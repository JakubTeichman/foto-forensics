from flask import Blueprint, request, jsonify
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import os
import torch
from extensions import db
from flask_sqlalchemy import SQLAlchemy
from noiseprint.model import FullConvNet as NoiseprintModel
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy import text

add_reference_bp = Blueprint("add_reference", __name__)

# Parametry fragmentacji (patchingu)
TILE_SIZE = 512 # Standardowy rozmiar fragmentu dla modelu Noiseprint
SIZE_THRESHOLD = 4000 # Próg, powyżej którego włącza się fragmentacja (patching)

# ==========================================
# 🧱 Model bazy danych
# ==========================================
class DeviceReference(db.Model):
    __tablename__ = "device_references"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    manufacturer = db.Column(db.String(100), nullable=False)
    model = db.Column(db.String(100), nullable=False)
    noiseprint = db.Column(LONGBLOB, nullable=False) # LONGBLOB dla dużych danych
    num_images = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.TIMESTAMP, server_default=db.text("CURRENT_TIMESTAMP"))


# ==========================================
# 🧩 Wymuszenie typu LONGBLOB (ALTER TABLE)
# ==========================================
@add_reference_bp.before_app_request
def enforce_longblob():
    """Wymusza typ LONGBLOB w kolumnie 'noiseprint'."""
    try:
        db.session.execute(text("""
            ALTER TABLE device_references 
            MODIFY COLUMN noiseprint LONGBLOB NOT NULL;
        """))
        db.session.commit()
    except Exception:
        pass


# ==========================================
# 🔧 Pomocnicze funkcje serializacji
# ==========================================
def serialize_noiseprint(np_array: np.ndarray) -> bytes:
    """Serializuje macierz NumPy do skompresowanego formatu .npz (bytes)."""
    mem = BytesIO()
    np.savez_compressed(mem, noise=np_array)
    mem.seek(0)
    return mem.read()


def deserialize_noiseprint(blob: bytes) -> np.ndarray:
    """Deserializuje bytes z powrotem do macierzy NumPy."""
    mem = BytesIO(blob)
    loaded = np.load(mem)
    return loaded["noise"]


# ==========================================
# 🔊 Noiseprint model i przetwarzanie
# ==========================================
MODEL_PATH = os.getenv("NOISEPRINT_MODEL_PATH", "/app/noiseprint/weights/model_noiseprint.pth")

noise_model = NoiseprintModel()
if os.path.exists(MODEL_PATH):
    try:
        noise_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    except RuntimeError as e:
        print(f"Błąd ładowania modelu (może być to normalne w tym środowisku, jeśli wagi są zewnętrzne): {e}")
else:
    print(f"⚠️ Ostrzeżenie: wagi modelu nie znalezione pod ścieżką: {MODEL_PATH}")
noise_model.eval()


def pil_to_tensor(pil_img: Image.Image):
    """Konwertuje obraz PIL do tensora gotowego dla modelu."""
    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    # Dodajemy wymiary batcha i kanału: [1,1,H,W]
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  
    return tensor

def generate_noiseprint_from_tensor(t: torch.Tensor) -> np.ndarray:
    """Generuje noiseprint z tensora."""
    with torch.no_grad():
        # Używamy squeeze(), aby usunąć wymiary batcha i kanału [H,W]
        out = noise_model(t).squeeze().cpu().numpy()
    return out

# NOWA FUNKCJA DO OBSŁUGI DUŻYCH OBRAZÓW POPRZEZ FRAGMENTACJĘ
def generate_noiseprint_from_pil_with_tiling(pil_img: Image.Image, tile_size: int, size_threshold: int) -> np.ndarray:
    """
    Generuje noiseprint z obrazu PIL. Dla dużych obrazów (> size_threshold) 
    używa techniki fragmentacji (tiling) w celu oszczędności pamięci.
    """
    width, height = pil_img.size
    
    # 1. PRZYPADEK 1: OBRAZ JEST WYSTARCZAJĄCO MAŁY (np. poniżej 4000 pikseli)
    if width <= size_threshold and height <= size_threshold:
        print(f"✅ Obraz {width}x{height} przetwarzany w całości (poniżej progu {size_threshold}).")
        t = pil_to_tensor(pil_img)
        return generate_noiseprint_from_tensor(t)

    # 2. PRZYPADEK 2: OBRAZ JEST ZBYT DUŻY - Używamy fragmentacji
    print(f"🧩 Obraz {width}x{height} przetwarzany z użyciem fragmentacji (powyżej progu {size_threshold}).")
    
    # Inicjalizacja macierzy wynikowej, która będzie trzymała złożone noiseprinty
    full_noiseprint = np.zeros((height, width), dtype=np.float32)
    
    # Obliczanie liczby fragmentów
    # Używamy zaokrąglenia w górę, aby pokryć cały obraz
    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size
    
    # Iteracja po fragmentach
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            
            # Definiowanie obszaru przycięcia
            left = x * tile_size
            top = y * tile_size
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            
            # Wymiar fragmentu (może być mniejszy na krawędzi)
            tile_width = right - left
            tile_height = bottom - top
            
            # Przycinanie fragmentu
            tile_img = pil_img.crop((left, top, right, bottom))
            
            # Wypełnienie fragmentu do pełnego rozmiaru, jeśli jest na krawędzi i mniejszy niż tile_size
            # To jest ważne, ponieważ model FullConvNet oczekuje stałego rozmiaru wejściowego.
            if tile_width != tile_size or tile_height != tile_size:
                 # Tworzymy czarny obraz (lub wypełniony innym kolorem) o rozmiarze TILE_SIZE
                padded_tile = Image.new('L', (tile_size, tile_size), 0)
                # Wklejamy przycięty fragment w lewy górny róg
                padded_tile.paste(tile_img, (0, 0))
                tile_img = padded_tile # Używamy padded_tile do generowania noiseprintu
            
            # Generowanie noiseprintu dla fragmentu
            t = pil_to_tensor(tile_img)
            tile_np = generate_noiseprint_from_tensor(t)
            
            # Wklejanie noiseprintu z powrotem do pełnej macierzy (uwzględniając ewentualne przycinanie krawędzi)
            full_noiseprint[top:bottom, left:right] = tile_np[:tile_height, :tile_width]

    return full_noiseprint.astype(np.float32)


# ==========================================
# 🧩 Endpoint: dodaj referencję
# ==========================================
@add_reference_bp.route("/add_reference", methods=["POST"])
def add_reference():
    manufacturer = request.form.get("manufacturer")
    model_name = request.form.get("model")

    if not manufacturer or not model_name:
        return jsonify({"error": "manufacturer and model are required"}), 400

    files = request.files.getlist("images")
    if not files or len(files) < 3:
        return jsonify({"error": "At least 3 images are required as reference"}), 400

    
    # 💥 INKREMENTALNE UŚREDNIANIE (Running Average)
    mean_np = None # Macierz przechowująca sumę
    num_processed = 0
    
    for i, f in enumerate(files):
        try:
            pil = Image.open(f.stream)
            
            # Używamy funkcji z fragmentacją
            np_noise = generate_noiseprint_from_pil_with_tiling(pil, TILE_SIZE, SIZE_THRESHOLD)
            
            if mean_np is None:
                # Inicjalizacja macierzy średniej (używamy pierwszej macierzy jako punktu odniesienia)
                mean_np = np_noise.astype(np.float32)
                num_processed = 1
            else:
                # Weryfikacja kształtu i skalowanie do kształtu pierwszej macierzy (jeśli obrazy miały inną rozdzielczość)
                if np_noise.shape != mean_np.shape:
                    target_shape = mean_np.shape
                    print(f"⚠️ Ostrzeżenie: Niezgodność kształtów. Skalowanie obrazu {i+1} do: {target_shape}")
                    np_noise = cv2.resize(np_noise, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Sumowanie inkrementalne
                mean_np += np_noise
                num_processed += 1
            
        except Exception as e:
            return jsonify({"error": f"Failed to process image {i+1} ({f.filename}): {str(e)}"}), 400

    # Ostateczne obliczenie średniej
    try:
        if num_processed == 0:
            return jsonify({"error": "No images were processed successfully"}), 500
            
        mean_np /= num_processed
        blob = serialize_noiseprint(mean_np)
    except Exception as e:
        return jsonify({"error": f"Failed to calculate mean noiseprint or serialize: {str(e)}"}), 500


    # Zapis do bazy
    try:
        ref = DeviceReference(
            manufacturer=manufacturer,
            model=model_name,
            noiseprint=blob,
            num_images=num_processed,
        )
        db.session.add(ref)
        db.session.commit()
        return jsonify({"message": "Reference added", "id": ref.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error during insert: {str(e)}"}), 500


# ==========================================
# 🧠 Endpoint: porównanie z bazą
# ==========================================
@add_reference_bp.route("/compare_with_db", methods=["POST"])
def compare_with_db():
    top_n = int(request.form.get("top_n", 5))
    evidence_file = request.files.get("evidence")

    if evidence_file is None:
        return jsonify({"error": "evidence file required"}), 400

    try:
        # Generowanie noiseprintu dla dowodu (również z obsługą fragmentacji)
        pil_e = Image.open(evidence_file.stream)
        evidence_np = generate_noiseprint_from_pil_with_tiling(pil_e, TILE_SIZE, SIZE_THRESHOLD)
    except Exception as e:
        return jsonify({"error": f"Failed to process evidence: {str(e)}"}), 400

    try:
        rows = db.session.query(DeviceReference).all()
        results = []

        # Kształt docelowy jest określany przez noiseprint dowodowy
        target_shape_comp = evidence_np.shape

        for row in rows:
            ref_np = deserialize_noiseprint(row.noiseprint)
            
            # Skalowanie, jeśli noiseprinty mają różną rozdzielczość (np. różne wejściowe obrazy)
            if ref_np.shape != target_shape_comp:
                ref_np = cv2.resize(ref_np, (target_shape_comp[1], target_shape_comp[0]), interpolation=cv2.INTER_LINEAR)

            a, b = evidence_np.flatten(), ref_np.flatten()
            
            # Obliczenie współczynnika korelacji
            corr = 0.0
            if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                 corr = float(np.corrcoef(a, b)[0, 1])

            results.append({
                "id": row.id,
                "manufacturer": row.manufacturer,
                "model": row.model,
                "num_images": row.num_images,
                "correlation": corr,
            })

        results_sorted = sorted(results, key=lambda x: x["correlation"], reverse=True)[:top_n]
        return jsonify({"results": results_sorted})
    except Exception as e:
        return jsonify({"error": f"Error during database comparison: {str(e)}"}), 500