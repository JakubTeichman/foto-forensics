import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import traceback
import datetime
import sys
import gc
import base64
from flask import Blueprint, request, jsonify
from numpy.linalg import norm
import math 

# Upewnij się, że import modelu jest poprawny dla Twojego środowiska
from noiseprint.model import FullConvNet as NoiseprintModel

# --- ⚙️ Konfiguracja ---
MAX_DIM_FOR_SIMPLE = 4000
TILE_SIZE = 512
OVERLAP = 32
MIN_TILE_DIM = 64
BATCH_SIZE = 4 

# PARAMETRY TRANSFORMACJI (nowe)
CENTER_C = 0.51 
SLOPE_K = 250 

noiseprint_bp = Blueprint("noiseprint", __name__)

# ---- 🧠 Ładowanie modelu (Wersja z akceleracją GPU/FP16 z Pliku 2) ----
def _get_device():
    """Wybiera CUDA jeśli jest dostępne, w przeciwnym razie CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = _get_device()
MODEL_PATH = "/app/noiseprint/weights/model_noiseprint.pth"

model = None
try:
    model = NoiseprintModel()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    
    # Przeniesienie modelu na urządzenie i użycie FP16 na GPU
    model.to(DEVICE)
    if DEVICE.type == "cuda":
        model.half()
        print("[INFO] Noiseprint model loaded on CUDA with FP16.")
    else:
        print("[INFO] Noiseprint model loaded on CPU.")
        
except Exception as e:
    print(f"[ERROR] Failed to load noiseprint model: {str(e)}", file=sys.stderr)

# --- Usunięte: Placeholder functions (preprocess_image, generate_noiseprint_from_tiles, cosine_similarity, compute_noise_stats)
# Zostały zastąpione ujednoliconymi i poprawionymi wersjami poniżej.

# ---- 🧩 Logging ----
def log_error(msg: str):
    """Zapisuje błąd do pliku log oraz do stderr."""
    log_path = "noiseprint_errors.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n{'-'*80}\n\n")
    except Exception:
        pass

    print(msg, file=sys.stderr)


# =========================================================================
# === LOGIKA DLA MAŁYCH OBRAZÓW (SIMPLE) ===
# =========================================================================

def preprocess_image_simple(image: Image.Image):
    """Konwersja obrazu PIL na tensor (1, 1, H, W)."""
    image = image.convert("L")
    img = np.array(image, dtype=np.float32)
    img = img / 255.0
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    return tensor

def generate_noiseprint_simple(img_tensor: torch.Tensor):
    """Generuje noiseprint dla jednego tensora w całości."""
    if model is None:
        raise RuntimeError("Noiseprint model not loaded")
    with torch.no_grad():
        # Używa DEVICE ustawionego globalnie
        noise = model(img_tensor.to(DEVICE)).squeeze().cpu().numpy()
    return noise


# =========================================================================
# === LOGIKA DLA DUŻYCH OBRAZÓW (TILED) ===
# =========================================================================

def preprocess_image_tiled(image: Image.Image):
    """
    Konwersja obrazu PIL -> numpy grayscale i podział na kafelki z metadanymi.
    Zwraca: (y, x, h, w), tiles_tensors (na CPU), (H, W)
    """
    image = image.convert("L")
    arr = np.array(image, dtype=np.float32)
    arr = arr / 255.0

    H, W = arr.shape
    tiles_meta = []
    tiles = []

    stride = max(1, TILE_SIZE - OVERLAP)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + TILE_SIZE, H)
            x2 = min(x + TILE_SIZE, W)
            tile = arr[y:y2, x:x2]
            h, w = tile.shape
            if h < MIN_TILE_DIM or w < MIN_TILE_DIM:
                continue
            
            # Wypełnienie (padding) do pełnego rozmiaru TILE_SIZE, jeśli kafel jest na krawędzi
            if h != TILE_SIZE or w != TILE_SIZE:
                pad_h = TILE_SIZE - h
                pad_w = TILE_SIZE - w
                tile_padded = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")
            else:
                tile_padded = tile
                
            # Konwersja do tensora (CPU)
            tensor = torch.from_numpy(tile_padded).unsqueeze(0).unsqueeze(0).float()
            tiles_meta.append((y, x, h, w))
            tiles.append(tensor)

    if not tiles:
        raise ValueError("Image too small for tiling.")

    return tiles_meta, tiles, (H, W)


def generate_noiseprint_from_tiles(tiles_meta, tiles, full_shape):
    """
    Przetwarza kafelki w batchach i scala wyniki.
    """
    if model is None:
        raise RuntimeError("Noiseprint model not loaded")

    H, W = full_shape
    acc = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    device = DEVICE
    use_fp16 = (device.type == "cuda")

    total = len(tiles)
    idx = 0

    with torch.no_grad():
        while idx < total:
            end = min(idx + BATCH_SIZE, total)
            batch = torch.cat(tiles[idx:end], dim=0)
            
            # Przeniesienie na urządzenie
            try:
                batch = batch.to(device)
                if use_fp16:
                    batch = batch.half()
            except Exception:
                pass # Kontynuuj na CPU jeśli błąd

            # Model forward
            try:
                out = model(batch)
                # Bierzemy pierwszy kanał, przenosimy na CPU i konwertujemy na float
                out_cpu = out[:, 0, :, :].cpu().float().numpy()
            except Exception as e:
                log_error(f"Model forward failed on batch idx {idx}-{end}: {e}\n{traceback.format_exc()}")
                idx = end
                continue # Pomiń tę partię

            # Akumulacja wyników
            for i in range(out_cpu.shape[0]):
                y, x, h, w = tiles_meta[idx + i]
                res = out_cpu[i]
                # Przycięcie do oryginalnego rozmiaru kafelka (w przypadku paddingu na krawędzi)
                res_crop = res[:h, :w]
                
                # Dodawanie do akumulatora i licznika
                acc[y:y + h, x:x + w] += res_crop
                count[y:y + h, x:x + w] += 1.0

            # Czyszczenie pamięci
            del batch, out, out_cpu
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            idx = end

    count[count == 0] = 1.0 # Zapobiega dzieleniu przez zero
    merged = acc / count
    return merged.astype(np.float32)


# =========================================================================
# === UNIFIED GENERATION FUNCTION ===
# =========================================================================

def generate_noiseprint_unified(image: Image.Image):
    """
    Wybiera metodę generowania noiseprintu w zależności od rozmiaru obrazu.
    """
    # Zapewnienie, że obraz jest w RGB do sprawdzenia rozmiaru i ewentualnej konwersji L w środku
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    max_dim = max(image.size)
    H, W = image.height, image.width

    if max_dim < MAX_DIM_FOR_SIMPLE:
        # 1. Prosta metoda (Non-Tiled) dla małych obrazów
        img_tensor = preprocess_image_simple(image)
        noise = generate_noiseprint_simple(img_tensor)
        # print(f"[INFO] Generated noiseprint ({H}x{W}) using SIMPLE (Non-Tiled) method.")
        return noise
    else:
        # 2. Metoda kafelkowa (TILED) dla dużych obrazów
        tiles_meta, tiles, full_shape = preprocess_image_tiled(image)
        noise = generate_noiseprint_from_tiles(tiles_meta, tiles, full_shape)
        # print(f"[INFO] Generated noiseprint ({H}x{W}) using TILED method.")
        return noise


# ---- 🧮 Helpers ----

def local_normalize(arr):
    """Lokalna normalizacja do średniej 0 i std 1. Zabezpieczona przed NaN i zerowym std."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(arr)
    if std < 1e-8:
        return arr
    return (arr - np.mean(arr)) / std


def cross_correlation(a, b):
    """Normalized cross-correlation (NCC). Wartość od -1 do 1."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def diff_energy(a, b):
    """Różnica energii między dwoma noiseprintami."""
    diff = np.abs(a - b)
    return float(np.sum(diff ** 2))

# =================================================================
# FUNKCJA - Transformuje surowe podobieństwo kosinusowe na skalę 0-100
# =================================================================
def transform_similarity_score(raw_correlation):
    """
    Przekształca surową korelację kosinusową (w zakresie -1 do 1) na 
    bardziej czytelną skalę 0-100 za pomocą funkcji logistycznej (Sigmoid).
    """
    
    # Krok 1: Normalizacja surowej korelacji kosinusowej do zakresu [0, 1]
    scaled_corr = (raw_correlation + 1.0) / 2.0
    
    # Zastosowanie funkcji logistycznej (Sigmoid)
    exponent = -SLOPE_K * (scaled_corr - CENTER_C)
    
    # Ograniczenie wartości wykładnika, aby uniknąć przepełnienia (overflow)
    if exponent > 50: 
        sigmoid_val = 0.0
    elif exponent < -50: 
        sigmoid_val = 1.0
    else:
        sigmoid_val = 1.0 / (1.0 + math.exp(exponent))
        
    # Skalowanie do zakresu 0-100
    score = 100.0 * sigmoid_val
    return float(score)


def compute_noise_stats(noise):
    """Oblicza podstawowe statystyki noiseprintu."""
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mean": float(np.mean(noise)),
        "std": float(np.std(noise)),
        "energy": float(np.sum(noise ** 2)),
        "entropy": float(-np.sum(noise * np.log(np.abs(noise) + 1e-10))),
    }


# ---- ⚙️ ROUTE: Noiseprint generation ----
@noiseprint_bp.route("/generate", methods=["POST"])
def generate():
    # ... (logika pozostaje bez zmian) ...
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided."}), 400

        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            log_error(f"❌ Failed to open image:\n{traceback.format_exc()}")
            return jsonify({"error": "Invalid or corrupted image file."}), 400

        try:
            noise = generate_noiseprint_unified(img)
        except MemoryError:
            log_error("❌ System ran out of memory (MemoryError)")
            return jsonify({"error": "Image too large for processing. Try smaller file."}), 500
        except Exception:
            log_error(f"❌ Exception in noiseprint generation:\n{traceback.format_exc()}")
            return jsonify({"error": "Noiseprint generation failed."}), 500

        stats = compute_noise_stats(noise)
        normalized = local_normalize(noise)
        
        # Kodowanie noiseprintu do PNG
        _, buffer = cv2.imencode(".png", (normalized * 127 + 128).astype(np.uint8))
        encoded = base64.b64encode(buffer).decode("utf-8")

        response = {"noiseprint": encoded, "stats": stats}

        if max(img.size) >= MAX_DIM_FOR_SIMPLE:
            response["info"] = f"✅ Image processed using memory-safe tiling (size > {MAX_DIM_FOR_SIMPLE}px)."
        else:
             response["info"] = "✅ Image processed using standard non-tiled method."

        return jsonify(response)

    except Exception:
        log_error(f"❗ Uncaught top-level error:\n{traceback.format_exc()}")
        return jsonify({"error": "Unexpected backend error."}), 500


# ---- ⚖️ ROUTE: Compare multiple noiseprints (Zmieniona logika) ----
@noiseprint_bp.route('/compare', methods=['POST'])
def compare():
    try:
        evidence_file = request.files.get('evidence')
        refs = request.files.getlist('references')

        if not evidence_file:
            return jsonify({'error': 'No evidence image provided.'}), 400

        # Wczytanie i przetworzenie próbki dowodowej
        evidence_img = Image.open(evidence_file.stream).convert("RGB")
        evidence_np = generate_noiseprint_unified(evidence_img)
        evidence_np = local_normalize(evidence_np) # Normalizacja dowodu

        raw_correlations, transformed_scores, diff_energies = [], [], []
        
        # NOWY LIST: Zbiera wszystkie znormalizowane i dopasowane do rozmiaru dowodu referencje
        normalized_resized_refs = [] 

        # Porównanie z każdą referencją
        for ref_file in refs:
            try:
                ref_img = Image.open(ref_file.stream).convert("RGB")
                
                # Generowanie i normalizacja noiseprintu referencyjnego
                ref_np = generate_noiseprint_unified(ref_img)
                ref_np_normalized = local_normalize(ref_np)

                # Dopasowanie rozmiaru - skalowanie referencji w górę do rozmiaru dowodu
                compare_ref_np = ref_np_normalized
                if ref_np_normalized.shape != evidence_np.shape:
                    compare_ref_np = cv2.resize(ref_np_normalized, 
                                                (evidence_np.shape[1], evidence_np.shape[0]), 
                                                interpolation=cv2.INTER_LINEAR)
                    # Ponowna normalizacja po skalowaniu
                    compare_ref_np = local_normalize(compare_ref_np)

                # Dodanie do listy do uśredniania
                normalized_resized_refs.append(compare_ref_np)

                raw_corr = cross_correlation(evidence_np, compare_ref_np) # NCC [-1, 1]
                transformed_score = transform_similarity_score(raw_corr) # Skala 0-100
                diff_en = diff_energy(evidence_np, compare_ref_np)

                raw_correlations.append(raw_corr)
                transformed_scores.append(transformed_score)
                diff_energies.append(diff_en)
            except Exception:
                log_error(f"⚠️ Failed to process reference file:\n{traceback.format_exc()}")
                continue
                
        if not transformed_scores:
             return jsonify({'error': 'No valid reference comparisons were produced.'}), 400

        # === NOWY KROK 1: Obliczanie uśrednionego Noiseprintu Referencyjnego ===
        mean_reference_np = np.mean(np.array(normalized_resized_refs), axis=0)
        mean_reference_np = local_normalize(mean_reference_np) # Finalna normalizacja

        # === NOWY KROK 2: Statystyki i kodowanie uśrednionego Noiseprintu ===
        stats_mean = compute_noise_stats(mean_reference_np)
        _, buf_mean_ref = cv2.imencode('.png', (mean_reference_np * 127 + 128).astype(np.uint8))
        encoded_mean_ref = base64.b64encode(buf_mean_ref).decode('utf-8')
        
        # Statystyki wyników
        mean_score = float(np.mean(transformed_scores))
        std_score = float(np.std(transformed_scores))
        mean_diff_en = float(np.mean(diff_energies))

        _, buf_evidence = cv2.imencode('.png', (evidence_np * 127 + 128).astype(np.uint8))

        return jsonify({
            'evidence_noiseprint': base64.b64encode(buf_evidence).decode('utf-8'),
            'mean_reference_noiseprint': encoded_mean_ref, # <-- NOWE POLE
            'stats_mean': stats_mean, # <-- NOWE POLE
            'pairwise_raw_correlations': raw_correlations, 
            'pairwise_transformed_scores': transformed_scores, 
            'mean_score': mean_score, # Średnia korelacja (0-100)
            'std_score': std_score,
            'mean_diff_energy': mean_diff_en,
            'stats_evidence': compute_noise_stats(evidence_np)
        })

    except Exception as e:
        log_error(f"❌ Noiseprint comparison failed:\n{traceback.format_exc()}")
        return jsonify({'error': f'Noiseprint comparison failed: {str(e)}'}), 500


# ---- 🧬 ROUTE: Embedding cosine similarity (Poprawiona logika) ----
@noiseprint_bp.route('/compare_embedding', methods=['POST'])
def compare_embedding():

    try:
        evidence_file = request.files.get('evidence')
        reference_file = request.files.get('reference')

        if not evidence_file or not reference_file:
            return jsonify({'error': 'Both evidence and reference images are required.'}), 400

        e_img = Image.open(evidence_file.stream).convert('RGB')
        r_img = Image.open(reference_file.stream).convert('RGB')

        # FIX: Użycie ujednoliconej funkcji generacji
        evidence_np = generate_noiseprint_unified(e_img)
        reference_np = generate_noiseprint_unified(r_img)

        evidence_np = local_normalize(evidence_np)
        reference_np = local_normalize(reference_np)

        # FIX: Resizing i ponowna normalizacja referencji do rozmiaru dowodu
        if evidence_np.shape != reference_np.shape:
            reference_np = cv2.resize(reference_np, (evidence_np.shape[1], evidence_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            reference_np = local_normalize(reference_np) # Ponowna normalizacja po skalowaniu

        # FIX: Użycie cross_correlation i transformacji do skali 0-100
        raw_corr = cross_correlation(evidence_np, reference_np)
        transformed_score = transform_similarity_score(raw_corr) 

        _, buf_evidence = cv2.imencode('.png', (evidence_np * 127 + 128).astype(np.uint8))
        _, buf_reference = cv2.imencode('.png', (reference_np * 127 + 128).astype(np.uint8))

        return jsonify({
            "method": "cross_correlation_transformed",
            "raw_correlation": raw_corr,
            "transformed_similarity_score": transformed_score, # <-- Zmieniona nazwa i skala 0-100
            "evidence_noiseprint": base64.b64encode(buf_evidence).decode('utf-8'),
            "reference_noiseprint": base64.b64encode(buf_reference).decode('utf-8'),
            "stats_evidence": compute_noise_stats(evidence_np),
            "stats_reference": compute_noise_stats(reference_np)
        })

    except Exception as e:
        log_error(f"❌ Embedding comparison failed:{traceback.format_exc()}")
        return jsonify({'error': f'Embedding comparison failed: {str(e)}'}), 500


# ---- 🧬 ROUTE: Compare with reference noiseprint database (SQL) ----
# ... (Ten fragment pomijam, zakładając, że działa poprawnie z nowymi funkcjami pomocniczymi) ...
# Wymagałby też dostosowania do nowej logiki transformacji, ale poza zakresem problemu z kompatybilnością.
@noiseprint_bp.route('/compare_with_db', methods=['POST'])
def compare_with_db():
    """
    Porównuje noiseprint obrazu dowodowego z bazą zapisanych noiseprintów referencyjnych (SQLAlchemy).
    """
    try:
        # Zakładamy, że potrzebne moduły są dostępne globalnie w kontekście Flask
        from extensions import db
        from routes.add_reference import DeviceReference, deserialize_noiseprint

        noiseprint_file = request.files.get("noiseprint")
        evidence_file = request.files.get("evidence")

        evidence_np = None

        if noiseprint_file:
            # === Wariant A: Wczytanie gotowego noiseprintu ===
            try:
                noiseprint_img = Image.open(noiseprint_file.stream).convert("L")
                evidence_np = np.array(noiseprint_img).astype(np.float32)
                evidence_np = local_normalize(evidence_np)
            except Exception:
                log_error(f"❌ Failed to load provided noiseprint:\n{traceback.format_exc()}")
                return jsonify({"error": "Failed to load provided noiseprint."}), 400

        elif evidence_file:
            # === Wariant B: Generowanie noiseprintu z obrazu dowodowego ===
            try:
                e_img = Image.open(evidence_file.stream).convert('RGB')
                evidence_np = generate_noiseprint_unified(e_img)
                evidence_np = local_normalize(evidence_np)
            except Exception:
                log_error(f"❌ Failed to generate noiseprint from evidence image:\n{traceback.format_exc()}")
                return jsonify({"error": "Failed to generate noiseprint from evidence image."}), 400
        else:
            return jsonify({"error": "No input provided. Expected 'noiseprint' or 'evidence'."}), 400

        # === 2️⃣ Wczytanie wszystkich referencji z bazy ===
        refs = db.session.query(DeviceReference).all()
        if not refs:
            return jsonify({'error': 'No reference entries found in database.'}), 404

        results = []
        
        for ref in refs:
            try:
                # Deserializacja noiseprintu referencyjnego
                ref_np = deserialize_noiseprint(ref.noiseprint)
                ref_np = np.nan_to_num(ref_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                # Normalizacja referencji
                ref_np_normalized = local_normalize(ref_np)

                # Utworzenie tymczasowej, skalowalnej kopii dowodu
                compare_evidence_np = evidence_np 
                
                # === ZMIANA LOGIKI SKALOWANIA: Skalowanie dużego dowodu w dół do rozmiaru referencji ===
                if ref_np_normalized.shape != compare_evidence_np.shape:
                    compare_evidence_np = cv2.resize(compare_evidence_np, 
                                                     (ref_np_normalized.shape[1], ref_np_normalized.shape[0]), 
                                                     interpolation=cv2.INTER_LINEAR)
                    # Po skalowaniu konieczna jest ponowna lokalna normalizacja
                    compare_evidence_np = local_normalize(compare_evidence_np)

                raw_corr = cross_correlation(compare_evidence_np, ref_np_normalized) # NCC [-1, 1]
                transformed_score = transform_similarity_score(raw_corr) # Skala 0-100

                results.append({
                    "id": ref.id,
                    "manufacturer": ref.manufacturer,
                    "model": ref.model,
                    "num_images": ref.num_images,
                    "raw_correlation": round(raw_corr, 4), # Zachowujemy dla referencji
                    "transformed_score": round(transformed_score, 2)
                })
            except Exception:
                log_error(f"⚠️ Failed to compare with reference ID={ref.id}\n{traceback.format_exc()}")

        # === 3️⃣ Sortowanie i zwrot wyników ===
        results_sorted = sorted(results, key=lambda x: x["transformed_score"], reverse=True)
        best_match = results_sorted[0] if results_sorted else None

        response = {
            "total_references": len(refs),
            "matches": results_sorted,
            "best_match": best_match,
            "transform_settings": {
                "center_c": CENTER_C,
                "slope_k": SLOPE_K,
                "description": "Similarity score (0-100) transformed using a Sigmoid function."
            }
        }
        return jsonify(response)

    except Exception:
        log_error(f"❌ compare_with_db failed:\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal error during database comparison.'}), 500