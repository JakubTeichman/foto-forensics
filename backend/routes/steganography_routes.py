import os
import tempfile
import io
import base64
import traceback
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from PIL import Image
from typing import Optional, Dict

# Standardowe importy ML
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# Globalna konfiguracja backendu Matplotlib (musi być przed pierwszym użyciem plt)
matplotlib.use('Agg')

# --- IMPORTY ZEWNĘTRZNYCH ANALIZATORÓW (ZACHOWANE) ---
# Użycie ImportError jest czystsze do obsługi opcjonalnych zależności
try:
    from stegano_compare.stegano_compare_main import analyze_images
except ImportError:
    analyze_images = None

try:
    from steganalysis.analyze_stegano_single import AnalyzeSteganoSingle
except ImportError:
    AnalyzeSteganoSingle = None

try:
    from steganalysis.aggregator import analyze as aggregator_analyze
except ImportError:
    aggregator_analyze = None

# --- IMPORTY MODELU SIAMESE ---
# Importujemy również sam moduł, aby znaleźć jego ścieżkę
import stegano_compare
from stegano_compare.preprocessing import prepare_pair_batch
from stegano_compare.siamese_model import load_siamese_model, SiameseMulti

# --- KONFIGURACJA ŚCIEŻEK I PARAMETRÓW DLA SIAMESE ---
MODEL_FILENAME = "siamese_stego_vs_pseudo_best.pth"
# Ustawiamy MODEL_PATH na samą nazwę pliku. Wyszukiwanie odbywa się w _get_model()
DEFAULT_MODEL_PATH = MODEL_FILENAME 

MODEL_PATH = os.environ.get("SIAMESE_MODEL_PATH", DEFAULT_MODEL_PATH)
print(f"DEBUG: Wyszukiwana nazwa pliku modelu: {MODEL_FILENAME}") # JAWNE LOGOWANIE NAZWY

THRESHOLD = float(os.environ.get("SIAMESE_THRESHOLD", "0.58")) 

# --- FUNKCJE POMOCNICZE DLA SIAMESE ---
_SIAMESE_MODEL = None
# Nowa globalna zmienna do przechowywania szczegółowego błędu ładowania
_SIAMESE_MODEL_ERROR = None 

def _get_model():
    """Ładuje model SiameseMulti jako singleton, aktywnie wyszukując plik wagi za pomocą ścieżek względnych."""
    global _SIAMESE_MODEL
    global _SIAMESE_MODEL_ERROR

    if _SIAMESE_MODEL is None:
        
        # Lista POTENCJALNYCH ścieżek względnych do przeszukania, w oparciu o PWD (/backend)
        possible_paths = [
            MODEL_FILENAME,                                            # 1. Bieżący katalog roboczy (/backend/)
            os.path.join("models", MODEL_FILENAME),                    # 2. Podkatalog 'models/' (/backend/models/)
            os.path.join("stegano_compare", "models", MODEL_FILENAME), # 3. Częsta ścieżka projektu (/backend/stegano_compare/models/)
            os.path.join("routes", MODEL_FILENAME),                    # 4. Jeśli plik wagi jest w tym samym folderze co routes.py
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.R_OK):
                found_path = path
                break
        
        if found_path is None:
            # Tworzenie komunikatu o błędzie z listą przeszukanych ścieżek
            searched_paths_str = ", ".join(possible_paths)
            _SIAMESE_MODEL_ERROR = f"Plik modelu '{MODEL_FILENAME}' nie został znaleziony w PWD ani w następujących ścieżkach względnych (względem PWD serwera, czyli /backend/): {searched_paths_str}."
            print(f"BŁĄD KRYTYCZNY: {_SIAMESE_MODEL_ERROR}")
            return None
        
        # Plik wagi został znaleziony. Używamy znalezionej ścieżki.
        MODEL_PATH_USED = found_path
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # KROK 1: Usunięcie ręcznego ładowania stanu.
            # load_siamese_model będzie teraz ładować wagi.
            print(f"INFO: Przekazywanie ścieżki {MODEL_PATH_USED} do load_siamese_model.")
            
            # ZMIANA: Wymuszamy in_channels=12, aby pasowało do dostępnego pliku wagi.
            _SIAMESE_MODEL = load_siamese_model(
                model_path=MODEL_PATH_USED, # Przekazujemy odnalezioną ścieżkę
                device=device,
                in_channels=12, # Zmienione z 30 na 12, aby pasowało do wagi
            )

            print(f"INFO: Model Siamese załadowany pomyślnie na urządzeniu: {device} (Architektura 12-kanałowa).")
            _SIAMESE_MODEL_ERROR = None # Zerowanie błędu po sukcesie
        except Exception as e:
            # ZAPIS SZCZEGÓŁOWEGO BŁĘDU DO GLOBALNEJ ZMIENNEJ
            detailed_error = f"Błąd wewnętrzny PyTorch: {type(e).__name__}: {str(e)}"
            _SIAMESE_MODEL_ERROR = detailed_error
            print(f"BŁĄD KRYTYCZNY: Nie udało się załadować modelu Siamese: {detailed_error}")
            # Nadal drukujemy traceback, aby zobaczyć, co dokładnie dzieje się wewnątrz load_siamese_model
            traceback.print_exc() 
            _SIAMESE_MODEL = None
            
    return _SIAMESE_MODEL

def _make_heatmap(full_a_tensor: torch.Tensor, full_b_tensor: torch.Tensor) -> Optional[np.ndarray]:
    """
    Funkcja tworząca mapę cieplną z uśrednionej kwadratowej różnicy tensorów wejściowych (C=30).
    """
    try:
        # Uproszczony wskaźnik aktywności: Średnia kwadratów różnicy kanałów
        diff = (full_a_tensor - full_b_tensor).pow(2)
        # Oblicz średnią wzdłuż wymiaru kanałów (C), zostawiając H i W
        heat_raw = diff.mean(dim=0).cpu().numpy()
        
        # Normalizacja do 0-1
        min_val, max_val = heat_raw.min(), heat_raw.max()
        if max_val > min_val:
            heat_norm = (heat_raw - min_val) / (max_val - min_val)
        else:
            heat_norm = np.zeros_like(heat_raw)

        return heat_norm
    except Exception as e:
        print(f"Błąd generowania heatmapy: {e}")
        return None

# --- INICJALIZACJA BLUEPRINTU ---
steganography_bp = Blueprint("steganography_bp", __name__) # Używamy nazwy z oryginalnego pliku
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- existing endpoints (kept) ----------------

@steganography_bp.route("/stegano/analyze", methods=["POST"])
def analyze_steganography():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided in the request."}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Cannot open image: {str(e)}"}), 400
        if AnalyzeSteganoSingle is None:
            return jsonify({"error": "AnalyzeSteganoSingle not available on server."}), 500
        analyzer = AnalyzeSteganoSingle()
        results = analyzer.analyze(pil_image=image)
        hidden_detected = results.get("hidden_detected", False)
        return jsonify({
            "status": "success",
            "hidden_detected": hidden_detected,
            "method": "multi-method",
            "details": results,
            "methods_results": results.get("methods_results", {})
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Steganography analysis failed: {str(e)}"}), 500

@steganography_bp.route("/stegano/analyze_full", methods=["POST"])
def analyze_steganography_full():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided in the request."}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400
        image_bytes = file.read()
        if aggregator_analyze is None:
            return jsonify({"error": "aggregator_analyze not available on server."}), 500
        report = aggregator_analyze(image_bytes=image_bytes, include_meta=True)
        analysis_results = report.get("analysis_results", {})
        hidden_detected = any(
            v.get("detected", False)
            for v in analysis_results.values()
            if isinstance(v, dict)
        )
        response = {
            "status": "success",
            "hidden_detected": bool(hidden_detected),
            "method": "aggregated",
            "report": report,
        }
        return jsonify(response), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Full steganography analysis failed: {str(e)}"}), 500

@steganography_bp.route("/stegano/compare", methods=["POST"])
def stegano_compare():
    if "original" not in request.files or "suspicious" not in request.files:
        return jsonify({"error": "Missing files"}), 400
    orig = request.files["original"]
    susp = request.files["suspicious"]
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, orig.filename)
        susp_path = os.path.join(tmpdir, susp.filename)
        orig.save(orig_path)
        susp.save(susp_path)
        if analyze_images is None:
            return jsonify({"error": "analyze_images (stegano_compare_main) not available."}), 500
        result = analyze_images(orig_path, susp_path)
    return jsonify(result)

# ---------------- NEW endpoint: Siamese CNN compare ----------------

@steganography_bp.route("/stegano/siamese", methods=["POST"])
def stegano_siamese():
    """
    Compare two images using trained SiameseMulti network.
    Output: JSON z polami 'score', 'threshold', 'stego_detected', 'confidence', 'heatmap_siamese'.
    """
    try:
        if "original" not in request.files or "suspicious" not in request.files:
            return jsonify({"error": "Missing files 'original' and 'suspicious'"}), 400

        orig = request.files["original"]
        susp = request.files["suspicious"]

        # 1. Ładowanie i weryfikacja modelu
        model = _get_model()
        if model is None:
            # Zwracamy szczegółowy błąd z globalnej zmiennej.
            error_message = _SIAMESE_MODEL_ERROR if _SIAMESE_MODEL_ERROR else f"Siamese model not initialized. Sprawdź, czy plik wagi pod {MODEL_PATH} jest poprawny."
            return jsonify({"status": "error", "error": error_message}), 500
        
        device = next(model.parameters()).device # Użyj urządzenia, na którym jest model

        # 2. Przygotowanie tensorów (batch=1, C=30, H, W).
        a_full_t, a_patch_t, b_full_t, b_patch_t = prepare_pair_batch(orig, susp, device=device, use_srm=True)

        # KROK 3: KOREKTA KANAŁÓW (konieczna dla modelu 12-kanałowego!)
        # Tensor wejściowy ma 30 kanałów, ale model tylko 12. Musimy obciąć.
        a_full_t = a_full_t[:, :12, :, :]
        b_full_t = b_full_t[:, :12, :, :]
        a_patch_t = a_patch_t[:, :12, :, :]
        b_patch_t = b_patch_t[:, :12, :, :]


        # 4. Inferencia
        with torch.no_grad():
            out, ea, eb = model(a_full_t, a_patch_t, b_full_t, b_patch_t)
            score = float(out.cpu().numpy()[0])

        stego_detected = bool(score >= THRESHOLD)
        confidence = float(score)  

        # 5. Generowanie Heatmapy (Wizualizacja)
        heat = None
        try:
            # Używamy 12 obciętych kanałów, które weszły do modelu
            heat_arr = _make_heatmap(a_full_t[0].cpu(), b_full_t[0].cpu()) 
            
            if heat_arr is not None:
                # Konwersja macierzy heatmapy na PNG w pamięci
                fig = plt.figure(frameon=False)
                fig.set_size_inches(4, 4)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(heat_arr, cmap='magma')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                buf.seek(0)
                img_bytes = buf.read()
                heat = "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            heat = None

        # 6. Zwrot wyników
        response = {
            "status": "success",
            "score": score,
            "threshold": THRESHOLD,
            "stego_detected": stego_detected,
            "confidence": confidence,
            "heatmap_siamese": heat
        }
        return jsonify(response), 200

    except Exception as e:
        # Złapanie każdego innego błędu (np. podczas przygotowywania danych)
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Siamese analysis failed during inference or preprocessing: {str(e)}"}), 500