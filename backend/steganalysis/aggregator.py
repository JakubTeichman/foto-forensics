import time
import io
from PIL import Image

# Importujemy klasę, która zawiera logikę detektora Ensemble CNN
# i realizuje skalowanie wyniku (procenty 0-100%)
from steganalysis_single_cnn import AnalyzeSteganoSingle

# Inicjalizacja detektora poza funkcją, aby uniknąć ponownego ładowania modeli
# w każdej analizie. Zakładamy, że kontekst pozwala na takie podejście.
try:
    SINGLE_CNN_ANALYZER = AnalyzeSteganoSingle()
    ANALYZER_READY = True
except Exception as e:
    SINGLE_CNN_ANALYZER = None
    ANALYZER_READY = False
    print(f"Błąd inicjalizacji AnalyzeSteganoSingle: {e}")


def _ensure_pil(image_bytes: bytes):
    """Konwertuje bajty obrazu na obiekt PIL.Image (RGB)."""
    if image_bytes is None:
        raise ValueError("image_bytes wymagane")
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def analyze(image_bytes: bytes, include_meta: bool = True, timeout_s: float = 30.0):
    """
    Uruchamia wyłącznie metodę Ensemble CNN (z AnalyzeSteganoSingle) i agreguje wyniki.
    
    Zwraca:
      {
        "analysis_results": {
            "ensemble_C1": {
                "method": "ensemble_C1",
                "score": 0.5,             # Surowy wynik (0-1)
                "score_percent": 80.0,    # Wynik po skalowaniu (0-100)
                "detected": True,
                "details": {},
            }
        },
        "meta": {"processing_time_s": float, "version": "v1.0"}
      }
    """
    t0 = time.time()
    pil_img = None
    
    if not ANALYZER_READY:
        return {"error": "Detektor nie został prawidłowo zainicjowany."}

    # 1. Wczytanie obrazu
    try:
        pil_img = _ensure_pil(image_bytes=image_bytes)
    except Exception as e:
        return {"error": f"Błąd wczytywania bajtów obrazu: {e}"}

    # 2. Uruchomienie analizy (wyłącznie CNN Ensemble Detector)
    try:
        # Analiza zwraca pełny wynik w formacie z poprzedniego pliku
        # ale nas interesuje tylko sekcja methods_results dla aggregation
        full_analysis_report = SINGLE_CNN_ANALYZER.analyze(pil_image=pil_img)
        
        # Ekstrakcja wyników (tylko jedna metoda: "ensemble_C1")
        results = full_analysis_report.get("methods_results", {})
        
        # Jeśli jest dostępna, dodajemy mapę cieplną do metadanych dla wygody
        # Choć zgodnie ze standardem agregatora, zwracamy tylko "methods_results"
        heatmap = full_analysis_report.get("average_heatmap_base64")
        
    except Exception as exc:
        # Zwrócenie błędu, jeśli analiza się nie powiodła
        results = {
            "ensemble_C1": {
                "method": "ensemble_C1", 
                "score": 0.0, 
                "score_percent": 0.0,
                "detected": False, 
                "details": {"error": str(exc)}
            }
        }

    # 3. Zgromadzenie metadanych
    t1 = time.time()
    out = {"analysis_results": results}
    
    if include_meta:
        meta_data = {
            "processing_time_s": float(t1 - t0), 
            "method_count": len(results),
            "version": "v1.0"
        }
        # Dodanie mapy cieplnej do metadanych, jeśli jest dostępna
        if heatmap:
            meta_data["average_heatmap_base64"] = heatmap
        
        out["meta"] = meta_data
        
    return out

# Example usage (wymaga zainicjowanej klasy AnalyzeSteganoSingle):
# with open("example.jpg", "rb") as f:
#     bytes_img = f.read()
# report = analyze(bytes_img)
# print(report)