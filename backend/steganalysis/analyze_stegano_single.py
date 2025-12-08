import os
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

# --- Deep Learning / Ensemble Detector ---
# Zgodnie z instrukcją, używamy tylko jednej, zaimportowanej metody.
from steganalysis.cnn.ensemble_detector import analyze as ensemble_cnn_analyze


class AnalyzeSteganoSingle:
    """
    Ujednolicony runner dla analizy steganograficznej,
    wykorzystujący wyłącznie detektor Ensemble CNN.
    Wyniki są skalowane do procentów zgodnie z podanymi progami.
    """

    def __init__(self):
        # Rejestrujemy tylko wymaganą metodę
        self.methods = {
            # --- Deep ensemble (CNN Ensemble Detector) ---
            "ensemble_C1": ensemble_cnn_analyze,
        }

    # ======================================================
    # 🔢 Logika skalowania wyniku do procentów (0-100%)
    # ======================================================
    def _calculate_stego_percentage(self, score):
        """
        Przelicza surowy wynik (0.0 do 1.0) na procent wykrycia (0 do 100)
        zgodnie z nieliniowymi progami:
        - <= 0.1: 0%
        - >= 0.6: 100%
        - 0.1 < score < 0.6: skalowanie liniowe od 0% do 100%
        """
        # Progi
        LOWER_THRESHOLD = 0.1  # Poniżej tej wartości wynik to 0%
        UPPER_THRESHOLD = 0.6  # Powyżej tej wartości wynik to 100%
        SCALING_RANGE = UPPER_THRESHOLD - LOWER_THRESHOLD  # Zakres skalowania: 0.5

        if score <= LOWER_THRESHOLD:
            return 0.0
        
        if score >= UPPER_THRESHOLD:
            return 100.0

        # Liniowe skalowanie dla wyniku w zakresie (0.1, 0.6)
        # 1. Przesunięcie: (score - 0.1) -> zakres (0, 0.5)
        # 2. Skalowanie do 0-1: / 0.5
        # 3. Mnożenie przez 100, aby uzyskać procent
        
        normalized_score = (score - LOWER_THRESHOLD) / SCALING_RANGE
        percentage = normalized_score * 100.0
        
        # Ograniczenie do [0.0, 100.0] (choć dzięki if/else powinno być to zbędne)
        return max(0.0, min(100.0, percentage))

    # ======================================================
    # 🔁 Pomocnicze metody (kodowanie / dekodowanie heatmap)
    # ======================================================
    def _decode_heatmap(self, heatmap_base64):
        """Convert base64 heatmap to NumPy grayscale array."""
        try:
            img_data = base64.b64decode(heatmap_base64)
            img = Image.open(BytesIO(img_data)).convert("L")
            return np.array(img, dtype=np.float32) / 255.0
        except Exception:
            return None

    def _encode_heatmap(self, array):
        """Convert NumPy array to base64-encoded PNG."""
        
        plt.figure(figsize=(4, 4))
        # Używamy cmap="inferno" jak w oryginalnym kodzie
        plt.imshow(array, cmap="inferno") 
        plt.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ======================================================
    # 🧠 Główna funkcja analizy
    # ======================================================
    def analyze(self, image_path=None, pil_image=None):
        """
        Uruchamia tylko detektor Ensemble CNN i agreguje wynik.
        """
        # 1. Wczytanie obrazu
        if pil_image is not None:
            pil_rgb = pil_image.convert("RGB")
        elif image_path:
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"❌ Cannot read image: {image_path}")
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            pil_rgb = Image.fromarray(img_rgb)
        else:
            raise ValueError("Provide either image_path or pil_image")

        # 2. Uruchomienie JEDYNEJ metody
        name = "ensemble_C1"
        fn = self.methods[name]
        methods_results = {}
        
        try:
            # CNN pracuje na kolorowym obrazie (pil_rgb)
            res = fn(pil_image=pil_rgb)
            
            # Weryfikacja formatu wyniku
            if not isinstance(res, dict) or "score" not in res:
                 # Zakładamy, że funkcja zwraca słownik z kluczem 'score' i opcjonalnie 'heatmap_base64'
                 # Jeśli nie jest słownikiem, rzucamy błąd
                 raise ValueError("Unexpected result format from ensemble_cnn_analyze.")
            
            raw_score = float(res.get("score", 0.0))
            
            # 3. Obliczenie wyniku procentowego zgodnie z nową logiką
            score_percent = self._calculate_stego_percentage(raw_score)
            
            # Warunek wykrycia: uznajemy, że wykryto, jeśli raw_score przekracza LOWER_THRESHOLD (0.1),
            # co odpowiada wynikowi procentowemu > 0%.
            detected = score_percent > 0.0
            
            methods_results[name] = {
                "method": res.get("method", name),
                # Zwracamy wynik surowy jako "score"
                "score": raw_score, 
                "score_percent": score_percent, # Dodatkowo zwracamy wynik procentowy
                "detected": detected,
                "details": res.get("details", {}),
            }

            # 4. Obsługa mapy cieplnej
            avg_heatmap_encoded = None
            if "heatmap_base64" in res:
                arr = self._decode_heatmap(res["heatmap_base64"])
                if arr is not None:
                    # Ponieważ jest tylko jedna metoda, "average_heatmap" to ta jedyna mapa
                    avg_heatmap_encoded = self._encode_heatmap(arr)

        except Exception as e:
            # Obsługa błędu, jeśli detektor zawiedzie
            avg_heatmap_encoded = None
            methods_results[name] = {
                "method": name,
                "score": 0.0,
                "score_percent": 0.0,
                "detected": False,
                "details": {"error": str(e)},
            }

        # 5. Określenie wyniku końcowego i zwrot
        overall_detected = methods_results[name]["detected"]
        detected_methods = [name] if overall_detected else []

        return {
            "hidden_detected": overall_detected,
            "detected_methods": detected_methods,
            "total_methods": len(methods_results), # Będzie zawsze 1
            "positive_count": len(detected_methods), # Będzie 1 lub 0
            "methods_results": methods_results,
            # "average_heatmap_base64" jest mapą z pojedynczej metody
            "average_heatmap_base64": avg_heatmap_encoded, 
        }