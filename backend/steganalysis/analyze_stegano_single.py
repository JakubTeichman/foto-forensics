import cv2
import numpy as np
from PIL import Image

# --- Importy metod ---
from .classic.chi_square_analysis import analyze as ChiSquareAnalysis
from .classic.lsb_histogram_analysis import analyze as LSBHistogramAnalysis
from .classic.rs_analysis import analyze as RSAnalysis
from .cnn.cnn_detector import CNNDetector
from .cnn.noiseprint_extractor import NoiseprintExtractor
from .ml.logistic_regression_detector import LogisticRegressionDetector
from .ml.svm_detector import SVMDetector
from .statistical.cooccurrence_analysis import analyze as CoOccurrenceAnalysis
from .statistical.noise_residuals import analyze as NoiseResiduals
from .statistical.wavelet_analysis import analyze as WaveletAnalysis


class AnalyzeSteganoSingle:
    """
    Klasa analizująca jeden obraz pod kątem steganografii.
    Łączy metody klasyczne, statystyczne i ML/CNN (jeśli aktywne).
    """

    def __init__(self):
        # --- Klasyczne metody ---
        self.his_square = ChiSquareAnalysis
        self.lsb_histogram = LSBHistogramAnalysis
        self.rs = RSAnalysis

        # --- Statystyczne metody ---
        self.cooccurrence = CoOccurrenceAnalysis
        self.noise_residuals = NoiseResiduals
        self.wavelet = WaveletAnalysis

        # --- Modele ML i CNN ---
        self.logreg = LogisticRegressionDetector()
        self.svm = SVMDetector()
        self.cnn = CNNDetector()
        self.noiseprint = NoiseprintExtractor()

    def analyze(self, image_path=None, pil_image=None):
        """
        Analizuje obraz i zwraca raport zbiorczy.
        """
        # --- Wczytanie obrazu ---
        if pil_image is not None:
            image = np.array(pil_image.convert("RGB"))
        elif image_path:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Nie udało się wczytać obrazu: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            raise ValueError("Musisz podać image_path lub pil_image")

        # --- Konwersja do odcieni szarości ---
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_pil = Image.fromarray(gray_image)

        results = {}

        # --- Klasyczne metody ---
        try:
            results["chi_square"] = self.his_square(image_bytes=None, pil_image=gray_pil)
        except Exception as e:
            results["chi_square"] = {"method": "chi_square", "score": 0.0, "detected": False, "error": str(e)}

        try:
            results["lsb_histogram"] = self.lsb_histogram(image_bytes=None, pil_image=pil_image)
        except Exception as e:
            results["lsb_histogram"] = {"method": "lsb_histogram", "score": 0.0, "detected": False, "error": str(e)}

        try:
            results["rs_analysis"] = self.rs(image_bytes=None, pil_image=gray_pil)
        except Exception as e:
            results["rs_analysis"] = {"method": "rs_analysis", "score": 0.0, "detected": False, "error": str(e)}

        # --- Statystyczne metody ---
        try:
            results["cooccurrence"] = self.cooccurrence(image_bytes=None, pil_image=gray_pil)
        except Exception as e:
            results["cooccurrence"] = {"method": "cooccurrence", "score": 0.0, "detected": False, "error": str(e)}

        try:
            results["noise_residuals"] = self.noise_residuals(image_bytes=None, pil_image=gray_pil)
        except Exception as e:
            results["noise_residuals"] = {"method": "noise_residuals", "score": 0.0, "detected": False, "error": str(e)}

        try:
            results["wavelet"] = self.wavelet(image_bytes=None, pil_image=gray_pil)
        except Exception as e:
            results["wavelet"] = {"method": "wavelet", "score": 0.0, "detected": False, "error": str(e)}

        # --- (opcjonalnie) Modele ML i CNN ---
        # Jeśli chcesz je włączyć, odkomentuj poniższe linie:
        """
        try:
            results["logistic_regression"] = self.logreg.analyze(image)
        except Exception as e:
            results["logistic_regression"] = {"method": "logreg", "score": 0.0, "detected": False, "error": str(e)}

        try:
            results["svm"] = self.svm.analyze(image)
        except Exception as e:
            results["svm"] = {"method": "svm", "score": 0.0, "detected": False, "error": str(e)}

        try:
            results["cnn"] = self.cnn.analyze(image)
        except Exception as e:
            results["cnn"] = {"method": "cnn", "score": 0.0, "detected": False, "error": str(e)}

        try:
            results["noiseprint"] = self.noiseprint.analyze(image)
        except Exception as e:
            results["noiseprint"] = {"method": "noiseprint", "score": 0.0, "detected": False, "error": str(e)}
        """

        # --- Analiza zbiorcza ---
        detected_methods = [
            name for name, res in results.items()
            if isinstance(res, dict) and res.get("detected", False)
        ]

        summary = {
            "hidden_detected": len(detected_methods) > 0,
            "detected_methods": detected_methods,
            "total_methods": len(results),
            "positive_count": len(detected_methods),
            "details": results
        }

        return summary
