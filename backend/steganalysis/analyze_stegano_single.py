import os
import json
import cv2
import numpy as np
from PIL import Image

# ✅ Upewniamy się, że importy działają poprawnie w projekcie i Dockerze
from steganalysis.classic.chi_square_analysis import analyze as chi_square_analyze
from steganalysis.classic.lsb_histogram_analysis import analyze as lsb_histogram_analyze
from steganalysis.classic.rs_analysis import analyze as rs_analyze

from steganalysis.statistical.cooccurrence_analysis import analyze as cooccurrence_analyze
from steganalysis.statistical.noise_residuals import analyze as noise_residuals_analyze
from steganalysis.statistical.wavelet_analysis import analyze as wavelet_analyze


class AnalyzeSteganoSingle:
    """
    Unified runner for per-image steganalysis methods.
    Optionally uses calibration.json for score adjustment.
    """

    def __init__(self, calibration_path=None):
        # 📁 Domyślna ścieżka do pliku kalibracji — obok pliku backendu
        if calibration_path is None:
            calibration_path = os.path.join(os.path.dirname(__file__), "calibration.json")

        self.methods = {
            "chi_square": chi_square_analyze,
            "lsb_histogram": lsb_histogram_analyze,
            "rs_analysis": rs_analyze,
            "cooccurrence": cooccurrence_analyze,
            "noise_residuals": noise_residuals_analyze,
            "wavelet": wavelet_analyze,
        }

        # 🔹 Wczytaj dane kalibracji (jeśli są dostępne)
        self.calibration_data = None
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    self.calibration_data = json.load(f)
                print(f"✅ Calibration loaded from: {calibration_path}")
            except Exception as e:
                print(f"⚠️ Could not load calibration file: {e}")
        else:
            print("ℹ️ No calibration file found — running without calibration")

    # --- KALIBRACJA ---
    def _apply_calibration(self, method_name, score):
        """Adjusts score based on calibration data if available."""
        if not self.calibration_data:
            return score

        delta_section = self.calibration_data.get("delta", {})
        cover_means = self.calibration_data.get("cover_means", {})
        stegano_means = self.calibration_data.get("stegano_means", {})

        if method_name not in delta_section:
            return score

        cover_mean = cover_means.get(method_name, 0.0)
        stegano_mean = stegano_means.get(method_name, 0.0)
        range_diff = stegano_mean - cover_mean

        if abs(range_diff) < 1e-9:
            return score  # brak sensownej różnicy

        # Normalizacja — przesuwamy wynik względem zakresu z kalibracji
        calibrated = (score - cover_mean) / range_diff
        return float(np.clip(calibrated, 0, 1))

    # --- ANALIZA ---
    def analyze(self, image_path=None, pil_image=None):
        """Main analysis pipeline for single image."""
        if pil_image is not None:
            pil_rgb = pil_image.convert("RGB")
        elif image_path:
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"❌ Cannot read image: {image_path}")
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            pil_rgb = Image.fromarray(img_rgb)
        else:
            raise ValueError("❗ Provide either image_path or pil_image")

        pil_gray = pil_rgb.convert("L")

        methods_results = {}

        for name, fn in self.methods.items():
            try:
                if name == "lsb_histogram":
                    res = fn(pil_image=pil_rgb)
                else:
                    res = fn(pil_image=pil_gray)

                if not isinstance(res, dict):
                    res = {"method": name, "score": float(res), "detected": float(res) >= 0.5, "details": {}}

                raw_score = float(res.get("score", 0.0))
                calibrated_score = self._apply_calibration(name, raw_score)

                methods_results[name] = {
                    "method": res.get("method", name),
                    "score_raw": raw_score,
                    "score_calibrated": calibrated_score,
                    "detected": calibrated_score >= 0.5,
                    "details": res.get("details", {}),
                }

            except Exception as e:
                methods_results[name] = {
                    "method": name,
                    "score_raw": 0.0,
                    "score_calibrated": 0.0,
                    "detected": False,
                    "details": {"error": str(e)},
                }

        detected_methods = [n for n, r in methods_results.items() if r.get("detected", False)]

        return {
            "hidden_detected": len(detected_methods) > 0,
            "detected_methods": detected_methods,
            "total_methods": len(methods_results),
            "positive_count": len(detected_methods),
            "methods_results": methods_results,
        }
