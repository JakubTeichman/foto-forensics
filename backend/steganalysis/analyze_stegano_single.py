import os
import cv2
import json
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# --- Classic methods ---
from steganalysis.classic.rs_analysis import analyze as rs_analyze

# --- Statistical methods ---
from steganalysis.statistical.cooccurrence_analysis import analyze as cooccurrence_analyze
from steganalysis.statistical.noise_residuals import analyze as noise_residuals_analyze
from steganalysis.statistical.wavelet_analysis import analyze as wavelet_analyze
from steganalysis.statistical.dct_stats_analysis import analyze_dct_stats as dct_stats_analyze
from steganalysis.statistical.high_frequency_residual_analysis import high_frequency_residual_analysis
from steganalysis.statistical.sample_pair_analysis import analyze as sample_pair_analyze
from steganalysis.statistical.pixel_pair_histogram import analyze as pixel_pair_diff_analyze
from steganalysis.statistical.noise_residual_correlation import analyze_noise_residual as noise_residual_corr_analyze
from steganalysis.statistical.markov_cooccurrence_analysis import markov_cooccurrence_analysis as markov_cooccurrence_analyze


class AnalyzeSteganoSingle:
    """
    Unified runner for per-image steganalysis methods.
    Runs all detectors, aggregates results, and computes average heatmap.
    """

    def __init__(self):
        # Register available analysis methods
        self.methods = {
            # --- Classic ---
            "rs_analysis": rs_analyze,

            # --- Statistical ---
            "cooccurrence": cooccurrence_analyze,
            "noise_residuals": noise_residuals_analyze,
            "wavelet": wavelet_analyze,
            "dct_stats": dct_stats_analyze,
            "high_freq_residual": high_frequency_residual_analysis,
            "sample_pair": sample_pair_analyze,
            "pixel_pair_diff": pixel_pair_diff_analyze,
            "noise_residual_corr": noise_residual_corr_analyze,
            "markov_cooccurrence": markov_cooccurrence_analyze,
        }

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
        from matplotlib import pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.imshow(array, cmap="inferno")
        plt.axis("off")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze(self, image_path=None, pil_image=None):
        """Run all registered steganalysis methods and aggregate results."""
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

        pil_gray = pil_rgb.convert("L")

        methods_results = {}
        heatmaps = []

        for name, fn in self.methods.items():
            try:
                # Choose appropriate input
                if name == "lsb_histogram":
                    res = fn(pil_image=pil_rgb)
                elif name == "high_freq_residual":
                    res = fn(np.array(pil_rgb))
                else:
                    res = fn(pil_image=pil_gray)

                # Normalize output format
                if not isinstance(res, dict):
                    res = {"method": name, "score": float(res), "detected": float(res) >= 0.5, "details": {}}

                methods_results[name] = {
                    "method": res.get("method", name),
                    "score": float(res.get("score", 0.0)),
                    "detected": bool(res.get("detected", False)),
                    "details": res.get("details", {}),
                }

                # Collect heatmap if available
                if "heatmap_base64" in res:
                    arr = self._decode_heatmap(res["heatmap_base64"])
                    if arr is not None:
                        heatmaps.append(arr)

            except Exception as e:
                methods_results[name] = {
                    "method": name,
                    "score": 0.0,
                    "detected": False,
                    "details": {"error": str(e)},
                }

        # --- Combine all heatmaps into a mean one ---
        if heatmaps:
            avg_heatmap = np.mean(heatmaps, axis=0)
            avg_heatmap_encoded = self._encode_heatmap(avg_heatmap)
        else:
            avg_heatmap_encoded = None

        # --- Determine overall detection ---
        detected_methods = [n for n, r in methods_results.items() if r.get("detected", False)]
        overall_detected = len(detected_methods) > 0

        return {
            "hidden_detected": overall_detected,
            "detected_methods": detected_methods,
            "total_methods": len(methods_results),
            "positive_count": len(detected_methods),
            "methods_results": methods_results,
            "average_heatmap_base64": avg_heatmap_encoded,
        }
