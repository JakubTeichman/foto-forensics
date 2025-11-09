import numpy as np
import pywt
from PIL import Image
import io

# ----------------------
# Helpers (local)
# ----------------------
def _compute_shannon_entropy(arr, bins=256):
    """Compute Shannon entropy of an array (0-255 range)."""
    hist, _ = np.histogram(arr.ravel(), bins=bins, range=(0, 255))
    probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _normalize_scores(values):
    """Normalize and average values between 0-1."""
    a = np.array(values, dtype=np.float64)
    if a.size == 0:
        return 0.0
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-12:
        return float(np.mean(a))
    norm = (a - mn) / (mx - mn)
    return float(np.clip(np.mean(norm), 0.0, 1.0))


def _rescale_score(value, mean_ref=0.5, scale_ref=0.25):
    """Rescale score to 0–1 range with soft normalization."""
    val = (value - mean_ref) / (scale_ref + 1e-12) / 2.0 + 0.5
    return float(np.clip(val, 0.0, 1.0))


# ----------------------
# Analyze (updated)
# ----------------------
def analyze(image_bytes=None, pil_image=None, wave='db1', levels=2, threshold=0.5):
    """
    Multi-scale Wavelet Statistics Analysis (updated).
    Detects inconsistencies in local textures and high-order coefficients.
    Returns structured result for aggregation layer.
    """
    # --- Load image ---
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.float32) / 255.0

    # --- Wavelet decomposition ---
    try:
        coeffs = pywt.wavedec2(arr, wavelet=wave, level=levels)
    except Exception as e:
        return {
            "method": "wavelet_analysis",
            "score": 0.0,
            "detected": False,
            "details": {"error": f"wavelet transform failed: {str(e)}"}
        }

    band_stats = []
    for lvl, tup in enumerate(coeffs[1:], start=1):
        cH, cV, cD = tup
        for name, band in [('cH', cH), ('cV', cV), ('cD', cD)]:
            band = band.astype(np.float64)
            energy = float(np.mean(np.abs(band)))
            std = float(np.std(band))
            kurtosis = float(np.mean(((band - band.mean()) / (std + 1e-12)) ** 4) - 3.0) if std > 0 else 0.0
            band_stats.append({
                "level": lvl,
                "band": name,
                "energy": energy,
                "std": std,
                "kurtosis": kurtosis
            })

    # --- Handle empty case ---
    if len(band_stats) == 0:
        return {
            "method": "wavelet_analysis",
            "score": 0.0,
            "detected": False,
            "details": {"error": "no valid coefficients"}
        }

    # --- Compute base metrics ---
    energies = np.array([b["energy"] for b in band_stats])
    kurtosis_vals = np.array([b["kurtosis"] for b in band_stats])

    mean_energy = float(np.mean(energies))
    std_energy = float(np.std(energies))
    mean_kurtosis = float(np.mean(kurtosis_vals))

    # --- Core scoring ---
    raw_score = float(np.clip(mean_energy / (np.max(energies) + 1e-12), 0.0, 1.0))
    kurtosis_norm = _normalize_scores(np.abs(kurtosis_vals))
    combined_raw = float(np.clip(0.7 * raw_score + 0.3 * kurtosis_norm, 0.0, 1.0))

    # --- Entropy correction ---
    approx = coeffs[0]
    approx_uint8 = np.clip(
        (approx - approx.min()) / (approx.max() - approx.min() + 1e-12) * 255.0,
        0, 255
    ).astype(np.uint8)
    entropy = _compute_shannon_entropy(approx_uint8)
    entropy_weight = float(np.clip(entropy / 7.5, 0.8, 1.25))

    combined_weighted = float(np.clip(combined_raw * entropy_weight, 0.0, 1.0))

    # --- Dynamic threshold ---
    hist, _ = np.histogram(energies, bins=6)
    probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-12)))
    dyn_threshold = float(np.clip(
        0.42 + 0.12 * (hist_entropy / np.log2(6)) + 0.04 * ((entropy - 3.0) / 4.0),
        0.25, 0.85
    ))

    # --- Final scoring ---
    score = _rescale_score(combined_weighted, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    # --- Return structured result ---
    details = {
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "mean_kurtosis": mean_kurtosis,
        "entropy": entropy,
        "combined_raw": combined_weighted,
        "dynamic_threshold": dyn_threshold,
        "n_bands": len(band_stats)
    }

    return {
        "method": "wavelet_analysis",
        "score": float(score),
        "detected": bool(detected),
        "details": details
    }
