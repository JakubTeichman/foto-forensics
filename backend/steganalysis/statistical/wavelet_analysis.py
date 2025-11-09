import pywt
import numpy as np
from PIL import Image
import io

# ----------------------
# Helpers (local)
# ----------------------
def _compute_shannon_entropy(arr, bins=256):
    hist, _ = np.histogram(arr.ravel(), bins=bins, range=(0, 255))
    probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def _normalize_scores(scores):
    a = np.array(scores, dtype=np.float64)
    if a.size == 0:
        return 0.0
    mn = a.min(); mx = a.max()
    if mx - mn < 1e-12:
        return float(np.mean(a))
    norm = (a - mn) / (mx - mn)
    return float(np.clip(np.mean(norm), 0.0, 1.0))

def _rescale_score(value, mean_ref=0.5, scale_ref=0.25):
    val = (value - mean_ref) / (scale_ref + 1e-12) / 2.0 + 0.5
    return float(np.clip(val, 0.0, 1.0))

# ----------------------
# analyze (updated)
# ----------------------
def analyze(image_bytes=None, pil_image=None, wave='db1', levels=2, threshold=0.5):
    """
    Wavelet multi-scale high-order statistics (updated).
    Returns dict: method, score (0..1), detected, details
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.float32) / 255.0
    coeffs = pywt.wavedec2(arr, wavelet=wave, level=levels)
    band_stats = []

    for lvl, tup in enumerate(coeffs[1:], start=1):
        cH, cV, cD = tup
        for name, band in [('cH', cH), ('cV', cV), ('cD', cD)]:
            e = float(np.mean(np.abs(band)))
            s = float(np.std(band))
            kurt = float(np.mean(((band - band.mean())/(s+1e-12))**4) - 3.0) if s > 0 else 0.0
            band_stats.append({"level": lvl, "band": name, "energy": e, "std": s, "kurtosis": kurt})

    energies = np.array([b['energy'] for b in band_stats]) if len(band_stats) > 0 else np.array([])
    if energies.size == 0:
        score = 0.0
        detected = False
        return {"method": "wavelet_analysis", "score": float(score), "detected": bool(detected), "details": {"error": "no bands"}}

    # raw score: mean energy normalized by max energy
    raw_score = float(np.clip(np.mean(energies) / (np.max(energies) + 1e-12), 0.0, 1.0))

    # kurtosis-based signal: high kurtosis may indicate manipulations
    kurtosis_list = [b["kurtosis"] for b in band_stats]
    kurt_norm = _normalize_scores(np.abs(kurtosis_list))

    combined = float(np.clip(0.7 * raw_score + 0.3 * kurt_norm, 0.0, 1.0))

    # entropy of approximation band (coeffs[0])
    approx = coeffs[0]
    approx_uint8 = np.clip((approx - approx.min()) / (approx.max() - approx.min() + 1e-12) * 255.0, 0, 255).astype(np.uint8)
    entropy = _compute_shannon_entropy(approx_uint8)
    entropy_weight = float(np.clip(entropy / 7.5, 0.8, 1.2))

    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold from energy distribution histogram
    hist, _ = np.histogram(energies, bins=6)
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    dyn_threshold = float(np.clip(0.4 + 0.12 * (hist_entropy / np.log2(6)) + 0.04 * ((entropy - 3.0)/4.0), 0.2, 0.85))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    details = {"bands": band_stats[:40], "entropy": entropy, "combined_raw": combined, "dynamic_threshold": dyn_threshold}
    return {"method": "wavelet_analysis", "score": float(score), "detected": bool(detected), "details": details}
