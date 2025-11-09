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

def _normalize_scores(values):
    a = np.array(values, dtype=np.float64)
    if a.size == 0:
        return 0.0
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-12:
        return float(np.mean(a))
    norm = (a - mn) / (mx - mn)
    return float(np.clip(np.mean(norm), 0.0, 1.0))

def _rescale_score(value, mean_ref=0.5, scale_ref=0.25):
    val = (value - mean_ref) / (scale_ref + 1e-12) / 2.0 + 0.5
    return float(np.clip(val, 0.0, 1.0))

# ----------------------
# Analyze (SPA method)
# ----------------------
def analyze(image_bytes=None, pil_image=None, threshold=0.5):
    """
    Sample Pair Analysis (SPA) for LSB steganalysis.
    Detects statistical irregularities in pixel pair parity distribution.
    Returns structured result for aggregation layer.
    """
    # --- Load image ---
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("L")

    arr = np.array(pil_image, dtype=np.uint8)
    h, w = arr.shape

    # --- Prepare pairs ---
    pairs_h = arr[:, :-1].astype(np.int16) - arr[:, 1:].astype(np.int16)
    pairs_v = arr[:-1, :].astype(np.int16) - arr[1:, :].astype(np.int16)
    diffs = np.concatenate([pairs_h.ravel(), pairs_v.ravel()])

    # --- Core SPA statistics ---
    even_diff = np.sum(diffs % 2 == 0)
    odd_diff = np.sum(diffs % 2 != 0)
    total = even_diff + odd_diff + 1e-12

    even_ratio = even_diff / total
    odd_ratio = odd_diff / total
    imbalance = abs(even_ratio - odd_ratio)

    # --- Local variance / entropy support ---
    entropy = _compute_shannon_entropy(arr)
    entropy_weight = float(np.clip(entropy / 7.5, 0.8, 1.25))

    # --- Score computation ---
    base_score = _normalize_scores([even_ratio, odd_ratio])
    combined = float(np.clip(imbalance * 1.2 + base_score * 0.6, 0.0, 1.0))
    weighted = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # --- Dynamic threshold based on histogram spread ---
    hist, _ = np.histogram(diffs, bins=32, range=(-32, 32))
    probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-12)))
    dyn_threshold = float(np.clip(0.45 + 0.1 * (hist_entropy / np.log2(32)), 0.25, 0.8))

    # --- Final score ---
    score = _rescale_score(weighted, mean_ref=0.45, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    # --- Return structured result ---
    details = {
        "even_ratio": float(even_ratio),
        "odd_ratio": float(odd_ratio),
        "imbalance": float(imbalance),
        "entropy": float(entropy),
        "hist_entropy": float(hist_entropy),
        "dynamic_threshold": float(dyn_threshold),
        "n_pairs": int(total)
    }

    return {
        "method": "sample_pair_analysis",
        "score": float(score),
        "detected": bool(detected),
        "details": details
    }
