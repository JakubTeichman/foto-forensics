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
# original helpers
# ----------------------
def _lsb_prop_channel(arr):
    flat = arr.ravel().astype(np.uint8)
    ones = int((flat & 1).sum())
    total = int(flat.size)
    return ones / (total + 1e-12), ones, total

def _moran_i(bits_arr):
    b = bits_arr.astype(float)
    n = b.size
    if n <= 1:
        return 0.0
    mean = b.mean()
    num = 0.0
    denom = ((b - mean)**2).sum() + 1e-12
    for i in range(n - 1):
        num += (b[i] - mean) * (b[i + 1] - mean)
    I = (n / (n - 1)) * (num / (denom + 1e-12))
    return float(I)

# ----------------------
# analyze (updated)
# ----------------------
def analyze(image_bytes=None, pil_image=None, patch_size=128, stride=64, threshold=0.6):
    """
    LSB histogram + patch-aware detector (updated).
    Returns dict: method, score (0..1), detected, details
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    img = np.array(pil_image, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    h, w, _ = img.shape
    y = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)

    channel_scores = []
    details = {}

    # per-channel coarse score
    for ch_name, ch_arr in [('Y', y), ('R', img[..., 0]), ('G', img[..., 1]), ('B', img[..., 2])]:
        prop, ones, total = _lsb_prop_channel(ch_arr)
        score_ch = max(0.0, 1.0 - 2.0 * abs(prop - 0.5))
        I = _moran_i((ch_arr.ravel() & 1))
        details[ch_name] = {"lsb_prop": float(prop), "ones": int(ones), "total": int(total), "score_raw": float(score_ch), "moran_I": float(I)}
        channel_scores.append(score_ch)

    # patch-wise scores
    patch_scores = []
    for y0 in range(0, max(1, h - patch_size + 1), max(1, stride)):
        for x0 in range(0, max(1, w - patch_size + 1), max(1, stride)):
            patch = y[y0:y0+patch_size, x0:x0+patch_size]
            if patch.size == 0:
                continue
            p_prop, p_ones, p_total = _lsb_prop_channel(patch)
            p_score = max(0.0, 1.0 - 2.0 * abs(p_prop - 0.5))
            patch_scores.append(p_score)

    patch_mean = float(np.mean(patch_scores)) if patch_scores else 0.0
    patch_std = float(np.std(patch_scores)) if patch_scores else 0.0

    # combine channel and patch scores
    raw_combined = 0.5 * float(np.mean(channel_scores)) + 0.5 * patch_mean

    # normalize patch distribution to reduce FP
    norm_patch = _normalize_scores(patch_scores) if patch_scores else raw_combined

    # entropy weight (image-level)
    entropy = _compute_shannon_entropy(y)
    entropy_weight = float(np.clip(entropy / 7.5, 0.75, 1.25))

    combined = float(np.clip(0.6 * raw_combined + 0.4 * norm_patch, 0.0, 1.0))
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold based on patch distribution histogram and entropy
    if patch_scores:
        hist, _ = np.histogram(patch_scores, bins=8, range=(0,1))
        hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
        hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    else:
        hist_entropy = 0.0
    dyn_threshold = float(np.clip(0.45 + 0.12 * (hist_entropy / np.log2(8)) + 0.05 * ((entropy - 4.0) / 4.0), 0.25, 0.8))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = score >= dyn_threshold

    details.update({
        "patch_mean": patch_mean,
        "patch_std": patch_std,
        "entropy": entropy,
        "combined_raw": combined,
        "dynamic_threshold": dyn_threshold
    })

    return {"method": "lsb_histogram", "score": float(score), "detected": bool(detected), "details": details}
