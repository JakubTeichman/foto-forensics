import numpy as np
import cv2
from PIL import Image
import io

# ----------------------
# Helpers (local)
# ----------------------
def _compute_shannon_entropy(arr, bins=None):
    if bins is None:
        # arr contains quantized small-range values -> choose range from min..max
        vals = arr.ravel()
        mn = int(vals.min()); mx = int(vals.max())
        bins = max(2, mx - mn + 1)
        hist, _ = np.histogram(vals, bins=bins, range=(mn, mx+1))
    else:
        hist, _ = np.histogram(arr.ravel(), bins=bins)
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
def _highpass(img):
    kernel = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)

def _quantize(arr, q=5):
    clipped = np.clip(np.round(arr), -q, q).astype(np.int8)
    shifted = clipped + q
    return shifted

# ----------------------
# analyze (updated)
# ----------------------
def analyze(image_bytes=None, pil_image=None, q=4, offsets=None, threshold=0.6):
    """
    Residual co-occurrence (simplified SRM-like) - updated with normalization, entropy weighting and dynamic threshold.
    Returns dict: method, score (0..1), detected, details
    """
    if offsets is None:
        offsets = [(0,1), (1,0), (1,1)]

    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.float32)
    res = _highpass(arr)
    qarr = _quantize(res, q=q)
    maxv = 2 * q + 1

    feats = []
    details = {}
    h, w = qarr.shape

    for dx, dy in offsets:
        H = np.zeros((maxv, maxv), dtype=np.float64)
        for y in range(0, h - dy):
            for x in range(0, w - dx):
                a = int(qarr[y, x])
                b = int(qarr[y + dy, x + dx])
                if 0 <= a < maxv and 0 <= b < maxv:
                    H[a, b] += 1
        total = H.sum() + 1e-12
        p = H / total
        p_nonzero = p[p > 0]
        ent = -np.sum(p_nonzero * np.log2(p_nonzero + 1e-12)) if p_nonzero.size > 0 else 0.0
        energy = float(np.sum(H**2))
        feats.append({"entropy": float(ent), "energy": energy})
        details[f"offset_{dx}_{dy}"] = {"entropy": float(ent), "energy": energy}

    if len(feats) == 0:
        score = 0.0
        detected = False
        return {"method": "cooccurrence_analysis", "score": score, "detected": detected, "details": {"error": "no features"}}

    ent_mean = float(np.mean([f["entropy"] for f in feats]))
    max_ent = np.log2((2*q+1)**2)
    raw_score = float(np.clip(ent_mean / (max_ent + 1e-12), 0.0, 1.0))

    # normalize energies to capture distribution differences
    energies = [f["energy"] for f in feats]
    norm_energy = _normalize_scores(energies)

    # entropy of quantized residuals (global)
    global_q_entropy = _compute_shannon_entropy(qarr, bins=None)
    entropy_weight = float(np.clip((global_q_entropy / (np.log2(maxv**2)+1e-12)), 0.7, 1.3))

    combined = float(np.clip(0.7 * raw_score + 0.3 * norm_energy, 0.0, 1.0))
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold from offsets entropy spread
    ent_list = [f["entropy"] for f in feats]
    hist, _ = np.histogram(ent_list, bins=6, range=(0, max_ent if max_ent>0 else 1))
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    dyn_threshold = float(np.clip(0.4 + 0.12 * (hist_entropy / np.log2(6)) + 0.05 * ((global_q_entropy - 2.0)/4.0), 0.2, 0.8))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    details.update({
        "ent_mean": ent_mean,
        "max_ent": max_ent,
        "global_q_entropy": global_q_entropy,
        "combined_raw": combined,
        "dynamic_threshold": dyn_threshold
    })

    return {"method": "cooccurrence_analysis", "score": float(score), "detected": bool(detected), "details": details}
