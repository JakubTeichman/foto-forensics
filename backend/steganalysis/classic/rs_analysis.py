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
# original group function
# ----------------------
def _group_function(block):
    p = block.astype(np.int32)
    s = 0
    if p.shape[1] > 1:
        s += np.abs(p[:,1:] - p[:,:-1]).sum()
    if p.shape[0] > 1:
        s += np.abs(p[1:,:] - p[:-1,:]).sum()
    return float(s)

# ----------------------
# analyze (updated)
# ----------------------
def analyze(image_bytes=None, pil_image=None, group_size=2, threshold=0.4):
    """
    RS analysis (simplified Fridrich/Goljan style) - updated with normalization, entropy weighting and dynamic threshold.
    Returns dict: method, score (0..1), detected, details
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.uint8)
    h, w = arr.shape

    R0 = S0 = Rf = Sf = 0.0
    total_groups = 0
    group_changes = []
    group_scores = []

    step = group_size
    for y0 in range(0, h - group_size + 1, step):
        for x0 in range(0, w - group_size + 1, step):
            blk = arr[y0:y0+group_size, x0:x0+group_size]
            if blk.size == 0:
                continue
            f_orig = _group_function(blk)
            flipped_blk = blk ^ 1
            f_flipped = _group_function(flipped_blk)
            if f_orig > f_flipped:
                R0 += 1
                group_scores.append(1.0)
            else:
                S0 += 1
                group_scores.append(0.0)

            mask = np.zeros_like(blk)
            mask[::2, ::2] = 1
            blk_masked = blk ^ mask
            f_masked = _group_function(blk_masked)
            if f_masked > f_flipped:
                Rf += 1
            else:
                Sf += 1

            total_groups += 1
            if len(group_changes) < 20:
                group_changes.append({"x": x0, "y": y0, "f_orig": f_orig, "f_flipped": f_flipped, "f_masked": f_masked})

    if total_groups == 0:
        raw_score = 0.0
    else:
        est = ((R0 - Rf) + (Sf - S0)) / (2.0 * total_groups + 1e-12)
        raw_score = float(max(0.0, min(1.0, abs(est))))

    # normalize group-wise distribution
    norm_group = _normalize_scores(group_scores) if group_scores else raw_score

    # entropy weight
    entropy = _compute_shannon_entropy(arr)
    entropy_weight = float(np.clip(entropy / 7.5, 0.75, 1.25))

    combined = float(np.clip(0.7 * raw_score + 0.3 * norm_group, 0.0, 1.0))
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold from histogram of group scores and entropy
    if group_scores:
        hist, _ = np.histogram(group_scores, bins=6, range=(0,1))
        hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
        hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    else:
        hist_entropy = 0.0
    dyn_threshold = float(np.clip(0.35 + 0.18 * (hist_entropy / np.log2(6)) + 0.06 * ((entropy - 4.0) / 4.0), 0.2, 0.8))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = score >= dyn_threshold

    details = {
        "R0": int(R0), "S0": int(S0), "Rf": int(Rf), "Sf": int(Sf),
        "groups": total_groups, "group_examples": group_changes,
        "entropy": entropy, "combined_raw": combined, "dynamic_threshold": dyn_threshold
    }
    return {"method": "rs_analysis", "score": float(score), "detected": bool(detected), "details": details}
