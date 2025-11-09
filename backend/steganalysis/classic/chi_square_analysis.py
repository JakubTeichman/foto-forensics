import numpy as np
from scipy.stats import chi2
from PIL import Image
import io

# ----------------------
# Helper functions (local)
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
        return float(np.mean(a))  # all equal
    norm = (a - mn) / (mx - mn)
    return float(np.clip(np.mean(norm), 0.0, 1.0))

def _rescale_score(value, mean_ref=0.5, scale_ref=0.25):
    # center around mean_ref, scale by scale_ref, and clamp to [0,1]
    val = (value - mean_ref) / (scale_ref + 1e-12) / 2.0 + 0.5
    return float(np.clip(val, 0.0, 1.0))

# ----------------------
# original chi2 helper
# ----------------------
def _chi2_for_array(arr):
    hist = np.bincount(arr.ravel().astype(np.uint8), minlength=256).astype(np.float64)
    pairs = hist.reshape(128, 2)
    evens = pairs[:, 0]; odds = pairs[:, 1]
    denom = evens + odds
    mask = denom > 0
    chi_terms = np.zeros_like(denom)
    chi_terms[mask] = (evens[mask] - odds[mask])**2 / (denom[mask] + 1e-12)
    chi_stat = float(np.sum(chi_terms))
    df = int(np.count_nonzero(mask) - 1)
    if df <= 0:
        p = 1.0
    else:
        p = float(1.0 - chi2.cdf(chi_stat, df))
    return chi_stat, p, df

# ----------------------
# analyze (updated)
# ----------------------
def analyze(image_bytes=None, pil_image=None, block_size=64, stride=64, threshold=0.5):
    """
    Chi-square PoV detector (patch-aware) - updated with normalization, entropy weighting and dynamic threshold.
    Returns dict: method, score (0..1), detected, details
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.uint8)
    h, w = arr.shape

    block_pvalues = []
    block_stats = []

    for y0 in range(0, max(1, h - block_size + 1), stride or block_size):
        for x0 in range(0, max(1, w - block_size + 1), stride or block_size):
            blk = arr[y0:y0+block_size, x0:x0+block_size]
            if blk.size == 0:
                continue
            chi_stat, p, df = _chi2_for_array(blk)
            block_pvalues.append(p)
            block_stats.append({"x": x0, "y": y0, "chi2": chi_stat, "p": p, "df": df})

    # combine p-values (Fisher) -> global_p
    if len(block_pvalues) == 0:
        global_p = 1.0
    else:
        T = -2.0 * sum(np.log(np.clip(block_pvalues, 1e-12, 1.0)))
        df_tot = 2 * len(block_pvalues)
        global_p = float(max(0.0, min(1.0, 1.0 - chi2.cdf(T, df_tot))))

    # raw score (1 - p) then normalize / rescale
    raw_score = float(max(0.0, min(1.0, 1.0 - global_p)))

    # normalize block-wise p-values to get dispersion-based modifier
    norm_block_score = _normalize_scores([1.0 - p for p in block_pvalues]) if block_pvalues else raw_score

    # entropy-based weight
    entropy = _compute_shannon_entropy(arr)
    entropy_weight = float(np.clip(entropy / 7.5, 0.7, 1.3))  # empirical scaling

    # combine raw, normalized and entropy weight
    combined = 0.6 * raw_score + 0.4 * norm_block_score
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold: histogram dispersion + entropy
    hist, _ = np.histogram([combined], bins=10, range=(0,1))
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    dyn_threshold = float(np.clip(0.4 + 0.15 * (hist_entropy / np.log2(10)) + 0.05 * ((entropy - 4.0) / 4.0), 0.2, 0.8))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = score >= dyn_threshold

    details = {
        "blocks": block_stats,
        "global_p": global_p,
        "n_blocks": len(block_pvalues),
        "entropy": entropy,
        "combined_raw": combined,
        "dynamic_threshold": dyn_threshold
    }
    return {"method": "chi_square", "score": float(score), "detected": bool(detected), "details": details}
