import numpy as np
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
# Analyze (Blockiness metric)
# ----------------------
def analyze(image_bytes=None, pil_image=None, block_size=8, threshold=0.5):
    """
    Blockiness / JPEG Artifact Analysis.
    Measures local discontinuities across 8x8 DCT block boundaries.
    Returns structured result for aggregation layer.
    """
    # --- Load image ---
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("L")

    arr = np.array(pil_image, dtype=np.float32)
    h, w = arr.shape

    # Ensure multiple of block size
    h_adj = h - (h % block_size)
    w_adj = w - (w % block_size)
    arr = arr[:h_adj, :w_adj]

    # --- Compute vertical and horizontal block boundaries ---
    vert_boundaries = np.arange(block_size, w_adj, block_size)
    horiz_boundaries = np.arange(block_size, h_adj, block_size)

    vert_diffs = np.zeros(len(vert_boundaries))
    horiz_diffs = np.zeros(len(horiz_boundaries))

    for i, vb in enumerate(vert_boundaries):
        left = arr[:, vb - 1]
        right = arr[:, vb]
        vert_diffs[i] = np.mean(np.abs(left - right))

    for i, hb in enumerate(horiz_boundaries):
        top = arr[hb - 1, :]
        bottom = arr[hb, :]
        horiz_diffs[i] = np.mean(np.abs(top - bottom))

    # --- Compute intra-block variation for reference ---
    block_means = []
    for y in range(0, h_adj, block_size):
        for x in range(0, w_adj, block_size):
            block = arr[y:y + block_size, x:x + block_size]
            block_means.append(np.mean(np.abs(block - np.mean(block))))
    block_means = np.array(block_means, dtype=np.float32)

    mean_block_var = float(np.mean(block_means))
    mean_edge_diff = float(np.mean(np.concatenate([vert_diffs, horiz_diffs])))
    blockiness_ratio = float(mean_edge_diff / (mean_block_var + 1e-12))

    # --- Entropy weighting ---
    entropy = _compute_shannon_entropy(arr)
    entropy_weight = float(np.clip(entropy / 7.5, 0.8, 1.25))

    # --- Core score ---
    normalized_edge = _normalize_scores(np.concatenate([vert_diffs, horiz_diffs]))
    normalized_var = _normalize_scores(block_means)
    combined_raw = float(np.clip(0.6 * normalized_edge + 0.4 * (1 - normalized_var), 0.0, 1.0))

    weighted = float(np.clip(combined_raw * entropy_weight * blockiness_ratio, 0.0, 1.0))

    # --- Dynamic threshold ---
    hist, _ = np.histogram(block_means, bins=16)
    probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-12)))
    dyn_threshold = float(np.clip(0.4 + 0.1 * (hist_entropy / np.log2(16)) + 0.05 * (blockiness_ratio - 1.0), 0.25, 0.85))

    # --- Final score ---
    score = _rescale_score(weighted, mean_ref=0.45, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    # --- Return structured result ---
    details = {
        "mean_edge_diff": float(mean_edge_diff),
        "mean_block_var": float(mean_block_var),
        "blockiness_ratio": float(blockiness_ratio),
        "entropy": float(entropy),
        "hist_entropy": float(hist_entropy),
        "dynamic_threshold": float(dyn_threshold),
        "n_blocks": int((h_adj / block_size) * (w_adj / block_size))
    }

    return {
        "method": "blockiness_analysis",
        "score": float(score),
        "detected": bool(detected),
        "details": details
    }
