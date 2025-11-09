import numpy as np
import cv2
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

# ----------------------
# Helpers
# ----------------------
def _normalize_scores(values):
    """Normalize and average values between 0–1."""
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
# Analyze
# ----------------------
def analyze(image_bytes=None, pil_image=None, threshold=0.5):
    """
    Pixel-Pair Difference Histogram Analysis
    ----------------------------------------
    Detects steganographic traces based on abnormal distributions
    of pixel-pair differences in horizontal and vertical directions.

    Returns a heatmap and statistical summary similar to other methods.
    """

    # --- Load image ---
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = np.array(pil_image, dtype=np.float32)

    # --- Compute pixel-pair differences (horizontal & vertical) ---
    diff_h = np.abs(img[:, 1:] - img[:, :-1])
    diff_v = np.abs(img[1:, :] - img[:-1, :])

    # --- Create combined difference map ---
    diff_combined = 0.5 * (diff_h[:-1, :] + diff_v[:, :-1])
    diff_combined_norm = cv2.normalize(diff_combined, None, 0, 1, cv2.NORM_MINMAX)

    # --- Compute histograms ---
    hist_h, _ = np.histogram(diff_h, bins=32, range=(0, 255), density=True)
    hist_v, _ = np.histogram(diff_v, bins=32, range=(0, 255), density=True)
    hist_diff = np.abs(hist_h - hist_v)
    hist_entropy = -np.sum(hist_diff * np.log2(hist_diff + 1e-12))

    # --- Extract statistical features ---
    mean_diff = np.mean(diff_combined)
    std_diff = np.std(diff_combined)
    var_ratio = std_diff / (mean_diff + 1e-12)
    entropy_norm = float(np.clip(hist_entropy / 5.0, 0.0, 1.0))

    # --- Combine metrics into one score ---
    irregularity_score = _normalize_scores([var_ratio, entropy_norm])
    score = _rescale_score(irregularity_score, mean_ref=0.55, scale_ref=0.25)
    detected = bool(score > threshold)

    # --- Generate heatmap (difference map visualization) ---
    plt.figure(figsize=(6, 6))
    plt.imshow(diff_combined_norm, cmap='inferno')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap_encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

    # --- Prepare output details ---
    details = {
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "var_ratio": float(var_ratio),
        "entropy_norm": float(entropy_norm),
        "irregularity_score": float(irregularity_score)
    }

    return {
        "method": "pixel_pair_histogram",
        "score": float(score),
        "detected": bool(detected),
        "heatmap": heatmap_encoded,
        "details": details
    }
