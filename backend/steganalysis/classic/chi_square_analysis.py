import numpy as np
from scipy.stats import chi2
from scipy.ndimage import gaussian_filter, laplace
from PIL import Image, ImageOps
import io, tempfile, os

# ----------------------
# Helper functions
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
    mn, mx = a.min(), a.max()
    if mx - mn < 1e-12:
        return float(np.mean(a))
    norm = (a - mn) / (mx - mn)
    return float(np.clip(np.mean(norm), 0.0, 1.0))

def _rescale_score(value, mean_ref=0.5, scale_ref=0.25):
    val = (value - mean_ref) / (scale_ref + 1e-12) / 2.0 + 0.5
    return float(np.clip(val, 0.0, 1.0))

def _remove_exif(pil_image):
    # usunięcie metadanych EXIF
    data = list(pil_image.getdata())
    no_exif = Image.new(pil_image.mode, pil_image.size)
    no_exif.putdata(data)
    return no_exif

def _resize_image(pil_image, max_dim=512):
    w, h = pil_image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)))
    return pil_image

def _prefilter_image(arr, mode="hypos"):
    # Prefiltracja (High-pass / Laplacian / Gaussian / LoG)
    if mode == "laplacian":
        return np.clip(laplace(arr.astype(np.float32)), 0, 255)
    elif mode == "gaussian":
        return np.clip(gaussian_filter(arr.astype(np.float32), sigma=1), 0, 255)
    elif mode == "log":
        g = gaussian_filter(arr.astype(np.float32), sigma=1)
        return np.clip(laplace(g), 0, 255)
    elif mode == "highpass" or mode == "hypas":
        low = gaussian_filter(arr.astype(np.float32), sigma=2)
        high = arr.astype(np.float32) - low
        return np.clip(high + 128, 0, 255)
    else:
        return arr

def _convert_to_luminance(pil_image, space="YCbCr"):
    if space.lower() == "lab":
        lab = pil_image.convert("LAB")
        return np.array(lab)[:, :, 0]
    else:
        ycbcr = pil_image.convert("YCbCr")
        return np.array(ycbcr)[:, :, 0]

# ----------------------
# Chi-square test helper
# ----------------------
def _chi2_for_array(arr):
    hist = np.bincount(arr.ravel().astype(np.uint8), minlength=256).astype(np.float64)
    pairs = hist.reshape(128, 2)
    evens, odds = pairs[:, 0], pairs[:, 1]
    denom = evens + odds
    mask = denom > 0
    chi_terms = np.zeros_like(denom)
    chi_terms[mask] = (evens[mask] - odds[mask]) ** 2 / (denom[mask] + 1e-12)
    chi_stat = float(np.sum(chi_terms))
    df = int(np.count_nonzero(mask) - 1)
    if df <= 0:
        p = 1.0
    else:
        p = float(1.0 - chi2.cdf(chi_stat, df))
    return chi_stat, p, df

# ----------------------
# Main analyze
# ----------------------
def analyze(image_bytes=None, pil_image=None,
            block_size=64, stride=64, threshold=0.5,
            max_dim=512, color_space="YCbCr", prefilter="hypos"):
    """
    Chi-square PoV detector (enhanced):
    - prefilter (HyPAS/LoG/Laplacian)
    - dynamic threshold
    - entropy weighting
    - luminance-based analysis
    - heatmap output
    """
    # --- load image
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes))

    # --- preproc
    pil_image = _remove_exif(pil_image)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = _resize_image(pil_image, max_dim=max_dim)
    arr = _convert_to_luminance(pil_image, space=color_space)
    arr = _prefilter_image(arr, mode=prefilter).astype(np.uint8)

    h, w = arr.shape
    block_pvalues, heatmap = [], []

    # --- block analysis
    for y0 in range(0, max(1, h - block_size + 1), stride or block_size):
        row_vals = []
        for x0 in range(0, max(1, w - block_size + 1), stride or block_size):
            blk = arr[y0:y0 + block_size, x0:x0 + block_size]
            if blk.size == 0:
                row_vals.append(0.5)
                continue
            chi_stat, p, df = _chi2_for_array(blk)
            val = 1.0 - p
            block_pvalues.append(p)
            row_vals.append(val)
        heatmap.append(row_vals)

    heatmap = np.array(heatmap, dtype=np.float32)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-12)

    # --- global p-value
    if len(block_pvalues) == 0:
        global_p = 1.0
    else:
        T = -2.0 * sum(np.log(np.clip(block_pvalues, 1e-12, 1.0)))
        df_tot = 2 * len(block_pvalues)
        global_p = float(max(0.0, min(1.0, 1.0 - chi2.cdf(T, df_tot))))

    raw_score = float(max(0.0, min(1.0, 1.0 - global_p)))
    norm_block_score = _normalize_scores([1.0 - p for p in block_pvalues]) if block_pvalues else raw_score

    # --- entropy-based weighting
    entropy = _compute_shannon_entropy(arr)
    entropy_weight = float(np.clip(entropy / 7.5, 0.7, 1.3))

    combined = 0.6 * raw_score + 0.4 * norm_block_score
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # --- dynamic threshold
    hist, _ = np.histogram(heatmap_norm.ravel(), bins=20, range=(0, 1))
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(hist_probs[hist_probs > 0] * np.log2(hist_probs[hist_probs > 0] + 1e-12)))
    dyn_threshold = float(np.clip(0.4 + 0.15 * (hist_entropy / np.log2(10)) + 0.05 * ((entropy - 4.0) / 4.0), 0.2, 0.8))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = score >= dyn_threshold

    # --- details
    details = {
        "global_p": global_p,
        "n_blocks": len(block_pvalues),
        "entropy": entropy,
        "combined_raw": combined,
        "dynamic_threshold": dyn_threshold,
        "params": {
            "block_size": block_size,
            "stride": stride,
            "max_dim": max_dim,
            "color_space": color_space,
            "prefilter": prefilter
        },
        "heatmap": heatmap_norm.tolist()
    }

    return {
        "method": "chi_square",
        "score": float(score),
        "detected": bool(detected),
        "details": details
    }
