import numpy as np
from skimage import restoration
from PIL import Image, ImageOps
import io
import cv2

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

def _remove_exif_temp(pil_image):
    """Return PIL.Image with EXIF removed (pixel copy)."""
    arr = np.array(pil_image)
    return Image.fromarray(arr)

def _resize_image(pil_image, max_dim=1024):
    w, h = pil_image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        return pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_image

def _convert_to_luminance(pil_image, space="YCbCr"):
    """Return uint8 luminance channel: Y (YCbCr) or L (Lab via cv2)."""
    if space and space.lower() == "lab":
        rgb = pil_image.convert("RGB")
        arr = np.array(rgb)[:, :, ::-1]  # RGB -> BGR
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        return L.astype(np.uint8)
    else:
        ycbcr = pil_image.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        return np.array(y, dtype=np.uint8)

def _prefilter_uint8(arr_uint8, mode="highpass"):
    a = arr_uint8.astype(np.float32)
    if mode == "none":
        return arr_uint8
    if mode == "laplacian":
        lap = cv2.Laplacian(a, ddepth=cv2.CV_32F, ksize=3)
        lap = lap - lap.min()
        if lap.max() <= 0:
            return np.zeros_like(arr_uint8)
        out = (lap / (lap.max() + 1e-12) * 255.0).astype(np.uint8)
        return out
    if mode == "log":
        g = cv2.GaussianBlur(a, (0, 0), sigmaX=1.0)
        lap = cv2.Laplacian(g, ddepth=cv2.CV_32F, ksize=3)
        lap = lap - lap.min()
        if lap.max() <= 0:
            return np.zeros_like(arr_uint8)
        out = (lap / (lap.max() + 1e-12) * 255.0).astype(np.uint8)
        return out
    # default highpass/hypas
    low = cv2.GaussianBlur(a, (0, 0), sigmaX=2.0)
    high = a - low
    high = high - high.min()
    if high.max() <= 0:
        return np.zeros_like(arr_uint8)
    out = (high / (high.max() + 1e-12) * 255.0).astype(np.uint8)
    return out

def _patch_statistics(patch):
    v = patch.ravel().astype(np.float64)
    mean = float(np.mean(v))
    std = float(np.std(v))
    if std > 0:
        skew = float(np.mean(((v - mean) / std) ** 3))
        kurt = float(np.mean(((v - mean) / std) ** 4) - 3.0)
    else:
        skew = 0.0; kurt = 0.0
    energy = float(np.sum(v ** 2))
    return {"mean": mean, "std": std, "skew": skew, "kurt": kurt, "energy": energy}

def _compute_block_heatmap_from_values(values_grid):
    """Normalize 2D grid to 0..1 (handles constant arrays)."""
    grid = np.array(values_grid, dtype=np.float32)
    if grid.size == 0:
        return np.zeros((1,1), dtype=np.float32)
    mn = grid.min(); ptp = grid.ptp()
    if ptp <= 1e-12:
        return np.clip(grid, 0.0, 1.0)
    return (grid - mn) / (ptp + 1e-12)

def _downsample_map_from_residuals(residual_float, block_size=16, max_map_dim=64):
    """Compute downsampled map based on mean absolute residual per block."""
    mag = np.abs(residual_float)
    h, w = mag.shape
    ph = ((h + block_size - 1) // block_size) * block_size
    pw = ((w + block_size - 1) // block_size) * block_size
    pad = np.zeros((ph, pw), dtype=np.float32)
    pad[:h, :w] = mag
    bh = ph // block_size
    bw = pw // block_size
    block_avg = pad.reshape(bh, block_size, bw, block_size).mean(axis=(1,3))
    if max(block_avg.shape) > max_map_dim:
        scale = max_map_dim / float(max(block_avg.shape))
        new_h = max(1, int(block_avg.shape[0] * scale))
        new_w = max(1, int(block_avg.shape[1] * scale))
        block_avg = cv2.resize(block_avg, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # normalize to 0..1
    if block_avg.size == 0:
        return block_avg
    mn = block_avg.min(); ptp = block_avg.ptp()
    if ptp <= 1e-12:
        return np.clip(block_avg, 0.0, 1.0)
    return ((block_avg - mn) / (ptp + 1e-12)).astype(np.float32)

# ----------------------
# Main analyze (enhanced noise residuals)
# ----------------------
def analyze(image_bytes=None,
            pil_image=None,
            patch_size=128,
            stride=64,
            threshold=0.6,
            max_dim=1024,
            color_space="YCbCr",
            prefilter="highpass",
            heatmap_block_size=16):
    """
    Noise residuals: denoise (wavelet), residual = image - denoised.
    Extract per-patch higher-order stats, map to score (enhanced).
    Returns: {"method","score","detected","details"}
    """
    # --- load and preprocess
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil = Image.open(io.BytesIO(image_bytes))
    else:
        pil = pil_image.copy()

    pil_proc = _remove_exif_temp(pil)
    pil_proc = ImageOps.exif_transpose(pil_proc)
    pil_proc = _resize_image(pil_proc, max_dim=max_dim)

    # work on luminance
    lum = _convert_to_luminance(pil_proc, space=color_space)
    lum_uint8 = np.array(lum, dtype=np.uint8)

    # optional prefilter on luminance (to emphasize HF residuals)
    work_uint8 = _prefilter_uint8(lum_uint8, mode=prefilter) if prefilter and prefilter != "none" else lum_uint8.copy()

    arr = work_uint8.astype(np.float32) / 255.0

    # denoise (wavelet) -> compute residual (keep float signed residual)
    try:
        den = restoration.denoise_wavelet(arr, channel_axis=None, rescale_sigma=True)
    except TypeError:
        den = restoration.denoise_wavelet(arr, multichannel=False, convert2ycbcr=False)
    residual = arr - den  # float in roughly [-1,1]

    h, w = residual.shape
    patch_stats = []
    block_grid = []

    ys = list(range(0, max(1, h - patch_size + 1), max(1, stride)))
    xs = list(range(0, max(1, w - patch_size + 1), max(1, stride)))

    for y0 in ys:
        row_vals = []
        for x0 in xs:
            patch = residual[y0:y0+patch_size, x0:x0+patch_size]
            if patch.size == 0:
                row_vals.append(0.0)
                continue
            # convert patch to scaled int16 like original code did for stats
            patch_int = np.clip((patch * 255.0), -32768, 32767).astype(np.int16)
            st = _patch_statistics(patch_int)
            patch_stats.append(st)
            # use energy or absolute-kurtosis magnitude as block value (energy emphasises anomalies)
            val = st["energy"]
            row_vals.append(val)
        block_grid.append(row_vals)

    if len(patch_stats) == 0:
        return {"method": "noise_residuals", "score": 0.0, "detected": False, "details": {"error": "image too small"}}

    energies = np.array([p["energy"] for p in patch_stats], dtype=np.float64)
    mean_skew = float(np.mean([p["skew"] for p in patch_stats]))
    mean_kurt = float(np.mean([p["kurt"] for p in patch_stats]))
    mean_energy = float(np.mean(energies))
    std_energy = float(np.std(energies))

    # original raw mapping (kept but bounded)
    score_raw = (abs(mean_kurt) / 10.0) + min(1.0, std_energy / (mean_energy + 1e-12) / 10.0)
    score_raw = float(np.clip(score_raw, 0.0, 1.0))

    # normalized energy distribution
    norm_energy = _normalize_scores(energies.tolist()) if energies.size > 0 else score_raw

    # entropy of residuals (image-level) — compute on scaled residual magnitude mapped to 0..255
    residual_uint8 = np.clip(((residual - residual.min()) / (residual.ptp() + 1e-12) * 255.0), 0, 255).astype(np.uint8)
    entropy = _compute_shannon_entropy(residual_uint8)
    entropy_weight = float(np.clip(entropy / 7.5, 0.75, 1.35))

    combined = float(np.clip(0.65 * score_raw + 0.35 * norm_energy, 0.0, 1.0))
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold based on energy histogram & entropy
    hist, _ = np.histogram(energies, bins=8)
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    dyn_threshold = float(np.clip(0.45 + 0.12 * (hist_entropy / np.log2(8)) + 0.04 * ((entropy - 2.5) / 4.0), 0.2, 0.85))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    # heatmaps:
    heatmap_blocks = _compute_block_heatmap_from_values(block_grid)  # normalized block grid (energies)
    down_map = _downsample_map_from_residuals(residual, block_size=heatmap_block_size, max_map_dim=64)

    details = {
        "mean_skew": mean_skew, "mean_kurt": mean_kurt,
        "mean_energy": mean_energy, "std_energy": std_energy,
        "n_patches": len(patch_stats),
        "entropy": float(entropy),
        "combined_raw": float(combined),
        "dynamic_threshold": float(dyn_threshold),
        "heatmap_blocks": heatmap_blocks.tolist(),
        "heatmap": down_map.tolist(),
        "params": {
            "patch_size": int(patch_size),
            "stride": int(stride),
            "max_dim": int(max_dim),
            "color_space": color_space,
            "prefilter": prefilter,
            "heatmap_block_size": int(heatmap_block_size)
        }
    }

    return {"method": "noise_residuals", "score": float(score), "detected": bool(detected), "details": details}
