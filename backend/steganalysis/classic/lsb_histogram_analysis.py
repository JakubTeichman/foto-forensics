import numpy as np
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

def _remove_exif(pil_image):
    # Create a new image from pixels to drop metadata (temporary)
    data = list(pil_image.getdata())
    no_exif = Image.new(pil_image.mode, pil_image.size)
    no_exif.putdata(data)
    return no_exif

def _resize_image(pil_image, max_dim=512):
    w, h = pil_image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_image

def _convert_to_luminance_and_channels(pil_image, space="YCbCr"):
    """
    Returns luminance (2D uint8) and full numpy RGB array (H,W,3 uint8)
    """
    rgb = pil_image.convert("RGB")
    rgb_arr = np.array(rgb, dtype=np.uint8)
    if space.lower() == "lab":
        lab = rgb.convert("LAB")
        L = np.array(lab)[:, :, 0].astype(np.uint8)
        return L, rgb_arr
    else:
        ycbcr = rgb.convert("YCbCr")
        Y = np.array(ycbcr)[:, :, 0].astype(np.uint8)
        return Y, rgb_arr

def _prefilter(arr_uint8, mode="highpass"):
    """
    arr_uint8: 2D uint8
    modes: 'highpass' (HP via subtracting gaussian), 'laplacian', 'log', 'none'
    returns uint8 array (0..255) suitable for subsequent analysis
    """
    a = arr_uint8.astype(np.float32)
    if mode == "none":
        return arr_uint8
    if mode == "laplacian":
        lap = cv2.Laplacian(a, ddepth=cv2.CV_32F, ksize=3)
        out = np.clip(lap - lap.min(), 0, lap.max() - lap.min() + 1e-12)
        out = (out / (out.max() + 1e-12) * 255.0).astype(np.uint8)
        return out
    if mode == "log":
        g = cv2.GaussianBlur(a, (0,0), sigmaX=1.0)
        lap = cv2.Laplacian(g, ddepth=cv2.CV_32F, ksize=3)
        out = np.clip(lap - lap.min(), 0, lap.max() - lap.min() + 1e-12)
        out = (out / (out.max() + 1e-12) * 255.0).astype(np.uint8)
        return out
    # default highpass: subtract blurred version
    low = cv2.GaussianBlur(a, (0,0), sigmaX=2.0)
    high = a - low
    # shift to positive domain and rescale
    high_shift = high - high.min()
    if high_shift.max() <= 0:
        return np.zeros_like(arr_uint8)
    out = (high_shift / (high_shift.max() + 1e-12) * 255.0).astype(np.uint8)
    return out

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
    # linear adjacency (row-major)
    num = ((b[:-1] - mean) * (b[1:] - mean)).sum()
    I = (n / (n - 1)) * (num / (denom + 1e-12))
    return float(I)

def _compute_lsb_heatmap(arr_uint8, patch_size=16, max_map_dim=64):
    """
    Compute downsampled LSB heatmap:
    - arr_uint8: 2D uint8 (luminance or channel)
    - patch_size: block size to average LSB
    - returns small 2D float32 array values 0..1
    """
    bits = (arr_uint8 & 1).astype(np.float32)
    h, w = bits.shape
    ph = ((h + patch_size - 1) // patch_size) * patch_size
    pw = ((w + patch_size - 1) // patch_size) * patch_size
    pad = np.zeros((ph, pw), dtype=np.float32)
    pad[:h, :w] = bits
    bh = ph // patch_size
    bw = pw // patch_size
    block_avg = pad.reshape(bh, patch_size, bw, patch_size).mean(axis=(1,3))
    # downsample if large
    if max(block_avg.shape) > max_map_dim:
        scale = max_map_dim / float(max(block_avg.shape))
        new_h = max(1, int(block_avg.shape[0] * scale))
        new_w = max(1, int(block_avg.shape[1] * scale))
        block_avg = cv2.resize(block_avg, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return block_avg.astype(np.float32)

# ----------------------
# Main analyze (enhanced LSB histogram)
# ----------------------
def analyze(image_bytes=None, pil_image=None,
            patch_size=128, stride=64, threshold=0.6,
            max_dim=512, color_space="YCbCr", prefilter="highpass"):
    """
    LSB histogram + patch-aware detector (enhanced).
    - returns: {method, score, detected, details}
    - details includes heatmap (downsampled), entropy, combined_raw, dynamic_threshold, params
    """
    # load and preprocess image (temporary EXIF strip, resize)
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil = Image.open(io.BytesIO(image_bytes))
    else:
        pil = pil_image.copy()

    pil = _remove_exif(pil)
    pil = ImageOps.exif_transpose(pil)
    pil = _resize_image(pil, max_dim=max_dim)

    # get luminance and rgb
    lum, rgb_arr = _convert_to_luminance_and_channels(pil, space=color_space)
    # optionally prefilter on luminance
    work_lum = _prefilter(lum, mode=prefilter)

    h, w = work_lum.shape

    # per-channel coarse LSB analysis (Y, R, G, B)
    channel_scores = []
    details = {}
    channels = {"Y": work_lum, "R": rgb_arr[..., 0], "G": rgb_arr[..., 1], "B": rgb_arr[..., 2]}
    for ch_name, ch_arr in channels.items():
        prop, ones, total = _lsb_prop_channel(ch_arr)
        score_ch = max(0.0, 1.0 - 2.0 * abs(prop - 0.5))
        I = _moran_i((ch_arr.ravel() & 1))
        details[ch_name] = {"lsb_prop": float(prop), "ones": int(ones), "total": int(total), "score_raw": float(score_ch), "moran_I": float(I)}
        channel_scores.append(score_ch)

    # patch-wise LSB prop on luminance (work_lum)
    patch_scores = []
    heatmap_blocks = []
    ys = list(range(0, max(1, h - patch_size + 1), max(1, stride)))
    xs = list(range(0, max(1, w - patch_size + 1), max(1, stride)))
    for y0 in ys:
        row = []
        for x0 in xs:
            patch = work_lum[y0:y0+patch_size, x0:x0+patch_size]
            if patch.size == 0:
                row.append(0.5)
                continue
            p_prop, p_ones, p_total = _lsb_prop_channel(patch)
            p_score = max(0.0, 1.0 - 2.0 * abs(p_prop - 0.5))
            patch_scores.append(p_score)
            row.append(p_score)
        heatmap_blocks.append(row)

    heatmap_blocks = np.array(heatmap_blocks, dtype=np.float32) if heatmap_blocks else np.zeros((1,1), dtype=np.float32)
    # normalize heatmap blocks to 0..1
    heatmap_norm = (heatmap_blocks - heatmap_blocks.min()) / (heatmap_blocks.ptp() + 1e-12)

    patch_mean = float(np.mean(patch_scores)) if patch_scores else 0.0
    patch_std = float(np.std(patch_scores)) if patch_scores else 0.0

    # combine channel and patch scores
    raw_combined = 0.5 * float(np.mean(channel_scores)) + 0.5 * patch_mean

    # normalize patch distribution to reduce FP
    norm_patch = _normalize_scores(patch_scores) if patch_scores else raw_combined

    # entropy weight (image-level on original luminance)
    entropy = _compute_shannon_entropy(lum)
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
    dyn_threshold = float(np.clip(0.45 + 0.12 * (hist_entropy / np.log2(8)) + 0.05 * ((entropy - 4.0) / 4.0), 0.25, 0.85))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    # compute LSB heatmap (downsampled) from original luminance for frontend visualization
    lsb_heatmap = _compute_lsb_heatmap(lum, patch_size=16, max_map_dim=64)
    # If heatmap_blocks grid exists, also include it (same scale as patch grid)
    heatmap_blocks_list = heatmap_norm.tolist()

    details.update({
        "patch_mean": patch_mean,
        "patch_std": patch_std,
        "entropy": float(entropy),
        "combined_raw": float(combined),
        "dynamic_threshold": float(dyn_threshold),
        "heatmap_blocks": heatmap_blocks_list,
        "heatmap": lsb_heatmap.tolist(),
        "params": {
            "patch_size": int(patch_size),
            "stride": int(stride),
            "max_dim": int(max_dim),
            "color_space": color_space,
            "prefilter": prefilter
        }
    })

    return {"method": "lsb_histogram", "score": float(score), "detected": bool(detected), "details": details}
