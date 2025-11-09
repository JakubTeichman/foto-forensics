import numpy as np
import cv2
from PIL import Image, ImageOps
import io

# ----------------------
# Helpers (local)
# ----------------------
def _compute_shannon_entropy(arr, bins=None):
    vals = arr.ravel()
    if bins is None:
        mn = int(vals.min()); mx = int(vals.max())
        bins = max(2, mx - mn + 1)
        hist, _ = np.histogram(vals, bins=bins, range=(mn, mx+1))
    else:
        hist, _ = np.histogram(vals, bins=bins)
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
    """Return new PIL.Image without EXIF (pixel copy)."""
    arr = np.array(pil_image)
    return Image.fromarray(arr)

def _resize_image(pil_image, max_dim=1024):
    w, h = pil_image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        return pil_image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return pil_image

def _convert_to_luminance(pil_image, space="YCbCr"):
    """Return uint8 luminance channel: Y (YCbCr) or L (Lab via cv2)."""
    if space and space.lower() == "lab":
        rgb = pil_image.convert("RGB")
        arr = np.array(rgb)[:, :, ::-1]  # RGB -> BGR for cv2
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        return L.astype(np.uint8)
    else:
        ycbcr = pil_image.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        return np.array(y, dtype=np.uint8)

def _prefilter(arr_uint8, mode="highpass"):
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
        g = cv2.GaussianBlur(a, (0,0), sigmaX=1.0)
        lap = cv2.Laplacian(g, ddepth=cv2.CV_32F, ksize=3)
        lap = lap - lap.min()
        if lap.max() <= 0:
            return np.zeros_like(arr_uint8)
        out = (lap / (lap.max() + 1e-12) * 255.0).astype(np.uint8)
        return out
    # default highpass/hypas
    low = cv2.GaussianBlur(a, (0,0), sigmaX=2.0)
    high = a - low
    high = high - high.min()
    if high.max() <= 0:
        return np.zeros_like(arr_uint8)
    out = (high / (high.max() + 1e-12) * 255.0).astype(np.uint8)
    return out

def _highpass_float(img):
    kernel = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)

def _quantize(arr, q=5):
    clipped = np.clip(np.round(arr), -q, q).astype(np.int8)
    shifted = clipped + q
    return shifted

def _compute_block_heatmap_from_values(values_grid):
    """Normalize 2D grid to 0..1 (handles constant arrays)."""
    grid = np.array(values_grid, dtype=np.float32)
    if grid.size == 0:
        return np.zeros((1,1), dtype=np.float32)
    mn = grid.min(); ptp = grid.ptp()
    if ptp <= 1e-12:
        return np.clip(grid, 0.0, 1.0)
    return (grid - mn) / (ptp + 1e-12)

def _downsample_to_map(arr_uint8, block_size=16, max_map_dim=64):
    """Produce downsampled map (e.g., entropy or LSB-like) for frontend."""
    # compute local entropy per non-overlapping blocks
    bits = (arr_uint8 & 1).astype(np.float32)
    h, w = bits.shape
    ph = ((h + block_size - 1) // block_size) * block_size
    pw = ((w + block_size - 1) // block_size) * block_size
    pad = np.zeros((ph, pw), dtype=np.float32)
    pad[:h, :w] = bits
    bh = ph // block_size
    bw = pw // block_size
    block_avg = pad.reshape(bh, block_size, bw, block_size).mean(axis=(1,3))
    if max(block_avg.shape) > max_map_dim:
        scale = max_map_dim / float(max(block_avg.shape))
        new_h = max(1, int(block_avg.shape[0] * scale))
        new_w = max(1, int(block_avg.shape[1] * scale))
        block_avg = cv2.resize(block_avg, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return block_avg.astype(np.float32)

# ----------------------
# Main analyze (enhanced co-occurrence)
# ----------------------
def analyze(image_bytes=None,
            pil_image=None,
            q=4,
            offsets=None,
            threshold=0.6,
            max_dim=1024,
            color_space="YCbCr",
            prefilter="highpass",
            patch_size=64,
            stride=64,
            heatmap_block_size=16):
    """
    Residual co-occurrence (SRM-like) - enhanced:
      - temp EXIF strip, resize, convert to luminance (YCbCr/Lab)
      - optional prefilter (highpass/laplacian/log/none)
      - patch-wise co-occurrence matrices -> entropy & energy per patch
      - block-level heatmap + downsampled map for frontend
      - normalization, entropy weighting, dynamic threshold
    Returns: {"method","score","detected","details"}
    """
    if offsets is None:
        offsets = [(0,1), (1,0), (1,1)]

    # load + preprocess
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil = Image.open(io.BytesIO(image_bytes))
    else:
        pil = pil_image.copy()

    pil_proc = _remove_exif_temp(pil)
    pil_proc = ImageOps.exif_transpose(pil_proc)
    pil_proc = _resize_image(pil_proc, max_dim=max_dim)

    # luminance
    lum = _convert_to_luminance(pil_proc, space=color_space)
    # prefilter (to emphasize residuals)
    if prefilter and prefilter != "none":
        work = _prefilter(lum, mode=prefilter)
    else:
        work = lum.copy()

    # residual: use highpass (float) before quantization (keeps sign)
    residual = _highpass_float(work)
    qarr = _quantize(residual, q=q)
    maxv = 2 * q + 1

    h, w = qarr.shape
    patch_stats = []       # per-patch ent/energy
    block_grid = []        # grid same shape as patch tiling

    ys = list(range(0, max(1, h - patch_size + 1), max(1, stride)))
    xs = list(range(0, max(1, w - patch_size + 1), max(1, stride)))

    for y0 in ys:
        row_vals = []
        for x0 in xs:
            patch = qarr[y0:y0+patch_size, x0:x0+patch_size]
            if patch.size == 0:
                row_vals.append(0.0)
                continue

            # build co-occurrence histogram for this patch over given offsets
            H_tot = np.zeros((maxv, maxv), dtype=np.float64)
            ent_sum = 0.0
            energy_sum = 0.0
            for dx, dy in offsets:
                H = np.zeros((maxv, maxv), dtype=np.float64)
                for yy in range(0, patch.shape[0] - dy):
                    for xx in range(0, patch.shape[1] - dx):
                        a = int(patch[yy, xx])
                        b = int(patch[yy + dy, xx + dx])
                        if 0 <= a < maxv and 0 <= b < maxv:
                            H[a, b] += 1
                total = H.sum() + 1e-12
                p = H / total
                p_nonzero = p[p > 0]
                ent = -np.sum(p_nonzero * np.log2(p_nonzero + 1e-12)) if p_nonzero.size > 0 else 0.0
                energy = float(np.sum(H**2))
                ent_sum += ent
                energy_sum += energy
                H_tot += H

            # average over offsets
            ent_avg = float(ent_sum / len(offsets))
            energy_avg = float(energy_sum / len(offsets))
            patch_stats.append({"x": x0, "y": y0, "entropy": ent_avg, "energy": energy_avg})
            # choose block value: entropy normalized later (higher ent -> more "uniform" residual co-occurrence)
            row_vals.append(ent_avg)
        block_grid.append(row_vals)

    if len(patch_stats) == 0:
        return {"method": "cooccurrence_analysis", "score": 0.0, "detected": False, "details": {"error": "no patches"}} 

    # collect entropies and energies
    ent_list = np.array([p["entropy"] for p in patch_stats], dtype=np.float64)
    energy_list = np.array([p["energy"] for p in patch_stats], dtype=np.float64)

    # raw score: normalized mean entropy (relative to theoretical max)
    ent_mean = float(ent_list.mean())
    max_ent = np.log2((2*q+1)**2)
    raw_score = float(np.clip(ent_mean / (max_ent + 1e-12), 0.0, 1.0))

    # normalized energy distribution helps capture structure changes
    norm_energy = _normalize_scores(energy_list.tolist())

    # global quantized residual entropy (image-level)
    global_q_entropy = _compute_shannon_entropy(qarr, bins=None)
    entropy_weight = float(np.clip((global_q_entropy / (np.log2(maxv**2)+1e-12)), 0.7, 1.3))

    combined = float(np.clip(0.7 * raw_score + 0.3 * norm_energy, 0.0, 1.0))
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold from patch entropy histogram + global residual entropy
    hist, _ = np.histogram(ent_list, bins=6, range=(0, max_ent if max_ent>0 else 1))
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    dyn_threshold = float(np.clip(0.4 + 0.12 * (hist_entropy / np.log2(6)) + 0.05 * ((global_q_entropy - 2.0)/4.0), 0.2, 0.85))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    # prepare heatmaps
    heatmap_blocks = _compute_block_heatmap_from_values(block_grid)  # normalized block grid (entropies)
    # downsampled "lsb-like" map from original luminance (gives intuitive front-end overlay)
    down_map = _downsample_to_map(lum, block_size=heatmap_block_size, max_map_dim=64)

    details = {
        "patch_count": len(patch_stats),
        "ent_mean": float(ent_mean),
        "max_ent": float(max_ent),
        "global_q_entropy": float(global_q_entropy),
        "combined_raw": float(combined),
        "dynamic_threshold": float(dyn_threshold),
        "heatmap_blocks": heatmap_blocks.tolist(),
        "heatmap": down_map.tolist(),
        "params": {
            "q": int(q),
            "offsets": offsets,
            "patch_size": int(patch_size),
            "stride": int(stride),
            "max_dim": int(max_dim),
            "color_space": color_space,
            "prefilter": prefilter,
            "heatmap_block_size": int(heatmap_block_size)
        }
    }

    return {"method": "cooccurrence_analysis", "score": float(score), "detected": bool(detected), "details": details}
