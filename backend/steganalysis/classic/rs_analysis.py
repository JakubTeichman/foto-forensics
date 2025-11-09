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

def _remove_exif_temp(pil_image):
    """
    Return new PIL.Image with EXIF removed (pixel data preserved).
    Does NOT modify original bytes or input reference.
    """
    arr = np.array(pil_image)
    return Image.fromarray(arr)

def _resize_image(pil_image, max_dim=1024):
    w, h = pil_image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        return pil_image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return pil_image

def _convert_to_luminance(pil_image, space="YCbCr"):
    """
    Convert to luminance array (uint8) using Y channel (YCbCr) or L channel (Lab).
    """
    if space and space.lower() == "lab":
        # PIL doesn't provide direct 'LAB' convert reliably everywhere; approximate via cv2 if available
        rgb = pil_image.convert("RGB")
        arr = np.array(rgb)[:,:,::-1]  # PIL RGB -> OpenCV BGR by reversing
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        L = lab[:,:,0]
        return L.astype(np.uint8)
    else:
        ycbcr = pil_image.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        return np.array(y, dtype=np.uint8)

def _prefilter(arr_uint8, mode="highpass"):
    """
    Prefilter options:
      - 'none' : return original
      - 'highpass' : subtract gaussian blurred image (HyPAS-like)
      - 'laplacian' : laplacian filter
      - 'log' : gaussian blur then laplacian (LoG)
    Returns uint8 array scaled to 0..255.
    """
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

def _group_function(block):
    p = block.astype(np.int32)
    s = 0
    if p.shape[1] > 1:
        s += np.abs(p[:,1:] - p[:,:-1]).sum()
    if p.shape[0] > 1:
        s += np.abs(p[1:,:] - p[:-1,:]).sum()
    return float(s)

def _compute_lsb_heatmap(arr_uint8, block_size=8, max_map_dim=64):
    """
    Compute downsampled LSB heatmap for frontend:
    - take LSB plane, compute averages in blocks of block_size,
      then downsample if any dimension > max_map_dim.
    """
    bits = (arr_uint8 & 1).astype(np.float32)
    h, w = bits.shape
    ph = ((h + block_size - 1) // block_size) * block_size
    pw = ((w + block_size - 1) // block_size) * block_size
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h, :w] = bits
    bh = ph // block_size
    bw = pw // block_size
    block_avg = padded.reshape(bh, block_size, bw, block_size).mean(axis=(1,3))
    # downsample if large
    if max(block_avg.shape) > max_map_dim:
        scale = max_map_dim / float(max(block_avg.shape))
        new_h = max(1, int(block_avg.shape[0]*scale))
        new_w = max(1, int(block_avg.shape[1]*scale))
        block_avg = cv2.resize(block_avg, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return block_avg.astype(np.float32)

# ----------------------
# Main analyze (enhanced RS)
# ----------------------
def analyze(image_bytes=None,
            pil_image=None,
            group_size=2,
            threshold=0.4,
            max_dim=1024,
            color_space="YCbCr",
            prefilter="highpass",
            compute_heatmap_block_size=8):
    """
    Enhanced RS analysis (Fridrich/Goljan style) with:
      - temp EXIF strip, resize, luminance conversion (YCbCr or Lab)
      - optional prefilter (highpass/laplacian/log/none)
      - normalization of group distributions
      - dynamic thresholding based on histogram + entropy
      - returns block-level heatmap and LSB heatmap (downsampled)
    Output: {"method","score","detected","details"}
    """

    # --- load and preprocess image (temporary EXIF strip only for processing)
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil = Image.open(io.BytesIO(image_bytes))
    else:
        pil = pil_image.copy()

    # make a pixel-only copy to drop EXIF (doesn't touch original source outside)
    pil_proc = _remove_exif_temp(pil)
    pil_proc = ImageOps.exif_transpose(pil_proc)
    pil_proc = _resize_image(pil_proc, max_dim=max_dim)

    # use luminance channel for analysis
    lum = _convert_to_luminance(pil_proc, space=color_space)
    # prefilter to emphasize residuals
    work = _prefilter(lum, mode=prefilter)

    arr = np.array(work, dtype=np.uint8)
    h, w = arr.shape

    # iterate groups
    R0 = S0 = Rf = Sf = 0.0
    total_groups = 0
    group_changes = []
    group_scores = []  # per-group coarse scores (0/1) used for heatmap/normalization

    step = group_size
    # create block grid for heatmap_blocks (rows x cols)
    ys = list(range(0, max(1, h - group_size + 1), step))
    xs = list(range(0, max(1, w - group_size + 1), step))
    heatmap_blocks = []

    for y0 in ys:
        row_vals = []
        for x0 in xs:
            blk = arr[y0:y0+group_size, x0:x0+group_size]
            if blk.size == 0:
                row_vals.append(0.5)
                continue
            f_orig = _group_function(blk)
            flipped_blk = blk ^ 1
            f_flipped = _group_function(flipped_blk)
            # coarse decision per group
            if f_orig > f_flipped:
                R0 += 1
                group_scores.append(1.0)
                row_vals.append(1.0)
            else:
                S0 += 1
                group_scores.append(0.0)
                row_vals.append(0.0)

            mask = np.zeros_like(blk, dtype=np.uint8)
            mask[::2, ::2] = 1
            blk_masked = blk ^ mask
            f_masked = _group_function(blk_masked)
            if f_masked > f_flipped:
                Rf += 1
            else:
                Sf += 1

            total_groups += 1
            if len(group_changes) < 200:
                group_changes.append({"x": x0, "y": y0, "f_orig": float(f_orig),
                                      "f_flipped": float(f_flipped), "f_masked": float(f_masked)})
        heatmap_blocks.append(row_vals)

    heatmap_blocks = np.array(heatmap_blocks, dtype=np.float32) if len(heatmap_blocks) > 0 else np.zeros((1,1), dtype=np.float32)
    # normalize block heatmap to 0..1
    if heatmap_blocks.size > 0:
        bh_min = heatmap_blocks.min(); bh_ptp = heatmap_blocks.ptp()
        if bh_ptp <= 1e-12:
            heatmap_blocks_norm = np.clip(heatmap_blocks, 0.0, 1.0)
        else:
            heatmap_blocks_norm = (heatmap_blocks - bh_min) / (bh_ptp + 1e-12)
    else:
        heatmap_blocks_norm = np.zeros((1,1), dtype=np.float32)

    # raw estimate
    if total_groups == 0:
        raw_score = 0.0
    else:
        est = ((R0 - Rf) + (Sf - S0)) / (2.0 * total_groups + 1e-12)
        raw_score = float(max(0.0, min(1.0, abs(est))))

    # normalized distribution
    norm_group = _normalize_scores(group_scores) if group_scores else raw_score

    # entropy weight (use original luminance, not prefiltered)
    orig_lum = np.array(_convert_to_luminance(pil_proc, space=color_space), dtype=np.uint8)
    entropy = _compute_shannon_entropy(orig_lum)
    entropy_weight = float(np.clip(entropy / 7.5, 0.7, 1.3))

    combined = float(np.clip(0.7 * raw_score + 0.3 * norm_group, 0.0, 1.0))
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold: use histogram entropy of block-level values and image entropy
    hist, _ = np.histogram(heatmap_blocks_norm.ravel(), bins=6, range=(0,1))
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_ent = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    dyn_threshold = float(np.clip(0.35 + 0.18 * (hist_ent / np.log2(6)) + 0.06 * ((entropy - 4.0) / 4.0), 0.2, 0.8))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    # LSB heatmap (downsampled) from original luminance for frontend visualization
    lsb_heatmap = _compute_lsb_heatmap(orig_lum, block_size=compute_heatmap_block_size, max_map_dim=64)

    details = {
        "R0": int(R0), "S0": int(S0), "Rf": int(Rf), "Sf": int(Sf),
        "groups": int(total_groups),
        "group_examples": group_changes[:200],
        "entropy": float(entropy),
        "combined_raw": float(combined),
        "dynamic_threshold": float(dyn_threshold),
        "heatmap_blocks": heatmap_blocks_norm.tolist(),
        "lsb_heatmap": lsb_heatmap.tolist(),
        "params": {
            "group_size": int(group_size),
            "max_dim": int(max_dim),
            "color_space": color_space,
            "prefilter": prefilter,
            "compute_heatmap_block_size": int(compute_heatmap_block_size)
        }
    }

    return {"method": "rs_analysis", "score": float(score), "detected": bool(detected), "details": details}
