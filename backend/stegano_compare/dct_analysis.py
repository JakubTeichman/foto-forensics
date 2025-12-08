# stegano_compare/dct_analysis.py
import numpy as np
from scipy.fftpack import dct

def block_dct_2d(block):
    """2D DCT type II on an 8x8 block (normalized)."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def image_dct_stats(img, block_size=8):
    """
    Compute DCT over non-overlapping blocks.
    Returns DC and AC coefficient stats for grayscale float [0,1].
    """
    if img is None or img.size == 0:
        return {"mean_abs_dc": 0.0, "mean_abs_ac": 0.0, "std_ac": 0.0, "median_ac": 0.0}

    H, W = img.shape
    pad_h = (block_size - (H % block_size)) % block_size
    pad_w = (block_size - (W % block_size)) % block_size
    img_p = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
    H2, W2 = img_p.shape

    ac_vals = []
    dc_vals = []

    for i in range(0, H2, block_size):
        for j in range(0, W2, block_size):
            blk = img_p[i:i+block_size, j:j+block_size]
            blk = (blk * 255.0).astype(np.float32)
            d = block_dct_2d(blk)
            dc_vals.append(np.abs(d[0, 0]))
            ac = np.abs(d).flatten()
            ac = np.delete(ac, 0)
            ac_vals.extend(ac.tolist())

    # fallback in case of empty lists
    if len(ac_vals) == 0:
        ac_vals = [0.0]
    if len(dc_vals) == 0:
        dc_vals = [0.0]

    ac = np.array(ac_vals)
    dc = np.array(dc_vals)

    stats = {
        "mean_abs_dc": float(np.nan_to_num(np.mean(dc))),
        "mean_abs_ac": float(np.nan_to_num(np.mean(ac))),
        "std_ac": float(np.nan_to_num(np.std(ac))),
        "median_ac": float(np.nan_to_num(np.median(ac))),
    }
    return stats

def compare_dct(orig, susp):
    """Compare DCT statistics; return normalized L1 distance and both stats."""
    s_orig = image_dct_stats(orig)
    s_susp = image_dct_stats(susp)

    keys = ["mean_abs_dc", "mean_abs_ac", "std_ac", "median_ac"]
    v1 = np.array([s_orig[k] for k in keys])
    v2 = np.array([s_susp[k] for k in keys])

    denom = np.maximum(np.abs(v1), 1e-6)
    rel_diff = np.abs(v1 - v2) / denom
    score = float(np.nan_to_num(np.mean(rel_diff)))

    return score, s_orig, s_susp
