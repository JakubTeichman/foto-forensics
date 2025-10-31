"""
Chi-square (Pairs of Value - PoV) detector, patch-aware and multi-scale.
- Dzieli obraz na bloki (np. 64x64) i liczy statystykę chi^2 dla par (0+1,2+3,...)
- Łączy p-wartości blokowo (Fisher's method) i daje score 0..1 (1 -> silna odchyłka wskazująca stego)
- Zwraca szczegóły: per-block chi^2, p oraz global p
"""
import numpy as np
from scipy.stats import chi2
import io
from PIL import Image

def _chi2_for_array(arr):
    hist = np.bincount(arr.ravel().astype(np.uint8), minlength=256).astype(np.float64)
    pairs = hist.reshape(128,2)
    evens = pairs[:,0]; odds = pairs[:,1]; denom = (evens+odds)
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

def analyze(image_bytes=None, pil_image=None, block_size=64, stride=64, threshold=0.5):
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
    arr = np.array(pil_image, dtype=np.uint8)
    h,w = arr.shape
    block_pvalues = []
    block_stats = []
    for y0 in range(0, h - block_size + 1, stride):
        for x0 in range(0, w - block_size + 1, stride):
            blk = arr[y0:y0+block_size, x0:x0+block_size]
            chi_stat, p, df = _chi2_for_array(blk)
            block_pvalues.append(p)
            block_stats.append({"x":x0,"y":y0,"chi2":chi_stat,"p":p,"df":df})
    # combine p-values using Fisher's method
    if len(block_pvalues)==0:
        global_p = 1.0
    else:
        import math
        T = -2.0 * sum(np.log(np.clip(block_pvalues, 1e-12, 1.0)))
        df_tot = 2 * len(block_pvalues)
        global_p = float(1.0 - chi2.cdf(T, df_tot))
    # score = 1 - global_p (small p -> high score)
    score = float(max(0.0, min(1.0, 1.0 - global_p)))
    detected = score >= threshold
    details = {"blocks": block_stats, "global_p": global_p, "n_blocks": len(block_pvalues)}
    return {"method":"chi_square", "score": score, "detected": bool(detected), "details": details}
