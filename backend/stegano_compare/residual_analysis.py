import numpy as np
from scipy import ndimage

# Wysokoprzepustowy filtr (High-Pass Filter)
HPF = np.array([
    [-1,  2, -1],
    [ 2, -4,  2],
    [-1,  2, -1]
])

def highpass(img: np.ndarray) -> np.ndarray:
    """
    Apply a simple high-pass filter to emphasize fine details or noise.
    Uses scipy.ndimage.correlate for modern compatibility.
    """
    return ndimage.correlate(img, HPF, mode='reflect')

def analyze_residuals(orig: np.ndarray, susp: np.ndarray):
    """
    Compute residual maps and return:
      - residual_diff_mean: mean absolute difference between residual maps
      - residual_map_diff: absolute difference map (for visualization)
    """
    orig_res = highpass(orig)
    susp_res = highpass(susp)
    diff_map = np.abs(orig_res - susp_res)
    diff_mean = float(np.mean(diff_map))
    return diff_mean, diff_map
