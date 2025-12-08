# stegano_compare/lsb_analysis.py
import numpy as np
from skimage.transform import resize

def analyze_lsb(orig, susp):
    """
    Analyze Least Significant Bits differences between two grayscale images.
    Works reliably on float [0,1] input, internally converted to uint8.
    Returns:
      - lsb_diff_prop: proportion of differing LSB bits
      - lsb_diff_map: 2D binary map of differing bits (0/1)
    """
    if orig is None or susp is None:
        return 0.0, np.zeros((1, 1))
    orig = np.nan_to_num(orig, nan=0.0, posinf=1.0, neginf=0.0)
    susp = np.nan_to_num(susp, nan=0.0, posinf=1.0, neginf=0.0)

    o = np.clip(orig * 255, 0, 255).astype(np.uint8)
    s = np.clip(susp * 255, 0, 255).astype(np.uint8)

    if o.shape != s.shape:
        s = resize(s, o.shape, anti_aliasing=False, preserve_range=True).astype(np.uint8)

    o_lsb = o & 1
    s_lsb = s & 1

    diff_map = np.abs(o_lsb.astype(np.int8) - s_lsb.astype(np.int8))
    prop = float(np.mean(diff_map)) if diff_map.size > 0 else 0.0

    if np.isnan(prop):
        prop = 0.0

    return prop, diff_map
