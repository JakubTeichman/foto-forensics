# stegano_compare/lsb_analysis.py
import numpy as np

def analyze_lsb(orig, susp):
    """
    Works on uint8 data. orig/susp expected float [0,1] -> convert to uint8.
    Returns:
      - lsb_diff_prop: proportion of differing LSB bits
      - lsb_diff_map: 2D map of differing bits (0/1)
    """
    import numpy as np
    o = (orig * 255).astype(np.uint8)
    s = (susp * 255).astype(np.uint8)

    # align shapes if needed (simple resize)
    if o.shape != s.shape:
        import skimage.transform as tf
        s = (tf.resize(s, o.shape, anti_aliasing=True) * 255).astype(np.uint8)

    o_lsb = o & 1
    s_lsb = s & 1
    diff_map = np.abs(o_lsb.astype(np.int8) - s_lsb.astype(np.int8))
    prop = float(diff_map.mean())
    return prop, diff_map
