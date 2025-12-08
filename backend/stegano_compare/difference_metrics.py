# stegano_compare/difference_metrics.py
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_differences(orig, susp):
    """
    orig, susp: grayscale images float [0,1]
    returns mse, ssim_val, diff_map (abs difference)
    """
    if orig.shape != susp.shape:
        import skimage.transform as tf
        susp = tf.resize(susp, orig.shape, anti_aliasing=True)

    mse = float(np.mean((orig - susp) ** 2))
    ssim_val = float(ssim(orig, susp, data_range=1.0))
    diff_map = np.abs(orig - susp)
    return mse, ssim_val, diff_map
