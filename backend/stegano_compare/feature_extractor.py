# stegano_compare/feature_extractor.py
import numpy as np

def extract_features(mse, ssim_val, residual_diff_mean, lsb_prop, dct_score):
    """
    Return 1D numpy feature vector ready for ML classifier.
    Order: [mse, 1-ssim, residual_diff_mean, lsb_prop, dct_score]
    We convert ssim to 1-ssim as 'difference' feature.
    """
    return np.array([mse, 1.0-ssim_val, residual_diff_mean, lsb_prop, dct_score]).reshape(1, -1)
