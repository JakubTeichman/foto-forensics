# stegano_compare/model_detector.py
import os
import numpy as np
from .feature_extractor import extract_features

# optional: load sklearn model if provided
try:
    from joblib import load as joblib_load
    model_available = True
except Exception:
    joblib_load = None
    model_available = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_model.pkl")

def load_rf_model():
    if joblib_load is None:
        return None
    try:
        if os.path.exists(MODEL_PATH):
            return joblib_load(MODEL_PATH)
    except Exception:
        return None
    return None

_RF_MODEL = load_rf_model()

def detect_anomaly(mse, ssim_val, residual_diff_mean, lsb_prop, dct_score):
    """
    Return tuple (score, method), where score is 0..1 probability:
     - if RF model exists -> use model.predict_proba
     - else fallback to heuristic weighted scoring
    """
    feats = extract_features(mse, ssim_val, residual_diff_mean, lsb_prop, dct_score)
    if _RF_MODEL is not None:
        try:
            proba = _RF_MODEL.predict_proba(feats)[:,1][0]
            return float(proba), "rf_model"
        except Exception:
            pass

    # fallback heuristic (normalized)
    # weights tuned roughly; you can re-train model later
    w = np.array([0.25, 0.2, 0.25, 0.2, 0.1])
    norm_feats = feats.flatten()
    # clamp some features to reasonable ranges
    norm_feats[0] = min(norm_feats[0], 1.0)   # mse (if images normalized)
    norm_feats[2] = min(norm_feats[2], 1.0)   # residual diff
    norm_feats[3] = min(norm_feats[3], 1.0)   # lsb prop
    score = float(np.clip(np.dot(w, norm_feats), 0.0, 1.0))
    return score, "heuristic"
