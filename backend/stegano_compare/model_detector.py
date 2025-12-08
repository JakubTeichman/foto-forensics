import os
import numpy as np
from .feature_extractor import extract_features

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

    w = np.array([
        0.00,  
        0.00,  
        0.40,  
        0.50,  
        0.10   
    ])

    norm_feats = feats.flatten()
    
    norm_feats[0] = min(norm_feats[0], 1.0)
    norm_feats[2] = min(norm_feats[2], 1.0)  
    norm_feats[3] = min(norm_feats[3], 1.0) 
    
    
    base_score = float(np.clip(np.dot(w, norm_feats), 0.0, 1.0))
    final_score = base_score
    
    
    MSE_THRESHOLD_BOOST = 0.00001           
    MSE_THRESHOLD_PENALTY_START = 0.0002   
    MSE_MAX_PENALTY_RANGE = 0.0050          

    if mse <= MSE_THRESHOLD_BOOST:
        boost_multiplier = 1000
        final_score = base_score * boost_multiplier
    
    elif mse > MSE_THRESHOLD_PENALTY_START:
        
        mse_excess = mse - MSE_THRESHOLD_PENALTY_START
        
        max_excess = MSE_MAX_PENALTY_RANGE - MSE_THRESHOLD_PENALTY_START
        
        if max_excess > 1e-10 and mse_excess > 0:
            
            penalty_strength = min(1.0, mse_excess / max_excess)
            
            min_multiplier = 0.0001 
            penalty_multiplier = max(min_multiplier, 1.0 - (1.0 - min_multiplier) * penalty_strength)
            
            final_score = base_score * penalty_multiplier
            
    score = float(np.clip(final_score, 0.0, 1.0))
    print(score)
    return score, "heuristic"