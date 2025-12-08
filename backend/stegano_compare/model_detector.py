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
    # Features in order: [mse, ssim_val, residual_diff_mean, lsb_prop, dct_score]
    feats = extract_features(mse, ssim_val, residual_diff_mean, lsb_prop, dct_score)
    if _RF_MODEL is not None:
        try:
            proba = _RF_MODEL.predict_proba(feats)[:,1][0]
            return float(proba), "rf_model"
        except Exception:
            pass

    # ---------- Modyfikacja Heurystyki Zgodna z Wymaganiami Użytkownika ----------
    
    # Wagi dostosowane, aby wynik bazowy (base_score) zależał głównie od cech detekcyjnych
    # (Residual Diff i LSB Prop). Wpływ MSE będzie aplikowany jako silna korekta/mnożnik na końcu.
    w = np.array([
        0.00,  # mse (Wpływ usunięty, bo będzie silna korekta MSE)
        0.00,  # ssim (Wpływ usunięty, j.w.)
        0.40,  # residual diff (bardzo ważne)
        0.50,  # lsb diff (najważniejsze)
        0.10   # dct score (pomocnicze)
    ])

    norm_feats = feats.flatten()
    # Ograniczenie niektórych cech do rozsądnych zakresów przed ważeniem
    norm_feats[0] = min(norm_feats[0], 1.0) # mse
    norm_feats[2] = min(norm_feats[2], 1.0)  
    norm_feats[3] = min(norm_feats[3], 1.0) # lsb prop
    
    # 1. Obliczenie wyniku bazowego (zależnego głównie od wskaźników LSB i Residual)
    base_score = float(np.clip(np.dot(w, norm_feats), 0.0, 1.0))
    final_score = base_score
    
    # 2. Aplikacja korekty na podstawie MSE
    
    # Wartości progowe i korekcyjne
    MSE_THRESHOLD_BOOST = 0.00001            # Wartość bliska 0 dla "dosłownie równe 0"
    MSE_THRESHOLD_PENALTY_START = 0.0002    # Próg 2/10000, od którego zaczyna się kara
    MSE_MAX_PENALTY_RANGE = 0.0050          # Maksymalne MSE (np. 50/10000), gdzie kara jest maksymalna

    if mse <= MSE_THRESHOLD_BOOST:
        # Wymóg 1: MSE dosłownie równe 0. Wynik ma być "o wiele wyższy".
        # Stosujemy duży mnożnik (np. x1.75), by znacznie zwiększyć wynik bazowy.
        boost_multiplier = 1000
        final_score = base_score * boost_multiplier
    
    elif mse > MSE_THRESHOLD_PENALTY_START:
        # Wymóg 2: MSE niezerowe, powyżej progu 0.0002. Wynik proporcjonalnie maleje.
        
        mse_excess = mse - MSE_THRESHOLD_PENALTY_START
        
        # Obliczenie zakresu, w którym kara jest aktywna: 0.0050 - 0.0002 = 0.0048
        max_excess = MSE_MAX_PENALTY_RANGE - MSE_THRESHOLD_PENALTY_START
        
        if max_excess > 1e-10 and mse_excess > 0:
            
            # Wskaźnik siły kary (0.0 przy progu, 1.0 przy MSE_MAX_PENALTY_RANGE)
            penalty_strength = min(1.0, mse_excess / max_excess)
            
            # Mnożnik maleje od 1.0 (przy progu) do 0.1 (przy max range).
            # Oznacza to, że przy max range zostaje tylko 10% wyniku bazowego (0.1).
            min_multiplier = 0.0001 
            penalty_multiplier = max(min_multiplier, 1.0 - (1.0 - min_multiplier) * penalty_strength)
            
            final_score = base_score * penalty_multiplier
            
    # Ostateczne ograniczenie wyniku do zakresu 0.0 - 1.0
    score = float(np.clip(final_score, 0.0, 1.0))
    print(score)
    return score, "heuristic"