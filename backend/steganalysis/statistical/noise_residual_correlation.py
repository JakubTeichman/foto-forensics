import cv2
import numpy as np
import io
import base64
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

def analyze_noise_residual(image_path):
    """
    Noise Residual Correlation Analysis
    -----------------------------------
    Detects potential steganographic alterations by analyzing
    correlations between residual noise maps from multiple filters.

    This method estimates local inconsistencies in noise structure,
    which are often introduced by embedding hidden data.
    """

    # --- 1️⃣ Load and preprocess image ---
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Failed to load image."}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray /= 255.0

    # --- 2️⃣ Compute residual maps from different filters ---
    blur3 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur5 = cv2.GaussianBlur(gray, (5, 5), 0)
    median = cv2.medianBlur((gray * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    bilateral = cv2.bilateralFilter(gray, 5, 75, 75)

    residual1 = gray - blur3
    residual2 = gray - blur5
    residual3 = gray - median
    residual4 = gray - bilateral

    # --- 3️⃣ Compute local correlations between residual maps ---
    corr_12 = np.corrcoef(residual1.ravel(), residual2.ravel())[0, 1]
    corr_13 = np.corrcoef(residual1.ravel(), residual3.ravel())[0, 1]
    corr_14 = np.corrcoef(residual1.ravel(), residual4.ravel())[0, 1]
    corr_23 = np.corrcoef(residual2.ravel(), residual3.ravel())[0, 1]
    corr_24 = np.corrcoef(residual2.ravel(), residual4.ravel())[0, 1]
    corr_34 = np.corrcoef(residual3.ravel(), residual4.ravel())[0, 1]

    correlations = np.array([corr_12, corr_13, corr_14, corr_23, corr_24, corr_34])
    mean_corr = float(np.nanmean(correlations))
    std_corr = float(np.nanstd(correlations))

    # --- 4️⃣ Compute entropy-like irregularity map ---
    residual_stack = np.stack([residual1, residual2, residual3, residual4], axis=0)
    local_var = np.var(residual_stack, axis=0)
    irregularity_map = cv2.normalize(local_var, None, 0, 1, cv2.NORM_MINMAX)

    plt.figure(figsize=(6, 6))
    plt.imshow(irregularity_map, cmap='inferno')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap_encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

    # --- 5️⃣ Compute final confidence score ---
    variance_mean = float(np.mean(local_var))
    irregularity_score = 1.0 - mean_corr  # lower correlation = more suspicious
    confidence = float(np.clip((irregularity_score * 0.7 + variance_mean * 1.2), 0.0, 1.0))
    suspicious = confidence > 0.55

    # --- 6️⃣ Metrics dictionary ---
    metrics = {
        "Mean correlation": round(mean_corr, 4),
        "Std correlation": round(std_corr, 4),
        "Mean residual variance": round(variance_mean, 4),
        "Irregularity score": round(irregularity_score, 4),
        "Confidence": round(confidence, 4)
    }

    # --- 7️⃣ Summary text ---
    summary = (
        f"Noise residual correlation shows an average correlation of {mean_corr:.3f}. "
        f"The irregularity score ({irregularity_score:.2f}) and residual variance ({variance_mean:.3f}) "
        f"indicate {'potential manipulation' if suspicious else 'normal noise structure'}. "
        f"Confidence score: {confidence:.2f}."
    )

    return {
        "heatmap": heatmap_encoded,
        "metrics": metrics,
        "summary": summary
    }
