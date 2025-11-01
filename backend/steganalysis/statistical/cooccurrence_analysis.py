import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

def analyze(image_bytes=None, pil_image=None, threshold=0.7):
    """
    Analiza współwystępowania (co-occurrence matrix) — ocenia korelacje sąsiednich pikseli.
    Steganografia zwiększa entropię i zmniejsza kontrast / homogeniczność.
    """
    if pil_image is None:
        raise ValueError("Brak obrazu PIL")

    img = np.array(pil_image.convert("L"))
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    # Normalizujemy entropię (0–1)
    score = np.clip(entropy / 15.0, 0, 1)
    detected = score >= threshold

    return {
        "method": "cooccurrence",
        "score": float(score),
        "detected": bool(detected),
        "details": {
            "contrast": float(contrast),
            "homogeneity": float(homogeneity),
            "entropy": float(entropy)
        }
    }
