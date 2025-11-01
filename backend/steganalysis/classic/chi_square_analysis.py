import numpy as np
from scipy.stats import chisquare
from PIL import Image

def analyze(image_bytes=None, pil_image=None, threshold=0.75):
    """
    Analiza chi-kwadrat — sprawdza, czy rozkład wartości pikseli jest losowy.
    Odchylenia od losowości mogą wskazywać na obecność steganografii.
    """
    if pil_image is None:
        raise ValueError("Brak obrazu PIL")

    gray = np.array(pil_image.convert("L")).flatten()
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    expected = np.ones_like(hist) * np.mean(hist)

    chi_stat, p_value = chisquare(hist, expected)
    score = 1 - p_value  # duży = odchylenie = możliwe stego
    detected = score >= threshold

    return {
        "method": "chi_square",
        "score": float(score),
        "detected": bool(detected),
        "details": {"p_value": float(p_value), "chi_stat": float(chi_stat)}
    }
