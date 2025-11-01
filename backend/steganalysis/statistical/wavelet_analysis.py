import numpy as np
import pywt
from PIL import Image

def analyze(image_bytes=None, pil_image=None, threshold=0.7):
    """
    Analiza falkowa (wavelet) — mierzy energię wysokoczęstotliwościowych współczynników.
    Większa energia = potencjalna manipulacja lub steganografia.
    """
    if pil_image is None:
        raise ValueError("Brak obrazu PIL")

    img = np.array(pil_image.convert("L"))
    coeffs2 = pywt.dwt2(img, 'haar')
    (LL, (LH, HL, HH)) = coeffs2
    energy = np.mean(np.abs(LH)) + np.mean(np.abs(HL)) + np.mean(np.abs(HH))

    score = float(np.clip(energy / 100.0, 0, 1))
    detected = score >= threshold

    return {
        "method": "wavelet",
        "score": score,
        "detected": bool(detected),
        "details": {"energy": energy}
    }
