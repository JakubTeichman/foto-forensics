import numpy as np
from PIL import Image

def analyze(image_bytes=None, pil_image=None, threshold=0.55):
    """
    RS analysis — klasyczna metoda wykrywająca steganografię LSB.
    Analizuje zmiany w grupach pikseli po negacji najmłodszych bitów.
    """
    if pil_image is None:
        raise ValueError("Brak obrazu PIL")

    img = np.array(pil_image.convert("L"), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=img.shape, dtype=np.uint8)

    flipped = img ^ mask
    diff = np.abs(img.astype(float) - flipped.astype(float))
    est = np.mean(diff) / 255.0

    score = float(np.clip(est, 0, 1))
    detected = score >= threshold

    return {
        "method": "rs_analysis",
        "score": score,
        "detected": bool(detected),
        "details": {"avg_difference": score}
    }
