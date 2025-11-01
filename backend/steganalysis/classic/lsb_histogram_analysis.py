import numpy as np
from PIL import Image

def analyze(image_bytes=None, pil_image=None, threshold=0.45):
    """
    Analiza histogramu bitów LSB — sprawdza równomierność rozkładu zer i jedynek.
    Obraz z ukrytą wiadomością ma zbyt "idealny" rozkład.
    """
    if pil_image is None:
        raise ValueError("Brak obrazu PIL")

    img = np.array(pil_image.convert("L"))
    lsb_plane = img & 1
    ones = np.sum(lsb_plane)
    zeros = lsb_plane.size - ones
    prop = ones / (ones + zeros)

    # Im bliżej 0.5, tym bardziej "podejrzany" obraz (bo zbyt idealny)
    score = 1 - 2 * abs(prop - 0.5)
    detected = score <= threshold  # UWAGA: odwrotna logika

    return {
        "method": "lsb_histogram",
        "score": float(score),
        "detected": bool(detected),
        "details": {"ones_ratio": float(prop)}
    }
