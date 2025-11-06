"""
Noiseprint model implementation based on:

D. Cozzolino and L. Verdoliva,
"Noiseprint: A CNN-Based Camera Model Fingerprint",
IEEE Transactions on Information Forensics and Security, vol. 15, pp. 144–159, 2020.
DOI: 10.1109/TIFS.2019.2916364

Original implementation (Matlab/TensorFlow) © 2019 GRIP-UNINA.
Adapted for research and educational use in Python/PyTorch by Jakub Teichman, 2025.
"""

from PIL import Image
from PIL.JpegImagePlugin import convert_dict_qtables
import numpy as np
from scipy.interpolate import interp1d
import io


def im2f(image_input, channel=1, dtype=np.float32):
    """
    Wczytuje obraz jako float32 (0-1), RGB lub grayscale.
    Obsługuje: PIL.Image, np.ndarray, bytes, file (Flask upload)
    """
    # Zamiana na obiekt PIL.Image
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8))
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):  # np. request.files['file']
        img = Image.open(image_input)
    else:
        raise ValueError("Nieobsługiwany format wejścia obrazu.")

    # Konwersja kanałów
    if channel == 1:
        img = img.convert('L')
    elif channel == 3:
        img = img.convert('RGB')

    arr = np.asarray(img).astype(dtype) / 256.0
    return arr, img.mode


def jpeg_qtableinv(image_input, tnum=0):
    """
    Odczytuje jakość JPEG (QF) na podstawie tablic kwantyzacji.
    Obsługuje: PIL.Image, bytes, file
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):
        img = Image.open(image_input)
    else:
        raise ValueError("Nieobsługiwany format wejścia obrazu.")

    try:
        h = np.asarray(convert_dict_qtables(img.quantization)[tnum]).reshape((8, 8))
    except Exception:
        return 101  # brak informacji o tablicy – traktuj jako surowy

    t = np.matrix([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])
    h_down = (2*h - 1) / (2*t)
    h_up = (2*h + 1) / (2*t)
    if np.all(h == 1):
        return 100
    x_down = (h_down[h > 1]).max()
    x_up = (h_up[h < 255]).min()
    s = 50 if x_up is None else (50 / x_up if x_up > 1 else 50*(2 - x_up))
    return int(round(s))


def resizeMapWithPadding(x, range0, range1, shapeOut):
    range0 = range0.flatten()
    range1 = range1.flatten()
    xv = np.arange(shapeOut[1])
    yv = np.arange(shapeOut[0])
    y = interp1d(range1, x, axis=1, kind='nearest', fill_value='extrapolate')
    y = interp1d(range0, y(xv), axis=0, kind='nearest', fill_value='extrapolate')
    return y(yv).astype(x.dtype)
