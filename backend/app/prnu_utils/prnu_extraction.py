# Prototyp ekstrakcji PRNU: modularne punkty wejścia
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from typing import Union
import io

def _load_image(path_or_bytes: Union[str, bytes]):
    if isinstance(path_or_bytes, bytes):
        from PIL import Image
        return np.array(Image.open(io.BytesIO(path_or_bytes)).convert("L"), dtype=np.float32)
    else:
        img = imread(path_or_bytes)
        if img is None:
            raise FileNotFoundError(path_or_bytes)
        if img.ndim == 3:
            img = rgb2gray(img)  # zwraca float w [0,1]
        return img.astype(np.float32)

def extract_prnu_from_path(path: str) -> np.ndarray:
    img = _load_image(path)
    return _extract_residual(img)

def extract_prnu_from_bytes(b: bytes) -> np.ndarray:
    img = _load_image(b)
    return _extract_residual(img)

def _extract_residual(img: np.ndarray) -> np.ndarray:
    # Prosty pipeline: gaussian smooth -> residual -> zero-mean -> norm
    smooth = gaussian_filter(img, sigma=1.0)
    residual = img - smooth
    residual -= residual.mean()
    std = residual.std()
    if std < 1e-8:
        std = 1.0
    residual = residual / std
    return residual.astype(np.float32)
