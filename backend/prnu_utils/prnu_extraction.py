# prnu_extraction.py
"""
Zaawansowany ekstraktor PRNU (zamiennik dla Twojego prnu_extractor).
Zawiera:
 - ładowanie obrazu z pliku / bytes (konwersja do grayscale float [0,1])
 - denoising wavelet (z fallbackami)
 - maskowanie obszarów o niskiej teksturze (na podstawie lokalnej wariancji)
 - high-pass FFT (opcjonalnie)
 - normalizacja residuala
Eksportuje: extract_prnu_from_path, extract_prnu_from_bytes
"""

import io
from typing import Union, Optional
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.restoration import denoise_wavelet, estimate_sigma
except Exception:
    imread = None
    rgb2gray = None
    denoise_wavelet = None
    estimate_sigma = None

from scipy.ndimage import gaussian_filter
from scipy.signal import wiener as scipy_wiener


def _load_image(path_or_bytes: Union[str, bytes]) -> np.ndarray:
    """
    Zwraca obraz grayscale float32 w zakresie [0,1].
    Akceptuje: path (str) lub bytes obrazu.
    """
    if isinstance(path_or_bytes, bytes):
        img = Image.open(io.BytesIO(path_or_bytes)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    else:
        if imread is None:
            img = Image.open(path_or_bytes)
            if img.mode == "RGB" or img.mode == "RGBA":
                img = img.convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr
        else:
            img = imread(path_or_bytes)
            if img is None:
                raise FileNotFoundError(path_or_bytes)
            if img.ndim == 3:
                if rgb2gray is not None:
                    img = rgb2gray(img)
                else:
                    img = np.asarray(Image.fromarray(img).convert("L"), dtype=np.float32) / 255.0
            else:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0
            return img


def _highpass_fft(img: np.ndarray, radius: int = 30) -> np.ndarray:
    """
    High-pass przez maskowanie niskich częstotliwości w dziedzinie FFT.
    radius: połowa szerokości kwadratu wyciętego w centrum (niskie częstotliwości).
    """
    if img.ndim != 2:
        raise ValueError("Wyjście _highpass_fft oczekuje obrazu 2D")
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones_like(img, dtype=np.float32)
    r = int(max(1, radius))
    r1 = max(0, crow - r)
    r2 = min(rows, crow + r)
    c1 = max(0, ccol - r)
    c2 = min(cols, ccol + r)
    mask[r1:r2, c1:c2] = 0.0
    fshift = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    if img_back.max() > 0:
        img_back = img_back / (img_back.max() + 1e-12)
    return img_back


def _local_texture_mask(img: np.ndarray, window: int = 7, var_thresh: float = 1e-4) -> np.ndarray:
    """
    Tworzy binarną maskę: 1 tam gdzie jest wystarczająco tekstury (wariancja lokalna > thresh),
    0 tam gdzie obszar jest jednolity — pomaga wyciszyć obszary zawierające mało informacji PRNU.
    """
    img_sq = img * img
    mean = gaussian_filter(img, sigma=window / 6.0)
    mean_sq = gaussian_filter(img_sq, sigma=window / 6.0)
    local_var = np.maximum(0.0, mean_sq - mean * mean)
    mask = (local_var > var_thresh).astype(np.float32)
    return mask


def _denoise(img: np.ndarray, method: str = "wavelet") -> np.ndarray:
    """
    Zwraca 'denoised' - estymację tła (bez PRNU).
    Metody: 'wavelet' (preferowane), 'wiener' (scipy), 'gauss' (fallback).
    """
    if method == "wavelet" and denoise_wavelet is not None:
        try:
            den = denoise_wavelet(img, method='BayesShrink', mode='soft', rescale_sigma=True)
            return den
        except Exception:
            method = "wiener"

    if method == "wiener":
        try:
            den = scipy_wiener(img, mysize=(5, 5))
            return den
        except Exception:
            method = "gauss"

    den = gaussian_filter(img, sigma=1.0)
    return den


def _normalize_residual(residual: np.ndarray) -> np.ndarray:
    """
    Zero-mean i unit-std normalization z zabezpieczeniem.
    """
    residual = residual - residual.mean()
    std = residual.std()
    if std < 1e-8:
        std = 1.0
    return (residual / std).astype(np.float32)


def _extract_residual(img: np.ndarray,
                      denoise_method: str = "wavelet",
                      apply_highpass: bool = True,
                      hp_radius: int = 30,
                      mask_window: int = 7,
                      mask_var_thresh: float = 1e-4) -> np.ndarray:
    """
    Główny pipeline ekstrakcji PRNU:
      1) normalizacja do [0,1]
      2) odszumianie (wavelet/wiener/gauss) -> denoised (estymacja obrazu bez PRNU)
      3) residual = img - denoised
      4) opcjonalny highpass FFT na residual (wzmocnienie wysokich częstotliwości)
      5) maskowanie obszarów o niskiej teksturze
      6) zero-mean i normalizacja
    """
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    den = _denoise(img, method=denoise_method)
    residual = img - den

    if apply_highpass:
        hp = _highpass_fft(residual, radius=hp_radius)
        if hp.std() > 1e-12:
            hp_scaled = (hp - hp.mean()) / (hp.std() + 1e-12)
            residual = (residual + hp_scaled) / 2.0

    mask = _local_texture_mask(img, window=mask_window, var_thresh=mask_var_thresh)
    residual = residual * mask

    residual = _normalize_residual(residual)
    return residual

def extract_prnu_from_path(path: str,
                           denoise_method: str = "wavelet",
                           apply_highpass: bool = True) -> np.ndarray:
    img = _load_image(path)
    return _extract_residual(img, denoise_method=denoise_method, apply_highpass=apply_highpass)


def extract_prnu_from_bytes(b: bytes,
                            denoise_method: str = "wavelet",
                            apply_highpass: bool = True) -> np.ndarray:
    img = _load_image(b)
    return _extract_residual(img, denoise_method=denoise_method, apply_highpass=apply_highpass)
