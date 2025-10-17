import numpy as np
from skimage import img_as_float
from skimage.restoration import denoise_wavelet
from bm3d import bm3d
import pywt
from scipy.signal import wiener
from scipy import fftpack

EPS = 1e-8

def load_image_as_gray(img):
    img = img_as_float(img)
    if img.ndim == 3:
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return img


def extract_noise_residual(img, method='bm3d'):
    """
    Return residual = image - denoised(image)
    method: 'bm3d', 'wavelet', 'wiener', or 'none'
    """
    gray = load_image_as_gray(img)

    if method == 'bm3d':
        # BM3D with estimated noise standard deviation
        sigma_est = np.mean(np.std(gray, axis=0))
        den = bm3d(gray, sigma_psd=sigma_est)
    elif method == 'wavelet':
        den = denoise_wavelet(gray, channel_axis=None, rescale_sigma=True)
    elif method == 'wiener':
        den = wiener(gray, mysize=(3,3))
    else:
        den = gray

    residual = gray - den
    return residual


def build_reference_pattern(images, denoise_method='bm3d'):
    imgs = [load_image_as_gray(im) for im in images]
    min_h = min(im.shape[0] for im in imgs)
    min_w = min(im.shape[1] for im in imgs)
    imgs = [im[:min_h, :min_w] for im in imgs]

    K_num = np.zeros_like(imgs[0], dtype=np.float64)
    K_den = np.zeros_like(imgs[0], dtype=np.float64)

    for im in imgs:
        W = extract_noise_residual(im, method=denoise_method)
        K_num += W * im
        K_den += im * im

    K_ref = K_num / (K_den + EPS)
    K_ref = K_ref - np.mean(K_ref)
    try:
        K_ref = wiener(K_ref, mysize=(3,3))
    except Exception:
        pass
    return K_ref


def cross_correlation_map(signal, template):
    s = signal - np.mean(signal)
    t = template - np.mean(template)
    T = np.flipud(np.fliplr(t))
    S = fftpack.fft2(s)
    Tf = fftpack.fft2(T)
    R = np.real(fftpack.ifft2(S * Tf))
    denom = (np.std(s) * np.std(t) * s.size)
    if denom < EPS:
        return np.zeros_like(R)
    return R / denom


def compute_pce(cc_map, peak_window=11):
    m, n = cc_map.shape
    idx = np.unravel_index(np.argmax(np.abs(cc_map)), cc_map.shape)
    peak_val = cc_map[idx]
    half = peak_window // 2
    mask = np.ones_like(cc_map, dtype=bool)
    y0, y1 = max(0, idx[0] - half), min(m, idx[0] + half + 1)
    x0, x1 = max(0, idx[1] - half), min(n, idx[1] + half + 1)
    mask[y0:y1, x0:x1] = False
    bg_energy = np.mean(cc_map[mask]**2)
    if bg_energy < EPS:
        return float('inf'), float(peak_val), (int(idx[0]), int(idx[1]))
    pce = (float(peak_val)**2) / float(bg_energy)
    return float(pce), float(peak_val), (int(idx[0]), int(idx[1]))


def detect_nua(img):
    gray = load_image_as_gray(img)
    try:
        coeffs = pywt.wavedec2(gray, 'db2', level=2)
        approx = coeffs[0]
        detail_vals = [np.mean(np.abs(a)) for level in coeffs[1:] for a in level]
        approx_energy = np.mean(np.abs(approx)) + EPS
        detail_energy = (np.mean(detail_vals) + EPS)
        ratio = approx_energy / detail_energy
    except Exception:
        ratio = 0.0

    try:
        den = denoise_wavelet(gray, multichannel=False, rescale_sigma=True)
        smoothness = np.var(gray) / (np.var(den) + EPS)
    except Exception:
        smoothness = 1.0

    nua_flag = (ratio > 2.5) or (smoothness < 1.2)
    diag = {'wavelet_ratio': float(ratio), 'smoothness': float(smoothness)}
    return bool(nua_flag), diag


def compare_with_reference(test_in, ref_list, denoise_method='bm3d'):
    from skimage import io

    def ensure_array(x):
        return io.imread(x) if isinstance(x, str) else x

    ref_arrays = [ensure_array(r) for r in ref_list]
    test_arr = ensure_array(test_in)

    K_ref = build_reference_pattern(ref_arrays, denoise_method=denoise_method)
    W_test = extract_noise_residual(test_arr, method=denoise_method)

    h, w = K_ref.shape
    W_test = W_test[:h, :w]
    test_lum = load_image_as_gray(test_arr)[:h, :w]

    template = K_ref * test_lum
    cc_map = cross_correlation_map(W_test, template)
    pce, peak, peak_coords = compute_pce(cc_map)

    try:
        ref_mean = np.mean(np.stack([load_image_as_gray(r)[:h, :w] for r in ref_arrays]), axis=0)
    except Exception:
        ref_mean = load_image_as_gray(ref_arrays[0])[:h, :w]
    nua_ref, diag_ref = detect_nua(ref_mean)
    nua_test, diag_test = detect_nua(test_arr)

    result = {
        'PCE': float(pce),
        'peak': float(peak),
        'peak_coords': peak_coords,
        'NUA_ref': bool(nua_ref),
        'NUA_test': bool(nua_test),
        'diag_ref': diag_ref,
        'diag_test': diag_test
    }
    return result
