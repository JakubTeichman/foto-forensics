import numpy as np
from skimage import img_as_float
from skimage.restoration import denoise_wavelet
from bm3d import bm3d
import pywt
from scipy.signal import wiener
from scipy import fftpack

EPS = 1e-8

BIG_DIM_THRESHOLD = 3000   
PATCH_SIZE = 512         
OVERLAP = 0.5              
MIN_SIGMA = 1e-4
MAX_SIGMA = 0.5


def load_image_as_gray(img):
    img = img_as_float(img)
    if img.ndim == 3:
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return img


def _hann2d(sz):
    """2D Hann window rozmiaru sz (h, w)"""
    h, w = sz
    wy = np.hanning(h) if h > 1 else np.array([1.0])
    wx = np.hanning(w) if w > 1 else np.array([1.0])
    return np.outer(wy, wx)


def _patch_indices(length, patch_size, step):
    """Generator par (start, end) dla osi, zapewniający pokrycie końca obrazu."""
    i = 0
    while True:
        end = i + patch_size
        if end >= length:
            start = max(0, length - patch_size)
            yield start, length
            break
        else:
            yield i, end
            i += step


def _denoise_patch_bm3d(patch, sigma_est):
    """
    Denoise pojedynczy patch przy użyciu BM3D.
    Przyjmuje patch w skali 0..1, zwraca denoised patch w tym samym zakresie/dtype.
    """
    try:
        p_in = np.ascontiguousarray(patch.astype(np.float64))
        den = bm3d(p_in, sigma_psd=float(sigma_est))
        return np.asarray(den, dtype=patch.dtype)
    except Exception as e:
        print("BM3D patch failed, fallback wiener. Exception:", repr(e))
        try:
            return wiener(patch, mysize=(3, 3))
        except Exception as e2:
            print("Wiener fallback failed:", repr(e2))


def extract_noise_residual_patchwise(img, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """
    Patch-wise extraction: dzieli obraz na patchy z overlap, odszumia BM3D per-patch,
    agreguje residuale przez overlap-add z oknem Hann, a na końcu normalizuje
    (zero-mean, unit-std) tak jak w pierwotnej funkcji.
    """
    gray = load_image_as_gray(img)
    H, W = gray.shape

    overlap_pixels = int(round(patch_size * overlap))
    step = patch_size - overlap_pixels
    if step < 1:
        step = 1

    residual_sum = np.zeros_like(gray, dtype=np.float64)
    weight_sum = np.zeros_like(gray, dtype=np.float64)

    sigma_global_est = max(MIN_SIGMA, min(MAX_SIGMA, float(np.std(gray) if np.std(gray) > 0 else MIN_SIGMA)))

    for ys in _patch_indices(H, patch_size, step):
        y0, y1 = ys
        for xs in _patch_indices(W, patch_size, step):
            x0, x1 = xs

            patch = gray[y0:y1, x0:x1]
            ph, pw = patch.shape

            sigma_patch = float(np.std(patch))
            if not np.isfinite(sigma_patch) or sigma_patch <= 0:
                sigma_patch = sigma_global_est
            sigma_patch = float(np.clip(sigma_patch, MIN_SIGMA, MAX_SIGMA))

            den = _denoise_patch_bm3d(patch, sigma_patch)

            resid_patch = patch - den
            resid_patch = resid_patch.astype(np.float64)

            win = _hann2d((ph, pw))

            residual_sum[y0:y1, x0:x1] += resid_patch * win
            weight_sum[y0:y1, x0:x1] += win

    assembled = residual_sum / (weight_sum + EPS)

    assembled = assembled - np.mean(assembled)
    std = assembled.std() + EPS
    return (assembled / std).astype(np.float32)


def extract_noise_residual(img, method='bm3d'):
    """Return residual = image - denoised(image)
       Dla BM3D: używa patch-wise jeśli obraz jest duży (big dim threshold).
    """
    gray = load_image_as_gray(img)

    if method == 'bm3d':
        H, W = gray.shape
        if max(H, W) > BIG_DIM_THRESHOLD:
            try:
                return extract_noise_residual_patchwise(gray)
            except Exception as e:
                print("Patch-wise BM3D failed, falling back to single-call BM3D. Exception:", repr(e))
        try:
            sigma_est = float(np.std(gray))
            sigma_est = float(np.clip(sigma_est if np.isfinite(sigma_est) and sigma_est > 0 else MIN_SIGMA,
                                      MIN_SIGMA, MAX_SIGMA))
            gray_c = np.ascontiguousarray(gray.astype(np.float64))
            den = bm3d(gray_c, sigma_psd=sigma_est)
            den = np.asarray(den, dtype=gray.dtype)
        except Exception as e:
            print("BM3D single-call failed, fallback to Wiener. Exception:", repr(e))
            try:
                den = wiener(gray, mysize=(3, 3))
            except Exception as e2:
                print("Wiener fallback failed:", repr(e2))
                den = gray

    elif method == 'wavelet':
        try:
            den = denoise_wavelet(gray, channel_axis=None, rescale_sigma=True)
        except Exception as e:
            print("Wavelet denoise failed, fallback to wiener:", repr(e))
            den = wiener(gray, mysize=(3, 3))
    elif method == 'wiener':
        den = wiener(gray, mysize=(3, 3))
    else:
        den = gray

    residual = gray - den
    residual -= residual.mean()
    std = residual.std() + EPS
    return (residual / std).astype(np.float32)


def build_reference_pattern(images, denoise_method='bm3d'):
    imgs = [load_image_as_gray(im) for im in images]
    min_h = min(im.shape[0] for im in imgs)
    min_w = min(im.shape[1] for im in imgs)
    imgs = [im[:min_h, :min_w] for im in imgs]

    K_num = np.zeros((min_h, min_w), dtype=np.float64)
    K_den = np.zeros((min_h, min_w), dtype=np.float64)

    for im in imgs:
        W = extract_noise_residual(im, method=denoise_method)
        W = W[:min_h, :min_w]
        im = im[:min_h, :min_w]
        K_num += W * im
        K_den += im * im

    K_ref = K_num / (K_den + EPS)
    K_ref -= np.mean(K_ref)
    try:
        K_ref = wiener(K_ref, mysize=(3, 3))
    except Exception as e:
        print("Wiener smoothing on K_ref failed:", repr(e))
    return K_ref.astype(np.float32)


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
    bg_energy = np.mean(cc_map[mask] ** 2)
    if bg_energy < EPS:
        return float('inf'), float(peak_val), (int(idx[0]), int(idx[1]))
    pce = (float(peak_val) ** 2) / float(bg_energy)
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
        den = denoise_wavelet(gray, channel_axis=None, rescale_sigma=True)
        smoothness = np.var(gray) / (np.var(den) + EPS)
    except Exception:
        smoothness = 1.0

    nua_flag = (ratio > 2.5) or (smoothness < 1.2)
    diag = {'wavelet_ratio': float(ratio), 'smoothness': float(smoothness)}
    return bool(nua_flag), diag


def _match_shapes(ref, test):
    """Ensure same shape by center crop"""
    h = min(ref.shape[0], test.shape[0])
    w = min(ref.shape[1], test.shape[1])
    return ref[:h, :w], test[:h, :w]


def compare_with_reference(test_in, ref_list, denoise_method='bm3d'):
    from skimage import io

    def ensure_array(x):
        return io.imread(x) if isinstance(x, str) else x

    ref_arrays = [ensure_array(r) for r in ref_list]
    test_arr = ensure_array(test_in)

    K_ref = build_reference_pattern(ref_arrays, denoise_method=denoise_method)
    W_test = extract_noise_residual(test_arr, method=denoise_method)

    K_ref, W_test = _match_shapes(K_ref, W_test)
    test_lum = load_image_as_gray(test_arr)
    _, test_lum = _match_shapes(K_ref, test_lum)

    template = K_ref * test_lum
    cc_map = cross_correlation_map(W_test, template)
    pce, peak, peak_coords = compute_pce(cc_map)

    try:
        ref_mean = np.mean(np.stack([load_image_as_gray(r) for r in ref_arrays]), axis=0)
    except Exception:
        ref_mean = load_image_as_gray(ref_arrays[0])

    ref_mean, _ = _match_shapes(ref_mean, K_ref)

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
