# statistical/noise_residuals.py
import numpy as np
from skimage import restoration
from PIL import Image
import io

def _patch_statistics(patch):
    v = patch.ravel().astype(np.float64)
    mean = float(np.mean(v))
    std = float(np.std(v))
    if std > 0:
        skew = float(np.mean(((v - mean)/std)**3))
        kurt = float(np.mean(((v - mean)/std)**4) - 3.0)
    else:
        skew = 0.0; kurt = 0.0
    energy = float(np.sum(v**2))
    return {"mean": mean, "std": std, "skew": skew, "kurt": kurt, "energy": energy}

def analyze(image_bytes=None, pil_image=None, patch_size=128, stride=64, threshold=0.6):
    """
    Noise residuals: denoise (wavelet), residual = image - denoised.
    Extract per-patch higher-order stats, map to score.
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.float32) / 255.0

    # use new API if available
    try:
        den = restoration.denoise_wavelet(arr, channel_axis=None, rescale_sigma=True)
    except TypeError:
        den = restoration.denoise_wavelet(arr, multichannel=False, convert2ycbcr=False)

    residual = arr - den
    h, w = residual.shape
    stats = []

    for y0 in range(0, max(1, h - patch_size + 1), max(1, stride)):
        for x0 in range(0, max(1, w - patch_size + 1), max(1, stride)):
            patch = (residual[y0:y0+patch_size, x0:x0+patch_size] * 255.0).astype(np.int16)
            if patch.size == 0:
                continue
            stats.append(_patch_statistics(patch))

    if len(stats) == 0:
        return {"method": "noise_residuals", "score": 0.0, "detected": False, "details": {"error": "image too small"}}

    mean_skew = float(np.mean([s["skew"] for s in stats]))
    mean_kurt = float(np.mean([s["kurt"] for s in stats]))
    mean_energy = float(np.mean([s["energy"] for s in stats]))
    std_energy = float(np.std([s["energy"] for s in stats]))

    score_raw = (abs(mean_kurt) / 10.0) + min(1.0, std_energy / (mean_energy + 1e-12) / 10.0)
    score = float(max(0.0, min(1.0, score_raw)))
    detected = score >= threshold
    details = {"mean_skew": mean_skew, "mean_kurt": mean_kurt, "mean_energy": mean_energy, "std_energy": std_energy, "n_patches": len(stats)}
    return {"method": "noise_residuals", "score": score, "detected": bool(detected), "details": details}
