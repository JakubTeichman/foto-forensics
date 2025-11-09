import numpy as np
from skimage import restoration
from PIL import Image
import io

# ----------------------
# Helpers (local)
# ----------------------
def _compute_shannon_entropy(arr, bins=256):
    hist, _ = np.histogram(arr.ravel(), bins=bins, range=(0, 255))
    probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def _normalize_scores(scores):
    a = np.array(scores, dtype=np.float64)
    if a.size == 0:
        return 0.0
    mn = a.min(); mx = a.max()
    if mx - mn < 1e-12:
        return float(np.mean(a))
    norm = (a - mn) / (mx - mn)
    return float(np.clip(np.mean(norm), 0.0, 1.0))

def _rescale_score(value, mean_ref=0.5, scale_ref=0.25):
    val = (value - mean_ref) / (scale_ref + 1e-12) / 2.0 + 0.5
    return float(np.clip(val, 0.0, 1.0))

# ----------------------
# original helpers
# ----------------------
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

# ----------------------
# analyze (updated)
# ----------------------
def analyze(image_bytes=None, pil_image=None, patch_size=128, stride=64, threshold=0.6):
    """
    Noise residuals: denoise (wavelet), residual = image - denoised.
    Extract per-patch higher-order stats, map to score (updated).
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
    energies = []

    for y0 in range(0, max(1, h - patch_size + 1), max(1, stride)):
        for x0 in range(0, max(1, w - patch_size + 1), max(1, stride)):
            patch = (residual[y0:y0+patch_size, x0:x0+patch_size] * 255.0).astype(np.int16)
            if patch.size == 0:
                continue
            st = _patch_statistics(patch)
            stats.append(st)
            energies.append(st["energy"])

    if len(stats) == 0:
        return {"method": "noise_residuals", "score": 0.0, "detected": False, "details": {"error": "image too small"}}

    mean_skew = float(np.mean([s["skew"] for s in stats]))
    mean_kurt = float(np.mean([s["kurt"] for s in stats]))
    mean_energy = float(np.mean(energies))
    std_energy = float(np.std(energies))

    # original raw mapping
    score_raw = (abs(mean_kurt) / 10.0) + min(1.0, std_energy / (mean_energy + 1e-12) / 10.0)
    score_raw = float(np.clip(score_raw, 0.0, 1.0))

    # normalized energy distribution
    norm_energy = _normalize_scores(energies) if energies else score_raw

    # entropy of residuals (image-level)
    residual_uint8 = np.clip(((residual * 255.0)).astype(np.int16), 0, 255).astype(np.uint8)
    entropy = _compute_shannon_entropy(residual_uint8)
    entropy_weight = float(np.clip(entropy / 7.5, 0.8, 1.3))

    combined = float(np.clip(0.65 * score_raw + 0.35 * norm_energy, 0.0, 1.0))
    combined = float(np.clip(combined * entropy_weight, 0.0, 1.0))

    # dynamic threshold based on energy histogram & entropy
    hist, _ = np.histogram(energies, bins=8)
    hist_probs = hist.astype(np.float64) / (hist.sum() + 1e-12)
    hist_entropy = float(-np.sum(hist_probs[hist_probs>0] * np.log2(hist_probs[hist_probs>0] + 1e-12)))
    dyn_threshold = float(np.clip(0.45 + 0.12 * (hist_entropy / np.log2(8)) + 0.04 * ((entropy - 2.5) / 4.0), 0.2, 0.85))

    score = _rescale_score(combined, mean_ref=0.5, scale_ref=0.25)
    detected = bool(score >= dyn_threshold)

    details = {
        "mean_skew": mean_skew, "mean_kurt": mean_kurt,
        "mean_energy": mean_energy, "std_energy": std_energy,
        "n_patches": len(stats), "entropy": entropy,
        "combined_raw": combined, "dynamic_threshold": dyn_threshold
    }
    return {"method": "noise_residuals", "score": float(score), "detected": bool(detected), "details": details}
