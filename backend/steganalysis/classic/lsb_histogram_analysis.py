# classic/lsb_histogram_analysis.py
import numpy as np
from PIL import Image
import io

def _lsb_prop_channel(arr):
    flat = arr.ravel().astype(np.uint8)
    ones = int((flat & 1).sum())
    total = int(flat.size)
    return ones / (total + 1e-12), ones, total

def _moran_i(bits_arr):
    b = bits_arr.astype(float)
    n = b.size
    if n <= 1:
        return 0.0
    mean = b.mean()
    num = 0.0
    denom = ((b - mean)**2).sum() + 1e-12
    for i in range(n - 1):
        num += (b[i] - mean) * (b[i + 1] - mean)
    I = (n / (n - 1)) * (num / (denom + 1e-12))
    return float(I)

def analyze(image_bytes=None, pil_image=None, patch_size=128, stride=64, threshold=0.6):
    """
    LSB histogram + patch-aware detector.
    Works with RGB or grayscale (will expand to 3 channels).
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    img = np.array(pil_image, dtype=np.uint8)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    h, w, _ = img.shape
    y = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)

    scores = []
    details = {}
    for ch_name, ch_arr in [('Y', y), ('R', img[..., 0]), ('G', img[..., 1]), ('B', img[..., 2])]:
        prop, ones, total = _lsb_prop_channel(ch_arr)
        score_ch = max(0.0, 1.0 - 2.0 * abs(prop - 0.5))
        I = _moran_i((ch_arr.ravel() & 1))
        details[ch_name] = {"lsb_prop": float(prop), "ones": int(ones), "total": int(total), "score_raw": float(score_ch), "moran_I": float(I)}
        scores.append(score_ch)

    patch_scores = []
    for y0 in range(0, max(1, h - patch_size + 1), max(1, stride)):
        for x0 in range(0, max(1, w - patch_size + 1), max(1, stride)):
            patch = y[y0:y0+patch_size, x0:x0+patch_size]
            if patch.size == 0:
                continue
            p_prop, p_ones, p_total = _lsb_prop_channel(patch)
            p_score = max(0.0, 1.0 - 2.0 * abs(p_prop - 0.5))
            patch_scores.append(p_score)

    patch_mean = float(np.mean(patch_scores)) if patch_scores else 0.0
    patch_std = float(np.std(patch_scores)) if patch_scores else 0.0

    final_score = float(0.5 * np.mean(scores) + 0.5 * patch_mean)
    detected = final_score >= threshold

    details["patch_mean"] = patch_mean
    details["patch_std"] = patch_std
    details["channel_scores"] = {k: details[k]["score_raw"] for k in ['Y', 'R', 'G', 'B']}

    return {"method": "lsb_histogram", "score": final_score, "detected": bool(detected), "details": details}
