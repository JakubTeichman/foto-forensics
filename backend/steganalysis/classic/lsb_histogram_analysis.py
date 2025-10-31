"""
LSB Histogram Analysis (rozszerzona wersja).
- Analiza globalna i patchowa LSB (luminancja i każdy kanał RGB)
- Pomiar odchylenia od oczekiwanego rozkładu 50/50 dla bitów LSB
- Dodatkowo: statystyki Moran I (autokorelacja bitów) i blokowe testy
Zwraca: dict z score (0..1), detected bool i szczegółami.
"""
import numpy as np
from PIL import Image

def _lsb_prop_channel(arr):
    """arr uint8 2D -> proportion of ones in LSB"""
    flat = arr.ravel().astype(np.uint8)
    ones = int((flat & 1).sum())
    total = flat.size
    return ones / (total + 1e-12), ones, total

def _moran_i(bits_arr):
    # bits_arr 1D of {0,1} -> center and compute Moran's I for adjacency
    b = bits_arr.astype(float)
    n = b.size
    if n <= 1:
        return 0.0
    mean = b.mean()
    num = 0.0
    denom = ((b - mean)**2).sum() + 1e-12
    # adjacency weight = 1 for neighbors (i,i+1)
    for i in range(n-1):
        num += (b[i] - mean)*(b[i+1] - mean)
    # normalize by number of links (n-1)
    denom_factor = denom
    I = (n / (n-1)) * (num / (denom_factor + 1e-12))
    return float(I)

def analyze(image_bytes=None, pil_image=None, patch_size=128, stride=64, threshold=0.6):
    """
    Wejście:
      - image_bytes (bytes) OR pil_image (PIL.Image)
    Zwraca:
      {"method":"lsb_histogram", "score":.., "detected": bool, "details": {...}}
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = np.array(pil_image, dtype=np.uint8)
    h,w,_ = img.shape
    # luminance
    y = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.uint8)
    # global: luminance + channels
    scores = []
    details = {}
    for ch_name, ch_arr in [('Y', y), ('R', img[...,0]), ('G', img[...,1]), ('B', img[...,2])]:
        prop, ones, total = _lsb_prop_channel(ch_arr)
        # ideal prop 0.5 => score = 1 - 2*abs(prop-0.5)
        score = max(0.0, 1.0 - 2.0 * abs(prop - 0.5))
        # moran
        I = _moran_i((ch_arr.ravel() & 1))
        details[ch_name] = {"lsb_prop": prop, "ones": int(ones), "total": int(total), "score_raw": float(score), "moran_I": float(I)}
        scores.append(score)
    # patch-based: compute mean and var of patch scores (luminance)
    patch_scores = []
    for y0 in range(0, h - patch_size + 1, stride):
        for x0 in range(0, w - patch_size + 1, stride):
            patch = y[y0:y0+patch_size, x0:x0+patch_size]
            p_prop, p_ones, p_total = _lsb_prop_channel(patch)
            p_score = max(0.0, 1.0 - 2.0 * abs(p_prop - 0.5))
            patch_scores.append(p_score)
    if len(patch_scores)==0:
        patch_mean = 0.0
        patch_std = 0.0
    else:
        patch_mean = float(np.mean(patch_scores))
        patch_std = float(np.std(patch_scores))
    # aggregate: weighted mean (global channels + patch mean)
    final_score = float(0.5 * np.mean(scores) + 0.5 * patch_mean)
    detected = final_score >= threshold
    details["patch_mean"] = patch_mean
    details["patch_std"] = patch_std
    return {"method":"lsb_histogram", "score": final_score, "detected": bool(detected), "details": details}

# Note: requires `io` import if using image_bytes; keep aggregator to pass PIL images or bytes.
import io
