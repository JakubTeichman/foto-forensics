# statistical/cooccurrence_analysis.py
import numpy as np
import cv2
from PIL import Image
import io

def _highpass(img):
    kernel = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)

def _quantize(arr, q=5):
    clipped = np.clip(np.round(arr), -q, q).astype(np.int8)
    shifted = clipped + q
    return shifted

def analyze(image_bytes=None, pil_image=None, q=4, offsets=None, threshold=0.6):
    """
    Residual co-occurrence (simplified SRM-like).
    Returns entropy/energy features mapped to 0..1 score.
    """
    if offsets is None:
        offsets = [(0,1), (1,0), (1,1)]

    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.float32)
    res = _highpass(arr)
    qarr = _quantize(res, q=q)
    maxv = 2 * q + 1

    feats = []
    details = {}
    h, w = qarr.shape

    for dx, dy in offsets:
        H = np.zeros((maxv, maxv), dtype=np.float64)
        for y in range(0, h - dy):
            for x in range(0, w - dx):
                a = int(qarr[y, x])
                b = int(qarr[y + dy, x + dx])
                if 0 <= a < maxv and 0 <= b < maxv:
                    H[a, b] += 1
        total = H.sum() + 1e-12
        p = H / total
        p_nonzero = p[p > 0]
        ent = -np.sum(p_nonzero * np.log2(p_nonzero + 1e-12)) if p_nonzero.size > 0 else 0.0
        energy = float(np.sum(H**2))
        feats.append({"entropy": float(ent), "energy": energy})
        details[f"offset_{dx}_{dy}"] = {"entropy": float(ent), "energy": energy}

    if len(feats) == 0:
        score = 0.0
    else:
        ent_mean = float(np.mean([f["entropy"] for f in feats]))
        max_ent = np.log2((2*q+1)**2)
        score = float(np.clip(ent_mean / (max_ent + 1e-12), 0.0, 1.0))

    detected = score >= threshold
    return {"method": "cooccurrence_analysis", "score": score, "detected": bool(detected), "details": details}
