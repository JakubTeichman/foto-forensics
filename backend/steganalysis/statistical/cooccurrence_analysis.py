"""
Residual co-occurrence statistics (SRM-like simplified):
- Compute HPF residual, quantize to small range, compute 2D co-occurrence histograms for offsets
- Flatten features -> compute entropy/energy stats -> map to score
"""
import numpy as np
import cv2
from PIL import Image
import io

def _highpass(img):
    kernel = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)

def _quantize(arr, q=5):
    # quantize residual to -q..q
    clipped = np.clip(np.round(arr), -q, q).astype(np.int8)
    shifted = clipped + q  # 0..2q
    return shifted

def analyze(image_bytes=None, pil_image=None, q=4, offsets=[(0,1),(1,0),(1,1)], threshold=0.6):
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
    arr = np.array(pil_image, dtype=np.float32)
    res = _highpass(arr)
    qarr = _quantize(res, q=q)
    maxv = 2*q + 1
    # compute co-occurrence for each offset
    feats = []
    details = {}
    h,w = qarr.shape
    for dx,dy in offsets:
        # build 2D hist: (value at (i,j), value at (i+dy, j+dx))
        H = np.zeros((maxv, maxv), dtype=np.float64)
        for y in range(h - dy):
            for x in range(w - dx):
                a = qarr[y,x]; b = qarr[y+dy, x+dx]
                H[a,b] += 1
        p = H / (H.sum() + 1e-12)
        p_nonzero = p[p>0]
        ent = -np.sum(p_nonzero * np.log2(p_nonzero))
        energy = float(np.sum(H**2))
        feats.append(ent); feats.append(energy)
        details[f"offset_{dx}_{dy}"] = {"entropy":float(ent),"energy":float(energy)}
    # score: normalize entropy relative to max entropy
    max_ent = np.log2((2*q+1)**2)
    mean_ent = float(np.mean([f for f in feats[::2]])) if len(feats)>0 else 0.0
    score = float(np.clip(mean_ent / (max_ent+1e-12), 0.0, 1.0))
    detected = score >= threshold
    return {"method":"cooccurrence_analysis","score":score,"detected":bool(detected),"details":details}
