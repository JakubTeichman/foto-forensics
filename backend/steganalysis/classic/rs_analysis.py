# classic/rs_analysis.py
import numpy as np
from PIL import Image
import io

def _group_function(block):
    p = block.astype(np.int32)
    s = 0
    if p.shape[1] > 1:
        s += np.abs(p[:,1:] - p[:,:-1]).sum()
    if p.shape[0] > 1:
        s += np.abs(p[1:,:] - p[:-1,:]).sum()
    return float(s)

def analyze(image_bytes=None, pil_image=None, group_size=2, threshold=0.4):
    """
    RS analysis (simplified Fridrich/Goljan style).
    """
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    arr = np.array(pil_image, dtype=np.uint8)
    h, w = arr.shape
    R0 = S0 = Rf = Sf = 0.0
    total_groups = 0
    group_changes = []

    step = group_size
    for y0 in range(0, h - group_size + 1, step):
        for x0 in range(0, w - group_size + 1, step):
            blk = arr[y0:y0+group_size, x0:x0+group_size]
            if blk.size == 0:
                continue
            f_orig = _group_function(blk)
            flipped_blk = blk ^ 1
            f_flipped = _group_function(flipped_blk)
            if f_orig > f_flipped:
                R0 += 1
            else:
                S0 += 1
            mask = np.zeros_like(blk)
            mask[::2, ::2] = 1
            blk_masked = blk ^ mask
            f_masked = _group_function(blk_masked)
            if f_masked > f_flipped:
                Rf += 1
            else:
                Sf += 1
            total_groups += 1
            if len(group_changes) < 20:
                group_changes.append({"x": x0, "y": y0, "f_orig": f_orig, "f_flipped": f_flipped, "f_masked": f_masked})

    if total_groups == 0:
        score = 0.0
    else:
        est = ((R0 - Rf) + (Sf - S0)) / (2.0 * total_groups + 1e-12)
        score = float(max(0.0, min(1.0, abs(est))))

    detected = score >= threshold
    details = {"R0": int(R0), "S0": int(S0), "Rf": int(Rf), "Sf": int(Sf), "groups": total_groups, "group_examples": group_changes}
    return {"method": "rs_analysis", "score": score, "detected": bool(detected), "details": details}
