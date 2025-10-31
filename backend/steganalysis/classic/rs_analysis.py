"""
RS method (Regular/Singular groups) — rozszerzona, zgodna z ideą Fridrich & Goljan.
Implementacja (praktyczna):
 - Definiujemy funkcję discriminant (f) jako suma bezwzględnych różnic w grupie
 - Grupujemy piksele w bloki (np. 2x2 lub 4x4) z ustalonym porządkiem
 - Pola R,S liczymy przed i po flipie LSB na określonym masce (flipping vector)
 - Obliczamy delta = (R - S) przed i po; na tej podstawie estymujemy payload
Zwraca score bazujący na oszacowanym udziale wstawienia i statystykach.
"""
import numpy as np
from PIL import Image
import io

def _group_function(block):
    # block: 2D uint8; discriminant = sum abs diff neighbors
    p = block.astype(np.int32)
    s = 0
    s += np.abs(p[:,1:] - p[:,:-1]).sum() if p.shape[1]>1 else 0
    s += np.abs(p[1:,:] - p[:-1,:]).sum() if p.shape[0]>1 else 0
    return float(s)

def analyze(image_bytes=None, pil_image=None, group_size=2, threshold=0.4):
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
    arr = np.array(pil_image, dtype=np.uint8)
    h,w = arr.shape
    R0 = 0.0; S0 = 0.0; Rf = 0.0; Sf = 0.0; total_groups = 0
    group_changes = []
    for y0 in range(0, h - group_size + 1, group_size):
        for x0 in range(0, w - group_size + 1, group_size):
            blk = arr[y0:y0+group_size, x0:x0+group_size]
            f_orig = _group_function(blk)
            # flip LSB of whole block
            flipped_blk = blk ^ 1
            f_flipped = _group_function(flipped_blk)
            # classification as regular or singular depending on if f increases/decreases
            # we follow Fridrich: if f_orig > f_flipped -> group considered regular (R), else singular (S)
            if f_orig > f_flipped:
                R0 += 1
            else:
                S0 += 1
            # now consider "mask flipping" (flip some pixels only) - approximate by flipping LSB only in alternate positions
            mask = np.zeros_like(blk)
            mask[::2,::2] = 1
            blk_masked = blk ^ mask
            f_masked = _group_function(blk_masked)
            if f_masked > f_flipped:
                Rf += 1
            else:
                Sf += 1
            total_groups += 1
            group_changes.append({"x":x0,"y":y0,"f_orig":f_orig,"f_flipped":f_flipped,"f_masked":f_masked})
    if total_groups == 0:
        score = 0.0
    else:
        # estimate embedding rate approximated by ( (R0 - Rf) + (Sf - S0) ) / total
        est = ( (R0 - Rf) + (Sf - S0) ) / (2.0 * total_groups + 1e-12)
        # scale and clamp
        score = float(max(0.0, min(1.0, abs(est))))
    detected = score >= threshold
    details = {"R0":int(R0),"S0":int(S0),"Rf":int(Rf),"Sf":int(Sf),"groups": total_groups, "group_examples": group_changes[:20]}
    return {"method":"rs_analysis", "score": score, "detected": bool(detected), "details": details}
