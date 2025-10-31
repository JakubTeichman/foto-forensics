"""
Wavelet multi-scale high-order statistics:
- DWT decomposition (multiple levels)
- Compute energy and higher-order moments for detail subbands (HL,LH,HH)
- Compare distribution across scales (stego often affects high-frequency bands)
"""
import pywt
import numpy as np
from PIL import Image
import io

def analyze(image_bytes=None, pil_image=None, wave='db1', levels=2, threshold=0.5):
    if pil_image is None:
        if image_bytes is None:
            raise ValueError("image_bytes or pil_image required")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    coeffs = pywt.wavedec2(arr, wavelet=wave, level=levels)
    # coeffs[0] is approximation at coarsest level, coeffs[1:] are tuples (cH,cV,cD)
    details = {}
    band_stats = []
    for lvl, tup in enumerate(coeffs[1:], start=1):
        cH, cV, cD = tup
        for name, band in [('cH',cH), ('cV',cV), ('cD',cD)]:
            e = float(np.mean(np.abs(band)))
            s = float(np.std(band))
            kurt = float(np.mean(((band - band.mean())/(s+1e-12))**4) - 3.0) if s>0 else 0.0
            band_stats.append({"level":lvl,"band":name,"energy":e,"std":s,"kurtosis":kurt})
    # heuristics: if mean energy in detail bands significantly larger -> score increases
    energies = np.array([b['energy'] for b in band_stats])
    if energies.size==0:
        score = 0.0
    else:
        score = float(np.clip(np.mean(energies) / (np.max(energies)+1e-12), 0, 1))
    details['bands'] = band_stats[:20]
    return {"method":"wavelet_analysis","score":score,"detected": score>=threshold, "details": details}
