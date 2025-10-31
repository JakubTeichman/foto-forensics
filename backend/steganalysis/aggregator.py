# backend/steganalysis/aggregator.py
import time
import io
from PIL import Image

# Classic
from .classic.lsb_histogram_analysis import analyze as lsb_analyze
from .classic.chi_square_analysis import analyze as chi_analyze
from .classic.rs_analysis import analyze as rs_analyze

# Statistical
from .statistical.noise_residuals import analyze as noise_residuals_analyze
from .statistical.wavelet_analysis import analyze as wavelet_analyze
from .statistical.cooccurrence_analysis import analyze as cooc_analyze

# CNN / ML hooks (commented until you have models ready)
# from .ml.svm_detector import predict as svm_predict
# from .ml.logistic_regression_detector import predict as logreg_predict
# from .cnn.cnn_detector import predict as cnn_predict
# from .cnn.noiseprint_extractor import analyze as noiseprint_analyze

SUPPORTED_METHODS = [
    ("lsb_histogram", lsb_analyze),
    ("chi_square", chi_analyze),
    ("rs_analysis", rs_analyze),
    ("noise_residuals", noise_residuals_analyze),
    ("wavelet", wavelet_analyze),
    ("cooccurrence", cooc_analyze),
    # add ML/CNN entries here when models available
    # ("ml_svm", svm_predict),
    # ("ml_logreg", logreg_predict),
    # ("cnn", cnn_predict),
    # ("noiseprint", noiseprint_analyze),
]

def _ensure_pil(image_bytes=None):
    """Return PIL.Image (RGB) from bytes; raises ValueError if none given."""
    if image_bytes is None:
        raise ValueError("image_bytes required")
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

def analyze(image_bytes: bytes, include_meta: bool = True, timeout_s: float = 30.0):
    """
    Run all supported methods on a single image (no aggregation).
    Returns:
      {
        "analysis_results": {
            "lsb_histogram": {method dict...},
            "chi_square": {...},
            ...
        },
        "meta": {"processing_time_s": float, "version": "v1.0"}
      }
    """
    t0 = time.time()
    pil_img = None
    results = {}

    # pre-create PIL image once to reuse (some analyzers accept pil_image)
    try:
        pil_img = _ensure_pil(image_bytes=image_bytes)
    except Exception as e:
        # if image can't be loaded -> return error
        return {"error": f"invalid image bytes: {e}"}

    for key, func in SUPPORTED_METHODS:
        try:
            # Try call signature: prefer `pil_image` if analyzer accepts it
            try:
                res = func(pil_image=pil_img)
            except TypeError:
                # fallback: pass bytes
                res = func(image_bytes=image_bytes)
            # Ensure returned format
            if not isinstance(res, dict):
                res = {"method": key, "score": 0.0, "detected": False, "details": {"raw_return": str(res)}}
        except Exception as exc:
            res = {"method": key, "score": 0.0, "detected": False, "details": {"error": str(exc)}}
        results[key] = res

    # ML/CNN hooks - commented: uncomment and implement predict wrappers that accept pil_image
    # try:
    #     svm_res = svm_predict(pil_image)
    #     results["ml_svm"] = svm_res
    # except Exception as e:
    #     results["ml_svm"] = {"method":"ml_svm","score":0.0,"detected":False,"details":{"error":str(e)}}

    t1 = time.time()
    out = {"analysis_results": results}
    if include_meta:
        out["meta"] = {"processing_time_s": float(t1 - t0), "method_count": len(results)}
    return out

# Example usage:
# with open("example.jpg", "rb") as f:
#     bytes_img = f.read()
# report = analyze(bytes_img)
# print(report)
