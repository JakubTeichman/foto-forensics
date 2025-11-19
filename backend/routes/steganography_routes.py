# stegano_routes.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os, tempfile, io, base64, traceback
from PIL import Image
import numpy as np

# existing analyzer imports (kept)
try:
    from stegano_compare.stegano_compare_main import analyze_images
except Exception:
    analyze_images = None

try:
    from steganalysis.analyze_stegano_single import AnalyzeSteganoSingle
except Exception:
    AnalyzeSteganoSingle = None

try:
    from steganalysis.aggregator import analyze as aggregator_analyze
except Exception:
    aggregator_analyze = None

# new imports
import torch
from stegano_compare.preprocessing import prepare_pair_batch
from stegano_compare.siamese_model import load_siamese_model
from stegano_compare.srm_filters import get_srm_bank
import matplotlib.pyplot as plt

steganography_bp = Blueprint("steganography_bp", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- existing endpoints (kept) ----------------
@steganography_bp.route("/stegano/analyze", methods=["POST"])
def analyze_steganography():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided in the request."}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Cannot open image: {str(e)}"}), 400
        if AnalyzeSteganoSingle is None:
            return jsonify({"error": "AnalyzeSteganoSingle not available on server."}), 500
        analyzer = AnalyzeSteganoSingle()
        results = analyzer.analyze(pil_image=image)
        hidden_detected = results.get("hidden_detected", False)
        return jsonify({
            "status": "success",
            "hidden_detected": hidden_detected,
            "method": "multi-method",
            "details": results,
            "methods_results": results.get("methods_results", {})
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Steganography analysis failed: {str(e)}"}), 500

@steganography_bp.route("/stegano/analyze_full", methods=["POST"])
def analyze_steganography_full():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided in the request."}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400
        image_bytes = file.read()
        if aggregator_analyze is None:
            return jsonify({"error": "aggregator_analyze not available on server."}), 500
        report = aggregator_analyze(image_bytes=image_bytes, include_meta=True)
        analysis_results = report.get("analysis_results", {})
        hidden_detected = any(
            v.get("detected", False)
            for v in analysis_results.values()
            if isinstance(v, dict)
        )
        response = {
            "status": "success",
            "hidden_detected": bool(hidden_detected),
            "method": "aggregated",
            "report": report,
        }
        return jsonify(response), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Full steganography analysis failed: {str(e)}"}), 500

@steganography_bp.route("/stegano/compare", methods=["POST"])
def stegano_compare():
    if "original" not in request.files or "suspicious" not in request.files:
        return jsonify({"error": "Missing files"}), 400
    orig = request.files["original"]
    susp = request.files["suspicious"]
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, orig.filename)
        susp_path = os.path.join(tmpdir, susp.filename)
        orig.save(orig_path)
        susp.save(susp_path)
        if analyze_images is None:
            return jsonify({"error": "analyze_images (stegano_compare_main) not available."}), 500
        result = analyze_images(orig_path, susp_path)
    return jsonify(result)

# ---------------- NEW endpoint: Siamese CNN compare ----------------
# configuration: model path and threshold (you can change path or set via env var)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /app/routes
MODEL_PATH = os.path.join(BASE_DIR, "..", "stegano_compare", "models", "siamese_multi_best.pth")
MODEL_PATH = os.path.abspath(MODEL_PATH)
THRESHOLD = float(os.environ.get("SIAMESE_THRESHOLD", "0.5920"))
IN_CHANNELS = int(os.environ.get("SIAMESE_IN_CHANNELS", "30"))
EMBED_DIM = int(os.environ.get("SIAMESE_EMBED_DIM", "256"))

# lazy model loader
_siamese = None
def _get_model():
    global _siamese
    if _siamese is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _siamese = load_siamese_model(MODEL_PATH, device=device, in_channels=IN_CHANNELS, embed_dim=EMBED_DIM)
    return _siamese

def _make_heatmap(a_full, b_full):
        """
        Heatmapa pokazująca realne różnice pikselowe.
        a_full, b_full: tensory (C,H,W) w przestrzeni obrazu [0..1]
        """
        import numpy as np
        from skimage.transform import resize

        # konwersja tensorów na numpy
        a = a_full.detach().cpu().numpy()
        b = b_full.detach().cpu().numpy()

        # bierzemy tylko kanały RGB (pierwsze 3)
        a = np.clip(a[:3], 0, 1)
        b = np.clip(b[:3], 0, 1)

        # różnica per-pixel → jedna mapa
        diff = np.abs(a - b).mean(axis=0)

        # podbijamy kontrast (ważne!)
        diff = diff ** 0.5     # gamma
        diff = diff / (diff.max() + 1e-9)

        # powiększamy do 256x256
        heat = resize(diff, (256, 256))

        return heat


@steganography_bp.route("/stegano/siamese", methods=["POST"])
def stegano_siamese():
    """
    Compare two images using trained SiameseMulti network.
    Input: multipart form-data with 'original' and 'suspicious' files.
    Output: JSON variant B:
      {
        "status":"success",
        "score": float (0..1),
        "threshold": float,
        "stego_detected": bool,
        "confidence": float (0..1),
        "heatmap_siamese": "data:image/png;base64,..." or None
      }
    """
    try:
        if "original" not in request.files or "suspicious" not in request.files:
            return jsonify({"error": "Missing files 'original' and 'suspicious'"}), 400

        orig = request.files["original"]
        susp = request.files["suspicious"]

        # prepare tensors (batched)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a_full_t, a_patch_t, b_full_t, b_patch_t = prepare_pair_batch(orig, susp, device=device, use_srm=True)
        # ensure shapes: (1, C, H, W)

        model = _get_model()
        if model is None:
            return jsonify({"error": "Siamese model not available on server."}), 500

        with torch.no_grad():
            out, ea, eb = model(a_full_t, a_patch_t, b_full_t, b_patch_t)
            score = float(out.cpu().numpy()[0])  # 0..1 ; larger -> more likely stego (per training head)
            distance = float(torch.norm(ea - eb, p=2, dim=1).cpu().numpy()[0])

        stego_detected = bool(score >= THRESHOLD)
        confidence = float(score)  # choose to expose score as confidence; alternative: abs(score-threshold)

        # try create heatmap
        heat = None
        try:
            heat_arr = _make_heatmap(a_full_t[0].cpu(), b_full_t[0].cpu())
            if heat_arr is not None:
                # convert heat to PNG in-memory
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig = plt.figure(frameon=False)
                fig.set_size_inches(4,4)
                ax = plt.Axes(fig, [0.,0.,1.,1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(heat_arr, cmap='magma')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                img_bytes = buf.read()
                heat = "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')
        except Exception:
            heat = None

        response = {
            "status": "success",
            "score": score,
            "threshold": THRESHOLD,
            "stego_detected": stego_detected,
            "confidence": confidence,
            "heatmap_siamese": heat
        }
        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Siamese analysis failed: {str(e)}"}), 500
