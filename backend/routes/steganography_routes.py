# === backend/steganography_routes.py ===

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from stegano_compare.stegano_compare_main import analyze_images
import os, tempfile
import traceback
from PIL import Image

# Importy z modułów analizy
from steganalysis.analyze_stegano_single import AnalyzeSteganoSingle
from steganalysis.aggregator import analyze as aggregator_analyze

# Tworzymy blueprint
steganography_bp = Blueprint("steganography_bp", __name__)

# Folder tymczasowy na przesłane pliki
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# === Endpoint 1: Szybka analiza (AnalyzeSteganoSingle) ===
@steganography_bp.route("/stegano/analyze", methods=["POST"])
def analyze_steganography():
    """
    Quick steganography analysis for a single image using classical, statistical, and ML methods.
    Compatible with the current frontend SteganoReport component.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided in the request."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Wczytaj obraz jako PIL.Image (bo analizy tego wymagają)
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Cannot open image: {str(e)}"}), 400

        # Run analysis
        analyzer = AnalyzeSteganoSingle()
        results = analyzer.analyze(pil_image=image)

        # Poprawne sprawdzenie wykrycia
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
        return jsonify({
            "error": f"Steganography analysis failed: {str(e)}"
        }), 500


# === Endpoint 2: Pełna analiza (Aggregator) ===
@steganography_bp.route("/stegano/analyze_full", methods=["POST"])
def analyze_steganography_full():
    """
    Full aggregated steganography analysis.
    Uses the aggregator.py pipeline with all supported classical/statistical methods.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided in the request."}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400

        # Read file as bytes (aggregator expects bytes)
        image_bytes = file.read()

        # Run full aggregated analysis
        report = aggregator_analyze(image_bytes=image_bytes, include_meta=True)

        # Determine detection result
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
        return jsonify({
            "error": f"Full steganography analysis failed: {str(e)}"
        }), 500

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

        result = analyze_images(orig_path, susp_path)

    return jsonify(result)