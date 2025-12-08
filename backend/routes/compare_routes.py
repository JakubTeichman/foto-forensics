import os
import sys
import traceback
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import numpy as np
from skimage import io
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# Add parent folder to path so imports work in Docker
sys.path.append(str(Path(__file__).resolve().parent.parent))

compare_bp = Blueprint("compare", __name__)

# Import PRNU utilities and pipeline
from prnu_utils.prnu_extraction import (
    extract_prnu_from_path,
    extract_prnu_from_bytes
)
from prnu_pipeline import compare_with_reference

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================================================================
# Endpoint: compare (two images)
# ================================================================
@compare_bp.route('/compare', methods=['POST'])
def compare_images():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Missing files'}), 400

        img1 = request.files['image1']
        img2 = request.files['image2']

        # OPTIONAL: allow selecting denoise method (default = bm3d)
        denoise_method = request.form.get('denoise_method', 'bm3d').lower()
        if denoise_method not in ['bm3d', 'wavelet']:
            denoise_method = 'bm3d'

        path1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
        path2 = os.path.join(UPLOAD_FOLDER, secure_filename(img2.filename))
        img1.save(path1)
        img2.save(path2)

        # Quick size check
        image1 = io.imread(path1)
        image2 = io.imread(path2)
        size_warning = image1.shape != image2.shape
        warnings = []
        if size_warning:
            warnings.append({
                "type": "warning",
                "message": f"Obrazy mają różne rozdzielczości: {image1.shape} vs {image2.shape}. "
                           "Wynik podobieństwa może być mniej dokładny."
            })

        # Run comparison pipeline
        pipeline_res = compare_with_reference(path1, [path2], denoise_method=denoise_method)

        resp = {
            'similarity': float(pipeline_res.get('PCE', 0.0)),
            'PCE': float(pipeline_res.get('PCE', 0.0)),
            'peak': float(pipeline_res.get('peak', 0.0)),
            'NUA_ref': pipeline_res.get('NUA_ref', False),
            'NUA_test': pipeline_res.get('NUA_test', False),
            'diag_ref': pipeline_res.get('diag_ref', {}),
            'diag_test': pipeline_res.get('diag_test', {}),
            'size_warning': bool(size_warning),
            'denoise_method': denoise_method,
            'warnings': warnings
        }
        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


# ================================================================
# Endpoint: compare-multiple
# ================================================================
@compare_bp.route('/compare-multiple', methods=['POST'])
def compare_multiple():
    try:
        if 'image1' not in request.files or 'images2' not in request.files:
            return jsonify({'error': 'Missing files'}), 400

        img1 = request.files['image1']
        images2 = request.files.getlist('images2')

        # OPTIONAL: allow selecting denoise method
        denoise_method = request.form.get('denoise_method', 'bm3d').lower()
        if denoise_method not in ['bm3d', 'wavelet']:
            denoise_method = 'bm3d'

        # Save test image
        path1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
        img1.save(path1)

        # Save reference images
        ref_paths = []
        for img in images2:
            p = os.path.join(UPLOAD_FOLDER, secure_filename(img.filename))
            img.save(p)
            ref_paths.append(p)

        # Resolution check
        img1_arr = io.imread(path1)
        size1 = img1_arr.shape
        warnings = []
        size_warning = False
        for rp in ref_paths:
            try:
                a = io.imread(rp)
                if a.shape != size1:
                    size_warning = True
                    warnings.append({
                        "type": "warning",
                        "message": f"Obraz '{os.path.basename(rp)}' ma inną rozdzielczość {a.shape} vs {size1}."
                    })
            except Exception:
                warnings.append({
                    "type": "error",
                    "message": f"Nie udało się odczytać obrazu '{os.path.basename(rp)}' do weryfikacji rozmiaru."
                })

        # Run pipeline
        pipeline_res = compare_with_reference(path1, ref_paths, denoise_method=denoise_method)

        resp = {
            'similarity': float(pipeline_res.get('PCE', 0.0)),
            'PCE': float(pipeline_res.get('PCE', 0.0)),
            'peak': float(pipeline_res.get('peak', 0.0)),
            'peak_coords': pipeline_res.get('peak_coords', None),
            'NUA_ref': pipeline_res.get('NUA_ref', False),
            'NUA_test': pipeline_res.get('NUA_test', False),
            'diag_ref': pipeline_res.get('diag_ref', {}),
            'diag_test': pipeline_res.get('diag_test', {}),
            'size_warning': bool(size_warning),
            'denoise_method': denoise_method,
            'warnings': warnings
        }
        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


# ================================================================
# Health endpoint
# ================================================================
@compare_bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200
