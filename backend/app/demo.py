import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import numpy as np
from skimage import io

# Dodaj katalog nadrzędny do sys.path, aby importy działały w Dockerze
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import funkcji PRNU
from .prnu_utils.prnu_extraction import (
    extract_prnu_from_path,
    extract_prnu_from_bytes
)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================================================================
# 🔹 Endpoint 1: Porównanie dwóch obrazów
# ================================================================
@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    img1 = request.files['image1']
    img2 = request.files['image2']

    path1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
    path2 = os.path.join(UPLOAD_FOLDER, secure_filename(img2.filename))
    img1.save(path1)
    img2.save(path2)

    # Odczyt obrazów i sprawdzenie rozdzielczości
    image1 = io.imread(path1)
    image2 = io.imread(path2)

    size_warning = image1.shape != image2.shape
    warnings = []

    if size_warning:
        warnings.append(
            f"The uploaded images have different resolutions: "
            f"{image1.shape} vs {image2.shape}. "
            f"This may affect the accuracy of the similarity result. "
            f"It is recommended to use images captured in the same resolution."
        )

    # ✅ Ekstrakcja PRNU z plików (poprawiona linia)
    prnu1 = extract_prnu_from_path(path1)
    prnu2 = extract_prnu_from_path(path2)

    # Jeśli różne rozmiary — przycinamy do wspólnego obszaru
    min_h = min(prnu1.shape[0], prnu2.shape[0])
    min_w = min(prnu1.shape[1], prnu2.shape[1])
    prnu1 = prnu1[:min_h, :min_w]
    prnu2 = prnu2[:min_h, :min_w]

    similarity = np.corrcoef(prnu1.flatten(), prnu2.flatten())[0, 1]

    return jsonify({
        'similarity': float(similarity),
        'size_warning': size_warning,
        'warnings': warnings
    })


# ================================================================
# 🔹 Endpoint 2: Porównanie jednego obrazu z wieloma
# ================================================================
@app.route('/compare-multiple', methods=['POST'])
def compare_multiple():
    if 'image1' not in request.files or 'images2' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    img1 = request.files['image1']
    images2 = request.files.getlist('images2')

    path1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
    img1.save(path1)

    image1 = io.imread(path1)
    prnu1 = extract_prnu_from_path(path1)  # ✅ poprawione
    size1 = image1.shape

    similarities = []
    warnings = []
    size_warning = False

    for img in images2:
        path2 = os.path.join(UPLOAD_FOLDER, secure_filename(img.filename))
        img.save(path2)

        image2 = io.imread(path2)
        prnu2 = extract_prnu_from_path(path2)  # ✅ poprawione

        # Sprawdzenie rozdzielczości
        if image2.shape != size1:
            size_warning = True
            warnings.append(
                f"Image '{img.filename}' has a different resolution ({image2.shape}) than the reference ({size1}). "
                "This may affect the reliability of the similarity result. "
                "It is recommended to use reference images with the same resolution."
            )

        # Dopasowanie rozmiarów do wspólnego minimum
        min_h = min(prnu1.shape[0], prnu2.shape[0])
        min_w = min(prnu1.shape[1], prnu2.shape[1])
        prnu1_cropped = prnu1[:min_h, :min_w]
        prnu2_cropped = prnu2[:min_h, :min_w]

        similarity = np.corrcoef(prnu1_cropped.flatten(), prnu2_cropped.flatten())[0, 1]
        similarities.append(float(similarity))

    avg_similarity = float(np.mean(similarities))

    return jsonify({
        'similarity': avg_similarity,
        'all': similarities,
        'size_warning': size_warning,
        'warnings': warnings
    })


# ================================================================
# 🔹 Uruchomienie aplikacji
# ================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
