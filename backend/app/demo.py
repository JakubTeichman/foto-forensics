import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import numpy as np

# Dodaj katalog nadrzędny do sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importuj funkcje z zewnętrznego pliku
from .prnu_utils.prnu_extraction import extract_prnu_from_bytes  # Użyjesz tej funkcji zamiast extract_prnu

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/compare', methods=['POST'])  # zmieniono na /app/...
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    img1 = request.files['image1']
    img2 = request.files['image2']

    path1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
    path2 = os.path.join(UPLOAD_FOLDER, secure_filename(img2.filename))
    img1.save(path1)
    img2.save(path2)

    # Przeczytaj obrazy jako numpy array
    from skimage import io

    image1 = io.imread(path1)
    image2 = io.imread(path2)

    prnu1 = extract_prnu_from_bytes(image1)
    prnu2 = extract_prnu_from_bytes(image2)

    similarity = np.corrcoef(prnu1.flatten(), prnu2.flatten())[0, 1]

    return jsonify({'similarity': float(similarity)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
