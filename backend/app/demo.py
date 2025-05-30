from flask import Flask, request, jsonify, Blueprint
import numpy as np
import cv2
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # umożliwia połączenie z Reactem

# Create blueprint - make sure this is at the module level
demo = Blueprint('demo', __name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@demo.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    img1 = request.files['image1']
    img2 = request.files['image2']

    path1 = os.path.join(UPLOAD_FOLDER, secure_filename(img1.filename))
    path2 = os.path.join(UPLOAD_FOLDER, secure_filename(img2.filename))
    img1.save(path1)
    img2.save(path2)

    prnu1 = extract_prnu(path1)
    prnu2 = extract_prnu(path2)

    similarity = np.corrcoef(prnu1.flatten(), prnu2.flatten())[0, 1]

    return jsonify({'similarity': float(similarity)})

def extract_prnu(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    denoised = cv2.medianBlur(image, 5)
    noise = cv2.subtract(image, denoised)
    return noise.astype(np.float32)

app.register_blueprint(demo)

if __name__ == '__main__':
    app.run(debug=True)
