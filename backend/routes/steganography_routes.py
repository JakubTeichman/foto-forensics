# === blueprint: steganography_routes.py ===

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os

# Tworzymy blueprint
steganography_bp = Blueprint('steganography', __name__)

# Folder tymczasowy (jeśli chcesz zapisać plik)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Endpoint: analiza steganograficzna ---
@steganography_bp.route('/stegano/analyze', methods=['POST'])
def analyze_steganography():
    """
    Analizuje przesłany obraz pod kątem ukrytych treści (steganografia).
    W przyszłości można tu dodać wywołanie modelu ML lub algorytmu detekcji.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku w żądaniu'}), 400

    file = request.files.get('file') or request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400


    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # TODO: tu wstaw analizę steganograficzną
    # result = analyze_stego(filepath)
    # Na razie zwracamy przykładową odpowiedź
    result = {
        "status": "success",
        "message": "Analiza steganograficzna zakończona pomyślnie",
        "filename": filename
    }

    return jsonify(result), 200


# --- Endpoint: porównanie steganograficzne (do przyszłej implementacji) ---
@steganography_bp.route('/steganography/compare', methods=['POST'])
def compare_steganography():
    """
    (TODO) Porównuje dwa obrazy pod kątem steganograficznym.
    """
    return jsonify({'message': 'Funkcjonalność w przygotowaniu'}), 501
