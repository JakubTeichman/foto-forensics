# analyze_routes.py
from flask import Blueprint, request, jsonify
from PIL import Image
import exifread
import io

analyze_bp = Blueprint("analyze", __name__)

@analyze_bp.route("/metadata", methods=["POST"])
def analyze_metadata():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = file.read()

    # Otwieranie obrazu z pamięci
    image = Image.open(io.BytesIO(file_bytes))

    # Odczyt EXIF (jeśli istnieje)
    file.seek(0)
    exif_tags = exifread.process_file(io.BytesIO(file_bytes), details=False)

    # Tworzenie struktury danych
    metadata = {
        "File Name": file.filename,
        "Format": image.format or "N/A",
        "Mode": image.mode,
        "Resolution": f"{image.width} x {image.height}",
        "File Size": f"{round(len(file_bytes) / 1024, 2)} KB",
        "EXIF Data": {tag: str(value) for tag, value in exif_tags.items()},
    }

    # Dodanie podstawowych pól EXIF, jeśli występują
    gps = {k: str(v) for k, v in exif_tags.items() if "GPS" in k}
    if gps:
        metadata["GPS Info"] = gps
    else:
        metadata["GPS Info"] = "No GPS metadata found"

    return jsonify(metadata)
