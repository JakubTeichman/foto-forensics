from flask import Blueprint, request, jsonify
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import os
import torch
from extensions import db
from flask_sqlalchemy import SQLAlchemy
from noiseprint.model import FullConvNet as NoiseprintModel

add_reference_bp = Blueprint("add_reference", __name__)

# ==========================================
# 🧱 Model bazy danych
# ==========================================
# 🧱 Model bazy danych
class DeviceReference(db.Model):
    __tablename__ = "device_references"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    manufacturer = db.Column(db.String(100), nullable=False)
    model = db.Column(db.String(100), nullable=False)
    # LONGBLOB zamiast domyślnego BLOB (~64KB)
    noiseprint = db.Column(db.LargeBinary(length=(4 * 1024 * 1024 * 1024)), nullable=False)
    num_images = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.TIMESTAMP, server_default=db.text("CURRENT_TIMESTAMP"))


# ==========================================
# 🔧 Pomocnicze funkcje serializacji
# ==========================================
def serialize_noiseprint(np_array: np.ndarray) -> bytes:
    mem = BytesIO()
    np.savez_compressed(mem, noise=np_array)
    mem.seek(0)
    return mem.read()


def deserialize_noiseprint(blob: bytes) -> np.ndarray:
    mem = BytesIO(blob)
    loaded = np.load(mem)
    return loaded["noise"]


# ==========================================
# 🔊 Noiseprint model
# ==========================================
MODEL_PATH = os.getenv("NOISEPRINT_MODEL_PATH", "/app/noiseprint/weights/model_noiseprint.pth")

noise_model = NoiseprintModel()
if os.path.exists(MODEL_PATH):
    noise_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
else:
    print(f"⚠️ Warning: model weights not found at {MODEL_PATH}")
noise_model.eval()


def pil_to_tensor(pil_img: Image.Image):
    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor


def generate_noiseprint_from_pil(pil_img: Image.Image):
    t = pil_to_tensor(pil_img)
    with torch.no_grad():
        out = noise_model(t).squeeze().cpu().numpy()
    return out


# ==========================================
# 🧩 Endpoint: dodaj referencję
# ==========================================
@add_reference_bp.route("/add_reference", methods=["POST"])
def add_reference():
    manufacturer = request.form.get("manufacturer")
    model_name = request.form.get("model")

    if not manufacturer or not model_name:
        return jsonify({"error": "manufacturer and model are required"}), 400

    files = request.files.getlist("images")
    if not files or len(files) < 3:
        return jsonify({"error": "At least 3 images are required as reference"}), 400

    # Generowanie noiseprintów
    noiseprints = []
    for f in files:
        try:
            pil = Image.open(f.stream)
            np_noise = generate_noiseprint_from_pil(pil)
            noiseprints.append(np_noise)
        except Exception as e:
            return jsonify({"error": f"Failed to process one image: {str(e)}"}), 400

    # Dopasowanie rozmiarów (jeśli różne)
    shapes = [n.shape for n in noiseprints]
    if len(set(shapes)) != 1:
        target_shape = shapes[0]
        noiseprints = [
            cv2.resize(n, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            for n in noiseprints
        ]

    # Uśrednienie noiseprintów
    mean_np = np.mean(noiseprints, axis=0).astype(np.float32)
    blob = serialize_noiseprint(mean_np)

    # Zapis do bazy
    try:
        ref = DeviceReference(
            manufacturer=manufacturer,
            model=model_name,
            noiseprint=blob,
            num_images=len(files),
        )
        db.session.add(ref)
        db.session.commit()
        return jsonify({"message": "Reference added", "id": ref.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ==========================================
# 🧠 Endpoint: porównanie z bazą
# ==========================================
@add_reference_bp.route("/compare_with_db", methods=["POST"])
def compare_with_db():
    top_n = int(request.form.get("top_n", 5))
    evidence_file = request.files.get("evidence")

    if evidence_file is None:
        return jsonify({"error": "evidence file required"}), 400

    try:
        pil_e = Image.open(evidence_file.stream)
        evidence_np = generate_noiseprint_from_pil(pil_e)
    except Exception as e:
        return jsonify({"error": f"Failed to process evidence: {str(e)}"}), 400

    try:
        rows = db.session.query(DeviceReference).all()
        results = []

        for row in rows:
            ref_np = deserialize_noiseprint(row.noiseprint)
            if ref_np.shape != evidence_np.shape:
                ref_np = cv2.resize(ref_np, (evidence_np.shape[1], evidence_np.shape[0]), interpolation=cv2.INTER_LINEAR)

            a, b = evidence_np.flatten(), ref_np.flatten()
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-10 and np.std(b) > 1e-10 else 0.0

            results.append({
                "id": row.id,
                "manufacturer": row.manufacturer,
                "model": row.model,
                "num_images": row.num_images,
                "correlation": corr,
            })

        results_sorted = sorted(results, key=lambda x: x["correlation"], reverse=True)[:top_n]
        return jsonify({"results": results_sorted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
