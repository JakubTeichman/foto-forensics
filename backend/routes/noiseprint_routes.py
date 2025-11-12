from flask import Blueprint, request, jsonify
import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import traceback
import datetime
import sys
from numpy.linalg import norm

from noiseprint.model import FullConvNet as NoiseprintModel

noiseprint_bp = Blueprint("noiseprint", __name__)

# ---- 🧠 Load model ----
model = NoiseprintModel()
try:
    model.load_state_dict(torch.load("/app/noiseprint/weights/model_noiseprint.pth", map_location="cpu"))
    model.eval()
except Exception as e:
    print(f"[ERROR] Failed to load noiseprint model: {str(e)}", file=sys.stderr)


# ---- 🧩 Logging ----
def log_error(msg: str):
    log_path = "noiseprint_errors.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n{'-'*80}\n\n")
    print(msg, file=sys.stderr)


# ---- 🧩 Image preprocessing ----
def preprocess_image(image):
    """
    Convert image to grayscale tensor.
    Automatically downscale if width or height > 3000 px.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image))
    image = image.convert("L")

    max_dim = max(image.size)
    MAX_SAFE_SIZE = 3000
    if max_dim > MAX_SAFE_SIZE:
        scale = MAX_SAFE_SIZE / max_dim
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        orig_size = image.size
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        log_error(f"🪶 Image automatically downscaled from {orig_size} to {new_size} for stability")

    img = np.array(image, dtype=np.float32)
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0)
    img = img / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)


# ---- 🧩 Noiseprint generation ----
def generate_noiseprint(img_tensor):
    """Generate noiseprint safely."""
    with torch.no_grad():
        noise = model(img_tensor).squeeze().cpu().numpy()
    return noise


# ---- 🧮 Helpers ----
def local_normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(arr)
    if std < 1e-8:
        return arr
    return (arr - np.mean(arr)) / std


def cross_correlation(a, b):
    """Normalized cross-correlation (NCC)."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def diff_energy(a, b):
    """Difference energy between two noiseprints."""
    diff = np.abs(a - b)
    return float(np.sum(diff ** 2))


def cosine_similarity(a, b):
    """Cosine similarity between embeddings."""
    a = a.flatten()
    b = b.flatten()
    denom = (norm(a) * norm(b))
    if denom == 0:
        return 0.0
    cos_sim = np.dot(a, b) / denom
    return float((cos_sim + 1) / 2)  # scale [0,1]


def compute_noise_stats(noise):
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mean": float(np.mean(noise)),
        "std": float(np.std(noise)),
        "energy": float(np.sum(noise ** 2)),
        "entropy": float(-np.sum(noise * np.log(np.abs(noise) + 1e-10))),
    }


# ---- ⚙️ ROUTE: Noiseprint generation ----
@noiseprint_bp.route("/generate", methods=["POST"])
def generate():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided."}), 400

        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            log_error(f"❌ Failed to open image:\n{traceback.format_exc()}")
            return jsonify({"error": "Invalid or corrupted image file."}), 400

        try:
            img_tensor = preprocess_image(img)
            noise = generate_noiseprint(img_tensor)
        except MemoryError:
            log_error("❌ System ran out of memory (MemoryError)")
            return jsonify({"error": "Image too large for processing. Try smaller file."}), 500
        except Exception:
            log_error(f"❌ Exception in noiseprint generation:\n{traceback.format_exc()}")
            return jsonify({"error": "Noiseprint generation failed."}), 500

        stats = compute_noise_stats(noise)
        normalized = local_normalize(noise)
        _, buffer = cv2.imencode(".png", (normalized * 127 + 128).astype(np.uint8))
        encoded = base64.b64encode(buffer).decode("utf-8")

        response = {"noiseprint": encoded, "stats": stats}

        if max(img.size) > 3000:
            response["info"] = "⚠️ Image was automatically downscaled for processing stability."

        return jsonify(response)

    except Exception:
        log_error(f"❗ Uncaught top-level error:\n{traceback.format_exc()}")
        return jsonify({"error": "Unexpected backend error."}), 500


# ---- ⚖️ ROUTE: Compare multiple noiseprints ----
@noiseprint_bp.route('/compare', methods=['POST'])
def compare():
    try:
        evidence_file = request.files['evidence']
        refs = request.files.getlist('references')

        # 1️⃣ Noiseprint próbki dowodowej
        evidence_tensor = preprocess_image(Image.open(evidence_file.stream))
        evidence_np = local_normalize(generate_noiseprint(evidence_tensor))

        correlations, diff_energies = [], []

        # 2️⃣ Porównanie z każdą referencją
        for ref_file in refs:
            ref_tensor = preprocess_image(Image.open(ref_file.stream))
            ref_np = local_normalize(generate_noiseprint(ref_tensor))

            # dopasowanie rozmiaru
            if ref_np.shape != evidence_np.shape:
                ref_np = cv2.resize(ref_np, (evidence_np.shape[1], evidence_np.shape[0]))

            corr = cross_correlation(evidence_np, ref_np)
            diff_en = diff_energy(evidence_np, ref_np)

            correlations.append(corr)
            diff_energies.append(diff_en)

        # 3️⃣ Statystyki wyników
        mean_corr = float(np.mean(correlations))
        std_corr = float(np.std(correlations))
        mean_diff_en = float(np.mean(diff_energies))

        _, buf_evidence = cv2.imencode('.png', (evidence_np * 127 + 128).astype(np.uint8))

        return jsonify({
            'evidence_noiseprint': base64.b64encode(buf_evidence).decode('utf-8'),
            'pairwise_correlations': correlations,
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'mean_diff_energy': mean_diff_en,
            'stats_evidence': compute_noise_stats(evidence_np)
        })

    except Exception as e:
        log_error(f"❌ Noiseprint comparison failed:\n{traceback.format_exc()}")
        return jsonify({'error': f'Noiseprint comparison failed: {str(e)}'}), 500


# ---- 🧬 ROUTE: Embedding cosine similarity ----
@noiseprint_bp.route('/compare_embedding', methods=['POST'])
def compare_embedding():
    """
    Porównuje dwa Noiseprinty (dowodowy i referencyjny) metodą embedding similarity.
    Zwraca wynik podobieństwa kosinusowego (0–1).
    """
    try:
        evidence_file = request.files['evidence']
        reference_file = request.files['reference']

        evidence_tensor = preprocess_image(Image.open(evidence_file.stream))
        reference_tensor = preprocess_image(Image.open(reference_file.stream))

        evidence_np = local_normalize(generate_noiseprint(evidence_tensor))
        reference_np = local_normalize(generate_noiseprint(reference_tensor))

        if evidence_np.shape != reference_np.shape:
            reference_np = cv2.resize(reference_np, (evidence_np.shape[1], evidence_np.shape[0]))

        similarity = cosine_similarity(evidence_np, reference_np)

        _, buf_evidence = cv2.imencode('.png', (evidence_np * 127 + 128).astype(np.uint8))
        _, buf_reference = cv2.imencode('.png', (reference_np * 127 + 128).astype(np.uint8))

        return jsonify({
            "method": "embedding_cosine_similarity",
            "similarity_score": similarity,
            "evidence_noiseprint": base64.b64encode(buf_evidence).decode('utf-8'),
            "reference_noiseprint": base64.b64encode(buf_reference).decode('utf-8'),
            "stats_evidence": compute_noise_stats(evidence_np),
            "stats_reference": compute_noise_stats(reference_np)
        })

    except Exception as e:
        log_error(f"❌ Embedding comparison failed:\n{traceback.format_exc()}")
        return jsonify({'error': f'Embedding comparison failed: {str(e)}'}), 500
    
# ---- 🧬 ROUTE: Compare with reference noiseprint database (SQL) ----
@noiseprint_bp.route('/compare_with_db', methods=['POST'])
def compare_with_db():
    """
    Porównuje noiseprint obrazu dowodowego z bazą zapisanych noiseprintów referencyjnych (SQLAlchemy).
    Wymaga endpointu add_reference z modelem DeviceReference.
    """
    try:
        from extensions import db
        from routes.add_reference import DeviceReference, deserialize_noiseprint

        evidence_file = request.files.get('evidence')
        if not evidence_file:
            return jsonify({'error': 'No evidence image provided.'}), 400

        # === 1️⃣ Wczytanie i przetworzenie próbki dowodowej ===
        evidence_tensor = preprocess_image(Image.open(evidence_file.stream))
        evidence_np = local_normalize(generate_noiseprint(evidence_tensor))

        # === 2️⃣ Wczytanie wszystkich referencji z bazy ===
        refs = db.session.query(DeviceReference).all()
        if not refs:
            return jsonify({'error': 'No reference entries found in database.'}), 404

        results = []
        for ref in refs:
            try:
                ref_np = deserialize_noiseprint(ref.noiseprint)
                ref_np = local_normalize(ref_np)

                if ref_np.shape != evidence_np.shape:
                    ref_np = cv2.resize(ref_np, (evidence_np.shape[1], evidence_np.shape[0]))

                sim = cosine_similarity(evidence_np, ref_np)
                results.append({
                    "id": ref.id,
                    "manufacturer": ref.manufacturer,
                    "model": ref.model,
                    "num_images": ref.num_images,
                    "similarity": round(sim, 4)
                })
            except Exception:
                log_error(f"⚠️ Failed to compare with reference ID={ref.id}\n{traceback.format_exc()}")

        # === 3️⃣ Sortowanie i zwrot wyników ===
        results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
        best_match = results_sorted[0] if results_sorted else None

        response = {
            "total_references": len(refs),
            "matches": results_sorted,
            "best_match": best_match,
        }
        return jsonify(response)

    except Exception:
        log_error(f"❌ Compare_with_db failed:\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal error during database comparison.'}), 500

