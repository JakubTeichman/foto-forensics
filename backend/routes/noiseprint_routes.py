from flask import Blueprint, request, jsonify
import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
from noiseprint.model import FullConvNet as NoiseprintModel
from numpy.linalg import norm

noiseprint_bp = Blueprint('noiseprint', __name__)

# ---- 🔧 Inicjalizacja modelu ----
model = NoiseprintModel()
model.load_state_dict(torch.load('/app/noiseprint/weights/model_noiseprint.pth', map_location='cpu'))
model.eval()


# ---- 🧩 Pomocnicze funkcje ----
def preprocess_image(image):
    """Przygotowuje obraz do generowania Noiseprintu (wyrównanie + filtracja)."""
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image)).convert('L')
    else:
        image = image.convert('L')

    img = np.array(image, dtype=np.float32)
    # 🔹 filtr high-pass (redukuje treść, wzmacnia szum)
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0)
    img = img / 255.0
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img_tensor


def compute_noise_stats(noise):
    """Oblicza statystyki Noiseprintu."""
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mean": float(np.mean(noise)),
        "std": float(np.std(noise)),
        "energy": float(np.sum(noise ** 2)),
        "entropy": float(-np.sum(noise * np.log(np.abs(noise) + 1e-10))),
    }


def generate_noiseprint(image_tensor):
    """Generuje Noiseprint z obrazu."""
    with torch.no_grad():
        noise = model(image_tensor).squeeze().cpu().numpy()
    return noise


def local_normalize(arr):
    """Lokalna normalizacja z zabezpieczeniem przed NaN i zerowym rozrzutem."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(arr)
    if std < 1e-8:
        return arr
    return (arr - np.mean(arr)) / std


def cross_correlation(a, b):
    """Oblicza współczynnik korelacji krzyżowej (NCC)."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def diff_energy(a, b):
    """Energia różnicowa między noiseprintami."""
    diff = np.abs(a - b)
    return float(np.sum(diff ** 2))


def cosine_similarity(a, b):
    """Oblicza podobieństwo kosinusowe między wektorami embeddingów."""
    a = a.flatten()
    b = b.flatten()
    denom = (norm(a) * norm(b))
    if denom == 0:
        return 0.0
    cos_sim = np.dot(a, b) / denom
    return float((cos_sim + 1) / 2)  # skala [0, 1]


# ---- 🧠 ROUTE: generowanie Noiseprint ----
@noiseprint_bp.route('/generate', methods=['POST'])
def generate():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img_tensor = preprocess_image(img)
        noise = generate_noiseprint(img_tensor)

        stats = compute_noise_stats(noise)
        _, buffer = cv2.imencode('.png', (local_normalize(noise) * 127 + 128).astype(np.uint8))
        encoded = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'noiseprint': encoded,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': f'Noiseprint generation failed: {str(e)}'}), 500


# ---- ⚖️ ROUTE: porównanie Noiseprintów (klasyczne NCC + energia różnicowa) ----
@noiseprint_bp.route('/compare', methods=['POST'])
def compare():
    try:
        evidence_file = request.files['evidence']
        refs = request.files.getlist('references')

        # --- 1️⃣ Noiseprint próbki dowodowej
        evidence_tensor = preprocess_image(Image.open(evidence_file.stream))
        evidence_np = generate_noiseprint(evidence_tensor)
        evidence_np = local_normalize(evidence_np)

        # --- 2️⃣ Porównanie z każdą referencją
        correlations = []
        diff_energies = []
        for ref_file in refs:
            ref_tensor = preprocess_image(Image.open(ref_file.stream))
            ref_np = generate_noiseprint(ref_tensor)
            ref_np = local_normalize(ref_np)

            # dopasowanie rozmiaru
            if ref_np.shape != evidence_np.shape:
                ref_np = cv2.resize(ref_np, (evidence_np.shape[1], evidence_np.shape[0]))

            corr = cross_correlation(evidence_np, ref_np)
            diff_en = diff_energy(evidence_np, ref_np)

            correlations.append(corr)
            diff_energies.append(diff_en)

        # --- 3️⃣ Statystyki wyników
        mean_corr = float(np.mean(correlations))
        std_corr = float(np.std(correlations))
        mean_diff_en = float(np.mean(diff_energies))

        # --- 4️⃣ Obrazy (dla wizualizacji)
        _, buf_evidence = cv2.imencode('.png', (local_normalize(evidence_np) * 127 + 128).astype(np.uint8))

        return jsonify({
            'evidence_noiseprint': base64.b64encode(buf_evidence).decode('utf-8'),
            'pairwise_correlations': correlations,
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'mean_diff_energy': mean_diff_en,
            'stats_evidence': compute_noise_stats(evidence_np)
        })

    except Exception as e:
        return jsonify({'error': f'Noiseprint comparison failed: {str(e)}'}), 500


# ---- 🧬 ROUTE: porównanie Noiseprintów (embedding similarity / cosine) ----
@noiseprint_bp.route('/compare_embedding', methods=['POST'])
def compare_embedding():
    """
    Porównuje dwa Noiseprinty (dowodowy i referencyjny) metodą embedding similarity.
    Zwraca wynik podobieństwa kosinusowego (0–1).
    """
    try:
        evidence_file = request.files['evidence']
        reference_file = request.files['reference']

        # --- Generowanie Noiseprintów ---
        evidence_tensor = preprocess_image(Image.open(evidence_file.stream))
        reference_tensor = preprocess_image(Image.open(reference_file.stream))

        evidence_np = local_normalize(generate_noiseprint(evidence_tensor))
        reference_np = local_normalize(generate_noiseprint(reference_tensor))

        # Dopasowanie rozmiarów
        if evidence_np.shape != reference_np.shape:
            reference_np = cv2.resize(reference_np, (evidence_np.shape[1], evidence_np.shape[0]))

        # --- Obliczenie embedding similarity ---
        similarity = cosine_similarity(evidence_np, reference_np)

        # --- Przygotowanie wizualizacji ---
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
        return jsonify({'error': f'Embedding comparison failed: {str(e)}'}), 500
