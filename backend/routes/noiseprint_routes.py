from flask import Blueprint, request, jsonify
import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
from noiseprint.model import FullConvNet as NoiseprintModel
noiseprint_bp = Blueprint('noiseprint', __name__)

model = NoiseprintModel()
model.load_state_dict(torch.load('/app/noiseprint/weights/model_noiseprint.pth', map_location='cpu'))
model.eval()

def image_to_tensor(image):
    # Jeśli obraz jest typu PIL.Image, konwertujemy go do skali szarości
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image)).convert('L')  # <---- tu zmiana: 'L' = grayscale
    else:
        image = image.convert('L')

    img = np.array(image, dtype=np.float32)
    img = img / 255.0
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [B, C, H, W] = [1, 1, H, W]
    return img_tensor



def compute_noise_stats(noise):
    return {
        "mean": float(np.mean(noise)),
        "std": float(np.std(noise)),
        "energy": float(np.sum(noise ** 2)),
        "entropy": float(-np.sum(noise * np.log(np.abs(noise) + 1e-10))),
    }

def generate_noiseprint(image_tensor):
    with torch.no_grad():
        return model(image_tensor).squeeze().cpu().numpy()

@noiseprint_bp.route('/generate', methods=['POST'])
def generate():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_tensor = image_to_tensor(img)
    noise = generate_noiseprint(img_tensor)
    stats = compute_noise_stats(noise)
    _, buffer = cv2.imencode('.png', (noise * 255).astype(np.uint8))
    encoded = base64.b64encode(buffer).decode('utf-8')
    return jsonify({
        'noiseprint': encoded,
        'stats': stats
    })

@noiseprint_bp.route('/compare', methods=['POST'])
def compare():
    evidence_file = request.files['evidence']
    refs = request.files.getlist('references')

    evidence_tensor = image_to_tensor(Image.open(evidence_file.stream))
    evidence_np = generate_noiseprint(evidence_tensor)

    ref_noiseprints = []
    for r in refs:
        img = Image.open(r.stream)
        ref_tensor = image_to_tensor(img)
        ref_noiseprints.append(generate_noiseprint(ref_tensor))

    mean_ref = np.mean(ref_noiseprints, axis=0)

    corr = cv2.matchTemplate(evidence_np, mean_ref, cv2.TM_CCOEFF_NORMED)
    peak_corr = float(np.max(corr))

    _, buf_evidence = cv2.imencode('.png', (evidence_np * 255).astype(np.uint8))
    _, buf_mean_ref = cv2.imencode('.png', (mean_ref * 255).astype(np.uint8))

    return jsonify({
        'evidence_noiseprint': base64.b64encode(buf_evidence).decode('utf-8'),
        'mean_reference_noiseprint': base64.b64encode(buf_mean_ref).decode('utf-8'),
        'peak_correlation': peak_corr
    })
