# ======================================
# analyze_routes.py
# ======================================
from flask import Blueprint, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import exifread
import io

# Tworzymy blueprint Flask
analyze_bp = Blueprint("analyze", __name__)

# ==============================
# 🧠 Definicja modelu (CNNEffNetV2)
# ==============================
class CNNEffNetV2(nn.Module):
    def __init__(self, backbone_name="efficientnetv2_rw_m", pretrained=False, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out

# ==============================
# 🔧 Ładowanie modelu
# ==============================
DEVICE = torch.device("cpu")

model = CNNEffNetV2()
model.load_state_dict(torch.load("best_cnn_v3.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==============================
# 🧩 Transformacje wejściowe
# ==============================
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# ==============================
# 🔍 Endpoint: analiza NUA
# ==============================
@analyze_bp.route("/nua/", methods=["POST"])
def analyze_nua():
    if "file" not in request.files:
        return jsonify({"error": "Brak pliku"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()  # prawdopodobieństwo klasy 1 (np. 'nua_paired')
        detected = prob > 0.5

    return jsonify({
        "detected": bool(detected),
        "confidence": round(prob, 4)
    })

# ==============================
# 🧾 Endpoint: analiza metadanych
# ==============================
@analyze_bp.route("/metadata", methods=["POST"])
def analyze_metadata():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = file.read()

    # Otwieranie obrazu
    image = Image.open(io.BytesIO(file_bytes))

    # Odczyt EXIF
    file.seek(0)
    exif_tags = exifread.process_file(io.BytesIO(file_bytes), details=False)

    metadata = {
        "File Name": file.filename,
        "Format": image.format or "N/A",
        "Mode": image.mode,
        "Resolution": f"{image.width} x {image.height}",
        "File Size": f"{round(len(file_bytes) / 1024, 2)} KB",
        "EXIF Data": {tag: str(value) for tag, value in exif_tags.items()},
    }

    gps = {k: str(v) for k, v in exif_tags.items() if "GPS" in k}
    metadata["GPS Info"] = gps if gps else "No GPS metadata found"

    return jsonify(metadata)
