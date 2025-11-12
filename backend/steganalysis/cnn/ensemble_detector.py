import os
import json
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model CNN ---
class CNNEffNetV2(torch.nn.Module):
    def __init__(self, backbone_name="efficientnetv2_rw_m", num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        self.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out, f


# --- Ścieżki ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CNN_PATH = os.path.join(MODELS_DIR, "best_stegano_cnn.pth")
SVM_PARAMS_PATH = os.path.join(MODELS_DIR, "svm_params.json")
GB_PARAMS_PATH = os.path.join(MODELS_DIR, "gb_params.json")

# --- Transformacja obrazu ---
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# --- Funkcja do odtwarzania modeli klasycznych ---
def load_sklearn_model(cls, json_path):
    with open(json_path, "r") as f:
        params = json.load(f)
    model = cls(**params)
    return model


# ===============================
# 🧠 Ładowanie wszystkich modeli
# ===============================
cnn = CNNEffNetV2().to(DEVICE)
cnn.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
cnn.eval()

svm = load_sklearn_model(SVC, SVM_PARAMS_PATH)
gb = load_sklearn_model(GradientBoostingClassifier, GB_PARAMS_PATH)

# ===============================
# 🔍 Analiza pojedynczego obrazu
# ===============================
def analyze(pil_image: Image.Image):
    try:
        img = pil_image.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs, features = cnn(tensor)
            cnn_prob = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()[0]
            features_np = features.cpu().numpy()

        svm_prob = svm.predict_proba(features_np)[:, 1][0]
        X_meta = np.array([[cnn_prob, svm_prob]])
        ensemble_prob = gb.predict_proba(X_meta)[:, 1][0]

        detected = ensemble_prob >= 0.5

        return {
            "method": "ensemble_C1",
            "score": float(ensemble_prob),
            "detected": bool(detected),
            "details": {
                "cnn_prob": float(cnn_prob),
                "svm_prob": float(svm_prob),
                "ensemble_prob": float(ensemble_prob),
                "threshold": 0.5,
            },
        }

    except Exception as e:
        return {"error": str(e)}
