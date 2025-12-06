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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from flask import Blueprint, request, jsonify
import joblib
import timm # Wymagane dla CNNEffNetV2
import os
from sklearn.ensemble import GradientBoostingClassifier # Zostawiamy importy klas, na wypadek gdyby były potrzebne przez inne moduły
from sklearn.svm import SVC 

# Utwórz Blueprint, jeśli nie był zdefiniowany w innym miejscu
analyze_bp = Blueprint('analyze', __name__) 

# =================================================================
# ARCHITEKTURA I TRANSFORMATOR (Skopiowane z pliku ewaluacyjnego)
# =================================================================

class ResidualTransform:
    """Konwertuje obraz na jego residuum (szum) i normalizuje."""
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.blur_filter = ImageFilter.GaussianBlur(kernel_size)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Obraz musi być typu PIL.Image")

        # Obliczanie residuum: Obraz - Rozmyty obraz
        img_np = np.array(img, dtype=np.float32)
        blurred_img = img.filter(self.blur_filter)
        blurred_np = np.array(blurred_img, dtype=np.float32)

        residual_np = img_np - blurred_np
        residual_norm = np.clip(residual_np + 128, 0, 255).astype(np.uint8)

        return Image.fromarray(residual_norm).convert('RGB')

class CNNEffNetV2(nn.Module):
    def __init__(self, backbone_name="efficientnetv2_rw_m", pretrained=False, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        # Zwracamy predykcję (out) i cechy (features)
        return out, features

# =================================================================
# 🔧 ŁADOWANIE MODELI ENSEMBLE (JEDNOKROTNIE PRZY STARCIE)
# Zmieniono logikę: ładujemy tylko CNN i SVM.
# =================================================================

DEVICE = torch.device("cpu") # Zgodnie z Twoim plikiem, używamy CPU
IMG_SIZE = (320, 320)
NUM_CLASSES = 2

# Ścieżki do plików
CNN_WEIGHTS_FILE = "best_cnn_nua.pth"
SVM_FILE = "re_trained_svm_nua.pkl"
# META_GB_FILE usunięto

# 1. Ładowanie modelu CNN i wag
try:
    cnn_model = CNNEffNetV2(num_classes=NUM_CLASSES).to(DEVICE)
    cnn_model.load_state_dict(torch.load(CNN_WEIGHTS_FILE, map_location=DEVICE))
    cnn_model.eval()
    print(f"✅ Załadowano model CNN z: {CNN_WEIGHTS_FILE}")
except Exception as e:
    cnn_model = None
    print(f"❌ Błąd ładowania CNN ({CNN_WEIGHTS_FILE}): {e}")

# 2. Ładowanie SVM (Meta-Klasyfikator 1)
try:
    svm_clf = joblib.load(SVM_FILE)
    print(f"✅ Załadowano model SVM z: {SVM_FILE}")
except Exception as e:
    svm_clf = None
    print(f"❌ Błąd ładowania SVM ({SVM_FILE}): {e}")

# Transformacje wejściowe (z Residuals)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    ResidualTransform(kernel_size=3),
    transforms.ToTensor(),
    # Używamy tej samej normalizacji, co w pliku treningowym
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# Próg dla uśrednionej predykcji (można dostosować, np. do 0.5)
BEST_THR = 0.55 

# =================================================================
# 🔍 Endpoint: analiza NUA (UŻYCIE UPROSZCZONEGO ENSEMBLE: CNN + SVM)
# =================================================================

@analyze_bp.route("/nua", methods=["POST"])
def analyze_nua_simplified():
    # Zmieniono warunek sprawdzający załadowanie modeli (tylko CNN i SVM)
    if not all([cnn_model, svm_clf]):
        # Komunikat 503, gdy modele nie są dostępne
        return jsonify({"error": "Błąd: Krytyczne modele CNN lub SVM nie są załadowane poprawnie na serwerze (błąd 503). Sprawdź logi serwera."}), 503

    if "file" not in request.files:
        return jsonify({"error": "Brak pliku"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
        # Krok 1: Transformacja i przygotowanie tensora
        tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return jsonify({"error": f"Błąd przetwarzania obrazu: {e}"}), 400

    # ------------------
    # FAZA PREDYKCJI UPROSZCZONEGO ENSEMBLE
    # ------------------
    with torch.no_grad():
        # A. CNN: Wykonanie forward pass
        cnn_output, cnn_features = cnn_model(tensor)
        
        # B. CNN: Generowanie prawdopodobieństwa
        cnn_prob_nua = torch.softmax(cnn_output, dim=1)[0, 1].item()
        
        # C. CNN: Ekstrakcja cech (input dla SVM)
        cnn_features_np = cnn_features.cpu().numpy()
        
    # D. SVM: Generowanie prawdopodobieństwa
    svm_prob_nua = svm_clf.predict_proba(cnn_features_np)[:, 1][0]
    
    # E. Uśrednianie predykcji (zastępuje model Meta-Ensemble)
    ensemble_prob = (cnn_prob_nua + svm_prob_nua) / 2.0
    
    # ------------------
    # WYNIK I KONFIDENCJA
    # ------------------
    prob = max(0.0, min(1.0, ensemble_prob))
    
    threshold = BEST_THR
    detected = prob > threshold
    
    # Obliczenie konfidencji
    k = 1.5
    if prob < threshold:
        # Konfidencja w brak NUA (klasa 0)
        confidence = (threshold - prob) / threshold * 100.0
    else:
        # Konfidencja w obecność NUA (klasa 1)
        confidence = (prob - threshold) / (1 - threshold) * 100.0

    confidence = round(confidence * k, 2)
    
    # Zmieniono nazwę funkcji endpointu z analyze_nua_ensemble na analyze_nua_simplified, 
    # aby odzwierciedlić uproszczoną logikę. Jeśli używasz tego endpointu pod adresem /nua, 
    # upewnij się, że inne części aplikacji odwołują się do tej nazwy lub pozostaw oryginalną.
    return jsonify({
        "detected": bool(detected),
        "confidence": confidence
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
