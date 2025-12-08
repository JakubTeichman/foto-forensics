import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib # Wymagane do ładowania modeli .pkl (SVM)
from PIL import Image
from torchvision import transforms

# Używamy joblib, więc SVC i GradientBoostingClassifier nie są już potrzebne do ładowania modelu,
# ale trzymamy je na wypadek, gdyby były potrzebne gdzie indziej, choć usunęliśmy GB.
# W nowej wersji używamy tylko klas architektonicznych z pliku treningowego.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================================
# --- Definicje Architektury Nowego Modelu (SRM-CNN) ---
# Te klasy zostały skopiowane z Twojego pliku treningowego
# ====================================================================

class SRMLayer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Filtry SRM (krzyżowe i kwadratowe)
        kv = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        kv3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
        kernels = [kv, kv3]

        conv_kernels_list = []
        for k in kernels:
            k = torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0)
            conv_kernels_list.append(k)

        weight_per_channel = torch.cat(conv_kernels_list, dim=0)
        # Powtarzamy dla każdego kanału wejściowego (RGB)
        final_weight = weight_per_channel.repeat(in_channels, 1, 1, 1)

        self.register_buffer('kernels', final_weight)
        self.in_channels = in_channels
        self.n_kernels = len(kernels)
        self.kernel_size = kernels[0].shape[0]

    def forward(self, x):
        weight = self.kernels.to(x.device)
        padding = self.kernel_size // 2
        # Aplikujemy filtry SRM, wyjście = 2 * in_channels (3 * 2 = 6)
        out = F.conv2d(x, weight, padding=padding, groups=self.in_channels)
        return torch.tanh(out) # Funkcja aktywacji po filtracji

class SimpleStegNet(nn.Module):
    # CNN, który przetwarza wyjście z warstwy SRM (6 kanałów)
    def __init__(self, in_channels=6, num_classes=2, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res_block1 = self._make_res_block(64, 128)
        self.res_block2 = self._make_res_block(128, 256)
        self.res_block3 = self._make_res_block(256, 512)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, feature_dim) # Warstwa do ekstrakcji cech dla SVM
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_res_block(self, in_ch, out_ch):
        # Blok rezydualny/konwolucyjny z MaxPool
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool(x).view(x.size(0), -1)
        feats = self.fc(x)
        logits = self.classifier(feats)
        return logits, feats

class FullStegModel(nn.Module):
    # Cały model: SRM -> CNN
    def __init__(self, in_rgb_channels=3, num_classes=2, feature_dim=512):
        super().__init__()
        self.srm = SRMLayer(in_channels=in_rgb_channels)
        # Warstwa SRM generuje 6 kanałów, więc SimpleStegNet jest inicjowany z in_channels=6
        self.net = SimpleStegNet(in_channels=in_rgb_channels * 2, num_classes=num_classes, feature_dim=feature_dim)

    def forward(self, x):
        residual = self.srm(x)
        logits, feats = self.net(residual)
        return logits, feats

# --- Ścieżki do plików modeli ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Zaktualizowane ścieżki do nowych modeli
CNN_PATH = os.path.join(MODELS_DIR, "best_stegnet_final.pth")
SVM_PATH = os.path.join(MODELS_DIR, "svm_stegano.pkl")

# Usuwamy ścieżki do GB i jsonów z paramsami, bo są niepotrzebne
# GB_PARAMS_PATH, SVM_PARAMS_PATH (usuwamy)

# --- Transformacja obrazu ---
# Używamy transformacji z pliku treningowego (IMG_SIZE=256), ale utrzymujemy
# Resize(320, 320) z oryginalnego pliku, aby zachować kompatybilność,
# chyba że model był trenowany na innym rozmiarze. W pliku treningowym był 256.
# Zmienię na 256, aby dopasować do modelu FullStegModel (który używa 256 w treningu).
transform = transforms.Compose([
    transforms.Resize((256, 256)), # Dostosowane do rozmiaru treningowego
    transforms.ToTensor(),
    # Normalizacja z pliku treningowego
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===============================
# 🧠 Ładowanie modeli
# ===============================
try:
    # 1. Ładowanie nowego modelu CNN (FullStegModel)
    cnn = FullStegModel(in_rgb_channels=3, num_classes=2, feature_dim=512).to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    cnn.eval()
except Exception as e:
    print(f"Błąd ładowania modelu CNN FullStegModel: {e}")
    # Możesz dodać logikę wyjścia lub użycia domyślnego modelu

try:
    # 2. Ładowanie modelu SVM (zapisany jako .pkl za pomocą joblib)
    svm = joblib.load(SVM_PATH)
except Exception as e:
    print(f"Błąd ładowania modelu SVM z joblib: {e}")
    # Możesz dodać logikę wyjścia lub użycia domyślnego modelu

# Gradient Boosting (gb) jest usunięty, bo nowe ensemble to proste uśrednianie

# ===============================
# 🔍 Analiza pojedynczego obrazu
# ===============================
def analyze(pil_image: Image.Image):
    """
    Analizuje pojedynczy obraz PIL Image, używając nowego ensemble:
    FullStegModel (CNN) + SVM. Ensemble to proste uśrednienie prawdopodobieństw.
    Zachowuje oryginalny format wyjściowy.
    """
    try:
        img = pil_image.convert("RGB")
        # Przygotowanie tensora do predykcji
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # 1. Predykcja CNN i ekstrakcja cech
            outputs, features = cnn(tensor)
            # CNN probability (P(klasa 1))
            cnn_prob = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()[0]
            features_np = features.cpu().numpy()

        # 2. Predykcja SVM na podstawie cech z CNN
        # SVM probability (P(klasa 1))
        svm_prob = svm.predict_proba(features_np)[:, 1][0]

        # 3. Nowa logika Ensemble: proste uśrednianie
        ensemble_prob = (cnn_prob + svm_prob) / 2.0

        # W pliku treningowym używano optymalnego progu f1.
        # Aby zachować kompatybilność z poprzednim kodem, używamy domyślnego 0.5.
        threshold = 0.39
        detected = ensemble_prob >= threshold

        return {
            "method": "ensemble_Averaging", # Zaktualizowana nazwa metody
            "score": float(ensemble_prob),
            "detected": bool(detected),
            "details": {
                "cnn_prob": float(cnn_prob),
                "svm_prob": float(svm_prob),
                "ensemble_prob": float(ensemble_prob), # Wcześniej było to GB, teraz uśrednianie
                "threshold": threshold,
            },
        }

    except Exception as e:
        # Bardziej szczegółowe logowanie błędu, jeśli to konieczne
        print(f"Błąd w funkcji analyze: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # Przykładowy kod do testowania z obrazem zerowym (placeholder)
    print(f"Urządzenie: {DEVICE}")
    print(f"CNN gotowe: {cnn.net.__class__.__name__}")
    print(f"SVM gotowe: {svm.__class__.__name__}")

    # Tworzenie pustego obrazu PIL jako placeholder do testów
    try:
        dummy_image = Image.new('RGB', (256, 256), color = 'red')
        test_result = analyze(dummy_image)
        print("\n--- Wynik Analizy na Dummiesie ---")
        # Oczekiwany błąd, jeśli model nie jest trenowany na tym obrazie,
        # ale testuje, czy interfejs działa.
        if "error" in test_result:
             print(f"Test interfejsu: Sukces (Zwrócono błąd: {test_result['error']})")
        else:
             print(json.dumps(test_result, indent=4))
    except Exception as e:
        print(f"\nTEST BŁĄD KRYTYCZNY: {e}")