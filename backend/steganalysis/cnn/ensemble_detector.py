import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib 
from PIL import Image
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SRMLayer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        kv = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        kv3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
        kernels = [kv, kv3]

        conv_kernels_list = []
        for k in kernels:
            k = torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0)
            conv_kernels_list.append(k)

        weight_per_channel = torch.cat(conv_kernels_list, dim=0)
        final_weight = weight_per_channel.repeat(in_channels, 1, 1, 1)

        self.register_buffer('kernels', final_weight)
        self.in_channels = in_channels
        self.n_kernels = len(kernels)
        self.kernel_size = kernels[0].shape[0]

    def forward(self, x):
        weight = self.kernels.to(x.device)
        padding = self.kernel_size // 2
        out = F.conv2d(x, weight, padding=padding, groups=self.in_channels)
        return torch.tanh(out) 

class SimpleStegNet(nn.Module):
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
        self.fc = nn.Linear(512, feature_dim) 
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_res_block(self, in_ch, out_ch):
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
    def __init__(self, in_rgb_channels=3, num_classes=2, feature_dim=512):
        super().__init__()
        self.srm = SRMLayer(in_channels=in_rgb_channels)
        self.net = SimpleStegNet(in_channels=in_rgb_channels * 2, num_classes=num_classes, feature_dim=feature_dim)

    def forward(self, x):
        residual = self.srm(x)
        logits, feats = self.net(residual)
        return logits, feats

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

CNN_PATH = os.path.join(MODELS_DIR, "best_stegnet_final.pth")
SVM_PATH = os.path.join(MODELS_DIR, "svm_stegano.pkl")


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

try:
    cnn = FullStegModel(in_rgb_channels=3, num_classes=2, feature_dim=512).to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    cnn.eval()
except Exception as e:
    print(f"Błąd ładowania modelu CNN FullStegModel: {e}")

try:
    svm = joblib.load(SVM_PATH)
except Exception as e:
    print(f"Błąd ładowania modelu SVM z joblib: {e}")
def analyze(pil_image: Image.Image):
    """
    Analizuje pojedynczy obraz PIL Image, używając nowego ensemble:
    FullStegModel (CNN) + SVM. Ensemble to proste uśrednienie prawdopodobieństw.
    Zachowuje oryginalny format wyjściowy.
    """
    try:
        img = pil_image.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs, features = cnn(tensor)
            cnn_prob = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()[0]
            features_np = features.cpu().numpy()

        svm_prob = svm.predict_proba(features_np)[:, 1][0]

        ensemble_prob = (cnn_prob + svm_prob) / 2.0

        threshold = 0.39
        detected = ensemble_prob >= threshold

        return {
            "method": "ensemble_Averaging",
            "score": float(ensemble_prob),
            "detected": bool(detected),
            "details": {
                "cnn_prob": float(cnn_prob),
                "svm_prob": float(svm_prob),
                "ensemble_prob": float(ensemble_prob), 
                "threshold": threshold,
            },
        }

    except Exception as e:
        print(f"Błąd w funkcji analyze: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    print(f"Urządzenie: {DEVICE}")
    print(f"CNN gotowe: {cnn.net.__class__.__name__}")
    print(f"SVM gotowe: {svm.__class__.__name__}")

    try:
        dummy_image = Image.new('RGB', (256, 256), color = 'red')
        test_result = analyze(dummy_image)
        print("\n--- Wynik Analizy na Dummiesie ---")
        if "error" in test_result:
             print(f"Test interfejsu: Sukces (Zwrócono błąd: {test_result['error']})")
        else:
             print(json.dumps(test_result, indent=4))
    except Exception as e:
        print(f"\nTEST BŁĄD KRYTYCZNY: {e}")