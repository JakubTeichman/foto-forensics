"""
Noiseprint model implementation based on:

D. Cozzolino and L. Verdoliva,
"Noiseprint: A CNN-Based Camera Model Fingerprint",
IEEE Transactions on Information Forensics and Security, vol. 15, pp. 144–159, 2020.
DOI: 10.1109/TIFS.2019.2916364

Original implementation (Matlab/TensorFlow) © 2019 GRIP-UNINA.
Adapted for research and educational use in Python/PyTorch by Jakub Teichman, 2025.
"""

import torch
import numpy as np
import torchvision.transforms as transforms
from .model import FullConvNet
from .utils import im2f, jpeg_qtableinv


def getNoiseprint(image_input, weights_dir="pretrained_weights"):
    """
    Główna funkcja: generuje mapę noiseprint danego obrazu.
    Przyjmuje bezpośrednio obraz (PIL, ndarray, bytes lub Flask file),
    a nie ścieżkę do pliku.
    """
    img, _ = im2f(image_input, channel=1)
    transform = transforms.ToTensor()

    try:
        QF = jpeg_qtableinv(image_input)
    except Exception:
        QF = 101

    # Inicjalizacja modelu
    net = FullConvNet(0.9, False)
    weight_path = f"{weights_dir}/model_qf{int(QF)}.pth"
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()

    # Przetwarzanie
    with torch.no_grad():
        tensor_image = transform(img).reshape(1, 1, img.shape[0], img.shape[1])
        result = net(tensor_image)[0][0].numpy()

    return img, result
