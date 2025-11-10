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
import traceback
import datetime
import os
from PIL import Image

from .model import FullConvNet
from .utils import im2f, jpeg_qtableinv


def log_error(msg: str):
    """Write error message to log file with timestamp."""
    log_path = "noiseprint_errors.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n\n")


def safe_infer(net, tensor_image):
    """Run model safely, returning None if fails."""
    try:
        with torch.no_grad():
            result = net(tensor_image)[0][0].numpy()
        return result
    except Exception as e:
        log_error(f"Model inference failed: {str(e)}\n{traceback.format_exc()}")
        return None


def getNoiseprint(image_input, weights_dir="pretrained_weights"):
    """
    Generates noiseprint for the given image.
    Handles uniform images and memory errors gracefully.
    Logs all issues to noiseprint_errors.log.
    """
    try:
        img, _ = im2f(image_input, channel=1)
        img = np.asarray(img, dtype=np.float32)

        if img.ndim == 3:
            img = img.mean(axis=2)

        # Normalize to [0, 255]
        if img.max() <= 1.0:
            img *= 255.0

        # Check for uniform image (very low variance)
        if np.std(img) < 1e-3:
            log_error("⚠️ Detected nearly uniform image. Adding small random noise to stabilize.")
            img = img + np.random.normal(0, 0.1, img.shape)

        transform = transforms.ToTensor()

        # Try to read JPEG quality factor
        try:
            QF = jpeg_qtableinv(image_input)
        except Exception:
            QF = 101

        # Load weights (fallback if not found)
        weight_path = f"{weights_dir}/model_qf{int(QF)}.pth"
        if not os.path.exists(weight_path):
            log_error(f"❗ Weights for QF={QF} not found. Falling back to model_qf101.pth.")
            weight_path = f"{weights_dir}/model_qf101.pth"

        # Initialize model
        net = FullConvNet(0.9, False)
        net.load_state_dict(torch.load(weight_path, map_location="cpu"))
        net.eval()

        tensor_image = transform(img).reshape(1, 1, img.shape[0], img.shape[1])
        tensor_image = tensor_image + 1e-8 * torch.randn_like(tensor_image)

        # --- 1st attempt (original size) ---
        result = safe_infer(net, tensor_image)
        if result is not None:
            return img, result

        # --- 2nd attempt: scale down if previous failed ---
        log_error("⚠️ First inference failed. Trying again with scaled-down image due to memory constraints.")
        MAX_SIZE = 2048
        if max(img.shape) > MAX_SIZE:
            scale = MAX_SIZE / max(img.shape)
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img_pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            img_resized = img_pil.resize(new_size, Image.LANCZOS)
            img_resized = np.asarray(img_resized, dtype=np.float32)

            tensor_image = transform(img_resized).reshape(1, 1, img_resized.shape[0], img_resized.shape[1])
            tensor_image = tensor_image + 1e-8 * torch.randn_like(tensor_image)

            result = safe_infer(net, tensor_image)
            if result is not None:
                log_error(f"✅ Successfully generated noiseprint after scaling to {new_size}.")
                return img_resized, result

        log_error("❌ Noiseprint generation failed after both attempts.")
        return None, None

    except Exception as e:
        log_error(f"❗ Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return None, None
