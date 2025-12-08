# preprocessing.py
# Helpers: load PIL image (file-like or path), produce full + patch grayscale tensors,
# apply SRM bank conv and return tensors ready for model inference.

from PIL import Image
import io
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from typing import Tuple
# Importujemy teraz funkcję z siamese_model.py, która zwraca kanoniczny bank SRM
from stegano_compare.siamese_model import get_srm_processor 

IMAGE_SIZE = 256
PATCH_SIZE = 128

_transform_full = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor()
])

def load_pil_from_file_storage(file_obj) -> Image.Image:
    # file_obj can be Flask FileStorage (has .read or .stream) or a file path (str)
    if hasattr(file_obj, "read"):
        file_obj.seek(0)
        data = file_obj.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    elif isinstance(file_obj, str):
        return Image.open(file_obj).convert("RGB")
    else:
        raise ValueError("Unsupported file object for image loading.")

def random_center_patch(pil_img: Image.Image, size=PATCH_SIZE):
    # deterministic-ish center crop fallback for inference: take center crop
    w, h = pil_img.size
    if w < size or h < size:
        return pil_img.resize((size, size))
    cx, cy = w // 2, h // 2
    half = size // 2
    left = max(cx - half, 0)
    upper = max(cy - half, 0)
    return pil_img.crop((left, upper, left + size, upper + size))

def prepare_pair_tensors(original_file, suspicious_file, device='cpu', use_srm=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (a_full_srm, a_patch_srm, b_full_srm, b_patch_srm) each tensor shape (C, H, W) where
    C = SRM filters count (30). Ready to stack into batch dim with unsqueeze(0).
    """
    a_img = load_pil_from_file_storage(original_file)
    b_img = load_pil_from_file_storage(suspicious_file)

    # full + patch
    a_full = _transform_full(a_img)             # 1xHxW
    b_full = _transform_full(b_img)
    a_patch = T.Compose([T.Lambda(lambda img: random_center_patch(img, PATCH_SIZE)),
                         T.Resize((PATCH_SIZE, PATCH_SIZE)),
                         T.Grayscale(num_output_channels=1),
                         T.ToTensor()])(a_img)              # 1xpxp
    b_patch = T.Compose([T.Lambda(lambda img: random_center_patch(img, PATCH_SIZE)),
                         T.Resize((PATCH_SIZE, PATCH_SIZE)),
                         T.Grayscale(num_output_channels=1),
                         T.ToTensor()])(b_img)

    device = torch.device(device)
    
    # NEW: Pobierz bank SRM z modułu modelu, aby zapewnić spójność.
    SRM_BANK, _ = get_srm_processor() 

    if not use_srm:
        # Just repeat single channel to fake channels (dla testów/legacy)
        n = SRM_BANK.shape[0] # Liczba kanałów (30)
        a_full_srm = a_full.repeat(n, 1, 1)
        b_full_srm = b_full.repeat(n, 1, 1)
        a_patch_srm = a_patch.repeat(n, 1, 1)
        b_patch_srm = b_patch.repeat(n, 1, 1)
        return a_full_srm.to(device), a_patch_srm.to(device), b_full_srm.to(device), b_patch_srm.to(device)

    # Teraz używamy SRM_BANK pobranego z siamese_model.py
    bank = SRM_BANK.to(device)  # Nx1xhw (N=30)
    pad = (bank.shape[2] // 2, bank.shape[3] // 2)

    # convert to device
    a_full_t = a_full.to(device)
    b_full_t = b_full.to(device)
    a_patch_t = a_patch.to(device)
    b_patch_t = b_patch.to(device) # POPRAWIONE: używamy b_patch

    # add batch dim for conv2d (1x1xHxW) -> conv -> 1xN x H x W => squeeze(0)
    a_full_srm = F.conv2d(a_full_t.unsqueeze(0), bank, padding=pad).squeeze(0)
    b_full_srm = F.conv2d(b_full_t.unsqueeze(0), bank, padding=pad).squeeze(0)
    a_patch_srm = F.conv2d(a_patch_t.unsqueeze(0), bank, padding=pad).squeeze(0)
    b_patch_srm = F.conv2d(b_patch_t.unsqueeze(0), bank, padding=pad).squeeze(0)

    return a_full_srm, a_patch_srm, b_full_srm, b_patch_srm

# convenience wrapper to build batched tensors (1, C, H, W)
def prepare_pair_batch(original_file, suspicious_file, device='cpu', use_srm=True):
    a_full_srm, a_patch_srm, b_full_srm, b_patch_srm = prepare_pair_tensors(original_file, suspicious_file, device=device, use_srm=use_srm)
    return (a_full_srm.unsqueeze(0), a_patch_srm.unsqueeze(0),
            b_full_srm.unsqueeze(0), b_patch_srm.unsqueeze(0))