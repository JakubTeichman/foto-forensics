import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

SRM_CHANNELS = 30
EMBED_DIM = 256


def make_srm_bank():
    """Tworzy bank 30 filtrów SRM."""
    kernels = []
    
    k1 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float32)
    k2 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    k3 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
    k4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) 
    
    k5 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
    k6 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
    k7 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32)
    
    base_kernels = [k1, k2, k3, k4, k5, k6, k7]
    
    for k in base_kernels:
        kernels.append(k)
        
        kernels.append(np.rot90(k))
        kernels.append(np.rot90(k, 2))
        
        if k.shape == (3, 3):
             if not np.array_equal(k, k.T):
                 kernels.append(k.T)
                 kernels.append(np.rot90(k.T))
                 
    
    if len(kernels) < SRM_CHANNELS:
        print(f"Ostrzeżenie: Wygenerowano tylko {len(kernels)} filtrów, ale model oczekuje {SRM_CHANNELS}.")
        
    kernels = kernels[:SRM_CHANNELS]
    arrs = []
    
    for k in kernels:
        kf = np.array(k, dtype=np.float32)
        
        if kf.sum() != 0:
            kf = kf - kf.mean()
            
        arrs.append(kf)
        
    bank = np.stack([k for k in arrs])[:, None, :, :]
    return torch.from_numpy(bank)

SRM_BANK = make_srm_bank().float()
if SRM_BANK.shape[0] != SRM_CHANNELS:
    raise RuntimeError(f"Oczekiwano {SRM_CHANNELS} filtrów SRM, znaleziono {SRM_BANK.shape[0]}.")

def apply_srm_filters(img_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Stosuje bank filtrów SRM do tensora obrazu wejściowego.
    
    Oczekuje wejścia: (B, 1, H, W) lub (1, H, W) - tensor jednokanałowy po konwersji do skali szarości.
    Zwraca wyjście: (B, 30, H, W) lub (30, H, W) - 30-kanałowy tensor.
    """
    bank = SRM_BANK.to(device)
    
    is_single = (img_tensor.dim() == 3)
    if is_single:
        img = img_tensor.unsqueeze(0)
    else:
        img = img_tensor
        
    if img.shape[1] != 1:
         raise ValueError(f"Oczekiwano 1 kanału wejściowego dla SRM, otrzymano {img.shape[1]}")

    kernel_h, kernel_w = bank.shape[2], bank.shape[3]
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    out = nn.functional.conv2d(img, bank, padding=(pad_h, pad_w))
    
    if is_single:
        return out.squeeze(0) 
    return out

class MultiScaleBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=EMBED_DIM):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=1,stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, embed_dim//2)
        )
        self.global_stream = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,padding=1,stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=1,stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, embed_dim//2)
        )

    def forward(self, full, patch):
        g = self.global_stream(full)
        l = self.local(patch)
        return torch.cat([g, l], dim=1) 

class SiameseMulti(nn.Module):
    def __init__(self, in_channels, embed_dim=EMBED_DIM):
        super().__init__()
        self.branch = MultiScaleBranch(in_channels, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, a_full, a_patch, b_full, b_patch):
        ea = self.branch(a_full, a_patch)
        eb = self.branch(b_full, b_patch)
        d = torch.abs(ea - eb)
        out = self.head(d).squeeze(1)
        return out, ea, eb

_SIAMESE = None
def load_siamese_model(model_path: str, device: str = None, in_channels:int = SRM_CHANNELS, embed_dim:int = EMBED_DIM):
    """
    Ładuje model z wagami z model_path.
    in_channels i embed_dim są ustawione na wartości użyte podczas treningu (30 i 256).
    """
    global _SIAMESE
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    if _SIAMESE is None:
        model = SiameseMulti(in_channels=in_channels, embed_dim=embed_dim) 
        
        map_loc = dev if dev.type == 'cpu' else None
        
        try:
            state = torch.load(model_path, map_location=map_loc)
        except Exception as e:
            print(f"Błąd ładowania pliku .pth: {e}")
            return None

        try:
            model.load_state_dict(state)
        except Exception:
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
                
        model.to(dev)
        model.eval()
        _SIAMESE = model
    return _SIAMESE

def get_srm_processor() -> Tuple[torch.Tensor, callable]:
    """Zwraca bank filtrów SRM i funkcję do ich aplikowania."""
    return SRM_BANK, apply_srm_filters