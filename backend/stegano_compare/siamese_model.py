# siamese_model.py
# Model definitions and loader (singleton). Compatible with training script you provided.

import torch
import torch.nn as nn

class MultiScaleBranch(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()
        # local stream
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=1,stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, embed_dim//2)
        )
        # global stream
        self.global_stream = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,padding=1,stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,padding=1,stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, embed_dim//2)
        )

    def forward(self, full, patch):
        g = self.global_stream(full)
        l = self.local(patch)
        return torch.cat([g, l], dim=1)  # embed_dim

class SiameseMulti(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
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

# Loader singleton
_SIAMESE = None
def load_siamese_model(model_path: str, device: str = None, in_channels:int = 30, embed_dim:int = 256):
    """
    Loads model from model_path. device: 'cpu' or 'cuda' or None (auto).
    Returns model on device and sets to eval().
    """
    global _SIAMESE
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    if _SIAMESE is None:
        model = SiameseMulti(in_channels=in_channels, embed_dim=embed_dim)
        # load map_location
        map_loc = dev if dev.type == 'cpu' else None
        state = torch.load(model_path, map_location=map_loc)
        try:
            model.load_state_dict(state)
        except Exception:
            # If state dict saved with module wrapper or different key layout, try common key fix
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
        model.to(dev)
        model.eval()
        _SIAMESE = model
    return _SIAMESE
