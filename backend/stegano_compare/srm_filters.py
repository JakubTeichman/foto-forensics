import numpy as np
import torch

def make_srm_bank(num_filters=30):
    kernels = []
    kernels += [np.array([[0,0,0],[0,1,-1],[0,0,0]]),
                np.array([[0,0,0],[0,1,0],[0,-1,0]]),
                np.array([[0,0,0],[0,1,0],[-1,0,0]]),
                np.array([[-1,1,0],[1,0,0],[0,0,0]])]
    kernels += [np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
                np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]])]
    kernels += [np.array([[1,-2,1],[-2,4,-2],[1,-2,1]]),
                np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
                np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])]

    base = kernels.copy()
    for k in base:
        for scale in [1.0, 0.5, 2.0]:
            kernels.append(k * scale)
        kernels.append(np.rot90(k))
        kernels.append(np.rot90(k,2))

    kernels = kernels[:num_filters]
    arrs = []
    for k in kernels:
        kf = np.array(k, dtype=np.float32)
        if kf.sum() != 0:
            kf = kf - kf.mean()
        arrs.append(kf)

    bank = np.stack([k for k in arrs])[:, None, :, :]
    return torch.from_numpy(bank).float()

_SRM_BANK = None
def get_srm_bank(num_filters=30):
    global _SRM_BANK
    if _SRM_BANK is None:
        _SRM_BANK = make_srm_bank(num_filters=num_filters)
    return _SRM_BANK
