# noiseprint_extractor.py
"""
Noiseprint model implementation based on:

D. Cozzolino and L. Verdoliva,
"Noiseprint: A CNN-Based Camera Model Fingerprint",
IEEE Transactions on Information Forensics and Security, vol. 15, pp. 144–159, 2020.
DOI: 10.1109/TIFS.2019.2916364

Original implementation (Matlab/TensorFlow) © 2019 GRIP-UNINA.
Adapted for research and educational use in Python/PyTorch by Jakub Teichman, 2025.

Extractor utilities: tiling (512x512), overlap, strict normalization,
outlier masking and base64 PNG encoder for noiseprint visualization.
"""
import torch
import numpy as np
import torchvision.transforms as transforms
import traceback
import datetime
import os
from PIL import Image
import io
import base64
import gc

from .model import FullConvNet
from .utils import im2f, jpeg_qtableinv

LOG_PATH = "noiseprint_errors.log"


def log_error(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n{'-'*80}\n\n")


def safe_infer(net, tensor_image):
    """Run model safely and return numpy map or None on failure.
    tensor_image must already be on the same device as net.
    """
    try:
        with torch.no_grad():
            out = net(tensor_image)
            result = out[0][0].cpu().numpy()
        return result
    except Exception as e:
        log_error(f"Model inference failed: {e}\n{traceback.format_exc()}")
        return None


def strict_normalize(arr):
    """
    Deterministic normalization:
    - NaNs -> 0
    - subtract mean, divide by std (if std tiny -> 1)
    - return float32
    """
    a = np.array(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mean = a.mean()
    std = a.std()
    if std < 1e-6:
        std = 1.0
    return ((a - mean) / std).astype(np.float32)


def mask_outliers(arr, z_thresh=5.0):
    """
    Simple outlier mask: values with |z| > z_thresh replaced by median.
    """
    a = np.array(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(a)
    std = a.std()
    if std < 1e-6:
        return a
    z = np.abs((a - med) / std)
    a[z > z_thresh] = med
    return a


def im_to_base64_png(np_image):
    """
    Encode a float numpy image (can be negative) into PNG base64 (no data: prefix).
    Visualization mapping: strict_normalize -> scale to 0..255 centered at 128.
    """
    img = np.array(np_image, dtype=np.float32)
    img_n = strict_normalize(img)
    # scale for visualization: amplify slightly for visibility
    disp = np.clip((img_n * 30) + 128, 0, 255).astype(np.uint8)
    pil = Image.fromarray(disp)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_device_from_net(net):
    try:
        p = next(net.parameters())
        return p.device
    except StopIteration:
        return torch.device("cpu")


def tile_infer_and_merge(net, img, transform, tile_size=512, overlap=64, batch_infer=False, batch_size=4):
    """
    Tile grayscale img (H,W) into overlapping tiles, infer noiseprint on each tile,
    and merge by weighted averaging.

    Memory-safe implementation: each tile (or small batch) is pushed to the
    model, result copied to CPU/Numpy, temporary tensors deleted and gc run.

    Arguments:
      net: torch model (should already be on desired device)
      img: 2D numpy array (H,W) grayscale float32
      transform: torchvision transform (e.g., ToTensor)
      tile_size: int
      overlap: int
      batch_infer: if True, will batch tiles before calling model (reduces calls, increases memory)
      batch_size: number of tiles per batch (only if batch_infer True)

    Returns merged noise map (float32) or None on failure.
    """
    H, W = img.shape
    stride = tile_size - overlap
    if stride <= 0:
        stride = tile_size

    ys = list(range(0, max(1, H - overlap), stride))
    xs = list(range(0, max(1, W - overlap), stride))

    # Ensure last tile covers the end
    if ys[-1] + tile_size < H:
        ys[-1] = max(0, H - tile_size)
    if xs[-1] + tile_size < W:
        xs[-1] = max(0, W - tile_size)

    acc = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    device = _get_device_from_net(net)

    # helper to free memory after each inference
    def _cleanup(vars_to_del=None):
        if vars_to_del:
            for v in vars_to_del:
                try:
                    del v
                except Exception:
                    pass
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # If batch_infer==False process tile-by-tile (recommended for low memory)
    if not batch_infer:
        for y in ys:
            for x in xs:
                y1 = y
                x1 = x
                y2 = min(y1 + tile_size, H)
                x2 = min(x1 + tile_size, W)

                tile = img[y1:y2, x1:x2]

                pad_h = tile_size - tile.shape[0]
                pad_w = tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile_padded = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")
                    tensor = transform(tile_padded).reshape(1, 1, tile_padded.shape[0], tile_padded.shape[1])
                else:
                    tensor = transform(tile).reshape(1, 1, tile.shape[0], tile.shape[1])

                # move tensor to model device
                try:
                    tensor = tensor.to(device)
                except Exception:
                    # fallback: keep on cpu
                    pass

                # Add tiny epsilon (avoid allocating large random tensors)
                # If you want noise-augmentation, enable it externally.
                # tensor = tensor + 1e-8 * torch.randn_like(tensor, device=tensor.device)

                res = safe_infer(net, tensor)
                # cleanup tensor asap
                _cleanup([tensor])

                if res is None:
                    log_error(f"Tile inference failed at x={x1} y={y1}")
                    return None

                # crop to original tile size (in case of padding)
                res_crop = res[: tile.shape[0], : tile.shape[1]]

                acc[y1:y1 + res_crop.shape[0], x1:x1 + res_crop.shape[1]] += res_crop
                weight[y1:y1 + res_crop.shape[0], x1:x1 + res_crop.shape[1]] += 1.0

        weight[weight == 0] = 1.0
        merged = acc / weight
        return merged.astype(np.float32)

    # Batch inference path (uses more memory, but fewer model calls)
    else:
        tiles_meta = []
        tiles_tensors = []
        try:
            for y in ys:
                for x in xs:
                    y1 = y
                    x1 = x
                    y2 = min(y1 + tile_size, H)
                    x2 = min(x1 + tile_size, W)

                    tile = img[y1:y2, x1:x2]
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    if pad_h > 0 or pad_w > 0:
                        tile_padded = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")
                        tensor = transform(tile_padded).reshape(1, 1, tile_padded.shape[0], tile_padded.shape[1])
                    else:
                        tensor = transform(tile).reshape(1, 1, tile.shape[0], tile.shape[1])

                    tiles_meta.append((y1, x1, tile.shape[0], tile.shape[1]))
                    tiles_tensors.append(tensor)

                    # if batch is full, run inference
                    if len(tiles_tensors) >= batch_size:
                        batch = torch.cat(tiles_tensors, dim=0)
                        try:
                            batch = batch.to(device)
                        except Exception:
                            pass

                        res_batch = safe_infer(net, batch) if False else None
                        # note: safe_infer expects single input; for batch we perform manual call
                        try:
                            with torch.no_grad():
                                out = net(batch)
                                out_cpu = out[:, 0, :, :].cpu().numpy()  # shape B x H x W
                        except Exception as e:
                            log_error(f"Batch inference failed: {e}\n{traceback.format_exc()}")
                            _cleanup(tiles_tensors + [batch])
                            return None

                        # iterate results and accumulate
                        for idx, (y1_, x1_, h_, w_) in enumerate(tiles_meta[:len(out_cpu)]):
                            res = out_cpu[idx]
                            res_crop = res[:h_, :w_]
                            acc[y1_:y1_ + res_crop.shape[0], x1_:x1_ + res_crop.shape[1]] += res_crop
                            weight[y1_:y1_ + res_crop.shape[0], x1_:x1_ + res_crop.shape[1]] += 1.0

                        # cleanup used tensors and metas
                        _cleanup(tiles_tensors + [batch])
                        tiles_meta = []
                        tiles_tensors = []

            # leftover tiles
            if tiles_tensors:
                batch = torch.cat(tiles_tensors, dim=0)
                try:
                    batch = batch.to(device)
                except Exception:
                    pass
                try:
                    with torch.no_grad():
                        out = net(batch)
                        out_cpu = out[:, 0, :, :].cpu().numpy()
                except Exception as e:
                    log_error(f"Batch inference failed on leftovers: {e}\n{traceback.format_exc()}")
                    _cleanup(tiles_tensors + [batch])
                    return None

                for idx, (y1_, x1_, h_, w_) in enumerate(tiles_meta[:len(out_cpu)]):
                    res = out_cpu[idx]
                    res_crop = res[:h_, :w_]
                    acc[y1_:y1_ + res_crop.shape[0], x1_:x1_ + res_crop.shape[1]] += res_crop
                    weight[y1_:y1_ + res_crop.shape[0], x1_:x1_ + res_crop.shape[1]] += 1.0

                _cleanup(tiles_tensors + [batch])
                tiles_meta = []
                tiles_tensors = []

            weight[weight == 0] = 1.0
            merged = acc / weight
            return merged.astype(np.float32)

        except Exception as e:
            log_error(f"Unexpected tiled batch processing error: {e}\n{traceback.format_exc()}")
            _cleanup(tiles_tensors)
            return None


def getNoiseprint_fullpipeline(image_input, weights_dir="pretrained_weights",
                               tile_size=512, overlap=64, block_threshold_wh=(3000, 4000),
                               net: torch.nn.Module = None, device: torch.device = None,
                               allow_scaled_fallback=True):
    """
    Full pipeline:
    - load image with im2f
    - uniform-image stabilization
    - load model with QF fallback (unless 'net' provided)
    - if large (H >= block_threshold_wh[0] and W >= block_threshold_wh[1]) or maxdim > block_threshold_wh[0] -> tiling
    - else full-image inference
    - postprocess: mask_outliers + strict_normalize

    Returns: (orig_img_float32, noise_map_float32, meta_dict) or (None, None, error)
    """
    meta = {"scaled": False, "method": None}
    try:
        img, _ = im2f(image_input, channel=1)
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 3:
            img = img.mean(axis=2)

        if img.max() <= 1.0:
            img *= 255.0

        if np.std(img) < 1e-3:
            log_error("⚠️ Uniform image detected, adding tiny random noise.")
            img = img + np.random.normal(0, 0.1, img.shape)

        transform = transforms.ToTensor()

        # determine QF if needed
        try:
            QF = jpeg_qtableinv(image_input)
        except Exception:
            QF = 101

        # Load or use provided net
        if net is None:
            weight_path = f"{weights_dir}/model_qf{int(QF)}.pth"
            if not os.path.exists(weight_path):
                log_error(f"Missing weights for QF={QF}, falling back to model_qf101.pth")
                weight_path = f"{weights_dir}/model_qf101.pth"

            net = FullConvNet(0.9, False)
            net.load_state_dict(torch.load(weight_path, map_location="cpu"))
            net.eval()

            # move to provided device or CPU
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                net.to(device)
            except Exception:
                pass
        else:
            # If net provided, infer device from it
            if device is None:
                device = _get_device_from_net(net)

        H, W = img.shape
        # trigger tiling if both dims >= threshold pair OR maxdim > first threshold
        trigger = (H >= block_threshold_wh[0] and W >= block_threshold_wh[1]) or max(H, W) > block_threshold_wh[0]
        if trigger:
            meta["method"] = "tile"
            log_error(f"📌 Large image {W}x{H} -> using tiling (tile={tile_size}, overlap={overlap})")
            noise = tile_infer_and_merge(net, img, transform, tile_size=tile_size, overlap=overlap)
            if noise is None:
                log_error("⚠️ Tiling failed, will try full-image then scaled fallback")
            else:
                noise = mask_outliers(noise)
                noise = strict_normalize(noise)
                return img.astype(np.float32), noise.astype(np.float32), meta

        # full-image attempt (may be memory heavy)
        meta["method"] = "full"
        tensor_image = transform(img).reshape(1, 1, H, W)
        try:
            tensor_image = tensor_image.to(device)
        except Exception:
            pass

        # avoid creating large random tensors; small epsilon if wanted
        # tensor_image = tensor_image + 1e-8 * torch.randn_like(tensor_image, device=tensor_image.device)

        res = safe_infer(net, tensor_image)
        # free tensor_image
        try:
            del tensor_image
        except Exception:
            pass
        gc.collect()
        if device is not None and device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if res is not None:
            res = mask_outliers(res)
            res = strict_normalize(res)
            return img.astype(np.float32), res.astype(np.float32), meta

        # scaled fallback (only if allowed)
        if allow_scaled_fallback:
            log_error("⚠️ Full-image failed; trying scaled fallback")
            MAX_SIZE = 2048
            maxdim = max(H, W)
            if maxdim > MAX_SIZE:
                scale = MAX_SIZE / maxdim
                new_size = (int(W * scale), int(H * scale))
                img_pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                img_resized = img_pil.resize(new_size, Image.LANCZOS)
                img_resized = np.asarray(img_resized, dtype=np.float32)
                tensor_image = transform(img_resized).reshape(1, 1, img_resized.shape[0], img_resized.shape[1])
                try:
                    tensor_image = tensor_image.to(device)
                except Exception:
                    pass
                res2 = safe_infer(net, tensor_image)
                try:
                    del tensor_image
                except Exception:
                    pass
                gc.collect()
                if device is not None and device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                if res2 is not None:
                    meta["method"] = "scaled_fallback"
                    meta["scaled"] = True
                    res2 = mask_outliers(res2)
                    res2 = strict_normalize(res2)
                    return img_resized.astype(np.float32), res2.astype(np.float32), meta

        log_error("❌ Noiseprint generation completely failed.")
        return None, None, {"error": "inference_failed"}

    except Exception as e:
        log_error(f"Unexpected error in getNoiseprint_fullpipeline: {e}\n{traceback.format_exc()}")
        return None, None, {"error": str(e)}
