from flask import Blueprint, request, jsonify
import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import traceback
import datetime
import sys
import gc
from numpy.linalg import norm

# NOTE: This module assumes noiseprint.extractor (the model) is located
# in the `noiseprint.model` package; keep compatibility with your project.
from noiseprint.model import FullConvNet as NoiseprintModel

noiseprint_bp = Blueprint("noiseprint", __name__)

# ---- 🧠 Global model load (single instance) ----
# We load model once on import and keep it in memory. If CUDA is available,
# move the model to GPU and enable half precision to speed up inference.

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = _get_device()
MODEL_PATH = "/app/noiseprint/weights/model_noiseprint.pth"

model = None
try:
    model = NoiseprintModel()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    # move model to device
    try:
        model.to(DEVICE)
        # use float16 on GPU to speed up and reduce memory
        if DEVICE.type == "cuda":
            model.half()
    except Exception:
        pass
except Exception as e:
    print(f"[ERROR] Failed to load noiseprint model: {str(e)}", file=sys.stderr)


# ---- 🧩 Logging ----
# ---- 📌 Global aggregated reference noiseprint ----
REFERENCE_ACC = None       # accumulator (sum of reference noiseprints)
REFERENCE_COUNT = 0        # number of reference images added

def log_error(msg: str):
    log_path = "noiseprint_errors.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n{'-'*80}\n\n")
    except Exception:
        pass

    # log to stderr as well
    print(msg, file=sys.stderr)


# ---- 🧩 Image preprocessing + tiling (memory-safe) ----
def preprocess_image(image, tile_size=512, overlap=32, min_tile_dim=64):
    """
    Convert PIL image -> grayscale numpy and split into tiles with metadata.
    Returns: tiles_meta: list of (y, x, h, w), tiles_tensors: list of torch tensors on CPU
    and full_shape (H, W).
    Each tensor shape: (1, 1, h, w) dtype=float32 (or float16 on GPU later)
    """
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image))

    image = image.convert("L")
    arr = np.array(image, dtype=np.float32)

    # Light sharpening - keep, but inexpensive
    try:
        img_blur = cv2.GaussianBlur(arr, (3, 3), 0)
        arr = cv2.addWeighted(arr, 1.5, img_blur, -0.5, 0)
    except Exception:
        pass

    arr = arr / 255.0

    H, W = arr.shape
    tiles_meta = []
    tiles = []

    stride = max(1, tile_size - overlap)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)
            tile = arr[y:y2, x:x2]
            h, w = tile.shape
            if h < min_tile_dim or w < min_tile_dim:
                continue
            # pad to tile_size only during inference (we will keep original h,w in meta)
            if h != tile_size or w != tile_size:
                pad_h = tile_size - h
                pad_w = tile_size - w
                tile_padded = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")
            else:
                tile_padded = tile
            # to tensor (CPU)
            tensor = torch.from_numpy(tile_padded).unsqueeze(0).unsqueeze(0).float()
            tiles_meta.append((y, x, h, w))
            tiles.append(tensor)

    if not tiles:
        raise ValueError("Image too small for tiling.")

    return tiles_meta, tiles, (H, W)


# ---- 🧩 Merge tiles results into full noiseprint map (memory-safe batching) ----
def generate_noiseprint_from_tiles(tiles_meta, tiles, full_shape, batch_size=4):
    """
    tiles_meta: list of (y,x,h,w) for each tile in same order as tiles
    tiles: list of torch tensors on CPU
    full_shape: (H, W)

    Returns merged noise numpy array (H, W) float32
    """
    if model is None:
        raise RuntimeError("Noiseprint model not loaded")

    H, W = full_shape
    acc = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    device = DEVICE
    use_fp16 = (device.type == "cuda")

    total = len(tiles)
    idx = 0

    with torch.no_grad():
        while idx < total:
            end = min(idx + batch_size, total)
            batch = torch.cat(tiles[idx:end], dim=0)  # B x 1 x H x W (tile_size)
            # move to device
            try:
                batch = batch.to(device)
                if use_fp16:
                    batch = batch.half()
            except Exception:
                pass

            # model forward
            try:
                out = model(batch)  # expected shape B x C x h x w where C>=1
                out_cpu = out[:, 0, :, :].cpu().float().numpy()
            except Exception as e:
                log_error(f"Model forward failed on batch idx {idx}-{end}: {e}{traceback.format_exc()}")
                # cleanup and continue with next batch
                try:
                    del batch
                except Exception:
                    pass
                gc.collect()
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                idx = end
                continue

            # accumulate outputs
            for i in range(out_cpu.shape[0]):
                y, x, h, w = tiles_meta[idx + i]
                res = out_cpu[i]
                # crop to original tile shape in case of padding
                res_crop = res[:h, :w]
                acc[y:y + h, x:x + w] += res_crop
                count[y:y + h, x:x + w] += 1.0

            # cleanup
            try:
                del batch, out, out_cpu
            except Exception:
                pass
            gc.collect()
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            idx = end

    count[count == 0] = 1.0
    merged = acc / count
    return merged.astype(np.float32)


# ---- 🧮 Helpers ----
def local_normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.std(arr)
    if std < 1e-8:
        return arr
    return (arr - np.mean(arr)) / std


def cross_correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def diff_energy(a, b):
    diff = np.abs(a - b)
    return float(np.sum(diff ** 2))


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    denom = (norm(a) * norm(b))
    if denom == 0:
        return 0.0
    return float((np.dot(a, b) / denom + 1) / 2)


# ---- 📌 Update mean reference noiseprint ----
def update_reference_noiseprint(ref_np):
    """
    Update the global mean reference noiseprint.
    ref_np must be a normalized numpy noiseprint, (H, W)
    """
    global REFERENCE_ACC, REFERENCE_COUNT

    if ref_np is None:
        return

    ref_np = np.nan_to_num(ref_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Initialize accumulator
    if REFERENCE_ACC is None:
        REFERENCE_ACC = ref_np.astype(np.float64)
        REFERENCE_COUNT = 1
    else:
        # Resize if different shape
        if REFERENCE_ACC.shape != ref_np.shape:
            ref_np = cv2.resize(ref_np, (REFERENCE_ACC.shape[1], REFERENCE_ACC.shape[0]))

        REFERENCE_ACC += ref_np
        REFERENCE_COUNT += 1


def get_mean_reference_noiseprint():
    """
    Returns the current mean reference noiseprint or None.
    """
    global REFERENCE_ACC, REFERENCE_COUNT

    if REFERENCE_ACC is None or REFERENCE_COUNT == 0:
        return None

    return (REFERENCE_ACC / REFERENCE_COUNT).astype(np.float32)


def compute_noise_stats(noise):
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mean": float(np.mean(noise)),
        "std": float(np.std(noise)),
        "energy": float(np.sum(noise ** 2)),
        "entropy": float(-np.sum(noise * np.log(np.abs(noise) + 1e-10))),
    }


# ---- ⚙️ ROUTE: Noiseprint generation (single image) ----
@noiseprint_bp.route("/generate", methods=["POST"])
def generate():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided."}), 400

        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            log_error(f"❌ Failed to open image:{traceback.format_exc()}")
            return jsonify({"error": "Invalid or corrupted image file."}), 400

        try:
            tiles_meta, tiles, full_shape = preprocess_image(img, tile_size=512, overlap=32)
            noise = generate_noiseprint_from_tiles(tiles_meta, tiles, full_shape, batch_size=4)
        except MemoryError:
            log_error("❌ MemoryError – image too large")
            return jsonify({"error": "Image too large. Try a smaller file."}), 500
        except Exception as e:
            log_error(f"❌ Exception in noiseprint generation:{traceback.format_exc()}")
            return jsonify({"error": "Noiseprint generation failed."}), 500

        stats = compute_noise_stats(noise)
        normalized = local_normalize(noise)
        _, buffer = cv2.imencode(".png", (normalized * 127 + 128).astype(np.uint8))
        encoded = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "noiseprint": encoded,
            "stats": stats
        })

    except Exception:
        log_error(f"❗ Uncaught top-level error:{traceback.format_exc()}")
        return jsonify({"error": "Unexpected backend error."}), 500


# ---- ⚖️ ROUTE: Compare multiple noiseprints ----
@noiseprint_bp.route('/compare', methods=['POST'])
def compare():
    try:
        evidence_file = request.files.get('evidence')
        refs = request.files.getlist('references')

        if not evidence_file:
            return jsonify({'error': 'No evidence image provided.'}), 400

        # evidence tiles -> noise
        e_img = Image.open(evidence_file.stream).convert('RGB')
        e_meta, e_tiles, e_shape = preprocess_image(e_img, tile_size=512, overlap=32)
        evidence_np = generate_noiseprint_from_tiles(e_meta, e_tiles, e_shape, batch_size=4)
        evidence_np = local_normalize(evidence_np)

        correlations, diff_energies = [], []

        for ref_file in refs:
            try:
                r_img = Image.open(ref_file.stream).convert('RGB')
                r_meta, r_tiles, r_shape = preprocess_image(r_img, tile_size=512, overlap=32)
                ref_np = generate_noiseprint_from_tiles(r_meta, r_tiles, r_shape, batch_size=4)
                ref_np = local_normalize(ref_np)
                update_reference_noiseprint(ref_np)


                # resize ref to evidence shape if needed
                if ref_np.shape != evidence_np.shape:
                    ref_np = cv2.resize(ref_np, (evidence_np.shape[1], evidence_np.shape[0]))

                corr = cross_correlation(evidence_np, ref_np)
                diff_en = diff_energy(evidence_np, ref_np)

                correlations.append(corr)
                diff_energies.append(diff_en)
            except Exception:
                log_error(f"⚠️ Failed to process reference file:{traceback.format_exc()}")
                continue

        if not correlations:
            return jsonify({'error': 'No valid reference comparisons were produced.'}), 400

        mean_corr = float(np.mean(correlations))
        std_corr = float(np.std(correlations))
        mean_diff_en = float(np.mean(diff_energies))

        _, buf_evidence = cv2.imencode('.png', (evidence_np * 127 + 128).astype(np.uint8))

        mean_ref_np = get_mean_reference_noiseprint()

        mean_ref_encoded = None
        if mean_ref_np is not None:
            _, buf = cv2.imencode(".png", (mean_ref_np * 127 + 128).astype(np.uint8))
            mean_ref_encoded = base64.b64encode(buf).decode("utf-8")


        return jsonify({
            'evidence_noiseprint': base64.b64encode(buf_evidence).decode('utf-8'),
            'pairwise_correlations': correlations,
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'mean_diff_energy': mean_diff_en,
            'stats_evidence': compute_noise_stats(evidence_np),
            "mean_reference_noiseprint": mean_ref_encoded,

        })

    except Exception as e:
        log_error(f"❌ Noiseprint comparison failed:{traceback.format_exc()}")
        return jsonify({'error': f'Noiseprint comparison failed: {str(e)}'}), 500


# ---- 🧬 ROUTE: Embedding cosine similarity ----
@noiseprint_bp.route('/compare_embedding', methods=['POST'])
def compare_embedding():
    try:
        evidence_file = request.files.get('evidence')
        reference_file = request.files.get('reference')

        if not evidence_file or not reference_file:
            return jsonify({'error': 'Both evidence and reference images are required.'}), 400

        e_img = Image.open(evidence_file.stream).convert('RGB')
        r_img = Image.open(reference_file.stream).convert('RGB')

        e_meta, e_tiles, e_shape = preprocess_image(e_img, tile_size=512, overlap=32)
        r_meta, r_tiles, r_shape = preprocess_image(r_img, tile_size=512, overlap=32)

        evidence_np = generate_noiseprint_from_tiles(e_meta, e_tiles, e_shape, batch_size=4)
        reference_np = generate_noiseprint_from_tiles(r_meta, r_tiles, r_shape, batch_size=4)

        evidence_np = local_normalize(evidence_np)
        reference_np = local_normalize(reference_np)

        if evidence_np.shape != reference_np.shape:
            reference_np = cv2.resize(reference_np, (evidence_np.shape[1], evidence_np.shape[0]))

        similarity = cosine_similarity(evidence_np, reference_np)

        _, buf_evidence = cv2.imencode('.png', (evidence_np * 127 + 128).astype(np.uint8))
        _, buf_reference = cv2.imencode('.png', (reference_np * 127 + 128).astype(np.uint8))

        return jsonify({
            "method": "embedding_cosine_similarity",
            "similarity_score": similarity,
            "evidence_noiseprint": base64.b64encode(buf_evidence).decode('utf-8'),
            "reference_noiseprint": base64.b64encode(buf_reference).decode('utf-8'),
            "stats_evidence": compute_noise_stats(evidence_np),
            "stats_reference": compute_noise_stats(reference_np)
        })

    except Exception as e:
        log_error(f"❌ Embedding comparison failed:{traceback.format_exc()}")
        return jsonify({'error': f'Embedding comparison failed: {str(e)}'}), 500


# ---- 🧬 ROUTE: Compare with reference noiseprint database (SQL) ----
@noiseprint_bp.route('/compare_with_db', methods=['POST'])
def compare_with_db():
    try:
        from extensions import db
        from routes.add_reference import DeviceReference, deserialize_noiseprint

        evidence_file = request.files.get('evidence')
        if not evidence_file:
            return jsonify({'error': 'No evidence image provided.'}), 400

        e_img = Image.open(evidence_file.stream).convert('RGB')
        e_meta, e_tiles, e_shape = preprocess_image(e_img, tile_size=512, overlap=32)
        evidence_np = generate_noiseprint_from_tiles(e_meta, e_tiles, e_shape, batch_size=4)
        evidence_np = local_normalize(evidence_np)

        refs = db.session.query(DeviceReference).all()
        if not refs:
            return jsonify({'error': 'No reference entries found in database.'}), 404

        results = []
        for ref in refs:
            try:
                ref_np = deserialize_noiseprint(ref.noiseprint)
                ref_np = local_normalize(ref_np)

                if ref_np.shape != evidence_np.shape:
                    ref_np = cv2.resize(ref_np, (evidence_np.shape[1], evidence_np.shape[0]))

                sim = cosine_similarity(evidence_np, ref_np)
                results.append({
                    "id": ref.id,
                    "manufacturer": ref.manufacturer,
                    "model": ref.model,
                    "num_images": ref.num_images,
                    "similarity": round(sim, 4)
                })
            except Exception:
                log_error(f"⚠️ Failed to compare with reference ID={ref.id}{traceback.format_exc()}")

        results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
        best_match = results_sorted[0] if results_sorted else None

        response = {
            "total_references": len(refs),
            "matches": results_sorted,
            "best_match": best_match,
        }
        return jsonify(response)

    except Exception:
        log_error(f"❌ Compare_with_db failed:{traceback.format_exc()}")
        return jsonify({'error': 'Internal error during database comparison.'}), 500
