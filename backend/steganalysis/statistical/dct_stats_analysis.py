import cv2
import numpy as np
import io
import base64
from matplotlib import pyplot as plt
from PIL import Image

def analyze_dct_stats(image_path):
    """
    DCT Coefficient Statistics Analysis (for JPEG images)

    Detects anomalies in DCT coefficient distributions that may indicate hidden data,
    e.g., steganography in the frequency domain (through DCT coefficient modification).
    Returns: heatmap, metrics, and textual summary.
    """

    # --- 1️⃣ Load and preprocess the image ---
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Failed to load image."}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adjust dimensions to multiples of 8
    h_crop, w_crop = h - (h % 8), w - (w % 8)
    gray = gray[:h_crop, :w_crop]

    # --- 2️⃣ Compute DCT blocks ---
    blocks_v = h_crop // 8
    blocks_h = w_crop // 8

    dct_map = np.zeros((blocks_v, blocks_h))
    hf_energy = []
    lf_energy = []

    for i in range(blocks_v):
        for j in range(blocks_h):
            block = gray[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
            dct_block = cv2.dct(block)
            abs_dct = np.abs(dct_block)

            # Energy of low vs high frequencies
            low_freq = abs_dct[:4, :4].sum()
            high_freq = abs_dct[4:, 4:].sum()
            ratio = high_freq / (low_freq + 1e-6)
            dct_map[i, j] = ratio

            lf_energy.append(low_freq)
            hf_energy.append(high_freq)

    # --- 3️⃣ Normalize and generate heatmap ---
    heatmap = cv2.normalize(dct_map, None, 0, 1, cv2.NORM_MINMAX)
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='hot')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap_encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

    # --- 4️⃣ Compute metrics ---
    hf_mean = np.mean(hf_energy)
    lf_mean = np.mean(lf_energy)
    hf_lf_ratio = hf_mean / (lf_mean + 1e-6)

    kurtosis_hf = float(np.mean((hf_energy - hf_mean)**4) / (np.std(hf_energy)**4 + 1e-6))
    skewness_hf = float(np.mean((hf_energy - hf_mean)**3) / (np.std(hf_energy)**3 + 1e-6))

    metrics = {
        "Mean HF energy": round(hf_mean, 4),
        "Mean LF energy": round(lf_mean, 4),
        "HF/LF ratio": round(hf_lf_ratio, 4),
        "HF energy skewness": round(skewness_hf, 4),
        "HF energy kurtosis": round(kurtosis_hf, 4),
    }

    # --- 5️⃣ Text summary ---
    summary = (
        f"DCT analysis revealed an average high-to-low frequency energy ratio of {hf_lf_ratio:.3f}. "
        f"The skewness ({skewness_hf:.3f}) and kurtosis ({kurtosis_hf:.3f}) values indicate a "
        f"{'non-standard' if hf_lf_ratio > 1.2 or kurtosis_hf > 4 else 'typical'} distribution of DCT coefficients. "
        "Abnormal values may suggest local block modifications or hidden information embedded in the DCT domain."
    )

    return {
        "heatmap": heatmap_encoded,
        "metrics": metrics,
        "summary": summary
    }
