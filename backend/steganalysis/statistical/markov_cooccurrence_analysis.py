import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def markov_cooccurrence_analysis(image):
    """
    Markov / Co-occurrence Statistics Analysis (SPAM-like).
    Detects subtle embedding artifacts in spatial domain using
    co-occurrence matrices of pixel difference transitions.
    Returns a heatmap and diagnostic metrics for aggregation.
    """

    # Convert to grayscale and resize for consistency
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32)

    # Compute pixel difference matrices in multiple directions
    diff_h = np.diff(gray, axis=1)
    diff_v = np.diff(gray, axis=0)

    # Quantize differences into small range [-4, 4] for co-occurrence
    q = 4
    diff_h_q = np.clip(np.round(diff_h / 2.0), -q, q).astype(np.int32)
    diff_v_q = np.clip(np.round(diff_v / 2.0), -q, q).astype(np.int32)

    # Create co-occurrence matrices
    cooc_h = np.zeros((2 * q + 1, 2 * q + 1), dtype=np.float32)
    cooc_v = np.zeros((2 * q + 1, 2 * q + 1), dtype=np.float32)

    for x in range(diff_h_q.shape[0]):
        for y in range(diff_h_q.shape[1] - 1):
            a, b = diff_h_q[x, y], diff_h_q[x, y + 1]
            cooc_h[a + q, b + q] += 1

    for x in range(diff_v_q.shape[0] - 1):
        for y in range(diff_v_q.shape[1]):
            a, b = diff_v_q[x, y], diff_v_q[x + 1, y]
            cooc_v[a + q, b + q] += 1

    # Normalize matrices
    cooc_h /= np.sum(cooc_h) + 1e-12
    cooc_v /= np.sum(cooc_v) + 1e-12

    # Symmetry deviation metric (strong indicator of embedding)
    sym_diff_h = np.mean(np.abs(cooc_h - np.flipud(np.fliplr(cooc_h))))
    sym_diff_v = np.mean(np.abs(cooc_v - np.flipud(np.fliplr(cooc_v))))
    symmetry_score = float((sym_diff_h + sym_diff_v) / 2.0)

    # Combined co-occurrence (for visualization)
    combined = (cooc_h + cooc_v) / 2.0

    # Compute entropy and energy
    nonzero_vals = combined[combined > 0]
    entropy = float(-np.sum(nonzero_vals * np.log2(nonzero_vals + 1e-12)))
    energy = float(np.sum(combined ** 2))

    # Create heatmap visualization
    plt.figure(figsize=(4, 4))
    plt.imshow(combined, cmap='viridis')
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return structured result
    return {
        "method": "Markov / Co-occurrence Analysis",
        "heatmap_base64": heatmap_base64,
        "details": {
            "entropy": entropy,
            "energy": energy,
            "symmetry_score": symmetry_score,
            "matrix_size": int(combined.shape[0])
        }
    }
