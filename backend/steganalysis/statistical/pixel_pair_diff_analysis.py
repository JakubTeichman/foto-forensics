import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def pixel_pair_diff_analysis(image):
    """
    Pixel-Pair Difference Histogram (PPDH) Analysis.
    Detects subtle pixel perturbations introduced by LSB or embedding operations.
    Returns a heatmap and diagnostic values.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image size for consistency
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    # Compute horizontal and vertical differences
    diff_h = np.abs(np.diff(gray, axis=1))
    diff_v = np.abs(np.diff(gray, axis=0))

    # Combine both difference maps
    diff_combined = (diff_h[:-1, :] + diff_v[:, :-1]) / 2.0
    diff_combined = diff_combined.astype(np.float32)

    # Apply Gaussian smoothing to stabilize noise patterns
    diff_smoothed = cv2.GaussianBlur(diff_combined, (3, 3), 0)

    # Normalize heatmap for visualization
    heatmap = cv2.normalize(diff_smoothed, None, 0, 1, cv2.NORM_MINMAX)

    # Dynamic threshold based on percentile
    threshold_value = np.percentile(diff_smoothed, 95)
    binary_map = (diff_smoothed > threshold_value).astype(np.uint8)

    # Diagnostic metrics
    mean_diff = np.mean(diff_smoothed)
    std_diff = np.std(diff_smoothed)
    edge_density = np.mean(binary_map)

    # Generate heatmap visualization (base64)
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap='inferno')
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return consistent output structure
    return {
        "method": "Pixel Pair Difference Analysis",
        "heatmap_base64": heatmap_base64,
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "edge_density": float(edge_density),
        "dynamic_threshold": float(threshold_value)
    }
