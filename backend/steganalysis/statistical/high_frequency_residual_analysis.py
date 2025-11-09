import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def high_frequency_residual_analysis(image):
    """
    High-Frequency Residual Analysis (HFR)
    -------------------------------------
    Detects inconsistencies in high-frequency bands using FFT-based residuals.
    Useful for revealing subtle embedding artifacts invisible in the spatial domain.
    """

    # Convert to grayscale and normalize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0

    # Perform 2D FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    # Create high-pass mask (remove low frequencies)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = 24  # radius for low frequency cutoff
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    # Apply mask to FFT
    fft_filtered = fft_shift * mask

    # Inverse FFT to get high-frequency residual
    f_ishift = np.fft.ifftshift(fft_filtered)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    img_back = np.clip(img_back, 0, 1)

    # Compute statistical descriptors
    mean_val = float(np.mean(img_back))
    std_val = float(np.std(img_back))
    kurtosis_val = float(np.mean(((img_back - mean_val) / (std_val + 1e-12)) ** 4) - 3.0)

    # Dynamic threshold for detection (based on kurtosis and std)
    dynamic_threshold = float(np.clip(0.45 + 0.15 * (kurtosis_val / 3.0), 0.3, 0.8))
    score = float(np.clip(std_val * (1 + abs(kurtosis_val)), 0.0, 1.0))
    detected = bool(score > dynamic_threshold)

    # Generate heatmap for visualization
    plt.figure(figsize=(4, 4))
    plt.imshow(img_back, cmap='inferno')
    plt.axis('off')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return structured output
    return {
        "method": "High-Frequency Residual Analysis",
        "heatmap_base64": heatmap_base64,
        "details": {
            "mean": mean_val,
            "std": std_val,
            "kurtosis": kurtosis_val,
            "dynamic_threshold": dynamic_threshold,
            "score": score,
            "detected": detected
        }
    }
