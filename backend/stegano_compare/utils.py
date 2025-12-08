# stegano_compare/utils.py
import logging
import numpy as np
from skimage import io, color, img_as_float
from PIL import Image
import io as _io
import base64

logging.basicConfig(level=logging.INFO, format="[SteganoCompare] %(message)s")


def load_image_safe(path_or_file):
    """
    Safely loads an image, converts to grayscale, and handles RGBA/CMYK formats.

    path_or_file: path string or file-like object (e.g., werkzeug FileStorage)
    returns image normalized to float [0,1], grayscale
    """
    try:
        # Load using Pillow for better format handling
        if hasattr(path_or_file, "read"):
            img = Image.open(path_or_file)
        else:
            img = Image.open(str(path_or_file))

        # Convert to RGB if RGBA or CMYK
        if img.mode in ("RGBA", "LA"):
            logging.info("Detected image with alpha channel — converting to RGB.")
            img = img.convert("RGB")
        elif img.mode == "CMYK":
            logging.info("Detected CMYK image — converting to RGB.")
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)

        # Convert to float [0,1]
        arr = img_as_float(arr)

        # Convert to grayscale
        gray = color.rgb2gray(arr)

        return gray

    except Exception as e:
        logging.error(f"Failed to load image: {e}")
        raise


def save_heatmap_to_base64(fig):
    """Save matplotlib figure to PNG base64 string (no file IO)."""
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_bytes = buf.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    buf.close()
    return b64
