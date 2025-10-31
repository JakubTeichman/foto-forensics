# steganalysis/cnn/noiseprint_extractor.py
# Uproszczony moduł imitujący ideę Noiseprint
import numpy as np
from skimage import io, restoration

class NoiseprintExtractor:
    name = "Noiseprint"

    @staticmethod
    def analyze(image_path: str) -> float:
        image = io.imread(image_path, as_gray=True)
        residual = image - restoration.denoise_nl_means(image)
        score = np.mean(np.abs(residual))
        return float(np.clip(score, 0, 1))
