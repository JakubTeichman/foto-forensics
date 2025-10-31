import cv2
import numpy as np
from .classic.chi_square_analysis import analyze as ChiSquareAnalysis
from .classic.lsb_histogram_analysis import analyze as LSBHistogramAnalysis
from .classic.rs_analysis import analyze as RSAnalysis
from .cnn.cnn_detector import CNNDetector
from .cnn.noiseprint_extractor import  NoiseprintExtractor
from .ml.logistic_regression_detector import LogisticRegressionDetector
from .ml.svm_detector import SVMDetector
from .statistical.cooccurrence_analysis import analyze as CoOccurrenceAnalysis
from .statistical.noise_residuals import analyze as NoiseResiduals
from .statistical.wavelet_analysis import analyze as WaveletAnalysis

class AnalyzeSteganoSingle:
    def __init__(self):
        # Klasyczne metody
        self.his_square = ChiSquareAnalysis()
        self.lsb_histogram = LSBHistogramAnalysis()
        self.rs = RSAnalysis()

        # Statystyczne metody
        self.cooccurrence = CoOccurrenceAnalysis()
        self.noise_residuals = NoiseResiduals()
        self.vivalet = WaveletAnalysis()

        # Modele ML i CNN
        self.logreg = LogisticRegressionDetector()
        self.svm = SVMDetector()
        self.cnn = CNNDetector()
        self.noiseprint = NoiseprintExtractor()

    def analyze(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nie udało się wczytać obrazu: {image_path}")

        results = {}

        # Klasyczne
        results["chi_square"] = self.his_square.analyze(image)
        results["lsb_histogram"] = self.lsb_histogram.analyze(image)
        results["rs_analysis"] = self.rs.analyze(image)

        # Statystyczne
        results["cooccurrence"] = self.cooccurrence.analyze(image)
        results["noise_residuals"] = self.noise_residuals.analyze(image)
        results["wavelet"] = self.vivalet.analyze(image)

        # Modele ML i CNN
        results["logistic_regression"] = self.logreg.analyze(image)
        results["svm"] = self.svm.analyze(image)
        results["cnn"] = self.cnn.analyze(image)
        results["noiseprint"] = self.noiseprint.analyze(image)

        return results
