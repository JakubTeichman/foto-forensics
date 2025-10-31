# steganalysis/ml/svm_detector.py
import joblib
import numpy as np
from skimage import io, feature

class SVMDetector:
    name = "SVM Detector"
    model_path = "models/svm_steg.pkl"  # <-- tutaj wkleisz swój wytrenowany model z Colaba

    @classmethod
    def analyze(cls, image_path: str) -> float:
        image = io.imread(image_path, as_gray=True)
        hog_features = feature.hog(image)
        model = joblib.load(cls.model_path)
        prob = model.predict_proba([hog_features])[0][1]
        return float(prob)
