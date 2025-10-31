# steganalysis/ml/logistic_regression_detector.py
import joblib
import numpy as np
from skimage import io, feature

class LogisticRegressionDetector:
    name = "Logistic Regression"
    model_path = "models/logreg_steg.pkl"  # <-- model wytrenowany w Colabie

    @classmethod
    def analyze(cls, image_path: str) -> float:
        image = io.imread(image_path, as_gray=True)
        features = feature.hog(image)
        model = joblib.load(cls.model_path)
        prob = model.predict_proba([features])[0][1]
        return float(prob)
