# steganalysis/cnn/cnn_detector.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

class CNNDetector:
    name = "CNN Detector"
    model_path = "models/cnn_steg.h5"  # <-- ścieżka do wytrenowanego modelu CNN

    @classmethod
    def analyze(cls, image_path: str) -> float:
        model = load_model(cls.model_path)
        img = keras_image.load_img(image_path, target_size=(128, 128), color_mode="grayscale")
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prob = model.predict(img_array)[0][0]
        return float(prob)
