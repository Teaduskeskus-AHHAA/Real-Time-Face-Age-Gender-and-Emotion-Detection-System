import numpy as np
from tensorflow.keras.models import load_model
import cv2


class EmotionDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)
        self.labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def _to_probabilities(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if np.min(raw) >= 0 and abs(np.sum(raw) - 1.0) < 0.05:
            return raw
        shifted = raw - np.max(raw)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def predict_emotion(self, face, return_input=False):
        """
        Returns:
            emotion, scores
            and optionally the 64x64 grayscale tensor visualized as BGR
        """
        gray = cv2.resize(face, (64, 64))
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        model_input = gray.astype("float32") / 255.0
        model_input = np.expand_dims(model_input, axis=-1)
        model_input = np.expand_dims(model_input, axis=0)

        raw = self.model.predict(model_input, verbose=0)[0]
        probs = self._to_probabilities(raw)
        emotion_idx = int(np.argmax(probs))
        scores = {label: float(probs[i]) for i, label in enumerate(self.labels)}
        emotion = self.labels[emotion_idx]

        if return_input:
            # Visualize the exact grayscale patch the model receives
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return emotion, scores, vis
        return emotion, scores
