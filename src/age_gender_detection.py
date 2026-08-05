import cv2
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor


class AgeGenderDetector:
    """
    MiVOLO v2 (iitolstykh/mivolo_v2) — SOTA age/gender without hand-tuned bias curves.
    Uses face crop + an approximate upper-body crop (model expects both streams).
    """

    def __init__(self, model_path="iitolstykh/mivolo_v2"):
        self.device = torch.device("cpu")
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Match MiVOLO id2label: 0=male, 1=female
        self.gender_labels = [
            self.config.gender_id2label[0].capitalize(),
            self.config.gender_id2label[1].capitalize(),
        ]

    def _face_crop(self, frame, box, pad=0.10):
        x, y, w, h = [float(v) for v in box]
        fh, fw = frame.shape[:2]
        x0 = max(0, int(x - pad * w))
        y0 = max(0, int(y - pad * h))
        x1 = min(fw, int(x + w + pad * w))
        y1 = min(fh, int(y + h + pad * h))
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1]

    def _body_crop(self, frame, box):
        """Approximate person crop from the face box when no person detector is available."""
        x, y, w, h = [float(v) for v in box]
        fh, fw = frame.shape[:2]
        x0 = max(0, int(x - 0.7 * w))
        x1 = min(fw, int(x + w + 0.7 * w))
        y0 = max(0, int(y - 0.4 * h))
        y1 = min(fh, int(y + h + 2.4 * h))
        if x1 <= x0 or y1 <= y0:
            return self._face_crop(frame, box, pad=0.2)
        return frame[y0:y1, x0:x1]

    @torch.inference_mode()
    def predict_age_gender(self, frame, box, landmarks=None, avoid_boxes=None, return_input=False):
        face = self._face_crop(frame, box)
        body = self._body_crop(frame, box)
        if face is None or face.size == 0:
            empty = (None,) if return_input else ()
            return ("Unknown", "?", None, None) + empty

        faces_input = self.processor(images=[face])["pixel_values"].to(self.device)
        body_input = self.processor(images=[body])["pixel_values"].to(self.device)

        output = self.model(faces_input=faces_input, body_input=body_input)

        age_value = float(np.clip(output.age_output[0].item(), 0.0, 100.0))
        age_label = str(int(round(age_value)))

        raw_gender = output.raw_gender_output[0].detach().cpu().numpy().astype(np.float64)
        e = np.exp(raw_gender - np.max(raw_gender))
        gender_probs = e / e.sum()  # [Male, Female] per MiVOLO ordering
        gender = self.gender_labels[int(np.argmax(gender_probs))]

        if return_input:
            # Show the face chip used for the face stream
            debug = cv2.resize(face, (224, 224))
            return gender, age_label, gender_probs, age_value, debug
        return gender, age_label, gender_probs, age_value
