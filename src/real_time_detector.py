import cv2
import numpy as np
from src.detect_faces import FaceDetector
from src.age_gender_detection import AgeGenderDetector
from src.emotion_detection import EmotionDetector
from src.utils import draw_face_overlay


class RealTimeDetector:
    def __init__(
        self,
        face_model_path,
        age_gender_model_path,
        emotion_model_path,
        debug=False,
    ):
        self.face_detector = FaceDetector(face_model_path)
        self.age_gender_detector = AgeGenderDetector(age_gender_model_path)
        self.emotion_detector = EmotionDetector(emotion_model_path)
        self.debug = debug
        self._infer_every = 5  # MiVOLO is heavier; refresh a bit less often
        self._frame_i = 0
        self._debug_tile = 160

    def _update_age_gender(self, track, frame, other_boxes):
        box = track.get("raw_box") or track["box"]
        result = self.age_gender_detector.predict_age_gender(
            frame,
            box,
            landmarks=None,  # genderage was trained with rotation=0 bbox crops
            avoid_boxes=other_boxes,
            return_input=self.debug,
        )
        if self.debug:
            gender, age_label, gender_probs, age_value, net_input = result
            if net_input is not None:
                track["debug_age_gender"] = net_input
        else:
            gender, age_label, gender_probs, age_value = result

        if gender_probs is None or age_value is None:
            return

        track["gender_hist"] = 0.55 * track["gender_hist"] + 0.45 * gender_probs
        track["gender"] = self.age_gender_detector.gender_labels[int(np.argmax(track["gender_hist"]))]

        # Light median smoothing only — no gender-specific calibration hacks
        samples = track.setdefault("age_samples", [])
        age_value = float(age_value)
        if samples:
            center = float(np.median(samples))
            age_value = center + float(np.clip(age_value - center, -6.0, 6.0))
        samples.append(age_value)
        if len(samples) > 20:
            del samples[:-20]

        track["age_value"] = float(np.median(samples))
        track["age"] = str(int(round(track["age_value"])))
        track["age_locked"] = None

    def _update_emotion(self, track, frame):
        x, y, w, h = track["box"]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return

        result = self.emotion_detector.predict_emotion(face, return_input=self.debug)
        if self.debug:
            emotion, scores, net_input = result
            track["debug_emotion"] = net_input
        else:
            emotion, scores = result

        if track["emotion_scores"] is None:
            track["emotion_scores"] = dict(scores)
        else:
            track["emotion_scores"] = {
                k: 0.55 * track["emotion_scores"].get(k, v) + 0.45 * v
                for k, v in scores.items()
            }
        track["emotion"] = max(track["emotion_scores"], key=track["emotion_scores"].get)

    def _label_tile(self, image, lines):
        tile = cv2.resize(image, (self._debug_tile, self._debug_tile), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(tile, (0, 0), (self._debug_tile - 1, 34), (20, 20, 20), -1)
        for i, line in enumerate(lines):
            cv2.putText(
                tile,
                line,
                (6, 14 + i * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        return tile

    def _show_debug_inputs(self, tracks):
        tiles = []
        for track in tracks:
            age_img = track.get("debug_age_gender")
            emo_img = track.get("debug_emotion")
            if age_img is None and emo_img is None:
                continue

            age_label = track.get("age") or "?"
            gender_label = track.get("gender") or "?"
            emotion_label = track.get("emotion") or "?"

            if age_img is not None:
                tiles.append(self._label_tile(age_img, [f"id {track['id']} age/gender", f"{gender_label} {age_label}"]))
            if emo_img is not None:
                tiles.append(self._label_tile(emo_img, [f"id {track['id']} emotion", emotion_label]))

        if not tiles:
            blank = np.zeros((self._debug_tile, self._debug_tile * 2, 3), dtype=np.uint8)
            cv2.putText(blank, "no detector inputs yet", (12, self._debug_tile // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("Detector Inputs (debug)", blank)
            return

        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        row_imgs = []
        for r in range(rows):
            chunk = tiles[r * cols:(r + 1) * cols]
            while len(chunk) < cols:
                chunk.append(np.zeros((self._debug_tile, self._debug_tile, 3), dtype=np.uint8))
            row_imgs.append(np.hstack(chunk))
        mosaic = np.vstack(row_imgs)
        cv2.imshow("Detector Inputs (debug)", mosaic)

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self._frame_i += 1
            tracks = self.face_detector.detect_faces(frame)
            occupied = []
            all_boxes = [t.get("raw_box") or t["box"] for t in tracks]

            for track in tracks:
                should_infer = track["visible"] and (
                    track["age"] is None
                    or self._frame_i % self._infer_every == track["id"] % self._infer_every
                )
                if should_infer:
                    my_box = track.get("raw_box") or track["box"]
                    other_boxes = [b for b in all_boxes if b is not my_box]
                    self._update_age_gender(track, frame, other_boxes)
                    self._update_emotion(track, frame)

                if track["age"] is None or track["emotion_scores"] is None:
                    continue

                x, y, w, h = track["box"]
                occupied = draw_face_overlay(
                    frame,
                    x, y, w, h,
                    track["gender"],
                    track["age"],
                    track["emotion"],
                    track["emotion_scores"],
                    occupied,
                )

            cv2.imshow('Face, Age, Gender, and Emotion Detection', frame)
            if self.debug:
                self._show_debug_inputs(tracks)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
