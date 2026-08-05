import cv2
import numpy as np


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


class FaceDetector:
    """
    YuNet DNN face detector with stable track IDs.
    Also keeps raw boxes + 5-point landmarks for age/gender alignment.
    """

    def __init__(
        self,
        model_path,
        score_threshold=0.7,
        nms_threshold=0.3,
        hold_frames=8,
        detect_width=640,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.hold_frames = hold_frames
        # Run YuNet on a downscaled frame — FHD input is far too slow on CPU
        self.detect_width = int(detect_width)
        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            5000,
        )
        self._stable = []
        self._next_id = 1

    def detect_faces(self, frame):
        """Return track dicts: id, box, raw_box, landmarks, misses, visible, ..."""
        h, w = frame.shape[:2]
        scale = 1.0
        detect = frame
        if w > self.detect_width:
            scale = self.detect_width / float(w)
            detect = cv2.resize(
                frame,
                (self.detect_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        dh, dw = detect.shape[:2]
        self._detector.setInputSize((dw, dh))
        _, faces = self._detector.detect(detect)
        inv = 1.0 / scale

        detections = []
        if faces is not None:
            for face in faces:
                score = float(face[-1])
                if score < self.score_threshold:
                    continue

                x, y, fw, fh = face[:4].astype(np.float32)
                # Map boxes/landmarks back to full-resolution frame
                x = int(round(x * inv))
                y = int(round(y * inv))
                fw = int(round(fw * inv))
                fh = int(round(fh * inv))
                x = max(0, x)
                y = max(0, y)
                fw = max(1, min(fw, w - x))
                fh = max(1, min(fh, h - y))
                if fw < 40 or fh < 40:
                    continue
                ratio = fw / float(fh)
                if ratio < 0.6 or ratio > 1.6:
                    continue

                # YuNet: right eye, left eye, nose, right mouth, left mouth
                landmarks = face[4:14].reshape(5, 2).astype(np.float32) * inv
                detections.append({
                    "box": (x, y, fw, fh),
                    "landmarks": landmarks,
                    "score": score,
                })

        return self._track(detections)

    def _new_track(self, det):
        track = {
            "id": self._next_id,
            "box": det["box"],
            "raw_box": det["box"],
            "landmarks": det["landmarks"],
            "misses": 0,
            "visible": True,
            "gender": None,
            "age": None,
            "emotion": None,
            "emotion_scores": None,
            "age_hist": np.zeros(8, dtype=np.float64),
            "gender_hist": np.zeros(2, dtype=np.float64),
            "age_value": None,
            "age_samples": [],
            "age_locked": None,
        }
        self._next_id += 1
        return track

    def _track(self, detections):
        matched = []
        used = set()

        for track in self._stable:
            best_i, best = -1, 0.0
            for i, det in enumerate(detections):
                if i in used:
                    continue
                score = _iou(track["box"], det["box"])
                if score > best:
                    best_i, best = i, score

            if best_i >= 0 and best >= 0.2:
                used.add(best_i)
                det = detections[best_i]
                px, py, pw, ph = track["box"]
                dx, dy, dw, dh = det["box"]
                track["box"] = (
                    int(0.65 * px + 0.35 * dx),
                    int(0.65 * py + 0.35 * dy),
                    int(0.65 * pw + 0.35 * dw),
                    int(0.65 * ph + 0.35 * dh),
                )
                # Keep unsmoothed geometry for model crops
                track["raw_box"] = det["box"]
                track["landmarks"] = det["landmarks"]
                track["misses"] = 0
                track["visible"] = True
                matched.append(track)
            else:
                track["misses"] += 1
                track["visible"] = False
                if track["misses"] <= self.hold_frames:
                    matched.append(track)

        for i, det in enumerate(detections):
            if i in used:
                continue
            if any(_iou(det["box"], t["box"]) >= 0.2 for t in matched):
                continue
            matched.append(self._new_track(det))

        self._stable = matched
        return self._stable
