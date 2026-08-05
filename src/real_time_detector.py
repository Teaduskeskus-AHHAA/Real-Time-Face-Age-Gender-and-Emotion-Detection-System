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

    def _open_camera(self, min_width=1920, min_height=1080):
        """Open webcam at least at Full HD when the device supports it."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera 0")

        # MJPG often unlocks higher resolutions on USB webcams
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(min_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(min_height))
        # Prefer a snappy stream if the camera allows it
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Warm up and read actual negotiated size
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError("Camera opened but failed to read frames")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera mode: {width}x{height}")
        if width < min_width or height < min_height:
            print(
                f"Warning: camera negotiated below FHD ({width}x{height}). "
                "Frames will be upscaled to at least 1920x1080 for display."
            )
        return cap

    def _ensure_fhd(self, frame, min_width=1920, min_height=1080):
        """Upscale frames that come in below Full HD."""
        h, w = frame.shape[:2]
        if w >= min_width and h >= min_height:
            return frame
        scale = max(min_width / float(w), min_height / float(h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _screen_size(self):
        """Best-effort primary display size for fullscreen setup."""
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            if w > 0 and h > 0:
                return int(w), int(h)
        except Exception:
            pass
        return 1920, 1080

    def _enter_fullscreen(self, window_name, frame=None):
        """Force the OpenCV window into fullscreen (more reliable on Windows)."""
        screen_w, screen_h = self._screen_size()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 0, 0)
        cv2.resizeWindow(window_name, screen_w, screen_h)
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)

    def run(self):
        window_name = "Face, Age, Gender, and Emotion Detection"
        cap = self._open_camera(min_width=1920, min_height=1080)

        # Prime one frame so fullscreen attaches to a real window
        ret, first = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Camera failed on startup frame")
        first = self._ensure_fhd(first, min_width=1920, min_height=1080)
        self._enter_fullscreen(window_name, first)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self._ensure_fhd(frame, min_width=1920, min_height=1080)

                self._frame_i += 1
                # Re-assert fullscreen for the first frames (some backends ignore the first call)
                if self._frame_i <= 5:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

                cv2.imshow(window_name, frame)
                if self.debug:
                    self._show_debug_inputs(tracks)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or Esc
                    break
                if key == ord("f"):
                    fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if fullscreen > 0 else cv2.WINDOW_FULLSCREEN,
                    )
        finally:
            cap.release()
            cv2.destroyAllWindows()
