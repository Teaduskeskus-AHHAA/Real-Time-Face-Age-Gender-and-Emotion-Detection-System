import cv2
import numpy as np
import sys
import threading
import queue
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
        # Display loop stays light; heavy models run on a worker thread
        self._infer_every = 10
        self._detect_every = 2
        self._frame_i = 0
        self._debug_tile = 160
        self._last_tracks = []
        self._infer_q = queue.Queue(maxsize=1)
        self._result_q = queue.Queue()
        self._worker_stop = threading.Event()
        self._worker = None

    def _update_age_gender(self, track, frame, other_boxes):
        box = track.get("raw_box") or track["box"]
        result = self.age_gender_detector.predict_age_gender(
            frame,
            box,
            landmarks=None,
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

    def _infer_worker(self):
        """Run age/gender + emotion off the display thread so the TV feed stays smooth."""
        while not self._worker_stop.is_set():
            try:
                job = self._infer_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if job is None:
                break

            track_id, frame, box, other_boxes, want_debug = job
            # Temporary track shell so existing update helpers can fill fields
            shell = {
                "id": track_id,
                "box": box,
                "raw_box": box,
                "gender_hist": np.zeros(2, dtype=np.float64),
                "age_samples": [],
                "age": None,
                "gender": None,
                "emotion": None,
                "emotion_scores": None,
                "age_value": None,
                "age_locked": None,
            }
            try:
                self._update_age_gender(shell, frame, other_boxes)
                self._update_emotion(shell, frame)
                self._result_q.put({
                    "id": track_id,
                    "gender_hist": shell["gender_hist"],
                    "gender": shell["gender"],
                    "age_samples": list(shell.get("age_samples") or []),
                    "age_value": shell.get("age_value"),
                    "age": shell.get("age"),
                    "emotion": shell.get("emotion"),
                    "emotion_scores": shell.get("emotion_scores"),
                    "debug_age_gender": shell.get("debug_age_gender") if want_debug else None,
                    "debug_emotion": shell.get("debug_emotion") if want_debug else None,
                })
            except Exception as exc:
                print(f"Inference worker error: {exc}")

    def _apply_infer_results(self, tracks):
        by_id = {t["id"]: t for t in tracks}
        while True:
            try:
                result = self._result_q.get_nowait()
            except queue.Empty:
                break
            track = by_id.get(result["id"])
            if track is None:
                continue

            if result.get("gender_hist") is not None and result.get("gender") is not None:
                # Merge with live history so async results don't thrash labels
                track["gender_hist"] = 0.5 * track["gender_hist"] + 0.5 * result["gender_hist"]
                track["gender"] = self.age_gender_detector.gender_labels[
                    int(np.argmax(track["gender_hist"]))
                ]

            if result.get("age_value") is not None:
                samples = track.setdefault("age_samples", [])
                samples.append(float(result["age_value"]))
                if len(samples) > 20:
                    del samples[:-20]
                track["age_value"] = float(np.median(samples))
                track["age"] = str(int(round(track["age_value"])))

            if result.get("emotion_scores") is not None:
                scores = result["emotion_scores"]
                if track["emotion_scores"] is None:
                    track["emotion_scores"] = dict(scores)
                else:
                    track["emotion_scores"] = {
                        k: 0.55 * track["emotion_scores"].get(k, v) + 0.45 * v
                        for k, v in scores.items()
                    }
                track["emotion"] = max(track["emotion_scores"], key=track["emotion_scores"].get)

            if result.get("debug_age_gender") is not None:
                track["debug_age_gender"] = result["debug_age_gender"]
            if result.get("debug_emotion") is not None:
                track["debug_emotion"] = result["debug_emotion"]

    def _schedule_inference(self, tracks, frame):
        """Queue at most one face for the worker; drop if busy (keeps display realtime)."""
        if not self._infer_q.empty():
            return

        all_boxes = [t.get("raw_box") or t["box"] for t in tracks]
        # Prefer faces that still lack labels, then round-robin by frame
        pending = [t for t in tracks if t["visible"] and t.get("age") is None]
        if not pending:
            pending = [
                t for t in tracks
                if t["visible"]
                and self._frame_i % self._infer_every == t["id"] % self._infer_every
            ]
        if not pending:
            return

        track = pending[0]
        box = track.get("raw_box") or track["box"]
        other_boxes = [b for b in all_boxes if b is not box]
        # Copy frame so the display thread can keep mutating the live buffer
        job = (track["id"], frame.copy(), box, other_boxes, self.debug)
        try:
            self._infer_q.put_nowait(job)
        except queue.Full:
            pass

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
        """Open webcam at Full HD with the lowest-latency backend we can get."""
        backends = []
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        cap = None
        for backend in backends:
            trial = cv2.VideoCapture(0, backend)
            if trial.isOpened():
                cap = trial
                break
            trial.release()

        if cap is None or not cap.isOpened():
            raise RuntimeError("Could not open camera 0")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(min_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(min_height))
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

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
                "Fullscreen will stretch the native stream (no CPU upscale)."
            )
        return cap

    def _read_latest(self, cap, max_grab=3):
        """Drain a few buffered frames and return the newest (cuts TV glass-to-glass lag)."""
        frame = None
        for _ in range(max_grab):
            if not cap.grab():
                break
            ok, img = cap.retrieve()
            if ok and img is not None:
                frame = img
        if frame is None:
            ok, frame = cap.read()
            if not ok:
                return None
        return frame

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

        first = self._read_latest(cap)
        if first is None:
            cap.release()
            raise RuntimeError("Camera failed on startup frame")
        self._enter_fullscreen(window_name, first)

        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._infer_worker, name="infer", daemon=True)
        self._worker.start()

        try:
            while True:
                frame = self._read_latest(cap)
                if frame is None:
                    break

                self._frame_i += 1
                if self._frame_i <= 5:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                if self._frame_i % self._detect_every == 1 or not self._last_tracks:
                    tracks = self.face_detector.detect_faces(frame)
                    self._last_tracks = tracks
                else:
                    tracks = self._last_tracks

                self._apply_infer_results(tracks)
                self._schedule_inference(tracks, frame)

                occupied = []
                for track in tracks:
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
                if key == ord("q") or key == 27:
                    break
                if key == ord("f"):
                    fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if fullscreen > 0 else cv2.WINDOW_FULLSCREEN,
                    )
        finally:
            self._worker_stop.set()
            try:
                self._infer_q.put_nowait(None)
            except queue.Full:
                pass
            if self._worker is not None:
                self._worker.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()
