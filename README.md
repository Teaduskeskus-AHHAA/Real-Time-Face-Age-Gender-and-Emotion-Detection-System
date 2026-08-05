# Real-Time Face, Age, Gender, and Emotion Detection

Webcam app that detects faces in real time and overlays **age**, **gender**, and **emotion** (with per-emotion likelihood bars).

| Stage | Model |
|---|---|
| Face detection | OpenCV **YuNet** (`face_detection_yunet_2023mar.onnx`) |
| Age + gender | **MiVOLO v2** ([`iitolstykh/mivolo_v2`](https://huggingface.co/iitolstykh/mivolo_v2)) |
| Emotion | Keras model in `data/emotion_model.hdf5` |

## Quick start (clone → run)

```bash
git clone <your-repo-url>
cd Real-Time-Face-Age-Gender-and-Emotion-Detection-System

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Pin OpenCV first (both packages at the same version)
pip install opencv-python==4.10.0.84
pip install opencv-contrib-python==4.10.0.84

# Remaining deps (includes MiVOLO from GitHub)
pip install -r requirements.txt

# Local face model (YuNet). Safe to re-run.
python scripts/download_models.py

python main.py
```

Press **`q`** in the video window to quit.

Debug crops (what age/gender and emotion nets see):

```bash
python main.py --debug
```

### First run notes

- **MiVOLO v2** weights download from Hugging Face the first time you start the app (needs network).
- If `pip install -r requirements.txt` fails on the `mivolo` git line, install it explicitly:

```bash
pip install 'setuptools>=68,<81'
pip install --no-build-isolation "git+https://github.com/WildChlamydia/MiVOLO.git"
```

- Installing **both** `opencv-python` and `opencv-contrib-python` can confuse `cv2` if versions differ. Keep them on **`4.10.0.84`**. If imports break, uninstall both and reinstall in that order again.

## Requirements

- Python **3.10 or 3.11** recommended (TensorFlow + torch)
- Webcam
- Network on first run (Hugging Face model pull)

Pinned / important packages (see `requirements.txt`):

- `opencv-python==4.10.0.84`
- `opencv-contrib-python==4.10.0.84`
- `transformers==4.51.0`
- `setuptools>=68,<81` (needed to build MiVOLO)
- `mivolo` from `git+https://github.com/WildChlamydia/MiVOLO.git`

## Project layout

```
main.py                      # Entry point (--debug optional)
requirements.txt
scripts/download_models.py   # Downloads YuNet into data/
data/
  face_detection_yunet_2023mar.onnx   # Face detector (tracked / downloadable)
  emotion_model.hdf5                  # Emotion classifier (tracked)
src/
  detect_faces.py            # YuNet + multi-face tracking
  age_gender_detection.py    # MiVOLO v2 age/gender
  emotion_detection.py       # Emotion probabilities
  real_time_detector.py      # Webcam loop + overlays
  utils.py                   # Modern on-screen UI
```

Large / obsolete weights (`*.caffemodel`, extra `*.onnx` experiments) are **gitignored** and not required.

## How it works

1. Capture frames from camera `0`.
2. YuNet finds faces and keeps stable track IDs across frames.
3. For each live face:
   - MiVOLO estimates **age** (years) and **gender** from a face crop plus an approximate upper-body crop.
   - The emotion network returns scores for Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
4. A panel is drawn per face (age, gender, fixed-order emotion bars).
5. Exit with `q`.

Processing stays on-device in memory for display; the stock app does not save or upload frames. The first MiVOLO load does download weights from Hugging Face.

## Troubleshooting

| Issue | Fix |
|---|---|
| `FaceDetectorYN` / OpenCV DNN errors | Reinstall OpenCV 4.10.0.84 (both wheels, same version) |
| `No module named mivolo` | `pip install --no-build-isolation "git+https://github.com/WildChlamydia/MiVOLO.git"` with `setuptools<81` |
| `pkg_resources` / MiVOLO build fails | `pip install 'setuptools>=68,<81'` then retry MiVOLO install |
| Missing YuNet file | `python scripts/download_models.py` |
| Slow first launch | Normal while Hugging Face caches MiVOLO |
| Camera not opening | Close other apps using the webcam; try another index in `VideoCapture` if needed |

## Privacy

Use only with consent. Predictions (especially binary gender and age) are approximate model outputs, not ground truth.

## License

See [LICENSE](LICENSE). Third-party models (YuNet, MiVOLO, emotion weights) have their own licenses—respect those when redistributing weights.
