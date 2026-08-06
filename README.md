# Real-Time Face, Age, Gender, and Emotion Detection

Webcam app that detects faces in real time and overlays **age**, **gender**, and **emotion** (with per-emotion likelihood bars).

| Stage | Model |
|---|---|
| Face detection | OpenCV **YuNet** (`data/face_detection_yunet_2023mar.onnx`) |
| Age + gender | **MiVOLO v2** ([`iitolstykh/mivolo_v2`](https://huggingface.co/iitolstykh/mivolo_v2)) |
| Emotion | Keras model in `data/emotion_model.hdf5` |

## Quick start — Windows (Desktop)

1. Install **Python 3.11 64-bit** from [python.org](https://www.python.org/downloads/)  
   - Enable **Add python.exe to PATH**
2. Clone this repo to your Desktop (full folder).
3. Double-click **`run.bat`**

The bat file will create `venv`, install **prebuilt wheels only** (no Visual Studio), download YuNet if needed, and start the app.

```bat
cd %USERPROFILE%\Desktop\Real-Time-Face-Age-Gender-and-Emotion-Detection-System
run.bat
```

Debug crops:

```bat
run.bat --debug
```

Press **`q`** in the video window to quit.

If install was half-broken before: delete the `venv` folder, then run `run.bat` again.

## Quick start — macOS / Linux

```bash
git clone <your-repo-url>
cd Real-Time-Face-Age-Gender-and-Emotion-Detection-System

python3 -m venv venv
source venv/bin/activate

pip install -U pip wheel
pip install "setuptools>=68,<81"
pip install opencv-python==4.10.0.84
pip install opencv-contrib-python==4.10.0.84
pip install -r requirements.txt

python scripts/download_models.py
python main.py
```

## Important Windows notes

- Use **Python 3.10 or 3.11 only**. 3.12/3.13 often lack matching TensorFlow/numpy wheels and pip tries to compile (needs VS — we avoid that).
- `run.bat` installs packages with `--only-binary` where possible so **Visual Studio Build Tools are not required**.
- **NVIDIA GPU (Quadro / GeForce):** `pip install torch` from PyPI is CPU-only. `run.bat` detects `nvidia-smi` and installs the **CUDA 12.1** PyTorch wheel. If you already have a CPU venv, re-run `run.bat` (it will upgrade torch) or:

```bat
venv\Scripts\activate
pip uninstall -y torch torchvision
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

You should see `Age/gender device: cuda` and your GPU name at startup.
- OpenCV is pinned to **`4.10.0.84`** for both `opencv-python` and `opencv-contrib-python`.
- The **`mivolo/`** folder is **vendored in this repo** (not a git submodule). Do not add a nested clone under `third_party/`.
- MiVOLO **weights** still download from Hugging Face on first `python main.py` (needs network once).

## Project layout

```
run.bat                      # Windows one-click setup + launch
main.py                      # Entry point (--debug optional)
requirements.txt             # Pinned deps (wheels; no git installs)
mivolo/                      # Vendored MiVOLO Python package
scripts/download_models.py   # Downloads YuNet into data/
data/
  face_detection_yunet_2023mar.onnx
  emotion_model.hdf5
src/
  detect_faces.py
  age_gender_detection.py
  emotion_detection.py
  real_time_detector.py
  utils.py
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `pip install -r requirements` fails | Delete `venv`, use Python **3.11 64-bit**, run `run.bat`. It installs packages one-by-one as wheels — no Visual Studio, no `git+` MiVOLO install. |
| `numpy` build / Visual Studio errors | Same fix: Python 3.11 + delete `venv` + `run.bat`. |
| `No module named mivolo` | Ensure the `mivolo/` folder exists in the repo root (re-clone if missing). |
| OpenCV / `FaceDetectorYN` errors | Same OpenCV version for both packages: `4.10.0.84`. Re-run `run.bat` after deleting `venv`. |
| Missing YuNet | `python scripts\download_models.py` |
| Slow first launch | Hugging Face download of MiVOLO weights |
| Camera not opening | Close other webcam apps |

## Privacy

Use only with consent. Age/gender/emotion outputs are approximate model predictions.

## License

See [LICENSE](LICENSE). Third-party models (YuNet, MiVOLO, emotion weights) and the vendored `mivolo` package have their own licenses.
