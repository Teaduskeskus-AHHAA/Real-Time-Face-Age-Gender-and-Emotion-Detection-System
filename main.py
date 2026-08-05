import argparse
from pathlib import Path

from scripts.download_models import ensure_models
from src.real_time_detector import RealTimeDetector

# Paths to model files
FACE_MODEL_PATH = 'data/face_detection_yunet_2023mar.onnx'
AGE_GENDER_MODEL_PATH = 'iitolstykh/mivolo_v2'
EMOTION_MODEL_PATH = 'data/emotion_model.hdf5'


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time face, age, gender, and emotion detection")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a second window with the image crops fed into age/gender and emotion models",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Make sure local assets exist (YuNet). MiVOLO pulls from Hugging Face on first use.
    if not Path(FACE_MODEL_PATH).exists():
        ensure_models()

    detector = RealTimeDetector(
        FACE_MODEL_PATH,
        AGE_GENDER_MODEL_PATH,
        EMOTION_MODEL_PATH,
        debug=args.debug,
    )
    detector.run()


if __name__ == "__main__":
    main()
