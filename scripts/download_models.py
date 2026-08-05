"""Download runtime model assets that are not vendored in git."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = DATA / "face_detection_yunet_2023mar.onnx"

EMOTION_PATH = DATA / "emotion_model.hdf5"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")


def ensure_models(force: bool = False) -> None:
    if force or not YUNET_PATH.exists() or YUNET_PATH.stat().st_size < 10_000:
        download(YUNET_URL, YUNET_PATH)
    else:
        print(f"OK  {YUNET_PATH.name}")

    if not EMOTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMOTION_PATH}. Re-clone the repo or restore data/emotion_model.hdf5."
        )
    print(f"OK  {EMOTION_PATH.name}")

    print(
        "\nAge/gender weights (MiVOLO v2) download automatically from Hugging Face\n"
        "on first run: iitolstykh/mivolo_v2"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model files into data/")
    parser.add_argument("--force", action="store_true", help="Re-download YuNet even if present")
    args = parser.parse_args()
    ensure_models(force=args.force)


if __name__ == "__main__":
    main()
