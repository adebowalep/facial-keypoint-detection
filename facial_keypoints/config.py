"""Central constants and paths for the facial_keypoints package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "keypoints_model_1_fp16.pt"
HAAR_CASCADE_PATH = PACKAGE_DIR / "assets" / "haarcascade_frontalface_default.xml"

# keypoint normalization used during training: (pixel - KEYPOINT_MEAN) / KEYPOINT_SCALE
KEYPOINT_MEAN = 100.0
KEYPOINT_SCALE = 50.0

# image normalization used during training: pixel / IMAGE_SCALE, range [0, 1]
IMAGE_SCALE = 255.0
