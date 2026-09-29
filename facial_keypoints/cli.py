"""Command-line demo: detect faces and predict keypoints on an image file."""

import argparse
from pathlib import Path

import cv2

from .config import DEFAULT_CHECKPOINT
from .inference import detect_and_predict, load_model
from .viz import draw_keypoints_on_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces and predict facial keypoints.")
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("keypoints_output.png"))
    args = parser.parse_args()

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model = load_model(str(args.checkpoint))
    results = detect_and_predict(model, image_rgb)

    print(f"Detected {len(results)} face(s).")

    fig = draw_keypoints_on_image(image_rgb, results)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
