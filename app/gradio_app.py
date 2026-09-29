"""Gradio inference app: upload an image, get faces + predicted keypoints back."""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facial_keypoints.inference import detect_and_predict, load_model
from facial_keypoints.viz import draw_keypoints_on_image

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None

    model = get_model()
    results = detect_and_predict(model, image)

    if not results:
        raise gr.Error("No faces detected. Try a clearer, more front-facing photo.")

    fig = draw_keypoints_on_image(image, results)
    fig.canvas.draw()
    rendered = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    import matplotlib.pyplot as plt

    plt.close(fig)
    return rendered


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "sample_images"
examples = [str(p) for p in sorted(EXAMPLES_DIR.glob("*.jpg"))] if EXAMPLES_DIR.exists() else []

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a photo with one or more faces"),
    outputs=gr.Image(type="numpy", label="Detected faces + predicted keypoints"),
    examples=examples if examples else None,
    title="Facial Keypoint Detection",
    description=(
        "Detects faces with a Haar cascade, then predicts 68 facial keypoints "
        "(eyes, eyebrows, nose, mouth, jaw) with a CNN trained from scratch. "
        "See the [repo README](https://github.com/adebowalep/facial-keypoint-detection) "
        "for accuracy numbers and known limitations."
    ),
)

if __name__ == "__main__":
    demo.launch()
