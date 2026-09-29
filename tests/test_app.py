import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from facial_keypoints.config import DEFAULT_CHECKPOINT

requires_checkpoint = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.exists(), reason="trained checkpoint not present"
)


@requires_checkpoint
def test_predict_returns_rendered_image_for_a_real_photo():
    import cv2

    from gradio_app import predict

    image_dir = Path(__file__).resolve().parent.parent / "docs" / "sample_images"
    image_path = image_dir / "obamas.jpg"
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    output = predict(image_rgb)

    assert isinstance(output, np.ndarray)
    assert output.ndim == 3
    assert output.shape[2] == 3


@requires_checkpoint
def test_predict_returns_none_for_none_input():
    from gradio_app import predict

    assert predict(None) is None
