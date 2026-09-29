import numpy as np
import pytest

from facial_keypoints.config import DEFAULT_CHECKPOINT
from facial_keypoints.inference import FaceKeypoints, load_model, predict_keypoints
from facial_keypoints.model import NUM_KEYPOINTS

requires_checkpoint = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.exists(),
    reason="trained checkpoint not present (expected at models/keypoints_model_1_fp16.pt)",
)


@requires_checkpoint
def test_load_model_returns_net_in_eval_mode():
    model = load_model()
    assert model.training is False


@requires_checkpoint
def test_predict_keypoints_on_synthetic_face(sample_face_roi):
    model = load_model()
    box = (0, 0, sample_face_roi.shape[1], sample_face_roi.shape[0])

    result = predict_keypoints(model, sample_face_roi, box)

    assert isinstance(result, FaceKeypoints)
    assert result.keypoints.shape == (NUM_KEYPOINTS, 2)
    assert np.isfinite(result.keypoints).all()


@requires_checkpoint
def test_predicted_keypoints_land_within_roi_bounds(sample_face_roi):
    model = load_model()
    h, w = sample_face_roi.shape[:2]
    box = (0, 0, w, h)

    result = predict_keypoints(model, sample_face_roi, box)

    # a well-behaved model on a roughly face-shaped crop should keep most
    # keypoints near the crop, not off in unrelated coordinate space;
    # this is a loose sanity bound, not an accuracy claim
    assert result.keypoints[:, 0].mean() == pytest.approx(w / 2, abs=w)
    assert result.keypoints[:, 1].mean() == pytest.approx(h / 2, abs=h)
