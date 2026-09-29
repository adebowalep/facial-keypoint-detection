"""Regression test for a BatchNorm instability found in the shipped checkpoint.

One channel in ``bn1`` had ``running_var == 0.0`` exactly, which caused
BatchNorm to divide by ~sqrt(eps) for that channel -- amplifying tiny
per-image deviations ~300x and cascading into predictions off by hundreds to
thousands of pixels on some (but not all) inputs. ``load_model`` floors
running_var to fix this without retraining (see the comment in
``facial_keypoints/inference.py``). This test fails loudly if that fix is
ever removed or a future checkpoint reintroduces the same defect.
"""

import numpy as np
import pytest
import torch

from facial_keypoints.config import DEFAULT_CHECKPOINT
from facial_keypoints.inference import (
    _BATCHNORM_RUNNING_VAR_FLOOR,
    load_model,
    predict_keypoints,
)

requires_checkpoint = pytest.mark.skipif(
    not DEFAULT_CHECKPOINT.exists(), reason="trained checkpoint not present"
)


@requires_checkpoint
def test_no_batchnorm_channel_has_near_zero_running_var():
    model = load_model()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.running_var.min().item() >= _BATCHNORM_RUNNING_VAR_FLOOR


@requires_checkpoint
def test_predictions_stay_within_a_sane_pixel_range(sample_face_roi):
    """A regression guard for the "predictions off by hundreds of pixels"
    failure mode: on any input, predicted keypoints should land within a
    generous multiple of the ROI size, not orders of magnitude beyond it.
    """
    model = load_model()
    h, w = sample_face_roi.shape[:2]
    box = (0, 0, w, h)

    result = predict_keypoints(model, sample_face_roi, box)

    margin = 3  # generous: real predictions stay within ~1x the box, not 3x
    assert result.keypoints[:, 0].min() > -margin * w
    assert result.keypoints[:, 0].max() < (1 + margin) * w
    assert result.keypoints[:, 1].min() > -margin * h
    assert result.keypoints[:, 1].max() < (1 + margin) * h
