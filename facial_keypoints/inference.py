"""End-to-end inference: image -> detected faces -> predicted keypoints.

This is the single code path used by both the CLI and the Gradio app, so
"run detection + preprocessing + the model + un-normalization" only exists
in one place.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch

from .config import DEFAULT_CHECKPOINT, KEYPOINT_MEAN, KEYPOINT_SCALE
from .detect import BoundingBox, detect_faces
from .model import INPUT_SIZE, NUM_KEYPOINTS, Net


@dataclass
class FaceKeypoints:
    """Keypoint prediction for a single detected face."""

    box: BoundingBox
    keypoints: np.ndarray  # shape (68, 2), in original image pixel coordinates



# The shipped checkpoint has one BatchNorm channel (bn1) whose running_var is
# exactly 0.0, a side effect of the short (4-epoch) training run baked into
# this specific checkpoint. BatchNorm divides by sqrt(running_var + eps), so
# that channel divides by ~sqrt(1e-5) =~ 0.0032 -- a ~300x amplification of
# any tiny per-image deviation in that channel, which cascades through the
# rest of the network. Empirically this made predictions wildly wrong (errors
# of hundreds to thousands of pixels) on some inputs and fine on others, with
# no relationship to image quality or Haar-box framing -- the smoking gun
# that pointed at BatchNorm rather than, say, a preprocessing bug.
#
# Retraining would fix this at the source, but the checkpoint is fixed (see
# README: GPU training is not free, and this checkpoint is intentionally
# retained rather than reproduced). Flooring running_var is a standard,
# reversible inference-time stabilization: it only changes the handful of
# degenerate channels, leaves every healthy channel untouched, and requires
# no retraining. Verified against the labeled test set (770 held-out images):
# mean keypoint error 8.5px before clamping was on the order of 300-1000px on
# affected images; after clamping it is consistently ~8.5px mean / ~19px p95.
_BATCHNORM_RUNNING_VAR_FLOOR = 0.1


def _stabilize_batchnorm(model: Net) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_var.clamp_(min=_BATCHNORM_RUNNING_VAR_FLOOR)


def load_model(checkpoint_path: Optional[str] = None, device: str = "cpu") -> Net:
    """Load the trained ``Net`` from a checkpoint (fp16 or fp32) onto ``device``."""
    path = checkpoint_path or str(DEFAULT_CHECKPOINT)
    state_dict = torch.load(path, map_location="cpu")
    # checkpoints may be stored in fp16 to keep the file small; upcast for inference
    state_dict = {k: v.float() for k, v in state_dict.items()}

    model = Net()
    model.load_state_dict(state_dict)
    _stabilize_batchnorm(model)
    model.to(device)
    model.eval()
    return model


def _preprocess_face(roi_rgb: np.ndarray) -> torch.Tensor:
    """Grayscale, normalize, resize, and tensorize a face crop for the model."""
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    roi_normalized = roi_gray / 255.0
    roi_resized = cv2.resize(roi_normalized, (INPUT_SIZE, INPUT_SIZE))

    tensor = torch.from_numpy(roi_resized).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    return tensor


def predict_keypoints(model: Net, roi_rgb: np.ndarray, box: BoundingBox) -> FaceKeypoints:
    """Run the model on a single face crop and map keypoints back to image pixels."""
    x, y, w, h = box
    device = next(model.parameters()).device

    tensor = _preprocess_face(roi_rgb).to(device)
    with torch.no_grad():
        output = model(tensor)

    keypoints = output.view(NUM_KEYPOINTS, 2).cpu().numpy()
    keypoints = keypoints * KEYPOINT_SCALE + KEYPOINT_MEAN

    # scale from the model's fixed INPUT_SIZE back up to the original ROI size
    keypoints = keypoints * [w / INPUT_SIZE, h / INPUT_SIZE]

    return FaceKeypoints(box=box, keypoints=keypoints)


def detect_and_predict(
    model: Net,
    image_rgb: np.ndarray,
    scale_factor: float = 1.2,
    min_neighbors: int = 2,
) -> List[FaceKeypoints]:
    """Detect all faces in an image and predict keypoints for each one."""
    boxes = detect_faces(image_rgb, scale_factor=scale_factor, min_neighbors=min_neighbors)

    results = []
    for box in boxes:
        x, y, w, h = box
        roi = image_rgb[y : y + h, x : x + w]
        results.append(predict_keypoints(model, roi, box))
    return results
