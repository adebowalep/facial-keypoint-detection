"""Pixel-space accuracy metrics against a labeled keypoints CSV."""

from dataclasses import dataclass

import numpy as np

from .dataset import FacialKeypointsDataset
from .inference import predict_keypoints
from .model import Net


@dataclass
class KeypointMetrics:
    n_images: int
    mean_px_error: float
    median_px_error: float
    p95_px_error: float
    max_px_error: float

    def to_dict(self) -> dict:
        return {
            "n_images": self.n_images,
            "mean_px_error": round(self.mean_px_error, 3),
            "median_px_error": round(self.median_px_error, 3),
            "p95_px_error": round(self.p95_px_error, 3),
            "max_px_error": round(self.max_px_error, 3),
        }


def evaluate(model: Net, csv_file: str, root_dir: str) -> KeypointMetrics:
    """Run the full inference path (not the training-time transform pipeline)
    against every image in a labeled CSV, and report per-keypoint pixel error.

    Uses the same ``predict_keypoints`` code path as the CLI/Gradio app, so
    these numbers reflect real inference behavior, not just training-loop loss.
    """
    dataset = FacialKeypointsDataset(csv_file=csv_file, root_dir=root_dir, transform=None)

    errors = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"]
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        h, w = image.shape[:2]
        result = predict_keypoints(model, image, (0, 0, w, h))
        errors.append(np.abs(result.keypoints - sample["keypoints"]).mean())

    errors_arr = np.array(errors)
    return KeypointMetrics(
        n_images=len(errors_arr),
        mean_px_error=float(errors_arr.mean()),
        median_px_error=float(np.median(errors_arr)),
        p95_px_error=float(np.percentile(errors_arr, 95)),
        max_px_error=float(errors_arr.max()),
    )
