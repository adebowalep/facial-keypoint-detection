"""Plotting helpers for keypoint predictions."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .inference import FaceKeypoints


def draw_keypoints_on_image(
    image_rgb: np.ndarray,
    results: List[FaceKeypoints],
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Return a matplotlib Figure with predicted keypoints drawn over each face."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_rgb)

    for result in results:
        x, y, w, h = result.box
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2)
        )
        pts = result.keypoints
        ax.scatter(x + pts[:, 0], y + pts[:, 1], s=12, marker=".", c="magenta")

    ax.axis("off")
    fig.tight_layout()
    return fig
