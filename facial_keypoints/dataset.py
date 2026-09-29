"""PyTorch Dataset for the Kaggle-derived facial keypoints CSV/image format."""

import os
from typing import Callable, Optional

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FacialKeypointsDataset(Dataset):
    """Loads (image, 68-keypoint) pairs from a CSV + image directory.

    Args:
        csv_file: path to a CSV with the image filename in column 0 and
            136 (x, y) keypoint values in the remaining columns.
        root_dir: directory containing the images referenced by the CSV.
        transform: optional callable applied to each
            ``{"image": ..., "keypoints": ...}`` sample.
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.key_pts_frame)

    def __getitem__(self, idx: int) -> dict:
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(image_name)

        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
