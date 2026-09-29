"""Image/keypoint transforms shared by training and inference.

Each transform operates on a ``{"image": np.ndarray, "keypoints": np.ndarray}``
sample dict, mirroring the torchvision ``Dataset``/``transforms.Compose``
convention, so the same pipeline can be reused for the training
``DataLoader`` and for single-image inference.
"""

from typing import Tuple, TypedDict, Union

import cv2
import numpy as np
import torch

from .config import IMAGE_SCALE, KEYPOINT_MEAN, KEYPOINT_SCALE


class Sample(TypedDict):
    image: np.ndarray
    keypoints: np.ndarray


class Normalize:
    """Convert a color image to grayscale and normalize image + keypoints."""

    def __call__(self, sample: Sample) -> Sample:
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if image_copy.ndim == 3 and image_copy.shape[2] == 3:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

        image_copy = image_copy / IMAGE_SCALE
        key_pts_copy = (key_pts_copy - KEYPOINT_MEAN) / KEYPOINT_SCALE

        return {"image": image_copy, "keypoints": key_pts_copy}


class Rescale:
    """Rescale the image (and keypoints) to a given size.

    Args:
        output_size: if an int, the smaller edge is matched to this value,
            preserving aspect ratio; if a tuple, output is matched exactly.
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]]) -> None:
        self.output_size = output_size

    def __call__(self, sample: Sample) -> Sample:
        image, key_pts = sample["image"], sample["keypoints"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        key_pts = key_pts * [new_w / w, new_h / h]

        return {"image": img, "keypoints": key_pts}


class RandomCrop:
    """Randomly crop the image (and shift keypoints) to a given size."""

    def __init__(self, output_size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample: Sample) -> Sample:
        image, key_pts = sample["image"], sample["keypoints"]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        key_pts = key_pts - [left, top]

        return {"image": image, "keypoints": key_pts}


class ToTensor:
    """Convert ndarrays in a sample to torch Tensors, HWC -> CHW."""

    def __call__(self, sample: Sample) -> dict:
        image, key_pts = sample["image"], sample["keypoints"]

        if image.ndim == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(image),
            "keypoints": torch.from_numpy(key_pts),
        }
