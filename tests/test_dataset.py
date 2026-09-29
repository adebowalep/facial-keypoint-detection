import numpy as np
import pandas as pd
import pytest

from facial_keypoints.dataset import FacialKeypointsDataset


@pytest.fixture
def toy_csv_dir(tmp_path):
    """A tiny 2-image CSV + PNG fixture, built on the fly."""
    import matplotlib.image as mpimg

    root = tmp_path / "images"
    root.mkdir()

    rows = []
    for i, name in enumerate(["a.png", "b.png"]):
        img = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
        mpimg.imsave(root / name, img)
        keypoints = np.random.uniform(0, 50, size=136)
        rows.append([name, *keypoints])

    columns = ["filename"] + [f"kp_{i}" for i in range(136)]
    df = pd.DataFrame(rows, columns=columns)
    csv_path = tmp_path / "keypoints.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, root


def test_dataset_length(toy_csv_dir):
    csv_path, root = toy_csv_dir
    ds = FacialKeypointsDataset(csv_file=str(csv_path), root_dir=str(root))
    assert len(ds) == 2


def test_dataset_item_shapes(toy_csv_dir):
    csv_path, root = toy_csv_dir
    ds = FacialKeypointsDataset(csv_file=str(csv_path), root_dir=str(root))
    sample = ds[0]
    assert sample["image"].shape == (50, 50, 3)
    assert sample["keypoints"].shape == (68, 2)


def test_dataset_applies_transform(toy_csv_dir):
    csv_path, root = toy_csv_dir
    calls = []

    def fake_transform(sample):
        calls.append(sample)
        return sample

    ds = FacialKeypointsDataset(csv_file=str(csv_path), root_dir=str(root), transform=fake_transform)
    _ = ds[0]
    assert len(calls) == 1
