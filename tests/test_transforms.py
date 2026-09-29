import numpy as np
import torch

from facial_keypoints.transforms import Normalize, RandomCrop, Rescale, ToTensor


def make_sample(h=280, w=280, n_points=68):
    image = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    keypoints = np.random.uniform(0, min(h, w), size=(n_points, 2))
    return {"image": image, "keypoints": keypoints}


def test_normalize_converts_to_grayscale_and_scales_range():
    sample = make_sample()
    out = Normalize()(sample)
    assert out["image"].ndim == 2
    assert out["image"].max() <= 1.0
    assert out["image"].min() >= 0.0


def test_normalize_centers_keypoints():
    sample = make_sample()
    out = Normalize()(sample)
    # keypoints were in [0, 280); after (pt - 100) / 50 they should roughly span [-2, 3.6]
    assert out["keypoints"].max() < 4
    assert out["keypoints"].min() > -3


def test_rescale_int_preserves_aspect_ratio():
    # smaller edge (h=200) is matched to output_size; w scales proportionally
    sample = make_sample(h=200, w=400)
    out = Rescale(100)(sample)
    h, w = out["image"].shape[:2]
    assert h == 100
    assert w == 200


def test_rescale_scales_keypoints_consistently():
    sample = make_sample(h=200, w=200, n_points=1)
    sample["keypoints"] = np.array([[100.0, 50.0]])
    out = Rescale(100)(sample)
    np.testing.assert_allclose(out["keypoints"], [[50.0, 25.0]])


def test_random_crop_output_size():
    sample = make_sample(h=250, w=250)
    out = RandomCrop(224)(sample)
    assert out["image"].shape[:2] == (224, 224)
    assert out["keypoints"].shape == (68, 2)


def test_to_tensor_shapes_and_dtype():
    sample = {
        "image": np.random.rand(224, 224).astype(np.float64),
        "keypoints": np.random.rand(68, 2),
    }
    out = ToTensor()(sample)
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (1, 224, 224)
    assert isinstance(out["keypoints"], torch.Tensor)


def test_pipeline_end_to_end_matches_model_input_shape():
    sample = make_sample(h=280, w=280)
    pipeline = [Rescale(250), RandomCrop(224), Normalize(), ToTensor()]
    for t in pipeline:
        sample = t(sample)
    assert sample["image"].shape == (1, 224, 224)
