from facial_keypoints.detect import detect_faces


def test_detect_faces_on_synthetic_image_returns_list(sample_image_rgb):
    # a random-noise image won't contain a real face; this just checks the
    # cascade runs end-to-end and returns a well-formed (possibly empty) list
    results = detect_faces(sample_image_rgb)
    assert isinstance(results, list)
    for box in results:
        assert len(box) == 4
        assert all(isinstance(v, int) for v in box)
