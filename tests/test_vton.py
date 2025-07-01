import os
import logging
import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")
from skimage.transform import PiecewiseAffineTransform
from vton import VTONPipeline

# Skip heavy tests when requested or when heavy dependencies are missing.
SKIP_HEAVY = os.getenv("SKIP_HEAVY_TESTS") == "1"


@pytest.mark.skipif(SKIP_HEAVY, reason="Heavy tests are skipped")
def test_segment_size():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pipe = VTONPipeline()
    img = np.zeros((64, 32, 3), dtype=np.uint8)
    mask = pipe.segment(img)
    assert mask.shape == img.shape[:2]


def test_run_person_imread_failure(tmp_path):
    pipe = VTONPipeline.__new__(VTONPipeline)  # bypass heavy init
    out = tmp_path / "out.jpg"
    with pytest.raises(RuntimeError, match="Failed to load person image"):
        pipe.run("nonexistent.jpg", "cloth.png", str(out))


def test_run_cloth_imread_failure(tmp_path):
    person_img = np.zeros((10, 10, 3), dtype=np.uint8)
    person_path = tmp_path / "person.jpg"
    cv2.imwrite(str(person_path), person_img)

    pipe = VTONPipeline.__new__(VTONPipeline)  # bypass heavy init
    out = tmp_path / "out.jpg"
    with pytest.raises(RuntimeError, match="Failed to load cloth image"):
        pipe.run(str(person_path), "nonexistent.jpg", str(out))


def test_get_pipeline_cache(monkeypatch):
    from vton import get_pipeline

    dummy = object()
    monkeypatch.setattr("vton.VTONPipeline", lambda: dummy)
    monkeypatch.setattr("vton._GLOBAL_PIPELINE", None)

    first = get_pipeline()
    second = get_pipeline()

    assert first is second is dummy


def test_invalid_segmentation_model():
    import vton
    if vton.torch is None:
        pytest.skip("torch not available")
    with pytest.raises(ValueError):
        VTONPipeline(segmentation_model="foo")


def test_extract_keypoints_fallback(monkeypatch, caplog):
    pipe = VTONPipeline.__new__(VTONPipeline)
    pipe.pose_backend = "openpose"
    pipe.op = type("Op", (), {"Datum": lambda self=None: type("Datum", (), {})(),
                               "VectorDatum": lambda self, arr: arr})()
    pipe.op_wrapper = type("Wrapper", (), {"emplaceAndPop": lambda self, d: None})()

    fallback_called = False

    def fake_mp(img):
        nonlocal fallback_called
        fallback_called = True
        return {"nose": (0, 0)}

    monkeypatch.setattr(pipe, "mp_pose", object(), raising=False)
    monkeypatch.setattr(pipe, "_extract_keypoints_mp", fake_mp)

    datum = pipe.op.Datum()
    datum.poseKeypoints = None
    monkeypatch.setattr(pipe.op, "Datum", lambda: datum)

    with caplog.at_level(logging.WARNING):
        result = pipe.extract_keypoints(np.zeros((1, 1, 3), dtype=np.uint8))
    assert "OpenPose detected no keypoints on (1, 1, 3); switching to Mediapipe" in caplog.text
    assert fallback_called
    assert result == {"nose": (0, 0)}


def test_warp_fallback_estimation(monkeypatch, caplog):
    pipe = VTONPipeline.__new__(VTONPipeline)
    cloth = np.ones((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    src = {"nose": (0, 0), "neck": (1, 0), "left_shoulder": (0, 1)}
    dst = {"nose": (1, 1), "neck": (2, 1), "left_shoulder": (1, 2)}

    monkeypatch.setattr(PiecewiseAffineTransform, "estimate", lambda self, s, d: False)
    with caplog.at_level(logging.WARNING):
        out_c, out_m, status = pipe.warp(
            cloth, mask, src, dst, person_shape=(4, 4), return_status=True
        )
    assert out_c.shape == cloth.shape
    assert out_m.shape == mask.shape
    assert status == "estimation_failed"
    assert "approximate overlay" in caplog.text.lower()
    assert "3 source" in caplog.text and "3 destination" in caplog.text


def test_warp_fallback_error(monkeypatch, caplog):
    pipe = VTONPipeline.__new__(VTONPipeline)
    cloth = np.ones((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    src = {"nose": (0, 0), "neck": (1, 0), "left_shoulder": (0, 1)}
    dst = {"nose": (1, 1), "neck": (2, 1), "left_shoulder": (1, 2)}

    monkeypatch.setattr(PiecewiseAffineTransform, "estimate", lambda self, s, d: True)
    monkeypatch.setattr(PiecewiseAffineTransform, "__call__", lambda self, pts: pts + 50)
    monkeypatch.setattr(
        PiecewiseAffineTransform,
        "inverse",
        property(lambda self: type("Inv", (), {"__call__": lambda _self, pts: pts + 50})()),
    )

    with caplog.at_level(logging.WARNING):
        out_c, out_m, status = pipe.warp(
            cloth, mask, src, dst, person_shape=(4, 4), return_status=True
        )
    assert out_c.shape == cloth.shape
    assert out_m.shape == mask.shape
    assert status == "error"
    assert "approximate overlay" in caplog.text.lower()


def test_extract_keypoints_mp_new_api(monkeypatch):
    pipe = VTONPipeline.__new__(VTONPipeline)

    class FakeImage:
        def __init__(self, image_format=None, data=None):
            self.image_format = image_format
            self.data = data

    mp_stub = type(
        "mp",
        (),
        {
            "Image": FakeImage,
            "ImageFormat": type("Fmt", (), {"SRGB": object()}),
            "solutions": type("sol", (), {"pose": type("p", (), {"PoseLandmark": {}})}),
        },
    )

    calls = []

    def process(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return type("Res", (), {"pose_landmarks": None})()

    pipe.mp = mp_stub
    pipe.mp_pose = type("Pose", (), {"process": process})()

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    assert pipe._extract_keypoints_mp(img) is None
    assert len(calls) == 1
    assert isinstance(calls[0][0], FakeImage)
    assert calls[0][1]["image_size"] == (2, 2)


def test_extract_keypoints_mp_fallback(monkeypatch):
    pipe = VTONPipeline.__new__(VTONPipeline)

    class FakeImage:
        def __init__(self, image_format=None, data=None):
            self.image_format = image_format
            self.data = data

    mp_stub = type(
        "mp",
        (),
        {
            "Image": FakeImage,
            "ImageFormat": type("Fmt", (), {"SRGB": object()}),
            "solutions": type("sol", (), {"pose": type("p", (), {"PoseLandmark": {}})}),
        },
    )

    class Pose:
        def __init__(self):
            self.calls = []

        def process(self, arg, **kwargs):
            self.calls.append((arg, kwargs))
            if isinstance(arg, FakeImage):
                raise AttributeError("object has no attribute 'shape'")
            return type("Res", (), {"pose_landmarks": None})()

    pose = Pose()
    pipe.mp = mp_stub
    pipe.mp_pose = pose

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    assert pipe._extract_keypoints_mp(img) is None
    assert len(pose.calls) == 2
    assert isinstance(pose.calls[0][0], FakeImage)
    assert pose.calls[0][1]["image_size"] == (2, 2)
    assert isinstance(pose.calls[1][0], np.ndarray)
