import os
import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")
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
