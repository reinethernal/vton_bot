import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is required for VTONPipeline")
from vton import VTONPipeline


def test_segment_size():
    pipe = VTONPipeline()
    img = np.zeros((64, 32, 3), dtype=np.uint8)
    mask = pipe.segment(img)
    assert mask.shape == img.shape[:2]
