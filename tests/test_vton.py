import numpy as np
from vton import VTONPipeline


def test_segment_size():
    pipe = VTONPipeline()
    img = np.zeros((64, 32, 3), dtype=np.uint8)
    mask = pipe.segment(img)
    assert mask.shape == img.shape[:2]
