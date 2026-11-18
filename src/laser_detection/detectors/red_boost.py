"""Fixed threshold on signed red-channel dominance in BGR images."""

import cv2
import numpy as np

from ..detection_settings import RedBoostSettings
from .base import Detector

_BOOST_WEIGHTS = np.array([[-0.5, -0.5, 1]], dtype=np.float32)


def red_boost(bgr):
    return cv2.transform(bgr.astype(np.float32), _BOOST_WEIGHTS).reshape(bgr.shape[:2])


class RedBoostDetector(Detector):
    name = "red_boost"

    def __init__(self, settings: RedBoostSettings = RedBoostSettings(), *, width=1536, height=1024):
        super().__init__(width, height)
        self.settings = settings

    def binary_mask(self, bgr):
        return (red_boost(bgr) > self.settings.boost_threshold).astype(np.uint8) * 255
