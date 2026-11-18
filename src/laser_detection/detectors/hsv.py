"""Fixed color thresholds in OpenCV's standard 8-bit HSV space."""

import cv2

from ..detection_settings import HSVSettings
from .base import Detector


class HSVDetector(Detector):
    def __init__(self, settings: HSVSettings = HSVSettings(), *, name="hsv_legacy",
                 width=1536, height=1024):
        super().__init__(width, height)
        self.name = name
        self.settings = settings

    def binary_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = self.settings
        mask1 = cv2.inRange(hsv, (s.hue_range_1[0], s.saturation_min, s.value_min),
                            (s.hue_range_1[1], s.saturation_max, s.value_max))
        mask2 = cv2.inRange(hsv, (s.hue_range_2[0], s.saturation_min, s.value_min),
                            (s.hue_range_2[1], s.saturation_max, s.value_max))
        return cv2.bitwise_or(mask1, mask2)
