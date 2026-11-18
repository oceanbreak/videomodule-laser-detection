"""Fixed color thresholds in OpenCV's standard 8-bit HLS space."""

import cv2

from ..detection_settings import HLSSettings
from .base import Detector


class HLSDetector(Detector):
    name = "hls"

    def __init__(self, settings: HLSSettings = HLSSettings(), *, width=1536, height=1024):
        super().__init__(width, height)
        self.settings = settings

    def binary_mask(self, bgr):
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        s = self.settings
        mask1 = cv2.inRange(hls, (s.hue_range_1[0], s.lightness_min, s.saturation_min),
                            (s.hue_range_1[1], s.lightness_max, s.saturation_max))
        mask2 = cv2.inRange(hls, (s.hue_range_2[0], s.lightness_min, s.saturation_min),
                            (s.hue_range_2[1], s.lightness_max, s.saturation_max))
        return cv2.bitwise_or(mask1, mask2)
