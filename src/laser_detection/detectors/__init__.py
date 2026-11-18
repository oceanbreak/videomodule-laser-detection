"""Color-only laser detection baselines."""

from .base import Detection, Detector
from .hls import HLSDetector
from .hsv import HSVDetector
from .red_boost import RedBoostDetector


def baseline_detectors(settings) -> tuple[Detector, ...]:
    size = {"width": settings.width, "height": settings.height}
    return (
        HLSDetector(settings.hls, **size),
        RedBoostDetector(settings.red_boost, **size),
        HSVDetector(settings.hsv_legacy, name="hsv_legacy", **size),
        HSVDetector(settings.hsv_narrow, name="hsv_narrow", **size),
    )


__all__ = ["Detection", "Detector", "HLSDetector", "HSVDetector", "RedBoostDetector", "baseline_detectors"]
