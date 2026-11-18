"""Color-only laser detection baselines."""

from .base import Detection, Detector
from .hls import HLSDetector
from .red_boost import RedBoostDetector


def baseline_detectors(settings) -> tuple[Detector, Detector]:
    size = {"width": settings.width, "height": settings.height}
    return HLSDetector(settings.hls, **size), RedBoostDetector(settings.red_boost, **size)


__all__ = ["Detection", "Detector", "HLSDetector", "RedBoostDetector", "baseline_detectors"]
