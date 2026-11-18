"""Validated settings shared by the baseline detectors and benchmark."""

from dataclasses import dataclass
from math import isfinite


def _integer(name, value, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer in {bounds}")


def _range(name, values, maximum):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"{name} must have two bounds")
    lower, upper = values
    for value in values:
        _integer(name, value, 0, maximum)
    if lower > upper:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return lower, upper


@dataclass(frozen=True)
class HLSSettings:
    hue_range_1: tuple[int, int] = (0, 10)
    hue_range_2: tuple[int, int] = (170, 180)
    lightness_min: int = 40
    lightness_max: int = 255
    saturation_min: int = 100
    saturation_max: int = 255

    def __post_init__(self):
        for name in ("hue_range_1", "hue_range_2"):
            object.__setattr__(self, name, _range(name, getattr(self, name), 180))
        _range("lightness", (self.lightness_min, self.lightness_max), 255)
        _range("saturation", (self.saturation_min, self.saturation_max), 255)


@dataclass(frozen=True)
class RedBoostSettings:
    boost_threshold: float = 40

    def __post_init__(self):
        value = self.boost_threshold
        if type(value) not in (int, float) or not isfinite(value) or not -255 <= value <= 255:
            raise ValueError("boost_threshold must be in [-255, 255]")


@dataclass(frozen=True)
class DetectionSettings:
    width: int = 1536
    height: int = 1024
    hls: HLSSettings = HLSSettings()
    red_boost: RedBoostSettings = RedBoostSettings()
    warmup_frames: int = 5
    opencv_threads: int = 1
    split_seed: int = 42

    def __post_init__(self):
        for name in ("width", "height", "opencv_threads"):
            _integer(name, getattr(self, name), 1)
        for name in ("warmup_frames", "split_seed"):
            _integer(name, getattr(self, name), 0)
        if isinstance(self.hls, dict):
            object.__setattr__(self, "hls", HLSSettings(**self.hls))
        if isinstance(self.red_boost, dict):
            object.__setattr__(self, "red_boost", RedBoostSettings(**self.red_boost))
        if not isinstance(self.hls, HLSSettings) or not isinstance(self.red_boost, RedBoostSettings):
            raise ValueError("hls and red_boost must be mappings")
