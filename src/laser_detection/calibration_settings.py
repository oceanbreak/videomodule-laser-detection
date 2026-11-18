"""Defaults for reproducible calibration experiments."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CalibrationSettings:
    ransac_threshold: float = 10
    ransac_trials: int = 256
    corridor_thresholds: tuple[float, ...] = (5, 10, 20, 30, 50)
    training_sizes: tuple[int, ...] = (10, 20, 30, 50, 100, 200)
    repetitions: int = 30
    contamination_fractions: tuple[float, ...] = (0, 0.05, 0.10, 0.20, 0.30)
    contamination_training_size: int = 50
    random_state: int = 42

    def __post_init__(self):
        for name, minimum in (
            ("ransac_trials", 1), ("repetitions", 1),
            ("contamination_training_size", 2), ("random_state", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not _positive(self.ransac_threshold):
            raise ValueError("ransac_threshold must be a finite positive number")
        for name, valid in (
            ("corridor_thresholds", _positive),
            ("training_sizes", lambda value: type(value) is int and value >= 2),
            ("contamination_fractions", lambda value: _number(value) and 0 <= value <= 1),
        ):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values or not all(map(valid, values)):
                raise ValueError(f"Invalid {name}")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, tuple(sorted(values)))


def _number(value):
    return type(value) in (float, int) and isfinite(value)


def _positive(value):
    return _number(value) and value > 0
