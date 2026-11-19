"""Shared baseline pipeline and pixel-center coordinate mapping."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Optional

import cv2
import numpy as np

from .blobs import find_components


def scale_centers(points, from_size: tuple[int, int], to_size: tuple[int, int]):
    """Map XY pixel centers between image grids given sizes in (width, height) order."""
    return (np.asarray(points, dtype=float) + 0.5) * (np.asarray(to_size) / from_size) - 0.5


@dataclass(frozen=True)
class Detection:
    detected: bool
    coordinates: tuple[float, float, float, float]
    component_count: int
    area1: Optional[int]
    area2: Optional[int]
    resize_ns: int
    threshold_ns: int
    components_ns: int
    total_ns: int
    mask: Optional[np.ndarray] = None


class Detector(ABC):
    name: str

    def __init__(self, width=1536, height=1024):
        self.size = (width, height)

    @abstractmethod
    def binary_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Threshold an already resized uint8 BGR image."""

    def resize(self, bgr):
        return cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)

    def detect(self, bgr: np.ndarray, *, return_mask=False) -> Detection:
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
            raise ValueError("Expected a nonempty uint8 BGR image")
        start = perf_counter_ns()
        resized = self.resize(bgr)
        after_resize = perf_counter_ns()
        mask = self.binary_mask(resized)
        after_threshold = perf_counter_ns()
        blobs = find_components(mask)
        after_components = perf_counter_ns()
        if blobs.selected_labels:
            indices = np.asarray(blobs.selected_labels) - 1
            xy = scale_centers(blobs.centroids[indices], self.size, bgr.shape[1::-1])
            coordinates = tuple(map(float, xy.ravel()))
            area1, area2 = map(int, blobs.areas[indices])
        else:
            coordinates = (float("nan"),) * 4
            area1 = area2 = None
        end = perf_counter_ns()
        return Detection(
            bool(blobs.selected_labels), coordinates, blobs.count, area1, area2,
            after_resize - start, after_threshold - after_resize,
            after_components - after_threshold, end - start, mask if return_mask else None,
        )
