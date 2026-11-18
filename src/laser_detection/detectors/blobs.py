"""Foreground components of an eight-connected binary mask."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Components:
    areas: np.ndarray
    centroids: np.ndarray
    selected_labels: tuple[int, ...]

    @property
    def count(self):
        return len(self.areas)


def find_components(mask: np.ndarray) -> Components:
    """Return foreground results in label order, indexed by label minus one."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    centroids = centroids[1:]
    selected = ()
    if count >= 3:
        largest = np.argsort(-areas, kind="stable")[:2]
        selected = tuple(int(index + 1) for index in sorted(largest, key=lambda i: (*centroids[i], i)))
    return Components(areas, centroids, selected)
