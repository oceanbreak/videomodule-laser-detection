"""Independent orthogonal line fits and joint corridor evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration_settings import CalibrationSettings
from .data import COORDINATES

SIDES = ("left", "right", "pair")


class CalibrationError(ValueError):
    """The supplied observations do not determine two usable lines."""


@dataclass(frozen=True)
class Line:
    a: float
    b: float
    c: float

    def __post_init__(self):
        norm = np.hypot(self.a, self.b)
        if not np.isfinite(norm) or norm == 0 or not np.isfinite(self.c):
            raise ValueError("Line requires a finite, nonzero normal and a finite offset")
        sign = -1 if self.a < 0 or (self.a == 0 and self.b < 0) else 1
        for name in ("a", "b", "c"):
            object.__setattr__(self, name, float(getattr(self, name) / norm * sign))

    def distance(self, points):
        return np.abs(np.asarray(points) @ [self.a, self.b] + self.c)


@dataclass(frozen=True)
class LaserLines:
    left: Line
    right: Line

    @property
    def angle_degrees(self):
        """Acute, unoriented angle between the lines, in degrees."""
        a, b = self.left.a, self.left.b
        u, v = self.right.a, self.right.b
        return float(np.degrees(np.arctan2(abs(a * v - b * u), abs(a * u + b * v))))


def _fit_line(points, config, rng):
    if len(points) < 2 or not np.any(points != points[0]):
        raise CalibrationError("at least two distinct points are required")
    best = None
    best_score = (-1, -np.inf)
    _, representatives = np.unique(points, axis=0, return_index=True)
    first = rng.integers(len(points), size=config.ransac_trials)
    second = rng.integers(len(points) - 1, size=config.ransac_trials)
    second += second >= first
    origins = points[first]
    deltas = points[second] - origins
    for i in np.flatnonzero(~deltas.any(axis=1)):
        alternatives = representatives[np.any(points[representatives] != origins[i], axis=1)]
        deltas[i] = points[rng.choice(alternatives)] - origins[i]
    normals = deltas[:, [1, 0]] * [1, -1]
    normals /= np.hypot(normals[:, 0], normals[:, 1])[:, None]
    offsets = -np.sum(normals * origins, axis=1)
    for start in range(0, len(normals), 32):
        stop = start + 32
        a, b = normals[start:stop].T
        distances = np.abs(
            a[:, None] * points[:, 0] + b[:, None] * points[:, 1] + offsets[start:stop, None]
        )
        masks = distances <= config.ransac_threshold
        counts = masks.sum(axis=1)
        costs = np.sum(distances, axis=1, where=masks)
        index = np.lexsort((costs, -counts))[0]
        score = (int(counts[index]), -float(costs[index]))
        if score > best_score:
            best, best_score = masks[index], score

    if best is None or best.sum() < 2:
        raise CalibrationError("no line has at least two RANSAC inliers")
    consensus = points[best]
    center = consensus.mean(axis=0)
    _, singular, axes = np.linalg.svd(consensus - center, full_matrices=False)
    if singular[0] == 0 or np.isclose(singular[0], singular[1], rtol=1e-10, atol=0):
        raise CalibrationError("RANSAC consensus does not determine a unique direction")
    normal = axes[-1]
    refined = Line(*normal, -normal @ center)
    final_inliers = points[refined.distance(points) <= config.ransac_threshold]
    if len(final_inliers) < 2 or not np.any(final_inliers != final_inliers[0]):
        raise CalibrationError("fitted line has fewer than two distinct final inliers")
    return refined


def labeled_pairs(points: pd.DataFrame) -> pd.DataFrame:
    mask = np.isfinite(points[COORDINATES].to_numpy(dtype=float)).all(axis=1)
    return points.loc[mask]


def fit_laser_lines(
    points: pd.DataFrame, config: CalibrationSettings = CalibrationSettings(), *,
    random_state: int | None = None,
) -> LaserLines:
    """Fit models from pairs normalized at import. Rows with missing pairs are omitted."""
    coordinates = labeled_pairs(points)[COORDINATES].to_numpy(dtype=float)
    seed = config.random_state if random_state is None else random_state
    left_rng, right_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(2))
    lines = []
    for side, start, rng in (("left", 0, left_rng), ("right", 2, right_rng)):
        try:
            lines.append(_fit_line(coordinates[:, start:start + 2], config, rng))
        except CalibrationError as error:
            raise CalibrationError(f"{side}: {error}") from error
    return LaserLines(*lines)


def calculate_residuals(points: pd.DataFrame, model: LaserLines) -> pd.DataFrame:
    coordinates = points[COORDINATES].to_numpy(dtype=float)
    left = model.left.distance(coordinates[:, :2])
    right = model.right.distance(coordinates[:, 2:])
    return pd.DataFrame({"left": left, "right": right, "pair": np.maximum(left, right)},
                        index=points.index)


def inlier_masks(residuals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return masks for each laser and their intersection for the pair."""
    return residuals.loc[:, list(SIDES)].le(threshold)


def evaluate_corridors(points, model, thresholds) -> pd.DataFrame:
    residuals = calculate_residuals(labeled_pairs(points), model)
    return pd.DataFrame(
        [inlier_masks(residuals, threshold).mean() for threshold in thresholds],
        index=pd.Index(thresholds, name="threshold"),
    )


def residual_statistics(residuals: pd.DataFrame) -> dict:
    summary = {}
    for side in SIDES:
        values = residuals[side]
        quantiles = values.quantile([0.5, 0.9, 0.95, 0.99])
        for label, value in zip(("median", "p90", "p95", "p99"), quantiles):
            summary[f"{side}_{label}"] = value
        summary[f"{side}_max"] = values.max()
    return summary
