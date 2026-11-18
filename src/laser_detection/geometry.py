"""Pair geometry in pixel coordinates and radians."""

import numpy as np
import pandas as pd

METRICS = ["xc", "yc", "dx", "dy", "d", "theta", "theta_lr"]


def add_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    x1, y1, x2, y2 = (frame[column] for column in ("x1", "y1", "x2", "y2"))
    dx, dy = x2 - x1, y2 - y1
    # For vertical pairs, order by increasing Y to preserve swap invariance.
    reverse = (dx < 0) | ((dx == 0) & (dy < 0))
    direction = np.where(reverse, -1, 1)
    return frame.assign(
        xc=(x1 + x2) / 2, yc=(y1 + y2) / 2,
        dx=dx, dy=dy, d=np.hypot(dx, dy),
        theta=np.arctan2(dy, dx),
        theta_lr=np.arctan2(dy * direction, np.abs(dx)),
    )
