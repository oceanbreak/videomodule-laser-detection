"""Dataset counts and per-setup descriptive statistics."""

import pandas as pd

from .geometry import METRICS


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = int(frame["d"].count())
    return pd.DataFrame([{
        "frames": len(frame),
        "labeled": labeled,
        "unlabeled": len(frame) - labeled,
        "setups": frame["setup_id"].nunique(),
    }])


def setup_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("setup_id", sort=False)
    summary = grouped["d"].agg(frames="size", labeled="count")
    summary["unlabeled"] = summary["frames"] - summary["labeled"]
    summary["missing_fraction"] = summary["unlabeled"] / summary["frames"]
    metrics = grouped[METRICS].agg(["median", "std", "min", "max"])
    metrics.columns = ["_".join(column) for column in metrics.columns]
    return summary.join(metrics).reset_index()
