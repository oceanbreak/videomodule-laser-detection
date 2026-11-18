"""Group-aware evaluation and pair-level detection metrics."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import COORDINATES, _read_csv, _unique_names

RECALL_THRESHOLDS = (5, 10, 20)


def dataset_split(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Split independent capture series as units, in deterministic 60/20/20 group proportions."""
    assignments = frame[["image"]].copy()
    if config.series_csv is None:
        return assignments.assign(series_id=pd.NA, split="exploratory")
    groups = _read_csv(config.series_csv, ["image", "series_id"])
    _unique_names(groups, "image", config.series_csv)
    groups["series_id"] = groups["series_id"].str.strip()
    if groups["series_id"].eq("").any():
        raise ValueError(f"{config.series_csv}: series_id cannot be empty")
    expected, provided = set(assignments["image"]), set(groups["image"])
    if expected != provided:
        raise ValueError(
            f"{config.series_csv}: series mapping must match the dataset "
            f"({len(expected - provided)} missing, {len(provided - expected)} extra images)"
        )
    series = sorted(groups["series_id"].unique())
    if len(series) < 3:
        raise ValueError(f"{config.series_csv}: need at least three independent capture series")
    shuffled = np.random.default_rng(config.detection.split_seed).permutation(series)
    held_out = max(1, round(0.2 * len(series)))
    labels = dict.fromkeys(shuffled[:held_out], "test")
    labels.update(dict.fromkeys(shuffled[held_out:2 * held_out], "validation"))
    labels.update(dict.fromkeys(shuffled[2 * held_out:], "train"))
    assignments = assignments.merge(groups, on="image", validate="one_to_one")
    return assignments.assign(split=assignments["series_id"].map(labels))


def score_predictions(predictions, annotations, splits):
    truth_columns = [f"true_{coord}" for coord in COORDINATES]
    truth = annotations[["image", *COORDINATES]].rename(columns=dict(zip(COORDINATES, truth_columns)))
    scored = predictions.merge(truth, on="image", validate="many_to_one")
    scored = scored.merge(splits, on="image", validate="many_to_one")
    scored["annotated"] = scored[truth_columns].notna().all(axis=1)
    readable = scored["status"].eq("ok")
    present = scored["detected"].fillna(False).astype(bool)
    annotated = scored["annotated"]
    scored["outcome"] = np.select(
        [~readable, annotated & present, ~annotated & present, annotated & ~present],
        ["unscored", "tp", "fp", "fn"], default="tn",
    )
    for point in (1, 2):
        dx = scored[f"x{point}"].astype(float) - scored[f"true_x{point}"]
        dy = scored[f"y{point}"].astype(float) - scored[f"true_y{point}"]
        scored[f"error{point}_px"] = np.hypot(dx, dy).where(scored["outcome"].eq("tp"))
    scored["pair_max_error_px"] = np.maximum(scored["error1_px"], scored["error2_px"])
    return scored


def _rate(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def _metrics(group):
    outcomes = group["outcome"].value_counts()
    tp, fp, fn, tn = (int(outcomes.get(label, 0)) for label in ("tp", "fp", "fn", "tn"))
    errors = group[["error1_px", "error2_px"]].to_numpy(dtype=float).ravel()
    errors = pd.Series(errors).dropna()
    quantiles = errors.quantile([0.5, 0.90, 0.95])
    return {
        "frames": len(group), "scored_frames": tp + fp + fn + tn,
        "image_errors": int(outcomes.get("unscored", 0)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": _rate(tp, tp + fp), "recall": _rate(tp, tp + fn),
        "f1": _rate(2 * tp, 2 * tp + fp + fn),
        "localized_pairs": tp,
        "mean_point_error_px": errors.mean(), "median_point_error_px": quantiles.loc[0.5],
        "p90_point_error_px": quantiles.loc[0.90], "p95_point_error_px": quantiles.loc[0.95],
        **{f"recall_at_{threshold}px": _rate(group["pair_max_error_px"].le(threshold).sum(), tp + fn)
           for threshold in RECALL_THRESHOLDS},
    }


def detection_metrics(scored, *, by_setup=False):
    """Summaries by split. Localization statistics pool both distances for matched pairs."""
    dimensions = ["split", "method"]
    if by_setup:
        dimensions.append("setup_id")
    return pd.DataFrame([
        {**dict(zip(dimensions, keys)), **_metrics(group)}
        for keys, group in scored.groupby(dimensions, sort=False)
    ])


def worst_cases(scored, method, split, count=10):
    """List missed pairs, false alarms, then matches ordered by maximum coordinate error."""
    rows = scored.loc[
        scored["method"].eq(method) & scored["split"].eq(split) & scored["outcome"].isin(["fn", "fp", "tp"])
    ].copy()
    rows["_priority"] = rows["outcome"].map({"fn": 0, "fp": 1, "tp": 2})
    return rows.sort_values(["_priority", "pair_max_error_px", "image"], ascending=[True, False, True]).head(count).drop(columns="_priority")


def comparison_table(metrics, timings, split):
    """Combine the same summary columns from live data or previously exported CSVs."""
    columns = ["method", "recall", *(f"recall_at_{t}px" for t in RECALL_THRESHOLDS),
               "median_point_error_px"]
    scores = metrics.loc[metrics["split"].eq(split), columns]
    return scores.merge(timings[["method", "total_mean_ms"]], on="method",
                        how="left", validate="one_to_one")


def load_detection_comparison(directory, split):
    """Rebuild the comparison from saved summaries without decoding images."""
    directory = Path(directory)
    metrics = pd.read_csv(directory / "metrics_by_method.csv")
    timings = pd.read_csv(directory / "timings.csv")
    if not metrics["split"].eq(split).any():
        raise ValueError(f"Split {split!r} is not present in {directory}")
    return comparison_table(metrics, timings, split)


def localization_disagreements(scored, first, second, split, threshold=10):
    """Return annotated frames where exactly one method localizes both points."""
    rows = scored.loc[
        scored["split"].eq(split) & scored["annotated"] & scored["status"].eq("ok"),
        ["image", "setup_id", "method", "outcome", "pair_max_error_px"],
    ]
    left = rows.loc[rows["method"].eq(first)].drop(columns="method")
    right = rows.loc[rows["method"].eq(second)].drop(columns=["method", "setup_id"])
    paired = left.merge(right, on="image", suffixes=(f"_{first}", f"_{second}"), validate="one_to_one")
    first_pass = paired[f"pair_max_error_px_{first}"].le(threshold)
    second_pass = paired[f"pair_max_error_px_{second}"].le(threshold)
    paired["better_method"] = np.where(first_pass, first, second)
    return paired.loc[first_pass ^ second_pass]
