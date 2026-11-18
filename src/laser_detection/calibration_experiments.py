"""Run full-data fits and repeated experiments on untouched holdout pairs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import (
    SIDES,
    CalibrationError,
    LaserLines,
    calculate_residuals,
    fit_laser_lines,
    inlier_masks,
    labeled_pairs,
    residual_statistics,
)
from .config import Config
from .data import COORDINATES


@dataclass
class CalibrationStudy:
    models: dict[str, LaserLines]
    parameters: pd.DataFrame
    summary: pd.DataFrame
    residuals: pd.DataFrame
    corridors: pd.DataFrame
    sample_runs: pd.DataFrame
    sample_summary: pd.DataFrame
    contamination_runs: pd.DataFrame
    contamination_summary: pd.DataFrame


def split_positions(count, size, rng):
    """Draw whole pairs without replacement and retain source order in both partitions."""
    if not 2 <= size < count:
        raise ValueError("Need at least two training pairs and one held-out pair")
    train = np.sort(rng.choice(count, size=size, replace=False))
    return train, np.setdiff1d(np.arange(count), train)


def contaminate_pairs(train, fraction, width, height, rng):
    """Replace complete pairs with uniform pixel coordinates in a training copy."""
    result = train.copy()
    count = int(np.floor(len(train) * fraction + 0.5))
    indices = rng.choice(len(train), count, replace=False)
    points = rng.integers([0, 0], [width, height], size=(count, 2, 2))
    # lexsort's last key is primary. Y provides the tie-break at equal X.
    order = np.lexsort((points[:, :, 1], points[:, :, 0]), axis=1)
    pairs = np.take_along_axis(points, order[:, :, None], axis=1)
    result.iloc[indices, result.columns.get_indexer(COORDINATES)] = pairs.reshape(count, 4)
    return result, count


def _coefficients(model):
    values = {
        f"{side}_{key}": getattr(getattr(model, side), key) if model else np.nan
        for side in ("left", "right") for key in ("a", "b", "c")
    }
    return {**values, "angle_degrees": model.angle_degrees if model else np.nan}


def _inlier_statistics(residuals, threshold, prefix=""):
    masks = inlier_masks(residuals, threshold)
    return {
        **{f"{prefix}{side}_inliers": int(masks[side].sum()) for side in SIDES},
        **{f"{prefix}{side}_inlier_fraction": masks[side].mean() for side in SIDES},
    }


def corridor_metric(side, threshold):
    return f"{side}_coverage_px_{threshold:g}"


def _run_metrics(train, test, settings, seed):
    try:
        model = fit_laser_lines(train, settings, random_state=seed)
    except CalibrationError as error:
        return {"status": "failed", "message": str(error)}
    test_residuals = calculate_residuals(test, model)
    quantiles = test_residuals.quantile([0.5, 0.95])
    return {
        "status": "ok", "message": "",
        **_coefficients(model),
        **_inlier_statistics(calculate_residuals(train, model), settings.ransac_threshold, "train_"),
        **{f"test_{side}_{label}": quantiles.loc[q, side]
           for side in SIDES for label, q in (("median", 0.5), ("p95", 0.95))},
        **{corridor_metric(side, t): value
           for t in settings.corridor_thresholds
           for side, value in inlier_masks(test_residuals, t).mean().items()},
    }


def _seed(rng):
    return int(rng.integers(2**32))


def _repeat_runs(points, setup, config, rng, sizes, *, contaminated=False):
    settings = config.calibration
    rows = []
    fractions = settings.contamination_fractions if contaminated else (None,)
    for size in sizes:
        base = {
            "setup_id": setup, "evaluation": "out_of_sample", "training_size": size,
            "test_pairs": max(0, len(points) - size),
        }
        if size >= len(points):
            for fraction in fractions:
                rows.append({
                    **base, "contamination_fraction": fraction,
                    "status": "skipped", "message": f"Need more than {size} annotated pairs; have {len(points)}",
                })
            continue
        for repeat in range(1, settings.repetitions + 1):
            split_seed, fit_seed = _seed(rng), _seed(rng)
            train_pos, test_pos = split_positions(len(points), size, np.random.default_rng(split_seed))
            train, test = points.iloc[train_pos], points.iloc[test_pos]
            # Use the same split and fit seed across levels within each repeat.
            for fraction in fractions:
                meta = {**base, "repeat": repeat, "split_seed": split_seed, "fit_seed": fit_seed}
                fit_points = train
                if contaminated:
                    noise_seed = _seed(rng)
                    fit_points, count = contaminate_pairs(
                        train, fraction, config.width, config.height, np.random.default_rng(noise_seed),
                    )
                    meta.update(contamination_fraction=fraction, replaced_pairs=count, noise_seed=noise_seed)
                rows.append({**meta, **_run_metrics(fit_points, test, settings, fit_seed)})
    result = pd.DataFrame(rows)
    return result if contaminated else result.drop(columns="contamination_fraction", errors="ignore")


def _metric_names(settings):
    return [
        *(f"train_{side}_inlier_fraction" for side in SIDES),
        *(f"test_{side}_{label}" for side in SIDES for label in ("median", "p95")),
        *(corridor_metric(side, t) for t in settings.corridor_thresholds for side in SIDES),
    ]


def _complete_runs(runs, settings, contaminated=False):
    columns = ["setup_id", "evaluation", "training_size", "test_pairs", "repeat",
               "status", "message", "split_seed", "fit_seed"]
    if contaminated:
        columns.extend(["contamination_fraction", "replaced_pairs", "noise_seed"])
    columns.extend(_coefficients(None))
    columns.extend(f"train_{side}_inliers" for side in SIDES)
    columns.extend(_metric_names(settings))
    runs = runs.reindex(columns=columns)
    integer_columns = ["repeat", "split_seed", "fit_seed"]
    if contaminated:
        integer_columns.extend(["replaced_pairs", "noise_seed"])
    runs[integer_columns] = runs[integer_columns].astype("Int64")
    return runs


def _aggregate(runs, keys, metrics):
    rows = []
    for group_key, group in runs.groupby(keys, sort=False):
        statuses = group["status"].value_counts()
        ok = group.loc[group["status"].eq("ok"), metrics].astype(float)
        quantiles = ok.quantile([0.05, 0.5, 0.95])
        for metric in metrics:
            rows.append({
                **dict(zip(keys, group_key)),
                "evaluation": "out_of_sample", "metric": metric,
                "successful_runs": int(statuses.get("ok", 0)),
                "failed_runs": int(statuses.get("failed", 0)),
                "skipped": bool(statuses.get("skipped", 0)),
                "p05": quantiles.at[0.05, metric],
                "median": quantiles.at[0.5, metric],
                "p95": quantiles.at[0.95, metric],
            })
    return pd.DataFrame(rows)


def run_calibration_study(frame: pd.DataFrame, config: Config) -> CalibrationStudy:
    settings = config.calibration
    full_rng, sample_rng, noise_rng = map(
        np.random.default_rng, np.random.SeedSequence(settings.random_state).spawn(3),
    )
    models, parameters, summaries, residual_tables, coverage, samples, noise = {}, [], [], [], [], [], []
    for setup, group in frame.groupby("setup_id", sort=False):
        points = labeled_pairs(group)
        seed = _seed(full_rng)
        row = {"setup_id": setup, "evaluation": "in_sample", "annotated_pairs": len(points), "fit_seed": seed}
        residuals = pd.DataFrame(np.nan, index=points.index, columns=SIDES)
        model = None
        try:
            model = fit_laser_lines(points, settings, random_state=seed)
        except CalibrationError as error:
            row.update(status="failed", message=str(error))
        else:
            models[setup] = model
            row.update(status="ok", message="")
            residuals = calculate_residuals(points, model)
        parameters.append({**row, **_coefficients(model)})
        fitted = _inlier_statistics(residuals, settings.ransac_threshold) if model else {}
        summaries.append({**row, **_coefficients(model), **residual_statistics(residuals), **fitted})
        per_frame = points[["image", "setup_id", *COORDINATES]].join(residuals.add_suffix("_residual"))
        per_frame["evaluation"] = "in_sample"
        per_frame["status"] = row["status"]
        for column, values in inlier_masks(residuals, settings.ransac_threshold).items():
            per_frame[f"{column}_inlier"] = values.where(residuals[column].notna()).astype("boolean")
        residual_tables.append(per_frame)

        for threshold in settings.corridor_thresholds:
            fractions = inlier_masks(residuals, threshold).mean() if model else dict.fromkeys(SIDES, np.nan)
            coverage.append({**row, "threshold": threshold, **fractions})
        samples.append(_repeat_runs(points, setup, config, sample_rng, settings.training_sizes))
        noise.append(_repeat_runs(
            points, setup, config, noise_rng, [settings.contamination_training_size], contaminated=True,
        ))
    sample_runs = _complete_runs(pd.concat(samples, ignore_index=True), settings)
    noise_runs = _complete_runs(pd.concat(noise, ignore_index=True), settings, contaminated=True)
    metrics = _metric_names(settings)
    summary = pd.DataFrame(summaries)
    for side in SIDES:
        for name in ("inliers", "inlier_fraction"):
            column = f"{side}_{name}"
            if column not in summary:
                summary[column] = np.nan
    return CalibrationStudy(
        models, pd.DataFrame(parameters), summary,
        pd.concat(residual_tables).sort_index(kind="stable"), pd.DataFrame(coverage),
        sample_runs, _aggregate(sample_runs, ["setup_id", "training_size"], metrics),
        noise_runs, _aggregate(noise_runs, ["setup_id", "contamination_fraction"], metrics),
    )
