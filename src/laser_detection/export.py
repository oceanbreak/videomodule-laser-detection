"""Write standalone charts and inspection tables to local disk."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import Config
from .statistics import dataset_summary, setup_summary


def export_inspection(frame, figures, config: Config) -> list[Path]:
    tables = {
        "dataset_summary.csv": dataset_summary(frame),
        "setup_summary.csv": setup_summary(frame),
        "geometry.csv": frame,
    }
    return _write_outputs(tables, figures, {}, config.output_dir, config, "inspection")


def export_calibration(study, figures, config: Config) -> list[Path]:
    tables = {
        "line_parameters.csv": study.parameters,
        "in_sample_summary.csv": study.summary,
        "frame_residuals.csv": study.residuals,
        "in_sample_corridors.csv": study.corridors,
        "sample_size_runs.csv": study.sample_runs,
        "sample_size_summary.csv": study.sample_summary,
        "contamination_runs.csv": study.contamination_runs,
        "contamination_summary.csv": study.contamination_summary,
    }
    metadata = {"experiment_config.json": {
        **asdict(config.calibration), "image_width": config.width, "image_height": config.height,
    }}
    return _write_outputs(tables, figures, metadata, config.calibration_output_dir, config, "calibration")


def export_detection(scored, splits, metrics, by_setup, timings, environment, figures, config, *,
                     run_name="hsv_comparison"):
    """Export to a new run directory. Existing detection exports are never replaced."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("run_name must contain only ASCII letters, digits, underscores and hyphens")
    predictions = {}
    for method, rows in scored.groupby("method", sort=False):
        if not re.fullmatch(r"[a-z0-9_]+", method):
            raise ValueError(f"Invalid detector name for export: {method!r}")
        predictions[f"predictions_{method}.csv"] = rows
    tables = {
        **predictions,
        "metrics_by_method.csv": metrics,
        "metrics_by_setup.csv": by_setup,
        "timings.csv": timings,
        "dataset_split.csv": splits,
    }
    metadata = {"experiment_config.json": {
        "detection": asdict(config.detection),
        "original_size": [config.width, config.height],
        "split_type": "capture_series" if config.series_csv else "exploratory",
        "benchmark": environment,
    }}
    output = config.detection_output_dir.resolve() / run_name
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Detection run already exists: {output}. Choose a different run_name.") from error
    figure_dir = output / "figures"
    return [
        *_write_outputs(tables, {}, metadata, output, config, "detection"),
        *_write_outputs({}, figures, {}, figure_dir, config, "detection"),
    ]


def _write_outputs(tables, figures, metadata, output, config, prefix):
    charts = {f"{name}.html": figure for name, figure in figures.items()}
    output = output.resolve()
    inputs = {config.annotations_csv.resolve(), config.setups_csv.resolve()}
    if config.series_csv is not None:
        inputs.add(config.series_csv.resolve())
    destinations = [output / name for name in (*tables, *metadata, *charts)]
    for path in destinations:
        if path.parent != output or path.resolve().parent != output or path.resolve() in inputs:
            raise ValueError(f"Unsafe export destination: {path}")
        if path.exists() and any(source.exists() and path.samefile(source) for source in inputs):
            raise ValueError(f"Export would overwrite an input CSV: {path}")

    output.mkdir(parents=True, exist_ok=True)
    for path in destinations:
        with NamedTemporaryFile(dir=output, delete=False) as file:
            temporary = Path(file.name)
        try:
            if path.name in tables:
                tables[path.name].to_csv(temporary, index=False, lineterminator="\n")
            elif path.name in metadata:
                temporary.write_text(json.dumps(metadata[path.name], indent=2) + "\n", encoding="utf-8")
            else:
                charts[path.name].write_html(
                    temporary, include_plotlyjs=True, div_id=f"{prefix}-{path.stem}",
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return destinations
