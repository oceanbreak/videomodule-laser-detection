"""Plotly visualizations for the calibration study."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .calibration_experiments import corridor_metric
from .visualization import _color, _image_axes, inspect_image

LASER_COLORS = {"left": "#1f77b4", "right": "#d62728"}


def clipped_line(line, width, height, offset=0):
    """Intersect a parallel offset of the line with the closed image rectangle."""
    normal = np.array([line.a, line.b])
    origin = (offset - line.c) * normal
    direction = np.array([-line.b, line.a])
    lower, upper = -np.inf, np.inf
    for start, slope, limit in zip(origin, direction, (width, height)):
        if slope == 0:
            if not 0 <= start <= limit:
                return None
        else:
            entry, exit_ = sorted((-start / slope, (limit - start) / slope))
            lower, upper = max(lower, entry), min(upper, exit_)
    if lower > upper:
        return None
    return np.clip(origin + np.array([[lower], [upper]]) * direction, 0, [width, height])


def _add_lines(figure, model, config, corridor=None):
    for side, color in LASER_COLORS.items():
        line = getattr(model, side)
        offsets = [0] if corridor is None else [0, -corridor, corridor]
        for offset in offsets:
            segment = clipped_line(line, config.width, config.height, offset)
            if segment is not None:
                figure.add_trace(go.Scatter(
                    x=segment[:, 0], y=segment[:, 1], mode="lines",
                    name=f"{side} fit" if not offset else f"{side} corridor ±{corridor:g} px",
                    line=dict(color=color, dash="dash" if offset else "solid"),
                    legendgroup=f"{side}-corridor" if offset else f"{side}-fit",
                    showlegend=offset <= 0,
                ))


def spatial_calibration(study, setup, config, corridor):
    rows = study.residuals.loc[study.residuals["setup_id"].eq(setup)]
    figure = go.Figure()
    for index, (side, color) in enumerate(LASER_COLORS.items(), start=1):
        inlier = rows[f"{side}_inlier"]
        for label, mask, symbol in (
            ("inliers", inlier.eq(True), "circle"),
            ("outliers", inlier.eq(False), "x"),
            ("unscored", inlier.isna(), "diamond"),
        ):
            subset = rows.loc[mask]
            if not subset.empty:
                figure.add_trace(go.Scattergl(
                    x=subset[f"x{index}"], y=subset[f"y{index}"], text=subset["image"],
                    mode="markers", name=f"{side} {label}", marker=dict(color=color, symbol=symbol),
                    customdata=subset[f"{side}_residual"],
                    hovertemplate="%{text}<br>(%{x}, %{y})<br>Residual: %{customdata:.2f} px",
                ))
    if setup in study.models:
        _add_lines(figure, study.models[setup], config, corridor)
    else:
        figure.add_annotation(text="No fitted model", xref="paper", yref="paper", x=0.5, y=0.95)
    _image_axes(figure, config.width, config.height)
    figure.update_layout(title=f"Setup {setup} · in-sample · corridor ±{corridor:g} px")
    return figure


def inspect_calibration_image(frame, filename, study, config):
    figure, details, message = inspect_image(frame, filename, config)
    if figure is not None and details["setup_id"] in study.models:
        _add_lines(figure, study.models[details["setup_id"]], config)
    return figure, details, message


def largest_residuals(study, setup, threshold, count=20):
    columns = ["image", "setup_id", "x1", "y1", "x2", "y2",
               "left_residual", "right_residual", "pair_residual"]
    rows = study.residuals.loc[
        study.residuals["setup_id"].eq(setup) & study.residuals["pair_residual"].notna(), columns
    ]
    return rows.nlargest(count, "pair_residual").assign(
        passes_corridor=lambda result: result["pair_residual"].le(threshold),
    )


def residual_ecdf(study):
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Left laser", "Right laser"])
    for index, (setup, rows) in enumerate(study.residuals.groupby("setup_id", sort=False)):
        for column, side in enumerate(LASER_COLORS, start=1):
            data = np.sort(rows[f"{side}_residual"].dropna())
            if data.size:
                figure.add_trace(go.Scatter(
                    x=np.r_[0, data], y=np.arange(len(data) + 1) / len(data),
                    name=str(setup), legendgroup=str(setup), showlegend=column == 1,
                    line=dict(color=_color(index), shape="hv"),
                ), row=1, col=column)
    figure.update_xaxes(title_text="Perpendicular residual (px)")
    figure.update_yaxes(title_text="Cumulative fraction", range=[0, 1])
    figure.update_layout(title="Residual distributions · in-sample", legend_title="Setup")
    return figure


def corridor_coverage(study):
    figure = go.Figure()
    for index, (setup, rows) in enumerate(study.corridors.groupby("setup_id", sort=False)):
        figure.add_trace(go.Scatter(
            x=rows["threshold"], y=rows["pair"], name=str(setup),
            mode="lines+markers", line=dict(color=_color(index)),
        ))
    figure.update_layout(
        title="Paired corridor coverage · in-sample", legend_title="Setup",
        xaxis_title="Corridor half-width (px)", yaxis_title="Fraction of pairs",
        yaxis_range=[0, 1],
    )
    return figure


def _repeated_figure(summary, metrics, x, title, y_label):
    figure = make_subplots(rows=1, cols=len(metrics), subplot_titles=list(metrics.values()))
    for index, (setup, group) in enumerate(summary.groupby("setup_id", sort=False)):
        for column, metric in enumerate(metrics, start=1):
            rows = group.loc[group["metric"].eq(metric)].sort_values(x)
            values = rows["median"]
            figure.add_trace(go.Scatter(
                x=rows[x], y=values, name=str(setup), mode="lines+markers",
                legendgroup=str(setup), showlegend=column == 1, line=dict(color=_color(index)),
                error_y=dict(type="data", symmetric=False, array=rows["p95"] - values,
                             arrayminus=values - rows["p05"]),
            ), row=1, col=column)
    figure.update_layout(title=title, legend_title="Setup")
    figure.update_xaxes(title_text="Training pairs" if x == "training_size" else "Contaminated fraction")
    figure.update_yaxes(title_text=y_label)
    return figure


def calibration_figures(study, config):
    thresholds = config.calibration.corridor_thresholds
    displayed = [t for t in thresholds if t in (10, 20)] or list(thresholds[:2])
    figures = {
        "residual_ecdf": residual_ecdf(study),
        "corridor_coverage": corridor_coverage(study),
        "sample_size_residual": _repeated_figure(
            study.sample_summary, {"test_pair_median": "Maximum of left and right residuals"},
            "training_size", "Holdout residual by sample size", "Median residual (px)",
        ),
        "sample_size_coverage": _repeated_figure(
            study.sample_summary, {corridor_metric("pair", t): f"±{t:g} px" for t in displayed},
            "training_size", "Holdout coverage by sample size", "Fraction of pairs",
        ),
        "contamination_residual": _repeated_figure(
            study.contamination_summary, {"test_pair_median": "Maximum of left and right residuals"},
            "contamination_fraction", "Holdout residual with synthetic training contamination",
            "Median residual (px)",
        ),
    }
    corridor = 20 if 20 in thresholds else thresholds[0]
    # Numeric suffixes avoid collisions and filesystem-dependent setup IDs.
    for index, setup in enumerate(study.parameters["setup_id"], start=1):
        figures[f"spatial_setup_{index:02d}"] = spatial_calibration(study, setup, config, corridor)
    return figures
