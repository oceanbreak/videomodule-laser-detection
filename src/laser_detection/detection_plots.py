"""Plotly comparisons and on-demand Matplotlib diagnostics."""

from io import BytesIO

import cv2
import numpy as np
import plotly.express as px
from matplotlib.figure import Figure

from .detectors.base import scale_centers
from .evaluation import RECALL_THRESHOLDS


def detection_figures(metrics, timings, split):
    scores = metrics.loc[metrics["split"].eq(split)]
    columns = ["precision", "recall", "f1", *(f"recall_at_{t}px" for t in RECALL_THRESHOLDS)]
    quality = scores.melt(id_vars="method", value_vars=columns, var_name="metric", value_name="score")
    stages = ["resize_mean_ms", "threshold_mean_ms", "components_mean_ms", "total_mean_ms"]
    durations = timings.melt(id_vars="method", value_vars=stages, var_name="stage", value_name="ms")
    return {
        "quality_comparison": px.bar(
            quality, x="metric", y="score", color="method", barmode="group",
            range_y=[0, 1], title=f"Pair detection and localization · {split}",
        ),
        "timing_comparison": px.bar(
            durations, x="stage", y="ms", color="method", barmode="group",
            title="Mean serial pipeline timing · all readable frames",
        ),
    }


def diagnostic_figure(image, row, detectors):
    """Render one frame, comparing masks and predicted and annotated point centers."""
    detectors = tuple(detectors)
    figure = Figure(figsize=(14, 4.8), layout="constrained")
    axes = figure.subplots(1, len(detectors) + 1)
    first, *others = axes
    reduced = detectors[0].resize(image)
    first.imshow(cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB))
    first.set_title("Reduced image")
    original_size = image.shape[1::-1]
    truth = np.array([[row.x1, row.y1], [row.x2, row.y2]])

    def add_points(axis, points, size, label, color, marker):
        if np.isfinite(points).all():
            x, y = scale_centers(points, original_size, size).T
            axis.scatter(x, y, edgecolors=color, facecolors="none", marker=marker,
                         s=85, linewidths=1.6, label=label)

    add_points(first, truth, detectors[0].size, "Annotation", "white", "o")
    for axis, detector, color, marker in zip(others, detectors, ("cyan", "yellow"), ("s", "D")):
        prediction = detector.detect(image, return_mask=True)
        points = np.array(prediction.coordinates).reshape(2, 2)
        axis.imshow(prediction.mask, cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"{detector.name} mask · {prediction.component_count} components")
        add_points(axis, truth, detector.size, "Annotation", "orange", "o")
        add_points(axis, points, detector.size, detector.name, color, marker)
        add_points(first, points, detectors[0].size, detector.name, color, marker)
    for axis in axes:
        height, width = axis.images[0].get_array().shape[:2]
        axis.set_xlim(-0.5, width - 0.5)
        axis.set_ylim(height - 0.5, -0.5)
        axis.set_axis_off()
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper right")
    return figure


def diagnostic_png(image, row, detectors):
    figure = diagnostic_figure(image, row, detectors)
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png")
        return buffer.getvalue()
