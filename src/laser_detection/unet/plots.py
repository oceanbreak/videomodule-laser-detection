"""On-demand visualization of targets and predictions."""

from io import BytesIO

import numpy as np
from matplotlib.figure import Figure

from ..detectors.base import scale_centers


def heatmap_figure(rgb, heatmap, *, coordinates=None, original_size=None):
    figure = Figure(figsize=(9, 4), layout="constrained")
    image_axis, heat_axis = figure.subplots(1, 2)
    image_axis.imshow(rgb)
    heat_axis.imshow(heatmap, vmin=0, vmax=1, cmap="magma")
    image_axis.set_title("RGB input")
    heat_axis.set_title("Heatmap")
    if coordinates is not None and np.isfinite(coordinates).all():
        x, y = scale_centers(np.asarray(coordinates).reshape(2, 2),
                             original_size, heatmap.shape[::-1]).T
        for axis in (image_axis, heat_axis):
            axis.scatter(x, y, facecolors="none", edgecolors="cyan", s=75)
    return figure


def figure_png(figure):
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png")
        return buffer.getvalue()
