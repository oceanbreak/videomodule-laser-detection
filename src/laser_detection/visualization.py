"""Interactive plots and on-demand inspection of a single local image."""

import base64
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.colors import qualitative

from .config import Config
from .data import COORDINATES
from .geometry import METRICS

AXIS_LABELS = {
    "yc": "Vertical pair center (px)",
    "d": "Distance (px)",
    "dx": "Horizontal displacement (px)",
    "dy": "Vertical displacement (px)",
    "theta_lr": "Left-to-right angle (rad)",
}


def _color(index):
    if index < len(qualitative.Plotly):
        return qualitative.Plotly[index]
    return f"hsl({index * 137.508 % 360:.3f}, 65%, 45%)"


def _image_axes(figure, width, height):
    figure.update_xaxes(title="X (px)", range=[0, width], constrain="domain")
    figure.update_yaxes(
        title="Y (px)", range=[height, 0], scaleanchor="x", scaleratio=1,
        constrain="domain",
    )


def _spatial(groups, width, height):
    figure = go.Figure()
    for setup, rows, color in groups:
        for laser, symbol in [(1, "circle"), (2, "diamond")]:
            figure.add_trace(go.Scattergl(
                x=rows[f"x{laser}"], y=rows[f"y{laser}"], text=rows["image"],
                mode="markers", name=str(setup), legendgroup=str(setup),
                showlegend=laser == 1, marker=dict(symbol=symbol, color=color),
                hovertemplate=(
                    "%{text}<br>Setup %{fullData.name}"
                    f"<br>Laser {laser}"
                    "<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
                ),
            ))
    _image_axes(figure, width, height)
    figure.update_layout(
        title="Laser positions · 1 circle / 2 diamond",
        legend=dict(title="Setup", groupclick="togglegroup"),
    )
    return figure


def _scatter(groups, metric):
    figure = go.Figure()
    for setup, rows, color in groups:
        figure.add_trace(go.Scattergl(
            x=rows["yc"], y=rows[metric], mode="markers", name=str(setup),
            text=rows["image"], customdata=rows[[*COORDINATES, "d"]].to_numpy(),
            marker=dict(color=color),
            hovertemplate=(
                "%{text}<br>Setup %{fullData.name}"
                "<br>Laser 1: (%{customdata[0]:.2f}, %{customdata[1]:.2f})"
                "<br>Laser 2: (%{customdata[2]:.2f}, %{customdata[3]:.2f})"
                "<br>Distance: %{customdata[4]:.2f} px"
                "<br>%{yaxis.title.text}: %{y:.2f}<extra></extra>"
            ),
        ))
    figure.update_layout(
        xaxis_title=AXIS_LABELS["yc"], yaxis_title=AXIS_LABELS[metric],
        legend_title="Setup",
    )
    return figure


def _box(groups, metric):
    figure = go.Figure()
    for setup, rows, color in groups:
        figure.add_trace(go.Box(y=rows[metric], name=str(setup), marker_color=color))
    figure.update_layout(xaxis_title="Setup", yaxis_title=AXIS_LABELS[metric])
    return figure


def inspection_figures(frame: pd.DataFrame, config: Config) -> dict[str, go.Figure]:
    groups = [
        (setup, rows.dropna(subset=["d"]), _color(index))
        for index, (setup, rows) in enumerate(frame.groupby("setup_id", sort=False))
    ]
    return {
        "laser_positions": _spatial(groups, config.width, config.height),
        "distance_vs_yc": _scatter(groups, "d"),
        "distance_by_setup": _box(groups, "d"),
        "dx_vs_yc": _scatter(groups, "dx"),
        "dy_vs_yc": _scatter(groups, "dy"),
        "angle_by_setup": _box(groups, "theta_lr"),
    }


def inspect_image(frame: pd.DataFrame, filename: str, config: Config):
    """Return the figure, the frame's geometry, and an optional status message."""
    selection = frame.loc[frame["image"].eq(filename)]
    if selection.empty:
        return None, None, f"Unknown frame: {filename}"
    row = selection.iloc[0]
    details = row[["setup_id", *COORDINATES, *METRICS]]
    path = (config.images_dir / filename).resolve()
    if not path.is_relative_to(config.images_dir.resolve()):
        return None, details, f"Image path is outside the configured directory: {filename}"
    try:
        with Image.open(path) as image, BytesIO() as buffer:
            image.convert("RGB").save(buffer, format="PNG")
            source = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except (OSError, ValueError) as error:
        return None, details, f"Image unavailable: {path} ({error})"

    figure = go.Figure(go.Image(source=source, hoverinfo="skip"))
    if pd.notna(row["d"]):
        figure.add_trace(go.Scatter(
            x=[row["x1"], row["x2"]], y=[row["y1"], row["y2"]],
            mode="lines+markers", name="Laser pair",
            line=dict(color="red"), marker=dict(size=10, symbol=["circle", "diamond"]),
        ))
    _image_axes(figure, config.width, config.height)
    figure.update_layout(title=filename)
    return figure, details, None
