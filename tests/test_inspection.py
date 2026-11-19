import os

import numpy as np
import pytest
from PIL import Image

from laser_detection.config import load_config
from laser_detection.data import load_annotations, load_dataset
from laser_detection.export import export_inspection
from laser_detection.geometry import METRICS, add_geometry
from laser_detection.statistics import dataset_summary, setup_summary
from laser_detection.visualization import inspect_image, inspection_figures


@pytest.fixture
def config(tmp_path):
    settings = tmp_path / "config.local.yaml"
    settings.write_text("""
data:
  annotations_csv: lasers.csv
  setups_csv: laser_setups.csv
  images_dir: photos
image:
  width: 3072
  height: 2048
output:
  directory: results/inspection
""")
    config = load_config(settings)
    config.annotations_csv.write_text(
        "image;x1;y1;x2;y2\nz.png;1;2;4;6\na.png;;;;\nm.png;4;6;1;2\n"
    )
    config.setups_csv.write_text("filename;setup\nz.png;A\nm.png;B\n")
    return config


def test_config_paths(config, tmp_path, monkeypatch):
    assert config.annotations_csv == tmp_path / "lasers.csv"
    assert config.output_dir == tmp_path / "results" / "inspection"
    config.output_dir.parent.mkdir()
    monkeypatch.chdir(config.output_dir.parent)
    assert load_config() == config


def test_setup_boundaries_and_missing_coordinates(config):
    frame = load_dataset(config)
    assert frame["image"].tolist() == ["z.png", "a.png", "m.png"]
    assert frame["setup_id"].tolist() == ["A", "A", "B"]
    assert frame.loc[1, ["x1", "y1", "x2", "y2"]].isna().all()


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ("z.png;A\nunknown.png;B\n", "missing from annotations"),
        ("z.png;A\nz.png;B\n", "duplicate filename"),
        ("z.png;A\nm.png;B\na.png;C\n", "row order"),
        ("a.png;A\n", "first annotation row"),
    ],
)
def test_invalid_boundaries(config, boundaries, message):
    config.setups_csv.write_text("filename;setup\n" + boundaries)
    with pytest.raises(ValueError, match=message):
        load_dataset(config)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("a;3072;0;1;1", "x1="),
        ("a;0;2048;1;1", "y1="),
        ("a;-1;0;1;1", "x1="),
        ("a;0;0;inf;1", "x2="),
        ("a;no;0;1;1", "x1="),
        ("a;;0;1;1", "all four"),
        ("a;1;2;3", "expected 5 fields"),
        ("a;1;2;3;4\na;1;2;3;4", "duplicate image"),
    ],
)
def test_invalid_annotations(config, row, message):
    config.annotations_csv.write_text("image;x1;y1;x2;y2\n" + row + "\n")
    with pytest.raises(ValueError, match=message):
        load_annotations(config.annotations_csv)


def test_missing_csv_and_columns(config):
    config.annotations_csv.unlink()
    with pytest.raises(ValueError, match="lasers.csv"):
        load_dataset(config)
    config.annotations_csv.write_text("image;x1\na;1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(config)


def test_geometry_and_swap(config):
    annotations = load_dataset(config)
    frame = add_geometry(annotations)
    assert frame[annotations.columns].equals(annotations)
    assert frame.loc[0, METRICS].tolist() == pytest.approx(
        [2.5, 4, 3, 4, 5, np.arctan2(4, 3), np.arctan2(4, 3)]
    )
    assert frame.loc[1, METRICS].isna().all()
    assert frame.loc[2, ["dx", "dy", "d", "theta", "theta_lr"]].tolist() == pytest.approx(
        [3, 4, 5, np.arctan2(4, 3), frame.at[0, "theta_lr"]]
    )


def test_vertical_swap(config):
    annotations = load_dataset(config)
    annotations.loc[[0, 2], ["x1", "x2"]] = 8
    frame = add_geometry(annotations)
    assert frame.loc[[0, 2], "theta_lr"].tolist() == pytest.approx([np.pi / 2] * 2)


def test_frame_view_with_and_without_local_image(config):
    frame = add_geometry(load_dataset(config))
    figure, details, message = inspect_image(frame, "z.png", config)
    assert figure is None
    assert details["d"] == 5
    assert "Image unavailable" in message

    config.images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(config.images_dir / "z.png")
    figure, details, message = inspect_image(frame, "z.png", config)
    assert message is None
    assert figure.data[0].source.startswith("data:image/png;base64,")
    assert list(figure.data[1].x) == [1, 4]
    assert list(figure.data[1].y) == [2, 6]


def test_statistics_plot_groups_and_local_export(config):
    frame = add_geometry(load_dataset(config))
    assert dataset_summary(frame).iloc[0].to_dict() == {
        "frames": 3, "labeled": 2, "unlabeled": 1, "setups": 2,
    }
    assert setup_summary(frame)["missing_fraction"].tolist() == [0.5, 0]
    figures = inspection_figures(frame, config)
    positions = figures["laser_positions"]
    assert list(positions.layout.yaxis.range) == [2048, 0]
    assert positions.layout.yaxis.scaleanchor == "x"
    assert positions.layout.legend.groupclick == "togglegroup"
    assert [trace.legendgroup for trace in positions.data] == ["A", "A", "B", "B"]
    assert [trace.showlegend for trace in positions.data] == [True, False, True, False]
    paths = export_inspection(frame, figures, config)
    assert len(paths) == 9
    assert all(path.parent == config.output_dir for path in paths)
    assert b"inspection-laser_positions" in (config.output_dir / "laser_positions.html").read_bytes()
    before = [path.read_bytes() for path in paths]
    export_inspection(frame, figures, config)
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("link_method", ["hardlink", "symlink"])
def test_export_protects_source_csv(config, link_method):
    frame = add_geometry(load_dataset(config))
    original = config.annotations_csv.read_bytes()
    config.output_dir.mkdir(parents=True)
    destination = config.output_dir / "geometry.csv"
    try:
        if link_method == "hardlink":
            os.link(config.annotations_csv, destination)
        else:
            destination.symlink_to(config.annotations_csv)
    except OSError as error:
        if link_method == "symlink" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows Developer Mode or elevated privileges")
        raise
    with pytest.raises(ValueError, match="Unsafe export destination|overwrite an input CSV"):
        export_inspection(frame, {}, config)
    assert config.annotations_csv.read_bytes() == original
    assert not (config.output_dir / "dataset_summary.csv").exists()
