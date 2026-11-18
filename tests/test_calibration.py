from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from laser_detection.calibration import (
    CalibrationError, LaserLines, Line, calculate_residuals,
    evaluate_corridors, fit_laser_lines, inlier_masks,
)
from laser_detection.calibration_experiments import (
    contaminate_pairs, run_calibration_study, split_positions,
)
from laser_detection.calibration_plots import (
    clipped_line, inspect_calibration_image, largest_residuals, spatial_calibration,
)
from laser_detection.calibration_settings import CalibrationSettings
from laser_detection.config import Config, load_config
from laser_detection.data import COORDINATES
from laser_detection.export import export_calibration
from laser_detection.geometry import add_geometry


@pytest.fixture
def pairs():
    rng = np.random.default_rng(71)
    y = np.linspace(50, 1800, 60)
    frame = pd.DataFrame({
        "image": [f"frame-{i}.png" for i in range(len(y))], "setup_id": "A",
        "x1": 240 + rng.normal(size=len(y)), "y1": y,
        "x2": 1700 - 0.4 * y + rng.normal(size=len(y)), "y2": y,
    })
    frame.loc[0, "x1"] = 600
    frame.loc[1, "x2"] = 2500
    frame.loc[2, COORDINATES] = np.nan
    return frame


@pytest.fixture
def config(tmp_path):
    return Config(
        tmp_path / "lasers.csv", tmp_path / "setups.csv", tmp_path / "photos",
        3072, 2048, tmp_path / "inspection",
        CalibrationSettings(training_sizes=(10, 20, 80), repetitions=3,
                            contamination_training_size=20),
        tmp_path / "calibration",
    )


def test_normal_form_distance_and_pair_logic():
    line = Line(-3, -4, 10)
    assert (line.a, line.b, line.c) == pytest.approx((0.6, 0.8, -2))
    assert line.distance([[0, 0], [0, 2.5], [0, 5]]) == pytest.approx([2, 0, 2])
    model = LaserLines(Line(1, 0, -20), Line(0, 1, -10))
    assert model.angle_degrees == 90
    points = pd.DataFrame([[22, 0, 50, 17], [24, 0, 50, 12]], columns=COORDINATES)
    residuals = calculate_residuals(points, model)
    assert residuals["pair"].tolist() == [7, 4]
    assert inlier_masks(residuals, 4).values.tolist() == [[True, False, False], [True, True, True]]
    assert evaluate_corridors(points, model, [4, 7])["pair"].tolist() == [0.5, 1.0]
    with pytest.raises(ValueError, match="nonzero normal"):
        Line(0, 0, 3)


def test_horizontal_and_vertical_models_and_clipping():
    xy = pd.DataFrame([[20, 2, 40, 60], [20, 50, 80, 60], [20, 60, 90, 60]],
                      columns=COORDINATES)
    model = fit_laser_lines(xy)
    np.testing.assert_allclose(calculate_residuals(xy, model), 0, atol=1e-10)
    np.testing.assert_allclose(clipped_line(model.left, 100, 100, 5), [[25, 0], [25, 100]])
    np.testing.assert_allclose(clipped_line(model.right, 100, 100), [[100, 60], [0, 60]])
    assert clipped_line(model.left, 100, 100, 100) is None


def test_noisy_lines_outliers_and_reproducibility(pairs, config):
    model = fit_laser_lines(pairs, config.calibration)
    assert model == fit_laser_lines(pairs, config.calibration)
    assert model.left.distance([[240, 50], [240, 1800]]).max() < 2
    assert model.right.distance([[1680, 50], [980, 1800]]).max() < 2
    residuals = calculate_residuals(pairs, model)
    masks = inlier_masks(residuals, config.calibration.ransac_threshold)
    assert masks.iloc[:3].values.tolist() == [
        [False, True, False], [True, False, False], [False, False, False],
    ]
    assert masks.iloc[3:]["pair"].all()


def test_holdout_uses_all_remaining_pairs(pairs, config):
    study = run_calibration_study(pairs, config)
    runs = study.sample_runs
    pd.testing.assert_frame_equal(runs, run_calibration_study(pairs, config).sample_runs)
    run = runs.loc[runs["status"].eq("ok")].iloc[0]
    clean = pairs.dropna(subset=COORDINATES)
    train, test = split_positions(len(clean), run.training_size, np.random.default_rng(run.split_seed))
    assert not set(train) & set(test)
    assert sorted([*train, *test]) == list(range(len(clean)))
    model = fit_laser_lines(clean.iloc[train], config.calibration, random_state=run.fit_seed)
    residuals = calculate_residuals(clean.iloc[test], model)
    assert run.test_pairs == len(test)
    assert run.test_pair_p95 == pytest.approx(residuals["pair"].quantile(0.95))
    assert run.pair_coverage_px_10 == pytest.approx(residuals["pair"].le(10).mean())
    assert runs.loc[runs.training_size.eq(80), "status"].tolist() == ["skipped"]
    assert study.summary.loc[0, "evaluation"] == "in_sample"
    assert run.evaluation == "out_of_sample"
    assert study.residuals["image"].tolist() == clean["image"].tolist()


def test_contamination_is_paired_and_preserves_holdout(pairs, config):
    study = run_calibration_study(pairs, config)
    rows = study.contamination_runs
    assert rows.groupby("repeat")["split_seed"].nunique().eq(1).all()
    assert rows.groupby("repeat")["fit_seed"].nunique().eq(1).all()
    run = rows.loc[rows.contamination_fraction.eq(0.30)].iloc[0]
    clean = pairs.dropna(subset=COORDINATES)
    train, test = split_positions(len(clean), run.training_size, np.random.default_rng(run.split_seed))
    training = clean.iloc[train]
    modified, count = contaminate_pairs(training, 0.30, 3072, 2048, np.random.default_rng(run.noise_seed))
    assert count == run.replaced_pairs == 6
    assert modified["image"].equals(training["image"])
    assert modified["x1"].le(modified["x2"]).all()
    assert modified[COORDINATES].ne(training[COORDINATES]).any(axis=1).sum() == count
    assert training.equals(clean.iloc[train])
    if run.status == "ok":
        model = fit_laser_lines(modified, config.calibration, random_state=run.fit_seed)
        assert run.test_pair_median == pytest.approx(
            calculate_residuals(clean.iloc[test], model)["pair"].median(),
        )


def test_degenerate_and_unlabeled_setups(pairs, config):
    pairs[COORDINATES] = [10.0, 11.0, 12.0, 13.0]
    with pytest.raises(CalibrationError, match="distinct points"):
        fit_laser_lines(pairs)
    unlabeled = pairs.iloc[:1].assign(setup_id="B").copy()
    unlabeled[COORDINATES] = np.nan
    study = run_calibration_study(pd.concat([pairs, unlabeled], ignore_index=True), config)
    assert study.summary["status"].tolist() == ["failed", "failed"]
    assert study.residuals["pair_inlier"].isna().all()
    assert study.sample_runs["test_pair_median"].isna().all()
    assert set(study.sample_runs["status"]) == {"failed", "skipped"}
    assert study.models == {}
    assert largest_residuals(study, "B", 20).empty


def test_spatial_plot_and_exports(pairs, config):
    study = run_calibration_study(pairs, config)
    figure = spatial_calibration(study, "A", config, 20)
    assert tuple(figure.layout.yaxis.range) == (2048, 0)
    assert figure.layout.yaxis.scaleanchor == "x"
    assert {"left outliers", "right outliers", "left fit", "right fit"} <= {
        trace.name for trace in figure.data
    }
    for trace in figure.data:
        assert all(0 <= x <= config.width for x in trace.x)
        assert all(0 <= y <= config.height for y in trace.y)
    paths = export_calibration(study, {"map": figure}, config)
    assert len(paths) == 10
    assert all(path.parent == config.calibration_output_dir for path in paths)
    before = [path.read_bytes() for path in paths]
    export_calibration(study, {"map": figure}, config)
    assert before == [path.read_bytes() for path in paths]


def test_config_defaults_and_validation(tmp_path):
    path = tmp_path / "config.local.yaml"
    path.write_text("""data:
  annotations_csv: lasers.csv
  setups_csv: setups.csv
  images_dir: photos
image: {width: 3072, height: 2048}
output: {directory: results/inspection}
calibration: {training_sizes: [20, 10], random_state: 9}
""")
    config = load_config(path)
    assert config.calibration.training_sizes == (10, 20)
    assert config.calibration.random_state == 9
    assert config.calibration_output_dir == tmp_path / "results/calibration"
    with pytest.raises(ValueError, match="corridor_thresholds"):
        replace(config.calibration, corridor_thresholds=(0,))


def test_outlying_frame_can_show_annotations_and_fits(pairs, config):
    frame = add_geometry(pairs)
    study = run_calibration_study(frame, config)
    outliers = largest_residuals(study, "A", 20)
    assert set(outliers.iloc[:2]["image"]) == {"frame-0.png", "frame-1.png"}
    assert not outliers.iloc[:2]["passes_corridor"].any()
    filename = outliers.iloc[0]["image"]
    figure, details, message = inspect_calibration_image(frame, filename, study, config)
    assert figure is None and details["setup_id"] == "A" and "Image unavailable" in message

    config.images_dir.mkdir()
    Image.new("RGB", (8, 8)).save(config.images_dir / filename)
    figure, _, message = inspect_calibration_image(frame, filename, study, config)
    assert message is None
    assert [trace.name for trace in figure.data[1:]] == ["Laser pair", "left fit", "right fit"]
