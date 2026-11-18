from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest

from laser_detection.benchmarking import benchmark_dataset, timing_summary
from laser_detection.config import Config, load_config
from laser_detection.data import COORDINATES
from laser_detection.detection_plots import diagnostic_png
from laser_detection.detection_settings import DetectionSettings, HLSSettings, RedBoostSettings
from laser_detection.detectors import HLSDetector, RedBoostDetector, baseline_detectors
from laser_detection.detectors.base import scale_centers
from laser_detection.detectors.blobs import find_components
from laser_detection.detectors.red_boost import red_boost
from laser_detection.evaluation import dataset_split, detection_metrics, score_predictions


def test_hls_bgr_conversion_and_hue_wrap():
    image = np.array([[[0, 0, 255], [20, 0, 255], [255, 0, 0], [150, 150, 150], [0, 0, 20]]],
                     dtype=np.uint8)
    assert cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[0].tolist() == [
        [0, 128, 255], [178, 128, 255], [120, 128, 255], [0, 150, 0], [0, 10, 255],
    ]
    mask = HLSDetector().binary_mask(image)
    assert mask.dtype == np.uint8
    assert mask.tolist() == [[255, 255, 0, 0, 0]]


def test_red_boost_signed_extremes_and_strict_threshold():
    image = np.array([[[255, 255, 0], [0, 0, 255], [255, 255, 255],
                       [1, 0, 40], [0, 0, 40], [1, 0, 41]]], dtype=np.uint8)
    np.testing.assert_array_equal(red_boost(image), [[-255, 255, 0, 39.5, 40, 40.5]])
    mask = RedBoostDetector().binary_mask(image)
    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 255, 0, 0, 0, 255]]
    negative = RedBoostDetector(RedBoostSettings(-20)).binary_mask(image[:, :1])
    assert negative.tolist() == [[0]]


def test_components_area_centers_and_tie_break():
    mask = np.zeros((12, 15), dtype=np.uint8)
    mask[1, 1] = mask[2, 2] = 255  # Diagonal counts as one component.
    mask[0:2, 10:13] = 255
    mask[6:8, 3:6] = 255
    mask[9:11, 7:10] = 255
    components = find_components(mask)
    assert components.count == 4
    assert sorted(components.areas) == [2, 6, 6, 6]
    np.testing.assert_allclose(components.centroids[components.areas == 2], [[1.5, 1.5]])
    labels = np.asarray(components.selected_labels)
    np.testing.assert_allclose(components.centroids[labels - 1], [[4, 6.5], [11, 0.5]])
    assert find_components(mask).selected_labels == components.selected_labels


@pytest.mark.parametrize("detector_class", [HLSDetector, RedBoostDetector])
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_pipeline_component_counts_and_source_coordinates(detector_class, count):
    detector = detector_class(width=12, height=8)
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    patches = [(slice(0, 4), slice(12, 16)), (slice(8, 12), slice(4, 10)),
               (slice(14, 16), slice(22, 24))]
    for y, x in patches[:count]:
        image[y, x] = [0, 0, 255]
    result = detector.detect(image, return_mask=True)
    assert result.component_count == count
    assert result.detected == (count >= 2)
    assert result.mask.shape == (8, 12)
    assert result.total_ns >= result.resize_ns + result.threshold_ns + result.components_ns
    assert detector.detect(image).mask is None
    if count < 2:
        assert np.isnan(result.coordinates).all()
        assert result.area1 is None and result.area2 is None
    else:
        assert result.coordinates == (6.5, 9.5, 13.5, 1.5)
        assert (result.area1, result.area2) == (6, 4)


def test_equal_x_center_sort_and_single_pixel_components():
    mask = np.zeros((10, 10), np.uint8)
    mask[1, 5] = mask[7, 5] = 255
    components = find_components(mask)
    assert components.selected_labels == (1, 2)
    assert components.areas.tolist() == [1, 1]
    assert components.centroids.tolist() == [[5, 1], [5, 7]]
    np.testing.assert_allclose(scale_centers([[0, 0], [1, 2]], (4, 4), (12, 8)),
                               [[1, 0.5], [4, 4.5]])
    np.testing.assert_allclose(scale_centers([[1, 0.5], [4, 4.5]], (12, 8), (4, 4)),
                               [[0, 0], [1, 2]])


def test_metrics_include_missed_pairs_and_exclude_unreadable_frames():
    truth = pd.DataFrame([
        ["good", 0, 0, 100, 0], ["poor", 0, 0, 100, 0], ["miss", 0, 0, 100, 0],
        ["false", *([np.nan] * 4)], ["empty", *([np.nan] * 4)], ["unreadable", 0, 0, 100, 0],
    ], columns=["image", *COORDINATES])
    predictions = pd.DataFrame([
        ["good", True, 3, 4, 100, 0], ["poor", True, 0, 0, 100, 12],
        ["miss", False, *([np.nan] * 4)], ["false", True, 0, 0, 100, 0],
        ["empty", False, *([np.nan] * 4)], ["unreadable", pd.NA, *([np.nan] * 4)],
    ], columns=["image", "detected", *COORDINATES]).assign(
        method="hls", setup_id="S", status=["ok"] * 5 + ["image_error"],
    )
    splits = truth[["image"]].assign(split="test", series_id="series")
    scored = score_predictions(predictions, truth, splits)
    assert scored["outcome"].tolist() == ["tp", "tp", "fn", "fp", "tn", "unscored"]
    assert scored.loc[2:, ["error1_px", "error2_px"]].isna().all().all()
    row = detection_metrics(scored, by_setup=True).iloc[0]
    assert (row.tp, row.fp, row.fn, row.tn, row.image_errors) == (2, 1, 1, 1, 1)
    assert row.precision == row.recall == row.f1 == pytest.approx(2 / 3)
    assert row.mean_point_error_px == pytest.approx(4.25)
    assert row.median_point_error_px == pytest.approx(2.5)
    assert row.p95_point_error_px == pytest.approx(10.95)
    assert row.recall_at_5px == row.recall_at_10px == pytest.approx(1 / 3)
    assert row.recall_at_20px == pytest.approx(2 / 3)


@pytest.fixture
def config(tmp_path):
    return Config(tmp_path / "lasers.csv", tmp_path / "setups.csv", tmp_path / "images",
                  24, 16, tmp_path / "inspection", detection=DetectionSettings(width=12, height=8),
                  detection_output_dir=tmp_path / "detection")


def test_series_split_is_grouped_reproducible_and_independent_of_setup(config, tmp_path):
    path = tmp_path / "series.csv"
    names = [f"{i}.png" for i in range(10)]
    path.write_text("image;series_id\n" + "".join(f"{name};{i // 2}\n" for i, name in enumerate(names)))
    frame = pd.DataFrame({"image": names, "setup_id": "same_setup"})
    grouped = replace(config, series_csv=path)
    split = dataset_split(frame, grouped)
    assert split["split"].value_counts().to_dict() == {"train": 6, "validation": 2, "test": 2}
    assert split.groupby("series_id")["split"].nunique().eq(1).all()
    shuffled = dataset_split(frame.iloc[::-1], grouped)
    pd.testing.assert_frame_equal(split.sort_values("image").reset_index(drop=True),
                                  shuffled.sort_values("image").reset_index(drop=True))
    assert dataset_split(frame, config)["split"].eq("exploratory").all()
    with pytest.raises(ValueError, match="series mapping"):
        dataset_split(frame.iloc[:-1], grouped)


def test_benchmark_matches_cohorts_restores_threads_and_renders(config):
    config.images_dir.mkdir()
    frame = pd.DataFrame([
        ["one.png", "S", 5, 6, 15, 8], ["missing.png", "S", *([np.nan] * 4)],
    ], columns=["image", "setup_id", *COORDINATES])
    bgr = np.zeros((16, 24, 3), dtype=np.uint8)
    bgr[4:8, 4:8, 2] = bgr[6:10, 14:18, 2] = 255
    cv2.imwrite(str(config.images_dir / "one.png"), bgr)
    detectors = baseline_detectors(config.detection)
    threads = cv2.getNumThreads()
    predictions, environment = benchmark_dataset(frame, detectors, config)
    assert cv2.getNumThreads() == threads
    assert len(predictions) == 8
    assert predictions.groupby("method")["status"].apply(list).tolist() == [["ok", "image_error"]] * 4
    assert environment["warmup_frames_completed"] == 1
    assert environment["opencv_threads"] == 1 and environment["cpu"]
    timing = timing_summary(predictions)
    assert (timing["fps"] > 0).all() and timing["frames"].eq(1).all()
    assert diagnostic_png(bgr, frame.iloc[0], detectors).startswith(b"\x89PNG")


def test_detection_configuration(tmp_path):
    source = tmp_path / "config.local.yaml"
    source.write_text("""data:
  annotations_csv: lasers.csv
  setups_csv: setups.csv
  images_dir: images
  series_csv: series.csv
image: {width: 3072, height: 2048}
output: {directory: results/inspection}
detection:
  hls: {hue_range_1: [2, 9]}
  red_boost: {boost_threshold: 20.5}
""")
    config = load_config(source)
    assert config.detection.width == 1536
    assert config.detection.hls.hue_range_1 == (2, 9)
    assert config.detection.red_boost.boost_threshold == 20.5
    assert config.series_csv == tmp_path / "series.csv"
    assert config.detection_output_dir == tmp_path / "results/detection_baselines"
    with pytest.raises(ValueError, match="hue_range_1"):
        HLSSettings(hue_range_1=(0, 181))
    with pytest.raises(ValueError, match="lower bound"):
        HLSSettings(lightness_min=180, lightness_max=100)
    with pytest.raises(ValueError, match="boost_threshold"):
        RedBoostSettings(float("nan"))
