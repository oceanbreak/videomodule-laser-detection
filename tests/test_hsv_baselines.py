import cv2
import numpy as np
import pandas as pd
import pytest

from laser_detection.detection_plots import diagnostic_figure
from laser_detection.detection_settings import DetectionSettings, HSVSettings
from laser_detection.detectors import HSVDetector, baseline_detectors
from laser_detection.evaluation import localization_disagreements


def test_hsv_masks_match_reference_for_all_hues():
    hues = np.arange(180, dtype=np.uint8)
    hsv = np.empty((4, 180, 3), dtype=np.uint8)
    hsv[..., 0] = hues
    hsv[..., 1] = [[255], [80], [255], [200]]
    hsv[..., 2] = [[255], [255], [120], [220]]
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    np.testing.assert_array_equal(converted[0, :, 0], hues)
    first = cv2.inRange(converted, (0, 100, 150), (10, 255, 255))
    for detector, low, high in zip(baseline_detectors(DetectionSettings())[2:], (150, 170), (200, 179)):
        second = cv2.inRange(converted, (low, 100, 150), (high, 255, 255))
        mask = detector.binary_mask(bgr)
        np.testing.assert_array_equal(mask, cv2.bitwise_or(first, second))
        np.testing.assert_array_equal(np.flatnonzero(mask[0]), np.r_[0:11, low:180])
        assert mask.dtype == np.uint8
        assert mask[0, 179] == 255 and not mask[1:3].any()


def test_hsv_partial_overrides_preserve_variant_defaults():
    settings = DetectionSettings(hsv_narrow={"value_min": 200})
    assert settings.hsv_narrow.hue_range_2 == (170, 179)
    assert settings.hsv_narrow.value_min == 200
    assert settings.hsv_legacy.hue_range_2 == (150, 179)
    with pytest.raises(ValueError, match="hue_range_2"):
        HSVSettings(hue_range_2=(150, 180))


def test_hsv_uses_shared_pipeline_and_diagnostics_show_four_masks():
    hsv = np.zeros((20, 32, 3), dtype=np.uint8)
    hsv[0:4, 18:22] = [155, 255, 255]
    hsv[6:12, 4:10] = [179, 255, 255]
    hsv[18:20, 30:32] = [0, 255, 255]
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detectors = baseline_detectors(DetectionSettings(width=16, height=10))
    legacy, narrow = detectors[2:]
    assert isinstance(legacy, HSVDetector) and isinstance(narrow, HSVDetector)
    result = legacy.detect(bgr)
    assert result.coordinates == (6.5, 8.5, 19.5, 1.5)
    assert (result.component_count, result.area1, result.area2) == (3, 9, 4)
    assert narrow.detect(bgr).coordinates == (6.5, 8.5, 30.5, 18.5)
    hsv[18:20, 30:32] = 0
    result = narrow.detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert result.component_count == 1 and not result.detected
    assert np.isnan(result.coordinates).all()
    truth = pd.Series(dict(x1=6.5, y1=8.5, x2=19.5, y2=1.5))
    figure = diagnostic_figure(bgr, truth, detectors)
    assert len(figure.axes) == 5
    assert all(name in axis.get_title() for name, axis in
               zip(("hls", "red_boost", "hsv_legacy", "hsv_narrow"), figure.axes[1:]))


def test_paired_disagreements_include_misses_in_both_directions():
    rows = pd.DataFrame({
        "image": ["a", "a", "b", "b", "c", "c"],
        "setup_id": "01", "split": "test", "annotated": True, "status": "ok",
        "method": ["hsv_legacy", "red_boost"] * 3,
        "pair_max_error_px": [10, np.nan, 11, 5, 1, 2],
        "outcome": ["tp", "fn", "tp", "tp", "tp", "tp"],
    })
    result = localization_disagreements(rows, "hsv_legacy", "red_boost", "test")
    assert dict(zip(result["image"], result["better_method"])) == {"a": "hsv_legacy", "b": "red_boost"}


def test_export_four_methods_preserves_existing_runs_and_reloads_csv(tmp_path):
    from types import SimpleNamespace

    from laser_detection.evaluation import comparison_table, load_detection_comparison
    from laser_detection.export import export_detection

    config = SimpleNamespace(
        annotations_csv=tmp_path / "a.csv", setups_csv=tmp_path / "s.csv", series_csv=None,
        detection=DetectionSettings(), detection_output_dir=tmp_path, width=3072, height=2048,
    )
    names = [detector.name for detector in baseline_detectors(config.detection)]
    scored = pd.DataFrame({"image": "frame.png", "method": names, "x1": 15.5, "error1_px": 0.5})
    metrics = pd.DataFrame({
        "split": "test", "method": names, "recall": 1, "recall_at_5px": 0.5,
        "recall_at_10px": 0.75, "recall_at_20px": 0.9, "median_point_error_px": 3.0,
    })
    timings = pd.DataFrame({"method": names, "total_mean_ms": [2.0, 3.0, 4.0, 5.0]})
    old = tmp_path / "predictions_hls.csv"
    old.write_text("old predictions\n")
    args = (scored, scored[["image"]], metrics, metrics, timings, {}, {}, config)
    paths = export_detection(*args)
    assert old.read_text() == "old predictions\n"
    assert {path.name for path in paths} >= {f"predictions_{name}.csv" for name in names}
    pd.testing.assert_frame_equal(
        load_detection_comparison(tmp_path / "hsv_comparison", "test"),
        comparison_table(metrics, timings, "test"),
    )
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError, match="already exists"):
        export_detection(*args)
    assert before == {path: path.read_bytes() for path in paths}
