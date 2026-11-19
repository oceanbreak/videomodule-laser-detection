"""Load local paths relative to the configuration file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from .calibration_settings import CalibrationSettings
from .detection_settings import DetectionSettings


@dataclass(frozen=True)
class Config:
    annotations_csv: Path
    setups_csv: Path
    images_dir: Path
    width: int
    height: int
    output_dir: Path
    calibration: CalibrationSettings = CalibrationSettings()
    calibration_output_dir: Path = Path("results/calibration")
    detection: DetectionSettings = DetectionSettings()
    detection_output_dir: Path = Path("results/detection_baselines")
    series_csv: Optional[Path] = None


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Read an explicit config or find config.local.yaml above the working directory."""
    if path is None:
        cwd = Path.cwd()
        path = next(
            (folder / "config.local.yaml" for folder in (cwd, *cwd.parents)
             if (folder / "config.local.yaml").is_file()),
            cwd / "config.local.yaml",
        )
    path = Path(path).expanduser().resolve()
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read configuration {path}: {error}") from error

    def setting(section, key):
        try:
            return content[section][key]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: missing setting {section}.{key}") from error

    def local_path(section, key):
        value = setting(section, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {section}.{key} must be a nonempty path")
        return (path.parent / Path(value).expanduser()).resolve()

    def dimension(key):
        value = setting("image", key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{path}: image.{key} must be a positive integer")
        return value

    annotations_csv = local_path("data", "annotations_csv")
    setups_csv = local_path("data", "setups_csv")
    images_dir = local_path("data", "images_dir")
    output_dir = local_path("output", "directory")

    try:
        calibration = CalibrationSettings(**content.get("calibration", {}))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid calibration settings: {error}") from error

    calibration_output = (path.parent / "results/calibration").resolve()
    if "calibration_directory" in content["output"]:
        calibration_output = local_path("output", "calibration_directory")

    try:
        detection = DetectionSettings(**content.get("detection", {}))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid detection settings: {error}") from error

    detection_output = (path.parent / "results/detection_baselines").resolve()
    if "detection_directory" in content["output"]:
        detection_output = local_path("output", "detection_directory")
    series_csv = None
    if content.get("data", {}).get("series_csv") is not None:
        series_csv = local_path("data", "series_csv")

    return Config(
        annotations_csv=annotations_csv,
        setups_csv=setups_csv,
        images_dir=images_dir,
        width=dimension("width"),
        height=dimension("height"),
        output_dir=output_dir,
        calibration=calibration,
        calibration_output_dir=calibration_output,
        detection=detection,
        detection_output_dir=detection_output,
        series_csv=series_csv,
    )
