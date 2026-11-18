"""Load local paths relative to the configuration file."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    annotations_csv: Path
    setups_csv: Path
    images_dir: Path
    width: int
    height: int
    output_dir: Path


def load_config(path: str | Path | None = None) -> Config:
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

    return Config(
        annotations_csv=local_path("data", "annotations_csv"),
        setups_csv=local_path("data", "setups_csv"),
        images_dir=local_path("data", "images_dir"),
        width=dimension("width"),
        height=dimension("height"),
        output_dir=local_path("output", "directory"),
    )
