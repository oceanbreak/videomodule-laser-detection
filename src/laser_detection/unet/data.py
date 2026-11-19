"""Local, lazy image loading and heatmap targets."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..benchmarking import read_image
from ..data import COORDINATES
from ..detectors.base import scale_centers
from ..evaluation import dataset_split

PARTITIONS = ("train", "validation", "test")


def make_heatmap(coordinates, profile):
    """Render two input-grid spots. Missing annotations produce an empty target."""
    width, height = profile.image_size
    target = np.zeros((height, width), dtype=np.float32)
    points = np.asarray(coordinates).reshape(2, 2)
    if not np.isfinite(points).all():
        return target
    y, x = np.ogrid[:height, :width]
    for cx, cy in points.astype(np.float32):
        distance = (x.astype(np.float32) - cx) ** 2 + (y.astype(np.float32) - cy) ** 2
        if profile.target == "binary_circle":
            spot = distance <= profile.radius ** 2
        else:
            spot = np.exp(-distance / (2 * profile.sigma ** 2))
        np.maximum(target, spot, out=target)
    return target


def image_tensor(bgr, size):
    reduced = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255)


class HeatmapDataset(Dataset):
    def __init__(self, frame, config, profile):
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.profile = profile

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        bgr = read_image(row["image"], self.config)
        coordinates = scale_centers(row[COORDINATES].to_numpy(dtype=float).reshape(2, 2),
                                    bgr.shape[1::-1], self.profile.image_size)
        target = make_heatmap(coordinates, self.profile)
        meta = {"image": row["image"], "setup_id": row["setup_id"]}
        return image_tensor(bgr, self.profile.image_size), torch.from_numpy(target[None]), meta


def _read_split(path, frame):
    split = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["image", "series_id", "split"]
    if not set(required).issubset(split):
        raise ValueError(f"{path}: expected columns {required}")
    if split["image"].duplicated().any() or set(split["image"]) != set(frame["image"]):
        raise ValueError(f"{path}: split must match each annotation exactly once")
    return frame[["image"]].merge(split, on="image", validate="one_to_one")


def _find_split(config):
    if config.dataset_split_csv is not None:
        return config.dataset_split_csv
    root = config.detection_output_dir
    legacy = root / "dataset_split.csv"
    if legacy.is_file():
        return legacy
    candidates = sorted(root.glob("*/dataset_split.csv"))
    if len(candidates) > 1:
        raise ValueError("Multiple saved splits found. Set data.dataset_split_csv explicitly.")
    return candidates[0] if candidates else None


def training_split(frame, config, seed, *, saved=None):
    """Keep saved partitions, or assign complete series and label frame-level splits."""
    path = Path(saved) if saved is not None else _find_split(config)
    grouped = config.series_csv is not None
    if path is not None:
        split = _read_split(path, frame)
        if grouped:
            known = dataset_split(frame, config)["series_id"]
            if split["series_id"].ne("").any() and not split["series_id"].eq(known).all():
                raise ValueError("Saved split and capture series disagree")
            split["series_id"] = known
    else:
        split = dataset_split(frame, config).fillna({"series_id": ""})

    if split["split"].eq("exploratory").all():
        if grouped:
            split = dataset_split(frame, config)
        else:
            if len(split) < 3:
                raise ValueError("Need at least three frames for train/validation/test")
            indices = np.random.default_rng(seed).permutation(len(split))
            count = max(1, round(0.2 * len(split)))
            split["split"] = "train"
            split.loc[indices[:count], "split"] = "test"
            split.loc[indices[count:2 * count], "split"] = "validation"

    if set(split["split"]) != set(PARTITIONS):
        raise ValueError("Expected nonempty train, validation and test partitions")
    series = split["series_id"]
    if series.ne("").any():
        if series.eq("").any() or split.groupby("series_id")["split"].nunique().gt(1).any():
            raise ValueError("Every frame must have a series, and series cannot cross splits")
    split["protocol"] = "capture_series" if series.ne("").all() else "exploratory"
    return split[["image", "series_id", "split", "protocol"]]
