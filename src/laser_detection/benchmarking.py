"""Process one local image at a time and record serial detector timings."""

import platform
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import COORDINATES

STAGES = ("resize_ns", "threshold_ns", "components_ns", "total_ns")


class ImageDataError(ValueError):
    """A dataset image is missing, unreadable, or has incorrect dimensions."""


def read_image(filename, config):
    root = config.images_dir.resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ImageDataError(f"Image path is outside the configured directory: {filename}")
    try:
        encoded = path.read_bytes()
        image = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        ) if encoded else None
    except (OSError, cv2.error) as error:
        raise ImageDataError(f"Cannot read image {filename}: {error}") from error
    if image is None:
        raise ImageDataError(f"Cannot decode image {filename}")
    expected = (config.width, config.height)
    if image.shape[1::-1] != expected:
        raise ImageDataError(f"{filename}: expected {expected}, found {image.shape[1::-1]}")
    return image


@contextmanager
def _opencv_threads(threads):
    previous = cv2.getNumThreads()
    cv2.setNumThreads(threads)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def _environment():
    cpu = platform.processor()
    if not cpu:
        try:
            lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            cpu = next((line.split(":", 1)[1].strip() for line in lines
                        if line.startswith("model name") and ":" in line), "")
        except OSError:
            pass
    return {
        "cpu": cpu or platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "opencv_optimized": cv2.useOptimized(),
        "opencv_build": cv2.getBuildInformation(),
    }


def _warmup(frame, detectors, config):
    count = 0
    for filename in frame["image"]:
        if count == config.detection.warmup_frames:
            break
        try:
            image = read_image(filename, config)
        except ImageDataError:
            continue
        for detector in detectors:
            detector.detect(image)
        count += 1
    return count


def benchmark_dataset(frame: pd.DataFrame, detectors, config):
    """Run both methods serially per image, alternating the first method per readable frame."""
    detectors = tuple(detectors)
    columns = ["image", "setup_id", "method", "status", "message", "detected",
               "component_count", "area1", "area2", *COORDINATES, *STAGES]
    results = []
    with _opencv_threads(config.detection.opencv_threads):
        environment = _environment()
        environment["warmup_frames_completed"] = _warmup(frame, detectors, config)
        environment["order"] = "serial per frame, alternating first detector"
        readable = 0
        for row in frame.itertuples(index=False):
            common = {"image": row.image, "setup_id": row.setup_id}
            try:
                image = read_image(row.image, config)
            except ImageDataError as error:
                results.extend({**common, "method": detector.name, "status": "image_error",
                                "message": str(error)} for detector in detectors)
                continue
            order = detectors if readable % 2 == 0 else reversed(detectors)
            for detector in order:
                result = detector.detect(image)
                results.append({
                    **common, "method": detector.name, "status": "ok", "message": "",
                    **dict(zip(COORDINATES, result.coordinates)),
                    **{name: getattr(result, name) for name in
                       ("detected", "component_count", "area1", "area2", *STAGES)},
                })
            readable += 1
    predictions = pd.DataFrame(results, columns=columns)
    predictions["detected"] = predictions["detected"].astype("boolean")
    integers = ["component_count", "area1", "area2", *STAGES]
    predictions[integers] = predictions[integers].astype("Int64")
    return predictions, environment


def timing_summary(predictions):
    rows = []
    for method, recorded in predictions.groupby("method", sort=False):
        measured = recorded.loc[recorded["status"].eq("ok")]
        row = {"method": method, "frames": len(measured)}
        for stage in STAGES:
            milliseconds = measured[stage].astype(float) / 1_000_000
            name = stage.removesuffix("_ns")
            row.update({f"{name}_mean_ms": milliseconds.mean(),
                        f"{name}_median_ms": milliseconds.median(),
                        f"{name}_p95_ms": milliseconds.quantile(0.95)})
        row["fps"] = 1000 / row["total_mean_ms"] if row["total_mean_ms"] > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
