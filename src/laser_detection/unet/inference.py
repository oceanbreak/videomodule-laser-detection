"""Synchronized inference and a paired comparison with HSV Legacy."""

import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
import pandas as pd
import torch

from ..benchmarking import ImageDataError, read_image, timing_summary
from ..data import COORDINATES
from ..detectors.hsv import HSVDetector
from ..evaluation import detection_metrics, score_predictions
from .data import image_tensor, training_split
from .model import peak_centers
from .training import _new_run, load_model, select_device, split_digest


class Predictor:
    name = "unet"

    def __init__(self, checkpoint, device):
        self.device = select_device(device)
        self.checkpoint = Path(checkpoint).resolve()
        self.model, self.profile, self.split_digest, self.original_size = load_model(
            self.checkpoint, self.device,
        )

    @torch.inference_mode()
    def predict(self, bgr):
        # Exclude previous GPU work from this frame's timing.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter_ns()
        image = image_tensor(bgr, self.profile.image_size)[None].to(self.device)
        probability = self.model(image).sigmoid()[0, 0].cpu().numpy()
        coordinates, scores = peak_centers(probability, self.profile, bgr.shape[1::-1])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return coordinates, scores, probability, perf_counter_ns() - start


def compare_checkpoint(frame, config, predictor, *, run_name=None, partition="test"):
    """Evaluate both methods on exactly the same readable frames and export a new run."""
    checkpoint = predictor.checkpoint
    splits = training_split(frame, config, predictor.profile.seed,
                            saved=checkpoint.parent / "dataset_split.csv")
    if split_digest(splits) != predictor.split_digest or [config.width, config.height] != predictor.original_size:
        raise ValueError("The saved split or original image size differs from the checkpoint")
    selected = frame.merge(splits.loc[splits["split"].eq(partition)], on="image", validate="one_to_one")
    if selected.empty:
        raise ValueError(f"No frames in {partition}")
    size = (config.detection.width, config.detection.height)
    hsv = HSVDetector(config.detection.hsv_legacy, name="hsv_legacy", width=size[0], height=size[1])
    records = []
    readable = 0
    for row in selected.itertuples(index=False):
        common = {"image": row.image, "setup_id": row.setup_id}
        try:
            image = read_image(row.image, config)
        except ImageDataError as error:
            records.extend({**common, "method": method, "status": "image_error",
                            "message": str(error)} for method in ("unet", "hsv_legacy"))
            continue
        if not readable:  # One unmeasured warm-up on the same hardware.
            predictor.predict(image)
            hsv.detect(image)
        coordinates, confidence, _, duration = predictor.predict(image)
        baseline = hsv.detect(image)
        records.append({
            **common, "method": "unet", "status": "ok", "device": str(predictor.device),
            "detected": bool(np.isfinite(coordinates).all()), **dict(zip(COORDINATES, coordinates)),
            "confidence1": confidence[0], "confidence2": confidence[1], "total_ns": duration,
        })
        records.append({
            **common, "method": "hsv_legacy", "status": "ok", "device": "cpu",
            "detected": baseline.detected, **dict(zip(COORDINATES, baseline.coordinates)),
            **{stage: getattr(baseline, stage) for stage in
               ("resize_ns", "threshold_ns", "components_ns", "total_ns")},
        })
        readable += 1
    if not readable:
        raise ValueError(f"No readable {partition} frames")
    predictions = pd.DataFrame(records)
    scored = score_predictions(predictions, frame, splits)
    protocol = splits["protocol"].iloc[0]
    scored["protocol"] = protocol
    timing = timing_summary(scored)
    timing["device"] = timing["method"].map({"unet": str(predictor.device), "hsv_legacy": "cpu"})
    timing["input_size"] = timing["method"].map({
        "unet": "x".join(map(str, predictor.profile.image_size)), "hsv_legacy": "x".join(map(str, size)),
    })
    summary = detection_metrics(scored).assign(protocol=protocol)
    output = _new_run(checkpoint.parent / "evaluations", run_name)
    tables = {
        "predictions": scored, "metrics": summary, "timings": timing,
        "metrics_by_setup": detection_metrics(scored, by_setup=True).assign(protocol=protocol),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "evaluation_config.json").write_text(json.dumps({
        "checkpoint": checkpoint.name, "partition": partition, "protocol": protocol,
        "unet": asdict(predictor.profile), "unet_device": str(predictor.device),
        "hsv_legacy": asdict(config.detection.hsv_legacy), "hsv_input_size": size,
        "torch": str(torch.__version__), "opencv": cv2.__version__, "platform": platform.platform(),
        "torch_cpu_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
        "cuda_device": torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else None,
    }, indent=2) + "\n", encoding="utf-8")
    return output, tables
