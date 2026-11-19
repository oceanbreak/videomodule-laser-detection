"""CPU/CUDA training, validation, and resumable local runs."""

import json
import random
import re
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..check_environment import _cuda_smoke_test
from .data import HeatmapDataset, training_split
from .model import UNet, heatmap_loss
from .settings import Profile


def select_device(requested):
    if requested == "cpu" or (requested == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run python -m laser_detection.check_environment --require-cuda")
    try:
        index = torch.cuda.current_device()
        _cuda_smoke_test(torch, index)
    except Exception as error:
        raise RuntimeError(
            f"CUDA forward/backward failed: {error}. "
            "Run python -m laser_detection.check_environment --require-cuda."
        ) from error
    return torch.device("cuda", index)


def _epoch(model, loader, profile, device, *, epoch, learning_rate, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    processed = 0
    phase = "Training" if training else "Validation"
    with torch.set_grad_enabled(training), tqdm(
        total=len(loader), desc=f"Epoch {epoch} · {phase}", unit="batch",
        disable=not profile.show_progress,
    ) as progress:
        for image, target, _ in loader:
            image, target = image.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = heatmap_loss(model(image), target, profile)
            value = loss.item()
            if not np.isfinite(value):
                raise RuntimeError("Non-finite training or validation loss")
            if training:
                loss.backward()
                optimizer.step()
            processed += len(image)
            total += value * len(image)
            if profile.show_progress:
                progress.set_postfix(loss=f"{total / processed:.6f}", lr=f"{learning_rate:.3g}", refresh=False)
                progress.update()
    return total / processed


def _save_checkpoint(path, state):
    temporary = path.with_suffix(".pt.tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported U-Net checkpoint format")
    return state


def load_model(checkpoint, device):
    state = read_checkpoint(checkpoint)
    profile = Profile(**state["profile"])
    model = UNet(profile.base_channels)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), profile, state["split_digest"], state["original_size"]


def split_digest(splits):
    return sha256(splits.sort_values("image").to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _new_run(root, name):
    if name is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Run name must contain only letters, numbers, hyphens or underscores")
    path = root / name
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def train_run(frame, config, profile, *, run_name=None, resume=None):
    """Train or resume from the last checkpoint. Epochs is the total target count."""
    device = select_device(profile.device)
    torch.manual_seed(profile.seed)
    random.seed(profile.seed)
    np.random.seed(profile.seed % 2**32)
    generator = torch.Generator().manual_seed(profile.seed)

    state = None
    if resume is not None:
        last = Path(resume).resolve()
        if last.name != "last.pt" or run_name is not None:
            raise ValueError("Resume requires last.pt and uses its existing directory")
        run = last.parent
        state = read_checkpoint(last)
        previous = Profile(**state["profile"])
        if replace(previous, epochs=profile.epochs, device=profile.device,
                   num_workers=profile.num_workers, show_progress=profile.show_progress) != profile:
            raise ValueError("Resume must keep training settings except epochs, device and num_workers")
        if profile.epochs <= state["epoch"]:
            raise ValueError("Set epochs above the checkpoint epoch to continue")
        splits = training_split(frame, config, profile.seed, saved=run / "dataset_split.csv")
        if state["split_digest"] != split_digest(splits) or state["original_size"] != [config.width, config.height]:
            raise ValueError("The saved split or original image size differs from the checkpoint")
        if not (run / "best.pt").is_file():
            raise ValueError("Cannot resume without the run's best.pt")
    else:
        splits = training_split(frame, config, profile.seed)
        run = _new_run(config.unet_output_dir, run_name)
        splits.to_csv(run / "dataset_split.csv", index=False)
    profile_settings = asdict(profile)
    profile_settings.pop("show_progress")  # Display preference is not part of saved training state.
    (run / "experiment_config.json").write_text(json.dumps({
        "profile": profile_settings, "resolved_device": str(device),
        "original_size": [config.width, config.height], "torch": str(torch.__version__),
        "numpy": np.__version__, "opencv": cv2.__version__,
        "protocol": splits["protocol"].iloc[0],
    }, indent=2) + "\n", encoding="utf-8")
    digest = split_digest(splits)
    assigned = frame.merge(splits[["image", "split"]], on="image", validate="one_to_one")

    def loader(partition):
        rows = assigned.loc[assigned["split"].eq(partition)]
        training = partition == "train"
        return DataLoader(HeatmapDataset(rows, config, profile), batch_size=profile.batch_size,
                          num_workers=profile.num_workers, shuffle=training,
                          generator=generator if training else None)

    train, validation = loader("train"), loader("validation")
    model = UNet(profile.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile.learning_rate)
    history, best, stale = [], float("inf"), 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        history = state["history"]
        best, stale = state["best_loss"], state["stale"]
        # An explicit resume can continue after a prior early stop.
        if stale >= profile.patience:
            stale = 0

    for epoch in range(len(history) + 1, profile.epochs + 1):
        started = perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = _epoch(model, train, profile, device, epoch=epoch,
                            learning_rate=learning_rate, optimizer=optimizer)
        val_loss = _epoch(model, validation, profile, device, epoch=epoch,
                          learning_rate=learning_rate)
        seconds = perf_counter() - started
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": val_loss, "seconds": seconds})
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={seconds:.2f}s")
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        checkpoint = {
            "format_version": 1, "epoch": epoch, "profile": profile_settings,
            "split_digest": digest, "original_size": [config.width, config.height],
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_loss": best, "stale": stale, "history": history,
            "loader_rng": generator.get_state(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        if improved:
            _save_checkpoint(run / "best.pt", checkpoint)
        _save_checkpoint(run / "last.pt", checkpoint)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        if stale >= profile.patience:
            break
    return run
