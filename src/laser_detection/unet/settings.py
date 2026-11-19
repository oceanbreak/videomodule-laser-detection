"""Configuration for the CPU and CUDA training profiles."""

from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class Profile:
    device: str = "cpu"
    image_size: tuple[int, int] = (384, 256)
    batch_size: int = 2
    base_channels: int = 8
    epochs: int = 5
    num_workers: int = 0
    target: str = "binary_circle"
    radius: float = 4
    sigma: float = 3
    loss: str = "bce_dice"
    foreground_weight: float = 20
    learning_rate: float = 0.001
    patience: int = 5
    seed: int = 42
    confidence: float = 0.5
    nms_radius: float = 8

    def __post_init__(self):
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")
        size = self.image_size
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
            type(value) is not int or value < 8 for value in size
        ):
            raise ValueError("image_size must be [width, height], each at least 8")
        object.__setattr__(self, "image_size", tuple(size))
        for name in ("batch_size", "base_channels", "epochs", "num_workers", "patience", "seed"):
            value = getattr(self, name)
            minimum = 0 if name in ("num_workers", "seed") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("radius", "sigma", "foreground_weight", "learning_rate", "confidence", "nms_radius"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.foreground_weight < 1 or self.confidence > 1:
            raise ValueError("foreground_weight must be >= 1 and confidence must be <= 1")
        losses = {"binary_circle": ("bce", "dice", "bce_dice"), "gaussian": ("weighted_mse",)}
        if self.target not in losses or self.loss not in losses[self.target]:
            raise ValueError("Use bce, dice or bce_dice for binary_circle and weighted_mse for gaussian")


@dataclass(frozen=True)
class UNetSettings:
    cpu: Profile = Profile()
    cuda: Profile = Profile(device="cuda", image_size=(768, 512), base_channels=16,
                           epochs=40, num_workers=2)

    def __post_init__(self):
        for name, default in (("cpu", Profile()), ("cuda", type(self).cuda)):
            profile = getattr(self, name)
            if isinstance(profile, dict):
                profile = replace(default, **profile)
                object.__setattr__(self, name, profile)
            if not isinstance(profile, Profile):
                raise ValueError(f"unet.{name} must be a mapping")
