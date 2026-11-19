"""Four-resolution U-Net, loss functions and heatmap decoding."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..detectors.base import scale_centers


def _block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base_channels=8):
        super().__init__()
        widths = [base_channels * 2 ** i for i in range(4)]
        self.encoder = nn.ModuleList(_block(source, target)
                                     for source, target in zip([3, *widths], widths))
        self.decoder = nn.ModuleList(_block(source + target, target)
                                     for source, target in zip(widths[:0:-1], widths[-2::-1]))
        self.pool = nn.MaxPool2d(2)
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, image):
        features = self.encoder[0](image)
        skips = []
        for block in self.encoder[1:]:
            skips.append(features)
            features = block(self.pool(features))
        for block, skip in zip(self.decoder, reversed(skips)):
            features = F.interpolate(features, size=skip.shape[-2:], mode="nearest")
            features = block(torch.cat((features, skip), dim=1))
        return self.output(features)


def heatmap_loss(logits, target, profile):
    if profile.loss == "bce":
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=logits.new_tensor(profile.foreground_weight),
        )
    probability = logits.sigmoid()
    if profile.loss == "weighted_mse":
        weights = 1 + (profile.foreground_weight - 1) * target
        return ((probability - target).square() * weights).sum() / weights.sum()
    p, t = probability.flatten(1), target.flatten(1)
    dice = 1 - ((2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    if profile.loss == "dice":
        return dice
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=logits.new_tensor(profile.foreground_weight),
    )
    return bce + dice


def peak_centers(probability, profile, original_size):
    """Decode two peaks, using a circular suppression radius on the network grid."""
    local_max = cv2.dilate(probability, np.ones((3, 3), dtype=np.uint8))
    local_min = cv2.erode(probability, np.ones((3, 3), dtype=np.uint8))
    y, x = np.where((probability >= profile.confidence) & (probability == local_max)
                    & (probability > local_min))
    scores = probability[y, x]
    peaks = []
    for index in np.argsort(-scores, kind="stable"):
        point = (int(x[index]), int(y[index]), float(scores[index]))
        if not peaks or ((point[0] - peaks[0][0]) ** 2 + (point[1] - peaks[0][1]) ** 2
                         > profile.nms_radius ** 2):
            peaks.append(point)
        if len(peaks) == 2:
            peaks.sort()
            coordinates = scale_centers([p[:2] for p in peaks], probability.shape[::-1], original_size)
            return tuple(map(float, coordinates.ravel())), tuple(p[2] for p in peaks)
    return (float("nan"),) * 4, (float("nan"),) * 2
