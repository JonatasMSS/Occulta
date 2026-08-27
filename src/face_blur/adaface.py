from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import List

import torch
from torch import nn


Block = namedtuple("Block", ["in_channel", "depth", "stride"])


def _stage(in_channel: int, depth: int, units: int) -> List[Block]:
    return [Block(in_channel, depth, 2)] + [Block(depth, depth, 1)] * (units - 1)


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class AdaFaceBackbone(nn.Module):
    """Backbone IR-101 usado pelo checkpoint AdaFace WebFace12M."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )
        blocks = (
            _stage(64, 64, 3)
            + _stage(64, 128, 13)
            + _stage(128, 256, 30)
            + _stage(256, 512, 3)
        )
        self.body = nn.Sequential(*(BasicBlockIR(*block) for block in blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.body(self.input_layer(x)))


class AdaFaceIR101(nn.Module):
    """Wrapper que mantém os nomes ``net.*`` presentes no checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.net = AdaFaceBackbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_adaface(model_path: Path, device: torch.device) -> AdaFaceIR101:
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint AdaFace não encontrado: {model_path}")

    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except TypeError:  # Compatibilidade com versões antigas do PyTorch.
        checkpoint = torch.load(str(model_path), map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("O checkpoint AdaFace não contém um state_dict válido.")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    model = AdaFaceIR101()
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
