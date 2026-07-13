from typing import Dict, Tuple

import torch
from torch import nn


MODEL_NAMES = ["lenet300100", "smallconv", "vggsmall"]
MODEL_DEFAULTS: Dict[str, str] = {
    "mnist": "lenet300100",
    "fashion_mnist": "lenet300100",
    "cifar10": "vggsmall",
}


class LeNet300100(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        channels, height, width = input_shape
        flat_dim = channels * height * width
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallConvNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            sample = torch.zeros(1, channels, height, width)
            feat_dim = int(self.features(sample).numel())
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class VGGSmall(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        channels, _, _ = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(
    model_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
) -> nn.Module:
    if model_name == "lenet300100":
        return LeNet300100(input_shape=input_shape, num_classes=num_classes)
    if model_name == "smallconv":
        return SmallConvNet(input_shape=input_shape, num_classes=num_classes)
    if model_name == "vggsmall":
        return VGGSmall(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f"Unknown model_name: {model_name}. Expected one of {MODEL_NAMES}")
