from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn


PRUNABLE_MODEL_NAMES = ["lenet300100", "smallconv"]


class PrunableLeNet300100(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        hidden_sizes: Sequence[int] = (300, 100),
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        flat_dim = channels * height * width
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        prev_dim = flat_dim
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_dim, hidden_size))
            prev_dim = hidden_size
        self.out = nn.Linear(prev_dim, num_classes)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        ablate: Optional[Tuple[int, int]] = None,
    ):
        x = self.flatten(x)
        activations = []
        for layer_idx, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            if ablate is not None and ablate[0] == layer_idx:
                x = x.clone()
                x[:, ablate[1]] = 0.0
            if return_activations:
                activations.append(x)
        logits = self.out(x)
        if return_activations:
            return logits, activations
        return logits

    def hidden_sizes(self) -> List[int]:
        return [layer.out_features for layer in self.layers]


class PrunableSmallConvNet(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        channels_per_layer: Sequence[int] = (64, 64, 128, 128),
        classifier_hidden: int = 256,
    ) -> None:
        super().__init__()
        in_channels, height, width = input_shape
        c1, c2, c3, c4 = channels_per_layer
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1),
                nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            ]
        )
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.pool_after_layers = {1, 3}

        with torch.no_grad():
            sample = torch.zeros(1, in_channels, height, width)
            features = self._forward_features(sample)
            self.feature_shape = tuple(features.shape[1:])
            flattened_dim = int(features.numel())

        self.fc1 = nn.Linear(flattened_dim, classifier_hidden)
        self.fc2 = nn.Linear(classifier_hidden, num_classes)

    def _forward_features(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        ablate: Optional[Tuple[int, int]] = None,
    ):
        activations = []
        for layer_idx, conv in enumerate(self.conv_layers):
            x = self.activation(conv(x))
            if ablate is not None and ablate[0] == layer_idx:
                x = x.clone()
                x[:, ablate[1]] = 0.0
            if return_activations:
                activations.append(x)
            if layer_idx in self.pool_after_layers:
                x = self.pool(x)
        if return_activations:
            return x, activations
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        ablate: Optional[Tuple[int, int]] = None,
    ):
        if return_activations:
            x, activations = self._forward_features(x, return_activations=True, ablate=ablate)
            x = torch.flatten(x, start_dim=1)
            x = self.activation(self.fc1(x))
            logits = self.fc2(x)
            return logits, activations
        x = self._forward_features(x, ablate=ablate)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)

    def hidden_sizes(self) -> List[int]:
        return [layer.out_channels for layer in self.conv_layers]


def build_prunable_model(
    model_name: str,
    input_shape: Tuple[int, int, int],
    num_classes: int,
) -> nn.Module:
    if model_name == "lenet300100":
        return PrunableLeNet300100(input_shape=input_shape, num_classes=num_classes)
    if model_name == "smallconv":
        return PrunableSmallConvNet(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(
        f"Structured init-only pruning currently supports {PRUNABLE_MODEL_NAMES}, got {model_name!r}"
    )
