from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: Sequence[int] = (64, 64),
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev, size))
            prev = size
        self.out = nn.Linear(prev, output_dim)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
        ablate: Optional[Tuple[int, int]] = None,
    ):
        activations = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            if ablate is not None and ablate[0] == idx:
                x = x.clone()
                x[:, ablate[1]] = 0.0
            if return_activations:
                activations.append(x)
        x = self.out(x)
        if return_activations:
            return x, activations
        return x

    def hidden_sizes(self) -> List[int]:
        return [layer.out_features for layer in self.layers]
