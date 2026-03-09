import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from neuron_kill.model import MLP


def compute_importance(
    model: MLP, data_loader: torch.utils.data.DataLoader, device: torch.device
) -> List[torch.Tensor]:
    model.eval()
    sizes = model.hidden_sizes()
    sums = [torch.zeros(size, device=device) for size in sizes]
    total = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _, activations = model(x, return_activations=True)
            batch = x.size(0)
            total += batch
            for idx, act in enumerate(activations):
                sums[idx] += act.abs().sum(dim=0)

    mean_abs = [s / max(total, 1) for s in sums]
    outgoing_norms: List[torch.Tensor] = []
    for idx in range(len(sizes)):
        if idx < len(sizes) - 1:
            next_weight = model.layers[idx + 1].weight
        else:
            next_weight = model.out.weight
        outgoing_norms.append(next_weight.norm(dim=0))

    importances = [mean_abs[i] * outgoing_norms[i] for i in range(len(sizes))]
    return importances


def _prune_layer(model: MLP, layer_idx: int, keep_idx: torch.Tensor) -> None:
    layer = model.layers[layer_idx]
    device = layer.weight.device
    in_features = layer.in_features
    out_features = keep_idx.numel()

    # Keep only selected neurons (rows) while preserving their incoming weights.
    new_layer = nn.Linear(in_features, out_features, bias=True).to(device)
    new_layer.weight.data.copy_(layer.weight.data[keep_idx, :])
    new_layer.bias.data.copy_(layer.bias.data[keep_idx])
    model.layers[layer_idx] = new_layer

    if layer_idx < len(model.layers) - 1:
        next_layer = model.layers[layer_idx + 1]
        # Remove columns that fed into the pruned neurons.
        new_next = nn.Linear(out_features, next_layer.out_features, bias=True).to(device)
        new_next.weight.data.copy_(next_layer.weight.data[:, keep_idx])
        new_next.bias.data.copy_(next_layer.bias.data)
        model.layers[layer_idx + 1] = new_next
    else:
        out_layer = model.out
        # Same column removal for the output projection.
        new_out = nn.Linear(out_features, out_layer.out_features, bias=True).to(device)
        new_out.weight.data.copy_(out_layer.weight.data[:, keep_idx])
        new_out.bias.data.copy_(out_layer.bias.data)
        model.out = new_out


def _rand_uniform(
    rng: np.random.Generator, shape: Tuple[int, ...], bound: float
) -> np.ndarray:
    return rng.uniform(-bound, bound, size=shape)


def _rand_normal(
    rng: np.random.Generator, shape: Tuple[int, ...], scale: float
) -> np.ndarray:
    return rng.normal(scale=scale, size=shape)


def _init_new_rows(layer: nn.Linear, start: int, rng: np.random.Generator) -> None:
    fan_in = layer.in_features
    bound = 1.0 / math.sqrt(fan_in)
    weight_shape = (layer.out_features - start, layer.in_features)
    bias_shape = (layer.out_features - start,)
    weight = _rand_uniform(rng, weight_shape, bound)
    bias = _rand_uniform(rng, bias_shape, bound)
    layer.weight.data[start:] = torch.tensor(
        weight, device=layer.weight.device, dtype=layer.weight.dtype
    )
    layer.bias.data[start:] = torch.tensor(
        bias, device=layer.bias.device, dtype=layer.bias.dtype
    )


def _init_new_cols(layer: nn.Linear, start: int, rng: np.random.Generator) -> None:
    fan_in = layer.in_features
    bound = 1.0 / math.sqrt(fan_in)
    weight_shape = (layer.out_features, layer.in_features - start)
    weight = _rand_uniform(rng, weight_shape, bound)
    layer.weight.data[:, start:] = torch.tensor(
        weight, device=layer.weight.device, dtype=layer.weight.dtype
    )


def _grow_layer_random(
    model: MLP, layer_idx: int, add_count: int, rng: np.random.Generator
) -> None:
    if add_count <= 0:
        return

    layer = model.layers[layer_idx]
    device = layer.weight.device
    old_out = layer.out_features
    new_out = old_out + add_count

    # Append randomly initialized neurons while keeping existing rows untouched.
    new_layer = nn.Linear(layer.in_features, new_out, bias=True).to(device)
    new_layer.weight.data[:old_out] = layer.weight.data
    new_layer.bias.data[:old_out] = layer.bias.data
    _init_new_rows(new_layer, old_out, rng)
    model.layers[layer_idx] = new_layer

    if layer_idx < len(model.layers) - 1:
        next_layer = model.layers[layer_idx + 1]
        # Append new columns so downstream weights for old neurons are preserved.
        new_next = nn.Linear(new_out, next_layer.out_features, bias=True).to(device)
        new_next.weight.data[:, :old_out] = next_layer.weight.data
        new_next.bias.data.copy_(next_layer.bias.data)
        _init_new_cols(new_next, old_out, rng)
        model.layers[layer_idx + 1] = new_next
    else:
        out_layer = model.out
        # Append new columns in the output projection.
        new_out_layer = nn.Linear(new_out, out_layer.out_features, bias=True).to(device)
        new_out_layer.weight.data[:, :old_out] = out_layer.weight.data
        new_out_layer.bias.data.copy_(out_layer.bias.data)
        _init_new_cols(new_out_layer, old_out, rng)
        model.out = new_out_layer


def _grow_layer_split(
    model: MLP,
    layer_idx: int,
    add_count: int,
    source_indices: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float = 0.02,
) -> None:
    if add_count <= 0:
        return

    layer = model.layers[layer_idx]
    device = layer.weight.device
    old_out = layer.out_features
    new_out = old_out + add_count

    # Duplicate strong neurons with small noise; original weights stay intact.
    new_layer = nn.Linear(layer.in_features, new_out, bias=True).to(device)
    new_layer.weight.data[:old_out] = layer.weight.data
    new_layer.bias.data[:old_out] = layer.bias.data

    for offset, src in enumerate(source_indices):
        dst = old_out + offset
        weight_noise = _rand_normal(
            rng, tuple(layer.weight.data[src].shape), noise_scale
        )
        bias_noise = _rand_normal(rng, tuple(layer.bias.data[src].shape), noise_scale)
        new_layer.weight.data[dst] = layer.weight.data[src] + torch.tensor(
            weight_noise, device=device, dtype=layer.weight.dtype
        )
        new_layer.bias.data[dst] = layer.bias.data[src] + torch.tensor(
            bias_noise, device=device, dtype=layer.bias.dtype
        )

    model.layers[layer_idx] = new_layer

    if layer_idx < len(model.layers) - 1:
        next_layer = model.layers[layer_idx + 1]
        # Copy downstream columns from the source neurons with perturbation.
        new_next = nn.Linear(new_out, next_layer.out_features, bias=True).to(device)
        new_next.weight.data[:, :old_out] = next_layer.weight.data
        new_next.bias.data.copy_(next_layer.bias.data)

        for offset, src in enumerate(source_indices):
            dst = old_out + offset
            noise = _rand_normal(
                rng, tuple(next_layer.weight.data[:, src].shape), noise_scale
            )
            new_next.weight.data[:, dst] = next_layer.weight.data[:, src] + torch.tensor(
                noise, device=device, dtype=next_layer.weight.dtype
            )

        model.layers[layer_idx + 1] = new_next
    else:
        out_layer = model.out
        # Same copying for the output layer when splitting the last hidden layer.
        new_out_layer = nn.Linear(new_out, out_layer.out_features, bias=True).to(device)
        new_out_layer.weight.data[:, :old_out] = out_layer.weight.data
        new_out_layer.bias.data.copy_(out_layer.bias.data)

        for offset, src in enumerate(source_indices):
            dst = old_out + offset
            noise = _rand_normal(
                rng, tuple(out_layer.weight.data[:, src].shape), noise_scale
            )
            new_out_layer.weight.data[:, dst] = out_layer.weight.data[:, src] + torch.tensor(
                noise, device=device, dtype=out_layer.weight.dtype
            )

        model.out = new_out_layer


def prune_and_grow(
    model: MLP,
    importances: List[torch.Tensor],
    mode: str,
    prune_fraction: float = 0.1,
    min_neurons: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], List[int]]:
    if mode == "fixed":
        return [0 for _ in model.hidden_sizes()], [0 for _ in model.hidden_sizes()]

    if rng is None:
        raise ValueError("rng must be provided for reproducible structural updates")

    sizes = model.hidden_sizes()
    prune_counts: List[int] = []
    keep_indices: List[torch.Tensor] = []
    for size, importance in zip(sizes, importances):
        max_prune = max(size - min_neurons, 0)
        prune_count = min(int(size * prune_fraction), max_prune)
        prune_counts.append(prune_count)
        if prune_count == 0:
            keep_indices.append(torch.arange(size, device=importance.device))
            continue
        keep_count = size - prune_count
        topk = torch.topk(importance, k=keep_count, largest=True)
        keep_idx = torch.sort(topk.indices).values
        keep_indices.append(keep_idx)

    for layer_idx, keep_idx in enumerate(keep_indices):
        if keep_idx.numel() < sizes[layer_idx]:
            _prune_layer(model, layer_idx, keep_idx)

    if mode == "prune_only":
        return prune_counts, [0 for _ in prune_counts]

    grown_counts: List[int] = []
    for layer_idx, prune_count in enumerate(prune_counts):
        if prune_count == 0:
            grown_counts.append(0)
            continue

        if mode == "prune_grow_random":
            _grow_layer_random(model, layer_idx, prune_count, rng)
            grown_counts.append(prune_count)
            continue

        if mode == "prune_grow_split":
            keep_idx = keep_indices[layer_idx]
            kept_importance = importances[layer_idx][keep_idx].detach().cpu().numpy()
            if kept_importance.size == 0:
                grown_counts.append(0)
                continue
            top_k = max(1, int(0.2 * kept_importance.size))
            source_pool = np.argsort(kept_importance)[-top_k:]
            source_indices = rng.choice(source_pool, size=prune_count, replace=True)
            _grow_layer_split(model, layer_idx, prune_count, source_indices, rng)
            grown_counts.append(prune_count)
            continue

        raise ValueError(f"Unknown mode: {mode}")

    return prune_counts, grown_counts
