from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuron_survival_dynamics.init_only.prunable_models import (
    PrunableLeNet300100,
    PrunableSmallConvNet,
)


def _mean_abs_per_unit(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim == 2:
        return activation.abs().sum(dim=0)
    if activation.ndim == 4:
        return activation.abs().sum(dim=(0, 2, 3))
    raise ValueError(f"Unsupported activation shape: {tuple(activation.shape)}")


def _normalize_activation_sum(total_sum: torch.Tensor, total_examples: int, activation_ndim: int, spatial_size: int) -> torch.Tensor:
    if activation_ndim == 2:
        denom = max(total_examples, 1)
    elif activation_ndim == 4:
        denom = max(total_examples * spatial_size, 1)
    else:
        raise ValueError(f"Unsupported activation ndim: {activation_ndim}")
    return total_sum / denom


def _linear_outgoing_norm(next_weight: torch.Tensor) -> torch.Tensor:
    return next_weight.norm(dim=0)


def _conv_input_channel_norm(weight: torch.Tensor) -> torch.Tensor:
    in_channels = weight.shape[1]
    return weight.permute(1, 0, 2, 3).reshape(in_channels, -1).norm(dim=1)


def compute_importance_components(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    model.eval()
    sizes = model.hidden_sizes()
    activation_sums = [torch.zeros(size, device=device) for size in sizes]
    activation_ndims = [None for _ in sizes]
    spatial_sizes = [1 for _ in sizes]
    total_examples = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _, activations = model(x, return_activations=True)
            batch_size = x.size(0)
            total_examples += batch_size
            for layer_idx, activation in enumerate(activations):
                activation_sums[layer_idx] += _mean_abs_per_unit(activation)
                activation_ndims[layer_idx] = activation.ndim
                if activation.ndim == 4:
                    spatial_sizes[layer_idx] = int(activation.shape[2] * activation.shape[3])

    mean_abs = [
        _normalize_activation_sum(
            total_sum=activation_sums[layer_idx],
            total_examples=total_examples,
            activation_ndim=int(activation_ndims[layer_idx]),
            spatial_size=spatial_sizes[layer_idx],
        )
        for layer_idx in range(len(sizes))
    ]

    if isinstance(model, PrunableLeNet300100):
        outgoing_norms: List[torch.Tensor] = []
        for layer_idx in range(len(sizes)):
            if layer_idx < len(sizes) - 1:
                next_weight = model.layers[layer_idx + 1].weight
            else:
                next_weight = model.out.weight
            outgoing_norms.append(_linear_outgoing_norm(next_weight))
        importances = [mean_abs[layer_idx] * outgoing_norms[layer_idx] for layer_idx in range(len(sizes))]
        return mean_abs, outgoing_norms, importances

    if isinstance(model, PrunableSmallConvNet):
        outgoing_norms = []
        for layer_idx in range(len(sizes)):
            if layer_idx < len(sizes) - 1:
                outgoing_norms.append(_conv_input_channel_norm(model.conv_layers[layer_idx + 1].weight))
            else:
                channels, height, width = model.feature_shape
                spatial_dim = int(height * width)
                fc1_weight = model.fc1.weight.view(model.fc1.out_features, channels, spatial_dim)
                outgoing_norms.append(
                    fc1_weight.permute(1, 0, 2).reshape(channels, -1).norm(dim=1)
                )
        importances = [mean_abs[layer_idx] * outgoing_norms[layer_idx] for layer_idx in range(len(sizes))]
        return mean_abs, outgoing_norms, importances

    raise TypeError(f"Unsupported prunable model type: {type(model)!r}")


def count_active_units(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    active_threshold: float,
) -> List[int]:
    mean_abs, _, _ = compute_importance_components(model=model, data_loader=data_loader, device=device)
    return [int((value > active_threshold).sum().item()) for value in mean_abs]


def ablation_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    layer_idx: int,
    neuron_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x, ablate=(layer_idx, neuron_idx))
            loss = criterion(logits, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
    return total_loss / max(total_examples, 1)


def _prune_linear_layer(
    model: PrunableLeNet300100,
    layer_idx: int,
    keep_idx: torch.Tensor,
) -> None:
    layer = model.layers[layer_idx]
    device = layer.weight.device
    new_out_features = keep_idx.numel()
    new_layer = nn.Linear(layer.in_features, new_out_features, bias=True).to(device)
    new_layer.weight.data.copy_(layer.weight.data[keep_idx, :])
    new_layer.bias.data.copy_(layer.bias.data[keep_idx])
    model.layers[layer_idx] = new_layer

    if layer_idx < len(model.layers) - 1:
        next_layer = model.layers[layer_idx + 1]
        new_next = nn.Linear(new_out_features, next_layer.out_features, bias=True).to(device)
        new_next.weight.data.copy_(next_layer.weight.data[:, keep_idx])
        new_next.bias.data.copy_(next_layer.bias.data)
        model.layers[layer_idx + 1] = new_next
    else:
        out_layer = model.out
        new_out = nn.Linear(new_out_features, out_layer.out_features, bias=True).to(device)
        new_out.weight.data.copy_(out_layer.weight.data[:, keep_idx])
        new_out.bias.data.copy_(out_layer.bias.data)
        model.out = new_out


def _build_conv_like(old_layer: nn.Conv2d, in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=old_layer.kernel_size,
        stride=old_layer.stride,
        padding=old_layer.padding,
        dilation=old_layer.dilation,
        groups=old_layer.groups,
        bias=old_layer.bias is not None,
        padding_mode=old_layer.padding_mode,
    ).to(old_layer.weight.device)


def _prune_smallconv_layer(
    model: PrunableSmallConvNet,
    layer_idx: int,
    keep_idx: torch.Tensor,
) -> None:
    layer = model.conv_layers[layer_idx]
    device = layer.weight.device
    new_out_channels = keep_idx.numel()
    new_layer = _build_conv_like(layer, in_channels=layer.in_channels, out_channels=new_out_channels)
    new_layer.weight.data.copy_(layer.weight.data[keep_idx, :, :, :])
    if layer.bias is not None:
        new_layer.bias.data.copy_(layer.bias.data[keep_idx])
    model.conv_layers[layer_idx] = new_layer

    if layer_idx < len(model.conv_layers) - 1:
        next_layer = model.conv_layers[layer_idx + 1]
        new_next = _build_conv_like(next_layer, in_channels=new_out_channels, out_channels=next_layer.out_channels)
        new_next.weight.data.copy_(next_layer.weight.data[:, keep_idx, :, :])
        if next_layer.bias is not None:
            new_next.bias.data.copy_(next_layer.bias.data)
        model.conv_layers[layer_idx + 1] = new_next
        return

    _, feature_h, feature_w = model.feature_shape
    spatial_dim = int(feature_h * feature_w)
    fc1 = model.fc1
    keep_idx_cpu = keep_idx.detach().cpu()
    selected_columns = []
    for channel_idx in keep_idx_cpu.tolist():
        start = channel_idx * spatial_dim
        selected_columns.extend(range(start, start + spatial_dim))
    selected_column_idx = torch.tensor(selected_columns, dtype=torch.long, device=fc1.weight.device)
    new_fc1 = nn.Linear(new_out_channels * spatial_dim, fc1.out_features, bias=True).to(device)
    new_fc1.weight.data.copy_(fc1.weight.data[:, selected_column_idx])
    new_fc1.bias.data.copy_(fc1.bias.data)
    model.fc1 = new_fc1
    model.feature_shape = (new_out_channels, feature_h, feature_w)


def prune_prunable_layer(
    model: nn.Module,
    layer_idx: int,
    keep_idx: torch.Tensor,
) -> None:
    if isinstance(model, PrunableLeNet300100):
        _prune_linear_layer(model, layer_idx, keep_idx)
        return
    if isinstance(model, PrunableSmallConvNet):
        _prune_smallconv_layer(model, layer_idx, keep_idx)
        return
    raise TypeError(f"Unsupported prunable model type: {type(model)!r}")


def evaluate_prune_screening(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    baseline_val_loss: float,
    mean_abs: List[torch.Tensor],
    outgoing_norms: List[torch.Tensor],
    importances: List[torch.Tensor],
    ema_state: Optional[List[torch.Tensor]],
    ema_beta: float,
    ema_z_threshold: float,
    max_candidates_per_layer: int,
    min_neurons: int,
    ablation_epsilon_ratio: float,
    allowed_layer_indices: Optional[List[int]] = None,
) -> Dict[str, object]:
    layer_count = len(model.hidden_sizes())
    mean_abs_cpu = [value.detach().cpu() for value in mean_abs]
    outgoing_norms_cpu = [value.detach().cpu() for value in outgoing_norms]
    importances_cpu = [value.detach().cpu() for value in importances]

    if ema_state is None:
        ema_state = [value.clone() for value in importances_cpu]
    else:
        for layer_idx, values in enumerate(importances_cpu):
            ema_state[layer_idx] = ema_beta * ema_state[layer_idx] + (1.0 - ema_beta) * values

    prune_indices: List[List[int]] = [[] for _ in range(layer_count)]
    candidate_counts = [0 for _ in range(layer_count)]
    would_prune_counts = [0 for _ in range(layer_count)]
    ema_means = [0.0 for _ in range(layer_count)]
    ema_stds = [0.0 for _ in range(layer_count)]
    thresholds = [0.0 for _ in range(layer_count)]
    snapshot_rows: List[Dict[str, object]] = []

    epsilon = ablation_epsilon_ratio * max(baseline_val_loss, 1e-12)
    allowed_layer_set = None if allowed_layer_indices is None else set(allowed_layer_indices)
    for layer_idx, ema in enumerate(ema_state):
        mean = float(ema.mean())
        std = float(ema.std(unbiased=False)) if ema.numel() > 1 else 0.0
        threshold = mean - ema_z_threshold * std
        ema_means[layer_idx] = mean
        ema_stds[layer_idx] = std
        thresholds[layer_idx] = threshold

        candidates = (ema < threshold).nonzero(as_tuple=False).flatten().tolist()
        if candidates:
            candidates = sorted(candidates, key=lambda neuron_idx: ema[neuron_idx].item())
            candidates = candidates[:max_candidates_per_layer]
        candidate_counts[layer_idx] = len(candidates)
        candidate_rank = {neuron_idx: rank for rank, neuron_idx in enumerate(candidates)}

        max_prune = max(int(ema.numel()) - min_neurons, 0)
        ablation_by_neuron: Dict[int, float] = {}
        delta_by_neuron: Dict[int, float] = {}
        would_prune_set = set()
        is_allowed_layer = allowed_layer_set is None or layer_idx in allowed_layer_set
        for neuron_idx in candidates:
            if not is_allowed_layer:
                break
            if len(prune_indices[layer_idx]) >= max_prune:
                break
            ablated_loss = ablation_loss(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                layer_idx=layer_idx,
                neuron_idx=neuron_idx,
            )
            delta = ablated_loss - baseline_val_loss
            ablation_by_neuron[neuron_idx] = ablated_loss
            delta_by_neuron[neuron_idx] = delta
            if delta < epsilon:
                prune_indices[layer_idx].append(neuron_idx)
                would_prune_set.add(neuron_idx)
        would_prune_counts[layer_idx] = len(would_prune_set)

        for neuron_idx in range(int(ema.numel())):
            snapshot_rows.append(
                {
                    "epoch": epoch,
                    "layer_idx": layer_idx,
                    "neuron_idx": neuron_idx,
                    "mean_abs_activation": float(mean_abs_cpu[layer_idx][neuron_idx]),
                    "outgoing_norm": float(outgoing_norms_cpu[layer_idx][neuron_idx]),
                    "importance": float(importances_cpu[layer_idx][neuron_idx]),
                    "ema_importance": float(ema[neuron_idx]),
                    "ema_mean_layer": mean,
                    "ema_std_layer": std,
                    "threshold": threshold,
                    "baseline_val_loss": baseline_val_loss,
                    "ablation_epsilon": epsilon,
                    "is_allowed_layer": int(is_allowed_layer),
                    "is_candidate": int(neuron_idx in candidate_rank),
                    "candidate_rank": candidate_rank.get(neuron_idx, -1),
                    "would_prune": int(neuron_idx in would_prune_set),
                    "ablated_val_loss": ablation_by_neuron.get(neuron_idx),
                    "delta_val_loss": delta_by_neuron.get(neuron_idx),
                }
            )

    return {
        "ema_state": ema_state,
        "prune_indices": prune_indices,
        "candidate_counts": candidate_counts,
        "would_prune_counts": would_prune_counts,
        "ema_means": ema_means,
        "ema_stds": ema_stds,
        "thresholds": thresholds,
        "snapshot_rows": snapshot_rows,
    }
