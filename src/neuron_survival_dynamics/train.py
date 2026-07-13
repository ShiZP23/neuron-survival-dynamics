import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neuron_survival_dynamics.data import DatasetBundle, make_dataset
from neuron_survival_dynamics.model import MLP
from neuron_survival_dynamics.structural import (
    compute_importance,
    compute_importance_components,
    prune_and_grow,
)
from neuron_survival_dynamics.utils import count_trainable_params


def _clone_state_dict(model: MLP) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _run_epoch(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_count = 0

    if optimizer is None:
        model.eval()
    else:
        model.train()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        preds = model(x)
        loss = criterion(preds, y)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch = x.size(0)
        total_loss += loss.item() * batch
        total_count += batch

    return total_loss / max(total_count, 1)


def _ablation_loss(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    layer_idx: int,
    neuron_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x, ablate=(layer_idx, neuron_idx))
            loss = criterion(preds, y)
            batch = x.size(0)
            total_loss += loss.item() * batch
            total_count += batch
    return total_loss / max(total_count, 1)


def _count_active_neurons(
    model: MLP,
    loader: DataLoader,
    device: torch.device,
    active_threshold: float,
) -> List[int]:
    model.eval()
    sizes = model.hidden_sizes()
    sums = [torch.zeros(size, device=device) for size in sizes]
    total = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, activations = model(x, return_activations=True)
            batch = x.size(0)
            total += batch
            for idx, act in enumerate(activations):
                sums[idx] += act.abs().sum(dim=0)

    mean_abs = [s / max(total, 1) for s in sums]
    return [int((m > active_threshold).sum().item()) for m in mean_abs]


def _evaluate_prune_screening(
    model: MLP,
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
        for neuron_idx in candidates:
            if len(prune_indices[layer_idx]) >= max_prune:
                break
            ablated_loss = _ablation_loss(
                model,
                val_loader,
                criterion,
                device,
                layer_idx,
                neuron_idx,
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


def train_one_run(
    task: str,
    mode: str,
    run_dir: str,
    seed: int,
    device: torch.device,
    epochs: int = 300,
    update_interval: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    min_neurons: int = 16,
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
    noise: float = 0.0,
    data_seed: int = 0,
    model_seed: int = 0,
    shuffle_seed: int = 0,
    structure_seed: int = 0,
    ema_beta: float = 0.9,
    ema_z_threshold: float = 1.0,
    max_candidates_per_layer: int = 8,
    ablation_epsilon_ratio: float = 0.01,
    active_threshold: float = 1e-3,
    freeze_structure_after_epoch: Optional[int] = None,
    freeze_structure_reference: Optional[str] = None,
    shadow_prune: bool = False,
) -> Tuple[MLP, List[Dict]]:
    data: DatasetBundle = make_dataset(
        task=task,
        seed=data_seed,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        noise=noise,
    )
    train_gen = torch.Generator()
    train_gen.manual_seed(shuffle_seed)
    train_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
    )
    val_loader = DataLoader(
        TensorDataset(data.x_val, data.y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(data.x_test, data.y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    active_loader = DataLoader(
        TensorDataset(data.x_val, data.y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    importance_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=False,
    )

    # Isolate model initialization RNG from the global training RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(structure_seed)
    ema_state: Optional[List[torch.Tensor]] = None
    shadow_ema_state: Optional[List[torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch = 0
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_param_count = 0
    best_hidden_sizes: List[int] = []
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    history: List[Dict] = []
    shadow_snapshot_rows: List[Dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
        test_loss = _run_epoch(model, test_loader, criterion, None, device)

        layer_count = len(model.hidden_sizes())
        pruned = [0 for _ in range(layer_count)]
        grown = [0 for _ in range(layer_count)]
        candidate_counts = [0 for _ in range(layer_count)]
        ema_means = [0.0 for _ in range(layer_count)]
        ema_stds = [0.0 for _ in range(layer_count)]
        is_shadow_update_epoch = 0
        shadow_candidate_counts = [0 for _ in range(layer_count)]
        shadow_would_prune_counts = [0 for _ in range(layer_count)]
        shadow_ema_means = [0.0 for _ in range(layer_count)]
        shadow_ema_stds = [0.0 for _ in range(layer_count)]
        shadow_thresholds = [0.0 for _ in range(layer_count)]
        is_update_epoch = 0
        updated = False
        structure_updates_allowed = (
            freeze_structure_after_epoch is None or epoch <= freeze_structure_after_epoch
        )
        screening_enabled = epoch % update_interval == 0 and structure_updates_allowed
        actual_structure_update = mode != "fixed" and screening_enabled
        shadow_structure_update = mode == "fixed" and shadow_prune and screening_enabled
        if actual_structure_update or shadow_structure_update:
            mean_abs, outgoing_norms, importances = compute_importance_components(
                model, importance_loader, device
            )
            baseline_val_loss = val_loss

        if actual_structure_update:
            is_update_epoch = 1
            screening = _evaluate_prune_screening(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                baseline_val_loss=baseline_val_loss,
                mean_abs=mean_abs,
                outgoing_norms=outgoing_norms,
                importances=importances,
                ema_state=ema_state,
                ema_beta=ema_beta,
                ema_z_threshold=ema_z_threshold,
                max_candidates_per_layer=max_candidates_per_layer,
                min_neurons=min_neurons,
                ablation_epsilon_ratio=ablation_epsilon_ratio,
            )
            ema_state = screening["ema_state"]
            candidate_counts = screening["candidate_counts"]
            ema_means = screening["ema_means"]
            ema_stds = screening["ema_stds"]

            pruned, grown, ema_state = prune_and_grow(
                model,
                screening["prune_indices"],
                mode=mode,
                importances=importances,
                rng=rng,
                ema_state=ema_state,
            )
            changed = any(p > 0 for p in pruned) or any(g > 0 for g in grown)
            if changed:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                updated = True

        if shadow_structure_update:
            is_shadow_update_epoch = 1
            shadow_screening = _evaluate_prune_screening(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                baseline_val_loss=baseline_val_loss,
                mean_abs=mean_abs,
                outgoing_norms=outgoing_norms,
                importances=importances,
                ema_state=shadow_ema_state,
                ema_beta=ema_beta,
                ema_z_threshold=ema_z_threshold,
                max_candidates_per_layer=max_candidates_per_layer,
                min_neurons=min_neurons,
                ablation_epsilon_ratio=ablation_epsilon_ratio,
            )
            shadow_ema_state = shadow_screening["ema_state"]
            shadow_candidate_counts = shadow_screening["candidate_counts"]
            shadow_would_prune_counts = shadow_screening["would_prune_counts"]
            shadow_ema_means = shadow_screening["ema_means"]
            shadow_ema_stds = shadow_screening["ema_stds"]
            shadow_thresholds = shadow_screening["thresholds"]
            shadow_snapshot_rows.extend(shadow_screening["snapshot_rows"])

        if updated:
            val_loss = _run_epoch(model, val_loader, criterion, None, device)
            test_loss = _run_epoch(model, test_loader, criterion, None, device)

        sizes = model.hidden_sizes()
        param_count = count_trainable_params(model)
        active_counts = _count_active_neurons(
            model=model,
            loader=active_loader,
            device=device,
            active_threshold=active_threshold,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_test_loss = test_loss
            best_train_loss = train_loss
            best_param_count = param_count
            best_hidden_sizes = list(sizes)
            best_state_dict = _clone_state_dict(model)
        total_pruned = sum(pruned)
        total_grown = sum(grown)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "param_count": param_count,
                "sizes": sizes,
                "pruned": pruned,
                "grown": grown,
                "active_counts": active_counts,
                "total_pruned": total_pruned,
                "total_grown": total_grown,
                "candidate_counts": candidate_counts,
                "ema_means": ema_means,
                "ema_stds": ema_stds,
                "is_update_epoch": is_update_epoch,
                "is_shadow_update_epoch": is_shadow_update_epoch,
                "shadow_candidate_counts": shadow_candidate_counts,
                "shadow_would_prune_counts": shadow_would_prune_counts,
                "shadow_ema_means": shadow_ema_means,
                "shadow_ema_stds": shadow_ema_stds,
                "shadow_thresholds": shadow_thresholds,
            }
        )

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        layer_count = len(history[0]["sizes"])
        fieldnames = [
            "epoch",
            "is_update_epoch",
            "is_shadow_update_epoch",
            "train_loss",
            "val_loss",
            "test_loss",
            "param_count",
            "sizes",
            "pruned",
            "grown",
            "total_pruned",
            "total_grown",
        ]
        for idx in range(layer_count):
            fieldnames.append(f"candidate_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"ema_mean_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"ema_std_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"size_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"pruned_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"grown_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"active_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"shadow_candidate_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"shadow_would_prune_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"shadow_ema_mean_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"shadow_ema_std_{idx}")
        for idx in range(layer_count):
            fieldnames.append(f"shadow_threshold_{idx}")

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in history:
            sizes = row["sizes"]
            pruned = row["pruned"]
            grown = row["grown"]
            active_counts = row["active_counts"]
            candidates = row["candidate_counts"]
            ema_means = row["ema_means"]
            ema_stds = row["ema_stds"]
            shadow_candidates = row["shadow_candidate_counts"]
            shadow_would_prune = row["shadow_would_prune_counts"]
            shadow_ema_means = row["shadow_ema_means"]
            shadow_ema_stds = row["shadow_ema_stds"]
            shadow_thresholds = row["shadow_thresholds"]
            row_dict = {
                "epoch": row["epoch"],
                "is_update_epoch": row["is_update_epoch"],
                "is_shadow_update_epoch": row["is_shadow_update_epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
                "test_loss": row["test_loss"],
                "param_count": row["param_count"],
                "sizes": ",".join(str(x) for x in sizes),
                "pruned": ",".join(str(x) for x in pruned),
                "grown": ",".join(str(x) for x in grown),
                "total_pruned": row["total_pruned"],
                "total_grown": row["total_grown"],
            }
            for idx in range(layer_count):
                row_dict[f"candidate_{idx}"] = candidates[idx]
                row_dict[f"ema_mean_{idx}"] = ema_means[idx]
                row_dict[f"ema_std_{idx}"] = ema_stds[idx]
                row_dict[f"size_{idx}"] = sizes[idx]
                row_dict[f"pruned_{idx}"] = pruned[idx]
                row_dict[f"grown_{idx}"] = grown[idx]
                row_dict[f"active_{idx}"] = active_counts[idx]
                row_dict[f"shadow_candidate_{idx}"] = shadow_candidates[idx]
                row_dict[f"shadow_would_prune_{idx}"] = shadow_would_prune[idx]
                row_dict[f"shadow_ema_mean_{idx}"] = shadow_ema_means[idx]
                row_dict[f"shadow_ema_std_{idx}"] = shadow_ema_stds[idx]
                row_dict[f"shadow_threshold_{idx}"] = shadow_thresholds[idx]
            writer.writerow(row_dict)

    if shadow_snapshot_rows:
        shadow_path = os.path.join(run_dir, "shadow_prune_snapshots.csv")
        with open(shadow_path, "w", newline="") as f:
            fieldnames = list(shadow_snapshot_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shadow_snapshot_rows)

    final_row = history[-1]
    if ema_state is None:
        ema_state = [torch.zeros(size) for size in model.hidden_sizes()]
    if shadow_ema_state is None:
        shadow_ema_state = [torch.zeros(size) for size in model.hidden_sizes()]
    final_checkpoint = {
        "state_dict": model.state_dict(),
        "hidden_sizes": model.hidden_sizes(),
        "task": task,
        "mode": mode,
        "seed": seed,
        "epochs": epochs,
        "update_interval": update_interval,
        "batch_size": batch_size,
        "lr": lr,
        "min_neurons": min_neurons,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "noise": noise,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "shuffle_seed": shuffle_seed,
        "structure_seed": structure_seed,
        "ema_state": ema_state,
        "ema_beta": ema_beta,
        "ema_z_threshold": ema_z_threshold,
        "max_candidates_per_layer": max_candidates_per_layer,
        "ablation_epsilon_ratio": ablation_epsilon_ratio,
        "active_threshold": active_threshold,
        "freeze_structure_after_epoch": freeze_structure_after_epoch,
        "freeze_structure_reference": freeze_structure_reference,
        "shadow_prune": shadow_prune,
        "shadow_ema_state": shadow_ema_state,
        "final_epoch": final_row["epoch"],
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
        "final_test_loss": final_row["test_loss"],
        "final_param_count": final_row["param_count"],
        "selection_protocol": "fixed_budget_then_select_best_val_checkpoint",
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "best_test_loss_at_best_val": best_test_loss,
        "best_param_count": best_param_count,
        "best_hidden_sizes": best_hidden_sizes,
    }
    torch.save(final_checkpoint, os.path.join(run_dir, "model.pt"))
    if best_state_dict is not None:
        best_checkpoint = {
            "state_dict": best_state_dict,
            "hidden_sizes": best_hidden_sizes,
            "task": task,
            "mode": mode,
            "seed": seed,
            "epochs": epochs,
            "update_interval": update_interval,
            "batch_size": batch_size,
            "lr": lr,
            "min_neurons": min_neurons,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "noise": noise,
            "data_seed": data_seed,
            "model_seed": model_seed,
            "shuffle_seed": shuffle_seed,
            "structure_seed": structure_seed,
            "ema_beta": ema_beta,
            "ema_z_threshold": ema_z_threshold,
            "max_candidates_per_layer": max_candidates_per_layer,
            "ablation_epsilon_ratio": ablation_epsilon_ratio,
            "active_threshold": active_threshold,
            "freeze_structure_after_epoch": freeze_structure_after_epoch,
            "freeze_structure_reference": freeze_structure_reference,
            "shadow_prune": shadow_prune,
            "selection_protocol": "validation_loss_minimization",
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_test_loss_at_best_val": best_test_loss,
            "best_param_count": best_param_count,
        }
        torch.save(best_checkpoint, os.path.join(run_dir, "best_model.pt"))
    return model, history
