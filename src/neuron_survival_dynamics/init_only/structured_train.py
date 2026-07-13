import csv
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuron_survival_dynamics.init_only.prunable_models import build_prunable_model
from neuron_survival_dynamics.init_only.structural import (
    compute_importance_components,
    count_active_units,
    evaluate_prune_screening,
    prune_prunable_layer,
)
from neuron_survival_dynamics.utils import count_trainable_params, save_json, set_seed


STRUCTURED_MODES = ["fixed", "prune_only"]


@dataclass
class StructuredRunSummary:
    dataset: str
    model: str
    mode: str
    init_seed: int
    runtime_seed: int
    split_seed: int
    train_order_seed: int
    epochs: int
    update_interval: int
    batch_size: int
    lr: float
    weight_decay: float
    selected_epoch: int
    best_val_loss: float
    selected_val_acc: float
    selected_test_acc: float
    selected_test_loss: float
    final_val_acc: float
    final_test_acc: float
    final_test_loss: float
    final_minus_selected_loss: float
    selected_minus_final_acc: float
    total_pruned: int
    final_param_count: int
    final_hidden_sizes: str
    run_dir: str


def _classification_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    if optimizer is None:
        model.eval()
    else:
        model.train()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_examples += batch_size

    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _apply_pruning(
    model: nn.Module,
    prune_indices: List[List[int]],
    ema_state: List[torch.Tensor],
) -> Tuple[List[int], List[torch.Tensor]]:
    pruned_counts: List[int] = []
    for layer_idx, indices in enumerate(prune_indices):
        if not indices:
            pruned_counts.append(0)
            continue
        size = model.hidden_sizes()[layer_idx]
        keep_mask = torch.ones(size, dtype=torch.bool)
        keep_mask[indices] = False
        keep_idx_cpu = torch.arange(size)[keep_mask]
        keep_idx = keep_idx_cpu.to(next(model.parameters()).device)
        prune_prunable_layer(model, layer_idx, keep_idx)
        ema_state[layer_idx] = ema_state[layer_idx][keep_idx_cpu]
        pruned_counts.append(len(indices))
    return pruned_counts, ema_state


def _save_metrics_csv(path: str, history: List[Dict[str, object]]) -> None:
    if not history:
        return
    layer_count = len(history[0]["sizes"])
    fieldnames = [
        "epoch",
        "is_update_epoch",
        "is_shadow_update_epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "param_count",
        "sizes",
        "pruned",
        "total_pruned_epoch",
        "shadow_would_prune",
        "total_shadow_would_prune_epoch",
        "is_best_epoch",
    ]
    for layer_idx in range(layer_count):
        fieldnames.append(f"size_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"pruned_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"active_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"candidate_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"would_prune_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"ema_mean_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"ema_std_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"threshold_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"shadow_candidate_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"shadow_would_prune_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"shadow_ema_mean_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"shadow_ema_std_{layer_idx}")
    for layer_idx in range(layer_count):
        fieldnames.append(f"shadow_threshold_{layer_idx}")

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            row_dict = {
                "epoch": row["epoch"],
                "is_update_epoch": row["is_update_epoch"],
                "is_shadow_update_epoch": row["is_shadow_update_epoch"],
                "train_loss": row["train_loss"],
                "train_acc": row["train_acc"],
                "val_loss": row["val_loss"],
                "val_acc": row["val_acc"],
                "test_loss": row["test_loss"],
                "test_acc": row["test_acc"],
                "param_count": row["param_count"],
                "sizes": ",".join(str(value) for value in row["sizes"]),
                "pruned": ",".join(str(value) for value in row["pruned"]),
                "total_pruned_epoch": row["total_pruned_epoch"],
                "shadow_would_prune": ",".join(str(value) for value in row["shadow_would_prune"]),
                "total_shadow_would_prune_epoch": row["total_shadow_would_prune_epoch"],
                "is_best_epoch": row["is_best_epoch"],
            }
            for layer_idx in range(layer_count):
                row_dict[f"size_{layer_idx}"] = row["sizes"][layer_idx]
                row_dict[f"pruned_{layer_idx}"] = row["pruned"][layer_idx]
                row_dict[f"active_{layer_idx}"] = row["active_counts"][layer_idx]
                row_dict[f"candidate_{layer_idx}"] = row["candidate_counts"][layer_idx]
                row_dict[f"would_prune_{layer_idx}"] = row["would_prune_counts"][layer_idx]
                row_dict[f"ema_mean_{layer_idx}"] = row["ema_means"][layer_idx]
                row_dict[f"ema_std_{layer_idx}"] = row["ema_stds"][layer_idx]
                row_dict[f"threshold_{layer_idx}"] = row["thresholds"][layer_idx]
                row_dict[f"shadow_candidate_{layer_idx}"] = row["shadow_candidate_counts"][layer_idx]
                row_dict[f"shadow_would_prune_{layer_idx}"] = row["shadow_would_prune"][layer_idx]
                row_dict[f"shadow_ema_mean_{layer_idx}"] = row["shadow_ema_means"][layer_idx]
                row_dict[f"shadow_ema_std_{layer_idx}"] = row["shadow_ema_stds"][layer_idx]
                row_dict[f"shadow_threshold_{layer_idx}"] = row["shadow_thresholds"][layer_idx]
            writer.writerow(row_dict)


def _save_snapshot_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_structured_classifier_run(
    dataset_name: str,
    model_name: str,
    mode: str,
    init_seed: int,
    runtime_seed: int,
    split_seed: int,
    train_order_seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    run_dir: str,
    device: torch.device,
    epochs: int,
    update_interval: int,
    lr: float,
    weight_decay: float = 0.0,
    min_neurons: int = 16,
    ema_beta: float = 0.9,
    ema_z_threshold: float = 1.0,
    max_candidates_per_layer: int = 8,
    ablation_epsilon_ratio: float = 0.01,
    active_threshold: float = 1e-3,
    shadow_prune: bool = False,
    freeze_actual_updates_after_epoch: Optional[int] = None,
    shadow_after_freeze: bool = False,
    actual_prune_layer_whitelist: Optional[List[int]] = None,
    actual_prune_layer_whitelist_by_epoch: Optional[Dict[int, Optional[List[int]]]] = None,
    deterministic_algorithms: bool = False,
    save_checkpoints: bool = True,
) -> StructuredRunSummary:
    if mode not in STRUCTURED_MODES:
        raise ValueError(f"mode must be one of {STRUCTURED_MODES}, got {mode!r}")

    os.makedirs(run_dir, exist_ok=True)
    set_seed(runtime_seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = build_prunable_model(
            model_name=model_name,
            input_shape=input_shape,
            num_classes=num_classes,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    init_state = _clone_state_dict(model)
    param_count = count_trainable_params(model)

    best_val_loss = float("inf")
    best_epoch = 0
    best_snapshot: Optional[Dict[str, float]] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, object]] = []
    screening_snapshot_rows: List[Dict[str, object]] = []
    shadow_snapshot_rows: List[Dict[str, object]] = []
    ema_state: Optional[List[torch.Tensor]] = None
    shadow_ema_state: Optional[List[torch.Tensor]] = None
    total_pruned_so_far = 0
    shadow_state_initialized_from_prune = False
    normalized_epoch_whitelist = None
    if actual_prune_layer_whitelist_by_epoch is not None:
        normalized_epoch_whitelist = {
            int(epoch_key): value
            for epoch_key, value in actual_prune_layer_whitelist_by_epoch.items()
        }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _classification_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _classification_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = _classification_epoch(model, test_loader, criterion, device)

        sizes = model.hidden_sizes()
        layer_count = len(sizes)
        pruned = [0 for _ in range(layer_count)]
        candidate_counts = [0 for _ in range(layer_count)]
        would_prune_counts = [0 for _ in range(layer_count)]
        ema_means = [0.0 for _ in range(layer_count)]
        ema_stds = [0.0 for _ in range(layer_count)]
        thresholds = [0.0 for _ in range(layer_count)]
        shadow_candidate_counts = [0 for _ in range(layer_count)]
        shadow_would_prune_counts = [0 for _ in range(layer_count)]
        shadow_ema_means = [0.0 for _ in range(layer_count)]
        shadow_ema_stds = [0.0 for _ in range(layer_count)]
        shadow_thresholds = [0.0 for _ in range(layer_count)]
        is_update_epoch = 0
        is_shadow_update_epoch = 0

        screening_enabled = epoch % update_interval == 0
        actual_updates_allowed = (
            mode == "prune_only"
            and (
                freeze_actual_updates_after_epoch is None
                or epoch <= freeze_actual_updates_after_epoch
            )
        )
        actual_structure_update = screening_enabled and actual_updates_allowed
        shadow_structure_update = screening_enabled and (
            (mode == "fixed" and shadow_prune)
            or (
                mode == "prune_only"
                and shadow_after_freeze
                and not actual_updates_allowed
            )
        )

        if actual_structure_update or shadow_structure_update:
            mean_abs, outgoing_norms, importances = compute_importance_components(
                model=model,
                data_loader=train_loader,
                device=device,
            )
            baseline_val_loss = val_loss

        if actual_structure_update:
            is_update_epoch = 1
            allowed_layer_indices = actual_prune_layer_whitelist
            if normalized_epoch_whitelist is not None and epoch in normalized_epoch_whitelist:
                allowed_layer_indices = normalized_epoch_whitelist[epoch]
            screening = evaluate_prune_screening(
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
                allowed_layer_indices=allowed_layer_indices,
            )
            ema_state = screening["ema_state"]
            candidate_counts = screening["candidate_counts"]
            would_prune_counts = screening["would_prune_counts"]
            ema_means = screening["ema_means"]
            ema_stds = screening["ema_stds"]
            thresholds = screening["thresholds"]
            screening_snapshot_rows.extend(screening["snapshot_rows"])

            pruned, ema_state = _apply_pruning(
                model=model,
                prune_indices=screening["prune_indices"],
                ema_state=ema_state,
            )
            total_pruned_so_far += sum(pruned)
            if any(count > 0 for count in pruned):
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                val_loss, val_acc = _classification_epoch(model, val_loader, criterion, device)
                test_loss, test_acc = _classification_epoch(model, test_loader, criterion, device)
                sizes = model.hidden_sizes()

        if shadow_structure_update:
            is_shadow_update_epoch = 1
            if (
                mode == "prune_only"
                and shadow_after_freeze
                and not shadow_state_initialized_from_prune
                and shadow_ema_state is None
                and ema_state is not None
            ):
                shadow_ema_state = [value.clone() for value in ema_state]
                shadow_state_initialized_from_prune = True
            shadow_screening = evaluate_prune_screening(
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

        param_count = count_trainable_params(model)
        active_counts = count_active_units(
            model=model,
            data_loader=val_loader,
            device=device,
            active_threshold=active_threshold,
        )
        is_best_epoch = int(val_loss < best_val_loss)
        if is_best_epoch:
            best_val_loss = val_loss
            best_epoch = epoch
            best_snapshot = {
                "selected_val_acc": val_acc,
                "selected_test_acc": test_acc,
                "selected_test_loss": test_loss,
            }
            best_state_dict = _clone_state_dict(model)

        history.append(
            {
                "epoch": epoch,
                "is_update_epoch": is_update_epoch,
                "is_shadow_update_epoch": is_shadow_update_epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "param_count": param_count,
                "sizes": list(sizes),
                "pruned": pruned,
                "total_pruned_epoch": sum(pruned),
                "candidate_counts": candidate_counts,
                "would_prune_counts": would_prune_counts,
                "ema_means": ema_means,
                "ema_stds": ema_stds,
                "thresholds": thresholds,
                "active_counts": active_counts,
                "shadow_candidate_counts": shadow_candidate_counts,
                "shadow_would_prune": shadow_would_prune_counts,
                "total_shadow_would_prune_epoch": sum(shadow_would_prune_counts),
                "shadow_ema_means": shadow_ema_means,
                "shadow_ema_stds": shadow_ema_stds,
                "shadow_thresholds": shadow_thresholds,
                "is_best_epoch": is_best_epoch,
            }
        )

    if best_snapshot is None or best_state_dict is None:
        raise RuntimeError("Best snapshot was not recorded during structured training.")

    final_row = history[-1]
    summary = StructuredRunSummary(
        dataset=dataset_name,
        model=model_name,
        mode=mode,
        init_seed=init_seed,
        runtime_seed=runtime_seed,
        split_seed=split_seed,
        train_order_seed=train_order_seed,
        epochs=epochs,
        update_interval=update_interval,
        batch_size=train_loader.batch_size or 0,
        lr=lr,
        weight_decay=weight_decay,
        selected_epoch=best_epoch,
        best_val_loss=best_val_loss,
        selected_val_acc=best_snapshot["selected_val_acc"],
        selected_test_acc=best_snapshot["selected_test_acc"],
        selected_test_loss=best_snapshot["selected_test_loss"],
        final_val_acc=final_row["val_acc"],
        final_test_acc=final_row["test_acc"],
        final_test_loss=final_row["test_loss"],
        final_minus_selected_loss=final_row["test_loss"] - best_snapshot["selected_test_loss"],
        selected_minus_final_acc=best_snapshot["selected_test_acc"] - final_row["test_acc"],
        total_pruned=total_pruned_so_far,
        final_param_count=final_row["param_count"],
        final_hidden_sizes=",".join(str(value) for value in final_row["sizes"]),
        run_dir=run_dir,
    )

    _save_metrics_csv(os.path.join(run_dir, "metrics.csv"), history)
    save_json(os.path.join(run_dir, "summary.json"), asdict(summary))
    _save_snapshot_csv(os.path.join(run_dir, "screening_snapshots.csv"), screening_snapshot_rows)
    _save_snapshot_csv(os.path.join(run_dir, "shadow_prune_snapshots.csv"), shadow_snapshot_rows)

    if save_checkpoints:
        torch.save(
            {
                "dataset": dataset_name,
                "model": model_name,
                "mode": mode,
                "init_seed": init_seed,
                "runtime_seed": runtime_seed,
                "split_seed": split_seed,
                "train_order_seed": train_order_seed,
                "input_shape": input_shape,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "best_state_dict": best_state_dict,
                "init_state_dict": init_state,
                "final_hidden_sizes": model.hidden_sizes(),
                "ema_state": ema_state,
                "shadow_ema_state": shadow_ema_state,
                "freeze_actual_updates_after_epoch": freeze_actual_updates_after_epoch,
                "shadow_after_freeze": shadow_after_freeze,
                "actual_prune_layer_whitelist": actual_prune_layer_whitelist,
                "actual_prune_layer_whitelist_by_epoch": actual_prune_layer_whitelist_by_epoch,
            },
            os.path.join(run_dir, "model_bundle.pt"),
        )

    return summary
