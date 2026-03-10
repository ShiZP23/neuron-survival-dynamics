import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neuron_survival_dynamics.data import DatasetBundle, make_dataset
from neuron_survival_dynamics.model import MLP
from neuron_survival_dynamics.structural import compute_importance, prune_and_grow
from neuron_survival_dynamics.utils import count_trainable_params


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
    ema_beta: float = 0.9,
    ema_z_threshold: float = 1.0,
    max_candidates_per_layer: int = 8,
    ablation_epsilon_ratio: float = 0.01,
) -> Tuple[MLP, List[Dict]]:
    data: DatasetBundle = make_dataset(
        task=task,
        seed=seed,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        noise=noise,
    )
    train_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
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
    importance_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=False,
    )

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(seed)
    ema_state: Optional[List[torch.Tensor]] = None

    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = _run_epoch(model, test_loader, criterion, None, device)

        layer_count = len(model.hidden_sizes())
        pruned = [0 for _ in range(layer_count)]
        grown = [0 for _ in range(layer_count)]
        candidate_counts = [0 for _ in range(layer_count)]
        ema_means = [0.0 for _ in range(layer_count)]
        ema_stds = [0.0 for _ in range(layer_count)]
        is_update_epoch = 0
        updated = False
        if mode != "fixed" and epoch % update_interval == 0:
            is_update_epoch = 1
            baseline_val_loss = _run_epoch(
                model, val_loader, criterion, None, device
            )
            importances = compute_importance(model, importance_loader, device)
            importances_cpu = [imp.detach().cpu() for imp in importances]

            if ema_state is None:
                # Initialize EMA from the first measured importance.
                ema_state = [imp.clone() for imp in importances_cpu]
            else:
                for idx, imp in enumerate(importances_cpu):
                    ema_state[idx] = ema_beta * ema_state[idx] + (1.0 - ema_beta) * imp

            prune_indices: List[List[int]] = [[] for _ in range(layer_count)]
            for layer_idx, ema in enumerate(ema_state):
                mean = float(ema.mean())
                std = float(ema.std(unbiased=False)) if ema.numel() > 1 else 0.0
                ema_means[layer_idx] = mean
                ema_stds[layer_idx] = std
                threshold = mean - ema_z_threshold * std
                candidates = (ema < threshold).nonzero(as_tuple=False).flatten().tolist()
                if candidates:
                    candidates = sorted(candidates, key=lambda i: ema[i].item())
                    candidates = candidates[:max_candidates_per_layer]
                candidate_counts[layer_idx] = len(candidates)

                max_prune = max(int(ema.numel()) - min_neurons, 0)
                if max_prune == 0:
                    continue

                epsilon = ablation_epsilon_ratio * max(baseline_val_loss, 1e-12)
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
                    if delta < epsilon:
                        prune_indices[layer_idx].append(neuron_idx)

            pruned, grown, ema_state = prune_and_grow(
                model,
                prune_indices,
                mode=mode,
                importances=importances,
                rng=rng,
                ema_state=ema_state,
            )
            changed = any(p > 0 for p in pruned) or any(g > 0 for g in grown)
            if changed:
                # Optimizer state is reset because parameter shapes changed.
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                updated = True

        if updated:
            test_loss = _run_epoch(model, test_loader, criterion, None, device)

        sizes = model.hidden_sizes()
        param_count = count_trainable_params(model)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "param_count": param_count,
                "sizes": sizes,
                "pruned": pruned,
                "grown": grown,
                "candidate_counts": candidate_counts,
                "ema_means": ema_means,
                "ema_stds": ema_stds,
                "is_update_epoch": is_update_epoch,
            }
        )

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "is_update_epoch",
                "train_loss",
                "test_loss",
                "param_count",
                "sizes",
                "pruned",
                "grown",
                "candidate_0",
                "candidate_1",
                "candidate_2",
                "ema_mean_0",
                "ema_mean_1",
                "ema_mean_2",
                "ema_std_0",
                "ema_std_1",
                "ema_std_2",
                "size_0",
                "size_1",
                "size_2",
                "pruned_0",
                "pruned_1",
                "pruned_2",
                "grown_0",
                "grown_1",
                "grown_2",
            ],
        )
        writer.writeheader()
        for row in history:
            sizes = row["sizes"]
            pruned = row["pruned"]
            grown = row["grown"]
            candidates = row["candidate_counts"]
            ema_means = row["ema_means"]
            ema_stds = row["ema_stds"]
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "is_update_epoch": row["is_update_epoch"],
                    "train_loss": row["train_loss"],
                    "test_loss": row["test_loss"],
                    "param_count": row["param_count"],
                    "sizes": ",".join(str(x) for x in sizes),
                    "pruned": ",".join(str(x) for x in pruned),
                    "grown": ",".join(str(x) for x in grown),
                    "candidate_0": candidates[0],
                    "candidate_1": candidates[1],
                    "candidate_2": candidates[2],
                    "ema_mean_0": ema_means[0],
                    "ema_mean_1": ema_means[1],
                    "ema_mean_2": ema_means[2],
                    "ema_std_0": ema_stds[0],
                    "ema_std_1": ema_stds[1],
                    "ema_std_2": ema_stds[2],
                    "size_0": sizes[0],
                    "size_1": sizes[1],
                    "size_2": sizes[2],
                    "pruned_0": pruned[0],
                    "pruned_1": pruned[1],
                    "pruned_2": pruned[2],
                    "grown_0": grown[0],
                    "grown_1": grown[1],
                    "grown_2": grown[2],
                }
            )

    final_row = history[-1]
    if ema_state is None:
        ema_state = [torch.zeros(size) for size in model.hidden_sizes()]
    checkpoint = {
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
        "ema_state": ema_state,
        "ema_beta": ema_beta,
        "ema_z_threshold": ema_z_threshold,
        "max_candidates_per_layer": max_candidates_per_layer,
        "ablation_epsilon_ratio": ablation_epsilon_ratio,
        "final_epoch": final_row["epoch"],
        "final_train_loss": final_row["train_loss"],
        "final_test_loss": final_row["test_loss"],
        "final_param_count": final_row["param_count"],
    }
    torch.save(checkpoint, os.path.join(run_dir, "model.pt"))
    return model, history
