import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neuron_kill.data import DatasetBundle, make_dataset
from neuron_kill.model import MLP
from neuron_kill.structural import compute_importance, prune_and_grow
from neuron_kill.utils import count_trainable_params


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
    prune_fraction: float = 0.1,
    min_neurons: int = 16,
    n_train: int = 5000,
    n_test: int = 1000,
    noise: float = 0.0,
) -> Tuple[MLP, List[Dict]]:
    data: DatasetBundle = make_dataset(
        task=task,
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        noise=noise,
    )
    train_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
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

    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = _run_epoch(model, test_loader, criterion, None, device)

        pruned = [0 for _ in model.hidden_sizes()]
        grown = [0 for _ in model.hidden_sizes()]
        if mode != "fixed" and epoch % update_interval == 0:
            importances = compute_importance(model, importance_loader, device)
            pruned, grown = prune_and_grow(
                model,
                importances,
                mode=mode,
                prune_fraction=prune_fraction,
                min_neurons=min_neurons,
                rng=rng,
            )
            # Optimizer state is reset because parameter shapes changed.
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            }
        )

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_loss",
                "param_count",
                "sizes",
                "pruned",
                "grown",
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
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "test_loss": row["test_loss"],
                    "param_count": row["param_count"],
                    "sizes": ",".join(str(x) for x in sizes),
                    "pruned": ",".join(str(x) for x in pruned),
                    "grown": ",".join(str(x) for x in grown),
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

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    return model, history
