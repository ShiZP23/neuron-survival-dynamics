import csv
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuron_survival_dynamics.init_only.models import build_model
from neuron_survival_dynamics.utils import count_trainable_params, save_json, set_seed


@dataclass
class DenseRunSummary:
    dataset: str
    model: str
    init_seed: int
    runtime_seed: int
    split_seed: int
    train_order_seed: int
    epochs: int
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
    param_count: int
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

    mean_loss = total_loss / max(total_examples, 1)
    mean_acc = total_correct / max(total_examples, 1)
    return mean_loss, mean_acc


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_history_csv(path: str, history: List[Dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "is_best_epoch",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_dense_classifier_run(
    dataset_name: str,
    model_name: str,
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
    lr: float,
    weight_decay: float = 0.0,
    deterministic_algorithms: bool = False,
    save_checkpoints: bool = True,
) -> DenseRunSummary:
    os.makedirs(run_dir, exist_ok=True)

    set_seed(runtime_seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = build_model(
            model_name=model_name,
            input_shape=input_shape,
            num_classes=num_classes,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    param_count = count_trainable_params(model)
    init_state = _clone_state_dict(model)

    best_val_loss = float("inf")
    best_epoch = 0
    best_snapshot: Optional[Dict[str, object]] = None
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _classification_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = _classification_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        test_loss, test_acc = _classification_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
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
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "is_best_epoch": is_best_epoch,
            }
        )

    final_row = history[-1]
    if best_snapshot is None or best_state_dict is None:
        raise RuntimeError("Best snapshot was not recorded during training.")

    summary = DenseRunSummary(
        dataset=dataset_name,
        model=model_name,
        init_seed=init_seed,
        runtime_seed=runtime_seed,
        split_seed=split_seed,
        train_order_seed=train_order_seed,
        epochs=epochs,
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
        param_count=param_count,
        run_dir=run_dir,
    )

    _save_history_csv(os.path.join(run_dir, "metrics.csv"), history)
    save_json(os.path.join(run_dir, "summary.json"), asdict(summary))

    checkpoint_payload = {
        "dataset": dataset_name,
        "model": model_name,
        "init_seed": init_seed,
        "runtime_seed": runtime_seed,
        "split_seed": split_seed,
        "train_order_seed": train_order_seed,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "state_dict": model.state_dict(),
        "best_state_dict": best_state_dict,
        "init_state_dict": init_state,
    }
    if save_checkpoints:
        torch.save(checkpoint_payload, os.path.join(run_dir, "model_bundle.pt"))

    return summary
