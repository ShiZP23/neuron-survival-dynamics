from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuron_survival_dynamics.data import evaluate_target, make_grid
from neuron_survival_dynamics.model import MLP


def plot_losses(history: List[Dict], path: str) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, test_loss, label="test")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("Loss over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_sizes(history: List[Dict], path: str) -> None:
    epochs = [row["epoch"] for row in history]
    sizes = [row["sizes"] for row in history]
    size_array = np.array(sizes)

    plt.figure(figsize=(6, 4))
    for idx in range(size_array.shape[1]):
        plt.plot(epochs, size_array[:, idx], label=f"layer_{idx}")
    plt.xlabel("epoch")
    plt.ylabel("neurons")
    plt.title("Hidden layer sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_param_count(history: List[Dict], path: str) -> None:
    epochs = [row["epoch"] for row in history]
    params = [row["param_count"] for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, params, label="params")
    plt.xlabel("epoch")
    plt.ylabel("trainable params")
    plt.title("Parameter count over time")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_surface(
    model: MLP,
    task: str,
    path: str,
    device: torch.device,
    grid_size: int = 200,
) -> None:
    model.eval()
    grid = make_grid(grid_size)
    with torch.no_grad():
        pred = model(grid.to(device)).cpu().numpy().reshape(grid_size, grid_size)
    target = evaluate_target(task, grid).numpy().reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(target, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    axes[0].set_title("target")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    axes[1].set_title("prediction")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
