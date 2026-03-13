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


def plot_active_neurons(history: List[Dict], path: str) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    active = [row["active_counts"] for row in history]
    active_array = np.array(active)

    plt.figure(figsize=(7, 4))
    for idx in range(active_array.shape[1]):
        plt.plot(epochs, active_array[:, idx], label=f"active_{idx}", linewidth=1.7)
    plt.xlabel("epoch")
    plt.ylabel("active neurons")
    plt.title("Active Neurons over Time")
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


def plot_turnover(history: List[Dict], path: str) -> None:
    update_rows = [row for row in history if row.get("is_update_epoch", 0) == 1]
    if not update_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No structural updates in this run",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return

    update_idx = np.arange(1, len(update_rows) + 1)
    xpos = np.arange(len(update_rows))
    total_pruned = [row["total_pruned"] for row in update_rows]
    total_grown = [row["total_grown"] for row in update_rows]

    # Growth bars are only shown when they add non-redundant information.
    show_grown = any(p != g for p, g in zip(total_pruned, total_grown))
    layer_count = len(update_rows[0]["pruned"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    if show_grown:
        bar_w = 0.42
        axes[0].bar(
            xpos - bar_w / 2,
            total_pruned,
            width=bar_w,
            label="total_pruned",
            color="#d62728",
            alpha=0.9,
        )
        axes[0].bar(
            xpos + bar_w / 2,
            total_grown,
            width=bar_w,
            label="total_grown",
            color="#1f77b4",
            alpha=0.9,
        )
    else:
        axes[0].bar(
            xpos,
            total_pruned,
            width=0.65,
            label="total_pruned",
            color="#d62728",
            alpha=0.9,
        )
        axes[0].text(
            0.99,
            0.95,
            "total_grown matches total_pruned at all updates",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="dimgray",
        )

    axes[0].set_ylabel("neurons per update", fontsize=11)
    axes[0].set_title("Discrete Structural Turnover (Update Epochs Only)", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # Layer-wise prune turnover as stacked bars to reduce clutter.
    colors = ["#ff9896", "#c5b0d5", "#9edae5"]
    bottom = np.zeros(len(update_rows))
    for idx in range(layer_count):
        pruned_i = np.array([row["pruned"][idx] for row in update_rows], dtype=float)
        axes[1].bar(
            xpos,
            pruned_i,
            bottom=bottom,
            width=0.65,
            label=f"pruned_{idx}",
            color=colors[idx % len(colors)],
            alpha=0.95,
        )
        bottom += pruned_i

    axes[1].set_ylabel("pruned neurons", fontsize=11)
    axes[1].set_xlabel("update index", fontsize=11)
    axes[1].set_title("Layer-wise Pruned Counts per Structural Update", fontsize=12)
    axes[1].legend(ncol=min(layer_count, 3), fontsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(update_idx)

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
