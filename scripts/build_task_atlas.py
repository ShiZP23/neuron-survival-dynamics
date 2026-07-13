import csv
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


TASKS = ["hard", "medium", "simple"]
MODES = ["fixed", "prune_only", "prune_grow_random", "prune_grow_split"]
MODE_COLORS = {
    "fixed": "#1f77b4",
    "prune_only": "#ff7f0e",
    "prune_grow_random": "#2ca02c",
    "prune_grow_split": "#d62728",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for task in TASKS:
        for mode in MODES:
            for metrics_path in sorted((root / task / mode).glob("seed_*/*/metrics.csv")):
                metrics = read_csv(metrics_path)
                if not metrics or "val_loss" not in metrics[0]:
                    continue
                best = min(metrics, key=lambda row: float(row["val_loss"]))
                final = metrics[-1]
                updates = [row for row in metrics if int(row["is_update_epoch"]) == 1]
                rows.append(
                    {
                        "task": task,
                        "mode": mode,
                        "seed": int(metrics_path.parent.parent.name.replace("seed_", "")),
                        "run_name": metrics_path.parent.name,
                        "run_dir": str(metrics_path.parent),
                        "best_epoch": int(best["epoch"]),
                        "best_val_loss": float(best["val_loss"]),
                        "selected_test_loss": float(best["test_loss"]),
                        "final_test_loss": float(final["test_loss"]),
                        "selected_params": int(float(best["param_count"])),
                        "final_params": int(float(final["param_count"])),
                        "instability_gap": float(final["test_loss"]) - float(best["test_loss"]),
                        "total_pruned": sum(int(row["total_pruned"]) for row in updates),
                        "total_grown": sum(int(row["total_grown"]) for row in updates),
                        "pruned_0_total": sum(int(row["pruned_0"]) for row in updates),
                        "pruned_1_total": sum(int(row["pruned_1"]) for row in updates),
                        "pruned_2_total": sum(int(row["pruned_2"]) for row in updates),
                        "selected_active_0": int(best.get("active_0", 0)),
                        "selected_active_1": int(best.get("active_1", 0)),
                        "selected_active_2": int(best.get("active_2", 0)),
                    }
                )
    return rows


def plot_loss_trajectory_by_task(root: Path, out_dir: Path, task: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes_list = [ax for row_axes in axes for ax in row_axes]
    for ax, mode in zip(axes_list, MODES):
        seed_curves: List[np.ndarray] = []
        epochs = None
        for metrics_path in sorted((root / task / mode).glob("seed_*/*/metrics.csv")):
            metrics = read_csv(metrics_path)
            if not metrics or "val_loss" not in metrics[0]:
                continue
            test = np.array([max(float(row["test_loss"]), 1e-12) for row in metrics], dtype=float)
            seed_curves.append(test)
            if epochs is None:
                epochs = np.array([int(row["epoch"]) for row in metrics], dtype=int)
            ax.plot(epochs, test, color=MODE_COLORS[mode], alpha=0.18, linewidth=0.9)
        if seed_curves and epochs is not None:
            curve_array = np.stack(seed_curves, axis=0)
            q25 = np.quantile(curve_array, 0.25, axis=0)
            q50 = np.quantile(curve_array, 0.50, axis=0)
            q75 = np.quantile(curve_array, 0.75, axis=0)
            ax.plot(epochs, q50, color=MODE_COLORS[mode], linewidth=2.2)
            ax.fill_between(epochs, q25, q75, color=MODE_COLORS[mode], alpha=0.22)
        ax.set_yscale("log")
        ax.set_title(mode)
        ax.grid(True, linestyle="--", alpha=0.28)
    axes[0, 0].set_ylabel("test loss")
    axes[1, 0].set_ylabel("test loss")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_xlabel("epoch")
    fig.suptitle(f"{task}: seed-wise test-loss trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_seedwise_test_loss_atlas.png", dpi=170)
    plt.close(fig)


def _stack_metric_curves(
    root: Path, task: str, mode: str, metric_key: str, cumulative: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    seed_curves: List[np.ndarray] = []
    epochs = None
    for metrics_path in sorted((root / task / mode).glob("seed_*/*/metrics.csv")):
        metrics = read_csv(metrics_path)
        if not metrics or metric_key not in metrics[0]:
            continue
        curve = np.array([float(row[metric_key]) for row in metrics], dtype=float)
        if cumulative:
            curve = np.cumsum(curve)
        seed_curves.append(curve)
        if epochs is None:
            epochs = np.array([int(row["epoch"]) for row in metrics], dtype=int)
    if not seed_curves or epochs is None:
        return None, None
    return epochs, np.stack(seed_curves, axis=0)


def _plot_iqr(
    ax, epochs: np.ndarray, curves: np.ndarray, color: str, label: str, log_scale: bool = False
) -> None:
    q25 = np.quantile(curves, 0.25, axis=0)
    q50 = np.quantile(curves, 0.50, axis=0)
    q75 = np.quantile(curves, 0.75, axis=0)
    if log_scale:
        q25 = np.maximum(q25, 1e-12)
        q50 = np.maximum(q50, 1e-12)
        q75 = np.maximum(q75, 1e-12)
        ax.set_yscale("log")
    ax.plot(epochs, q50, color=color, linewidth=2.0, label=label)
    ax.fill_between(epochs, q25, q75, color=color, alpha=0.18)
    ax.grid(True, linestyle="--", alpha=0.28)


def plot_structural_dynamics(root: Path, out_dir: Path, task: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    panels = [
        ("param_count", False, "parameter count"),
        ("pruned_0", True, "cumulative pruned_0"),
        ("pruned_1", True, "cumulative pruned_1"),
        ("pruned_2", True, "cumulative pruned_2"),
    ]
    for ax, (metric_key, cumulative, title) in zip([ax for row_axes in axes for ax in row_axes], panels):
        for mode in MODES:
            epochs, curves = _stack_metric_curves(root, task, mode, metric_key, cumulative=cumulative)
            if epochs is None or curves is None:
                continue
            _plot_iqr(ax, epochs, curves, MODE_COLORS[mode], mode)
        ax.set_title(title)
    axes[0, 0].set_ylabel("count")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_xlabel("epoch")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"{task}: structural dynamics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_structural_dynamics.png", dpi=170)
    plt.close(fig)


def plot_active_dynamics(root: Path, out_dir: Path, task: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, metric_key in zip(axes, ["active_0", "active_1", "active_2"]):
        for mode in MODES:
            epochs, curves = _stack_metric_curves(root, task, mode, metric_key, cumulative=False)
            if epochs is None or curves is None:
                continue
            _plot_iqr(ax, epochs, curves, MODE_COLORS[mode], mode)
        ax.set_title(metric_key)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("active neurons")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{task}: active-neuron dynamics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_active_dynamics.png", dpi=170)
    plt.close(fig)


def plot_selected_vs_final_by_task(rows: List[Dict], out_dir: Path, task: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    subset = [row for row in rows if row["task"] == task]
    for mode in MODES:
        group = [row for row in subset if row["mode"] == mode]
        axes[0].scatter(
            [row["selected_test_loss"] for row in group],
            [row["final_test_loss"] for row in group],
            color=MODE_COLORS[mode],
            s=48,
            alpha=0.9,
            label=mode,
        )
        axes[1].scatter(
            [row["selected_params"] for row in group],
            [row["selected_test_loss"] for row in group],
            color=MODE_COLORS[mode],
            s=48,
            alpha=0.9,
            label=mode,
        )
    low = min(min(row["selected_test_loss"] for row in subset), min(row["final_test_loss"] for row in subset))
    high = max(max(row["selected_test_loss"] for row in subset), max(row["final_test_loss"] for row in subset))
    axes[0].plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1.1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("test loss @ best val")
    axes[0].set_ylabel("final test loss")
    axes[0].set_title(f"{task}: selected vs final")
    axes[0].grid(True, linestyle="--", alpha=0.28)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("selected params")
    axes[1].set_ylabel("test loss @ best val")
    axes[1].set_title(f"{task}: parameter-performance frontier")
    axes[1].grid(True, linestyle="--", alpha=0.28)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_selected_final_frontier.png", dpi=170)
    plt.close(fig)


def plot_task_bar_panels(rows: List[Dict], out_dir: Path, task: str) -> None:
    subset = [row for row in rows if row["task"] == task]
    summary = {}
    for mode in MODES:
        group = [row for row in subset if row["mode"] == mode]
        summary[mode] = {
            "selected_loss": mean(row["selected_test_loss"] for row in group),
            "epoch": mean(row["best_epoch"] for row in group),
            "params": mean(row["selected_params"] for row in group),
            "gap": mean(row["instability_gap"] for row in group),
            "active0": mean(row["selected_active_0"] for row in group),
            "active1": mean(row["selected_active_1"] for row in group),
            "active2": mean(row["selected_active_2"] for row in group),
            "pruned0": mean(row["pruned_0_total"] for row in group),
            "pruned1": mean(row["pruned_1_total"] for row in group),
            "pruned2": mean(row["pruned_2_total"] for row in group),
        }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes_list = [ax for row_axes in axes for ax in row_axes]
    mode_x = np.arange(len(MODES))
    colors = [MODE_COLORS[mode] for mode in MODES]

    axes_list[0].bar(mode_x, [summary[mode]["selected_loss"] for mode in MODES], color=colors)
    axes_list[0].set_yscale("log")
    axes_list[0].set_title("mean selected test loss")

    axes_list[1].bar(mode_x, [summary[mode]["epoch"] for mode in MODES], color=colors)
    axes_list[1].set_title("mean best epoch")

    axes_list[2].bar(mode_x, [summary[mode]["params"] for mode in MODES], color=colors)
    axes_list[2].set_title("mean selected params")

    axes_list[3].bar(mode_x, [max(summary[mode]["gap"], 1e-12) for mode in MODES], color=colors)
    axes_list[3].set_yscale("log")
    axes_list[3].set_title("mean instability gap")

    width = 0.25
    axes_list[4].bar(mode_x - width, [summary[mode]["active0"] for mode in MODES], width=width, label="active_0", color="#4c78a8")
    axes_list[4].bar(mode_x, [summary[mode]["active1"] for mode in MODES], width=width, label="active_1", color="#f58518")
    axes_list[4].bar(mode_x + width, [summary[mode]["active2"] for mode in MODES], width=width, label="active_2", color="#54a24b")
    axes_list[4].set_title("mean active neurons @ best val")
    axes_list[4].legend(fontsize=8)

    axes_list[5].bar(mode_x - width, [summary[mode]["pruned0"] for mode in MODES], width=width, label="pruned_0", color="#4c78a8")
    axes_list[5].bar(mode_x, [summary[mode]["pruned1"] for mode in MODES], width=width, label="pruned_1", color="#f58518")
    axes_list[5].bar(mode_x + width, [summary[mode]["pruned2"] for mode in MODES], width=width, label="pruned_2", color="#54a24b")
    axes_list[5].set_title("mean cumulative pruned neurons")
    axes_list[5].legend(fontsize=8)

    for ax in axes_list:
        ax.set_xticks(mode_x)
        ax.set_xticklabels([mode.replace("prune_", "") for mode in MODES], rotation=18, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle(f"{task}: method comparison panel", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_method_panels.png", dpi=170)
    plt.close(fig)


def plot_seed_tables(rows: List[Dict], out_dir: Path, task: str) -> None:
    subset = [row for row in rows if row["task"] == task]
    seeds = sorted({row["seed"] for row in subset})
    metrics = [
        ("selected_test_loss", "selected_loss"),
        ("best_epoch", "best_epoch"),
        ("selected_params", "selected_params"),
        ("instability_gap", "instability_gap"),
        ("pruned_0_total", "pruned_0_total"),
        ("pruned_1_total", "pruned_1_total"),
        ("pruned_2_total", "pruned_2_total"),
    ]
    for key, label in metrics:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(MODES))
        for seed in seeds:
            vals = [next(row[key] for row in subset if row["seed"] == seed and row["mode"] == mode) for mode in MODES]
            if "loss" in key or "gap" in key:
                vals = [max(float(v), 1e-12) for v in vals]
            ax.plot(x, vals, color="#bbbbbb", linewidth=0.9, alpha=0.7, zorder=1)
        for idx, mode in enumerate(MODES):
            vals = [next(row[key] for row in subset if row["seed"] == seed and row["mode"] == mode) for seed in seeds]
            if "loss" in key or "gap" in key:
                vals = [max(float(v), 1e-12) for v in vals]
            jitter = np.linspace(-0.06, 0.06, len(vals))
            ax.scatter(
                np.full(len(vals), idx, dtype=float) + jitter,
                vals,
                color=MODE_COLORS[mode],
                s=40,
                alpha=0.95,
                label=mode if idx == 0 else None,
                zorder=2,
            )
        if "loss" in key or "gap" in key:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([mode.replace("prune_", "") for mode in MODES], rotation=18, ha="right")
        ax.set_xlabel("method")
        ax.set_title(f"{task}: paired-seed {label}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"{task}_{label}_seedwise.png", dpi=170)
        plt.close(fig)


def main() -> None:
    root = Path("results/publishable_pilot_20260313")
    atlas_dir = root / "task_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    primary_dir = atlas_dir / "primary"
    supplementary_dir = atlas_dir / "supplementary"
    archive_dir = atlas_dir / "archive"
    primary_dir.mkdir(exist_ok=True)
    supplementary_dir.mkdir(exist_ok=True)
    archive_dir.mkdir(exist_ok=True)

    rows = collect_runs(root)
    write_csv(atlas_dir / "task_atlas_rows.csv", rows)

    summary_rows: List[Dict] = []
    for task in TASKS:
        for mode in MODES:
            subset = [row for row in rows if row["task"] == task and row["mode"] == mode]
            summary_rows.append(
                {
                    "task": task,
                    "mode": mode,
                    "n": len(subset),
                    "mean_selected_loss": mean(row["selected_test_loss"] for row in subset),
                    "median_selected_loss": median(row["selected_test_loss"] for row in subset),
                    "std_selected_loss": pstdev(row["selected_test_loss"] for row in subset),
                    "mean_best_epoch": mean(row["best_epoch"] for row in subset),
                    "mean_selected_params": mean(row["selected_params"] for row in subset),
                    "mean_instability_gap": mean(row["instability_gap"] for row in subset),
                    "mean_pruned_0_total": mean(row["pruned_0_total"] for row in subset),
                    "mean_pruned_1_total": mean(row["pruned_1_total"] for row in subset),
                    "mean_pruned_2_total": mean(row["pruned_2_total"] for row in subset),
                }
            )
        plot_loss_trajectory_by_task(root, primary_dir, task)
        plot_structural_dynamics(root, primary_dir, task)
        plot_active_dynamics(root, primary_dir, task)
        plot_selected_vs_final_by_task(rows, primary_dir, task)
        plot_task_bar_panels(rows, archive_dir, task)
        plot_seed_tables(rows, supplementary_dir, task)

    write_csv(atlas_dir / "task_atlas_summary.csv", summary_rows)
    print(atlas_dir)


if __name__ == "__main__":
    main()
