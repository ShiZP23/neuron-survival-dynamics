import csv
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("results/publishable_pilot_20260313/summary_three_task.csv")
OUT_DIR = Path("results/studies/prune_only_vs_fixed_final_loss_20260314")
TASKS = ["simple", "medium", "hard"]
MODES = ["prune_only", "fixed"]
MODE_COLORS = {
    "fixed": "#1f77b4",
    "prune_only": "#ff7f0e",
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


def collect_rows() -> List[Dict]:
    rows = []
    for row in read_csv(INPUT):
        if row["mode"] not in MODES:
            continue
        rows.append(
            {
                "task": row["task"],
                "mode": row["mode"],
                "seed": int(row["seed"]),
                "run_name": row["run_name"],
                "best_epoch": int(row["best_epoch"]),
                "test_loss_at_best_val": float(row["test_loss_at_best_val"]),
                "best_param_count": int(float(row["best_param_count"])),
                "final_test_loss": float(row["final_test_loss"]),
                "final_param_count": int(float(row["final_param_count"])),
            }
        )
    return rows


def summarize_task_modes(rows: List[Dict]) -> List[Dict]:
    out = []
    for task in TASKS:
        for mode in MODES:
            group = [row for row in rows if row["task"] == task and row["mode"] == mode]
            out.append(
                {
                    "task": task,
                    "mode": mode,
                    "n": len(group),
                    "mean_final_test_loss": mean(row["final_test_loss"] for row in group),
                    "median_final_test_loss": median(row["final_test_loss"] for row in group),
                    "mean_selected_test_loss": mean(row["test_loss_at_best_val"] for row in group),
                    "median_selected_test_loss": median(row["test_loss_at_best_val"] for row in group),
                    "mean_best_param_count": mean(row["best_param_count"] for row in group),
                    "median_best_param_count": median(row["best_param_count"] for row in group),
                }
            )
    return out


def pairwise_summary(rows: List[Dict]) -> List[Dict]:
    out = []
    for task in TASKS:
        final_deltas = []
        final_log_ratios = []
        final_wins = 0
        selected_deltas = []
        selected_log_ratios = []
        selected_wins = 0
        param_deltas = []
        seeds = sorted({row["seed"] for row in rows if row["task"] == task})
        for seed in seeds:
            prune = next(row for row in rows if row["task"] == task and row["mode"] == "prune_only" and row["seed"] == seed)
            fixed = next(row for row in rows if row["task"] == task and row["mode"] == "fixed" and row["seed"] == seed)
            final_delta = prune["final_test_loss"] - fixed["final_test_loss"]
            final_log_ratio = np.log10(prune["final_test_loss"] / fixed["final_test_loss"])
            selected_delta = prune["test_loss_at_best_val"] - fixed["test_loss_at_best_val"]
            selected_log_ratio = np.log10(prune["test_loss_at_best_val"] / fixed["test_loss_at_best_val"])
            param_delta = prune["best_param_count"] - fixed["best_param_count"]
            final_deltas.append(final_delta)
            final_log_ratios.append(final_log_ratio)
            selected_deltas.append(selected_delta)
            selected_log_ratios.append(selected_log_ratio)
            param_deltas.append(param_delta)
            if prune["final_test_loss"] < fixed["final_test_loss"]:
                final_wins += 1
            if prune["test_loss_at_best_val"] < fixed["test_loss_at_best_val"]:
                selected_wins += 1
        out.append(
            {
                "task": task,
                "n": len(seeds),
                "prune_only_final_win_rate": final_wins / len(seeds),
                "mean_delta_final_test_loss": mean(final_deltas),
                "median_delta_final_test_loss": median(final_deltas),
                "mean_log10_ratio_final_prune_over_fixed": mean(final_log_ratios),
                "median_log10_ratio_final_prune_over_fixed": median(final_log_ratios),
                "prune_only_selected_win_rate": selected_wins / len(seeds),
                "mean_delta_selected_test_loss": mean(selected_deltas),
                "median_delta_selected_test_loss": median(selected_deltas),
                "mean_log10_ratio_selected_prune_over_fixed": mean(selected_log_ratios),
                "median_log10_ratio_selected_prune_over_fixed": median(selected_log_ratios),
                "mean_delta_best_param_count": mean(param_deltas),
                "median_delta_best_param_count": median(param_deltas),
            }
        )
    return out


def plot_distribution(rows: List[Dict], key: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        prune_group = [row[key] for row in rows if row["task"] == task and row["mode"] == "prune_only"]
        fixed_group = [row[key] for row in rows if row["task"] == task and row["mode"] == "fixed"]
        vals = [prune_group, fixed_group]
        bp = ax.boxplot(vals, patch_artist=True, tick_labels=["prune_only", "fixed"], showfliers=False, widths=0.55)
        for patch, color in zip(bp["boxes"], [MODE_COLORS["prune_only"], MODE_COLORS["fixed"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for idx, (mode, group) in enumerate([("prune_only", prune_group), ("fixed", fixed_group)], start=1):
            jitter = np.linspace(-0.08, 0.08, len(group))
            ax.scatter(
                np.full(len(group), idx, dtype=float) + jitter,
                group,
                color=MODE_COLORS[mode],
                s=38,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_title(task)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    if "loss" in key:
        for ax in axes:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_paired_slopegraphs(rows: List[Dict], key: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        seeds = sorted({row["seed"] for row in rows if row["task"] == task})
        for seed in seeds:
            prune = next(row for row in rows if row["task"] == task and row["mode"] == "prune_only" and row["seed"] == seed)
            fixed = next(row for row in rows if row["task"] == task and row["mode"] == "fixed" and row["seed"] == seed)
            y = [prune[key], fixed[key]]
            color = MODE_COLORS["prune_only"] if prune[key] < fixed[key] else MODE_COLORS["fixed"]
            ax.plot([0, 1], y, color=color, linewidth=1.3, alpha=0.9)
            ax.scatter([0, 1], y, color=[MODE_COLORS["prune_only"], MODE_COLORS["fixed"]], s=26, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["prune_only", "fixed"], rotation=18, ha="right")
        ax.set_title(task)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if "loss" in key:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_win_rates(pairwise_rows: List[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    x = np.arange(len(TASKS))
    final_vals = [row["prune_only_final_win_rate"] for row in pairwise_rows]
    selected_vals = [row["prune_only_selected_win_rate"] for row in pairwise_rows]
    axes[0].bar(x, final_vals, color=MODE_COLORS["prune_only"], alpha=0.85)
    axes[1].bar(x, selected_vals, color=MODE_COLORS["fixed"], alpha=0.85)
    for ax, title in zip(axes, ["final-loss win rate", "selected-loss win rate"]):
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS)
        ax.set_ylabel("prune_only win rate")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hard_frontier(rows: List[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    hard_rows = [row for row in rows if row["task"] == "hard"]
    for mode in MODES:
        group = [row for row in hard_rows if row["mode"] == mode]
        ax.scatter(
            [row["best_param_count"] for row in group],
            [row["final_test_loss"] for row in group],
            color=MODE_COLORS[mode],
            s=52,
            alpha=0.9,
            label=mode,
        )
    ax.set_yscale("log")
    ax.set_xlabel("best parameter count")
    ax.set_ylabel("final test loss")
    ax.set_title("Hard task: prune_only vs fixed frontier")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(task_rows: List[Dict], pairwise_rows: List[Dict], out_dir: Path) -> None:
    lines = [
        "# Prune-Only vs Fixed Final-Loss Study",
        "",
        "Primary question:",
        "- does `prune_only` finish training with lower `final_test_loss` than `fixed`, especially on harder tasks?",
        "",
        "Reading order:",
        "1. `primary/final_loss_distribution_by_task.png`",
        "2. `primary/paired_seed_final_loss.png`",
        "3. `primary/win_rate_summary.png`",
        "4. `primary/hard_final_loss_frontier.png`",
        "5. `controls/selected_loss_distribution_by_task.png`",
        "6. `controls/paired_seed_selected_loss.png`",
        "",
        "Current evidence:",
    ]
    for task in TASKS:
        prune = next(row for row in task_rows if row["task"] == task and row["mode"] == "prune_only")
        fixed = next(row for row in task_rows if row["task"] == task and row["mode"] == "fixed")
        pair = next(row for row in pairwise_rows if row["task"] == task)
        lines.append(
            f"- {task}: final mean prune_only={prune['mean_final_test_loss']:.6g}, fixed={fixed['mean_final_test_loss']:.6g}; "
            f"selected mean prune_only={prune['mean_selected_test_loss']:.6g}, fixed={fixed['mean_selected_test_loss']:.6g}"
        )
        lines.append(
            f"  - win rates: final={pair['prune_only_final_win_rate']:.2f}, selected={pair['prune_only_selected_win_rate']:.2f}; "
            f"median log10 ratios: final={pair['median_log10_ratio_final_prune_over_fixed']:.4f}, selected={pair['median_log10_ratio_selected_prune_over_fixed']:.4f}"
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "- if prune_only does not beat fixed on `hard` final loss, the branch hypothesis is not supported.",
            "- if prune_only only beats fixed on `selected` but not `final`, the difference should be described as peak-performance or stability mismatch, not final-performance superiority.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    primary_dir = OUT_DIR / "primary"
    controls_dir = OUT_DIR / "controls"
    primary_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    task_rows = summarize_task_modes(rows)
    pairwise_rows = pairwise_summary(rows)
    write_csv(OUT_DIR / "rows.csv", rows)
    write_csv(OUT_DIR / "task_mode_summary.csv", task_rows)
    write_csv(OUT_DIR / "pairwise_summary.csv", pairwise_rows)

    plot_distribution(
        rows,
        "final_test_loss",
        "final test loss",
        "Final test loss by task: prune_only vs fixed",
        primary_dir / "final_loss_distribution_by_task.png",
    )
    plot_paired_slopegraphs(
        rows,
        "final_test_loss",
        "final test loss",
        "Seed-matched final-loss comparison: prune_only vs fixed",
        primary_dir / "paired_seed_final_loss.png",
    )
    plot_win_rates(pairwise_rows, primary_dir / "win_rate_summary.png")
    plot_hard_frontier(rows, primary_dir / "hard_final_loss_frontier.png")

    plot_distribution(
        rows,
        "test_loss_at_best_val",
        "test loss @ best val",
        "Validation-selected test loss by task: prune_only vs fixed",
        controls_dir / "selected_loss_distribution_by_task.png",
    )
    plot_paired_slopegraphs(
        rows,
        "test_loss_at_best_val",
        "test loss @ best val",
        "Seed-matched selected-loss comparison: prune_only vs fixed",
        controls_dir / "paired_seed_selected_loss.png",
    )

    write_readme(task_rows, pairwise_rows, OUT_DIR)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
