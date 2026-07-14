import csv
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("results/publishable_pilot_20260313/summary_three_task.csv")
OUT_DIR = Path("results/studies/prune_only_vs_grow_final_loss_20260314")
TASKS = ["simple", "medium", "hard"]
MODES = ["prune_only", "prune_grow_random", "prune_grow_split", "fixed"]
GROW_MODES = ["prune_grow_random", "prune_grow_split"]
MODE_COLORS = {
    "fixed": "#7f7f7f",
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


def collect_rows() -> List[Dict]:
    rows = []
    for row in read_csv(INPUT):
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
        for grow_mode in GROW_MODES:
            final_deltas = []
            final_log_ratios = []
            final_wins = 0
            selected_deltas = []
            selected_log_ratios = []
            selected_wins = 0
            seeds = sorted({row["seed"] for row in rows if row["task"] == task})
            for seed in seeds:
                prune = next(row for row in rows if row["task"] == task and row["mode"] == "prune_only" and row["seed"] == seed)
                grow = next(row for row in rows if row["task"] == task and row["mode"] == grow_mode and row["seed"] == seed)
                final_delta = prune["final_test_loss"] - grow["final_test_loss"]
                final_log_ratio = np.log10(prune["final_test_loss"] / grow["final_test_loss"])
                selected_delta = prune["test_loss_at_best_val"] - grow["test_loss_at_best_val"]
                selected_log_ratio = np.log10(prune["test_loss_at_best_val"] / grow["test_loss_at_best_val"])
                final_deltas.append(final_delta)
                final_log_ratios.append(final_log_ratio)
                selected_deltas.append(selected_delta)
                selected_log_ratios.append(selected_log_ratio)
                if prune["final_test_loss"] < grow["final_test_loss"]:
                    final_wins += 1
                if prune["test_loss_at_best_val"] < grow["test_loss_at_best_val"]:
                    selected_wins += 1
            out.append(
                {
                    "task": task,
                    "grow_mode": grow_mode,
                    "n": len(seeds),
                    "prune_only_final_win_rate": final_wins / len(seeds),
                    "mean_delta_final_test_loss": mean(final_deltas),
                    "median_delta_final_test_loss": median(final_deltas),
                    "mean_log10_ratio_final_prune_over_grow": mean(final_log_ratios),
                    "median_log10_ratio_final_prune_over_grow": median(final_log_ratios),
                    "prune_only_selected_win_rate": selected_wins / len(seeds),
                    "mean_delta_selected_test_loss": mean(selected_deltas),
                    "median_delta_selected_test_loss": median(selected_deltas),
                    "mean_log10_ratio_selected_prune_over_grow": mean(selected_log_ratios),
                    "median_log10_ratio_selected_prune_over_grow": median(selected_log_ratios),
                }
            )
    return out


def plot_final_loss_distribution(rows: List[Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    plot_modes = ["prune_only", "prune_grow_random", "prune_grow_split"]
    for ax, task in zip(axes, TASKS):
        vals = []
        labels = []
        colors = []
        fixed_group = [row["final_test_loss"] for row in rows if row["task"] == task and row["mode"] == "fixed"]
        fixed_median = median(fixed_group)
        for mode in plot_modes:
            group = [row["final_test_loss"] for row in rows if row["task"] == task and row["mode"] == mode]
            vals.append(group)
            labels.append(mode.replace("prune_", ""))
            colors.append(MODE_COLORS[mode])
        bp = ax.boxplot(vals, patch_artist=True, tick_labels=labels, showfliers=False, widths=0.55)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for idx, mode in enumerate(plot_modes, start=1):
            group = [row["final_test_loss"] for row in rows if row["task"] == task and row["mode"] == mode]
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
        ax.axhline(fixed_median, color=MODE_COLORS["fixed"], linestyle="--", linewidth=1.2, label="fixed median")
        ax.set_yscale("log")
        ax.set_title(task)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("final test loss")
    axes[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Final test loss by task: prune_only vs grow strategies", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "final_loss_distribution_by_task.png", dpi=180)
    plt.close(fig)


def plot_selected_loss_distribution(rows: List[Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    plot_modes = ["prune_only", "prune_grow_random", "prune_grow_split"]
    for ax, task in zip(axes, TASKS):
        vals = []
        labels = []
        colors = []
        fixed_group = [row["test_loss_at_best_val"] for row in rows if row["task"] == task and row["mode"] == "fixed"]
        fixed_median = median(fixed_group)
        for mode in plot_modes:
            group = [row["test_loss_at_best_val"] for row in rows if row["task"] == task and row["mode"] == mode]
            vals.append(group)
            labels.append(mode.replace("prune_", ""))
            colors.append(MODE_COLORS[mode])
        bp = ax.boxplot(vals, patch_artist=True, tick_labels=labels, showfliers=False, widths=0.55)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for idx, mode in enumerate(plot_modes, start=1):
            group = [row["test_loss_at_best_val"] for row in rows if row["task"] == task and row["mode"] == mode]
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
        ax.axhline(fixed_median, color=MODE_COLORS["fixed"], linestyle="--", linewidth=1.2, label="fixed median")
        ax.set_yscale("log")
        ax.set_title(task)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("test loss @ best val")
    axes[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Validation-selected test loss by task: prune_only vs grow strategies", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "selected_loss_distribution_by_task.png", dpi=180)
    plt.close(fig)


def plot_paired_slopegraphs(rows: List[Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(len(TASKS), len(GROW_MODES), figsize=(11.5, 10.5), sharey="row")
    for row_idx, task in enumerate(TASKS):
        for col_idx, grow_mode in enumerate(GROW_MODES):
            ax = axes[row_idx, col_idx]
            seeds = sorted({row["seed"] for row in rows if row["task"] == task})
            for seed in seeds:
                prune = next(row for row in rows if row["task"] == task and row["mode"] == "prune_only" and row["seed"] == seed)
                grow = next(row for row in rows if row["task"] == task and row["mode"] == grow_mode and row["seed"] == seed)
                y = [prune["final_test_loss"], grow["final_test_loss"]]
                color = MODE_COLORS["prune_only"] if prune["final_test_loss"] < grow["final_test_loss"] else MODE_COLORS[grow_mode]
                ax.plot([0, 1], y, color=color, linewidth=1.3, alpha=0.9)
                ax.scatter([0, 1], y, color=[MODE_COLORS["prune_only"], MODE_COLORS[grow_mode]], s=26, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["prune_only", grow_mode.replace("prune_", "")], rotation=18, ha="right")
            ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            if row_idx == 0:
                ax.set_title(grow_mode.replace("prune_", ""))
            if col_idx == 0:
                ax.set_ylabel(f"{task}\nfinal test loss")
    fig.suptitle("Seed-matched final-loss comparisons", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_seed_slopegraphs.png", dpi=180)
    plt.close(fig)


def plot_win_rate_summary(pairwise_rows: List[Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(TASKS))
    width = 0.32
    for idx, grow_mode in enumerate(GROW_MODES):
        group = [row for row in pairwise_rows if row["grow_mode"] == grow_mode]
        axes[0].bar(
            x + (idx - 0.5) * width,
            [row["prune_only_final_win_rate"] for row in group],
            width=width,
            color=MODE_COLORS[grow_mode],
            alpha=0.85,
            label=grow_mode.replace("prune_", ""),
        )
        axes[1].bar(
            x + (idx - 0.5) * width,
            [row["median_log10_ratio_final_prune_over_grow"] for row in group],
            width=width,
            color=MODE_COLORS[grow_mode],
            alpha=0.85,
            label=grow_mode.replace("prune_", ""),
        )
    axes[0].axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("prune_only win rate over grow")
    axes[0].set_ylabel("win rate")
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1].set_title("median log10(final prune_only / final grow)")
    axes[1].set_ylabel("log10 ratio")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_summary.png", dpi=180)
    plt.close(fig)


def plot_selected_win_rate_summary(pairwise_rows: List[Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(TASKS))
    width = 0.32
    for idx, grow_mode in enumerate(GROW_MODES):
        group = [row for row in pairwise_rows if row["grow_mode"] == grow_mode]
        axes[0].bar(
            x + (idx - 0.5) * width,
            [row["prune_only_selected_win_rate"] for row in group],
            width=width,
            color=MODE_COLORS[grow_mode],
            alpha=0.85,
            label=grow_mode.replace("prune_", ""),
        )
        axes[1].bar(
            x + (idx - 0.5) * width,
            [row["median_log10_ratio_selected_prune_over_grow"] for row in group],
            width=width,
            color=MODE_COLORS[grow_mode],
            alpha=0.85,
            label=grow_mode.replace("prune_", ""),
        )
    axes[0].axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("prune_only selected-loss win rate over grow")
    axes[0].set_ylabel("win rate")
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1].set_title("median log10(selected prune_only / selected grow)")
    axes[1].set_ylabel("log10 ratio")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "selected_win_rate_summary.png", dpi=180)
    plt.close(fig)


def plot_hard_frontier(rows: List[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    hard_rows = [row for row in rows if row["task"] == "hard" and row["mode"] in MODES]
    for mode in MODES:
        group = [row for row in hard_rows if row["mode"] == mode]
        ax.scatter(
            [row["best_param_count"] for row in group],
            [row["final_test_loss"] for row in group],
            color=MODE_COLORS[mode],
            s=48,
            alpha=0.9,
            label=mode.replace("prune_", ""),
        )
    ax.set_yscale("log")
    ax.set_xlabel("best parameter count")
    ax.set_ylabel("final test loss")
    ax.set_title("Hard task: final-loss / parameter frontier")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hard_final_loss_frontier.png", dpi=180)
    plt.close(fig)


def write_readme(task_rows: List[Dict], pairwise_rows: List[Dict], out_dir: Path) -> None:
    lines = [
        "# Prune-Only vs Grow Final-Loss Study",
        "",
        "Branch objective:",
        "- test whether `prune_only` is more likely than grow strategies to finish with lower `final_test_loss` on harder tasks.",
        "",
        "Primary endpoint:",
        "- `final_test_loss`",
        "",
        "Secondary control endpoint:",
        "- `test_loss_at_best_val`",
        "- `best_param_count`",
        "- seed-matched win rate",
        "",
        "Primary reading order:",
        "1. `primary/final_loss_distribution_by_task.png`",
        "2. `primary/paired_seed_slopegraphs.png`",
        "3. `primary/win_rate_summary.png`",
        "4. `primary/hard_final_loss_frontier.png`",
        "",
        "Control reading order:",
        "1. `controls/selected_loss_distribution_by_task.png`",
        "2. `controls/selected_win_rate_summary.png`",
        "",
        "Current evidence:",
    ]
    for task in TASKS:
        prune = next(row for row in task_rows if row["task"] == task and row["mode"] == "prune_only")
        rand = next(row for row in task_rows if row["task"] == task and row["mode"] == "prune_grow_random")
        split = next(row for row in task_rows if row["task"] == task and row["mode"] == "prune_grow_split")
        lines.append(
            f"- {task}: mean final loss prune_only={prune['mean_final_test_loss']:.6g}, "
            f"grow_random={rand['mean_final_test_loss']:.6g}, grow_split={split['mean_final_test_loss']:.6g}"
        )
        lines.append(
            f"  - selected control: prune_only={prune['mean_selected_test_loss']:.6g}, "
            f"grow_random={rand['mean_selected_test_loss']:.6g}, grow_split={split['mean_selected_test_loss']:.6g}"
        )
        for grow_mode in GROW_MODES:
            pair = next(row for row in pairwise_rows if row["task"] == task and row["grow_mode"] == grow_mode)
            lines.append(
                f"  - vs {grow_mode.replace('prune_', '')}: final win rate={pair['prune_only_final_win_rate']:.2f}, "
                f"final median log10 ratio={pair['median_log10_ratio_final_prune_over_grow']:.4f}, "
                f"selected win rate={pair['prune_only_selected_win_rate']:.2f}"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- `hard` is the strongest supportive case if prune_only beats both grow strategies on final loss while using fewer parameters than grow-based methods.",
            "- `simple` and `medium` act as controls for the “harder tasks” qualifier.",
            "- `test_loss_at_best_val` should be read as a control: if final-loss differences are large while selected-loss differences are small, the claim is about end-of-training stability rather than peak attainable performance.",
            "- If prune_only only beats `split` but not `random`, the headline should be narrowed accordingly.",
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

    plot_final_loss_distribution(rows, primary_dir)
    plot_paired_slopegraphs(rows, primary_dir)
    plot_win_rate_summary(pairwise_rows, primary_dir)
    plot_hard_frontier(rows, primary_dir)
    plot_selected_loss_distribution(rows, controls_dir)
    plot_selected_win_rate_summary(pairwise_rows, controls_dir)
    write_readme(task_rows, pairwise_rows, OUT_DIR)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
