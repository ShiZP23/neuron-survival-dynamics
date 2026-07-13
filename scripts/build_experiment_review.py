import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


TASKS = ["hard", "medium", "simple"]
MODES = ["fixed", "prune_only", "prune_grow_random", "prune_grow_split"]
MODE_LABELS = {
    "fixed": "fixed",
    "prune_only": "prune_only",
    "prune_grow_random": "grow_random",
    "prune_grow_split": "grow_split",
}
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


def summarize_main_results(summary_path: Path) -> Tuple[List[Dict], List[Dict]]:
    rows = read_csv(summary_path)
    stats_rows: List[Dict] = []
    targets: Dict[str, float] = {}
    for task in TASKS:
        fixed_rows = [row for row in rows if row["task"] == task and row["mode"] == "fixed"]
        targets[task] = median(float(row["test_loss_at_best_val"]) for row in fixed_rows)

    time_to_target_rows: List[Dict] = []
    root = summary_path.parent
    for row in rows:
        metrics_path = (
            root
            / row["task"]
            / row["mode"]
            / f"seed_{row['seed']}"
            / row["run_name"]
            / "metrics.csv"
        )
        metrics = read_csv(metrics_path)
        target = targets[row["task"]]
        reach_epoch = None
        for item in metrics:
            if float(item["val_loss"]) <= target:
                reach_epoch = int(item["epoch"])
                break
        time_to_target_rows.append(
            {
                "task": row["task"],
                "mode": row["mode"],
                "seed": row["seed"],
                "target_val_loss": target,
                "epoch_to_fixed_target": reach_epoch,
                "best_epoch": int(row["best_epoch"]),
                "selected_test_loss": float(row["test_loss_at_best_val"]),
                "selected_params": int(row["best_param_count"]),
                "final_test_loss": float(row["final_test_loss"]),
                "instability_gap": float(row["final_test_loss"]) - float(row["test_loss_at_best_val"]),
            }
        )

    for task in TASKS:
        for mode in MODES:
            subset = [row for row in time_to_target_rows if row["task"] == task and row["mode"] == mode]
            reach_epochs = [row["epoch_to_fixed_target"] for row in subset if row["epoch_to_fixed_target"] is not None]
            stats_rows.append(
                {
                    "task": task,
                    "mode": mode,
                    "n": len(subset),
                    "mean_selected_loss": mean(row["selected_test_loss"] for row in subset),
                    "median_selected_loss": median(row["selected_test_loss"] for row in subset),
                    "std_selected_loss": pstdev(row["selected_test_loss"] for row in subset),
                    "mean_final_loss": mean(row["final_test_loss"] for row in subset),
                    "median_final_loss": median(row["final_test_loss"] for row in subset),
                    "mean_instability_gap": mean(row["instability_gap"] for row in subset),
                    "median_instability_gap": median(row["instability_gap"] for row in subset),
                    "mean_selected_params": mean(row["selected_params"] for row in subset),
                    "median_selected_params": median(row["selected_params"] for row in subset),
                    "mean_best_epoch": mean(row["best_epoch"] for row in subset),
                    "mean_epoch_to_fixed_target": mean(reach_epochs) if reach_epochs else None,
                }
            )
    return stats_rows, time_to_target_rows


def classify_pattern(pruned0: int, pruned1: int, pruned2: int) -> str:
    if pruned0 > 0 and pruned2 > 0 and pruned1 == 0:
        return "p0+p2"
    if pruned0 > 0 and pruned1 == 0 and pruned2 == 0:
        return "p0_only"
    if pruned0 == 0 and pruned1 == 0 and pruned2 == 0:
        return "none"
    return "other"


def summarize_legacy_patterns(root: Path) -> Tuple[List[Dict], List[Dict]]:
    rows: List[Dict] = []
    for metrics_path in sorted(root.glob("prune_grow_split/seed_*/*/metrics.csv")):
        metrics = read_csv(metrics_path)
        if not metrics:
            continue
        updates = [row for row in metrics if int(row["is_update_epoch"]) == 1]
        pruned0 = sum(int(row["pruned_0"]) for row in updates)
        pruned1 = sum(int(row["pruned_1"]) for row in updates)
        pruned2 = sum(int(row["pruned_2"]) for row in updates)
        pattern = classify_pattern(pruned0, pruned1, pruned2)
        if pattern == "other":
            continue
        final = float(metrics[-1]["test_loss"])
        best_oracle = min(float(row["test_loss"]) for row in metrics)
        rows.append(
            {
                "run_path": str(metrics_path.parent),
                "seed_dir": metrics_path.parent.parent.name,
                "run_name": metrics_path.parent.name,
                "pattern": pattern,
                "final_test_loss": final,
                "oracle_best_test_loss": best_oracle,
                "oracle_gap": final - best_oracle,
                "total_pruned": sum(int(row["total_pruned"]) for row in updates),
                "pruned_0_total": pruned0,
                "pruned_1_total": pruned1,
                "pruned_2_total": pruned2,
            }
        )
    stats: List[Dict] = []
    for pattern in ["none", "p0_only", "p0+p2"]:
        subset = [row for row in rows if row["pattern"] == pattern]
        if not subset:
            continue
        stats.append(
            {
                "pattern": pattern,
                "n": len(subset),
                "mean_final_test_loss": mean(row["final_test_loss"] for row in subset),
                "median_final_test_loss": median(row["final_test_loss"] for row in subset),
                "mean_oracle_best_test_loss": mean(row["oracle_best_test_loss"] for row in subset),
                "mean_oracle_gap": mean(row["oracle_gap"] for row in subset),
                "mean_total_pruned": mean(row["total_pruned"] for row in subset),
                "mean_pruned_0_total": mean(row["pruned_0_total"] for row in subset),
                "mean_pruned_2_total": mean(row["pruned_2_total"] for row in subset),
            }
        )
    return rows, stats


def plot_task_panels(stats_rows: List[Dict], out_dir: Path) -> None:
    metrics = [
        ("mean_selected_loss", "Mean test loss @ best val", "review_selected_loss.png", "log"),
        ("mean_epoch_to_fixed_target", "Mean epoch to fixed-target val loss", "review_speed.png", "linear"),
        ("mean_selected_params", "Mean selected params", "review_selected_params.png", "linear"),
        ("mean_instability_gap", "Mean final-selected loss gap", "review_instability_gap.png", "log"),
    ]
    for key, ylabel, filename, scale in metrics:
        fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 4.6))
        for ax, task in zip(axes, TASKS):
            subset = [row for row in stats_rows if row["task"] == task]
            vals = []
            labels = []
            colors = []
            for mode in MODES:
                value = next(row[key] for row in subset if row["mode"] == mode)
                vals.append(np.nan if value is None else value)
                labels.append(MODE_LABELS[mode])
                colors.append(MODE_COLORS[mode])
            ax.bar(np.arange(len(vals)), vals, color=colors)
            ax.set_xticks(np.arange(len(vals)))
            ax.set_xticklabels(labels, rotation=18, ha="right")
            ax.set_title(task)
            if scale == "log":
                positive = [value for value in vals if value is not None and value > 0]
                if positive:
                    ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def plot_legacy_patterns(stats_rows: List[Dict], out_dir: Path) -> None:
    patterns = [row["pattern"] for row in stats_rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    axes[0].bar(patterns, [row["mean_final_test_loss"] for row in stats_rows], color=["#7f7f7f", "#1f77b4", "#d62728"])
    axes[0].set_yscale("log")
    axes[0].set_title("Legacy split controls: final test loss")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(patterns, [row["mean_oracle_gap"] for row in stats_rows], color=["#7f7f7f", "#1f77b4", "#d62728"])
    axes[1].set_yscale("log")
    axes[1].set_title("Legacy split controls: oracle gap")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[2].bar(patterns, [row["mean_total_pruned"] for row in stats_rows], color=["#7f7f7f", "#1f77b4", "#d62728"])
    axes[2].set_title("Legacy split controls: total pruned")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "review_legacy_split_patterns.png", dpi=160)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    main_stats: List[Dict],
    legacy_stats: List[Dict],
    review_dir: Path,
) -> None:
    def find_row(task: str, mode: str) -> Dict:
        return next(row for row in main_stats if row["task"] == task and row["mode"] == mode)

    lines: List[str] = []
    lines.append("# Experiment Review")
    lines.append("")
    lines.append("## Main protocol")
    lines.append("")
    lines.append("Primary model-selection rule: choose the checkpoint with minimum validation loss under a fixed training budget, then report test loss at that checkpoint.")
    lines.append("Secondary stability rule: also report final test loss and the final-minus-selected gap.")
    lines.append("")
    lines.append("## Main findings")
    lines.append("")
    hard_fixed = find_row("hard", "fixed")
    hard_prune = find_row("hard", "prune_only")
    hard_random = find_row("hard", "prune_grow_random")
    hard_split = find_row("hard", "prune_grow_split")
    lines.append(
        f"- On `hard`, `prune_only` has the best mean selected test loss ({hard_prune['mean_selected_loss']:.6g}) and reduces mean selected parameters to {hard_prune['mean_selected_params']:.1f} from {hard_fixed['mean_selected_params']:.0f}."
    )
    lines.append(
        f"- On `hard`, `prune_grow_split` has a selected-loss mean close to the other methods ({hard_split['mean_selected_loss']:.6g}) but a much larger mean instability gap ({hard_split['mean_instability_gap']:.6g})."
    )
    medium_fixed = find_row("medium", "fixed")
    medium_prune = find_row("medium", "prune_only")
    medium_random = find_row("medium", "prune_grow_random")
    lines.append(
        f"- On `medium`, `fixed` and `grow_random` are slightly better than `prune_only` in selected loss, but `prune_only` still cuts mean selected parameters to {medium_prune['mean_selected_params']:.1f}."
    )
    simple_fixed = find_row("simple", "fixed")
    simple_prune = find_row("simple", "prune_only")
    lines.append(
        f"- On `simple`, `fixed` remains the strongest selected-loss baseline ({simple_fixed['mean_selected_loss']:.6g}), while `prune_only` trades some accuracy for a much smaller mean selected model ({simple_prune['mean_selected_params']:.1f})."
    )
    lines.append("")
    lines.append("## Strategy guidance")
    lines.append("")
    lines.append("- `prune_only` is the strongest candidate for the main paper claim because it already shows a favorable accuracy/parameter tradeoff on the hardest task.")
    lines.append("- `prune_grow_random` is a useful control because it changes structure without reducing parameter count.")
    lines.append("- `prune_grow_split` currently looks more useful as a failure-mode or mechanism-analysis condition than as the lead method.")
    lines.append("")
    lines.append("## Pruning-pattern sensitivity")
    lines.append("")
    if legacy_stats:
        pattern_lines = {
            row["pattern"]: row for row in legacy_stats
        }
        if "p0_only" in pattern_lines and "p0+p2" in pattern_lines:
            p0 = pattern_lines["p0_only"]
            p02 = pattern_lines["p0+p2"]
            lines.append(
                f"- In the legacy hard split sweeps, `p0+p2` runs show much larger mean final loss ({p02['mean_final_test_loss']:.6g}) than `p0_only` runs ({p0['mean_final_test_loss']:.6g})."
            )
            lines.append(
                f"- `p0+p2` also corresponds to much larger total pruning volume ({p02['mean_total_pruned']:.1f} vs {p0['mean_total_pruned']:.1f}), supporting the idea that deeper-layer turnover is linked to instability."
            )
    lines.append("- These pattern analyses are diagnostic only because the legacy sweeps did not record validation loss and therefore cannot be compared with the new protocol as main-table evidence.")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    for filename in [
        "review_selected_loss.png",
        "review_speed.png",
        "review_selected_params.png",
        "review_instability_gap.png",
        "review_legacy_split_patterns.png",
    ]:
        lines.append(f"- `{filename}`")
    path.write_text("\n".join(lines))


def main() -> None:
    root = Path("results/publishable_pilot_20260313")
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    main_stats, speed_rows = summarize_main_results(root / "summary_three_task.csv")
    write_csv(review_dir / "main_task_mode_summary.csv", main_stats)
    write_csv(review_dir / "speed_and_stability_rows.csv", speed_rows)

    legacy_rows, legacy_stats = summarize_legacy_patterns(
        Path("former results/3.12results，128/hard")
    )
    write_csv(review_dir / "legacy_pattern_rows.csv", legacy_rows)
    write_csv(review_dir / "legacy_pattern_summary.csv", legacy_stats)

    plot_task_panels(main_stats, review_dir)
    plot_legacy_patterns(legacy_stats, review_dir)
    write_markdown_report(review_dir / "experiment_review.md", main_stats, legacy_stats, review_dir)
    print(review_dir)


if __name__ == "__main__":
    main()
