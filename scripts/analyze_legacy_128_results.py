import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


STRATEGIES = ("fixed", "prune_grow_random", "prune_grow_split")
SPEED_THRESHOLDS = (1e-2, 5e-3, 1e-3, 5e-4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze legacy 128-width hard-task experiment results"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("former results/3.12results，128/hard"),
        help="Root directory containing hard/{strategy}/seed_* runs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <root>/analysis",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def derived_seed_config(seed: int) -> Dict[str, int]:
    return {
        "data_seed": seed + 11,
        "model_seed": seed + 23,
        "shuffle_seed": seed + 37,
        "structure_seed": seed + 53,
    }


def is_timestamp_run(path: Path) -> bool:
    return len(path.name) == 15 and path.name[:8].isdigit() and path.name[8] == "_"


def canonical_run(seed_dir: Path) -> Optional[Path]:
    try:
        seed = int(seed_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None

    expected = derived_seed_config(seed)
    matches: List[Path] = []
    for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            continue
        cfg = load_json(config_path)
        if all(cfg.get(key) == value for key, value in expected.items()):
            matches.append(run_dir)

    if not matches:
        return None

    named_seed = seed_dir / f"seed{seed}"
    if named_seed in matches:
        return named_seed

    timestamp_matches = [path for path in matches if is_timestamp_run(path)]
    if timestamp_matches:
        return sorted(timestamp_matches)[0]
    return sorted(matches)[0]


def speed_epoch(rows: Sequence[Dict[str, str]], threshold: float) -> Optional[int]:
    for row in rows:
        if float(row["test_loss"]) <= threshold:
            return int(row["epoch"])
    return None


def first_update_epoch(rows: Sequence[Dict[str, str]]) -> Optional[int]:
    for row in rows:
        if int(row["is_update_epoch"]) == 1 and int(row["total_pruned"]) > 0:
            return int(row["epoch"])
    return None


def prune_pattern(rows: Sequence[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
    updates = [row for row in rows if int(row["is_update_epoch"]) == 1]
    totals = {
        "pruned_0": sum(int(row["pruned_0"]) for row in updates),
        "pruned_1": sum(int(row["pruned_1"]) for row in updates),
        "pruned_2": sum(int(row["pruned_2"]) for row in updates),
    }
    if totals["pruned_0"] > 0 and totals["pruned_2"] > 0 and totals["pruned_1"] == 0:
        return "p0+p2", totals
    if totals["pruned_0"] > 0 and totals["pruned_1"] == 0 and totals["pruned_2"] == 0:
        return "p0_only", totals
    if all(value == 0 for value in totals.values()):
        return "none", totals
    return "other", totals


def summarize_run(strategy: str, seed_name: str, run_dir: Path) -> Dict:
    cfg = load_json(run_dir / "config.json")
    rows = load_metrics(run_dir / "metrics.csv")
    test_losses = np.array([float(row["test_loss"]) for row in rows], dtype=float)
    train_losses = np.array([float(row["train_loss"]) for row in rows], dtype=float)
    params = np.array([int(float(row["param_count"])) for row in rows], dtype=int)
    best_idx = int(test_losses.argmin())
    pattern, totals = prune_pattern(rows)
    updates = [row for row in rows if int(row["is_update_epoch"]) == 1]

    summary = {
        "strategy": strategy,
        "seed_name": seed_name,
        "run_name": run_dir.name,
        "seed": int(cfg["seed"]),
        "data_seed": int(cfg["data_seed"]),
        "model_seed": int(cfg["model_seed"]),
        "shuffle_seed": int(cfg["shuffle_seed"]),
        "structure_seed": int(cfg["structure_seed"]),
        "epochs": int(cfg["epochs"]),
        "final_test_loss": float(test_losses[-1]),
        "final_train_loss": float(train_losses[-1]),
        "best_test_loss": float(test_losses[best_idx]),
        "best_epoch": best_idx + 1,
        "final_param_count": int(params[-1]),
        "min_param_count": int(params.min()),
        "mean_test_loss": float(test_losses.mean()),
        "median_test_loss": float(np.median(test_losses)),
        "final_minus_best": float(test_losses[-1] - test_losses[best_idx]),
        "generalization_gap_final": float(train_losses[-1] - test_losses[-1]),
        "first_prune_epoch": first_update_epoch(rows),
        "pattern": pattern,
        "total_pruned": sum(int(row["total_pruned"]) for row in updates),
        "total_grown": sum(int(row["total_grown"]) for row in updates),
        "pruned_0_total": totals["pruned_0"],
        "pruned_1_total": totals["pruned_1"],
        "pruned_2_total": totals["pruned_2"],
    }
    for threshold in SPEED_THRESHOLDS:
        summary[f"reach_{threshold}"] = speed_epoch(rows, threshold)
    return summary


def strategy_runs(root: Path) -> List[Dict]:
    records: List[Dict] = []
    for strategy in STRATEGIES:
        strategy_dir = root / strategy
        for seed_dir in sorted(strategy_dir.glob("seed_*")):
            run_dir = canonical_run(seed_dir)
            if run_dir is None:
                continue
            records.append(summarize_run(strategy, seed_dir.name, run_dir))
    return records


def seed0_sweeps(root: Path) -> List[Dict]:
    split_seed0 = root / "prune_grow_split" / "seed_0"
    records: List[Dict] = []
    for run_dir in sorted(p for p in split_seed0.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            continue
        record = summarize_run("prune_grow_split", "seed_0", run_dir)
        if run_dir.name.startswith("data"):
            record["sweep_type"] = "data_seed_sweep"
        elif run_dir.name == "seed0":
            record["sweep_type"] = "canonical_seed0"
        elif is_timestamp_run(run_dir):
            record["sweep_type"] = "model_seed_sweep"
        else:
            record["sweep_type"] = "other"
        records.append(record)
    return records


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_strategy_distributions(records: Sequence[Dict], out_dir: Path) -> None:
    metrics = [
        ("final_test_loss", "Final test loss", "strategy_final_test_loss_box.png", True),
        ("best_test_loss", "Best test loss", "strategy_best_test_loss_box.png", True),
        ("final_minus_best", "Final minus best test loss", "strategy_instability_box.png", True),
        ("mean_test_loss", "Mean test loss over training", "strategy_mean_test_loss_box.png", True),
        ("final_param_count", "Final parameter count", "strategy_final_params_box.png", False),
    ]
    for metric, ylabel, filename, log_scale in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        data = [
            [record[metric] for record in records if record["strategy"] == strategy]
            for strategy in STRATEGIES
        ]
        ax.boxplot(data, tick_labels=STRATEGIES, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by strategy")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def plot_strategy_speed(records: Sequence[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(SPEED_THRESHOLDS))
    width = 0.22
    for idx, strategy in enumerate(STRATEGIES):
        means = []
        for threshold in SPEED_THRESHOLDS:
            values = [
                record[f"reach_{threshold}"]
                for record in records
                if record["strategy"] == strategy and record[f"reach_{threshold}"] is not None
            ]
            means.append(mean(values) if values else np.nan)
        ax.bar(x + (idx - 1) * width, means, width=width, label=strategy)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{threshold:g}" for threshold in SPEED_THRESHOLDS])
    ax.set_xlabel("Test-loss threshold")
    ax.set_ylabel("Epoch to reach threshold")
    ax.set_title("Learning speed by strategy")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_speed_thresholds.png", dpi=160)
    plt.close(fig)


def plot_strategy_scatter(records: Sequence[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    colors = {
        "fixed": "#1f77b4",
        "prune_grow_random": "#2ca02c",
        "prune_grow_split": "#d62728",
    }
    for strategy in STRATEGIES:
        subset = [record for record in records if record["strategy"] == strategy]
        ax.scatter(
            [record["best_test_loss"] for record in subset],
            [record["final_test_loss"] for record in subset],
            s=46,
            alpha=0.85,
            color=colors[strategy],
            label=strategy,
        )
    low = min(
        min(record["best_test_loss"] for record in records),
        min(record["final_test_loss"] for record in records),
    )
    high = max(
        max(record["best_test_loss"] for record in records),
        max(record["final_test_loss"] for record in records),
    )
    ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Best test loss")
    ax.set_ylabel("Final test loss")
    ax.set_title("Best-vs-final test loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_best_vs_final_scatter.png", dpi=160)
    plt.close(fig)


def plot_strategy_trajectories(root: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = {
        "fixed": "#1f77b4",
        "prune_grow_random": "#2ca02c",
        "prune_grow_split": "#d62728",
    }
    for strategy in STRATEGIES:
        curves: List[np.ndarray] = []
        epochs: Optional[np.ndarray] = None
        for seed_dir in sorted((root / strategy).glob("seed_*")):
            run_dir = canonical_run(seed_dir)
            if run_dir is None:
                continue
            rows = load_metrics(run_dir / "metrics.csv")
            losses = np.array([float(row["test_loss"]) for row in rows], dtype=float)
            curves.append(losses)
            if epochs is None:
                epochs = np.array([int(row["epoch"]) for row in rows], dtype=int)
        if not curves or epochs is None:
            continue
        curve_array = np.stack(curves, axis=0)
        q25 = np.quantile(curve_array, 0.25, axis=0)
        q50 = np.quantile(curve_array, 0.50, axis=0)
        q75 = np.quantile(curve_array, 0.75, axis=0)
        ax.plot(epochs, q50, color=colors[strategy], linewidth=2.0, label=strategy)
        ax.fill_between(epochs, q25, q75, color=colors[strategy], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test loss")
    ax.set_title("Strategy comparison: median test-loss trajectory with IQR")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_test_loss_trajectory_iqr.png", dpi=160)
    plt.close(fig)


def plot_pattern_distributions(records: Sequence[Dict], out_dir: Path, sweep_type: str) -> None:
    subset = [
        record
        for record in records
        if record["sweep_type"] == sweep_type and record["pattern"] in {"p0_only", "p0+p2", "none"}
    ]
    if not subset:
        return

    pattern_order = ["none", "p0_only", "p0+p2"]
    metrics = [
        ("final_test_loss", "Final test loss", f"{sweep_type}_pattern_final_test_loss.png"),
        ("best_test_loss", "Best test loss", f"{sweep_type}_pattern_best_test_loss.png"),
        ("final_minus_best", "Final minus best test loss", f"{sweep_type}_pattern_instability.png"),
        ("total_pruned", "Total pruned neurons", f"{sweep_type}_pattern_total_pruned.png"),
    ]
    for metric, ylabel, filename in metrics:
        data = [
            [record[metric] for record in subset if record["pattern"] == pattern]
            for pattern in pattern_order
        ]
        if not any(group for group in data):
            continue
        labels = [pattern for pattern, group in zip(pattern_order, data) if group]
        values = [group for group in data if group]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        if metric != "total_pruned":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by prune pattern ({sweep_type})")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def plot_pattern_seed_scatter(records: Sequence[Dict], out_dir: Path, sweep_type: str) -> None:
    subset = [
        record
        for record in records
        if record["sweep_type"] == sweep_type and record["pattern"] in {"p0_only", "p0+p2", "none"}
    ]
    if not subset:
        return

    color_map = {"none": "#7f7f7f", "p0_only": "#1f77b4", "p0+p2": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for pattern, color in color_map.items():
        group = [record for record in subset if record["pattern"] == pattern]
        if not group:
            continue
        if sweep_type == "model_seed_sweep":
            xvals = [record["model_seed"] for record in group]
            xlabel = "model_seed"
        else:
            xvals = [record["data_seed"] for record in group]
            xlabel = "data_seed"
        ax.scatter(xvals, [record["final_test_loss"] for record in group], s=54, color=color, label=pattern)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final test loss")
    ax.set_title(f"Seed sensitivity and prune pattern ({sweep_type})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sweep_type}_pattern_seed_scatter.png", dpi=160)
    plt.close(fig)


def write_text_report(strategy_records: Sequence[Dict], sweep_records: Sequence[Dict], out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("Legacy 128-width hard-task analysis")
    lines.append("")
    lines.append("Strategy summary")
    for strategy in STRATEGIES:
        subset = [record for record in strategy_records if record["strategy"] == strategy]
        lines.append(
            (
                f"- {strategy}: n={len(subset)}, "
                f"median final={median(record['final_test_loss'] for record in subset):.6g}, "
                f"median best={median(record['best_test_loss'] for record in subset):.6g}, "
                f"median instability={median(record['final_minus_best'] for record in subset):.6g}, "
                f"final params={subset[0]['final_param_count'] if subset else 'NA'}"
            )
        )
    lines.append("")
    lines.append("Split sweep pattern summary")
    for sweep_type in ("model_seed_sweep", "data_seed_sweep", "canonical_seed0"):
        subset = [record for record in sweep_records if record["sweep_type"] == sweep_type]
        if not subset:
            continue
        lines.append(f"- {sweep_type}:")
        pattern_counts: Dict[str, int] = {}
        for record in subset:
            pattern_counts[record["pattern"]] = pattern_counts.get(record["pattern"], 0) + 1
        for pattern in sorted(pattern_counts):
            lines.append(f"  {pattern}: {pattern_counts[pattern]}")
    lines.append("")
    lines.append("Interpretation notes")
    lines.append("- fixed is the stability baseline because its architecture never changes.")
    lines.append("- random and split keep the same final width in this legacy setup, so final_param_count is not a discriminative metric.")
    lines.append("- split should be judged with both best and final loss because several runs degrade late in training.")
    (out_dir / "analysis_notes.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_records = strategy_runs(root)
    sweep_records = seed0_sweeps(root)

    write_csv(out_dir / "strategy_summary.csv", strategy_records)
    write_csv(out_dir / "split_seed0_sweeps.csv", sweep_records)

    plot_strategy_distributions(strategy_records, out_dir)
    plot_strategy_speed(strategy_records, out_dir)
    plot_strategy_scatter(strategy_records, out_dir)
    plot_strategy_trajectories(root, out_dir)
    plot_pattern_distributions(sweep_records, out_dir, "model_seed_sweep")
    plot_pattern_distributions(sweep_records, out_dir, "data_seed_sweep")
    plot_pattern_seed_scatter(sweep_records, out_dir, "model_seed_sweep")
    plot_pattern_seed_scatter(sweep_records, out_dir, "data_seed_sweep")
    write_text_report(strategy_records, sweep_records, out_dir)

    print(out_dir)


if __name__ == "__main__":
    main()
