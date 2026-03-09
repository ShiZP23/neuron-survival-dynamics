import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from neuron_kill.data import TASKS


def _parse_run_path(results_dir: str, metrics_path: str) -> Optional[Tuple[str, str, int, str]]:
    rel = os.path.relpath(metrics_path, results_dir)
    parts = rel.split(os.sep)
    if len(parts) < 5:
        return None
    task, mode, seed_part, timestamp = parts[0], parts[1], parts[2], parts[3]
    if not seed_part.startswith("seed_"):
        return None
    try:
        seed = int(seed_part.replace("seed_", ""))
    except ValueError:
        return None
    return task, mode, seed, timestamp


def _latest_runs(results_dir: str) -> List[Dict]:
    candidates: Dict[Tuple[str, str, int], Dict] = {}
    for root, _, files in os.walk(results_dir):
        if "metrics.csv" not in files:
            continue
        metrics_path = os.path.join(root, "metrics.csv")
        parsed = _parse_run_path(results_dir, metrics_path)
        if parsed is None:
            continue
        task, mode, seed, timestamp = parsed
        key = (task, mode, seed)
        existing = candidates.get(key)
        if existing is None or timestamp > existing["timestamp"]:
            candidates[key] = {
                "task": task,
                "mode": mode,
                "seed": seed,
                "timestamp": timestamp,
                "metrics_path": metrics_path,
            }
    return list(candidates.values())


def _read_final_metrics(metrics_path: str) -> Optional[Dict]:
    last_row = None
    with open(metrics_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    if last_row is None:
        return None
    return {
        "epoch": int(last_row["epoch"]),
        "test_loss": float(last_row["test_loss"]),
        "param_count": int(float(last_row["param_count"])),
    }


def summarize_results(results_dir: str) -> List[Dict]:
    runs = _latest_runs(results_dir)
    records: List[Dict] = []
    for run in runs:
        metrics = _read_final_metrics(run["metrics_path"])
        if metrics is None:
            continue
        records.append(
            {
                "task": run["task"],
                "mode": run["mode"],
                "seed": run["seed"],
                "timestamp": run["timestamp"],
                "final_epoch": metrics["epoch"],
                "final_test_loss": metrics["test_loss"],
                "final_param_count": metrics["param_count"],
            }
        )
    return records


def write_summary_csv(records: List[Dict], path: str) -> None:
    if not records:
        return
    fieldnames = [
        "task",
        "mode",
        "seed",
        "timestamp",
        "final_epoch",
        "final_test_loss",
        "final_param_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def plot_summary(records: List[Dict], path: str) -> None:
    if len(records) < 2:
        return

    modes = sorted({row["mode"] for row in records})
    mode_colors = {
        "fixed": "#1f77b4",
        "prune_only": "#ff7f0e",
        "prune_grow_random": "#2ca02c",
        "prune_grow_split": "#d62728",
    }
    task_markers = {
        "simple": "o",
        "medium": "s",
        "hard": "^",
    }

    plt.figure(figsize=(7, 5))
    for mode in modes:
        subset = [row for row in records if row["mode"] == mode]
        if not subset:
            continue
        for task in TASKS:
            task_subset = [row for row in subset if row["task"] == task]
            if not task_subset:
                continue
            x_vals = [row["final_param_count"] for row in task_subset]
            y_vals = [row["final_test_loss"] for row in task_subset]
            plt.scatter(
                x_vals,
                y_vals,
                label=f"{mode}:{task}",
                color=mode_colors.get(mode, "black"),
                marker=task_markers.get(task, "o"),
                alpha=0.8,
            )

    plt.xlabel("final trainable params")
    plt.ylabel("final test loss")
    plt.title("Final params vs final test loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate neuron_kill results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-plot", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = summarize_results(args.results_dir)
    if not records:
        return
    out_csv = args.out_csv or os.path.join(args.results_dir, "summary.csv")
    out_plot = args.out_plot or os.path.join(args.results_dir, "summary.png")
    write_summary_csv(records, out_csv)
    plot_summary(records, out_plot)


if __name__ == "__main__":
    main()
