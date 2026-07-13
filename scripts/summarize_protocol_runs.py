import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize runs with validation-selected checkpoints")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_run(run_dir: Path) -> Optional[Dict]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = load_metrics(metrics_path)
    if not rows:
        return None
    has_val = "val_loss" in rows[0]
    final = rows[-1]
    if has_val:
        best = min(rows, key=lambda row: float(row["val_loss"]))
    else:
        best = final
    return {
        "task": run_dir.parents[2].name,
        "mode": run_dir.parents[1].name,
        "seed": run_dir.parents[0].name.replace("seed_", ""),
        "run_name": run_dir.name,
        "final_epoch": int(final["epoch"]),
        "best_epoch": int(best["epoch"]),
        "final_val_loss": float(final["val_loss"]) if has_val else None,
        "best_val_loss": float(best["val_loss"]) if has_val else None,
        "final_test_loss": float(final["test_loss"]),
        "test_loss_at_best_val": float(best["test_loss"]),
        "final_param_count": int(float(final["param_count"])),
        "best_param_count": int(float(best["param_count"])),
        "selection_protocol": "best_val" if has_val else "legacy_final",
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_selected_vs_final(rows: List[Dict], out_dir: Path) -> None:
    modes = sorted({row["mode"] for row in rows})
    colors = {
        "fixed": "#1f77b4",
        "prune_only": "#ff7f0e",
        "prune_grow_random": "#2ca02c",
        "prune_grow_split": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        ax.scatter(
            [row["test_loss_at_best_val"] for row in subset],
            [row["final_test_loss"] for row in subset],
            color=colors.get(mode, "black"),
            s=55,
            alpha=0.85,
            label=mode,
        )
    low = min(min(row["test_loss_at_best_val"] for row in rows), min(row["final_test_loss"] for row in rows))
    high = max(max(row["test_loss_at_best_val"] for row in rows), max(row["final_test_loss"] for row in rows))
    ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Test loss at best val checkpoint")
    ax.set_ylabel("Final test loss")
    ax.set_title("Selected-vs-final test loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "selected_vs_final_test_loss.png", dpi=160)
    plt.close(fig)


def plot_mode_bars(rows: List[Dict], out_dir: Path) -> None:
    modes = sorted({row["mode"] for row in rows})
    med_selected = [np.median([row["test_loss_at_best_val"] for row in rows if row["mode"] == mode]) for mode in modes]
    med_final = [np.median([row["final_test_loss"] for row in rows if row["mode"] == mode]) for mode in modes]
    x = np.arange(len(modes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, med_selected, width=width, label="test@best_val")
    ax.bar(x + width / 2, med_final, width=width, label="final_test")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Median test loss")
    ax.set_title("Mode comparison under selection protocol")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "mode_selected_vs_final_bars.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for metrics_path in sorted(args.results_dir.glob("*/*/*/metrics.csv")):
        summary = summarize_run(metrics_path.parent)
        if summary is not None:
            rows.append(summary)
    write_csv(out_dir / "protocol_summary.csv", rows)
    if rows:
        plot_selected_vs_final(rows, out_dir)
        plot_mode_bars(rows, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
