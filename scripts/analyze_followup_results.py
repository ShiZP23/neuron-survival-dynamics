import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze targeted follow-up experiments")
    parser.add_argument(
        "--followup-root",
        type=Path,
        default=Path("results/followup_20260313/hard"),
        help="Root directory of targeted follow-up runs",
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("former results/3.12results，128/hard"),
        help="Root directory of legacy 128-width runs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <followup-root>/analysis",
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


def canonical_legacy_run(seed_dir: Path) -> Optional[Path]:
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


def summarize_run(run_dir: Path) -> Dict:
    cfg = load_json(run_dir / "config.json")
    rows = load_metrics(run_dir / "metrics.csv")
    updates = [row for row in rows if int(row["is_update_epoch"]) == 1]
    test = np.array([float(row["test_loss"]) for row in rows], dtype=float)
    best_idx = int(test.argmin())
    return {
        "mode": cfg["mode"],
        "seed": int(cfg["seed"]),
        "run_name": run_dir.name,
        "epsilon": float(cfg["ablation_epsilon_ratio"]),
        "final_test_loss": float(test[-1]),
        "best_test_loss": float(test[best_idx]),
        "best_epoch": best_idx + 1,
        "final_param_count": int(float(rows[-1]["param_count"])),
        "min_param_count": int(min(float(row["param_count"]) for row in rows)),
        "total_pruned": sum(int(row["total_pruned"]) for row in updates),
        "pruned_0_total": sum(int(row["pruned_0"]) for row in updates),
        "pruned_1_total": sum(int(row["pruned_1"]) for row in updates),
        "pruned_2_total": sum(int(row["pruned_2"]) for row in updates),
        "final_sizes": rows[-1]["sizes"],
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_prune_only_vs_fixed(prune_only_rows: List[Dict], fixed_rows: List[Dict], out_dir: Path) -> None:
    by_seed_fixed = {row["seed"]: row for row in fixed_rows}
    seeds = [row["seed"] for row in prune_only_rows]
    x = np.arange(len(seeds))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(x - width / 2, [by_seed_fixed[seed]["final_test_loss"] for seed in seeds], width=width, label="fixed")
    axes[0].bar(x + width / 2, [row["final_test_loss"] for row in prune_only_rows], width=width, label="prune_only")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(seed) for seed in seeds])
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Final test loss")
    axes[0].set_title("Final test loss: prune_only vs fixed")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x - width / 2, [by_seed_fixed[seed]["final_param_count"] for seed in seeds], width=width, label="fixed")
    axes[1].bar(x + width / 2, [row["final_param_count"] for row in prune_only_rows], width=width, label="prune_only")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(seed) for seed in seeds])
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Final parameter count")
    axes[1].set_title("Final parameter count: prune_only vs fixed")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "prune_only_vs_fixed_seedwise.png", dpi=160)
    plt.close(fig)


def plot_prune_only_efficiency(prune_only_rows: List[Dict], fixed_rows: List[Dict], out_dir: Path) -> None:
    by_seed_fixed = {row["seed"]: row for row in fixed_rows}
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for row in prune_only_rows:
        seed = row["seed"]
        fixed = by_seed_fixed[seed]
        ax.scatter(
            fixed["final_param_count"],
            fixed["final_test_loss"],
            color="#1f77b4",
            s=56,
            marker="o",
        )
        ax.scatter(
            row["final_param_count"],
            row["final_test_loss"],
            color="#ff7f0e",
            s=64,
            marker="s",
        )
        ax.annotate(f"seed {seed}", (row["final_param_count"], row["final_test_loss"]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Final parameter count")
    ax.set_ylabel("Final test loss")
    ax.set_title("Parameter-efficiency follow-up: fixed vs prune_only")
    ax.grid(True, linestyle="--", alpha=0.3)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="fixed"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=8, label="prune_only"),
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(out_dir / "prune_only_param_efficiency.png", dpi=160)
    plt.close(fig)


def plot_split_epsilon(split_rows: List[Dict], out_dir: Path) -> None:
    rows = sorted(split_rows, key=lambda row: row["epsilon"], reverse=True)
    eps = [row["epsilon"] for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(x - width / 2, [row["best_test_loss"] for row in rows], width=width, label="best")
    axes[0].bar(x + width / 2, [row["final_test_loss"] for row in rows], width=width, label="final")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{epsilon:g}" for epsilon in eps])
    axes[0].set_xlabel("ablation_epsilon_ratio")
    axes[0].set_ylabel("Test loss")
    axes[0].set_title("Problematic split seed: best vs final loss")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x, [row["total_pruned"] for row in rows], color="#d62728", width=0.6, label="total_pruned")
    axes[1].plot(x, [row["pruned_2_total"] for row in rows], color="#1f77b4", marker="o", linewidth=1.8, label="pruned_2_total")
    axes[1].plot(x, [row["pruned_0_total"] for row in rows], color="#2ca02c", marker="o", linewidth=1.8, label="pruned_0_total")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{epsilon:g}" for epsilon in eps])
    axes[1].set_xlabel("ablation_epsilon_ratio")
    axes[1].set_ylabel("Pruned neurons across training")
    axes[1].set_title("Problematic split seed: turnover by epsilon")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "split_epsilon_sensitivity.png", dpi=160)
    plt.close(fig)


def write_notes(prune_only_rows: List[Dict], fixed_rows: List[Dict], split_rows: List[Dict], out_dir: Path) -> None:
    by_seed_fixed = {row["seed"]: row for row in fixed_rows}
    lines = ["Targeted follow-up analysis", ""]
    lines.append("prune_only vs fixed")
    for row in prune_only_rows:
        fixed = by_seed_fixed[row["seed"]]
        delta_params = row["final_param_count"] - fixed["final_param_count"]
        delta_loss = row["final_test_loss"] - fixed["final_test_loss"]
        lines.append(
            f"- seed {row['seed']}: prune_only final params {row['final_param_count']} vs fixed {fixed['final_param_count']}, "
            f"final test loss delta {delta_loss:+.6g}, total_pruned={row['total_pruned']}"
        )
    lines.append("")
    lines.append("split epsilon sweep")
    for row in sorted(split_rows, key=lambda item: item["epsilon"], reverse=True):
        lines.append(
            f"- epsilon={row['epsilon']}: final={row['final_test_loss']:.6g}, best={row['best_test_loss']:.6g}, "
            f"total_pruned={row['total_pruned']}, p0={row['pruned_0_total']}, p2={row['pruned_2_total']}"
        )
    (out_dir / "followup_notes.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.followup_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    prune_only_rows = [
        summarize_run(run_dir)
        for run_dir in sorted((args.followup_root / "prune_only").glob("seed_*/*"))
        if (run_dir / "config.json").exists()
    ]
    split_rows = [
        summarize_run(run_dir)
        for run_dir in sorted((args.followup_root / "prune_grow_split").glob("seed_*/*"))
        if (run_dir / "config.json").exists()
    ]
    fixed_rows: List[Dict] = []
    for seed_dir in sorted((args.legacy_root / "fixed").glob("seed_*")):
        run_dir = canonical_legacy_run(seed_dir)
        if run_dir is not None:
            fixed_rows.append(summarize_run(run_dir))

    write_csv(out_dir / "followup_prune_only_summary.csv", prune_only_rows)
    write_csv(out_dir / "followup_split_epsilon_summary.csv", split_rows)
    plot_prune_only_vs_fixed(prune_only_rows, fixed_rows, out_dir)
    plot_prune_only_efficiency(prune_only_rows, fixed_rows, out_dir)
    plot_split_epsilon(split_rows, out_dir)
    write_notes(prune_only_rows, fixed_rows, split_rows, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
