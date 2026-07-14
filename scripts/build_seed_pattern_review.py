import csv
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("former results/3.12results，128/hard/prune_grow_split")
OUT_DIR = Path("results/publishable_pilot_20260313/seed_pattern_review")
PATTERNS = ["none", "p0_only", "p0+p2"]
PATTERN_COLORS = {
    "none": "#4c78a8",
    "p0_only": "#f58518",
    "p0+p2": "#e45756",
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


def classify_pattern(pruned_0: int, pruned_1: int, pruned_2: int) -> str:
    if pruned_0 > 0 and pruned_2 > 0:
        return "p0+p2"
    if pruned_0 > 0 and pruned_2 == 0:
        return "p0_only"
    if pruned_0 == 0 and pruned_1 == 0 and pruned_2 == 0:
        return "none"
    return "other"


def collect_rows() -> List[Dict]:
    rows: List[Dict] = []
    for metrics_path in sorted(ROOT.glob("seed_*/*/metrics.csv")):
        metrics = read_csv(metrics_path)
        if not metrics:
            continue
        updates = [row for row in metrics if int(row["is_update_epoch"]) == 1]
        pruned_0_total = sum(int(row["pruned_0"]) for row in updates)
        pruned_1_total = sum(int(row["pruned_1"]) for row in updates)
        pruned_2_total = sum(int(row["pruned_2"]) for row in updates)
        pattern = classify_pattern(pruned_0_total, pruned_1_total, pruned_2_total)
        if pattern not in PATTERNS:
            continue

        test_losses = [float(row["test_loss"]) for row in metrics]
        best_idx = min(range(len(metrics)), key=lambda i: test_losses[i])
        best = metrics[best_idx]
        final = metrics[-1]
        rows.append(
            {
                "run_path": str(metrics_path.parent),
                "seed_dir": metrics_path.parent.parent.name,
                "run_name": metrics_path.parent.name,
                "pattern": pattern,
                "best_epoch": int(best["epoch"]),
                "best_test_loss": float(best["test_loss"]),
                "final_test_loss": float(final["test_loss"]),
                "final_minus_best": float(final["test_loss"]) - float(best["test_loss"]),
                "mean_test_loss": mean(test_losses),
                "median_test_loss": median(test_losses),
                "reach_1e_3": first_reach(metrics, 1e-3),
                "reach_5e_4": first_reach(metrics, 5e-4),
                "reach_2e_4": first_reach(metrics, 2e-4),
                "total_pruned": sum(int(row["total_pruned"]) for row in updates),
                "total_grown": sum(int(row["total_grown"]) for row in updates),
                "pruned_0_total": pruned_0_total,
                "pruned_1_total": pruned_1_total,
                "pruned_2_total": pruned_2_total,
                "active_0_best": int(best.get("active_0", 0)),
                "active_1_best": int(best.get("active_1", 0)),
                "active_2_best": int(best.get("active_2", 0)),
            }
        )
    return rows


def first_reach(metrics: List[Dict[str, str]], threshold: float):
    for row in metrics:
        if float(row["test_loss"]) <= threshold:
            return int(row["epoch"])
    return None


def summarize(rows: List[Dict]) -> List[Dict]:
    summary: List[Dict] = []
    for pattern in PATTERNS:
        group = [row for row in rows if row["pattern"] == pattern]
        if not group:
            continue
        summary.append(
            {
                "pattern": pattern,
                "n": len(group),
                "mean_final_test_loss": mean(row["final_test_loss"] for row in group),
                "median_final_test_loss": median(row["final_test_loss"] for row in group),
                "mean_best_test_loss": mean(row["best_test_loss"] for row in group),
                "median_best_test_loss": median(row["best_test_loss"] for row in group),
                "mean_best_epoch": mean(row["best_epoch"] for row in group),
                "mean_final_minus_best": mean(row["final_minus_best"] for row in group),
                "median_final_minus_best": median(row["final_minus_best"] for row in group),
                "mean_total_pruned": mean(row["total_pruned"] for row in group),
                "mean_pruned_0_total": mean(row["pruned_0_total"] for row in group),
                "mean_pruned_1_total": mean(row["pruned_1_total"] for row in group),
                "mean_pruned_2_total": mean(row["pruned_2_total"] for row in group),
                "mean_active_0_best": mean(row["active_0_best"] for row in group),
                "mean_active_1_best": mean(row["active_1_best"] for row in group),
                "mean_active_2_best": mean(row["active_2_best"] for row in group),
            }
        )
    return summary


def pattern_dirs() -> Dict[str, Path]:
    dirs = {
        "primary": OUT_DIR / "primary",
        "supplementary": OUT_DIR / "supplementary",
        "archive": OUT_DIR / "archive",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def plot_distribution_with_points(
    rows: List[Dict], key: str, title: str, filename: str, out_dir: Path, log_scale: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    vals = []
    colors = []
    labels = []
    for pattern in PATTERNS:
        group = [row[key] for row in rows if row["pattern"] == pattern]
        if not group:
            continue
        vals.append(group)
        colors.append(PATTERN_COLORS[pattern])
        labels.append(pattern)
    bp = ax.boxplot(vals, patch_artist=True, tick_labels=labels, showfliers=False, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for idx, (pattern, group) in enumerate(
        [(pattern, [row[key] for row in rows if row["pattern"] == pattern]) for pattern in PATTERNS if any(row["pattern"] == pattern for row in rows)],
        start=1,
    ):
        jitter = np.linspace(-0.10, 0.10, len(group)) if len(group) > 1 else np.array([0.0])
        plot_vals = [max(float(v), 1e-12) if log_scale else float(v) for v in group]
        ax.scatter(
            np.full(len(group), idx, dtype=float) + jitter,
            plot_vals,
            color=PATTERN_COLORS[pattern],
            edgecolor="white",
            linewidth=0.5,
            s=38,
            alpha=0.95,
            zorder=3,
        )
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def plot_pattern_trajectory_iqr(rows: List[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for pattern in PATTERNS:
        group = [row for row in rows if row["pattern"] == pattern]
        if not group:
            continue
        curves = []
        epochs = None
        for row in group:
            metrics = read_csv(Path(row["run_path"]) / "metrics.csv")
            test = np.array([max(float(m["test_loss"]), 1e-12) for m in metrics], dtype=float)
            curves.append(test)
            if epochs is None:
                epochs = np.array([int(m["epoch"]) for m in metrics], dtype=int)
            ax.plot(epochs, test, color=PATTERN_COLORS[pattern], alpha=0.08, linewidth=0.8)
        curve_array = np.stack(curves, axis=0)
        q25 = np.quantile(curve_array, 0.25, axis=0)
        q50 = np.quantile(curve_array, 0.50, axis=0)
        q75 = np.quantile(curve_array, 0.75, axis=0)
        ax.plot(epochs, q50, color=PATTERN_COLORS[pattern], linewidth=2.2, label=pattern)
        ax.fill_between(epochs, q25, q75, color=PATTERN_COLORS[pattern], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test loss")
    ax.set_title("pattern-stratified test-loss trajectories")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pattern_loss_trajectory_iqr.png", dpi=180)
    plt.close(fig)


def plot_best_vs_gap(rows: List[Dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for pattern in PATTERNS:
        group = [row for row in rows if row["pattern"] == pattern]
        if not group:
            continue
        ax.scatter(
            [max(row["best_test_loss"], 1e-12) for row in group],
            [max(row["final_minus_best"], 1e-12) for row in group],
            color=PATTERN_COLORS[pattern],
            label=pattern,
            s=52,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("oracle best test loss")
    ax.set_ylabel("final minus best")
    ax.set_title("peak performance vs instability")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pattern_best_vs_gap.png", dpi=180)
    plt.close(fig)


def plot_pruned2_tradeoff(rows: List[Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for pattern in PATTERNS:
        group = [row for row in rows if row["pattern"] == pattern]
        if not group:
            continue
        axes[0].scatter(
            [row["pruned_2_total"] for row in group],
            [max(row["best_test_loss"], 1e-12) for row in group],
            color=PATTERN_COLORS[pattern],
            label=pattern,
            s=50,
            alpha=0.9,
        )
        axes[1].scatter(
            [row["pruned_2_total"] for row in group],
            [max(row["final_minus_best"], 1e-12) for row in group],
            color=PATTERN_COLORS[pattern],
            label=pattern,
            s=50,
            alpha=0.9,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cumulative pruned_2")
    axes[0].set_ylabel("oracle best test loss")
    axes[0].set_title("layer-2 pruning vs peak performance")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cumulative pruned_2")
    axes[1].set_ylabel("final minus best")
    axes[1].set_title("layer-2 pruning vs instability")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pattern_pruned2_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_exemplar_curves(rows: List[Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, pattern in zip(axes, PATTERNS):
        group = [row for row in rows if row["pattern"] == pattern]
        if not group:
            ax.set_visible(False)
            continue
        exemplar = min(group, key=lambda row: row["best_test_loss"])
        metrics = read_csv(Path(exemplar["run_path"]) / "metrics.csv")
        epochs = np.array([int(row["epoch"]) for row in metrics], dtype=int)
        test = np.array([max(float(row["test_loss"]), 1e-12) for row in metrics], dtype=float)
        train = np.array([max(float(row["train_loss"]), 1e-12) for row in metrics], dtype=float)
        ax.plot(epochs, train, color="#888888", linewidth=1.2, alpha=0.9, label="train")
        ax.plot(epochs, test, color=PATTERN_COLORS[pattern], linewidth=1.8, label="test")
        ax.set_yscale("log")
        ax.set_title(f"{pattern}\n{Path(exemplar['run_path']).name}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("loss")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "pattern_exemplar_loss_curves.png", dpi=180)
    plt.close(fig)


def write_report(summary_rows: List[Dict]) -> None:
    lines = [
        "# Seed Pattern Review",
        "",
        "Data source: `former results/3.12results，128/hard/prune_grow_split`.",
        "",
        "Figure organization:",
        "- `primary/`: figures suitable for main-text comparison",
        "- `supplementary/`: diagnostic and case-study figures",
        "- `archive/`: older low-information figures kept only for traceability",
        "",
        "Pattern definition:",
        "- `none`: no pruning in layers 0, 1, or 2",
        "- `p0_only`: pruning occurs in layer 0 only",
        "- `p0+p2`: pruning occurs in both layers 0 and 2",
        "",
        "Caveat:",
        "- `none` currently has only one run. Treat it as a case study, not a statistical group.",
        "- `best_test_loss` here is an oracle best test loss because these legacy sweeps do not store validation loss. Use it as a diagnostic primary metric for this sweep, not as a final paper protocol.",
        "",
        "Presentation rationale:",
        "- Main comparisons emphasize distribution plots with run-level points, training trajectories with median/IQR, and budget-performance tradeoff scatter plots.",
        "- This follows the general style used in pruning and dynamic sparsity papers, which usually compare performance under a fixed parameter or compute budget rather than relying on summary bars alone.",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['pattern']}",
                f"- n = {row['n']}",
                f"- mean oracle best test loss = {row['mean_best_test_loss']:.6g}",
                f"- mean final test loss = {row['mean_final_test_loss']:.6g}",
                f"- mean final minus best = {row['mean_final_minus_best']:.6g}",
                f"- mean best epoch = {row['mean_best_epoch']:.1f}",
                f"- mean total pruned = {row['mean_total_pruned']:.1f}",
                f"- mean pruned_0_total = {row['mean_pruned_0_total']:.1f}",
                f"- mean pruned_1_total = {row['mean_pruned_1_total']:.1f}",
                f"- mean pruned_2_total = {row['mean_pruned_2_total']:.1f}",
                f"- mean active neurons at oracle best: ({row['mean_active_0_best']:.1f}, {row['mean_active_1_best']:.1f}, {row['mean_active_2_best']:.1f})",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `p0+p2` and `p0_only` achieve similarly low oracle best test loss, so the main difference is not peak capability.",
            "- The main difference is stability: `p0+p2` has much worse final loss and much larger `final_minus_best`.",
            "- The deeper-layer pruning in `p0+p2` is the clearest correlational marker of unstable runs in this sweep.",
            "- `none` looks stable in the single available example, but the sample count is too small for generalization.",
        ]
    )
    (OUT_DIR / "seed_pattern_review.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = pattern_dirs()
    rows = collect_rows()
    summary_rows = summarize(rows)
    write_csv(OUT_DIR / "pattern_rows.csv", rows)
    write_csv(OUT_DIR / "pattern_summary.csv", summary_rows)

    plot_distribution_with_points(
        rows, "best_test_loss", "oracle best test loss by pattern", "pattern_best_test_loss.png", dirs["primary"], log_scale=True
    )
    plot_pattern_trajectory_iqr(rows, dirs["primary"])
    plot_best_vs_gap(rows, dirs["primary"])
    plot_pruned2_tradeoff(rows, dirs["primary"])

    plot_distribution_with_points(
        rows, "final_test_loss", "final test loss by pattern", "pattern_final_test_loss.png", dirs["supplementary"], log_scale=True
    )
    plot_distribution_with_points(
        rows, "final_minus_best", "final minus best by pattern", "pattern_instability_gap.png", dirs["supplementary"], log_scale=True
    )
    plot_distribution_with_points(
        rows, "best_epoch", "best epoch by pattern", "pattern_best_epoch.png", dirs["supplementary"], log_scale=False
    )
    plot_distribution_with_points(
        rows, "total_pruned", "total pruned by pattern", "pattern_total_pruned.png", dirs["supplementary"], log_scale=False
    )
    plot_exemplar_curves(rows, dirs["supplementary"])
    write_report(summary_rows)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
