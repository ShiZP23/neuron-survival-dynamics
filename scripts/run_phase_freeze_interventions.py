import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from neuron_survival_dynamics.train import train_one_run  # noqa: E402
from neuron_survival_dynamics.utils import get_device, save_json, set_seed  # noqa: E402


PHASE_ROWS_PATH = Path("results/studies/structural_phase_effects_20260314/phase_rows.csv")
RAW_ROOT = Path("results/followup_20260314/phase_freeze_interventions_pilot_runs")
STUDY_OUT = Path("results/studies/phase_freeze_interventions_pilot_20260314")
PATTERN_ORDER = ["prune0_only", "prune0+2", "no_prune"]
PATTERN_COLORS = {
    "prune0_only": "#f58518",
    "prune0+2": "#e45756",
    "no_prune": "#4c78a8",
}
CONDITION_ORDER = ["baseline_reference", "freeze_after_commit", "freeze_after_best"]
CONDITION_LABELS = {
    "baseline_reference": "baseline",
    "freeze_after_commit": "freeze@commit",
    "freeze_after_best": "freeze@best",
}
CONDITION_MARKERS = {
    "baseline_reference": "o",
    "freeze_after_commit": "s",
    "freeze_after_best": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run freeze-after-best/commit pilot interventions for structural-phase runs")
    parser.add_argument("--phase-rows", type=Path, default=PHASE_ROWS_PATH)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--study-out", type=Path, default=STUDY_OUT)
    parser.add_argument("--family", type=str, default="base_seed")
    parser.add_argument("--per-phase", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_component(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def share_vector(active_counts: Sequence[int]) -> Tuple[float, float, float]:
    active = np.array(list(active_counts), dtype=float)
    total = max(float(active.sum()), 1.0)
    shares = active / total
    return float(shares[0]), float(shares[1]), float(shares[2])


def share_reallocation_magnitude(best_shares: Tuple[float, float, float], final_shares: Tuple[float, float, float]) -> float:
    deltas = np.abs(np.array(final_shares, dtype=float) - np.array(best_shares, dtype=float))
    return 0.5 * float(deltas.sum())


def classify_pattern(pruned_0: int, pruned_1: int, pruned_2: int) -> str:
    if pruned_0 > 0 and pruned_2 > 0:
        return "prune0+2"
    if pruned_0 > 0 and pruned_2 == 0:
        return "prune0_only"
    if pruned_0 == 0 and pruned_1 == 0 and pruned_2 == 0:
        return "no_prune"
    return "other"


def select_representative_rows(rows: Sequence[Dict[str, str]], family: str, per_phase: int) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    family_rows = [row for row in rows if row["family"] == family]
    for pattern in PATTERN_ORDER:
        group = [row for row in family_rows if row["pattern"] == pattern]
        if not group:
            continue
        if len(group) <= per_phase:
            selected.extend(sorted(group, key=lambda row: int(row["seed"])))
            continue
        log_degradation = [np.log10(max(float(row["final_minus_best"]), 1e-12)) for row in group]
        share2 = [float(row["share_2_drift"]) for row in group]
        log_center = median(log_degradation)
        share2_center = median(share2)
        log_scale = median([abs(value - log_center) for value in log_degradation]) or 1.0
        share2_scale = median([abs(value - share2_center) for value in share2]) or 1.0
        ordered = sorted(
            group,
            key=lambda row: (
                abs(np.log10(max(float(row["final_minus_best"]), 1e-12)) - log_center) / log_scale
                + abs(float(row["share_2_drift"]) - share2_center) / share2_scale,
                int(row["seed"]),
            ),
        )
        selected.extend(ordered[:per_phase])
    return sorted(selected, key=lambda row: (PATTERN_ORDER.index(row["pattern"]), int(row["seed"])))


def load_source_config(run_path: Path) -> Dict:
    return json.loads((run_path / "config.json").read_text())


def summarize_history(
    history: Sequence[Dict],
    run_dir: Path,
    source_row: Dict[str, str],
    condition: str,
    freeze_epoch: Optional[int],
) -> Dict:
    test_losses = np.array([float(row["test_loss"]) for row in history], dtype=float)
    train_losses = np.array([float(row["train_loss"]) for row in history], dtype=float)
    best_idx = int(np.argmin(test_losses))
    best_row = history[best_idx]
    final_row = history[-1]
    best_active = [int(value) for value in best_row["active_counts"]]
    final_active = [int(value) for value in final_row["active_counts"]]
    best_shares = share_vector(best_active)
    final_shares = share_vector(final_active)
    total_pruned = [sum(int(epoch_row["pruned"][layer_idx]) for epoch_row in history) for layer_idx in range(3)]
    final_pattern = classify_pattern(total_pruned[0], total_pruned[1], total_pruned[2])
    post_best_epochs = int(final_row["epoch"]) - int(best_row["epoch"])
    post_best_share_drift = share_reallocation_magnitude(best_shares, final_shares)
    return {
        "source_run_path": source_row["run_path"],
        "source_seed": int(source_row["seed"]),
        "source_run_name": source_row["run_name"],
        "source_pattern": source_row["pattern"],
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "freeze_epoch": freeze_epoch if freeze_epoch is not None else -1,
        "run_dir": str(run_dir),
        "best_epoch": int(best_row["epoch"]),
        "final_epoch": int(final_row["epoch"]),
        "post_best_epochs": post_best_epochs,
        "final_pattern": final_pattern,
        "best_test_loss": float(test_losses[best_idx]),
        "final_test_loss": float(test_losses[-1]),
        "final_minus_best": float(test_losses[-1] - test_losses[best_idx]),
        "best_test_train_gap": float(test_losses[best_idx] - train_losses[best_idx]),
        "final_test_train_gap": float(test_losses[-1] - train_losses[-1]),
        "active_0_best": best_active[0],
        "active_1_best": best_active[1],
        "active_2_best": best_active[2],
        "active_0_final": final_active[0],
        "active_1_final": final_active[1],
        "active_2_final": final_active[2],
        "share_0_best": best_shares[0],
        "share_1_best": best_shares[1],
        "share_2_best": best_shares[2],
        "share_0_final": final_shares[0],
        "share_1_final": final_shares[1],
        "share_2_final": final_shares[2],
        "share_0_drift": final_shares[0] - best_shares[0],
        "share_1_drift": final_shares[1] - best_shares[1],
        "share_2_drift": final_shares[2] - best_shares[2],
        "post_best_share_drift": post_best_share_drift,
        "post_best_drift_rate": post_best_share_drift / max(post_best_epochs, 1),
        "share_2_drift_rate": (final_shares[2] - best_shares[2]) / max(post_best_epochs, 1),
    }


def baseline_reference_row(source_row: Dict[str, str]) -> Dict:
    return {
        "source_run_path": source_row["run_path"],
        "source_seed": int(source_row["seed"]),
        "source_run_name": source_row["run_name"],
        "source_pattern": source_row["pattern"],
        "condition": "baseline_reference",
        "condition_label": CONDITION_LABELS["baseline_reference"],
        "freeze_epoch": -1,
        "run_dir": source_row["run_path"],
        "best_epoch": int(source_row["best_epoch"]),
        "final_epoch": int(source_row["final_epoch"]),
        "post_best_epochs": int(source_row["post_best_epochs"]),
        "final_pattern": source_row["pattern"],
        "best_test_loss": float(source_row["best_test_loss"]),
        "final_test_loss": float(source_row["final_test_loss"]),
        "final_minus_best": float(source_row["final_minus_best"]),
        "best_test_train_gap": float(source_row["best_test_train_gap"]),
        "final_test_train_gap": float(source_row["final_test_train_gap"]),
        "active_0_best": int(source_row["active_0_best"]),
        "active_1_best": int(source_row["active_1_best"]),
        "active_2_best": int(source_row["active_2_best"]),
        "active_0_final": int(source_row["active_0_final"]),
        "active_1_final": int(source_row["active_1_final"]),
        "active_2_final": int(source_row["active_2_final"]),
        "share_0_best": float(source_row["share_0_best"]),
        "share_1_best": float(source_row["share_1_best"]),
        "share_2_best": float(source_row["share_2_best"]),
        "share_0_final": float(source_row["share_0_final"]),
        "share_1_final": float(source_row["share_1_final"]),
        "share_2_final": float(source_row["share_2_final"]),
        "share_0_drift": float(source_row["share_0_drift"]),
        "share_1_drift": float(source_row["share_1_drift"]),
        "share_2_drift": float(source_row["share_2_drift"]),
        "post_best_share_drift": float(source_row["post_best_share_drift"]),
        "post_best_drift_rate": float(source_row["post_best_drift_rate"]),
        "share_2_drift_rate": float(source_row["share_2_drift_rate"]),
    }


def intervention_run_dir(raw_root: Path, config: Dict, condition: str, source_row: Dict[str, str]) -> Path:
    condition_mode = f"{config['mode']}__{condition}"
    label = f"source_seed{source_row['seed']}__{safe_component(source_row['run_name'])}"
    return raw_root / config["task"] / condition_mode / f"seed_{config['seed']}" / label


def run_intervention(
    source_row: Dict[str, str],
    condition: str,
    freeze_epoch: int,
    raw_root: Path,
    device,
    force: bool,
) -> Dict:
    source_config = load_source_config(Path(source_row["run_path"]))
    run_dir = intervention_run_dir(raw_root, source_config, condition, source_row)
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and not force:
        history = []
        for row in read_csv(metrics_path):
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "test_loss": float(row["test_loss"]),
                    "active_counts": [int(row[f"active_{layer_idx}"]) for layer_idx in range(3)],
                    "pruned": [int(row[f"pruned_{layer_idx}"]) for layer_idx in range(3)],
                }
            )
        return summarize_history(history, run_dir, source_row, condition, freeze_epoch)

    ensure_dir(run_dir)
    run_config = dict(source_config)
    run_config.update(
        {
            "intervention": condition,
            "freeze_structure_after_epoch": freeze_epoch,
            "freeze_structure_reference": condition.replace("freeze_after_", ""),
            "source_run_path": source_row["run_path"],
            "source_pattern": source_row["pattern"],
        }
    )
    save_json(run_dir / "config.json", run_config)
    set_seed(int(source_config["seed"]))
    _, history = train_one_run(
        task=source_config["task"],
        mode=source_config["mode"],
        run_dir=str(run_dir),
        seed=int(source_config["seed"]),
        device=device,
        epochs=int(source_config["epochs"]),
        update_interval=int(source_config["update_interval"]),
        batch_size=int(source_config["batch_size"]),
        lr=float(source_config["lr"]),
        min_neurons=int(source_config["min_neurons"]),
        n_train=int(source_config["n_train"]),
        n_val=int(source_config["n_val"]),
        n_test=int(source_config["n_test"]),
        noise=float(source_config["noise"]),
        data_seed=int(source_config["data_seed"]),
        model_seed=int(source_config["model_seed"]),
        shuffle_seed=int(source_config["shuffle_seed"]),
        structure_seed=int(source_config["structure_seed"]),
        ema_beta=float(source_config["ema_beta"]),
        ema_z_threshold=float(source_config["ema_z_threshold"]),
        max_candidates_per_layer=int(source_config["max_candidates_per_layer"]),
        ablation_epsilon_ratio=float(source_config["ablation_epsilon_ratio"]),
        active_threshold=float(source_config["active_threshold"]),
        freeze_structure_after_epoch=freeze_epoch,
        freeze_structure_reference=condition.replace("freeze_after_", ""),
    )
    return summarize_history(history, run_dir, source_row, condition, freeze_epoch)


def add_deltas(rows: Sequence[Dict]) -> List[Dict]:
    baseline_by_source = {
        (row["source_seed"], row["source_run_name"]): row
        for row in rows
        if row["condition"] == "baseline_reference"
    }
    out: List[Dict] = []
    for row in rows:
        baseline = baseline_by_source[(row["source_seed"], row["source_run_name"])]
        row_copy = dict(row)
        row_copy["delta_final_test_loss_vs_baseline"] = float(row["final_test_loss"]) - float(baseline["final_test_loss"])
        row_copy["delta_final_minus_best_vs_baseline"] = float(row["final_minus_best"]) - float(baseline["final_minus_best"])
        row_copy["delta_share_2_drift_vs_baseline"] = float(row["share_2_drift"]) - float(baseline["share_2_drift"])
        row_copy["delta_post_best_drift_rate_vs_baseline"] = float(row["post_best_drift_rate"]) - float(
            baseline["post_best_drift_rate"]
        )
        out.append(row_copy)
    return out


def summarize_conditions(rows: Sequence[Dict]) -> List[Dict]:
    summary: List[Dict] = []
    for pattern in PATTERN_ORDER:
        pattern_rows = [row for row in rows if row["source_pattern"] == pattern]
        if not pattern_rows:
            continue
        for condition in CONDITION_ORDER:
            group = [row for row in pattern_rows if row["condition"] == condition]
            if not group:
                continue
            summary.append(
                {
                    "source_pattern": pattern,
                    "condition": condition,
                    "n": len(group),
                    "median_final_test_loss": median(float(row["final_test_loss"]) for row in group),
                    "median_final_minus_best": median(float(row["final_minus_best"]) for row in group),
                    "median_share_2_drift": median(float(row["share_2_drift"]) for row in group),
                    "median_post_best_drift_rate": median(float(row["post_best_drift_rate"]) for row in group),
                    "median_delta_final_test_loss_vs_baseline": median(
                        float(row["delta_final_test_loss_vs_baseline"]) for row in group
                    ),
                    "median_delta_final_minus_best_vs_baseline": median(
                        float(row["delta_final_minus_best_vs_baseline"]) for row in group
                    ),
                    "median_delta_share_2_drift_vs_baseline": median(
                        float(row["delta_share_2_drift_vs_baseline"]) for row in group
                    ),
                }
            )
    return summary


def plot_intervention_effects(rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    panels = [
        ("final_minus_best", "final-best", True),
        ("share_2_drift", "layer-2 share drift", False),
        ("post_best_drift_rate", "post-best drift rate", False),
    ]
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["source_seed"]), row["source_run_name"]), []).append(row)

    for ax, (metric, title, log_scale) in zip(axes, panels):
        for (seed, run_name), group in sorted(grouped.items()):
            ordered = sorted(group, key=lambda row: CONDITION_ORDER.index(row["condition"]))
            x = np.arange(len(ordered))
            y = [float(row[metric]) for row in ordered]
            pattern = ordered[0]["source_pattern"]
            ax.plot(x, y, color=PATTERN_COLORS[pattern], linewidth=1.4, alpha=0.75)
            for xpos, row, value in zip(x, ordered, y):
                ax.scatter(
                    xpos,
                    value,
                    marker=CONDITION_MARKERS[row["condition"]],
                    s=58,
                    color=PATTERN_COLORS[pattern],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
        ax.set_xticks(np.arange(len(CONDITION_ORDER)))
        ax.set_xticklabels([CONDITION_LABELS[condition] for condition in CONDITION_ORDER])
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        if log_scale:
            ax.set_yscale("log")
        if metric.endswith("drift"):
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("metric value")
    axes[1].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=PATTERN_COLORS[pattern],
                markeredgecolor="white",
                label=pattern,
            )
            for pattern in PATTERN_ORDER
            if any(row["source_pattern"] == pattern for row in rows)
        ],
        fontsize=8,
        loc="upper right",
        title="phase",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(
    selected_rows: Sequence[Dict[str, str]],
    comparison_rows: Sequence[Dict],
    condition_summary: Sequence[Dict],
    out_path: Path,
) -> None:
    summary_lookup = {(row["source_pattern"], row["condition"]): row for row in condition_summary}
    lines = [
        "# Phase Freeze Interventions Pilot",
        "",
        "Purpose:",
        "- test whether late structural movement is causally connected to final degradation",
        "- compare `freeze_after_commit` against `freeze_after_best` on representative baseline runs",
        "",
        "Selection rule:",
        "- source pool: `results/studies/structural_phase_effects_20260314/phase_rows.csv`",
        "- family filter: `base_seed`",
        "- representative runs chosen by closeness to phase medians in `(log final_minus_best, share_2_drift)`",
        "",
        "Selected source runs:",
    ]
    for row in selected_rows:
        lines.append(
            f"- {row['pattern']} seed={row['seed']} run={row['run_name']} "
            f"(baseline final-best={float(row['final_minus_best']):.6g}, share_2_drift={float(row['share_2_drift']):.4f})"
        )
    lines.extend(
        [
            "",
            "Condition medians by source phase:",
        ]
    )
    for pattern in PATTERN_ORDER:
        pattern_rows = [row for row in condition_summary if row["source_pattern"] == pattern]
        if not pattern_rows:
            continue
        for condition in CONDITION_ORDER:
            row = summary_lookup.get((pattern, condition))
            if row is None:
                continue
            lines.append(
                f"- {pattern} / {condition}: n={row['n']}, "
                f"median final-best={row['median_final_minus_best']:.6g}, "
                f"median final loss={row['median_final_test_loss']:.6g}, "
                f"median share_2_drift={row['median_share_2_drift']:.4f}, "
                f"median post_best_drift_rate={row['median_post_best_drift_rate']:.6g}"
            )
            if condition != "baseline_reference":
                lines.append(
                    f"  delta vs baseline: final-best={row['median_delta_final_minus_best_vs_baseline']:.6g}, "
                    f"final loss={row['median_delta_final_test_loss_vs_baseline']:.6g}, "
                    f"share_2_drift={row['median_delta_share_2_drift_vs_baseline']:.4f}"
                )
    lines.extend(
        [
            "",
            "Artifacts:",
            "- `comparison_rows.csv`",
            "- `condition_phase_summary.csv`",
            "- `primary/freeze_intervention_effects.png`",
            "",
            "Caveat:",
            "- this is a pilot on representative base-seed runs, not a full rerun of every structural-phase sample",
            "- intervention runs are deterministic reruns from the original config with structural updates disabled after a baseline-derived epoch",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.raw_root)
    ensure_dir(args.study_out / "primary")

    phase_rows = read_csv(args.phase_rows)
    selected_rows = select_representative_rows(phase_rows, family=args.family, per_phase=args.per_phase)

    comparison_rows: List[Dict] = [baseline_reference_row(row) for row in selected_rows]
    device = get_device(args.device)
    for row in selected_rows:
        commit_epoch = int(row["commit_epoch"])
        best_epoch = int(row["best_epoch"])
        comparison_rows.append(
            run_intervention(
                source_row=row,
                condition="freeze_after_commit",
                freeze_epoch=commit_epoch,
                raw_root=args.raw_root,
                device=device,
                force=args.force,
            )
        )
        comparison_rows.append(
            run_intervention(
                source_row=row,
                condition="freeze_after_best",
                freeze_epoch=best_epoch,
                raw_root=args.raw_root,
                device=device,
                force=args.force,
            )
        )

    comparison_rows = add_deltas(comparison_rows)
    condition_summary = summarize_conditions(comparison_rows)

    write_csv(args.study_out / "selection_manifest.csv", selected_rows)
    write_csv(args.study_out / "comparison_rows.csv", comparison_rows)
    write_csv(args.study_out / "condition_phase_summary.csv", condition_summary)
    plot_intervention_effects(comparison_rows, args.study_out / "primary" / "freeze_intervention_effects.png")
    write_readme(selected_rows, comparison_rows, condition_summary, args.study_out / "README.md")
    print(args.study_out)


if __name__ == "__main__":
    main()
