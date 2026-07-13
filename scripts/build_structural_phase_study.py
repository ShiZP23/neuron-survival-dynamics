import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("former results/3.12results，128/hard/prune_grow_split")
OUT_DIR = Path("results/studies/structural_phase_effects_20260314")
PRIMARY_DIR = OUT_DIR / "primary"
SUPPLEMENTARY_DIR = OUT_DIR / "supplementary"
PATTERNS = ["prune0_only", "prune0+2", "no_prune"]
PATTERN_COLORS = {
    "prune0_only": "#f58518",
    "prune0+2": "#e45756",
    "no_prune": "#4c78a8",
}
PATTERN_LABELS = {
    "prune0_only": "prune0_only",
    "prune0+2": "prune0+2",
    "no_prune": "no_prune",
}
FAMILY_ORDER = ["base_seed", "initialization", "data_realization", "anchor"]
FAMILY_LABELS = {
    "base_seed": "base seed",
    "initialization": "init sweep",
    "data_realization": "data sweep",
    "anchor": "shared anchor",
}
FAMILY_MARKERS = {
    "base_seed": "o",
    "initialization": "^",
    "data_realization": "s",
    "anchor": "D",
}
FACTOR_ORDER = ["base_seed", "initialization", "data_realization"]
FACTOR_LABELS = {
    "base_seed": "Base seed sweep",
    "initialization": "Initialization sweep",
    "data_realization": "Data realization sweep",
}
BOOTSTRAP_SAMPLES = 20000
PERMUTATION_SAMPLES = 20000


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


def valid_values(rows: Sequence[Dict], key: str) -> List[float]:
    values = [row[key] for row in rows]
    if key.startswith("reach_") or key.startswith("first_prune"):
        return [value for value in values if value >= 0]
    return list(values)


def valid_rows(rows: Sequence[Dict], key: str) -> List[Dict]:
    if key.startswith("reach_") or key.startswith("first_prune"):
        return [row for row in rows if row[key] >= 0]
    return list(rows)


def classify_pattern(pruned_0: int, pruned_1: int, pruned_2: int) -> str:
    if pruned_0 > 0 and pruned_2 > 0:
        return "prune0+2"
    if pruned_0 > 0 and pruned_2 == 0:
        return "prune0_only"
    if pruned_0 == 0 and pruned_1 == 0 and pruned_2 == 0:
        return "no_prune"
    return "other"


def classify_family(seed_dir: str, run_name: str) -> str:
    if seed_dir == "seed_0" and run_name.startswith("data"):
        return "data_realization"
    if seed_dir == "seed_0" and run_name == "seed0":
        return "anchor"
    if seed_dir == "seed_0":
        return "initialization"
    return "base_seed"


def first_reach(metrics: Sequence[Dict[str, str]], threshold: float) -> int:
    for row in metrics:
        if float(row["test_loss"]) <= threshold:
            return int(row["epoch"])
    return -1


def first_prune_epoch(metrics: Sequence[Dict[str, str]], layer_idx: int) -> int:
    key = f"pruned_{layer_idx}"
    for row in metrics:
        if int(row["is_update_epoch"]) == 1 and int(row[key]) > 0:
            return int(row["epoch"])
    return -1


def phase_commit_epoch(metrics: Sequence[Dict[str, str]], pattern: str) -> int:
    if not metrics:
        return -1
    if pattern == "no_prune":
        return int(metrics[0]["epoch"])
    cumulative_pruned = [0, 0, 0]
    for row in metrics:
        if int(row["is_update_epoch"]) == 1:
            for layer_idx in range(3):
                cumulative_pruned[layer_idx] += int(row[f"pruned_{layer_idx}"])
        if pattern == "prune0_only" and cumulative_pruned[0] > 0:
            return int(row["epoch"])
        if pattern == "prune0+2" and cumulative_pruned[0] > 0 and cumulative_pruned[2] > 0:
            return int(row["epoch"])
    return -1


def active_share_vector(metric_row: Dict[str, str]) -> Tuple[float, float, float]:
    active = np.array([int(metric_row[f"active_{layer_idx}"]) for layer_idx in range(3)], dtype=float)
    total_active = max(float(active.sum()), 1.0)
    shares = active / total_active
    return float(shares[0]), float(shares[1]), float(shares[2])


def share_reallocation_magnitude(best_shares: Tuple[float, float, float], final_shares: Tuple[float, float, float]) -> float:
    deltas = np.abs(np.array(final_shares, dtype=float) - np.array(best_shares, dtype=float))
    return 0.5 * float(deltas.sum())


def collect_rows() -> List[Dict]:
    rows: List[Dict] = []
    for metrics_path in sorted(ROOT.glob("seed_*/*/metrics.csv")):
        metrics = read_csv(metrics_path)
        if not metrics:
            continue
        config = json.loads((metrics_path.parent / "config.json").read_text())
        updates = [row for row in metrics if int(row["is_update_epoch"]) == 1]
        pruned_0_total = sum(int(row["pruned_0"]) for row in updates)
        pruned_1_total = sum(int(row["pruned_1"]) for row in updates)
        pruned_2_total = sum(int(row["pruned_2"]) for row in updates)
        pattern = classify_pattern(pruned_0_total, pruned_1_total, pruned_2_total)
        if pattern not in PATTERNS:
            continue
        family = classify_family(metrics_path.parent.parent.name, metrics_path.parent.name)
        test_losses = np.array([float(row["test_loss"]) for row in metrics], dtype=float)
        train_losses = np.array([float(row["train_loss"]) for row in metrics], dtype=float)
        best_idx = int(np.argmin(test_losses))
        best = metrics[best_idx]
        final = metrics[-1]
        final_epoch = int(final["epoch"])
        commit_epoch = phase_commit_epoch(metrics, pattern)
        best_shares = active_share_vector(best)
        final_shares = active_share_vector(final)
        post_best_epochs = final_epoch - int(best["epoch"])
        post_best_share_drift = share_reallocation_magnitude(best_shares, final_shares)
        rows.append(
            {
                "run_path": str(metrics_path.parent),
                "seed_dir": metrics_path.parent.parent.name,
                "run_name": metrics_path.parent.name,
                "family": family,
                "pattern": pattern,
                "seed": int(config["seed"]),
                "data_seed": int(config["data_seed"]),
                "model_seed": int(config["model_seed"]),
                "shuffle_seed": int(config["shuffle_seed"]),
                "structure_seed": int(config["structure_seed"]),
                "best_epoch": int(best["epoch"]),
                "final_epoch": final_epoch,
                "commit_epoch": commit_epoch,
                "best_after_commit_lag": int(best["epoch"]) - commit_epoch if commit_epoch >= 0 else -1,
                "post_best_epochs": post_best_epochs,
                "best_test_loss": float(test_losses[best_idx]),
                "final_test_loss": float(test_losses[-1]),
                "final_minus_best": float(test_losses[-1] - test_losses[best_idx]),
                "first_prune0_epoch": first_prune_epoch(metrics, 0),
                "first_prune2_epoch": first_prune_epoch(metrics, 2),
                "reach_1e_3": first_reach(metrics, 1e-3),
                "reach_5e_4": first_reach(metrics, 5e-4),
                "reach_2e_4": first_reach(metrics, 2e-4),
                "best_train_loss": float(train_losses[best_idx]),
                "final_train_loss": float(train_losses[-1]),
                "best_test_train_gap": float(test_losses[best_idx] - train_losses[best_idx]),
                "final_test_train_gap": float(test_losses[-1] - train_losses[-1]),
                "total_pruned": sum(int(row["total_pruned"]) for row in updates),
                "total_grown": sum(int(row["total_grown"]) for row in updates),
                "pruned_0_total": pruned_0_total,
                "pruned_1_total": pruned_1_total,
                "pruned_2_total": pruned_2_total,
                "active_0_best": int(best["active_0"]),
                "active_1_best": int(best["active_1"]),
                "active_2_best": int(best["active_2"]),
                "active_0_final": int(final["active_0"]),
                "active_1_final": int(final["active_1"]),
                "active_2_final": int(final["active_2"]),
                "active_0_drift": int(final["active_0"]) - int(best["active_0"]),
                "active_1_drift": int(final["active_1"]) - int(best["active_1"]),
                "active_2_drift": int(final["active_2"]) - int(best["active_2"]),
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
        )
    return rows


def build_factor_view_rows(rows: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        if row["family"] == "base_seed":
            out.append(
                {
                    "factor": "base_seed",
                    "factor_value": row["seed"],
                    "pattern": row["pattern"],
                    "run_name": row["run_name"],
                    "family": row["family"],
                }
            )
        elif row["family"] == "initialization":
            out.append(
                {
                    "factor": "initialization",
                    "factor_value": row["model_seed"],
                    "pattern": row["pattern"],
                    "run_name": row["run_name"],
                    "family": row["family"],
                }
            )
        elif row["family"] == "data_realization":
            out.append(
                {
                    "factor": "data_realization",
                    "factor_value": row["data_seed"],
                    "pattern": row["pattern"],
                    "run_name": row["run_name"],
                    "family": row["family"],
                }
            )
        elif row["family"] == "anchor":
            out.extend(
                [
                    {
                        "factor": "base_seed",
                        "factor_value": row["seed"],
                        "pattern": row["pattern"],
                        "run_name": row["run_name"],
                        "family": row["family"],
                    },
                    {
                        "factor": "data_realization",
                        "factor_value": row["data_seed"],
                        "pattern": row["pattern"],
                        "run_name": row["run_name"],
                        "family": row["family"],
                    },
                ]
            )
    return out


def summarize_patterns(rows: Sequence[Dict]) -> List[Dict]:
    summary: List[Dict] = []
    keys = [
        "best_test_loss",
        "final_test_loss",
        "final_minus_best",
        "commit_epoch",
        "best_after_commit_lag",
        "post_best_epochs",
        "post_best_share_drift",
        "post_best_drift_rate",
        "share_2_drift_rate",
        "first_prune0_epoch",
        "first_prune2_epoch",
        "reach_5e_4",
        "best_epoch",
        "best_test_train_gap",
        "final_test_train_gap",
        "active_0_best",
        "active_1_best",
        "active_2_best",
        "active_0_final",
        "active_1_final",
        "active_2_final",
        "active_0_drift",
        "active_1_drift",
        "active_2_drift",
        "share_0_best",
        "share_1_best",
        "share_2_best",
        "share_0_final",
        "share_1_final",
        "share_2_final",
        "share_0_drift",
        "share_1_drift",
        "share_2_drift",
        "pruned_0_total",
        "pruned_1_total",
        "pruned_2_total",
    ]
    for pattern in PATTERNS:
        group = [row for row in rows if row["pattern"] == pattern]
        if not group:
            continue
        summary_row = {"pattern": pattern, "n": len(group)}
        for key in keys:
            values = valid_values(group, key)
            if values:
                summary_row[f"mean_{key}"] = mean(values)
                summary_row[f"median_{key}"] = median(values)
            else:
                summary_row[f"mean_{key}"] = -1
                summary_row[f"median_{key}"] = -1
        summary.append(summary_row)
    return summary


def summarize_factors(view_rows: Sequence[Dict]) -> List[Dict]:
    summary: List[Dict] = []
    for factor in FACTOR_ORDER:
        group = [row for row in view_rows if row["factor"] == factor]
        if not group:
            continue
        counts = {pattern: sum(1 for row in group if row["pattern"] == pattern) for pattern in PATTERNS}
        dominant_pattern, dominant_count = max(counts.items(), key=lambda item: item[1])
        summary.append(
            {
                "factor": factor,
                "n": len(group),
                "count_prune0_only": counts["prune0_only"],
                "count_prune0+2": counts["prune0+2"],
                "count_no_prune": counts["no_prune"],
                "dominant_pattern": dominant_pattern,
                "dominant_fraction": dominant_count / len(group),
                "distinct_patterns": sum(count > 0 for count in counts.values()),
            }
        )
    return summary


def bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    stat_fn,
    rng: np.random.Generator,
    n_samples: int = BOOTSTRAP_SAMPLES,
) -> Tuple[float, float]:
    a_arr = np.array(list(a), dtype=float)
    b_arr = np.array(list(b), dtype=float)
    diffs = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        sample_a = rng.choice(a_arr, size=len(a_arr), replace=True)
        sample_b = rng.choice(b_arr, size=len(b_arr), replace=True)
        diffs[idx] = float(stat_fn(sample_a) - stat_fn(sample_b))
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high)


def permutation_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    stat_fn,
    rng: np.random.Generator,
    n_samples: int = PERMUTATION_SAMPLES,
) -> float:
    a_arr = np.array(list(a), dtype=float)
    b_arr = np.array(list(b), dtype=float)
    observed = float(stat_fn(a_arr) - stat_fn(b_arr))
    pool = np.concatenate([a_arr, b_arr])
    count = 0
    for _ in range(n_samples):
        permuted = rng.permutation(pool)
        diff = float(stat_fn(permuted[: len(a_arr)]) - stat_fn(permuted[len(a_arr) :]))
        if abs(diff) >= abs(observed):
            count += 1
    return (count + 1) / (n_samples + 1)


def build_pairwise_stats(rows: Sequence[Dict]) -> List[Dict]:
    rng = np.random.default_rng(0)
    a_group = [row for row in rows if row["pattern"] == "prune0_only"]
    b_group = [row for row in rows if row["pattern"] == "prune0+2"]
    metrics = [
        ("best_test_loss", "best test loss"),
        ("final_test_loss", "final test loss"),
        ("final_minus_best", "final minus best"),
        ("commit_epoch", "phase commit epoch"),
        ("best_after_commit_lag", "best-after-commit lag"),
        ("post_best_drift_rate", "post-best drift rate"),
        ("share_2_drift_rate", "share layer-2 drift rate"),
        ("reach_5e_4", "epoch reaching 5e-4"),
        ("best_epoch", "best epoch"),
        ("final_test_train_gap", "final test-train gap"),
        ("active_1_best", "active layer-1 @ best"),
        ("active_2_best", "active layer-2 @ best"),
        ("active_1_drift", "active layer-1 drift"),
        ("active_2_drift", "active layer-2 drift"),
        ("share_1_drift", "share layer-1 drift"),
        ("share_2_drift", "share layer-2 drift"),
    ]
    out: List[Dict] = []
    for key, label in metrics:
        a_values = valid_values(a_group, key)
        b_values = valid_values(b_group, key)
        median_diff = median(a_values) - median(b_values)
        ci_low, ci_high = bootstrap_ci(a_values, b_values, np.median, rng)
        out.append(
            {
                "metric": key,
                "label": label,
                "group_a": "prune0_only",
                "group_b": "prune0+2",
                "n_group_a": len(a_values),
                "n_group_b": len(b_values),
                "median_group_a": median(a_values),
                "median_group_b": median(b_values),
                "median_diff_a_minus_b": median_diff,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_pvalue": permutation_pvalue(a_values, b_values, np.median, rng),
            }
        )
    return out


def _active_patterns(rows: Sequence[Dict]) -> List[str]:
    return [pattern for pattern in PATTERNS if any(row["pattern"] == pattern for row in rows)]


def family_legend_handles() -> List[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="",
            markersize=8,
            color="#555555",
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]


def run_sort_value(row: Dict) -> Tuple[int, int, int]:
    family_rank = FAMILY_ORDER.index(row["family"])
    if row["family"] == "base_seed":
        factor_value = int(row["seed"])
    elif row["family"] == "initialization":
        factor_value = int(row["model_seed"])
    elif row["family"] == "data_realization":
        factor_value = int(row["data_seed"])
    else:
        factor_value = int(row["seed"])
    return (PATTERNS.index(row["pattern"]), family_rank, factor_value)


def run_label(row: Dict) -> str:
    if row["family"] == "base_seed":
        return f"s{row['seed']}"
    if row["family"] == "initialization":
        return f"m{row['model_seed']}"
    if row["family"] == "data_realization":
        return f"d{row['data_seed']}"
    return "anchor"


def plot_phase_assignments(view_rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 5.8), sharex=False)
    for ax, factor in zip(axes, FACTOR_ORDER):
        group = sorted(
            [row for row in view_rows if row["factor"] == factor],
            key=lambda row: row["factor_value"],
        )
        counts = {pattern: sum(1 for row in group if row["pattern"] == pattern) for pattern in PATTERNS}
        positions = np.arange(len(group))
        for idx, row in enumerate(group):
            ax.scatter(
                idx,
                0,
                s=280,
                marker="s",
                color=PATTERN_COLORS[row["pattern"]],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        ax.set_xlim(-0.6, len(group) - 0.4)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks(positions)
        ax.set_xticklabels([str(row["factor_value"]) for row in group])
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_title(
            f"{FACTOR_LABELS[factor]}  "
            f"(prune0_only={counts['prune0_only']}, prune0+2={counts['prune0+2']}, no_prune={counts['no_prune']})"
        )
    axes[0].legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=10,
                markerfacecolor=PATTERN_COLORS[pattern],
                markeredgecolor="white",
                label=PATTERN_LABELS[pattern],
            )
            for pattern in PATTERNS
        ],
        ncol=3,
        fontsize=8,
        loc="upper right",
    )
    axes[0].set_xlabel("factor value")
    axes[1].set_xlabel("factor value")
    axes[2].set_xlabel("factor value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _scatter_by_family(
    ax,
    rows: Sequence[Dict],
    key: str,
    log_scale: bool = False,
) -> None:
    patterns = _active_patterns(rows)
    positions = np.arange(1, len(patterns) + 1)
    groups = [valid_values([row for row in rows if row["pattern"] == pattern], key) for pattern in patterns]
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=patterns, showfliers=False, widths=0.58)
    for patch, pattern in zip(bp["boxes"], patterns):
        patch.set_facecolor(PATTERN_COLORS[pattern])
        patch.set_alpha(0.3)
    for pos, pattern in zip(positions, patterns):
        group = valid_rows([row for row in rows if row["pattern"] == pattern], key)
        jitter = np.linspace(-0.11, 0.11, len(group)) if len(group) > 1 else np.array([0.0])
        for offset, row in zip(jitter, group):
            ax.scatter(
                pos + offset,
                float(row[key]),
                marker=FAMILY_MARKERS[row["family"]],
                color=PATTERN_COLORS[pattern],
                edgecolor="white",
                linewidth=0.6,
                s=46,
                zorder=3,
                alpha=0.95,
            )
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_endpoint_panels(rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    panels = [
        ("best_test_loss", "best test loss", True),
        ("final_test_loss", "final test loss", True),
        ("final_test_train_gap", "final test-train gap", True),
    ]
    for ax, (key, title, log_scale) in zip(axes, panels):
        _scatter_by_family(ax, rows, key, log_scale=log_scale)
        ax.set_title(title)
    axes[0].legend(handles=family_legend_handles(), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_speed_panels(rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    panels = [
        ("reach_5e_4", "epoch reaching 5e-4"),
        ("best_epoch", "best epoch"),
    ]
    for ax, (key, title) in zip(axes, panels):
        _scatter_by_family(ax, rows, key, log_scale=False)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_active_trajectories(rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharex=True)
    for layer_idx, ax in enumerate(axes):
        for pattern in PATTERNS:
            group = [row for row in rows if row["pattern"] == pattern]
            if not group:
                continue
            curves = []
            epochs = None
            for row in group:
                metrics = read_csv(Path(row["run_path"]) / "metrics.csv")
                active = np.array([int(metric[f"active_{layer_idx}"]) for metric in metrics], dtype=float)
                curves.append(active)
                if epochs is None:
                    epochs = np.array([int(metric["epoch"]) for metric in metrics], dtype=int)
            curve_array = np.stack(curves, axis=0)
            q25 = np.quantile(curve_array, 0.25, axis=0)
            q50 = np.quantile(curve_array, 0.50, axis=0)
            q75 = np.quantile(curve_array, 0.75, axis=0)
            ax.plot(epochs, q50, color=PATTERN_COLORS[pattern], linewidth=2.0, label=PATTERN_LABELS[pattern])
            ax.fill_between(epochs, q25, q75, color=PATTERN_COLORS[pattern], alpha=0.18)
        ax.set_title(f"layer {layer_idx}")
        ax.set_xlabel("epoch")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("active neurons")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss_trajectories(rows: Sequence[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for pattern in PATTERNS:
        group = [row for row in rows if row["pattern"] == pattern]
        if not group:
            continue
        curves = []
        epochs = None
        for row in group:
            metrics = read_csv(Path(row["run_path"]) / "metrics.csv")
            test = np.array([max(float(metric["test_loss"]), 1e-12) for metric in metrics], dtype=float)
            curves.append(test)
            if epochs is None:
                epochs = np.array([int(metric["epoch"]) for metric in metrics], dtype=int)
        curve_array = np.stack(curves, axis=0)
        q25 = np.quantile(curve_array, 0.25, axis=0)
        q50 = np.quantile(curve_array, 0.50, axis=0)
        q75 = np.quantile(curve_array, 0.75, axis=0)
        ax.plot(epochs, q50, color=PATTERN_COLORS[pattern], linewidth=2.0, label=PATTERN_LABELS[pattern])
        ax.fill_between(epochs, q25, q75, color=PATTERN_COLORS[pattern], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test loss")
    ax.set_title("Phase-stratified test-loss trajectories")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_phase_commit_vs_best_epoch(rows: Sequence[Dict], out_path: Path) -> None:
    ordered_rows = sorted(rows, key=run_sort_value)
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    x = np.arange(len(ordered_rows))

    segment_start = 0
    for pattern in PATTERNS:
        group = [row for row in ordered_rows if row["pattern"] == pattern]
        if not group:
            continue
        segment_end = segment_start + len(group)
        ax.axvspan(
            segment_start - 0.5,
            segment_end - 0.5,
            color=PATTERN_COLORS[pattern],
            alpha=0.08,
            zorder=0,
        )
        ax.text(
            (segment_start + segment_end - 1) / 2,
            1.02,
            PATTERN_LABELS[pattern],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color=PATTERN_COLORS[pattern],
        )
        segment_start = segment_end

    for idx, row in enumerate(ordered_rows):
        event_values = [value for value in [row["first_prune0_epoch"], row["first_prune2_epoch"], row["best_epoch"]] if value >= 0]
        if event_values:
            ax.vlines(idx, min(event_values), max(event_values), color="#b0b0b0", linewidth=1.0, alpha=0.8, zorder=1)
        if row["first_prune0_epoch"] >= 0:
            ax.scatter(
                idx,
                row["first_prune0_epoch"],
                marker="^",
                s=58,
                color="#f58518",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        if row["first_prune2_epoch"] >= 0:
            ax.scatter(
                idx,
                row["first_prune2_epoch"],
                marker="s",
                s=48,
                color="#e45756",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.scatter(
            idx,
            row["best_epoch"],
            marker="o",
            s=40,
            color="#222222",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([run_label(row) for row in ordered_rows], rotation=90, fontsize=7)
    ax.set_ylabel("epoch")
    ax.set_xlabel("run")
    ax.set_title("Phase commit vs best epoch")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="^", linestyle="", markersize=8, color="#f58518", label="first prune0 epoch"),
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, color="#e45756", label="first prune2 epoch"),
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color="#222222", label="best epoch"),
        ],
        fontsize=8,
        loc="upper left",
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_best_to_final_degradation(rows: Sequence[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _scatter_by_family(ax, rows, "final_minus_best", log_scale=True)
    ax.set_ylabel("final test loss - best test loss")
    ax.set_title("Best-to-final degradation by phase")
    ax.legend(handles=family_legend_handles(), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_best_after_commit_lag(rows: Sequence[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _scatter_by_family(ax, rows, "best_after_commit_lag", log_scale=False)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylabel("best epoch - commit epoch")
    ax.set_title("Best-after-commit lag by phase")
    ax.legend(handles=family_legend_handles(), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_post_best_drift_rate(rows: Sequence[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _scatter_by_family(ax, rows, "post_best_drift_rate", log_scale=False)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylabel("0.5 * L1(share(final)-share(best)) / (final epoch - best epoch)")
    ax.set_title("Post-best drift rate by phase")
    ax.legend(handles=family_legend_handles(), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_active_drift(rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    panels = [
        ("active_1_drift", "layer 1"),
        ("active_2_drift", "layer 2"),
    ]
    for ax, (key, title) in zip(axes, panels):
        _scatter_by_family(ax, rows, key, log_scale=False)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(title)
    axes[0].set_ylabel("active(final) - active(best)")
    axes[1].legend(handles=family_legend_handles(), fontsize=8, loc="upper right")
    fig.suptitle("Active drift from best to final", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_share_drift_after_best(rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharey=True)
    panels = [
        ("share_0_drift", "layer 0"),
        ("share_1_drift", "layer 1"),
        ("share_2_drift", "layer 2"),
    ]
    for ax, (key, title) in zip(axes, panels):
        _scatter_by_family(ax, rows, key, log_scale=False)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(title)
    axes[0].set_ylabel("share(final) - share(best)")
    axes[2].legend(handles=family_legend_handles(), fontsize=8, loc="upper right")
    fig.suptitle("Active-share drift after best", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_degradation_vs_layer2_share_drift(rows: Sequence[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        for pattern in _active_patterns(family_rows):
            group = [row for row in family_rows if row["pattern"] == pattern]
            if not group:
                continue
            ax.scatter(
                [row["share_2_drift"] for row in group],
                [max(row["final_minus_best"], 1e-12) for row in group],
                marker=FAMILY_MARKERS[family],
                s=58,
                color=PATTERN_COLORS[pattern],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.95,
            )
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("layer-2 share(final) - share(best)")
    ax.set_ylabel("final test loss - best test loss")
    ax.set_title("Degradation vs layer-2 share drift")
    ax.grid(True, linestyle="--", alpha=0.3)
    phase_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=PATTERN_COLORS[pattern],
            markeredgecolor="white",
            label=PATTERN_LABELS[pattern],
        )
        for pattern in PATTERNS
        if any(row["pattern"] == pattern for row in rows)
    ]
    family_handles = family_legend_handles()
    legend1 = ax.legend(handles=phase_handles, fontsize=8, loc="lower right", title="phase")
    ax.add_artist(legend1)
    ax.legend(handles=family_handles, fontsize=8, loc="upper left", title="family")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_active_share_dynamics(rows: Sequence[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharex=True, sharey=True)
    for layer_idx, ax in enumerate(axes):
        for pattern in PATTERNS:
            group = [row for row in rows if row["pattern"] == pattern]
            if not group:
                continue
            curves = []
            epochs = None
            for row in group:
                metrics = read_csv(Path(row["run_path"]) / "metrics.csv")
                active = np.array(
                    [
                        [
                            int(metric["active_0"]),
                            int(metric["active_1"]),
                            int(metric["active_2"]),
                        ]
                        for metric in metrics
                    ],
                    dtype=float,
                )
                total_active = np.maximum(active.sum(axis=1, keepdims=True), 1.0)
                share = active[:, layer_idx] / total_active[:, 0]
                curves.append(share)
                if epochs is None:
                    epochs = np.array([int(metric["epoch"]) for metric in metrics], dtype=int)
            curve_array = np.stack(curves, axis=0)
            q25 = np.quantile(curve_array, 0.25, axis=0)
            q50 = np.quantile(curve_array, 0.50, axis=0)
            q75 = np.quantile(curve_array, 0.75, axis=0)
            ax.plot(epochs, q50, color=PATTERN_COLORS[pattern], linewidth=2.0, label=PATTERN_LABELS[pattern])
            ax.fill_between(epochs, q25, q75, color=PATTERN_COLORS[pattern], alpha=0.18)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"layer {layer_idx}")
        ax.set_xlabel("epoch")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("active share")
    axes[2].legend(fontsize=8)
    fig.suptitle("Active share dynamics by phase", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_readme(
    pattern_summary: Sequence[Dict],
    factor_summary: Sequence[Dict],
    pairwise_stats: Sequence[Dict],
    out_path: Path,
) -> None:
    summary_by_pattern = {row["pattern"]: row for row in pattern_summary}
    summary_by_factor = {row["factor"]: row for row in factor_summary}
    pairwise_by_metric = {row["metric"]: row for row in pairwise_stats}
    lines = [
        "# Structural Phase Effects Study",
        "",
        "Data source:",
        "- `former results/3.12results，128/hard/prune_grow_split`",
        "- hard task only, legacy sweep without validation-selected checkpoints",
        "",
        "Structural phases:",
        "- `prune0_only`: pruning in layer 0 only",
        "- `prune0+2`: pruning in layers 0 and 2",
        "- `no_prune`: no pruning in any hidden layer",
        "",
        "Reading order:",
        "1. `primary/phase_assignments_by_random_factor.png`",
        "2. `primary/phase_commit_vs_best_epoch.png`",
        "3. `primary/phase_best_after_commit_lag.png`",
        "4. `primary/phase_post_best_drift_rate.png`",
        "5. `primary/phase_best_to_final_degradation.png`",
        "6. `primary/phase_degradation_vs_layer2_share_drift.png`",
        "7. `primary/phase_active_drift.png`",
        "8. `primary/phase_share_drift_after_best.png`",
        "9. `primary/phase_active_share_dynamics.png`",
        "10. `primary/phase_endpoint_summary.png`",
        "11. `primary/phase_speed_summary.png`",
        "12. `primary/phase_active_dynamics.png`",
        "13. `supplementary/phase_loss_trajectories.png`",
        "",
        "Random-factor evidence:",
    ]
    for factor in FACTOR_ORDER:
        row = summary_by_factor[factor]
        lines.append(
            f"- {FACTOR_LABELS[factor]}: n={row['n']}, "
            f"prune0_only={row['count_prune0_only']}, prune0+2={row['count_prune0+2']}, no_prune={row['count_no_prune']}"
        )
    lines.extend(
        [
            "",
            "Pattern-level endpoint summary:",
        ]
    )

    def epoch_str(value: float) -> str:
        return "N/A" if value < 0 else f"{value:.0f}"

    for pattern in PATTERNS:
        row = summary_by_pattern[pattern]
        lines.append(
            f"- {pattern}: n={row['n']}, median best={row['median_best_test_loss']:.6g}, "
            f"median final={row['median_final_test_loss']:.6g}, "
            f"median final-best={row['median_final_minus_best']:.6g}, "
            f"median final gap={row['median_final_test_train_gap']:.6g}"
        )
        lines.append(
            f"  median commit={epoch_str(row['median_commit_epoch'])}, "
            f"best-after-commit lag={epoch_str(row['median_best_after_commit_lag'])}, "
            f"first prune epochs=(p0:{epoch_str(row['median_first_prune0_epoch'])}, "
            f"p2:{epoch_str(row['median_first_prune2_epoch'])}), "
            f"median best epoch={epoch_str(row['median_best_epoch'])}"
        )
        lines.append(
            f"  median reach_5e-4={row['median_reach_5e_4']:.0f}, "
            f"median active@best=({row['median_active_0_best']:.1f}, {row['median_active_1_best']:.1f}, {row['median_active_2_best']:.1f}), "
            f"median active drift=({row['median_active_0_drift']:.1f}, {row['median_active_1_drift']:.1f}, {row['median_active_2_drift']:.1f})"
        )
        lines.append(
            f"  median share drift=({row['median_share_0_drift']:.4f}, {row['median_share_1_drift']:.4f}, {row['median_share_2_drift']:.4f})"
        )
    final_loss_stats = pairwise_by_metric["final_test_loss"]
    best_loss_stats = pairwise_by_metric["best_test_loss"]
    lag_stats = pairwise_by_metric["best_after_commit_lag"]
    drift_rate_stats = pairwise_by_metric["post_best_drift_rate"]
    gap_stats = pairwise_by_metric["final_test_train_gap"]
    active_l1_stats = pairwise_by_metric["active_1_best"]
    active_l2_stats = pairwise_by_metric["active_2_best"]
    active_l1_drift_stats = pairwise_by_metric["active_1_drift"]
    active_l2_drift_stats = pairwise_by_metric["active_2_drift"]
    share_l1_drift_stats = pairwise_by_metric["share_1_drift"]
    share_l2_drift_stats = pairwise_by_metric["share_2_drift"]
    lines.extend(
        [
            "",
            "Key pairwise contrasts (`prune0_only - prune0+2`, median-based):",
            f"- best test loss diff = {best_loss_stats['median_diff_a_minus_b']:.6g} "
            f"(p={best_loss_stats['permutation_pvalue']:.4f})",
            f"- final test loss diff = {final_loss_stats['median_diff_a_minus_b']:.6g} "
            f"(p={final_loss_stats['permutation_pvalue']:.4f})",
            f"- best-after-commit lag diff = {lag_stats['median_diff_a_minus_b']:.0f} "
            f"(p={lag_stats['permutation_pvalue']:.4f})",
            f"- post-best drift rate diff = {drift_rate_stats['median_diff_a_minus_b']:.6g} "
            f"(p={drift_rate_stats['permutation_pvalue']:.4f})",
            f"- final test-train gap diff = {gap_stats['median_diff_a_minus_b']:.6g} "
            f"(p={gap_stats['permutation_pvalue']:.4f})",
            f"- active layer-1 @ best diff = {active_l1_stats['median_diff_a_minus_b']:.3g} "
            f"(p={active_l1_stats['permutation_pvalue']:.4f})",
            f"- active layer-2 @ best diff = {active_l2_stats['median_diff_a_minus_b']:.3g} "
            f"(p={active_l2_stats['permutation_pvalue']:.4f})",
            f"- active layer-1 drift diff = {active_l1_drift_stats['median_diff_a_minus_b']:.3g} "
            f"(p={active_l1_drift_stats['permutation_pvalue']:.4f})",
            f"- active layer-2 drift diff = {active_l2_drift_stats['median_diff_a_minus_b']:.3g} "
            f"(p={active_l2_drift_stats['permutation_pvalue']:.4f})",
            f"- share layer-1 drift diff = {share_l1_drift_stats['median_diff_a_minus_b']:.4f} "
            f"(p={share_l1_drift_stats['permutation_pvalue']:.4f})",
            f"- share layer-2 drift diff = {share_l2_drift_stats['median_diff_a_minus_b']:.4f} "
            f"(p={share_l2_drift_stats['permutation_pvalue']:.4f})",
            "",
            "Interpretation:",
            "- changing only `model_seed` can flip the run between `prune0_only` and `prune0+2`.",
            "- changing only `data_seed` can also flip the phase, so the pattern is not tied to one canonical dataset realization.",
            "- the new commit-vs-best panel shows that phase-defining prune events happen far earlier than the epoch of best performance.",
            "- the new best-after-commit lag makes it explicit how much optimization still happens after the phase is already locked in.",
            "- the new post-best drift rate turns late active-share movement into a scalar reallocation speed rather than a purely visual effect.",
            "- `prune0+2` reaches a slightly better oracle best loss, but it finishes with much worse final loss and a much larger test-train gap.",
            "- the degradation-vs-layer2-share-drift scatter checks directly whether runs that keep moving budget into layer 2 are also the ones that degrade most.",
            "- the new degradation and active-drift panels make the late-phase instability visible directly, not only through endpoint loss.",
            "- the new share-drift panel tests whether layer-wise functional budget keeps moving even after the run has passed its best checkpoint.",
            "- the active-share panel exposes how functional budget redistributes across layers rather than only showing raw active counts.",
            "- `no_prune` appears only once in this dataset and should be treated as a case study, not a statistical group.",
            "",
            "Current limitations:",
            "- no independent `structure_seed` sweep in this legacy source",
            "- no validation loss, so these are diagnostic legacy results rather than paper-final endpoints",
            "- the shared `seed0` anchor is intentionally reused in both the base-seed and data-realization views",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs([OUT_DIR, PRIMARY_DIR, SUPPLEMENTARY_DIR])
    rows = collect_rows()
    view_rows = build_factor_view_rows(rows)
    pattern_summary = summarize_patterns(rows)
    factor_summary = summarize_factors(view_rows)
    pairwise_stats = build_pairwise_stats(rows)

    write_csv(OUT_DIR / "phase_rows.csv", rows)
    write_csv(OUT_DIR / "phase_factor_assignments.csv", view_rows)
    write_csv(OUT_DIR / "phase_summary.csv", pattern_summary)
    write_csv(OUT_DIR / "phase_random_factor_summary.csv", factor_summary)
    write_csv(OUT_DIR / "phase_pairwise_stats.csv", pairwise_stats)

    plot_phase_assignments(view_rows, PRIMARY_DIR / "phase_assignments_by_random_factor.png")
    plot_phase_commit_vs_best_epoch(rows, PRIMARY_DIR / "phase_commit_vs_best_epoch.png")
    plot_best_after_commit_lag(rows, PRIMARY_DIR / "phase_best_after_commit_lag.png")
    plot_post_best_drift_rate(rows, PRIMARY_DIR / "phase_post_best_drift_rate.png")
    plot_best_to_final_degradation(rows, PRIMARY_DIR / "phase_best_to_final_degradation.png")
    plot_degradation_vs_layer2_share_drift(rows, PRIMARY_DIR / "phase_degradation_vs_layer2_share_drift.png")
    plot_active_drift(rows, PRIMARY_DIR / "phase_active_drift.png")
    plot_share_drift_after_best(rows, PRIMARY_DIR / "phase_share_drift_after_best.png")
    plot_active_share_dynamics(rows, PRIMARY_DIR / "phase_active_share_dynamics.png")
    plot_endpoint_panels(rows, PRIMARY_DIR / "phase_endpoint_summary.png")
    plot_speed_panels(rows, PRIMARY_DIR / "phase_speed_summary.png")
    plot_active_trajectories(rows, PRIMARY_DIR / "phase_active_dynamics.png")
    plot_loss_trajectories(rows, SUPPLEMENTARY_DIR / "phase_loss_trajectories.png")
    write_readme(pattern_summary, factor_summary, pairwise_stats, OUT_DIR / "README.md")


if __name__ == "__main__":
    main()
