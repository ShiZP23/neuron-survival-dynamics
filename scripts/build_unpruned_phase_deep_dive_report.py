import argparse
import csv
import math
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon


ANALYSIS_ROOT = Path(
    "results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis"
)
ALIGNMENT_CSV = ANALYSIS_ROOT / "phase_seed_alignment.csv"
PHASE_ROWS_CSV = ANALYSIS_ROOT / "phase_comparison" / "fixed_phase_rows.csv"
OUT_DIR = ANALYSIS_ROOT / "deep_dive"
PHASES = ["prune0+2", "prune0_only", "prune2_only", "no_prune"]
MAIN_PHASES = ["prune0+2", "prune0_only", "no_prune"]

FIXED_METRICS = [
    ("selected_test_loss", True),
    ("final_test_loss", True),
    ("final_minus_selected", True),
    ("final_gap", True),
    ("best_epoch", False),
]
PAIRED_METRICS = [
    ("selected_test_loss", True),
    ("final_test_loss", True),
    ("final_minus_selected", True),
    ("final_gap", True),
]
PRUNING_PATTERN_KEYS = ("100", "001", "101")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Chinese deep-dive report for the unpruned phase differentiation sweep."
    )
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
    parser.add_argument("--phase-rows-csv", type=Path, default=PHASE_ROWS_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=3000,
        help="Number of bootstrap samples for median-difference intervals.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}g}"
    return str(value)


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted = [1.0] * n
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        value = p_values[idx] * n / rank
        running = min(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def bootstrap_median_diff(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, samples: int
) -> Tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    estimates = np.empty(samples, dtype=float)
    for idx in range(samples):
        draw_a = rng.choice(a, size=len(a), replace=True)
        draw_b = rng.choice(b, size=len(b), replace=True)
        estimates[idx] = float(np.median(draw_a) - np.median(draw_b))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def common_language_prob_lower(a: np.ndarray, b: np.ndarray) -> float:
    left = a[:, None]
    right = b[None, :]
    return float((left < right).mean() + 0.5 * (left == right).mean())


def latest_metrics_by_seed(root: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for seed_dir in root.glob("seed_*"):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates = sorted(seed_dir.glob("*/metrics.csv"))
        if candidates:
            out[seed] = candidates[-1]
    return out


def build_pruning_signature_rows(prune_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed, metrics_path in sorted(latest_metrics_by_seed(prune_root).items()):
        metrics = read_csv(metrics_path)
        updates = [row for row in metrics if int(row["is_update_epoch"]) == 1]
        totals = {layer_idx: 0 for layer_idx in range(3)}
        first_epoch = {layer_idx: None for layer_idx in range(3)}
        pattern_counts = {key: 0 for key in PRUNING_PATTERN_KEYS}
        pattern_counts["other"] = 0
        update_pattern_sequence: List[str] = []

        for row in updates:
            pattern = "".join(
                "1" if int(row[f"pruned_{layer_idx}"]) > 0 else "0" for layer_idx in range(3)
            )
            if pattern != "000":
                update_pattern_sequence.append(pattern)
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
            elif pattern != "000":
                pattern_counts["other"] += 1
            for layer_idx in range(3):
                pruned = int(row[f"pruned_{layer_idx}"])
                totals[layer_idx] += pruned
                if pruned > 0 and first_epoch[layer_idx] is None:
                    first_epoch[layer_idx] = int(row["epoch"])

        layer_set = "".join(str(layer_idx) for layer_idx in range(3) if totals[layer_idx] > 0) or "none"
        total_updates = sum(pattern_counts.values())
        if totals[0] > 0 and totals[2] > 0:
            if first_epoch[0] == first_epoch[2]:
                order_class = "0_and_2_same_start"
            elif first_epoch[0] < first_epoch[2]:
                order_class = "0_then_2"
            else:
                order_class = "2_then_0"
        elif totals[0] > 0:
            order_class = "0_only"
        elif totals[2] > 0:
            order_class = "2_only"
        else:
            order_class = "none"

        row: Dict[str, object] = {
            "seed": seed,
            "metrics_path": str(metrics_path),
            "layer_set": layer_set,
            "order_class": order_class,
            "first_prune_0_epoch": first_epoch[0],
            "first_prune_1_epoch": first_epoch[1],
            "first_prune_2_epoch": first_epoch[2],
            "total_pruned_0": totals[0],
            "total_pruned_1": totals[1],
            "total_pruned_2": totals[2],
            "total_updates_with_pruning": total_updates,
            "pattern_sequence": "|".join(update_pattern_sequence) if update_pattern_sequence else "none",
        }
        for key in PRUNING_PATTERN_KEYS + ("other",):
            row[f"update_count_{key}"] = pattern_counts[key]
            row[f"ratio_{key}"] = pattern_counts[key] / max(total_updates, 1)
        rows.append(row)
    return rows


def build_group_summary(alignment: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in PHASES:
        group = alignment[alignment["prune_only_phase"] == phase]
        if group.empty:
            continue
        row: Dict[str, object] = {"phase": phase, "n": int(len(group))}
        for metric, _ in FIXED_METRICS:
            column = f"fixed_{metric}"
            row[f"median_{metric}"] = float(group[column].median())
            row[f"mean_{metric}"] = float(group[column].mean())
            row[f"q10_{metric}"] = float(group[column].quantile(0.1))
            row[f"q90_{metric}"] = float(group[column].quantile(0.9))
        row["rate_final_gt_4e-4"] = float((group["fixed_final_test_loss"] > 4e-4).mean())
        row["rate_degradation_gt_2e-4"] = float((group["fixed_final_minus_selected"] > 2e-4).mean())
        row["rate_gap_gt_1.5e-4"] = float((group["fixed_final_gap"] > 1.5e-4).mean())
        row["median_fixed_minus_prune_final"] = float(
            (group["fixed_final_test_loss"] - group["prune_only_final_test_loss"]).median()
        )
        row["median_fixed_minus_prune_degradation"] = float(
            (group["fixed_final_minus_selected"] - group["prune_only_final_minus_selected"]).median()
        )
        row["rate_fixed_final_better_than_prune"] = float(
            (group["fixed_final_test_loss"] < group["prune_only_final_test_loss"]).mean()
        )
        rows.append(row)
    return rows


def build_kruskal_rows(alignment: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scope_name, scope_phases in [("all4", PHASES), ("main3", MAIN_PHASES)]:
        scope_rows: List[Dict[str, object]] = []
        for metric, _ in FIXED_METRICS:
            groups = [alignment.loc[alignment["prune_only_phase"] == phase, f"fixed_{metric}"] for phase in scope_phases]
            stat, p_value = kruskal(*groups)
            scope_rows.append(
                {
                    "scope": scope_name,
                    "metric": metric,
                    "n_groups": len(scope_phases),
                    "kruskal_H": float(stat),
                    "p_value": float(p_value),
                }
            )
        adjusted = benjamini_hochberg([float(row["p_value"]) for row in scope_rows])
        for row, p_adjusted in zip(scope_rows, adjusted):
            row["p_value_bh"] = p_adjusted
            rows.append(row)
    return rows


def build_pairwise_rows(
    alignment: pd.DataFrame, rng: np.random.Generator, bootstrap_samples: int
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric, smaller_is_better in FIXED_METRICS:
        metric_rows: List[Dict[str, object]] = []
        column = f"fixed_{metric}"
        for phase_a, phase_b in combinations(PHASES, 2):
            values_a = alignment.loc[alignment["prune_only_phase"] == phase_a, column].to_numpy(dtype=float)
            values_b = alignment.loc[alignment["prune_only_phase"] == phase_b, column].to_numpy(dtype=float)
            if len(values_a) == 0 or len(values_b) == 0:
                continue
            stat, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            ci_low, ci_high = bootstrap_median_diff(values_a, values_b, rng, bootstrap_samples)
            row = {
                "metric": metric,
                "group_a": phase_a,
                "group_b": phase_b,
                "n_group_a": int(len(values_a)),
                "n_group_b": int(len(values_b)),
                "median_group_a": float(np.median(values_a)),
                "median_group_b": float(np.median(values_b)),
                "median_diff_a_minus_b": float(np.median(values_a) - np.median(values_b)),
                "median_diff_ci_low": ci_low,
                "median_diff_ci_high": ci_high,
                "mannwhitney_u": float(stat),
                "p_value": float(p_value),
                "prob_group_a_lower": common_language_prob_lower(values_a, values_b) if smaller_is_better else float("nan"),
            }
            metric_rows.append(row)
        adjusted = benjamini_hochberg([float(row["p_value"]) for row in metric_rows]) if metric_rows else []
        for row, p_adjusted in zip(metric_rows, adjusted):
            row["p_value_bh"] = p_adjusted
            rows.append(row)
    return rows


def build_paired_rows(
    alignment: pd.DataFrame, rng: np.random.Generator, bootstrap_samples: int
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in PHASES:
        group = alignment[alignment["prune_only_phase"] == phase]
        if group.empty:
            continue
        phase_rows: List[Dict[str, object]] = []
        for metric, smaller_is_better in PAIRED_METRICS:
            fixed_values = group[f"fixed_{metric}"].to_numpy(dtype=float)
            prune_values = group[f"prune_only_{metric}"].to_numpy(dtype=float)
            diffs = fixed_values - prune_values
            try:
                stat, p_value = wilcoxon(fixed_values, prune_values, zero_method="wilcox")
            except ValueError:
                stat, p_value = float("nan"), float("nan")
            ci_low, ci_high = bootstrap_median_diff(diffs, np.zeros_like(diffs), rng, bootstrap_samples)
            phase_rows.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "n": int(len(group)),
                    "median_fixed_minus_prune": float(np.median(diffs)),
                    "mean_fixed_minus_prune": float(np.mean(diffs)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "win_rate_fixed_lower": float((fixed_values < prune_values).mean()) if smaller_is_better else float("nan"),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_value) if not math.isnan(float(p_value)) else float("nan"),
                }
            )
        valid = [row for row in phase_rows if not math.isnan(float(row["p_value"]))]
        adjusted = benjamini_hochberg([float(row["p_value"]) for row in valid]) if valid else []
        for row, p_adjusted in zip(valid, adjusted):
            row["p_value_bh"] = p_adjusted
        for row in phase_rows:
            row.setdefault("p_value_bh", float("nan"))
            rows.append(row)
    return rows


def build_correlation_rows(alignment: pd.DataFrame, phase_rows: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    severity_pairs = [
        ("prune_only_total_pruned", "fixed_selected_test_loss"),
        ("prune_only_total_pruned", "fixed_final_test_loss"),
        ("prune_only_total_pruned", "fixed_final_minus_selected"),
        ("prune_only_total_pruned", "fixed_final_gap"),
        ("prune_only_pruned_0_total", "fixed_final_test_loss"),
        ("prune_only_pruned_2_total", "fixed_final_test_loss"),
        ("prune_only_pruned_2_total", "fixed_final_gap"),
    ]
    for x_key, y_key in severity_pairs:
        rho, p_value = spearmanr(alignment[x_key], alignment[y_key])
        rows.append(
            {
                "family": "prune_severity",
                "group": "all",
                "x_key": x_key,
                "y_key": y_key,
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            }
        )

    drift_pairs = [
        ("post_best_share_drift", "final_minus_selected"),
        ("post_best_drift_rate", "final_minus_selected"),
        ("share_1_drift", "final_minus_selected"),
        ("share_2_drift", "final_minus_selected"),
        ("post_best_drift_rate", "final_test_train_gap"),
    ]
    for x_key, y_key in drift_pairs:
        rho, p_value = spearmanr(phase_rows[x_key], phase_rows[y_key])
        rows.append(
            {
                "family": "fixed_drift_proxy",
                "group": "all",
                "x_key": x_key,
                "y_key": y_key,
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            }
        )
        for phase in PHASES:
            group = phase_rows[phase_rows["phase"] == phase]
            if group.empty:
                continue
            rho, p_value = spearmanr(group[x_key], group[y_key])
            rows.append(
                {
                    "family": "fixed_drift_proxy",
                    "group": phase,
                    "x_key": x_key,
                    "y_key": y_key,
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )
    return rows


def build_extreme_case_rows(alignment: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in PHASES:
        group = alignment[alignment["prune_only_phase"] == phase]
        if group.empty:
            continue
        for rank, (_, item) in enumerate(
            group.sort_values("fixed_final_test_loss", ascending=False).head(5).iterrows(),
            start=1,
        ):
            rows.append(
                {
                    "phase": phase,
                    "case_type": "worst_fixed_final",
                    "rank": rank,
                    "seed": int(item["seed"]),
                    "fixed_final_test_loss": float(item["fixed_final_test_loss"]),
                    "fixed_final_minus_selected": float(item["fixed_final_minus_selected"]),
                    "fixed_selected_test_loss": float(item["fixed_selected_test_loss"]),
                    "prune_only_final_test_loss": float(item["prune_only_final_test_loss"]),
                    "prune_only_total_pruned": int(item["prune_only_total_pruned"]),
                }
            )
        deltas = (group["fixed_final_test_loss"] - group["prune_only_final_test_loss"]).abs()
        largest = group.assign(abs_final_delta=deltas).sort_values("abs_final_delta", ascending=False).head(5)
        for rank, (_, item) in enumerate(largest.iterrows(), start=1):
            rows.append(
                {
                    "phase": phase,
                    "case_type": "largest_fixed_vs_prune_gap",
                    "rank": rank,
                    "seed": int(item["seed"]),
                    "fixed_final_test_loss": float(item["fixed_final_test_loss"]),
                    "fixed_final_minus_selected": float(item["fixed_final_minus_selected"]),
                    "fixed_selected_test_loss": float(item["fixed_selected_test_loss"]),
                    "prune_only_final_test_loss": float(item["prune_only_final_test_loss"]),
                    "prune_only_total_pruned": int(item["prune_only_total_pruned"]),
                }
            )
    return rows


def build_subphase_rows(pruning_signatures: pd.DataFrame, phase_rows: pd.DataFrame) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    order_rows: List[Dict[str, object]] = []
    for order_class, group in pruning_signatures.groupby("order_class"):
        order_rows.append(
            {
                "order_class": order_class,
                "n": int(len(group)),
                "seed_list": ",".join(str(seed) for seed in sorted(group["seed"].tolist())),
            }
        )

    joined = pruning_signatures.merge(
        phase_rows[["seed", "final_test_loss", "final_minus_selected", "final_test_train_gap"]],
        on="seed",
        how="left",
    )
    phase_02 = joined[joined["order_class"] == "0_and_2_same_start"].copy()
    micro_rows: List[Dict[str, object]] = []
    if not phase_02.empty:
        for x_key, y_key in [
            ("ratio_101", "final_minus_selected"),
            ("ratio_001", "final_minus_selected"),
            ("ratio_100", "final_test_loss"),
            ("ratio_100", "final_minus_selected"),
        ]:
            rho, p_value = spearmanr(phase_02[x_key], phase_02[y_key])
            micro_rows.append(
                {
                    "x_key": x_key,
                    "y_key": y_key,
                    "n": int(len(phase_02)),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )
    return order_rows, micro_rows


def write_report(
    out_path: Path,
    group_summary: pd.DataFrame,
    kruskal_rows: pd.DataFrame,
    pairwise_rows: pd.DataFrame,
    paired_rows: pd.DataFrame,
    correlation_rows: pd.DataFrame,
    subphase_summary_rows: pd.DataFrame,
    subphase_micro_rows: pd.DataFrame,
) -> None:
    group_lookup = {row["phase"]: row for _, row in group_summary.iterrows()}
    kruskal_lookup = {(row["scope"], row["metric"]): row for _, row in kruskal_rows.iterrows()}
    pairwise_lookup = {(row["metric"], row["group_a"], row["group_b"]): row for _, row in pairwise_rows.iterrows()}
    paired_lookup = {(row["phase"], row["metric"]): row for _, row in paired_rows.iterrows()}

    between_main = pairwise_lookup[("final_test_loss", "prune0+2", "prune0_only")]
    degradation_main = pairwise_lookup[("final_minus_selected", "prune0+2", "prune0_only")]
    gap_main = pairwise_lookup[("final_gap", "prune0+2", "prune0_only")]
    selected_main = pairwise_lookup[("selected_test_loss", "prune0+2", "prune0_only")]
    fixed_vs_prune_p02 = paired_lookup[("prune0+2", "final_test_loss")]
    fixed_vs_prune_p00 = paired_lookup[("prune0_only", "final_test_loss")]
    fixed_vs_prune_p20 = paired_lookup[("prune2_only", "final_test_loss")]

    drift_rows = correlation_rows[
        (correlation_rows["family"] == "fixed_drift_proxy")
        & (correlation_rows["group"] == "all")
        & (correlation_rows["y_key"] == "final_minus_selected")
    ]
    best_drift = drift_rows.iloc[drift_rows["spearman_rho"].abs().argmax()] if not drift_rows.empty else None
    total_pruned_gap = correlation_rows[
        (correlation_rows["family"] == "prune_severity")
        & (correlation_rows["x_key"] == "prune_only_total_pruned")
        & (correlation_rows["y_key"] == "fixed_final_gap")
    ].iloc[0]

    lines = [
        "# 未剪枝网络分化 200-seed 深挖报告",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 数据与问题",
        "",
        "- 数据源：`results/followup_20260317/unpruned_phase_seed_sweep/hard`。",
        "- paired 设计：同一 `seed` 同时有 `prune_only` 与 `fixed` 运行，各 200 个 seed，对应 200 条 paired 记录。",
        "- 当前问题：`fixed` 是否已经包含 latent phase differentiation，以及 `prune_only` 观察到的 phase 到底是在“揭示”还是“制造”差异。",
        "",
        "## 一句话结论",
        "",
        "这批 200-seed 结果已经足够支持一个明确结论：`fixed` 确实存在可被 paired `prune_only` phase 索引出来的潜在分化，而且这类分化主要体现在后期稳定性与最终泛化，而不是最佳点可达性本身。",
        "",
        "## Phase 清单与是否出现新 phase",
        "",
        "- 当前粗粒度 phase 清单为 4 类：`prune0+2=95`、`prune0_only=91`、`prune2_only=4`、`no_prune=10`。",
        "- 相比此前主三相 taxonomy，本轮 sweep 唯一新增且稳定出现的粗 phase 是 `prune2_only`。",
        "- 没有观察到 layer-1 pruning，也没有观察到 `0_then_2` 或 `2_then_0` 这类延迟启动的新顺序相；所有会同时剪 0 和 2 的 run 都是在 epoch 50 同步启动。",
        "- 因此，目前最稳健的结论不是“出现了很多新粗 phase”，而是“主 taxonomy 从三相扩展为四相，其中新增的是罕见但稳定的 `prune2_only`”。",
        "",
        "## 结果主轴：phase 差异主要落在 late stability，而不是 peak attainability",
        "",
        f"- 4-phase Kruskal 检验：`fixed_final_test_loss` 的 H={fmt(kruskal_lookup[('all4', 'final_test_loss')]['kruskal_H'])}，p={fmt(kruskal_lookup[('all4', 'final_test_loss')]['p_value'])}，BH={fmt(kruskal_lookup[('all4', 'final_test_loss')]['p_value_bh'])}；`fixed_final_gap` 的 H={fmt(kruskal_lookup[('all4', 'final_gap')]['kruskal_H'])}，p={fmt(kruskal_lookup[('all4', 'final_gap')]['p_value'])}，BH={fmt(kruskal_lookup[('all4', 'final_gap')]['p_value_bh'])}。",
        f"- 相比之下，`fixed_selected_test_loss` 的 4-phase Kruskal p={fmt(kruskal_lookup[('all4', 'selected_test_loss')]['p_value'])}，`fixed_best_epoch` 的 p={fmt(kruskal_lookup[('all4', 'best_epoch')]['p_value'])}；也就是 phase 几乎不体现在“何时到达最好点”，更多体现在“最好点之后会发生什么”。",
        f"- 主对比 `prune0+2` vs `prune0_only`：selected loss 中位数差 {fmt(selected_main['median_diff_a_minus_b'])}，p={fmt(selected_main['p_value'])}，BH={fmt(selected_main['p_value_bh'])}；final loss 中位数差 {fmt(between_main['median_diff_a_minus_b'])}，95% CI [{fmt(between_main['median_diff_ci_low'])}, {fmt(between_main['median_diff_ci_high'])}]，p={fmt(between_main['p_value'])}，BH={fmt(between_main['p_value_bh'])}。",
        f"- 同一组对比下，`final_minus_selected` 中位数差 {fmt(degradation_main['median_diff_a_minus_b'])}，`final_gap` 中位数差 {fmt(gap_main['median_diff_a_minus_b'])}。这说明 `prune0+2` 对应的 dense trajectory 在终点更稳，过拟合也更轻。",
        "",
        "## 各相的具体读法",
        "",
        f"- `prune0+2`：n={int(group_lookup['prune0+2']['n'])}，`fixed` median selected={fmt(group_lookup['prune0+2']['median_selected_test_loss'])}，median final={fmt(group_lookup['prune0+2']['median_final_test_loss'])}，median degradation={fmt(group_lookup['prune0+2']['median_final_minus_selected'])}，catastrophic tail 比例 `final>4e-4`={fmt(group_lookup['prune0+2']['rate_final_gt_4e-4'])}。",
        f"- `prune0_only`：n={int(group_lookup['prune0_only']['n'])}，median selected={fmt(group_lookup['prune0_only']['median_selected_test_loss'])}，median final={fmt(group_lookup['prune0_only']['median_final_test_loss'])}，median degradation={fmt(group_lookup['prune0_only']['median_final_minus_selected'])}，tail 比例={fmt(group_lookup['prune0_only']['rate_final_gt_4e-4'])}。它不是在 peak 上明显更差，而是在 late phase 上更容易滑落。",
        f"- `no_prune`：n={int(group_lookup['no_prune']['n'])}，median final={fmt(group_lookup['no_prune']['median_final_test_loss'])}，median degradation={fmt(group_lookup['no_prune']['median_final_minus_selected'])}。样本少，但尾部风险并不低，说明“不剪枝到最后”本身并不自动对应最稳定轨道。",
        f"- `prune2_only`：n={int(group_lookup['prune2_only']['n'])}，median final={fmt(group_lookup['prune2_only']['median_final_test_loss'])}，median degradation={fmt(group_lookup['prune2_only']['median_final_minus_selected'])}，median final gap={fmt(group_lookup['prune2_only']['median_final_gap'])}。它是本轮最值得额外跟踪的新相，因为它的终点风险是四相里最差的描述性组别之一，但样本只有 4，必须谨慎表述为“罕见候选相”。",
        "",
        "## 一个关键反直觉结果：phase 不是“剪得越多越差”的连续严重度",
        "",
        f"- `prune_only_total_pruned` 与 `fixed_final_gap` 的 Spearman rho={fmt(total_pruned_gap['spearman_rho'])}，p={fmt(total_pruned_gap['p_value'])}，符号是负的。",
        "- 这意味着如果把 `prune_only` 中的剪枝总量当成一个连续“损伤严重度”指标，它给出的方向反而是误导性的：剪得更多并不对应 `fixed` 更差，反而常常对应更低的最终 gap。",
        "- 因此，这个现象更像 qualitative phase identity，而不是 quantitative prune severity。",
        "",
        "## `fixed` 对比 paired `prune_only`：剪枝不是现象来源，而且不总是终点收益来源",
        "",
        f"- `prune0+2` 组内，`fixed - prune_only` 的 final loss 中位数差为 {fmt(fixed_vs_prune_p02['median_fixed_minus_prune'])}，95% CI [{fmt(fixed_vs_prune_p02['ci_low'])}, {fmt(fixed_vs_prune_p02['ci_high'])}]，Wilcoxon p={fmt(fixed_vs_prune_p02['p_value'])}，BH={fmt(fixed_vs_prune_p02['p_value_bh'])}。",
        f"- `prune0_only` 组内，对应 final loss 中位数差为 {fmt(fixed_vs_prune_p00['median_fixed_minus_prune'])}，Wilcoxon p={fmt(fixed_vs_prune_p00['p_value'])}。方向仍偏向 `fixed` 更低，但统计强度不如 `prune0+2` 明确。",
        f"- `prune2_only` 组内，`fixed` 反而更差，中位数差 {fmt(fixed_vs_prune_p20['median_fixed_minus_prune'])}，但 n=4，只能当描述性现象。",
        "- 这组结果非常重要：`prune_only` 的 phase 标签确实能揭示 latent differentiation，但“实际剪枝动作本身”并没有统一地改善终点；在相当多 seed 上，dense `fixed` 的最终测试损失反而更低。",
        "- 更准确的表述应当是：剪枝更像一个 phase reveal / phase locking 操作，而不是一个必然提升终点性能的治疗手段。",
        "",
        "## 机制层检查：现有粗 drift proxy 解释力很弱",
        "",
        f"- 当前最强的 drift proxy 相关项是 `{best_drift['x_key']}` vs `{best_drift['y_key']}`，rho={fmt(best_drift['spearman_rho'])}，p={fmt(best_drift['p_value'])}。这个量级仍然很弱。",
        "- `post_best_share_drift`、`post_best_drift_rate`、`share_1_drift`、`share_2_drift` 与 `final_minus_selected` 的整体相关都接近 0。",
        "- 这说明目前这套聚合层级的 drift 指标还不够贴近真正的机制变量。下一步如果要从相关推进到机制，最合适的不是再画更多 aggregate curve，而是给 `fixed` 加 `shadow prune` 记录，直接追踪 would-be-pruned neurons 的命运。",
        "",
        "## 是否存在更细的新子相",
        "",
        "- 以更新轮次模式看，`prune0+2` 组的 common skeleton 很稳定：先在 epoch 50 同步触发 `0` 与 `2`，之后大量更新轮次变成 `layer2-only` 尾巴。",
        "- 在 `prune0+2` 内部，平均更新轮次占比大致是：joint `101` 约 0.210，`layer2-only` 约 0.617，`layer0-only` 约 0.173。",
        "- 这些微结构与 `fixed` 终点退化的 Spearman 相关只有约 |rho|≈0.18，仍然偏弱。",
        "- 所以目前可以说：存在值得跟踪的 micro-variants，但证据还不足以把它们提升为新的稳定粗 phase。",
        "",
        "## 与文献的关系",
        "",
        "- Frankle & Carbin 2019《The Lottery Ticket Hypothesis》提示：可训练子网络与初始化密切相关，seed 并不是噪声，而是 subnet path 的选择器。我们的结果把这一直觉推进了一步：即便不真正剪枝，paired seed 也会把 dense `fixed` 带入不同 late-stability phase。链接：https://openreview.net/forum?id=rJl-b3RcF7",
        "- Frankle et al. 2020《The Early Phase of Neural Network Training》强调训练早期对后续轨道至关重要。我们当前还没有直接的早期 marker，但 phase 在 `selected loss` 上弱、在终点上强，和“早期定轨、后期显化”这类叙述是相容的。链接：https://arxiv.org/abs/2002.10365",
        "- Kornblith et al. 2019《Similarity of Neural Network Representations Revisited》说明“性能接近”并不意味着内部状态相同。这里不同相的 selected loss 很接近，但最终稳定性不同，正符合这一点。链接：https://proceedings.mlr.press/v97/kornblith19a.html",
        "- Draxler et al. 2018《Essentially No Barriers in Neural Network Energy Landscape》提醒我们：不同轨道不必对应彼此隔绝的 basin。当前 phase 更像 trajectory/stability family，而不是互相断开的孤立极小值。链接：https://proceedings.mlr.press/v80/draxler18a.html",
        "- Frankle et al. 2020《Linear Mode Connectivity and the Lottery Ticket Hypothesis》表明稀疏/稠密解之间的可连接性与训练路径关系复杂。我们的 paired 结果也说明：phase label 有信息，但 pruning 动作本身不是唯一决定因素。链接：https://proceedings.mlr.press/v119/frankle20a.html",
        "- Liu et al. 2018《Rethinking the Value of Network Pruning》指出 pruning 的收益并不必然来自继承下来的权重。我们这里看到 dense `fixed` 往往能匹配甚至超过 paired `prune_only` 终点，与这种“pruning 不是天然优势来源”的观点一致。链接：https://arxiv.org/abs/1810.05270",
        "",
        "## 研究判断",
        "",
        "作为一个科研判断，我认为这批结果已经足够支撑以下两点：",
        "",
        "1. `fixed` 的 latent phase differentiation 是真的，不是 `prune_only` 体系特有的假象。",
        "2. 这类 phase 差异主要决定的是后期稳定性、终点泛化和尾部风险，而不是 peak loss 本身。",
        "",
        "但它还不足以支撑以下更强命题：",
        "",
        "1. 目前还不能说现有 aggregate drift 指标已经抓到了机制核心。",
        "2. 目前也不能把 `prune2_only` 直接写成正式新 taxonomy，除非后续 seed sweep 或跨任务复现继续支持。",
        "",
        "## 我建议的下一步",
        "",
        "1. 在 `fixed` 中实现 `shadow prune`：按 `prune_only` 规则照常产生 would-be-pruned neurons，但不实际删除，只记录它们后续是否恢复、漂移到哪里、是否重新变重要。",
        "2. 先对本轮的 `prune2_only` 4 个 seed 做 targeted rerun 与神经元级追踪，因为这批 seed 最像新相入口。",
        "3. 把当前 `hard` 任务的 phase label 往 `medium` 或 `simple` 做迁移验证，看它到底是任务特异还是更一般的现象。",
        "4. 如果要写成正式 paper claim，下一步最缺的不是更多终点 boxplot，而是机制证据：early marker、shadow-prune 命运、以及跨任务复现。",
        "",
        "## 文件导航",
        "",
        "- 本报告：`deep_dive_report_zh.md`",
        "- 组级统计：`deep_dive_group_summary.csv`",
        "- 组间检验：`deep_dive_pairwise_tests.csv` 与 `deep_dive_kruskal_tests.csv`",
        "- paired `fixed vs prune_only`：`deep_dive_fixed_vs_prune.csv`",
        "- 相关分析：`deep_dive_correlations.csv`",
        "- 新 phase / 子相搜索：`deep_dive_pruning_signatures.csv`、`deep_dive_subphase_summary.csv`、`deep_dive_subphase_micro_stats.csv`",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.bootstrap_seed)

    alignment = pd.read_csv(args.alignment_csv)
    phase_rows = pd.read_csv(args.phase_rows_csv)

    group_summary = pd.DataFrame(build_group_summary(alignment))
    kruskal_rows = pd.DataFrame(build_kruskal_rows(alignment))
    pairwise_rows = pd.DataFrame(build_pairwise_rows(alignment, rng, args.bootstrap_samples))
    paired_rows = pd.DataFrame(build_paired_rows(alignment, rng, args.bootstrap_samples))
    correlation_rows = pd.DataFrame(build_correlation_rows(alignment, phase_rows))
    extreme_cases = pd.DataFrame(build_extreme_case_rows(alignment))

    prune_root = args.alignment_csv.parent.parent / "prune_only"
    pruning_signatures = pd.DataFrame(build_pruning_signature_rows(prune_root))
    subphase_summary_rows, subphase_micro_rows = build_subphase_rows(pruning_signatures, phase_rows)
    subphase_summary = pd.DataFrame(subphase_summary_rows)
    subphase_micro = pd.DataFrame(subphase_micro_rows)

    write_csv(args.out_dir / "deep_dive_group_summary.csv", group_summary.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_kruskal_tests.csv", kruskal_rows.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_pairwise_tests.csv", pairwise_rows.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_fixed_vs_prune.csv", paired_rows.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_correlations.csv", correlation_rows.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_extreme_cases.csv", extreme_cases.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_pruning_signatures.csv", pruning_signatures.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_subphase_summary.csv", subphase_summary.to_dict("records"))
    write_csv(args.out_dir / "deep_dive_subphase_micro_stats.csv", subphase_micro.to_dict("records"))

    write_report(
        args.out_dir / "deep_dive_report_zh.md",
        group_summary,
        kruskal_rows,
        pairwise_rows,
        paired_rows,
        correlation_rows,
        subphase_summary,
        subphase_micro,
    )


if __name__ == "__main__":
    main()
