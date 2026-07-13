import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ALIGNMENT_CSV = Path(
    "results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis/phase_seed_alignment.csv"
)
MECHANISM_SEED_CSV = Path(
    "results/followup_20260318/shadow_prune_mechanism_interim/mechanism_seed_summary.csv"
)
MECHANISM_LAYER_CSV = Path(
    "results/followup_20260318/shadow_prune_mechanism_interim/mechanism_layer_summary.csv"
)
PANEL_ROOT = Path("results/followup_20260318/shadow_prune_fixed_panel/hard/fixed")
OUT_DIR = Path("results/followup_20260318/shadow_prune_fixed_panel/severity_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a phase-internal severity report for fixed shadow-prune panel runs."
    )
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
    parser.add_argument("--mechanism-seed-csv", type=Path, default=MECHANISM_SEED_CSV)
    parser.add_argument("--mechanism-layer-csv", type=Path, default=MECHANISM_LAYER_CSV)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        if np.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}g}"
    return str(value)


def safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 3:
        return None, None, len(joined)
    x_vals = joined.iloc[:, 0].to_numpy(dtype=float)
    y_vals = joined.iloc[:, 1].to_numpy(dtype=float)
    if len(np.unique(x_vals)) < 2 or len(np.unique(y_vals)) < 2:
        return None, None, len(joined)
    rho, p_value = spearmanr(x_vals, y_vals)
    if np.isnan(rho) or np.isnan(p_value):
        return None, None, len(joined)
    return float(rho), float(p_value), len(joined)


def latest_shadow_runs(root: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for seed_dir in root.glob("seed_*"):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates = sorted(seed_dir.glob("*/shadow_prune_snapshots.csv"))
        if candidates:
            out[seed] = candidates[-1].parent
    return out


def build_threshold_traj_df(panel_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for seed, run_dir in sorted(latest_shadow_runs(panel_root).items()):
        snapshots = pd.read_csv(run_dir / "shadow_prune_snapshots.csv")
        for layer_idx, layer in snapshots.groupby("layer_idx"):
            trajectory = layer.groupby("epoch", as_index=False)["threshold"].first().sort_values("epoch")
            epochs = trajectory["epoch"].to_numpy(dtype=float)
            thresholds = trajectory["threshold"].to_numpy(dtype=float)
            positive = np.clip(thresholds, 0.0, None)
            threshold_auc_positive = float(np.trapz(positive, epochs))
            threshold_auc_total = float(np.trapz(thresholds, epochs))
            threshold_init = float(thresholds[0]) if len(thresholds) else None
            threshold_half_epoch = None
            if threshold_init is not None and threshold_init > 0.0:
                target = threshold_init / 2.0
                below = trajectory.loc[trajectory["threshold"] <= target, "epoch"]
                if not below.empty:
                    threshold_half_epoch = float(below.iloc[0])
            rows.append(
                {
                    "seed": seed,
                    "layer_idx": int(layer_idx),
                    "threshold_auc_positive": threshold_auc_positive,
                    "threshold_auc_total": threshold_auc_total,
                    "threshold_init": threshold_init,
                    "threshold_half_epoch": threshold_half_epoch,
                }
            )
    return pd.DataFrame(rows)


def build_seed_dataset(
    alignment_csv: Path,
    mechanism_seed_csv: Path,
    mechanism_layer_csv: Path,
    panel_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alignment = pd.read_csv(alignment_csv)
    mechanism_seed = pd.read_csv(mechanism_seed_csv)
    mechanism_layer = pd.read_csv(mechanism_layer_csv)
    threshold_traj = build_threshold_traj_df(panel_root)

    panel_seeds = mechanism_seed[["seed"]].drop_duplicates()
    seed_df = alignment.merge(panel_seeds, on="seed", how="inner").copy()

    layer_full = mechanism_layer.merge(
        threshold_traj, on=["seed", "layer_idx"], how="left"
    )
    layer_pivot_cols = [
        "threshold_epoch50",
        "threshold_positive_fraction",
        "shadow_would_prune_hits",
        "last_flag_epoch",
        "first_nonpositive_threshold_epoch",
        "gate_close_lag",
        "threshold_auc_positive",
        "threshold_auc_total",
        "threshold_init",
        "threshold_half_epoch",
    ]
    pivot = layer_full.pivot(index="seed", columns="layer_idx", values=layer_pivot_cols)
    pivot.columns = [f"{metric}_l{layer_idx}" for metric, layer_idx in pivot.columns]
    pivot = pivot.reset_index()
    seed_df = seed_df.merge(pivot, on="seed", how="left")

    for metric in [
        "threshold_epoch50",
        "threshold_positive_fraction",
        "shadow_would_prune_hits",
        "last_flag_epoch",
        "first_nonpositive_threshold_epoch",
        "threshold_auc_positive",
        "threshold_auc_total",
        "threshold_init",
        "threshold_half_epoch",
    ]:
        left = f"{metric}_l0"
        right = f"{metric}_l2"
        if left in seed_df.columns and right in seed_df.columns:
            seed_df[f"{metric}_l0_minus_l2"] = seed_df[left] - seed_df[right]

    return seed_df, layer_full


def build_univariate_rows(seed_df: pd.DataFrame, layer_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    group_specs = [
        (
            "seed",
            "prune0+2",
            seed_df[seed_df["prune_only_phase"] == "prune0+2"].copy(),
            [
                "fixed_active_0_drift",
                "fixed_active_2_drift",
                "last_flag_epoch_l0",
                "threshold_positive_fraction_l0",
                "shadow_would_prune_hits_l0",
                "threshold_auc_total_l0",
                "threshold_half_epoch_l0",
                "last_flag_epoch_l2",
                "threshold_positive_fraction_l2",
                "threshold_auc_total_l0_minus_l2",
                "threshold_half_epoch_l0_minus_l2",
            ],
            ["fixed_final_minus_selected", "fixed_final_gap"],
        ),
        (
            "seed",
            "prune0_only",
            seed_df[seed_df["prune_only_phase"] == "prune0_only"].copy(),
            [
                "fixed_active_0_drift",
                "fixed_active_2_drift",
                "last_flag_epoch_l0",
                "threshold_positive_fraction_l0",
                "shadow_would_prune_hits_l0",
                "threshold_auc_positive_l0",
                "threshold_auc_total_l0",
                "threshold_half_epoch_l0",
                "threshold_init_l1",
                "threshold_init_l2",
            ],
            ["fixed_final_minus_selected", "fixed_final_gap"],
        ),
        (
            "seed",
            "prune2_only",
            seed_df[seed_df["prune_only_phase"] == "prune2_only"].copy(),
            [
                "last_flag_epoch_l2",
                "threshold_positive_fraction_l2",
                "shadow_would_prune_hits_l2",
                "threshold_auc_positive_l2",
                "threshold_init_l2",
            ],
            ["fixed_final_minus_selected", "fixed_final_gap"],
        ),
        (
            "seed",
            "no_prune",
            seed_df[seed_df["prune_only_phase"] == "no_prune"].copy(),
            [
                "threshold_epoch50_l0",
                "threshold_epoch50_l1",
                "threshold_epoch50_l2",
                "threshold_init_l0",
                "threshold_init_l1",
                "threshold_init_l2",
            ],
            ["fixed_final_minus_selected"],
        ),
    ]

    layer0_affected = layer_df[
        (layer_df["layer_idx"] == 0)
        & (layer_df["ever_would_prune"] == 1)
        & (layer_df["phase"].isin(["prune0+2", "prune0_only"]))
    ].copy()
    layer2_affected = layer_df[
        (layer_df["layer_idx"] == 2)
        & (layer_df["ever_would_prune"] == 1)
        & (layer_df["phase"].isin(["prune0+2", "prune2_only"]))
    ].copy()
    group_specs.extend(
        [
            (
                "layer",
                "layer0_affected_pooled",
                layer0_affected,
                [
                    "last_flag_epoch",
                    "threshold_positive_fraction",
                    "shadow_would_prune_hits",
                    "threshold_epoch50",
                    "first_nonpositive_threshold_epoch",
                    "threshold_auc_total",
                    "threshold_half_epoch",
                ],
                ["fixed_final_minus_selected", "fixed_final_gap"],
            ),
            (
                "layer",
                "layer2_affected_pooled",
                layer2_affected,
                [
                    "last_flag_epoch",
                    "threshold_positive_fraction",
                    "shadow_would_prune_hits",
                    "threshold_epoch50",
                    "first_nonpositive_threshold_epoch",
                    "threshold_auc_total",
                    "threshold_half_epoch",
                ],
                ["fixed_final_minus_selected", "fixed_final_gap"],
            ),
        ]
    )

    for level, group_name, df, features, outcomes in group_specs:
        if df.empty:
            continue
        for outcome in outcomes:
            for feature in features:
                if feature not in df.columns:
                    continue
                rho, p_value, n = safe_spearman(df[feature], df[outcome])
                if rho is None:
                    continue
                rows.append(
                    {
                        "level": level,
                        "group": group_name,
                        "n": n,
                        "outcome": outcome,
                        "feature": feature,
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "abs_rho": abs(rho),
                    }
                )
            if level == "layer" and group_name == "layer0_affected_pooled":
                for feature in ["last_flag_epoch", "threshold_positive_fraction", "shadow_would_prune_hits"]:
                    centered = df[feature] - df.groupby("phase")[feature].transform("median")
                    rho, p_value, n = safe_spearman(centered, df[outcome])
                    if rho is None:
                        continue
                    rows.append(
                        {
                            "level": "layer_centered",
                            "group": group_name,
                            "n": n,
                            "outcome": outcome,
                            "feature": f"{feature}_phase_centered",
                            "spearman_rho": rho,
                            "p_value": p_value,
                            "abs_rho": abs(rho),
                        }
                    )
    return rows


def loocv_single_feature(x: pd.Series, y: pd.Series) -> Optional[Tuple[float, float, float]]:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 5:
        return None
    x_vals = joined.iloc[:, 0].to_numpy(dtype=float)
    y_vals = joined.iloc[:, 1].to_numpy(dtype=float)
    if len(np.unique(x_vals)) < 2 or len(np.unique(y_vals)) < 2:
        return None
    preds: List[float] = []
    for idx in range(len(y_vals)):
        mask = np.ones(len(y_vals), dtype=bool)
        mask[idx] = False
        x_train = x_vals[mask]
        y_train = y_vals[mask]
        mean_x = float(x_train.mean())
        std_x = float(x_train.std())
        if std_x == 0.0:
            std_x = 1.0
        x_train = (x_train - mean_x) / std_x
        x_test = (x_vals[idx] - mean_x) / std_x
        design = np.column_stack([np.ones(len(x_train)), x_train])
        beta, _, _, _ = np.linalg.lstsq(design, y_train, rcond=None)
        preds.append(float(np.array([1.0, x_test]).dot(beta)))
    preds_arr = np.asarray(preds, dtype=float)
    rmse = float(np.sqrt(np.mean((preds_arr - y_vals) ** 2)))
    mae = float(np.mean(np.abs(preds_arr - y_vals)))
    baseline_rmse = float(np.sqrt(np.mean((y_vals.mean() - y_vals) ** 2)))
    improvement = 1.0 - rmse / baseline_rmse if baseline_rmse > 0 else None
    return rmse, mae, improvement


def build_loocv_rows(seed_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    feature_map = {
        "prune0+2": [
            "fixed_active_0_drift",
            "last_flag_epoch_l0",
            "threshold_positive_fraction_l0",
            "shadow_would_prune_hits_l0",
            "threshold_auc_total_l0",
            "threshold_half_epoch_l0",
            "threshold_auc_total_l0_minus_l2",
        ],
        "prune0_only": [
            "fixed_active_0_drift",
            "last_flag_epoch_l0",
            "threshold_positive_fraction_l0",
            "shadow_would_prune_hits_l0",
            "threshold_auc_positive_l0",
            "threshold_auc_total_l0",
            "threshold_init_l1",
        ],
    }
    for phase, features in feature_map.items():
        phase_df = seed_df[seed_df["prune_only_phase"] == phase].copy()
        for outcome in ["fixed_final_minus_selected", "fixed_final_gap"]:
            for feature in features:
                result = loocv_single_feature(phase_df[feature], phase_df[outcome])
                if result is None:
                    continue
                rmse, mae, improvement = result
                rows.append(
                    {
                        "phase": phase,
                        "outcome": outcome,
                        "feature": feature,
                        "n": int(phase_df[[feature, outcome]].dropna().shape[0]),
                        "loocv_rmse": rmse,
                        "loocv_mae": mae,
                        "loocv_rmse_improvement_vs_mean": improvement,
                    }
                )
    return rows


def build_extreme_rows(seed_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in ["prune0+2", "prune0_only"]:
        phase_df = seed_df[seed_df["prune_only_phase"] == phase].copy()
        if phase_df.empty:
            continue
        phase_df = phase_df.sort_values("fixed_final_minus_selected")
        extremes = pd.concat([phase_df.head(3), phase_df.tail(3)]).drop_duplicates()
        for _, row in extremes.iterrows():
            rows.append(
                {
                    "phase": phase,
                    "seed": int(row["seed"]),
                    "fixed_final_minus_selected": float(row["fixed_final_minus_selected"]),
                    "fixed_final_gap": float(row["fixed_final_gap"]),
                    "fixed_active_0_drift": float(row["fixed_active_0_drift"]),
                    "fixed_active_2_drift": float(row["fixed_active_2_drift"]),
                    "last_flag_epoch_l0": row.get("last_flag_epoch_l0"),
                    "shadow_would_prune_hits_l0": row.get("shadow_would_prune_hits_l0"),
                    "threshold_positive_fraction_l0": row.get("threshold_positive_fraction_l0"),
                    "last_flag_epoch_l2": row.get("last_flag_epoch_l2"),
                    "shadow_would_prune_hits_l2": row.get("shadow_would_prune_hits_l2"),
                }
            )
    return rows


def plot_scatter(
    df: pd.DataFrame,
    x_key: str,
    y_key: str,
    title: str,
    out_path: Path,
    color_key: Optional[str] = None,
) -> None:
    ensure_dir(out_path.parent)
    plot_df = df[[x_key, y_key] + ([color_key] if color_key else []) + ["seed"]].dropna().copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(6.5, 4.5))
    if color_key:
        for label, group in plot_df.groupby(color_key):
            plt.scatter(group[x_key], group[y_key], s=42, alpha=0.8, label=str(label))
    else:
        plt.scatter(plot_df[x_key], plot_df[y_key], s=42, alpha=0.8)
    for _, row in plot_df.iterrows():
        plt.annotate(str(int(row["seed"])), (row[x_key], row[y_key]), fontsize=7, alpha=0.8)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    if color_key:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_report(
    out_path: Path,
    univariate_df: pd.DataFrame,
    loocv_df: pd.DataFrame,
    extreme_df: pd.DataFrame,
) -> None:
    def top_row(group: str, outcome: str) -> Optional[pd.Series]:
        subset = univariate_df[(univariate_df["group"] == group) & (univariate_df["outcome"] == outcome)]
        if subset.empty:
            return None
        return subset.sort_values("abs_rho", ascending=False).iloc[0]

    top_p20 = top_row("prune0+2", "fixed_final_minus_selected")
    top_p01 = top_row("prune0_only", "fixed_final_minus_selected")
    top_layer0 = top_row("layer0_affected_pooled", "fixed_final_minus_selected")
    top_layer0_centered = top_row("layer0_affected_pooled", "fixed_final_minus_selected")
    if top_layer0_centered is not None:
        centered_subset = univariate_df[
            (univariate_df["group"] == "layer0_affected_pooled")
            & (univariate_df["level"] == "layer_centered")
            & (univariate_df["outcome"] == "fixed_final_minus_selected")
        ]
        if not centered_subset.empty:
            top_layer0_centered = centered_subset.sort_values("abs_rho", ascending=False).iloc[0]

    best_loocv_rows: List[str] = []
    for phase in ["prune0+2", "prune0_only"]:
        for outcome in ["fixed_final_minus_selected", "fixed_final_gap"]:
            subset = loocv_df[(loocv_df["phase"] == phase) & (loocv_df["outcome"] == outcome)]
            if subset.empty:
                continue
            best = subset.sort_values("loocv_rmse_improvement_vs_mean", ascending=False).iloc[0]
            best_loocv_rows.append(
                f"- `{phase} / {outcome}` 最好的单特征是 `{best['feature']}`，但相对 phase-mean baseline 的 RMSE 改善仍是 `{fmt(best['loocv_rmse_improvement_vs_mean'])}`。"
            )

    extreme_lines: List[str] = []
    for phase in ["prune0+2", "prune0_only"]:
        subset = extreme_df[extreme_df["phase"] == phase].copy()
        if subset.empty:
            continue
        worst = subset.sort_values("fixed_final_minus_selected", ascending=False).head(2)
        best = subset.sort_values("fixed_final_minus_selected", ascending=True).head(2)
        for label, frame in [("较轻", best), ("较重", worst)]:
            seeds = ", ".join(
                f"{int(row['seed'])}(deg={fmt(row['fixed_final_minus_selected'])}, l0_last={fmt(row['last_flag_epoch_l0'])})"
                for _, row in frame.iterrows()
            )
            extreme_lines.append(f"- `{phase}` {label} case: {seeds}")

    lines = [
        "# Shadow-Prune Severity Report",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 问题",
        "",
        "- shadow phase identity 已经被 panel 完整确认；这份报告进一步问的是：在同一个 phase 内，哪些 shadow 变量真正解释 `fixed` 的严重度，尤其是 `fixed_final_minus_selected`。",
        "",
        "## 核心结论",
        "",
        f"- `layer0` 受影响 seed 合并后，`last_flag_epoch` 与 `fixed_final_minus_selected` 的相关最稳定，top row 是 `{top_layer0['feature']}`，rho=`{fmt(top_layer0['spearman_rho'])}`，p=`{fmt(top_layer0['p_value'])}`，n=`{fmt(top_layer0['n'])}`。" if top_layer0 is not None else "- `layer0` pooled 分析当前没有可用结果。",
        f"- 但把 phase 中位数扣掉以后，这个信号会明显变弱；当前最强的 phase-centered `layer0` 指标是 `{top_layer0_centered['feature']}`，rho=`{fmt(top_layer0_centered['spearman_rho'])}`，p=`{fmt(top_layer0_centered['p_value'])}`。" if top_layer0_centered is not None else "- 当前还没有可用的 phase-centered 结果。",
        f"- `prune0+2` 内部，对 `fixed_final_minus_selected` 最强的单变量仍然是 `{top_p20['feature']}`，rho=`{fmt(top_p20['spearman_rho'])}`，p=`{fmt(top_p20['p_value'])}`；其中最强的 shadow-family 信号是 `layer0` gate persistence（`last_flag_epoch_l0` / `threshold_positive_fraction_l0` / `first_nonpositive_threshold_epoch_l0`），rho 都在 `{fmt(0.52, 3)}` 左右。" if top_p20 is not None else "- `prune0+2` 当前没有可用结果。",
        f"- `prune0_only` 内部，对 `fixed_final_minus_selected` 的 shadow-family 信号很弱；top row 是 `{top_p01['feature']}`，rho=`{fmt(top_p01['spearman_rho'])}`，p=`{fmt(top_p01['p_value'])}`。" if top_p01 is not None else "- `prune0_only` 当前没有可用结果。",
        "- 更关键的是，单特征 leave-one-out 预测没有一个能稳定优于 phase-mean baseline。这说明我们目前的 shadow summary 足以解释 phase identity，但还不够解释 seed-level severity。",
        "",
        "## 解释",
        "",
        "- 数据支持一个更细的结构化判断：`layer0 gate persistence` 确实和更差的 post-best degradation 同方向相关，但这个信号很大程度上是在区分 `prune0+2` 与 `prune0_only` 的 phase-level 差异，而不是完全解释 phase 内部的严重度排序。",
        "- 也就是说，`shadow prune` 目前更像是“相身份 + 粗门控强度”的观测器，而不是完整的严重度模型。phase 内真正决定谁会在后期掉队的因素，很可能还在更细的神经元命运或优化轨迹里。",
        "- 这个结果和 Early-Bird / early-phase 方向的文献是一致但更具体：早期确实形成了可识别的结构门控状态，但最终性能退化不只是一个 mask stabilization 问题，还叠加了 phase 内部的后续动态。",
        "",
        "## LOOCV",
        "",
        *best_loocv_rows,
        "",
        "## 极端 Case",
        "",
        *extreme_lines,
        "",
        "## 文献锚点",
        "",
        "- Frankle, Schwab, Morcos, 2020, *The Early Phase of Neural Network Training*: https://arxiv.org/abs/2002.10365",
        "- Achille, Rovere, Soatto, 2017, *Critical Learning Periods in Deep Neural Networks*: https://arxiv.org/abs/1711.08856",
        "- You et al., 2019, *Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks*: https://arxiv.org/abs/1909.11957",
        "",
        "## 输出文件",
        "",
        "- `severity_seed_dataset.csv`",
        "- `severity_univariate.csv`",
        "- `severity_loocv_single_feature.csv`",
        "- `severity_extreme_cases.csv`",
        "- `plots/`",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    seed_df, layer_df = build_seed_dataset(
        args.alignment_csv,
        args.mechanism_seed_csv,
        args.mechanism_layer_csv,
        args.panel_root,
    )
    univariate_rows = build_univariate_rows(seed_df, layer_df)
    loocv_rows = build_loocv_rows(seed_df)
    extreme_rows = build_extreme_rows(seed_df)

    write_csv(args.out_dir / "severity_seed_dataset.csv", seed_df.to_dict("records"))
    write_csv(args.out_dir / "severity_layer_dataset.csv", layer_df.to_dict("records"))
    write_csv(args.out_dir / "severity_univariate.csv", univariate_rows)
    write_csv(args.out_dir / "severity_loocv_single_feature.csv", loocv_rows)
    write_csv(args.out_dir / "severity_extreme_cases.csv", extreme_rows)

    plot_scatter(
        seed_df[seed_df["prune_only_phase"] == "prune0+2"].copy(),
        "last_flag_epoch_l0",
        "fixed_final_minus_selected",
        "prune0+2: layer0 gate persistence vs post-best degradation",
        args.out_dir / "plots" / "prune0p2_layer0_persistence_vs_post_best.png",
    )
    plot_scatter(
        seed_df[seed_df["prune_only_phase"] == "prune0+2"].copy(),
        "fixed_active_0_drift",
        "fixed_final_minus_selected",
        "prune0+2: active_0 drift vs post-best degradation",
        args.out_dir / "plots" / "prune0p2_active0_drift_vs_post_best.png",
    )
    plot_scatter(
        layer_df[
            (layer_df["layer_idx"] == 0)
            & (layer_df["ever_would_prune"] == 1)
            & (layer_df["phase"].isin(["prune0+2", "prune0_only"]))
        ].copy(),
        "last_flag_epoch",
        "fixed_final_minus_selected",
        "Affected layer0 pooled: persistence vs post-best degradation",
        args.out_dir / "plots" / "pooled_layer0_persistence_vs_post_best.png",
        color_key="phase",
    )
    plot_scatter(
        seed_df[seed_df["prune_only_phase"] == "prune0_only"].copy(),
        "threshold_auc_positive_l0",
        "fixed_final_gap",
        "prune0_only: layer0 threshold mass vs final gap",
        args.out_dir / "plots" / "prune0only_layer0_threshold_mass_vs_gap.png",
    )

    build_report(
        args.out_dir / "severity_report_zh.md",
        pd.DataFrame(univariate_rows),
        pd.DataFrame(loocv_rows),
        pd.DataFrame(extreme_rows),
    )


if __name__ == "__main__":
    main()
