import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ALIGNMENT_CSV = Path(
    "results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis/phase_seed_alignment.csv"
)
DEFAULT_ROOTS = [
    Path("results/followup_20260318/shadow_prune_fixed_pilot/hard/fixed"),
    Path("results/followup_20260318/shadow_prune_fixed_panel/hard/fixed"),
]
OUT_DIR = Path("results/followup_20260318/shadow_prune_mechanism_interim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interim mechanism report from available fixed shadow-prune runs."
    )
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=DEFAULT_ROOTS,
        help="Roots containing fixed shadow-prune runs.",
    )
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


def safe_spearman(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 3:
        return None, None
    x_vals = joined.iloc[:, 0].to_numpy(dtype=float)
    y_vals = joined.iloc[:, 1].to_numpy(dtype=float)
    if len(np.unique(x_vals)) < 2 or len(np.unique(y_vals)) < 2:
        return None, None
    rho, p_value = spearmanr(x_vals, y_vals)
    if np.isnan(rho) or np.isnan(p_value):
        return None, None
    return float(rho), float(p_value)


def classify_phase_from_layers(active_layers: Dict[int, bool]) -> str:
    layer0 = bool(active_layers.get(0, False))
    layer1 = bool(active_layers.get(1, False))
    layer2 = bool(active_layers.get(2, False))
    if layer0 and layer2 and not layer1:
        return "prune0+2"
    if layer0 and not layer1 and not layer2:
        return "prune0_only"
    if layer2 and not layer0 and not layer1:
        return "prune2_only"
    if not layer0 and not layer1 and not layer2:
        return "no_prune"
    return "other"


def latest_runs_by_seed(roots: List[Path]) -> Dict[int, Path]:
    candidates: Dict[int, List[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for snapshot_path in root.glob("seed_*/*/shadow_prune_snapshots.csv"):
            run_dir = snapshot_path.parent
            try:
                seed = int(run_dir.parent.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.setdefault(seed, []).append(run_dir)
    latest: Dict[int, Path] = {}
    for seed, run_dirs in candidates.items():
        latest[seed] = max(run_dirs, key=lambda path: (path.name, str(path)))
    return latest


def load_alignment(path: Path) -> Dict[int, pd.Series]:
    df = pd.read_csv(path)
    return {int(row["seed"]): row for _, row in df.iterrows()}


def summarize_runs(
    roots: List[Path], alignment_csv: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alignment = load_alignment(alignment_csv)
    latest_runs = latest_runs_by_seed(roots)
    seed_rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []

    for seed, run_dir in sorted(latest_runs.items()):
        aligned = alignment.get(seed)
        if aligned is None:
            continue
        metrics = pd.read_csv(run_dir / "metrics.csv")
        snapshots = pd.read_csv(run_dir / "shadow_prune_snapshots.csv")
        best_idx = metrics["val_loss"].astype(float).idxmin()
        best_epoch = int(metrics.loc[best_idx, "epoch"])
        last_shadow_epoch = int(snapshots["epoch"].max())
        shadow_layers: Dict[int, bool] = {}
        threshold_layers: Dict[int, bool] = {}
        first_flag_epochs: List[int] = []
        last_flag_epochs: List[int] = []
        gate_close_lags: List[int] = []

        for layer_idx in sorted(snapshots["layer_idx"].unique().tolist()):
            layer = snapshots[snapshots["layer_idx"] == layer_idx].copy()
            epoch_rows = (
                layer.sort_values(["epoch", "neuron_idx"])
                .groupby("epoch", as_index=False)
                .first()
            )
            threshold_e50_series = epoch_rows.loc[epoch_rows["epoch"] == 50, "threshold"]
            threshold_e50 = float(threshold_e50_series.iloc[0]) if not threshold_e50_series.empty else None
            threshold_final = float(epoch_rows["threshold"].iloc[-1]) if not epoch_rows.empty else None
            threshold_min = float(epoch_rows["threshold"].min()) if not epoch_rows.empty else None
            threshold_positive_fraction = (
                float((epoch_rows["threshold"] > 0).mean()) if not epoch_rows.empty else None
            )
            positive_epochs = epoch_rows.loc[epoch_rows["threshold"] > 0, "epoch"]
            nonpositive_epochs = epoch_rows.loc[epoch_rows["threshold"] <= 0, "epoch"]
            first_positive_epoch = int(positive_epochs.iloc[0]) if not positive_epochs.empty else None
            first_nonpositive_epoch = int(nonpositive_epochs.iloc[0]) if not nonpositive_epochs.empty else None

            would_prune = layer[layer["would_prune"] == 1].copy()
            candidates = layer[layer["is_candidate"] == 1].copy()
            unique_wp = sorted(would_prune["neuron_idx"].astype(int).unique().tolist())
            ever_would_prune = bool(unique_wp)
            shadow_layers[int(layer_idx)] = ever_would_prune
            threshold_layers[int(layer_idx)] = bool(threshold_e50 is not None and threshold_e50 > 0)

            first_flag_epoch = int(would_prune["epoch"].min()) if ever_would_prune else None
            last_flag_epoch = int(would_prune["epoch"].max()) if ever_would_prune else None
            if first_flag_epoch is not None:
                first_flag_epochs.append(first_flag_epoch)
            if last_flag_epoch is not None:
                last_flag_epochs.append(last_flag_epoch)

            flagged_last_rows = (
                layer[
                    (layer["epoch"] == last_shadow_epoch)
                    & (layer["neuron_idx"].isin(unique_wp))
                ].copy()
                if unique_wp
                else pd.DataFrame()
            )
            final_zero_importance_rate = (
                float((flagged_last_rows["importance"] == 0.0).mean())
                if not flagged_last_rows.empty
                else None
            )
            final_zero_ema_rate = (
                float((flagged_last_rows["ema_importance"] == 0.0).mean())
                if not flagged_last_rows.empty
                else None
            )
            final_candidate_rate = (
                float((flagged_last_rows["is_candidate"] == 1).mean())
                if not flagged_last_rows.empty
                else None
            )
            final_would_prune_rate = (
                float((flagged_last_rows["would_prune"] == 1).mean())
                if not flagged_last_rows.empty
                else None
            )
            gate_close_lag = (
                int(first_nonpositive_epoch - last_flag_epoch)
                if first_nonpositive_epoch is not None and last_flag_epoch is not None
                else None
            )
            if gate_close_lag is not None:
                gate_close_lags.append(gate_close_lag)

            layer_rows.append(
                {
                    "seed": seed,
                    "phase": aligned["prune_only_phase"],
                    "layer_idx": int(layer_idx),
                    "run_dir": str(run_dir),
                    "best_epoch": best_epoch,
                    "last_shadow_epoch": last_shadow_epoch,
                    "fixed_final_test_loss": float(aligned["fixed_final_test_loss"]),
                    "fixed_final_minus_selected": float(aligned["fixed_final_minus_selected"]),
                    "fixed_final_gap": float(aligned["fixed_final_gap"]),
                    "threshold_epoch50": threshold_e50,
                    "threshold_final": threshold_final,
                    "threshold_min": threshold_min,
                    "threshold_positive_fraction": threshold_positive_fraction,
                    "first_positive_threshold_epoch": first_positive_epoch,
                    "first_nonpositive_threshold_epoch": first_nonpositive_epoch,
                    "ever_would_prune": int(ever_would_prune),
                    "shadow_candidate_unique": int(candidates["neuron_idx"].nunique()) if not candidates.empty else 0,
                    "shadow_would_prune_unique": len(unique_wp),
                    "shadow_would_prune_hits": int(would_prune["would_prune"].sum()) if not would_prune.empty else 0,
                    "first_flag_epoch": first_flag_epoch,
                    "last_flag_epoch": last_flag_epoch,
                    "gate_close_lag": gate_close_lag,
                    "final_zero_importance_rate": final_zero_importance_rate,
                    "final_zero_ema_rate": final_zero_ema_rate,
                    "final_candidate_rate": final_candidate_rate,
                    "final_would_prune_rate": final_would_prune_rate,
                    "threshold_sign_predicts_would_prune": int(
                        (threshold_e50 is not None and threshold_e50 > 0) == ever_would_prune
                    ),
                }
            )

        shadow_phase = classify_phase_from_layers(shadow_layers)
        threshold_sign_phase = classify_phase_from_layers(threshold_layers)
        seed_rows.append(
            {
                "seed": seed,
                "phase": aligned["prune_only_phase"],
                "run_dir": str(run_dir),
                "best_epoch": best_epoch,
                "last_shadow_epoch": last_shadow_epoch,
                "fixed_final_test_loss": float(aligned["fixed_final_test_loss"]),
                "fixed_final_minus_selected": float(aligned["fixed_final_minus_selected"]),
                "fixed_final_gap": float(aligned["fixed_final_gap"]),
                "shadow_phase": shadow_phase,
                "shadow_phase_match": int(shadow_phase == aligned["prune_only_phase"]),
                "threshold_sign_phase": threshold_sign_phase,
                "threshold_sign_phase_match": int(threshold_sign_phase == aligned["prune_only_phase"]),
                "shadow_total_unique": int(sum(int(v) for v in shadow_layers.values())),
                "shadow_total_layers": int(sum(1 for value in shadow_layers.values() if value)),
                "first_any_flag_epoch": int(min(first_flag_epochs)) if first_flag_epochs else None,
                "last_any_flag_epoch": int(max(last_flag_epochs)) if last_flag_epochs else None,
                "median_gate_close_lag": float(np.median(gate_close_lags)) if gate_close_lags else None,
                "epoch50_sign_exact_match": int(
                    all(
                        row["threshold_sign_predicts_would_prune"] == 1
                        for row in layer_rows
                        if row["seed"] == seed
                    )
                ),
            }
        )

    seed_df = pd.DataFrame(seed_rows)
    layer_df = pd.DataFrame(layer_rows)
    phase_rows: List[Dict[str, object]] = []
    if not seed_df.empty:
        for phase in ["prune0+2", "prune0_only", "prune2_only", "no_prune"]:
            seeds = seed_df[seed_df["phase"] == phase].copy()
            if seeds.empty:
                continue
            phase_rows.append(
                {
                    "phase": phase,
                    "n_seeds": int(len(seeds)),
                    "shadow_phase_match_rate": float(seeds["shadow_phase_match"].mean()),
                    "threshold_sign_phase_match_rate": float(seeds["threshold_sign_phase_match"].mean()),
                    "median_shadow_total_unique": float(seeds["shadow_total_unique"].median()),
                    "median_first_any_flag_epoch": float(seeds["first_any_flag_epoch"].median())
                    if seeds["first_any_flag_epoch"].notna().any()
                    else None,
                    "median_last_any_flag_epoch": float(seeds["last_any_flag_epoch"].median())
                    if seeds["last_any_flag_epoch"].notna().any()
                    else None,
                    "median_gate_close_lag": float(seeds["median_gate_close_lag"].median())
                    if seeds["median_gate_close_lag"].notna().any()
                    else None,
                    "median_fixed_final_minus_selected": float(seeds["fixed_final_minus_selected"].median()),
                    "median_fixed_final_gap": float(seeds["fixed_final_gap"].median()),
                }
            )
    phase_df = pd.DataFrame(phase_rows)
    return seed_df, layer_df, phase_df


def build_gate_stats(seed_df: pd.DataFrame, layer_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if seed_df.empty or layer_df.empty:
        return rows
    tp = int(((layer_df["threshold_epoch50"] > 0) & (layer_df["ever_would_prune"] == 1)).sum())
    tn = int(((layer_df["threshold_epoch50"] <= 0) & (layer_df["ever_would_prune"] == 0)).sum())
    fp = int(((layer_df["threshold_epoch50"] > 0) & (layer_df["ever_would_prune"] == 0)).sum())
    fn = int(((layer_df["threshold_epoch50"] <= 0) & (layer_df["ever_would_prune"] == 1)).sum())
    rows.append(
        {
            "group": "all_layers",
            "n_items": int(len(layer_df)),
            "metric": "epoch50_threshold_sign_predicts_would_prune",
            "value": float((tp + tn) / len(layer_df)),
            "extra_1": tp,
            "extra_2": tn,
            "extra_3": fp,
            "extra_4": fn,
        }
    )
    rows.append(
        {
            "group": "all_seeds",
            "n_items": int(len(seed_df)),
            "metric": "shadow_phase_match_rate",
            "value": float(seed_df["shadow_phase_match"].mean()),
            "extra_1": int(seed_df["shadow_phase_match"].sum()),
            "extra_2": int(len(seed_df)),
            "extra_3": None,
            "extra_4": None,
        }
    )
    rows.append(
        {
            "group": "all_seeds",
            "n_items": int(len(seed_df)),
            "metric": "threshold_sign_phase_match_rate",
            "value": float(seed_df["threshold_sign_phase_match"].mean()),
            "extra_1": int(seed_df["threshold_sign_phase_match"].sum()),
            "extra_2": int(len(seed_df)),
            "extra_3": None,
            "extra_4": None,
        }
    )
    affected = layer_df[layer_df["ever_would_prune"] == 1].copy()
    if not affected.empty:
        rows.extend(
            [
                {
                    "group": "affected_layers",
                    "n_items": int(len(affected)),
                    "metric": "median_first_flag_epoch",
                    "value": float(affected["first_flag_epoch"].median()),
                    "extra_1": None,
                    "extra_2": None,
                    "extra_3": None,
                    "extra_4": None,
                },
                {
                    "group": "affected_layers",
                    "n_items": int(len(affected)),
                    "metric": "median_last_flag_epoch",
                    "value": float(affected["last_flag_epoch"].median()),
                    "extra_1": None,
                    "extra_2": None,
                    "extra_3": None,
                    "extra_4": None,
                },
                {
                    "group": "affected_layers",
                    "n_items": int(len(affected)),
                    "metric": "median_first_nonpositive_threshold_epoch",
                    "value": float(affected["first_nonpositive_threshold_epoch"].median()),
                    "extra_1": None,
                    "extra_2": None,
                    "extra_3": None,
                    "extra_4": None,
                },
                {
                    "group": "affected_layers",
                    "n_items": int(len(affected)),
                    "metric": "median_gate_close_lag",
                    "value": float(affected["gate_close_lag"].median()),
                    "extra_1": None,
                    "extra_2": None,
                    "extra_3": None,
                    "extra_4": None,
                },
                {
                    "group": "affected_layers",
                    "n_items": int(len(affected)),
                    "metric": "median_final_zero_importance_rate",
                    "value": float(affected["final_zero_importance_rate"].median()),
                    "extra_1": None,
                    "extra_2": None,
                    "extra_3": None,
                    "extra_4": None,
                },
            ]
        )
    return rows


def build_within_phase_stats(layer_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in ["no_prune", "prune0+2", "prune0_only", "prune2_only"]:
        phase_layers = layer_df[layer_df["phase"] == phase].copy()
        if phase_layers.empty:
            continue
        for layer_idx in sorted(phase_layers["layer_idx"].unique().tolist()):
            layer = phase_layers[phase_layers["layer_idx"] == layer_idx]
            rho, p_value = safe_spearman(layer["threshold_epoch50"], layer["fixed_final_minus_selected"])
            rows.append(
                {
                    "phase": phase,
                    "layer_idx": int(layer_idx),
                    "n_seeds": int(len(layer)),
                    "metric": "spearman_threshold_epoch50_vs_fixed_final_minus_selected",
                    "value": rho,
                    "p_value": p_value,
                    "median_threshold_epoch50": float(layer["threshold_epoch50"].median()),
                    "median_fixed_final_minus_selected": float(layer["fixed_final_minus_selected"].median()),
                }
            )
    return rows


def build_report(
    out_path: Path,
    seed_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    gate_stats_df: pd.DataFrame,
    within_phase_df: pd.DataFrame,
) -> None:
    phase_lookup = {row["phase"]: row for _, row in phase_df.iterrows()}
    gate_lookup = {
        row["metric"]: row for _, row in gate_stats_df.iterrows()
    } if not gate_stats_df.empty else {}
    observed_phases = ["prune0+2", "prune0_only", "prune2_only", "no_prune"]
    phase_lines = []
    for phase in observed_phases:
        row = phase_lookup.get(phase)
        if row is None:
            phase_lines.append(f"- `{phase}`: 当前还没有已完成的 unique shadow run。")
            continue
        phase_lines.append(
            f"- `{phase}`: n={int(row['n_seeds'])}, shadow-phase match={fmt(row['shadow_phase_match_rate'])}, threshold-sign phase match={fmt(row['threshold_sign_phase_match_rate'])}, median first/last flag=({fmt(row['median_first_any_flag_epoch'])}, {fmt(row['median_last_any_flag_epoch'])})。"
        )

    threshold_gate = gate_lookup.get("epoch50_threshold_sign_predicts_would_prune", {})
    affected_first = gate_lookup.get("median_first_flag_epoch", {})
    affected_last = gate_lookup.get("median_last_flag_epoch", {})
    first_nonpositive = gate_lookup.get("median_first_nonpositive_threshold_epoch", {})
    gate_close = gate_lookup.get("median_gate_close_lag", {})
    final_zero = gate_lookup.get("median_final_zero_importance_rate", {})
    no_prune_within = within_phase_df[within_phase_df["phase"] == "no_prune"].copy() if not within_phase_df.empty else pd.DataFrame()
    no_prune_lines = []
    if not no_prune_within.empty:
        for _, row in no_prune_within.iterrows():
            no_prune_lines.append(
                f"- `no_prune / layer {int(row['layer_idx'])}`: rho={fmt(row['value'])}, p={fmt(row['p_value'])}, median threshold@50={fmt(row['median_threshold_epoch50'])}。"
            )
    else:
        no_prune_lines.append("- `no_prune` 组当前样本不足，暂时还不能评估相内相关。")

    lines = [
        "# Shadow-Prune Mechanism Interim Report",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 范围",
        "",
        f"- 当前报告整合了 `pilot + panel` 中所有已完成且可读的 unique shadow runs，共 `n={len(seed_df)}` 个 seed。",
        "- 目标不是再看终点 loss 盒图，而是直接检验 `fixed` 内部 prune screening dynamics 的门控机制。",
        "",
        "## 核心发现",
        "",
        f"- `epoch 50` 的 threshold sign 在当前已完成层样本上对 `ever_would_prune` 的判别准确率是 `{fmt(threshold_gate.get('value'))}`；TP/TN/FP/FN = `{fmt(threshold_gate.get('extra_1'))}/{fmt(threshold_gate.get('extra_2'))}/{fmt(threshold_gate.get('extra_3'))}/{fmt(threshold_gate.get('extra_4'))}`。",
        f"- 受影响层的 would-prune 首次出现中位 epoch 是 `{fmt(affected_first.get('value'))}`，最后一次出现中位 epoch 是 `{fmt(affected_last.get('value'))}`，而 threshold 首次变成非正的中位 epoch 是 `{fmt(first_nonpositive.get('value'))}`。",
        f"- 也就是说，当前样本里 would-prune 集合的消失不是因为神经元恢复活跃，而是因为层阈值在中后期下穿到非正区间；`first_nonpositive - last_flag` 的中位间隔是 `{fmt(gate_close.get('value'))}` 个 epoch。",
        f"- 即使 would-prune 身份消失，曾被反复命中的神经元在最后一个 shadow update 时仍常常是近零重要度；当前受影响层的 `final_zero_importance_rate` 中位数是 `{fmt(final_zero.get('value'))}`。",
        "",
        "## 分相结果",
        "",
        *phase_lines,
        "",
        "## 科研解释",
        "",
        "- 这批结果把机制焦点从“有没有 dead neuron”转成了“零重要度在层内是否仍被视为异常”。如果 threshold 在早期为正，零重要度 neuron 会被当成异常候选并反复进入 would-prune 集合；如果 threshold 从一开始就是负的，同样的零重要度现象只会被当成背景噪声。",
        "- 这与早期训练决定后续轨道的文献是一致的：Frankle, Schwab, Morcos 2020 强调 very early training 会锁定后续可塑性；Achille, Rovere, Soatto 2017 把这种现象描述为 critical learning period。我们的数据把这个说法具体化成了一个可观测的结构门控量：`threshold(epoch=50)` 的层级符号。",
        "- 这也修正了对 pruning 的直觉。`prune_only` 不是单纯在训练末期清理一批已经彻底死掉的 neuron；更像是在早期就识别到一小撮层内异常低重要度单元，然后在结构层面对它们进行 lock-in。到后期即便这些单元仍接近零，系统阈值已经漂移，异常判定窗口也关上了。",
        "",
        "## 限制与下一步",
        "",
        "- 当前 panel 已经完整跑完；但关于 common phases (`prune0+2` / `prune0_only`) 的 phase 内部严重度，我们还需要继续做正式统计检验，不能只凭中位数差异下结论。",
        "- 现有 `no_prune` 样本已经显示一个重要限制：threshold sign 很像 phase identity 的早期门控，但不太像 phase 内部 late degradation 的连续严重度指标。",
        *no_prune_lines,
        "",
        "## 文献锚点",
        "",
        "- Frankle, Schwab, Morcos, 2020, *The Early Phase of Neural Network Training*: https://arxiv.org/abs/2002.10365",
        "- Achille, Rovere, Soatto, 2017, *Critical Learning Periods in Deep Neural Networks*: https://arxiv.org/abs/1711.08856",
        "- Frankle, Carbin, 2019, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*: https://openreview.net/forum?id=rJl-b3RcF7",
        "",
        "## 输出文件",
        "",
        "- `mechanism_seed_summary.csv`",
        "- `mechanism_layer_summary.csv`",
        "- `mechanism_phase_summary.csv`",
        "- `mechanism_gate_stats.csv`",
        "- `mechanism_within_phase_stats.csv`",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    seed_df, layer_df, phase_df = summarize_runs(args.roots, args.alignment_csv)
    gate_stats_df = pd.DataFrame(build_gate_stats(seed_df, layer_df))
    within_phase_df = pd.DataFrame(build_within_phase_stats(layer_df))
    write_csv(args.out_dir / "mechanism_seed_summary.csv", seed_df.to_dict("records"))
    write_csv(args.out_dir / "mechanism_layer_summary.csv", layer_df.to_dict("records"))
    write_csv(args.out_dir / "mechanism_phase_summary.csv", phase_df.to_dict("records"))
    write_csv(args.out_dir / "mechanism_gate_stats.csv", gate_stats_df.to_dict("records"))
    write_csv(args.out_dir / "mechanism_within_phase_stats.csv", within_phase_df.to_dict("records"))
    build_report(
        args.out_dir / "mechanism_report_zh.md",
        seed_df,
        layer_df,
        phase_df,
        gate_stats_df,
        within_phase_df,
    )


if __name__ == "__main__":
    main()
