import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ALIGNMENT_CSV = Path(
    "results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis/phase_seed_alignment.csv"
)
PANEL_ROOT = Path("results/followup_20260318/shadow_prune_fixed_panel/hard/fixed")
OUT_DIR = Path("results/followup_20260318/shadow_prune_fixed_panel/analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a panel-level report for fixed shadow-prune follow-up runs."
    )
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
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


def latest_run_by_seed(root: Path) -> Dict[int, Path]:
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


def classify_shadow_phase(layer_hits: Dict[int, int]) -> str:
    layer0 = layer_hits.get(0, 0) > 0
    layer1 = layer_hits.get(1, 0) > 0
    layer2 = layer_hits.get(2, 0) > 0
    if layer0 and layer2 and not layer1:
        return "prune0+2"
    if layer0 and not layer1 and not layer2:
        return "prune0_only"
    if layer2 and not layer0 and not layer1:
        return "prune2_only"
    if not layer0 and not layer1 and not layer2:
        return "no_prune"
    return "other"


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


def build_rows(panel_root: Path, alignment_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alignment = pd.read_csv(alignment_csv)
    alignment_by_seed = {int(row["seed"]): row for _, row in alignment.iterrows()}
    run_rows: List[Dict[str, object]] = []
    neuron_rows: List[Dict[str, object]] = []

    for seed, run_dir in sorted(latest_run_by_seed(panel_root).items()):
        if seed not in alignment_by_seed:
            continue
        aligned = alignment_by_seed[seed]
        metrics = pd.read_csv(run_dir / "metrics.csv")
        snapshots = pd.read_csv(run_dir / "shadow_prune_snapshots.csv")
        best_idx = metrics["val_loss"].astype(float).idxmin()
        best_epoch = int(metrics.loc[best_idx, "epoch"])
        last_shadow_epoch = int(metrics.loc[metrics["is_shadow_update_epoch"] == 1, "epoch"].max())
        layer_hits: Dict[int, int] = {}
        total_unique = 0
        total_hits = 0

        for layer_idx in sorted(snapshots["layer_idx"].unique().tolist()):
            layer = snapshots[snapshots["layer_idx"] == layer_idx].copy()
            candidates = layer[layer["is_candidate"] == 1]
            would_prune = layer[layer["would_prune"] == 1]
            grouped_would_prune = would_prune.groupby("neuron_idx")
            unique_neurons = sorted(grouped_would_prune.groups.keys())
            layer_hits[int(layer_idx)] = len(unique_neurons)
            total_unique += len(unique_neurons)
            total_hits += int(would_prune["would_prune"].sum()) if not would_prune.empty else 0

            final_flag_neurons = set(
                would_prune.loc[would_prune["epoch"] == last_shadow_epoch, "neuron_idx"].astype(int).tolist()
            )
            first_epochs = grouped_would_prune["epoch"].min() if unique_neurons else pd.Series(dtype=float)
            last_epochs = grouped_would_prune["epoch"].max() if unique_neurons else pd.Series(dtype=float)
            hits = grouped_would_prune["would_prune"].sum() if unique_neurons else pd.Series(dtype=float)
            last_importance = (
                grouped_would_prune.apply(lambda g: float(g.sort_values("epoch")["importance"].iloc[-1]))
                if unique_neurons
                else pd.Series(dtype=float)
            )
            last_ema = (
                grouped_would_prune.apply(lambda g: float(g.sort_values("epoch")["ema_importance"].iloc[-1]))
                if unique_neurons
                else pd.Series(dtype=float)
            )

            run_rows.append(
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
                    "prune_only_pruned_total_layer": int(aligned[f"prune_only_pruned_{layer_idx}_total"]),
                    "shadow_candidate_unique": int(candidates["neuron_idx"].nunique()) if not candidates.empty else 0,
                    "shadow_would_prune_unique": len(unique_neurons),
                    "shadow_candidate_hits": int(candidates["is_candidate"].sum()) if not candidates.empty else 0,
                    "shadow_would_prune_hits": int(would_prune["would_prune"].sum()) if not would_prune.empty else 0,
                    "shadow_final_would_prune_unique": len(final_flag_neurons),
                    "shadow_resolved_unique": len([neuron for neuron in unique_neurons if neuron not in final_flag_neurons]),
                    "shadow_before_best_unique": int((first_epochs <= best_epoch).sum()) if not first_epochs.empty else 0,
                    "shadow_after_best_only_unique": int((first_epochs > best_epoch).sum()) if not first_epochs.empty else 0,
                    "shadow_first_flag_epoch_median": float(first_epochs.median()) if not first_epochs.empty else None,
                    "shadow_last_flag_epoch_median": float(last_epochs.median()) if not last_epochs.empty else None,
                    "shadow_last_to_best_gap_median": float((best_epoch - last_epochs).median()) if not last_epochs.empty else None,
                    "shadow_last_to_end_gap_median": float((last_shadow_epoch - last_epochs).median()) if not last_epochs.empty else None,
                    "shadow_hit_count_median": float(hits.median()) if not hits.empty else None,
                    "shadow_last_importance_median": float(last_importance.median()) if not last_importance.empty else None,
                    "shadow_last_ema_importance_median": float(last_ema.median()) if not last_ema.empty else None,
                    "shadow_zero_last_importance_rate": float((last_importance == 0.0).mean()) if not last_importance.empty else None,
                    "shadow_zero_last_ema_rate": float((last_ema == 0.0).mean()) if not last_ema.empty else None,
                }
            )

            for neuron_idx, group in grouped_would_prune:
                group = group.sort_values("epoch")
                neuron_rows.append(
                    {
                        "seed": seed,
                        "phase": aligned["prune_only_phase"],
                        "layer_idx": int(layer_idx),
                        "neuron_idx": int(neuron_idx),
                        "best_epoch": best_epoch,
                        "last_shadow_epoch": last_shadow_epoch,
                        "would_prune_hits": int(group["would_prune"].sum()),
                        "first_flag_epoch": int(group["epoch"].min()),
                        "last_flag_epoch": int(group["epoch"].max()),
                        "first_flag_before_best": int(int(group["epoch"].min()) <= best_epoch),
                        "flagged_at_last_shadow_epoch": int(neuron_idx in final_flag_neurons),
                        "last_importance": float(group["importance"].iloc[-1]),
                        "last_ema_importance": float(group["ema_importance"].iloc[-1]),
                    }
                )

        # seed-level total row
        run_rows.append(
            {
                "seed": seed,
                "phase": aligned["prune_only_phase"],
                "layer_idx": -1,
                "run_dir": str(run_dir),
                "best_epoch": best_epoch,
                "last_shadow_epoch": last_shadow_epoch,
                "fixed_final_test_loss": float(aligned["fixed_final_test_loss"]),
                "fixed_final_minus_selected": float(aligned["fixed_final_minus_selected"]),
                "fixed_final_gap": float(aligned["fixed_final_gap"]),
                "prune_only_pruned_total_layer": int(aligned["prune_only_total_pruned"]),
                "shadow_candidate_unique": None,
                "shadow_would_prune_unique": total_unique,
                "shadow_candidate_hits": None,
                "shadow_would_prune_hits": total_hits,
                "shadow_final_would_prune_unique": None,
                "shadow_resolved_unique": None,
                "shadow_before_best_unique": None,
                "shadow_after_best_only_unique": None,
                "shadow_first_flag_epoch_median": None,
                "shadow_last_flag_epoch_median": None,
                "shadow_last_to_best_gap_median": None,
                "shadow_last_to_end_gap_median": None,
                "shadow_hit_count_median": None,
                "shadow_last_importance_median": None,
                "shadow_last_ema_importance_median": None,
                "shadow_zero_last_importance_rate": None,
                "shadow_zero_last_ema_rate": None,
                "shadow_phase": classify_shadow_phase(layer_hits),
                "shadow_phase_match": int(classify_shadow_phase(layer_hits) == aligned["prune_only_phase"]),
            }
        )

    run_df = pd.DataFrame(run_rows)
    neuron_df = pd.DataFrame(neuron_rows)
    return run_df, neuron_df


def build_phase_summary(run_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seed_level = run_df[run_df["layer_idx"] == -1].copy()
    layer_level = run_df[run_df["layer_idx"] >= 0].copy()
    for phase in ["prune0+2", "prune0_only", "prune2_only", "no_prune"]:
        seeds = seed_level[seed_level["phase"] == phase]
        if seeds.empty:
            continue
        rows.append(
            {
                "phase": phase,
                "n_seeds": int(len(seeds)),
                "shadow_phase_match_rate": float(seeds["shadow_phase_match"].mean()),
                "median_shadow_total_unique": float(seeds["shadow_would_prune_unique"].median()),
                "median_shadow_total_hits": float(seeds["shadow_would_prune_hits"].median()),
                "median_fixed_final_minus_selected": float(seeds["fixed_final_minus_selected"].median()),
                "median_fixed_final_gap": float(seeds["fixed_final_gap"].median()),
            }
        )
        phase_layers = layer_level[layer_level["phase"] == phase]
        for layer_idx in sorted(phase_layers["layer_idx"].unique().tolist()):
            layer = phase_layers[phase_layers["layer_idx"] == layer_idx]
            rows.append(
                {
                    "phase": phase,
                    "n_seeds": int(len(layer)),
                    "layer_idx": int(layer_idx),
                    "shadow_phase_match_rate": None,
                    "median_shadow_unique": float(layer["shadow_would_prune_unique"].median()),
                    "median_shadow_hits": float(layer["shadow_would_prune_hits"].median()),
                    "median_first_flag_epoch": float(layer["shadow_first_flag_epoch_median"].median())
                    if layer["shadow_first_flag_epoch_median"].notna().any()
                    else None,
                    "median_last_flag_epoch": float(layer["shadow_last_flag_epoch_median"].median())
                    if layer["shadow_last_flag_epoch_median"].notna().any()
                    else None,
                    "median_last_to_best_gap": float(layer["shadow_last_to_best_gap_median"].median())
                    if layer["shadow_last_to_best_gap_median"].notna().any()
                    else None,
                    "median_last_to_end_gap": float(layer["shadow_last_to_end_gap_median"].median())
                    if layer["shadow_last_to_end_gap_median"].notna().any()
                    else None,
                    "median_zero_last_importance_rate": float(layer["shadow_zero_last_importance_rate"].median())
                    if layer["shadow_zero_last_importance_rate"].notna().any()
                    else None,
                }
            )
    return rows


def build_contrasts(run_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    layer_level = run_df[run_df["layer_idx"] >= 0].copy()
    contrasts = [
        ("prune0+2", 0, "prune0_only", 0),
        ("prune0+2", 2, "prune2_only", 2),
    ]
    for phase_a, layer_a, phase_b, layer_b in contrasts:
        group_a = layer_level[(layer_level["phase"] == phase_a) & (layer_level["layer_idx"] == layer_a)]
        group_b = layer_level[(layer_level["phase"] == phase_b) & (layer_level["layer_idx"] == layer_b)]
        if group_a.empty or group_b.empty:
            continue
        for metric in ["shadow_would_prune_hits", "shadow_last_flag_epoch_median", "shadow_last_to_best_gap_median"]:
            a = group_a[metric].dropna().to_numpy(dtype=float)
            b = group_b[metric].dropna().to_numpy(dtype=float)
            if len(a) == 0 or len(b) == 0:
                continue
            stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
            rows.append(
                {
                    "group_a": f"{phase_a}:layer{layer_a}",
                    "group_b": f"{phase_b}:layer{layer_b}",
                    "metric": metric,
                    "median_group_a": float(np.median(a)),
                    "median_group_b": float(np.median(b)),
                    "median_diff_a_minus_b": float(np.median(a) - np.median(b)),
                    "p_value": float(p_value),
                }
            )
    return rows


def build_correlations(run_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seed_level = run_df[run_df["layer_idx"] == -1].copy()
    for x_key, y_key in [
        ("shadow_would_prune_hits", "fixed_final_minus_selected"),
        ("shadow_would_prune_hits", "fixed_final_gap"),
        ("shadow_would_prune_unique", "fixed_final_minus_selected"),
    ]:
        rho, p_value = safe_spearman(seed_level[x_key], seed_level[y_key])
        rows.append(
            {
                "group": "all_seed_level",
                "x_key": x_key,
                "y_key": y_key,
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    layer_level = run_df[run_df["layer_idx"] >= 0].copy()
    for phase in ["prune0+2", "prune0_only", "prune2_only"]:
        phase_rows = layer_level[(layer_level["phase"] == phase) & (layer_level["shadow_would_prune_unique"] > 0)]
        if phase_rows.empty:
            continue
        for x_key, y_key in [
            ("shadow_would_prune_hits", "fixed_final_minus_selected"),
            ("shadow_last_flag_epoch_median", "fixed_final_minus_selected"),
            ("shadow_last_to_best_gap_median", "fixed_final_minus_selected"),
        ]:
            rho, p_value = safe_spearman(phase_rows[x_key], phase_rows[y_key])
            rows.append(
                {
                    "group": phase,
                    "x_key": x_key,
                    "y_key": y_key,
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    return rows


def build_report(
    out_path: Path,
    phase_summary: pd.DataFrame,
    contrast_rows: pd.DataFrame,
    correlation_rows: pd.DataFrame,
) -> None:
    seed_summary = phase_summary[phase_summary.get("layer_idx").isna()].copy() if "layer_idx" in phase_summary.columns else phase_summary
    phase_lookup = {row["phase"]: row for _, row in seed_summary.iterrows()}
    observed_phases = ["prune0+2", "prune0_only", "prune2_only", "no_prune"]
    phase_lines = []
    for phase in observed_phases:
        row = phase_lookup.get(phase)
        if row is None:
            phase_lines.append(f"- `{phase}` panel: 当前尚无已完成 seed。")
            continue
        phase_lines.append(
            f"- `{phase}` panel: n={int(row['n_seeds'])}, shadow-phase match rate={fmt(row['shadow_phase_match_rate'])}, median shadow total hits={fmt(row['median_shadow_total_hits'])}。"
        )

    partial_note = ""
    if seed_summary["n_seeds"].sum() < 40:
        partial_note = f"当前是中期快照，只覆盖已完成的 {int(seed_summary['n_seeds'].sum())}/40 个 panel seed。"

    lines = [
        "# Shadow-Prune Panel Report",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 设计",
        "",
        "- 该 panel 不是纯 exemplar，而是分层机制跟踪：`prune2_only` 与 `no_prune` 全量纳入，`prune0+2` 与 `prune0_only` 按 `fixed_final_minus_selected` 分层抽样。",
        "- 每个 run 都是完整 3000 epoch 的 `fixed --shadow-prune`。",
        f"- {partial_note}" if partial_note else "- 当前 panel 已完整跑完。",
        "",
        "## 核心结果",
        "",
        *phase_lines,
        "",
        "## 解释方向",
        "",
        "- 如果 panel 里 shadow-phase 与 paired prune phase 高度一致，那么 latent phase 就不只是终点统计现象，而是 `fixed` 内部 screening dynamics 的真实投影。",
        "- 如果 would-be-pruned hits / last-flag timing 与 `fixed_final_minus_selected` 或 `fixed_final_gap` 的相关强于旧 drift proxy，那么后续机制分析应该转向神经元命运，而不是继续堆 share drift。",
        "",
        "## 文件导航",
        "",
        "- `shadow_panel_run_summary.csv`",
        "- `shadow_panel_neuron_summary.csv`",
        "- `shadow_panel_phase_summary.csv`",
        "- `shadow_panel_contrasts.csv`",
        "- `shadow_panel_correlations.csv`",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    run_df, neuron_df = build_rows(args.panel_root, args.alignment_csv)
    phase_summary = pd.DataFrame(build_phase_summary(run_df))
    contrast_rows = pd.DataFrame(build_contrasts(run_df))
    correlation_rows = pd.DataFrame(build_correlations(run_df))
    write_csv(args.out_dir / "shadow_panel_run_summary.csv", run_df.to_dict("records"))
    write_csv(args.out_dir / "shadow_panel_neuron_summary.csv", neuron_df.to_dict("records"))
    write_csv(args.out_dir / "shadow_panel_phase_summary.csv", phase_summary.to_dict("records"))
    write_csv(args.out_dir / "shadow_panel_contrasts.csv", contrast_rows.to_dict("records"))
    write_csv(args.out_dir / "shadow_panel_correlations.csv", correlation_rows.to_dict("records"))
    build_report(args.out_dir / "shadow_panel_report_zh.md", phase_summary, contrast_rows, correlation_rows)


if __name__ == "__main__":
    main()
