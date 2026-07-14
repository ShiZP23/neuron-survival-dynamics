import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd


ALIGNMENT_CSV = Path(
    "results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis/phase_seed_alignment.csv"
)
SHADOW_ROOT = Path("results/followup_20260318/shadow_prune_fixed_pilot/hard/fixed")
OUT_DIR = Path("results/followup_20260318/shadow_prune_fixed_pilot/analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Chinese case-study report for fixed shadow-prune pilot runs."
    )
    parser.add_argument("--shadow-root", type=Path, default=SHADOW_ROOT)
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


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    alignment = pd.read_csv(args.alignment_csv)
    alignment_by_seed = {int(row["seed"]): row for _, row in alignment.iterrows()}
    shadow_runs = latest_run_by_seed(args.shadow_root)

    run_rows: List[Dict[str, object]] = []
    neuron_rows: List[Dict[str, object]] = []

    for seed, run_dir in sorted(shadow_runs.items()):
        if seed not in alignment_by_seed:
            continue
        aligned = alignment_by_seed[seed]
        metrics = pd.read_csv(run_dir / "metrics.csv")
        snapshots = pd.read_csv(run_dir / "shadow_prune_snapshots.csv")
        best_idx = metrics["val_loss"].astype(float).idxmin()
        best_epoch = int(metrics.loc[best_idx, "epoch"])
        shadow_epochs = sorted(
            int(epoch) for epoch in metrics.loc[metrics["is_shadow_update_epoch"] == 1, "epoch"].tolist()
        )
        last_shadow_epoch = shadow_epochs[-1] if shadow_epochs else None

        for layer_idx in sorted(snapshots["layer_idx"].unique().tolist()):
            layer = snapshots[snapshots["layer_idx"] == layer_idx].copy()
            candidates = layer[layer["is_candidate"] == 1]
            would_prune = layer[layer["would_prune"] == 1]

            grouped_candidates = candidates.groupby("neuron_idx")
            grouped_would_prune = would_prune.groupby("neuron_idx")
            candidate_neurons = sorted(grouped_candidates.groups.keys())
            would_prune_neurons = sorted(grouped_would_prune.groups.keys())
            final_flag_neurons = set(
                would_prune.loc[would_prune["epoch"] == last_shadow_epoch, "neuron_idx"].astype(int).tolist()
            )

            run_rows.append(
                {
                    "seed": seed,
                    "phase": aligned["prune_only_phase"],
                    "layer_idx": int(layer_idx),
                    "run_dir": str(run_dir),
                    "best_epoch": best_epoch,
                    "last_shadow_epoch": last_shadow_epoch,
                    "prune_only_pruned_total_layer": int(aligned[f"prune_only_pruned_{layer_idx}_total"]),
                    "shadow_candidate_unique": len(candidate_neurons),
                    "shadow_would_prune_unique": len(would_prune_neurons),
                    "shadow_candidate_hits": int(candidates["is_candidate"].sum()) if not candidates.empty else 0,
                    "shadow_would_prune_hits": int(would_prune["would_prune"].sum()) if not would_prune.empty else 0,
                    "shadow_final_would_prune_unique": len(final_flag_neurons),
                    "shadow_resolved_unique": len(
                        [neuron for neuron in would_prune_neurons if neuron not in final_flag_neurons]
                    ),
                    "shadow_before_best_unique": len(
                        [
                            neuron
                            for neuron, group in grouped_would_prune
                            if int(group["epoch"].min()) <= best_epoch
                        ]
                    )
                    if would_prune_neurons
                    else 0,
                    "shadow_after_best_only_unique": len(
                        [
                            neuron
                            for neuron, group in grouped_would_prune
                            if int(group["epoch"].min()) > best_epoch
                        ]
                    )
                    if would_prune_neurons
                    else 0,
                    "shadow_first_flag_epoch_median": float(
                        grouped_would_prune["epoch"].min().median()
                    )
                    if would_prune_neurons
                    else None,
                    "shadow_last_flag_epoch_median": float(
                        grouped_would_prune["epoch"].max().median()
                    )
                    if would_prune_neurons
                    else None,
                    "shadow_hit_count_median": float(grouped_would_prune["would_prune"].sum().median())
                    if would_prune_neurons
                    else None,
                }
            )

            for neuron_idx, group in grouped_would_prune:
                neuron_rows.append(
                    {
                        "seed": seed,
                        "phase": aligned["prune_only_phase"],
                        "layer_idx": int(layer_idx),
                        "neuron_idx": int(neuron_idx),
                        "would_prune_hits": int(group["would_prune"].sum()),
                        "first_flag_epoch": int(group["epoch"].min()),
                        "last_flag_epoch": int(group["epoch"].max()),
                        "flagged_at_last_shadow_epoch": int(neuron_idx in final_flag_neurons),
                        "best_epoch": best_epoch,
                        "first_flag_before_best": int(int(group["epoch"].min()) <= best_epoch),
                        "last_importance": float(group.sort_values("epoch")["importance"].iloc[-1]),
                        "last_ema_importance": float(group.sort_values("epoch")["ema_importance"].iloc[-1]),
                    }
                )

    write_csv(args.out_dir / "shadow_pilot_run_summary.csv", run_rows)
    write_csv(args.out_dir / "shadow_pilot_neuron_summary.csv", neuron_rows)

    run_summary = pd.DataFrame(run_rows)
    neuron_summary = pd.DataFrame(neuron_rows)
    lines = [
        "# Fixed Shadow-Prune Pilot Case Report",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 设计",
        "",
        "- 这是一组机制性 pilot，而不是新的大样本 sweep。",
        "- 对 `fixed` 运行启用 `shadow prune`：按 `prune_only` 同样的筛选与消融规则评估 would-be-pruned neurons，但不实际删 neuron。",
        "- 当前 pilot 使用 4 个 seed exemplar，分别覆盖 `prune0+2`、`prune0_only`、`prune2_only`、`no_prune`。",
        "",
        "## 核心观察",
        "",
    ]
    for phase in ["prune0+2", "prune0_only", "prune2_only", "no_prune"]:
        subset = run_summary[run_summary["phase"] == phase]
        if subset.empty:
            continue
        seed = int(subset.iloc[0]["seed"])
        lines.append(f"- {phase} exemplar: seed {seed}")
        for _, row in subset.sort_values("layer_idx").iterrows():
            lines.append(
                f"  layer {int(row['layer_idx'])}: unique would-prune={int(row['shadow_would_prune_unique'])}, "
                f"hits={int(row['shadow_would_prune_hits'])}, final-persistent={int(row['shadow_final_would_prune_unique'])}, "
                f"resolved={int(row['shadow_resolved_unique'])}, before-best={int(row['shadow_before_best_unique'])}, "
                f"after-best-only={int(row['shadow_after_best_only_unique'])}, "
                f"median first/last flag=({fmt(row['shadow_first_flag_epoch_median'])}, {fmt(row['shadow_last_flag_epoch_median'])})"
            )

    lines.extend(
        [
            "",
            "## 初步机制解释",
            "",
            "- 在 `prune0+2`、`prune0_only`、`prune2_only` exemplar 中，would-be-pruned neurons 都在 epoch 50 首次出现，说明 shadow-prune 目标集几乎是训练一开始就确定的。",
            "- 但这些 neuron 并没有一直持续到训练结束；它们通常在 550 到 700 左右停止满足 would-prune 条件，远早于 best epoch。",
            "- 结合快照文件可见，一个典型模式是：这些 neuron 的 importance 与 EMA 后期仍接近 0，但层内阈值已经漂到负值，因此“零重要度”不再被视为异常候选。",
            "- 这意味着当前规则更像是在追踪“相对异常的脆弱单元”，而不是绝对意义上的死神经元。这正是下一步需要 shadow-prune 扩样和改进阈值定义的原因。",
            "",
            "## 如何读这份 pilot",
            "",
            "- `shadow_would_prune_unique` 代表有多少固定身份 neuron 至少一次满足 would-prune 条件。",
            "- `shadow_would_prune_hits` 代表这些 neuron 被重复命中的总次数，用来区分一次性短暂低重要度和长期持续脆弱。",
            "- `shadow_final_would_prune_unique` 与 `shadow_resolved_unique` 则给出一个最粗的 persistence / rescue proxy。",
            "",
            "## 研究用途",
            "",
            "- 这份 pilot 主要用于验证记录层是否足够支撑后续机制研究。",
            "- 如果 `prune0+2`、`prune0_only`、`prune2_only` 在 would-be-pruned neuron 的复发、持续、best 前后分布上有系统差异，就说明 shadow-prune 是值得扩大样本的。",
            "",
            "## 文件导航",
            "",
            "- `shadow_pilot_run_summary.csv`",
            "- `shadow_pilot_neuron_summary.csv`",
        ]
    )
    (args.out_dir / "shadow_pilot_report_zh.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
