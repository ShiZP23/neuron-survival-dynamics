import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a family overview across timing-gap intervention runs."
    )
    parser.add_argument("--run-roots", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/init_only_lth_20260401/timing_gap_overview"),
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: List[Dict[str, object]] = []
    for run_root in args.run_roots:
        summary_rows = load_csv_rows(run_root / "analysis" / "timing_gap_summary.csv")
        fixed_row = next(row for row in summary_rows if row["run_label"] == "fixed_shadow_baseline")
        prune_row = next(row for row in summary_rows if row["run_label"] == "prune_only_baseline")
        timing_cols = sorted(key for key in fixed_row.keys() if key.startswith("timing_outcome_"))
        timing_layers = [int(key.rsplit("_", 1)[-1]) for key in timing_cols]
        restored = []
        fixed_kept = []
        side_effect = []
        for row in summary_rows:
            if row["run_label"] in {"fixed_shadow_baseline", "prune_only_baseline"}:
                continue
            outcomes = [row[f"timing_outcome_{layer_idx}"] for layer_idx in timing_layers]
            if all(outcome == "match_prune_timing" for outcome in outcomes):
                restored.append(row["whitelist_label"])
            if all(outcome == "keep_fixed_timing" for outcome in outcomes):
                fixed_kept.append(row["whitelist_label"])
            if row["induced_shadow_layers"] != "none":
                side_effect.append(f"{row['whitelist_label']}->{row['induced_shadow_layers']}")
        seed_rows.append(
            {
                "run_root": str(run_root),
                "seed": fixed_row["run_label"].split("_")[-1] if False else run_root.parent.name.replace("seed_", ""),
                "timing_layers": ",".join(str(layer_idx) for layer_idx in timing_layers),
                "fixed_shadow_signature": fixed_row["final_shadow_signature"],
                "prune_signature": prune_row["final_actual_signature"],
                "restored_whitelists": ",".join(restored) if restored else "none",
                "fixed_timing_whitelists": ",".join(fixed_kept) if fixed_kept else "none",
                "side_effect_whitelists": ",".join(side_effect) if side_effect else "none",
            }
        )

    lines: List[str] = []
    lines.append("# Init-Only Timing-Gap Family Overview")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告汇总所有 timing-gap intervention run，比较不同 timing-gap seeds 的最小 rewrite 前缀和 side effects。")
    lines.append("")
    lines.append("## 2. Seed 汇总")
    lines.append("")
    lines.append("| seed | timing layers | prune signature | fixed shadow signature | restored whitelists | fixed-timing whitelists | side effects |")
    lines.append("| ---: | --- | --- | --- | --- | --- | --- |")
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['timing_layers']} | {row['prune_signature']} | {row['fixed_shadow_signature']} | "
            f"{row['restored_whitelists']} | {row['fixed_timing_whitelists']} | {row['side_effect_whitelists']} |"
        )
    lines.append("")
    lines.append("## 3. 当前判断")
    lines.append("")
    lines.append("- 如果不同 timing-gap seeds 依赖不同的 restored whitelists，说明 timing-gap 不是单一 family，而是至少包含多种 upstream trigger route。")
    lines.append("- 如果某些 restored whitelists 同时伴随 side effects，说明 timing advance 与新增 late layers 可以被同一次真实 pruning 联动触发。")
    lines.append("- 因此 timing-gap seeds 应单独作为一类 intervention 对象，而不应简单并入 absence-type boundary seeds。")
    lines.append("")
    lines.append("## 4. 生成产物")
    lines.append("")
    lines.append(f"- seed 汇总：`{(args.out_dir / 'timing_gap_family_summary.csv').as_posix()}`")

    write_csv(args.out_dir / "timing_gap_family_summary.csv", seed_rows)
    (args.out_dir / "timing_gap_family_overview_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
