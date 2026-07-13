import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cross-run cohort report for init-only structured sweeps."
    )
    parser.add_argument("--run-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                text = value.strip()
                if text == "":
                    parsed[key] = text
                    continue
                try:
                    if "." in text or "e" in text.lower():
                        parsed[key] = float(text)
                    else:
                        parsed[key] = int(text)
                except ValueError:
                    parsed[key] = text
            rows.append(parsed)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_seed_rows(run_roots: Sequence[Path]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for run_root in run_roots:
        config = load_json(run_root / "run_config.json")
        phase_rows = load_csv_rows(run_root / "analysis" / "seed_phase_summary.csv")

        threshold_rows_by_seed: Dict[int, Dict[str, object]] = {}
        threshold_path = run_root / "analysis" / "threshold_sign_seed_summary.csv"
        if threshold_path.exists():
            for row in load_csv_rows(threshold_path):
                threshold_rows_by_seed[int(row["init_seed"])] = row

        early_rows_by_seed: Dict[int, Dict[str, object]] = {}
        early_path = run_root / "analysis" / "early_marker_seed_summary.csv"
        if early_path.exists():
            for row in load_csv_rows(early_path):
                early_rows_by_seed[int(row["init_seed"])] = row

        for row in phase_rows:
            init_seed = int(row["init_seed"])
            threshold_row = threshold_rows_by_seed.get(init_seed, {})
            early_row = early_rows_by_seed.get(init_seed, {})
            out.append(
                {
                    "run_root": str(run_root),
                    "dataset": str(config["dataset"]),
                    "model": str(config["model"]),
                    "init_seed": init_seed,
                    "coarse_phase": str(row["coarse_phase"]),
                    "fine_phase": str(row["fine_phase"]),
                    "shadow_coarse_phase": str(row["shadow_coarse_phase"]),
                    "shadow_fine_phase": str(row["shadow_fine_phase"]),
                    "shadow_coarse_match": int(row["shadow_coarse_match"]),
                    "shadow_fine_match": int(row["shadow_fine_match"]),
                    "fixed_selected_test_acc": float(row["fixed_selected_test_acc"]),
                    "prune_selected_test_acc": float(row["prune_selected_test_acc"]),
                    "fixed_final_test_acc": float(row["fixed_final_test_acc"]),
                    "prune_final_test_acc": float(row["prune_final_test_acc"]),
                    "prune_total_pruned": int(row["prune_total_pruned"]),
                    "epoch24_prefix": str(early_row.get("epoch24_prefix", "")),
                    "epoch2_shadow_signature": str(early_row.get("epoch2_shadow_signature", "")),
                    "layer3_threshold_sign_traj": str(threshold_row.get("layer3_threshold_sign_traj", "")),
                    "layer3_threshold_final": threshold_row.get("layer3_threshold_final", ""),
                }
            )
    return sorted(out, key=lambda item: int(item["init_seed"]))


def summarize_by_key(rows: Sequence[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    out: List[Dict[str, object]] = []
    for value, subset in sorted(grouped.items()):
        out.append(
            {
                "group_key": key,
                "group_value": value,
                "n_seeds": len(subset),
                "shadow_coarse_match_rate": float(np.mean([int(row["shadow_coarse_match"]) for row in subset])),
                "shadow_fine_match_rate": float(np.mean([int(row["shadow_fine_match"]) for row in subset])),
                "mean_fixed_selected_test_acc": float(np.mean([float(row["fixed_selected_test_acc"]) for row in subset])),
                "mean_prune_selected_test_acc": float(np.mean([float(row["prune_selected_test_acc"]) for row in subset])),
                "mean_prune_total_pruned": float(np.mean([float(row["prune_total_pruned"]) for row in subset])),
            }
        )
    return out


def render_report(
    seed_rows: Sequence[Dict[str, object]],
    coarse_rows: Sequence[Dict[str, object]],
    fine_rows: Sequence[Dict[str, object]],
    prefix_rows: Sequence[Dict[str, object]],
    sign_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    mismatch_rows = [
        row for row in seed_rows
        if int(row["shadow_coarse_match"]) == 0 or int(row["shadow_fine_match"]) == 0
    ]
    coarse_counter = Counter(str(row["coarse_phase"]) for row in seed_rows)
    fine_counter = Counter(str(row["fine_phase"]) for row in seed_rows)

    lines: List[str] = []
    lines.append("# Init-Only Structured Cohort Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告把多个 `structured_init_only` runs 合并成一个 cohort，用于回答：扩大 seed 之后，phase 多样性、mismatch 比例和 threshold-sign 机制是否仍然成立。")
    lines.append("")
    lines.append("## 2. 总体统计")
    lines.append("")
    lines.append(f"- cohort 总 seed 数：`{len(seed_rows)}`")
    lines.append(f"- coarse phase 类数：`{len(coarse_counter)}`")
    lines.append(f"- fine phase 类数：`{len(fine_counter)}`")
    lines.append(f"- shadow coarse match rate：`{fmt(float(np.mean([int(row['shadow_coarse_match']) for row in seed_rows])))} `")
    lines.append(f"- shadow fine match rate：`{fmt(float(np.mean([int(row['shadow_fine_match']) for row in seed_rows])))} `")
    lines.append(f"- mismatch seed 数：`{len(mismatch_rows)}` / `{len(seed_rows)}`")
    if mismatch_rows:
        lines.append(
            "- mismatch seeds："
            + "、".join(
                f"`{row['init_seed']}:{row['fine_phase']} -> {row['shadow_fine_phase']}`"
                for row in mismatch_rows
            )
        )
    lines.append("")
    lines.append("## 3. Coarse Phase 汇总")
    lines.append("")
    lines.append("| phase | seeds | shadow coarse match | fixed selected acc | prune selected acc | mean pruned |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in coarse_rows:
        lines.append(
            f"| {row['group_value']} | {row['n_seeds']} | {fmt(float(row['shadow_coarse_match_rate']))} | {fmt(float(row['mean_fixed_selected_test_acc']))} | {fmt(float(row['mean_prune_selected_test_acc']))} | {fmt(float(row['mean_prune_total_pruned']))} |"
        )
    lines.append("")
    lines.append("## 4. Late-Commit Prefix 汇总")
    lines.append("")
    lines.append("| prefix | seeds | shadow fine match |")
    lines.append("| --- | ---: | ---: |")
    for row in prefix_rows:
        lines.append(
            f"| {row['group_value']} | {row['n_seeds']} | {fmt(float(row['shadow_fine_match_rate']))} |"
        )
    lines.append("")
    lines.append("## 5. Layer-3 Sign 汇总")
    lines.append("")
    lines.append("| sign traj | seeds | shadow fine match |")
    lines.append("| --- | ---: | ---: |")
    for row in sign_rows:
        lines.append(
            f"| {row['group_value']} | {row['n_seeds']} | {fmt(float(row['shadow_fine_match_rate']))} |"
        )
    lines.append("")
    lines.append("## 6. 当前判断")
    lines.append("")
    lines.append("- 如果 mismatch 仍然稀少而且集中在少数 boundary seeds，说明结构反馈改写是一个边界现象，而不是普遍现象。")
    lines.append("- 如果 coarse/fine phase 数继续增加，但 shadow match rate 仍高，说明初始化诱导的 latent phase 在更大 cohort 上依然稳。")
    lines.append("- 如果 late-commit prefix 和 layer-3 sign trajectory 继续对应最终 phase，threshold-sign persistence 就可以被视为 cohort-level 机制变量。")
    lines.append("")
    lines.append("## 7. 生成产物")
    lines.append("")
    lines.append(f"- cohort seed 表：`{out_dir / 'cohort_seed_summary.csv'}`")
    lines.append(f"- coarse phase 表：`{out_dir / 'cohort_coarse_phase_summary.csv'}`")
    lines.append(f"- fine phase 表：`{out_dir / 'cohort_fine_phase_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path("results/init_only_lth_20260401/structured_cohort_overview")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = build_seed_rows(args.run_roots)
    coarse_rows = summarize_by_key(seed_rows, "coarse_phase")
    fine_rows = summarize_by_key(seed_rows, "fine_phase")
    prefix_rows = summarize_by_key([row for row in seed_rows if str(row["epoch24_prefix"])], "epoch24_prefix")
    sign_rows = summarize_by_key([row for row in seed_rows if str(row["layer3_threshold_sign_traj"])], "layer3_threshold_sign_traj")

    write_csv(out_dir / "cohort_seed_summary.csv", seed_rows)
    write_csv(out_dir / "cohort_coarse_phase_summary.csv", coarse_rows)
    write_csv(out_dir / "cohort_fine_phase_summary.csv", fine_rows)
    write_csv(out_dir / "cohort_prefix_summary.csv", prefix_rows)
    write_csv(out_dir / "cohort_layer3_sign_summary.csv", sign_rows)

    report = render_report(
        seed_rows=seed_rows,
        coarse_rows=coarse_rows,
        fine_rows=fine_rows,
        prefix_rows=prefix_rows,
        sign_rows=sign_rows,
        out_dir=out_dir,
    )
    (out_dir / "structured_cohort_overview_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
