import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a late-commit branch report for init-only structured runs."
    )
    parser.add_argument("--run-root", type=Path, required=True)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def shadow_signature(row: Dict[str, object], epoch: int) -> str:
    layers = []
    layer_idx = 0
    while f"shadow_would_prune_{layer_idx}" in row:
        if int(row[f"shadow_would_prune_{layer_idx}"]) > 0:
            layers.append(str(layer_idx))
        layer_idx += 1
    return "none" if not layers else "+".join(layers)


def render_report(
    run_root: Path,
    config: Dict[str, object],
    branch_rows: Sequence[Dict[str, object]],
    branch_summary_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Init-Only Late-Commit Branch Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告只看那些在早期 shadow prefix 完全相同、但在后续 epoch 才分叉的 seeds，用来回答：late-commit 的触发信号是什么。")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append("")
    lines.append("## 3. 分叉前缀汇总")
    lines.append("")
    lines.append("| prefix | seeds | final phases | mean e6 threshold_3 |")
    lines.append("| --- | ---: | --- | ---: |")
    for row in branch_summary_rows:
        lines.append(
            f"| {row['prefix']} | {row['n_seeds']} | {row['final_phases']} | {fmt(float(row['mean_epoch6_shadow_threshold_3']))} |"
        )
    lines.append("")
    lines.append("## 4. 种子级细节")
    lines.append("")
    lines.append("| seed | prefix | e6 shadow sig | coarse phase | e6 cand_3 | e6 wp_3 | e6 threshold_3 | e6 mean_3 | e6 std_3 |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in branch_rows:
        lines.append(
            f"| {row['init_seed']} | {row['prefix']} | {row['epoch6_shadow_signature']} | {row['coarse_phase']} | {row['epoch6_shadow_candidate_3']} | {row['epoch6_shadow_would_prune_3']} | {fmt(float(row['epoch6_shadow_threshold_3']))} | {fmt(float(row['epoch6_shadow_mean_3']))} | {fmt(float(row['epoch6_shadow_std_3']))} |"
        )
    lines.append("")
    lines.append("## 5. 当前判断")
    lines.append("")
    lines.append("- 如果同一前缀组里，只有 threshold 过零的 seed 在最后一层新增 shadow hits，这说明 late-commit 不是随机抖动，而是由阈值符号翻转触发。")
    lines.append("- 这和你之前在 shadow-prune fixed panel 里看到的现象是同一类机制：候选层并不是一直稳定存在，而是在阈值是否为正这件事上发生开关。")
    lines.append("- 因此下一阶段不应只记录 `would_prune`，还应把 `threshold sign persistence` 当作主机制变量。")
    lines.append("")
    lines.append("## 6. 生成产物")
    lines.append("")
    lines.append(f"- branch seed 表：`{out_dir / 'late_commit_seed_summary.csv'}`")
    lines.append(f"- branch prefix 表：`{out_dir / 'late_commit_prefix_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    early_rows = load_csv_rows(args.run_root / "analysis" / "early_marker_seed_summary.csv")
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in early_rows:
        grouped[str(row["epoch24_prefix"])].append(row)

    branch_rows: List[Dict[str, object]] = []
    for prefix, rows in grouped.items():
        final_phases = {str(row["coarse_phase"]) for row in rows}
        if len(final_phases) <= 1:
            continue
        for row in rows:
            seed = int(row["init_seed"])
            metrics_rows = load_csv_rows(args.run_root / "fixed_shadow" / f"init_seed_{seed}" / "metrics.csv")
            epoch6 = [item for item in metrics_rows if int(item["epoch"]) == 6][0]
            branch_rows.append(
                {
                    "init_seed": seed,
                    "prefix": prefix,
                    "coarse_phase": row["coarse_phase"],
                    "fine_phase": row["fine_phase"],
                    "epoch6_shadow_signature": shadow_signature(epoch6, 6),
                    "epoch6_shadow_candidate_3": int(epoch6["shadow_candidate_3"]),
                    "epoch6_shadow_would_prune_3": int(epoch6["shadow_would_prune_3"]),
                    "epoch6_shadow_threshold_3": float(epoch6["shadow_threshold_3"]),
                    "epoch6_shadow_mean_3": float(epoch6["shadow_ema_mean_3"]),
                    "epoch6_shadow_std_3": float(epoch6["shadow_ema_std_3"]),
                }
            )

    branch_rows = sorted(branch_rows, key=lambda row: (str(row["prefix"]), int(row["init_seed"])))
    branch_summary_rows: List[Dict[str, object]] = []
    if branch_rows:
        grouped_branch: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in branch_rows:
            grouped_branch[str(row["prefix"])].append(row)
        for prefix, rows in sorted(grouped_branch.items()):
            branch_summary_rows.append(
                {
                    "prefix": prefix,
                    "n_seeds": len(rows),
                    "final_phases": "; ".join(sorted({str(row["coarse_phase"]) for row in rows})),
                    "mean_epoch6_shadow_threshold_3": float(np.mean([float(row["epoch6_shadow_threshold_3"]) for row in rows])),
                }
            )

    write_csv(out_dir / "late_commit_seed_summary.csv", branch_rows)
    write_csv(out_dir / "late_commit_prefix_summary.csv", branch_summary_rows)
    report = render_report(
        run_root=args.run_root,
        config=config,
        branch_rows=branch_rows,
        branch_summary_rows=branch_summary_rows,
        out_dir=out_dir,
    )
    (out_dir / "late_commit_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
