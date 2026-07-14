import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a threshold-sign persistence report for an init-only structured run."
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


def detect_layer_count(metrics_rows: Sequence[Dict[str, object]]) -> int:
    count = 0
    while metrics_rows and f"shadow_threshold_{count}" in metrics_rows[0]:
        count += 1
    return count


def sign_symbol(value: float) -> str:
    return "+" if float(value) > 0.0 else "-"


def shadow_signature(row: Dict[str, object], layer_count: int) -> str:
    layers = [str(layer_idx) for layer_idx in range(layer_count) if int(row[f"shadow_would_prune_{layer_idx}"]) > 0]
    return "none" if not layers else "+".join(layers)


def build_seed_rows(run_root: Path) -> List[Dict[str, object]]:
    phase_rows = load_csv_rows(run_root / "analysis" / "seed_phase_summary.csv")
    out: List[Dict[str, object]] = []
    for phase_row in phase_rows:
        seed = int(phase_row["init_seed"])
        metrics_rows = load_csv_rows(run_root / "fixed_shadow" / f"init_seed_{seed}" / "metrics.csv")
        update_rows = [row for row in metrics_rows if int(row["is_shadow_update_epoch"]) == 1]
        layer_count = detect_layer_count(metrics_rows)
        row: Dict[str, object] = {
            "init_seed": seed,
            "coarse_phase": phase_row["coarse_phase"],
            "fine_phase": phase_row["fine_phase"],
            "shadow_coarse_match": int(phase_row["shadow_coarse_match"]),
            "shadow_fine_match": int(phase_row["shadow_fine_match"]),
        }
        for layer_idx in range(layer_count):
            sign_traj = "".join(sign_symbol(float(item[f"shadow_threshold_{layer_idx}"])) for item in update_rows)
            row[f"layer{layer_idx}_threshold_sign_traj"] = sign_traj
            row[f"layer{layer_idx}_threshold_final"] = float(update_rows[-1][f"shadow_threshold_{layer_idx}"])
            row[f"layer{layer_idx}_candidate_final"] = int(update_rows[-1][f"shadow_candidate_{layer_idx}"])
            row[f"layer{layer_idx}_would_prune_final"] = int(update_rows[-1][f"shadow_would_prune_{layer_idx}"])
        row["epoch2_shadow_signature"] = shadow_signature(update_rows[0], layer_count)
        row["epoch4_shadow_signature"] = shadow_signature(update_rows[1], layer_count) if len(update_rows) >= 2 else "none"
        row["epoch6_shadow_signature"] = shadow_signature(update_rows[2], layer_count) if len(update_rows) >= 3 else "none"
        out.append(row)
    return sorted(out, key=lambda item: int(item["init_seed"]))


def summarize_by_key(seed_rows: Sequence[Dict[str, object]], group_key: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row[group_key])].append(row)
    out: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        coarse_counter = Counter(str(row["coarse_phase"]) for row in rows)
        fine_counter = Counter(str(row["fine_phase"]) for row in rows)
        modal_coarse, modal_coarse_count = coarse_counter.most_common(1)[0]
        modal_fine, modal_fine_count = fine_counter.most_common(1)[0]
        out.append(
            {
                "group_key": group_key,
                "group_value": key,
                "n_seeds": len(rows),
                "n_coarse_phases": len(coarse_counter),
                "n_fine_phases": len(fine_counter),
                "modal_coarse_phase": modal_coarse,
                "modal_coarse_share": modal_coarse_count / len(rows),
                "modal_fine_phase": modal_fine,
                "modal_fine_share": modal_fine_count / len(rows),
                "mean_layer3_threshold_final": float(np.mean([float(row["layer3_threshold_final"]) for row in rows])),
                "mean_layer3_candidate_final": float(np.mean([float(row["layer3_candidate_final"]) for row in rows])),
                "mean_layer3_would_prune_final": float(np.mean([float(row["layer3_would_prune_final"]) for row in rows])),
            }
        )
    return out


def render_report(
    run_root: Path,
    config: Dict[str, object],
    seed_rows: Sequence[Dict[str, object]],
    layer3_rows: Sequence[Dict[str, object]],
    prefix_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Init-Only Threshold Sign Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告把 `shadow_threshold` 的符号轨迹提升成主机制变量，重点检查两个问题：")
    lines.append("")
    lines.append("- 不同 phase 是否对应不同的 threshold sign trajectory")
    lines.append("- late-commit 分叉是否由某一层 threshold 的过零来触发")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seeds：`{config['init_seeds']}`")
    lines.append("")
    lines.append("## 3. Layer-3 Sign Trajectory 汇总")
    lines.append("")
    lines.append("| sign traj | seeds | coarse phases | fine phases | modal coarse share | final threshold_3 | final wp_3 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in layer3_rows:
        lines.append(
            f"| {row['group_value']} | {row['n_seeds']} | {row['n_coarse_phases']} | {row['n_fine_phases']} | {fmt(float(row['modal_coarse_share']))} | {fmt(float(row['mean_layer3_threshold_final']))} | {fmt(float(row['mean_layer3_would_prune_final']))} |"
        )
    lines.append("")
    lines.append("## 4. Late-Commit Prefix 汇总")
    lines.append("")
    lines.append("| prefix | seeds | coarse phases | fine phases | modal coarse share | final threshold_3 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in prefix_rows:
        lines.append(
            f"| {row['group_value']} | {row['n_seeds']} | {row['n_coarse_phases']} | {row['n_fine_phases']} | {fmt(float(row['modal_coarse_share']))} | {fmt(float(row['mean_layer3_threshold_final']))} |"
        )
    lines.append("")
    lines.append("## 5. 种子级表")
    lines.append("")
    lines.append("| seed | coarse phase | e2 sig | e4 sig | e6 sig | layer3 sign traj | final threshold_3 | final wp_3 |")
    lines.append("| ---: | --- | --- | --- | --- | --- | ---: | ---: |")
    for row in seed_rows:
        lines.append(
            f"| {row['init_seed']} | {row['coarse_phase']} | {row['epoch2_shadow_signature']} | {row['epoch4_shadow_signature']} | {row['epoch6_shadow_signature']} | {row['layer3_threshold_sign_traj']} | {fmt(float(row['layer3_threshold_final']))} | {row['layer3_would_prune_final']} |"
        )
    lines.append("")
    lines.append("## 6. 当前判断")
    lines.append("")
    lines.append("- 如果同一前缀组在 `layer3_threshold_sign_traj` 上分成 `--+` 与 `---` 两类，而最终 phase 也随之分叉，就可以把 threshold sign persistence 视为真正的 branch variable。")
    lines.append("- 这比单纯记录 `would_prune` 更强，因为 `would_prune` 是结果，threshold sign 是驱动筛选开关的连续机制变量。")
    lines.append("- 下一阶段的系统统计应优先围绕 `threshold sign persistence` 展开，而不是只继续堆 seed 数。")
    lines.append("")
    lines.append("## 7. 生成产物")
    lines.append("")
    lines.append(f"- 种子级 sign 表：`{out_dir / 'threshold_sign_seed_summary.csv'}`")
    lines.append(f"- layer3 sign 汇总：`{out_dir / 'layer3_sign_summary.csv'}`")
    lines.append(f"- prefix sign 汇总：`{out_dir / 'prefix_sign_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    seed_rows = build_seed_rows(args.run_root)
    layer3_rows = summarize_by_key(seed_rows, group_key="layer3_threshold_sign_traj")

    seed_rows_with_prefix = []
    for row in seed_rows:
        new_row = dict(row)
        new_row["epoch24_prefix"] = f"{row['epoch2_shadow_signature']} -> {row['epoch4_shadow_signature']}"
        seed_rows_with_prefix.append(new_row)
    prefix_rows = summarize_by_key(seed_rows_with_prefix, group_key="epoch24_prefix")

    write_csv(out_dir / "threshold_sign_seed_summary.csv", seed_rows)
    write_csv(out_dir / "layer3_sign_summary.csv", layer3_rows)
    write_csv(out_dir / "prefix_sign_summary.csv", prefix_rows)
    report = render_report(
        run_root=args.run_root,
        config=config,
        seed_rows=seed_rows,
        layer3_rows=layer3_rows,
        prefix_rows=prefix_rows,
        out_dir=out_dir,
    )
    (out_dir / "threshold_sign_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
