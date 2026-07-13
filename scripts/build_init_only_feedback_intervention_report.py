import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a report for init-only feedback intervention runs."
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
    while metrics_rows and f"size_{count}" in metrics_rows[0]:
        count += 1
    return count


def signature_from_row(row: Dict[str, object], prefix: str, layer_count: int) -> str:
    layers = [str(layer_idx) for layer_idx in range(layer_count) if int(row[f"{prefix}_{layer_idx}"]) > 0]
    return "none" if not layers else "+".join(layers)


def encode_run_path(run_dir: Path, run_label: str) -> Dict[str, object]:
    metrics_rows = load_csv_rows(run_dir / "metrics.csv")
    layer_count = detect_layer_count(metrics_rows)
    actual_rows = [row for row in metrics_rows if int(row["is_update_epoch"]) == 1]
    shadow_rows = [row for row in metrics_rows if int(row["is_shadow_update_epoch"]) == 1]

    actual_path = []
    for row in actual_rows:
        actual_path.append(f"e{int(row['epoch'])}:{signature_from_row(row, 'pruned', layer_count)}")
    shadow_path = []
    for row in shadow_rows:
        shadow_path.append(f"e{int(row['epoch'])}:{signature_from_row(row, 'shadow_would_prune', layer_count)}")

    return {
        "run_label": run_label,
        "actual_path": ";".join(actual_path) if actual_path else "none",
        "shadow_path": ";".join(shadow_path) if shadow_path else "none",
        "final_shadow_signature": "no_shadow" if not shadow_rows else signature_from_row(shadow_rows[-1], "shadow_would_prune", layer_count),
        "final_actual_signature": "no_actual" if not actual_rows else signature_from_row(actual_rows[-1], "pruned", layer_count),
    }


def render_report(
    run_root: Path,
    config: Dict[str, object],
    aggregate_rows: Sequence[Dict[str, object]],
    path_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    row_by_label = {str(row["run_label"]): row for row in aggregate_rows}
    path_by_label = {str(row["run_label"]): row for row in path_rows}

    lines: List[str] = []
    lines.append("# Init-Only Feedback Intervention Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告围绕一个 mismatch seed，比较四种路径：")
    lines.append("")
    lines.append("- `fixed_shadow_baseline`")
    lines.append("- `prune_only_baseline`")
    lines.append("- `prune_until_e2_then_shadow`")
    lines.append("- `prune_until_e4_then_shadow`")
    lines.append("")
    lines.append("目标是判断：早期真实 pruning 是否会改变后续 shadow screening，尤其是最后一次 update 的 layer-level 事件。")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seed：`{config['init_seed']}`")
    lines.append(f"- epochs / update interval：`{config['epochs']}` / `{config['update_interval']}`")
    lines.append("")
    lines.append("## 3. 路径对照")
    lines.append("")
    lines.append("| run label | selected acc | final acc | total pruned | actual path | shadow path |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for row in aggregate_rows:
        path_row = path_by_label[str(row["run_label"])]
        lines.append(
            f"| {row['run_label']} | {fmt(float(row['selected_test_acc']))} | {fmt(float(row['final_test_acc']))} | {row['total_pruned']} | {path_row['actual_path']} | {path_row['shadow_path']} |"
        )
    lines.append("")
    lines.append("## 4. 核心判断")
    lines.append("")
    fixed_shadow = path_by_label.get("fixed_shadow_baseline", {})
    prune_baseline = path_by_label.get("prune_only_baseline", {})
    until_e2 = path_by_label.get("prune_until_e2_then_shadow", {})
    until_e4 = path_by_label.get("prune_until_e4_then_shadow", {})
    lines.append(f"- baseline shadow 最终路径：`{fixed_shadow.get('shadow_path', 'NA')}`")
    lines.append(f"- baseline prune 最终路径：`{prune_baseline.get('actual_path', 'NA')}`")
    lines.append(f"- prune 到 epoch 2 后改 shadow：`{until_e2.get('shadow_path', 'NA')}`")
    lines.append(f"- prune 到 epoch 4 后改 shadow：`{until_e4.get('shadow_path', 'NA')}`")
    lines.append("")
    lines.append("## 5. 解释框架")
    lines.append("")
    lines.append("- 如果 `prune_until_e4_then_shadow` 在最后一步已经不再预测 baseline shadow 里的额外层，那么可以直接支持“前两次真实 pruning 改写了最后一次 screening”。")
    lines.append("- 如果 `prune_until_e2_then_shadow` 仍保留 baseline shadow 的额外层，而 `prune_until_e4_then_shadow` 消失，则说明关键反馈发生在第二次真实 pruning 之后。")
    lines.append("- 这类干预比单纯对比 prune/shadow 更强，因为它把因果问题收缩到了具体 update 前缀。")
    lines.append("")
    lines.append("## 6. 生成产物")
    lines.append("")
    lines.append(f"- 干预汇总表：`{out_dir / 'intervention_path_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    aggregate_rows = load_csv_rows(args.run_root / "aggregate_summary.csv")
    path_rows = []
    for row in aggregate_rows:
        run_label = str(row["run_label"])
        run_dir = args.run_root / run_label
        path_rows.append(encode_run_path(run_dir, run_label))

    write_csv(out_dir / "intervention_path_summary.csv", path_rows)
    report = render_report(
        run_root=args.run_root,
        config=config,
        aggregate_rows=aggregate_rows,
        path_rows=path_rows,
        out_dir=out_dir,
    )
    (out_dir / "feedback_intervention_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
