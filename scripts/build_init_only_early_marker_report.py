import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an early-marker report for an init-only structured paired run."
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
    while metrics_rows and f"shadow_would_prune_{count}" in metrics_rows[0]:
        count += 1
    return count


def shadow_signature(row: Dict[str, object], layer_count: int) -> str:
    layers = [str(layer_idx) for layer_idx in range(layer_count) if int(row[f"shadow_would_prune_{layer_idx}"]) > 0]
    return "none" if not layers else "+".join(layers)


def shadow_count_vector(row: Dict[str, object], layer_count: int) -> str:
    return ",".join(str(int(row[f"shadow_would_prune_{layer_idx}"])) for layer_idx in range(layer_count))


def load_seed_feature_rows(run_root: Path) -> List[Dict[str, object]]:
    phase_rows = load_csv_rows(run_root / "analysis" / "seed_phase_summary.csv")
    out: List[Dict[str, object]] = []
    for phase_row in phase_rows:
        seed = int(phase_row["init_seed"])
        metrics_rows = load_csv_rows(run_root / "fixed_shadow" / f"init_seed_{seed}" / "metrics.csv")
        layer_count = detect_layer_count(metrics_rows)
        update_rows = [row for row in metrics_rows if int(row["is_shadow_update_epoch"]) == 1]
        first = update_rows[0]
        second = update_rows[1] if len(update_rows) >= 2 else None

        out.append(
            {
                "init_seed": seed,
                "coarse_phase": phase_row["coarse_phase"],
                "fine_phase": phase_row["fine_phase"],
                "shadow_coarse_match": int(phase_row["shadow_coarse_match"]),
                "shadow_fine_match": int(phase_row["shadow_fine_match"]),
                "fixed_final_test_acc": float(phase_row["fixed_final_test_acc"]),
                "prune_final_test_acc": float(phase_row["prune_final_test_acc"]),
                "epoch2_shadow_signature": shadow_signature(first, layer_count),
                "epoch2_shadow_counts": shadow_count_vector(first, layer_count),
                "epoch2_val_loss": float(first["val_loss"]),
                "epoch2_test_acc": float(first["test_acc"]),
                "epoch4_shadow_signature": shadow_signature(second, layer_count) if second is not None else "none",
                "epoch4_shadow_counts": shadow_count_vector(second, layer_count) if second is not None else "",
                "epoch4_val_loss": float(second["val_loss"]) if second is not None else float("nan"),
                "epoch4_test_acc": float(second["test_acc"]) if second is not None else float("nan"),
            }
        )
    return sorted(out, key=lambda row: int(row["init_seed"]))


def summarize_prefix_groups(seed_rows: Sequence[Dict[str, object]], prefix_key: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row[prefix_key])].append(row)

    out: List[Dict[str, object]] = []
    for prefix, rows in sorted(grouped.items()):
        coarse_counter = Counter(str(row["coarse_phase"]) for row in rows)
        fine_counter = Counter(str(row["fine_phase"]) for row in rows)
        modal_coarse, modal_coarse_count = coarse_counter.most_common(1)[0]
        modal_fine, modal_fine_count = fine_counter.most_common(1)[0]
        out.append(
            {
                "prefix_key": prefix_key,
                "prefix": prefix,
                "n_seeds": len(rows),
                "n_coarse_phases": len(coarse_counter),
                "n_fine_phases": len(fine_counter),
                "modal_coarse_phase": modal_coarse,
                "modal_coarse_share": modal_coarse_count / len(rows),
                "modal_fine_phase": modal_fine,
                "modal_fine_share": modal_fine_count / len(rows),
                "mean_fixed_final_test_acc": float(np.mean([float(row["fixed_final_test_acc"]) for row in rows])),
                "mean_prune_final_test_acc": float(np.mean([float(row["prune_final_test_acc"]) for row in rows])),
            }
        )
    return out


def majority_accuracy(summary_rows: Sequence[Dict[str, object]], share_key: str, count_key: str) -> float:
    total = int(sum(int(row[count_key]) for row in summary_rows))
    if total == 0:
        return float("nan")
    correct = sum(float(row[share_key]) * int(row[count_key]) for row in summary_rows)
    return float(correct / total)


def render_report(
    run_root: Path,
    config: Dict[str, object],
    seed_rows: Sequence[Dict[str, object]],
    epoch2_rows: Sequence[Dict[str, object]],
    epoch24_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    coarse_acc_e2 = majority_accuracy(epoch2_rows, "modal_coarse_share", "n_seeds")
    fine_acc_e2 = majority_accuracy(epoch2_rows, "modal_fine_share", "n_seeds")
    coarse_acc_e24 = majority_accuracy(epoch24_rows, "modal_coarse_share", "n_seeds")
    fine_acc_e24 = majority_accuracy(epoch24_rows, "modal_fine_share", "n_seeds")

    ambiguous_e2 = [row for row in epoch2_rows if int(row["n_coarse_phases"]) > 1 or int(row["n_fine_phases"]) > 1]
    ambiguous_e24 = [row for row in epoch24_rows if int(row["n_coarse_phases"]) > 1 or int(row["n_fine_phases"]) > 1]

    lines: List[str] = []
    lines.append("# Init-Only Early Marker Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告检验：只看早期 shadow-prune 轨迹，能否预测后续的 coarse / fine phase。这里重点看两个前缀：")
    lines.append("")
    lines.append("- `epoch2_shadow_signature`")
    lines.append("- `epoch2+epoch4_shadow_signature_prefix`")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seeds：`{config['init_seeds']}`")
    lines.append("")
    lines.append("## 3. 预测性总结")
    lines.append("")
    lines.append(f"- 只看 `epoch 2` shadow signature，基于 modal 映射的 coarse phase 预测准确率为 `{fmt(coarse_acc_e2)}`。")
    lines.append(f"- 只看 `epoch 2` shadow signature，基于 modal 映射的 fine phase 预测准确率为 `{fmt(fine_acc_e2)}`。")
    lines.append(f"- 看 `epoch 2 + epoch 4` prefix 后，coarse phase 预测准确率为 `{fmt(coarse_acc_e24)}`。")
    lines.append(f"- 看 `epoch 2 + epoch 4` prefix 后，fine phase 预测准确率为 `{fmt(fine_acc_e24)}`。")
    lines.append("")
    lines.append("## 4. Epoch-2 分组")
    lines.append("")
    lines.append("| prefix | seeds | coarse phases | fine phases | modal coarse share | modal fine share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in epoch2_rows:
        lines.append(
            f"| {row['prefix']} | {row['n_seeds']} | {row['n_coarse_phases']} | {row['n_fine_phases']} | {fmt(float(row['modal_coarse_share']))} | {fmt(float(row['modal_fine_share']))} |"
        )
    lines.append("")
    lines.append("## 5. Epoch-2+4 Prefix 分组")
    lines.append("")
    lines.append("| prefix | seeds | coarse phases | fine phases | modal coarse share | modal fine share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in epoch24_rows:
        lines.append(
            f"| {row['prefix']} | {row['n_seeds']} | {row['n_coarse_phases']} | {row['n_fine_phases']} | {fmt(float(row['modal_coarse_share']))} | {fmt(float(row['modal_fine_share']))} |"
        )
    lines.append("")
    lines.append("## 6. 分析")
    lines.append("")
    if ambiguous_e2:
        lines.append("- `epoch 2` 已经能明显分出一部分 phase，但仍有前缀分支会在更晚阶段继续分化。")
    else:
        lines.append("- `epoch 2` 的 shadow signature 已经足够唯一确定最终 phase。")
    if ambiguous_e24:
        lines.append("- 即使加入 `epoch 4`，仍有至少一个前缀到 `epoch 6` 才真正分叉，这说明有 late-commit phase。")
    else:
        lines.append("- 加入 `epoch 4` 后，phase 已基本完全可识别。")
    lines.append("- 这类前缀分叉非常重要：它告诉我们哪类 seed 的结构命运在早期就决定，哪类 seed 要到后期才 commit。")
    lines.append("")
    lines.append("## 7. 种子级特征表")
    lines.append("")
    lines.append("| seed | e2 sig | e4 sig | coarse phase | fine phase | fixed final acc | prune final acc |")
    lines.append("| ---: | --- | --- | --- | --- | ---: | ---: |")
    for row in seed_rows:
        lines.append(
            f"| {row['init_seed']} | {row['epoch2_shadow_signature']} | {row['epoch4_shadow_signature']} | {row['coarse_phase']} | {row['fine_phase']} | {fmt(float(row['fixed_final_test_acc']))} | {fmt(float(row['prune_final_test_acc']))} |"
        )
    lines.append("")
    lines.append("## 8. 生成产物")
    lines.append("")
    lines.append(f"- 种子级 early-marker 表：`{out_dir / 'early_marker_seed_summary.csv'}`")
    lines.append(f"- epoch2 分组表：`{out_dir / 'epoch2_prefix_summary.csv'}`")
    lines.append(f"- epoch2+4 分组表：`{out_dir / 'epoch24_prefix_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    seed_rows = load_seed_feature_rows(args.run_root)
    epoch2_rows = summarize_prefix_groups(seed_rows, prefix_key="epoch2_shadow_signature")

    seed_rows_with_prefix = []
    for row in seed_rows:
        row_copy = dict(row)
        row_copy["epoch24_prefix"] = f"{row['epoch2_shadow_signature']} -> {row['epoch4_shadow_signature']}"
        seed_rows_with_prefix.append(row_copy)
    epoch24_rows = summarize_prefix_groups(seed_rows_with_prefix, prefix_key="epoch24_prefix")

    write_csv(out_dir / "early_marker_seed_summary.csv", seed_rows_with_prefix)
    write_csv(out_dir / "epoch2_prefix_summary.csv", epoch2_rows)
    write_csv(out_dir / "epoch24_prefix_summary.csv", epoch24_rows)
    report = render_report(
        run_root=args.run_root,
        config=config,
        seed_rows=seed_rows_with_prefix,
        epoch2_rows=epoch2_rows,
        epoch24_rows=epoch24_rows,
        out_dir=out_dir,
    )
    (out_dir / "early_marker_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
