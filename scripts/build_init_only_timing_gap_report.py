import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a timing-gap intervention report for layer-targeted init-only feedback runs."
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
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_layer_count(metrics_rows: Sequence[Dict[str, object]]) -> int:
    count = 0
    while metrics_rows and f"size_{count}" in metrics_rows[0]:
        count += 1
    return count


def signature_from_row(row: Dict[str, object], prefix: str, layer_count: int) -> str:
    layers = [str(layer_idx) for layer_idx in range(layer_count) if int(row[f"{prefix}_{layer_idx}"]) > 0]
    return "none" if not layers else "+".join(layers)


def parse_signature_layers(signature: str) -> Set[int]:
    text = signature.strip()
    if text in {"", "none", "no_shadow", "no_actual"}:
        return set()
    return {int(part) for part in text.split("+")}


def first_commit_epoch(metrics_rows: Sequence[Dict[str, object]], event_key: str) -> Optional[int]:
    for row in metrics_rows:
        epoch = int(row["epoch"])
        if int(row[event_key]) > 0:
            return epoch
    return None


def build_run_summary(
    run_dir: Path,
    aggregate_row: Dict[str, object],
    timing_layers: Sequence[int],
    prune_baseline_actual_epochs: Dict[int, Optional[int]],
    fixed_baseline_shadow_epochs: Dict[int, Optional[int]],
    fixed_baseline_final_shadow_layers: Set[int],
) -> Dict[str, object]:
    metrics_rows = load_csv_rows(run_dir / "metrics.csv")
    layer_count = detect_layer_count(metrics_rows)
    actual_rows = [row for row in metrics_rows if int(row["is_update_epoch"]) == 1]
    shadow_rows = [row for row in metrics_rows if int(row["is_shadow_update_epoch"]) == 1]
    final_actual = actual_rows[-1] if actual_rows else None
    final_shadow = shadow_rows[-1] if shadow_rows else None
    final_actual_signature = "no_actual" if final_actual is None else signature_from_row(final_actual, "pruned", layer_count)
    final_shadow_signature = "no_shadow" if final_shadow is None else signature_from_row(final_shadow, "shadow_would_prune", layer_count)
    final_shadow_layers = parse_signature_layers(final_shadow_signature)

    row: Dict[str, object] = {
        "run_label": aggregate_row["run_label"],
        "whitelist_label": aggregate_row.get("whitelist_label", "all"),
        "selected_test_acc": aggregate_row["selected_test_acc"],
        "final_test_acc": aggregate_row["final_test_acc"],
        "total_pruned": aggregate_row["total_pruned"],
        "final_actual_signature": final_actual_signature,
        "final_shadow_signature": final_shadow_signature,
        "induced_shadow_layers": "none"
        if not (final_shadow_layers - fixed_baseline_final_shadow_layers)
        else "+".join(str(layer) for layer in sorted(final_shadow_layers - fixed_baseline_final_shadow_layers)),
    }

    outcome_labels: List[str] = []
    for layer_idx in timing_layers:
        actual_epoch = first_commit_epoch(actual_rows, f"pruned_{layer_idx}")
        shadow_epoch = first_commit_epoch(shadow_rows, f"shadow_would_prune_{layer_idx}")
        row[f"actual_commit_epoch_{layer_idx}"] = actual_epoch
        row[f"shadow_commit_epoch_{layer_idx}"] = shadow_epoch
        row[f"prune_baseline_commit_epoch_{layer_idx}"] = prune_baseline_actual_epochs[layer_idx]
        row[f"fixed_shadow_commit_epoch_{layer_idx}"] = fixed_baseline_shadow_epochs[layer_idx]
        if shadow_epoch == prune_baseline_actual_epochs[layer_idx]:
            outcome = "match_prune_timing"
        elif shadow_epoch == fixed_baseline_shadow_epochs[layer_idx]:
            outcome = "keep_fixed_timing"
        elif shadow_epoch is None:
            outcome = "suppress_target"
        else:
            outcome = "other_timing"
        row[f"timing_outcome_{layer_idx}"] = outcome
        outcome_labels.append(f"L{layer_idx}:{outcome}")
    row["timing_outcomes"] = ", ".join(outcome_labels)
    return row


def render_report(
    run_root: Path,
    config: Dict[str, object],
    summary_rows: Sequence[Dict[str, object]],
    timing_layers: Sequence[int],
    out_dir: Path,
) -> str:
    row_by_label = {str(row["run_label"]): row for row in summary_rows}
    fixed_row = row_by_label["fixed_shadow_baseline"]
    prune_row = row_by_label["prune_only_baseline"]
    targeted_rows = [
        row for row in summary_rows
        if str(row["run_label"]) not in {"fixed_shadow_baseline", "prune_only_baseline"}
    ]

    timing_restored_rows = []
    fixed_timing_rows = []
    side_effect_rows = []
    for row in targeted_rows:
        restored = True
        fixed_kept = True
        for layer_idx in timing_layers:
            if row[f"timing_outcome_{layer_idx}"] != "match_prune_timing":
                restored = False
            if row[f"timing_outcome_{layer_idx}"] != "keep_fixed_timing":
                fixed_kept = False
        if restored:
            timing_restored_rows.append(row)
        if fixed_kept:
            fixed_timing_rows.append(row)
        if str(row["induced_shadow_layers"]) != "none":
            side_effect_rows.append(row)

    timing_layer_label = ",".join(str(layer_idx) for layer_idx in timing_layers)
    lines: List[str] = []
    lines.append("# Init-Only Timing-Gap Intervention Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告专门分析 `timing-gap` seed：真实 prune 与 fixed-shadow 最终包含同一目标层，但 commit epoch 不同。我们关心哪些最小真实 pruning 前缀足以把 shadow 的 commit timing 拉回 prune baseline。")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seed：`{config['init_seed']}`")
    lines.append(f"- timing-gap layers：`{timing_layer_label}`")
    lines.append("")
    lines.append("## 3. Baseline")
    lines.append("")
    for layer_idx in timing_layers:
        lines.append(
            f"- `layer {layer_idx}`：prune baseline commit 在 `epoch {prune_row[f'prune_baseline_commit_epoch_{layer_idx}']}`，"
            f"fixed shadow commit 在 `epoch {fixed_row[f'fixed_shadow_commit_epoch_{layer_idx}']}`"
        )
    lines.append("")
    lines.append("## 4. 结果总表")
    lines.append("")
    lines.append("| run label | whitelist | final actual | final shadow | timing outcomes | induced shadow layers |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summary_rows:
        lines.append(
            f"| {row['run_label']} | {row['whitelist_label']} | {row['final_actual_signature']} | "
            f"{row['final_shadow_signature']} | {row['timing_outcomes']} | {row['induced_shadow_layers']} |"
        )
    lines.append("")
    lines.append("## 5. 核心判断")
    lines.append("")
    if timing_restored_rows:
        lines.append(
            "- 足以把 shadow commit timing 拉回 prune baseline 的前缀："
            + "、".join(f"`{row['whitelist_label']}`" for row in timing_restored_rows)
        )
    else:
        lines.append("- 当前没有任何 tested prefix 足以把 shadow commit timing 拉回 prune baseline。")
    if fixed_timing_rows:
        lines.append(
            "- 仍保持 fixed-shadow 原始 timing 的前缀："
            + "、".join(f"`{row['whitelist_label']}`" for row in fixed_timing_rows)
        )
    if side_effect_rows:
        lines.append(
            "- 会诱发额外 late shadow layers 的前缀："
            + "、".join(f"`{row['whitelist_label']}`→`{row['induced_shadow_layers']}`" for row in side_effect_rows)
        )
    lines.append("")
    lines.append("## 6. 解释框架")
    lines.append("")
    lines.append("- 如果某个 targeted prefix 已经足以把 shadow commit 从较晚的 epoch 拉回 prune baseline，更合理的解释是：第一次真实 pruning 已经改变了后续 gate 的翻正时机。")
    lines.append("- 如果同样的 prefix 还诱发额外 late layers，说明 timing-gap 与 layer-identity 边界并不完全独立，真实 pruning 既能提前原目标层，也可能顺带打开新的晚期层。")
    lines.append("- 因此 timing-gap seed 的关键量不是最终 coarse phase，而是目标层的 commit epoch 和是否伴随 side effects。")
    lines.append("")
    lines.append("## 7. 生成产物")
    lines.append("")
    lines.append(f"- timing-gap 摘要：`{out_dir / 'timing_gap_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    aggregate_rows = load_csv_rows(args.run_root / "aggregate_summary.csv")
    aggregate_by_label = {str(row["run_label"]): row for row in aggregate_rows}

    fixed_metrics = load_csv_rows(args.run_root / "fixed_shadow_baseline" / "metrics.csv")
    layer_count = detect_layer_count(fixed_metrics)
    fixed_shadow_rows = [row for row in fixed_metrics if int(row["is_shadow_update_epoch"]) == 1]

    prune_metrics = load_csv_rows(args.run_root / "prune_only_baseline" / "metrics.csv")
    prune_actual_rows = [row for row in prune_metrics if int(row["is_update_epoch"]) == 1]

    prune_baseline_actual_epochs = {
        layer_idx: first_commit_epoch(prune_actual_rows, f"pruned_{layer_idx}")
        for layer_idx in range(layer_count)
    }
    fixed_baseline_shadow_epochs = {
        layer_idx: first_commit_epoch(fixed_shadow_rows, f"shadow_would_prune_{layer_idx}")
        for layer_idx in range(layer_count)
    }

    timing_layers = [
        layer_idx
        for layer_idx in range(layer_count)
        if prune_baseline_actual_epochs[layer_idx] is not None
        and fixed_baseline_shadow_epochs[layer_idx] is not None
        and prune_baseline_actual_epochs[layer_idx] != fixed_baseline_shadow_epochs[layer_idx]
    ]
    if not timing_layers:
        raise RuntimeError("No timing-gap layers detected in this run root.")

    fixed_baseline_final_shadow = fixed_shadow_rows[-1]
    fixed_baseline_final_shadow_layers = parse_signature_layers(
        signature_from_row(fixed_baseline_final_shadow, "shadow_would_prune", layer_count)
    )

    summary_rows = [
        build_run_summary(
            run_dir=args.run_root / str(aggregate_row["run_label"]),
            aggregate_row=aggregate_row,
            timing_layers=timing_layers,
            prune_baseline_actual_epochs=prune_baseline_actual_epochs,
            fixed_baseline_shadow_epochs=fixed_baseline_shadow_epochs,
            fixed_baseline_final_shadow_layers=fixed_baseline_final_shadow_layers,
        )
        for aggregate_row in aggregate_rows
    ]
    report = render_report(
        run_root=args.run_root,
        config=config,
        summary_rows=summary_rows,
        timing_layers=timing_layers,
        out_dir=out_dir,
    )
    write_csv(out_dir / "timing_gap_summary.csv", summary_rows)
    (out_dir / "timing_gap_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
