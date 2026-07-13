import argparse
import csv
import json
from statistics import mean
from pathlib import Path
from typing import Dict, List, Sequence, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a report for layer-targeted init-only feedback interventions."
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
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def parse_signature_layers(signature: str) -> Set[int]:
    text = signature.strip()
    if text in {"", "none", "no_shadow", "no_actual"}:
        return set()
    return {int(part) for part in text.split("+")}


def build_run_row(run_dir: Path, aggregate_row: Dict[str, object]) -> Dict[str, object]:
    metrics_rows = load_csv_rows(run_dir / "metrics.csv")
    layer_count = detect_layer_count(metrics_rows)
    actual_rows = [row for row in metrics_rows if int(row["is_update_epoch"]) == 1]
    shadow_rows = [row for row in metrics_rows if int(row["is_shadow_update_epoch"]) == 1]
    first_actual = actual_rows[0] if actual_rows else None
    final_actual = actual_rows[-1] if actual_rows else None
    final_shadow = shadow_rows[-1] if shadow_rows else None
    row: Dict[str, object] = {
        "run_label": aggregate_row["run_label"],
        "whitelist_label": aggregate_row.get("whitelist_label", "all"),
        "selected_test_acc": aggregate_row["selected_test_acc"],
        "final_test_acc": aggregate_row["final_test_acc"],
        "total_pruned": aggregate_row["total_pruned"],
        "first_actual_signature": "no_actual" if first_actual is None else signature_from_row(first_actual, "pruned", layer_count),
        "final_actual_signature": "no_actual" if final_actual is None else signature_from_row(final_actual, "pruned", layer_count),
        "final_shadow_signature": "no_shadow" if final_shadow is None else signature_from_row(final_shadow, "shadow_would_prune", layer_count),
    }
    if final_shadow is not None:
        for layer_idx in range(layer_count):
            row[f"final_shadow_threshold_{layer_idx}"] = float(final_shadow[f"shadow_threshold_{layer_idx}"])
            row[f"final_shadow_would_prune_{layer_idx}"] = int(final_shadow[f"shadow_would_prune_{layer_idx}"])
    return row


def render_report(
    run_root: Path,
    config: Dict[str, object],
    run_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    row_by_label = {str(row["run_label"]): row for row in run_rows}
    fixed_row = row_by_label["fixed_shadow_baseline"]
    prune_row = row_by_label["prune_only_baseline"]
    family_extra_layers = sorted(
        parse_signature_layers(str(fixed_row["final_shadow_signature"]))
        - parse_signature_layers(str(prune_row["final_actual_signature"]))
    )
    family_extra_label = "none" if not family_extra_layers else "+".join(str(layer_idx) for layer_idx in family_extra_layers)

    targeted_rows = [
        row for row in run_rows
        if str(row["run_label"]) not in {"fixed_shadow_baseline", "prune_only_baseline"}
    ]
    for row in targeted_rows:
        row["matches_fixed_shadow"] = int(row["final_shadow_signature"] == fixed_row["final_shadow_signature"])
        row["matches_prune_baseline"] = int(row["final_shadow_signature"] == prune_row["final_actual_signature"])
    for row in run_rows:
        row["family_extra_layers"] = family_extra_label
        if family_extra_layers and str(row["final_shadow_signature"]) != "no_shadow":
            thresholds = [float(row[f"final_shadow_threshold_{layer_idx}"]) for layer_idx in family_extra_layers]
            wp_values = [int(row[f"final_shadow_would_prune_{layer_idx}"]) for layer_idx in family_extra_layers]
            row["final_shadow_extra_threshold_mean"] = mean(thresholds)
            row["final_shadow_extra_would_prune_sum"] = sum(wp_values)
        else:
            row["final_shadow_extra_threshold_mean"] = None
            row["final_shadow_extra_would_prune_sum"] = None

    sufficient_rows = [row for row in targeted_rows if int(row["matches_prune_baseline"]) == 1]
    insufficient_rows = [row for row in targeted_rows if int(row["matches_prune_baseline"]) == 0]

    lines: List[str] = []
    lines.append("# Init-Only Layer-Targeted Feedback Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告专门回答一个更细的因果问题：对 mismatch seed 来说，哪一次 targeted 真实 pruning 以及其中哪些层足以改写后续 screening path。")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seed：`{config['init_seed']}`")
    lines.append(f"- epochs / update interval：`{config['epochs']}` / `{config['update_interval']}`")
    lines.append(f"- family extra layers：`{family_extra_label}`")
    lines.append("")
    lines.append("## 3. 结果总表")
    lines.append("")
    lines.append("| run label | whitelist | first actual | final actual | final shadow | extra th mean@final | extra wp sum@final | match prune |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: |")
    for row in run_rows:
        match_prune = "-"
        if str(row["run_label"]) not in {"fixed_shadow_baseline", "prune_only_baseline"}:
            match_prune = str(row["matches_prune_baseline"])
        extra_th = "NA" if row["final_shadow_extra_threshold_mean"] is None else fmt(float(row["final_shadow_extra_threshold_mean"]))
        extra_wp = "NA" if row["final_shadow_extra_would_prune_sum"] is None else str(row["final_shadow_extra_would_prune_sum"])
        lines.append(
            f"| {row['run_label']} | {row['whitelist_label']} | {row['first_actual_signature']} | {row['final_actual_signature']} | {row['final_shadow_signature']} | {extra_th} | {extra_wp} | {match_prune} |"
        )
    lines.append("")
    lines.append("## 4. 核心判断")
    lines.append("")
    lines.append(f"- fixed shadow baseline 最终为：`{fixed_row['final_shadow_signature']}`")
    lines.append(f"- prune baseline 最终为：`{prune_row['final_actual_signature']}`")
    if sufficient_rows:
        lines.append(
            "- 足以把最终 shadow path 拉回 prune baseline 的层级前缀："
            + "、".join(f"`{row['whitelist_label']}`" for row in sufficient_rows)
        )
    else:
        lines.append("- 当前没有任何单层或给定层前缀足以把最终 shadow path 拉回 prune baseline。")
    if insufficient_rows:
        lines.append(
            "- 仍保留 dense baseline 晚期偏差的前缀："
            + "、".join(f"`{row['whitelist_label']}`" for row in insufficient_rows)
        )
    lines.append("")
    lines.append("## 5. 解释框架")
    lines.append("")
    lines.append("- 如果某个单层前缀已经足以消除最终的额外 late shadow event，说明分叉主因可以收缩到该层的早期真实 pruning。")
    lines.append("- 如果只有联合前缀有效，则更像跨层耦合：单层 pruning 不够，层间组合才会改写后续 screening。")
    lines.append("- 被追踪的 family extra-layer threshold / would-prune 是否在 rewrite 条件下回到 prune baseline，是判断该 late event 是否被真正关掉的连续机制证据。")
    lines.append("")
    lines.append("## 6. 生成产物")
    lines.append("")
    lines.append(f"- 干预摘要：`{out_dir / 'layer_feedback_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    aggregate_rows = load_csv_rows(args.run_root / "aggregate_summary.csv")
    run_rows = [
        build_run_row(args.run_root / str(aggregate_row["run_label"]), aggregate_row)
        for aggregate_row in aggregate_rows
    ]
    report = render_report(
        run_root=args.run_root,
        config=config,
        run_rows=run_rows,
        out_dir=out_dir,
    )
    write_csv(out_dir / "layer_feedback_summary.csv", run_rows)
    (out_dir / "layer_feedback_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
