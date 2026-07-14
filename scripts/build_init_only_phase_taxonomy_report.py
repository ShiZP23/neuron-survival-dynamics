import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a phase-taxonomy report for an init-only structured paired run."
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
    if not metrics_rows:
        return 0
    count = 0
    while f"size_{count}" in metrics_rows[0]:
        count += 1
    return count


def layers_from_row(row: Dict[str, object], prefix: str, layer_count: int) -> List[int]:
    hit_layers = []
    for layer_idx in range(layer_count):
        value = row.get(f"{prefix}_{layer_idx}")
        if value is None:
            continue
        if float(value) > 0.0:
            hit_layers.append(layer_idx)
    return hit_layers


def format_layers(layers: Sequence[int]) -> str:
    if not layers:
        return "none"
    return "+".join(str(layer) for layer in layers)


def encode_event_path(metrics_rows: Sequence[Dict[str, object]], event_flag: str, prefix: str, layer_count: int) -> str:
    parts: List[str] = []
    for row in metrics_rows:
        if int(row[event_flag]) != 1:
            continue
        epoch = int(row["epoch"])
        layers = layers_from_row(row, prefix, layer_count)
        parts.append(f"e{epoch}:{format_layers(layers)}")
    return ";".join(parts) if parts else "none"


def encode_union_signature(metrics_rows: Sequence[Dict[str, object]], event_flag: str, prefix: str, layer_count: int) -> str:
    hit_layers = set()
    for row in metrics_rows:
        if int(row[event_flag]) != 1:
            continue
        hit_layers.update(layers_from_row(row, prefix, layer_count))
    return "no_prune" if not hit_layers else "+".join(f"prune{layer}" for layer in sorted(hit_layers))


def summarize_seed(run_root: Path, aggregate_row: Dict[str, object]) -> Dict[str, object]:
    init_seed = int(aggregate_row["init_seed"])
    fixed_run_dir = run_root / "fixed" / f"init_seed_{init_seed}"
    prune_run_dir = run_root / "prune_only" / f"init_seed_{init_seed}"
    shadow_run_dir = run_root / "fixed_shadow" / f"init_seed_{init_seed}"

    prune_metrics = load_csv_rows(prune_run_dir / "metrics.csv")
    shadow_metrics = load_csv_rows(shadow_run_dir / "metrics.csv")
    layer_count = detect_layer_count(prune_metrics)

    prune_event_path = encode_event_path(
        metrics_rows=prune_metrics,
        event_flag="is_update_epoch",
        prefix="pruned",
        layer_count=layer_count,
    )
    shadow_event_path = encode_event_path(
        metrics_rows=shadow_metrics,
        event_flag="is_shadow_update_epoch",
        prefix="shadow_would_prune",
        layer_count=layer_count,
    )
    coarse_signature = encode_union_signature(
        metrics_rows=prune_metrics,
        event_flag="is_update_epoch",
        prefix="pruned",
        layer_count=layer_count,
    )
    shadow_signature = encode_union_signature(
        metrics_rows=shadow_metrics,
        event_flag="is_shadow_update_epoch",
        prefix="shadow_would_prune",
        layer_count=layer_count,
    )

    fixed_summary = load_json(fixed_run_dir / "summary.json")
    prune_summary = load_json(prune_run_dir / "summary.json")
    shadow_summary = load_json(shadow_run_dir / "summary.json")

    return {
        "init_seed": init_seed,
        "coarse_phase": coarse_signature,
        "fine_phase": prune_event_path,
        "shadow_coarse_phase": shadow_signature,
        "shadow_fine_phase": shadow_event_path,
        "shadow_coarse_match": int(coarse_signature == shadow_signature),
        "shadow_fine_match": int(prune_event_path == shadow_event_path),
        "fixed_selected_test_acc": float(fixed_summary["selected_test_acc"]),
        "prune_selected_test_acc": float(prune_summary["selected_test_acc"]),
        "shadow_selected_test_acc": float(shadow_summary["selected_test_acc"]),
        "fixed_final_test_acc": float(fixed_summary["final_test_acc"]),
        "prune_final_test_acc": float(prune_summary["final_test_acc"]),
        "shadow_final_test_acc": float(shadow_summary["final_test_acc"]),
        "prune_total_pruned": int(prune_summary["total_pruned"]),
        "prune_final_hidden_sizes": str(prune_summary["final_hidden_sizes"]),
    }


def summarize_phase_groups(seed_rows: Sequence[Dict[str, object]], phase_key: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    phase_values = sorted({str(row[phase_key]) for row in seed_rows})
    for phase in phase_values:
        subset = [row for row in seed_rows if str(row[phase_key]) == phase]
        out.append(
            {
                "phase_key": phase_key,
                "phase": phase,
                "n_seeds": len(subset),
                "shadow_coarse_match_rate": float(np.mean([int(row["shadow_coarse_match"]) for row in subset])),
                "shadow_fine_match_rate": float(np.mean([int(row["shadow_fine_match"]) for row in subset])),
                "fixed_selected_test_acc_mean": float(np.mean([float(row["fixed_selected_test_acc"]) for row in subset])),
                "prune_selected_test_acc_mean": float(np.mean([float(row["prune_selected_test_acc"]) for row in subset])),
                "fixed_final_test_acc_mean": float(np.mean([float(row["fixed_final_test_acc"]) for row in subset])),
                "prune_final_test_acc_mean": float(np.mean([float(row["prune_final_test_acc"]) for row in subset])),
                "prune_total_pruned_mean": float(np.mean([float(row["prune_total_pruned"]) for row in subset])),
            }
        )
    return out


def render_report(
    run_root: Path,
    config: Dict[str, object],
    seed_rows: Sequence[Dict[str, object]],
    coarse_rows: Sequence[Dict[str, object]],
    fine_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    coarse_counter = Counter(str(row["coarse_phase"]) for row in seed_rows)
    fine_counter = Counter(str(row["fine_phase"]) for row in seed_rows)
    coarse_match_rate = float(np.mean([int(row["shadow_coarse_match"]) for row in seed_rows])) if seed_rows else float("nan")
    fine_match_rate = float(np.mean([int(row["shadow_fine_match"]) for row in seed_rows])) if seed_rows else float("nan")

    lines: List[str] = []
    lines.append("# Init-Only Phase Taxonomy Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告针对一个 `init-only paired structured` run，抽取两个层级的 phase：")
    lines.append("")
    lines.append("- `coarse phase`：整个 run 中哪些层曾经被 prune")
    lines.append("- `fine phase`：每个 update epoch 的逐步 pruning 路径")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seeds：`{config['init_seeds']}`")
    lines.append(f"- epochs / update interval：`{config['epochs']}` / `{config['update_interval']}`")
    lines.append("")
    lines.append("## 3. 总体观察")
    lines.append("")
    lines.append(f"- coarse phase 共发现 `{len(coarse_counter)}` 类：`{'; '.join(sorted(coarse_counter.keys()))}`")
    lines.append(f"- fine phase 共发现 `{len(fine_counter)}` 类。")
    lines.append(f"- shadow 对 coarse phase 的匹配率为 `{fmt(coarse_match_rate)}`。")
    lines.append(f"- shadow 对 fine phase 的匹配率为 `{fmt(fine_match_rate)}`。")
    lines.append("")
    lines.append("## 4. Coarse Phase 汇总")
    lines.append("")
    lines.append("| phase | n seeds | shadow coarse match | fixed selected acc | prune selected acc | mean pruned |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in coarse_rows:
        lines.append(
            f"| {row['phase']} | {row['n_seeds']} | {fmt(float(row['shadow_coarse_match_rate']))} | {fmt(float(row['fixed_selected_test_acc_mean']))} | {fmt(float(row['prune_selected_test_acc_mean']))} | {fmt(float(row['prune_total_pruned_mean']))} |"
        )
    lines.append("")
    lines.append("## 5. Fine Phase 汇总")
    lines.append("")
    lines.append("| phase | n seeds | shadow fine match | fixed final acc | prune final acc |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in fine_rows:
        lines.append(
            f"| {row['phase']} | {row['n_seeds']} | {fmt(float(row['shadow_fine_match_rate']))} | {fmt(float(row['fixed_final_test_acc_mean']))} | {fmt(float(row['prune_final_test_acc_mean']))} |"
        )
    lines.append("")
    lines.append("## 6. 种子级表")
    lines.append("")
    lines.append("| seed | coarse phase | fine phase | shadow coarse | shadow fine | coarse match | fine match |")
    lines.append("| ---: | --- | --- | --- | --- | ---: | ---: |")
    for row in seed_rows:
        lines.append(
            f"| {row['init_seed']} | {row['coarse_phase']} | {row['fine_phase']} | {row['shadow_coarse_phase']} | {row['shadow_fine_phase']} | {row['shadow_coarse_match']} | {row['shadow_fine_match']} |"
        )
    lines.append("")
    lines.append("## 7. 当前判断")
    lines.append("")
    lines.append("- 如果 coarse phase 已经分化，说明不同初始化 seed 的结构命运开始分成离散簇。")
    lines.append("- 如果 fine phase 也开始分化，说明差异不只是最终剪到哪些层，而是逐步 screening 路径本身就不同。")
    lines.append("- 如果 shadow 对 coarse 与 fine phase 都高度对齐，就可以把 phase 解释为 dense latent dynamics 的显影，而不是 pruning 人为造出的结果。")
    lines.append("")
    lines.append("## 8. 生成产物")
    lines.append("")
    lines.append(f"- 种子级 phase 表：`{out_dir / 'seed_phase_summary.csv'}`")
    lines.append(f"- coarse group 表：`{out_dir / 'coarse_phase_summary.csv'}`")
    lines.append(f"- fine group 表：`{out_dir / 'fine_phase_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    aggregate_rows = load_csv_rows(args.run_root / "aggregate_summary.csv")
    prune_rows = [row for row in aggregate_rows if str(row["run_label"]) == "prune_only"]
    seed_rows = [summarize_seed(args.run_root, row) for row in prune_rows]
    seed_rows = sorted(seed_rows, key=lambda row: int(row["init_seed"]))
    coarse_rows = summarize_phase_groups(seed_rows, phase_key="coarse_phase")
    fine_rows = summarize_phase_groups(seed_rows, phase_key="fine_phase")

    write_csv(out_dir / "seed_phase_summary.csv", seed_rows)
    write_csv(out_dir / "coarse_phase_summary.csv", coarse_rows)
    write_csv(out_dir / "fine_phase_summary.csv", fine_rows)
    report = render_report(
        run_root=args.run_root,
        config=config,
        seed_rows=seed_rows,
        coarse_rows=coarse_rows,
        fine_rows=fine_rows,
        out_dir=out_dir,
    )
    (out_dir / "phase_taxonomy_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
