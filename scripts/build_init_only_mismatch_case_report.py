import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mismatch case-study report for init-only structured paired runs."
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


def event_signature(row: Dict[str, object], prefix: str, layer_count: int) -> str:
    layers = [str(layer_idx) for layer_idx in range(layer_count) if int(row[f"{prefix}_{layer_idx}"]) > 0]
    return "none" if not layers else "+".join(layers)


def build_epoch_compare_rows(run_root: Path, seed: int) -> List[Dict[str, object]]:
    prune_metrics = load_csv_rows(run_root / "prune_only" / f"init_seed_{seed}" / "metrics.csv")
    shadow_metrics = load_csv_rows(run_root / "fixed_shadow" / f"init_seed_{seed}" / "metrics.csv")
    layer_count = detect_layer_count(prune_metrics)
    prune_update_rows = [row for row in prune_metrics if int(row["is_update_epoch"]) == 1]
    shadow_update_rows = [row for row in shadow_metrics if int(row["is_shadow_update_epoch"]) == 1]

    rows: List[Dict[str, object]] = []
    for prune_row, shadow_row in zip(prune_update_rows, shadow_update_rows):
        epoch = int(prune_row["epoch"])
        prune_sig = event_signature(prune_row, "pruned", layer_count)
        shadow_sig = event_signature(shadow_row, "shadow_would_prune", layer_count)
        rows.append(
            {
                "epoch": epoch,
                "prune_signature": prune_sig,
                "shadow_signature": shadow_sig,
                "match": int(prune_sig == shadow_sig),
                **{
                    f"prune_{layer_idx}": int(prune_row[f"pruned_{layer_idx}"])
                    for layer_idx in range(layer_count)
                },
                **{
                    f"shadow_{layer_idx}": int(shadow_row[f"shadow_would_prune_{layer_idx}"])
                    for layer_idx in range(layer_count)
                },
                **{
                    f"shadow_threshold_{layer_idx}": float(shadow_row[f"shadow_threshold_{layer_idx}"])
                    for layer_idx in range(layer_count)
                },
                **{
                    f"shadow_candidate_{layer_idx}": int(shadow_row[f"shadow_candidate_{layer_idx}"])
                    for layer_idx in range(layer_count)
                },
            }
        )
    return rows


def divergence_epoch(rows: Sequence[Dict[str, object]]) -> int:
    for row in rows:
        if int(row["match"]) == 0:
            return int(row["epoch"])
    return -1


def summarize_shadow_layer_details(snapshot_rows: Sequence[Dict[str, object]], epoch: int, layer_idx: int) -> List[Dict[str, object]]:
    rows = [row for row in snapshot_rows if int(row["epoch"]) == epoch and int(row["layer_idx"]) == layer_idx and int(row["is_candidate"]) == 1]
    rows = sorted(rows, key=lambda row: int(row["candidate_rank"]))
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "epoch": epoch,
                "layer_idx": layer_idx,
                "neuron_idx": int(row["neuron_idx"]),
                "candidate_rank": int(row["candidate_rank"]),
                "would_prune": int(row["would_prune"]),
                "importance": float(row["importance"]),
                "ema_importance": float(row["ema_importance"]),
                "threshold": float(row["threshold"]),
                "delta_val_loss": float(row["delta_val_loss"]) if row["delta_val_loss"] != "" else np.nan,
            }
        )
    return out


def render_report(
    run_root: Path,
    config: Dict[str, object],
    mismatch_rows: Sequence[Dict[str, object]],
    per_seed_epoch_rows: Dict[int, List[Dict[str, object]]],
    detail_rows_by_seed: Dict[int, List[Dict[str, object]]],
    out_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Init-Only Shadow Mismatch Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告专门检查 `shadow` 与真实 `prune_only` 不一致的 seed，确认分歧出现在哪个 update 阶段、哪些层，以及这种分歧更像 early mismatch 还是 late feedback。")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seeds：`{config['init_seeds']}`")
    lines.append("")
    lines.append("## 3. Mismatch Seeds")
    lines.append("")
    if not mismatch_rows:
        lines.append("- 本轮没有 mismatch seeds。")
    else:
        lines.append("| seed | coarse phase | shadow coarse | fine phase | shadow fine |")
        lines.append("| ---: | --- | --- | --- | --- |")
        for row in mismatch_rows:
            lines.append(
                f"| {row['init_seed']} | {row['coarse_phase']} | {row['shadow_coarse_phase']} | {row['fine_phase']} | {row['shadow_fine_phase']} |"
            )
    lines.append("")
    lines.append("## 4. 逐 epoch 分歧")
    lines.append("")
    for seed, epoch_rows in per_seed_epoch_rows.items():
        first_div = divergence_epoch(epoch_rows)
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append(f"- 首个分歧 epoch：`{first_div}`")
        lines.append("")
        lines.append("| epoch | prune signature | shadow signature | match |")
        lines.append("| ---: | --- | --- | ---: |")
        for row in epoch_rows:
            lines.append(
                f"| {row['epoch']} | {row['prune_signature']} | {row['shadow_signature']} | {row['match']} |"
            )
        details = detail_rows_by_seed.get(seed, [])
        if details:
            lines.append("")
            lines.append("- 首个分歧 epoch 的 shadow 候选详情：")
            lines.append("")
            lines.append("| epoch | layer | neuron | rank | would prune | ema | threshold | delta loss |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in details:
                lines.append(
                    f"| {row['epoch']} | {row['layer_idx']} | {row['neuron_idx']} | {row['candidate_rank']} | {row['would_prune']} | {fmt(float(row['ema_importance']))} | {fmt(float(row['threshold']))} | {fmt(float(row['delta_val_loss']))} |"
                )
        lines.append("")
    lines.append("## 5. 判断")
    lines.append("")
    lines.append("- 如果前几个 update 完全一致、只在最后一个 update 才分歧，这更支持“真实 pruning 改写了后续 screening”而不是“shadow 一开始就错了”。")
    lines.append("- 这种 mismatch seed 是下一阶段 intervention 的最佳对象，因为它们正好落在 `latent dynamics` 与 `structural feedback` 的分界线上。")
    lines.append("")
    lines.append("## 6. 生成产物")
    lines.append("")
    lines.append(f"- mismatch seed 表：`{out_dir / 'mismatch_seed_summary.csv'}`")
    for seed in per_seed_epoch_rows:
        lines.append(f"- seed {seed} epoch 对比：`{out_dir / f'mismatch_seed_{seed}_epoch_compare.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    phase_rows = load_csv_rows(args.run_root / "analysis" / "seed_phase_summary.csv")
    mismatch_rows = [
        row for row in phase_rows
        if int(row["shadow_coarse_match"]) == 0 or int(row["shadow_fine_match"]) == 0
    ]

    per_seed_epoch_rows: Dict[int, List[Dict[str, object]]] = {}
    detail_rows_by_seed: Dict[int, List[Dict[str, object]]] = {}
    for row in mismatch_rows:
        seed = int(row["init_seed"])
        epoch_rows = build_epoch_compare_rows(args.run_root, seed)
        per_seed_epoch_rows[seed] = epoch_rows
        first_div = divergence_epoch(epoch_rows)
        if first_div > 0:
            div_row = next(item for item in epoch_rows if int(item["epoch"]) == first_div)
            layer_count = len([key for key in div_row if key.startswith("shadow_") and key.split("_")[1].isdigit() and key.count("_") == 1])
            detail_rows: List[Dict[str, object]] = []
            shadow_snapshots = load_csv_rows(args.run_root / "fixed_shadow" / f"init_seed_{seed}" / "shadow_prune_snapshots.csv")
            for layer_idx in range(layer_count):
                if int(div_row[f"shadow_{layer_idx}"]) > int(div_row[f"prune_{layer_idx}"]):
                    detail_rows.extend(
                        summarize_shadow_layer_details(
                            snapshot_rows=shadow_snapshots,
                            epoch=first_div,
                            layer_idx=layer_idx,
                        )
                    )
            detail_rows_by_seed[seed] = detail_rows
            write_csv(out_dir / f"mismatch_seed_{seed}_epoch_compare.csv", epoch_rows)
            if detail_rows:
                write_csv(out_dir / f"mismatch_seed_{seed}_shadow_candidate_details.csv", detail_rows)

    write_csv(out_dir / "mismatch_seed_summary.csv", mismatch_rows)
    report = render_report(
        run_root=args.run_root,
        config=config,
        mismatch_rows=mismatch_rows,
        per_seed_epoch_rows=per_seed_epoch_rows,
        detail_rows_by_seed=detail_rows_by_seed,
        out_dir=out_dir,
    )
    (out_dir / "mismatch_case_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
