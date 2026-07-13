import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set


UPDATE_EPOCHS: Sequence[int] = (2, 4, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cohort-level threshold-persistence report for init-only structured runs."
    )
    parser.add_argument(
        "--cohort-root",
        type=Path,
        default=Path("results/init_only_lth_20260401/structured_cohort_overview"),
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_signature_map(phase: str) -> Dict[int, Set[int]]:
    mapping: Dict[int, Set[int]] = {}
    for part in phase.split(";"):
        epoch_text, signature = part.split(":", 1)
        epoch = int(epoch_text[1:])
        mapping[epoch] = {int(item) for item in signature.split("+")} if signature else set()
    return mapping


def sign_char(value: float) -> str:
    return "+" if value > 0 else "-"


def format_layer_set(layer_indices: Sequence[int]) -> str:
    if not layer_indices:
        return "none"
    return "+".join(str(layer_idx) for layer_idx in layer_indices)


def classify_event(
    actual_late: bool,
    shadow_late: bool,
    actual_sign_traj: str,
    shadow_sign_traj: str,
) -> str:
    if actual_late and shadow_late:
        return "stable_late_commit"
    if shadow_late and not actual_late:
        if actual_sign_traj == "---":
            return "boundary_absent_in_actual"
        if actual_sign_traj[1] == "+" or actual_sign_traj[2] == "+":
            return "boundary_timing_gap"
        return "boundary_shadow_only_other"
    if actual_late and not shadow_late:
        if shadow_sign_traj == "---":
            return "boundary_absent_in_shadow"
        if shadow_sign_traj[1] == "+" or shadow_sign_traj[2] == "+":
            return "boundary_actual_timing_gap"
        return "boundary_actual_only_other"
    return "no_late_event"


def load_epoch_rows(metrics_path: Path, threshold_prefix: str) -> Dict[int, Dict[str, float]]:
    rows = load_csv_rows(metrics_path)
    epoch_rows: Dict[int, Dict[str, float]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        if epoch not in UPDATE_EPOCHS:
            continue
        parsed = {"epoch": float(epoch)}
        for key, value in row.items():
            if key.startswith(threshold_prefix):
                parsed[key] = float(value)
        epoch_rows[epoch] = parsed
    return epoch_rows


def main() -> None:
    args = parse_args()
    cohort_root = args.cohort_root
    cohort_rows = load_csv_rows(cohort_root / "cohort_seed_summary.csv")

    event_rows: List[Dict[str, object]] = []
    for row in cohort_rows:
        run_root = Path(row["run_root"])
        init_seed = int(row["init_seed"])
        actual_map = parse_signature_map(row["fine_phase"])
        shadow_map = parse_signature_map(row["shadow_fine_phase"])

        actual_extra_layers = sorted(actual_map[6] - actual_map[4])
        shadow_extra_layers = sorted(shadow_map[6] - shadow_map[4])
        union_layers = sorted(set(actual_extra_layers) | set(shadow_extra_layers))
        if not union_layers:
            continue

        fixed_metrics = load_epoch_rows(
            run_root / "fixed_shadow" / f"init_seed_{init_seed}" / "metrics.csv",
            threshold_prefix="shadow_threshold_",
        )
        prune_metrics = load_epoch_rows(
            run_root / "prune_only" / f"init_seed_{init_seed}" / "metrics.csv",
            threshold_prefix="threshold_",
        )

        for layer_idx in union_layers:
            actual_values = [prune_metrics[epoch][f"threshold_{layer_idx}"] for epoch in UPDATE_EPOCHS]
            shadow_values = [fixed_metrics[epoch][f"shadow_threshold_{layer_idx}"] for epoch in UPDATE_EPOCHS]
            actual_sign_traj = "".join(sign_char(value) for value in actual_values)
            shadow_sign_traj = "".join(sign_char(value) for value in shadow_values)
            actual_late = layer_idx in actual_extra_layers
            shadow_late = layer_idx in shadow_extra_layers
            event_type = classify_event(
                actual_late=actual_late,
                shadow_late=shadow_late,
                actual_sign_traj=actual_sign_traj,
                shadow_sign_traj=shadow_sign_traj,
            )
            event_rows.append(
                {
                    "run_root": row["run_root"],
                    "init_seed": init_seed,
                    "epoch24_prefix": row["epoch24_prefix"],
                    "coarse_phase": row["coarse_phase"],
                    "fine_phase": row["fine_phase"],
                    "shadow_fine_phase": row["shadow_fine_phase"],
                    "layer_idx": layer_idx,
                    "actual_late": int(actual_late),
                    "shadow_late": int(shadow_late),
                    "event_type": event_type,
                    "actual_sign_traj": actual_sign_traj,
                    "shadow_sign_traj": shadow_sign_traj,
                    "actual_threshold_e2": actual_values[0],
                    "actual_threshold_e4": actual_values[1],
                    "actual_threshold_e6": actual_values[2],
                    "shadow_threshold_e2": shadow_values[0],
                    "shadow_threshold_e4": shadow_values[1],
                    "shadow_threshold_e6": shadow_values[2],
                    "threshold_gap_e4": shadow_values[1] - actual_values[1],
                    "threshold_gap_e6": shadow_values[2] - actual_values[2],
                }
            )

    event_type_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    prefix_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        event_type_groups[str(row["event_type"])].append(row)
        prefix_groups[str(row["epoch24_prefix"])].append(row)

    event_summary_rows: List[Dict[str, object]] = []
    for event_type, rows in sorted(event_type_groups.items()):
        actual_traj_counter = Counter(str(row["actual_sign_traj"]) for row in rows)
        shadow_traj_counter = Counter(str(row["shadow_sign_traj"]) for row in rows)
        seed_list = sorted({int(row["init_seed"]) for row in rows})
        prefix_list = sorted({str(row["epoch24_prefix"]) for row in rows})
        event_summary_rows.append(
            {
                "event_type": event_type,
                "event_count": len(rows),
                "seed_count": len(seed_list),
                "seeds": ",".join(str(seed) for seed in seed_list),
                "prefix_count": len(prefix_list),
                "prefixes": " | ".join(prefix_list),
                "modal_actual_sign_traj": actual_traj_counter.most_common(1)[0][0],
                "modal_shadow_sign_traj": shadow_traj_counter.most_common(1)[0][0],
                "mean_actual_threshold_e6": f"{sum(float(row['actual_threshold_e6']) for row in rows) / len(rows):.4f}",
                "mean_shadow_threshold_e6": f"{sum(float(row['shadow_threshold_e6']) for row in rows) / len(rows):.4f}",
                "mean_threshold_gap_e4": f"{sum(float(row['threshold_gap_e4']) for row in rows) / len(rows):.4f}",
                "mean_threshold_gap_e6": f"{sum(float(row['threshold_gap_e6']) for row in rows) / len(rows):.4f}",
            }
        )

    prefix_summary_rows: List[Dict[str, object]] = []
    for prefix, rows in sorted(prefix_groups.items()):
        counts = Counter(str(row["event_type"]) for row in rows)
        prefix_summary_rows.append(
            {
                "epoch24_prefix": prefix,
                "event_count": len(rows),
                "stable_late_commit": counts.get("stable_late_commit", 0),
                "boundary_absent_in_actual": counts.get("boundary_absent_in_actual", 0),
                "boundary_timing_gap": counts.get("boundary_timing_gap", 0),
                "other_boundary": sum(
                    counts.get(name, 0)
                    for name in (
                        "boundary_shadow_only_other",
                        "boundary_absent_in_shadow",
                        "boundary_actual_timing_gap",
                        "boundary_actual_only_other",
                    )
                ),
            }
        )

    event_rows_sorted = sorted(
        event_rows,
        key=lambda row: (str(row["epoch24_prefix"]), int(row["init_seed"]), int(row["layer_idx"])),
    )

    lines: List[str] = []
    lines.append("# Init-Only Threshold Persistence Cohort Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告把 `late-layer event` 提升成 cohort 级主机制变量。每个 event 对应“某一层在 `epoch 6` 相对 `epoch 4` 新增”，并比较它在真实 `prune_only` 与 `fixed+shadow` 中的 threshold sign persistence。")
    lines.append("")
    lines.append("我们关心三类事件：")
    lines.append("")
    lines.append("- `stable_late_commit`：该层在真实 prune 和 shadow 中都属于晚期新增层")
    lines.append("- `boundary_absent_in_actual`：该层只在 shadow 中晚期出现，真实 prune 的 threshold 一直未翻正")
    lines.append("- `boundary_timing_gap`：该层只在 shadow 中表现为“晚期新增”，但真实 prune 已在更早 update 中翻正并提交")
    lines.append("")
    lines.append("## 2. Event 类型汇总")
    lines.append("")
    lines.append("| event type | events | seeds | modal actual sign | modal shadow sign | mean actual th@e6 | mean shadow th@e6 | mean gap@e4 | mean gap@e6 |")
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for row in event_summary_rows:
        lines.append(
            f"| {row['event_type']} | {row['event_count']} | {row['seed_count']} | "
            f"{row['modal_actual_sign_traj']} | {row['modal_shadow_sign_traj']} | "
            f"{row['mean_actual_threshold_e6']} | {row['mean_shadow_threshold_e6']} | "
            f"{row['mean_threshold_gap_e4']} | {row['mean_threshold_gap_e6']} |"
        )

    lines.append("")
    lines.append("## 3. Prefix 汇总")
    lines.append("")
    lines.append("| prefix | events | stable late-commit | absent in actual | timing gap | other boundary |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in prefix_summary_rows:
        lines.append(
            f"| {row['epoch24_prefix']} | {row['event_count']} | {row['stable_late_commit']} | "
            f"{row['boundary_absent_in_actual']} | {row['boundary_timing_gap']} | {row['other_boundary']} |"
        )

    lines.append("")
    lines.append("## 4. Event 级细节")
    lines.append("")
    lines.append("| seed | prefix | layer | event type | actual sign | shadow sign | actual th@e6 | shadow th@e6 |")
    lines.append("| ---: | --- | ---: | --- | --- | --- | ---: | ---: |")
    for row in event_rows_sorted:
        lines.append(
            f"| {row['init_seed']} | {row['epoch24_prefix']} | {row['layer_idx']} | {row['event_type']} | "
            f"{row['actual_sign_traj']} | {row['shadow_sign_traj']} | "
            f"{float(row['actual_threshold_e6']):.4f} | {float(row['shadow_threshold_e6']):.4f} |"
        )

    lines.append("")
    lines.append("## 5. 当前判断")
    lines.append("")
    lines.append("- 如果 `stable_late_commit` 的模态轨迹稳定为 `--+ / --+`，说明真正的稳定 late branch 可以直接理解成 target-layer threshold 在最后一次 update 翻正。")
    lines.append("- 如果 `boundary_absent_in_actual` 的模态轨迹是 `--- / --+`，说明这类边界并不是 timing 差，而是 shadow 中出现了真实 prune 没有出现的阈值翻正。")
    lines.append("- 如果 `boundary_timing_gap` 的模态轨迹是 `-++ / --+`，说明这类边界本质上是 commit timing 被真实 pruning 提前了，而不是最终 target layer 身份不同。")
    lines.append("- 因此，`threshold-sign persistence` 不只是 late-commit 的相关变量，而是已经足够把 stable branch、absence boundary 和 timing boundary 分成三种机制类型。")
    lines.append("")
    lines.append("## 6. 生成产物")
    lines.append("")
    lines.append(f"- event 汇总：`{(cohort_root / 'threshold_persistence_event_summary.csv').as_posix()}`")
    lines.append(f"- prefix 汇总：`{(cohort_root / 'threshold_persistence_prefix_summary.csv').as_posix()}`")
    lines.append(f"- event 明细：`{(cohort_root / 'threshold_persistence_event_details.csv').as_posix()}`")

    event_summary_path = cohort_root / "threshold_persistence_event_summary.csv"
    with event_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_summary_rows)

    prefix_summary_path = cohort_root / "threshold_persistence_prefix_summary.csv"
    with prefix_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prefix_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prefix_summary_rows)

    details_path = cohort_root / "threshold_persistence_event_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows_sorted)

    report_path = cohort_root / "threshold_persistence_cohort_report_zh.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
