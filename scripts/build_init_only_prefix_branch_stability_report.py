import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prefix-branch stability report from a structured cohort summary."
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


def parse_epoch_signature(fine_phase: str) -> List[str]:
    parts = []
    for item in fine_phase.split(";"):
        _, signature = item.split(":", 1)
        parts.append(signature)
    return parts


def main() -> None:
    args = parse_args()
    cohort_csv = args.cohort_root / "cohort_seed_summary.csv"
    rows = load_csv_rows(cohort_csv)

    prefix_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        e2, e4, e6 = parse_epoch_signature(row["fine_phase"])
        shadow_e2, shadow_e4, shadow_e6 = parse_epoch_signature(row["shadow_fine_phase"])
        row["e2_sig"] = e2
        row["e4_sig"] = e4
        row["e6_sig"] = e6
        row["shadow_e2_sig"] = shadow_e2
        row["shadow_e4_sig"] = shadow_e4
        row["shadow_e6_sig"] = shadow_e6
        row["prefix"] = f"{e2} -> {e4}"
        prefix_groups[row["prefix"]].append(row)

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    lines: List[str] = []
    lines.append("# Init-Only Prefix Branch Stability Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告把 cohort 中的种子按 `epoch 2 -> epoch 4` 的早期 prefix 分组，专门区分两类后续分叉：")
    lines.append("")
    lines.append("- `stable late-commit`：`epoch 6` 才新增层，但 shadow 与真实 prune 一致")
    lines.append("- `boundary late-commit`：`epoch 6` 的新增层只出现在 shadow 或只出现在真实 prune")
    lines.append("")
    lines.append("## 2. Prefix 汇总")
    lines.append("")
    lines.append("| prefix | seeds | final paths | late-commit seeds | stable late-commit | boundary late-commit | shadow fine match |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for prefix in sorted(prefix_groups):
        group = prefix_groups[prefix]
        final_paths = sorted({row["fine_phase"] for row in group})
        stable_late_commit = 0
        boundary_late_commit = 0
        for row in group:
            actual_late_commit = row["e6_sig"] != row["e4_sig"]
            shadow_late_commit = row["shadow_e6_sig"] != row["shadow_e4_sig"]
            if actual_late_commit and shadow_late_commit and int(row["shadow_fine_match"]) == 1:
                stable_late_commit += 1
            elif actual_late_commit or shadow_late_commit:
                boundary_late_commit += 1
        late_commit_count = stable_late_commit + boundary_late_commit
        shadow_match_rate = sum(int(row["shadow_fine_match"]) for row in group) / len(group)
        summary_rows.append(
            {
                "prefix": prefix,
                "seed_count": len(group),
                "final_path_count": len(final_paths),
                "late_commit_seed_count": late_commit_count,
                "stable_late_commit_count": stable_late_commit,
                "boundary_late_commit_count": boundary_late_commit,
                "shadow_fine_match_rate": f"{shadow_match_rate:.4f}",
                "final_paths": " | ".join(final_paths),
            }
        )
        lines.append(
            f"| {prefix} | {len(group)} | {len(final_paths)} | {late_commit_count} | "
            f"{stable_late_commit} | {boundary_late_commit} | {shadow_match_rate:.4f} |"
        )

    lines.append("")
    lines.append("## 3. 多终点 prefix 细节")
    lines.append("")
    lines.append("| prefix | seed | final path | late-commit type |")
    lines.append("| --- | ---: | --- | --- |")

    for prefix in sorted(prefix_groups):
        group = prefix_groups[prefix]
        final_paths = sorted({row["fine_phase"] for row in group})
        if len(final_paths) <= 1:
            continue
        for row in sorted(group, key=lambda item: int(item["init_seed"])):
            actual_late_commit = row["e6_sig"] != row["e4_sig"]
            shadow_late_commit = row["shadow_e6_sig"] != row["shadow_e4_sig"]
            if actual_late_commit and shadow_late_commit and int(row["shadow_fine_match"]) == 1:
                late_commit_type = "stable late-commit"
            elif actual_late_commit or shadow_late_commit:
                late_commit_type = "boundary late-commit"
            else:
                late_commit_type = "no late-commit"
            detail_rows.append(
                {
                    "prefix": prefix,
                    "init_seed": row["init_seed"],
                    "final_path": row["fine_phase"],
                    "late_commit_type": late_commit_type,
                }
            )
            lines.append(
                f"| {prefix} | {row['init_seed']} | {row['fine_phase']} | {late_commit_type} |"
            )

    lines.append("")
    lines.append("## 4. 当前判断")
    lines.append("")
    lines.append("- 如果同一个早期 prefix 下已经出现多个最终路径，但绝大多数分叉都属于 `stable late-commit`，说明晚期分叉本身是 latent dynamics 的一部分，不应全部解释成 boundary 反馈。")
    lines.append("- 如果 `boundary late-commit` 只占少数，说明结构反馈改写主要发生在少量边界 seed 上，而不是同 prefix 的普遍命运。")
    lines.append("- 这一步把“late-commit 现象”与“feedback rewrite 现象”明确拆开：前者可以是稳定相，后者只是其中的边界子集。")
    lines.append("")
    lines.append("## 5. 生成产物")
    lines.append("")
    lines.append(f"- prefix 汇总：`{(args.cohort_root / 'prefix_branch_summary.csv').as_posix()}`")
    lines.append(f"- prefix 细节：`{(args.cohort_root / 'prefix_branch_details.csv').as_posix()}`")

    summary_path = args.cohort_root / "prefix_branch_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_path = args.cohort_root / "prefix_branch_details.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    report_path = args.cohort_root / "prefix_branch_stability_report_zh.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
