import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a family-level report for boundary seeds in init-only layer feedback studies."
    )
    parser.add_argument("--layer-feedback-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


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


def collect_float_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if value in {"", None}:
            continue
        values.append(float(value))
    return values


def boundary_family_key(seed_rows: Sequence[Dict[str, object]]) -> str:
    fixed_signature = next(str(row["final_shadow_signature"]) for row in seed_rows if str(row["run_label"]) == "fixed_shadow_baseline")
    prune_signature = next(str(row["final_actual_signature"]) for row in seed_rows if str(row["run_label"]) == "prune_only_baseline")
    return f"{prune_signature} -> {fixed_signature}"


def build_seed_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_seed: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["init_seed"])].append(row)

    out: List[Dict[str, object]] = []
    for seed, seed_rows in sorted(by_seed.items()):
        branch_label = str(seed_rows[0]["branch_label"])
        if branch_label != "boundary_mismatch":
            continue
        targeted_rows = [row for row in seed_rows if int(row["is_targeted"]) == 1]
        rewrite_rows = [row for row in targeted_rows if int(row["branch_rewrite"]) == 1]
        out.append(
            {
                "init_seed": seed,
                "boundary_family": boundary_family_key(seed_rows),
                "family_extra_layers": str(seed_rows[0]["family_extra_layers"]),
                "n_targeted_runs": len(targeted_rows),
                "n_branch_rewrites": len(rewrite_rows),
                "rewrite_whitelists": "none" if not rewrite_rows else ",".join(str(row["whitelist_label"]) for row in rewrite_rows),
                "mean_rewrite_extra_threshold": float(np.mean(collect_float_values(rewrite_rows, "final_shadow_extra_threshold_mean")))
                if collect_float_values(rewrite_rows, "final_shadow_extra_threshold_mean")
                else np.nan,
            }
        )
    return out


def build_family_rows(seed_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["boundary_family"])].append(row)

    out: List[Dict[str, object]] = []
    for family, rows in sorted(grouped.items()):
        whitelist_counter: Dict[str, int] = defaultdict(int)
        for row in rows:
            for whitelist in str(row["rewrite_whitelists"]).split(","):
                if whitelist and whitelist != "none":
                    whitelist_counter[whitelist] += 1
        out.append(
            {
                "boundary_family": family,
                "family_extra_layers": str(rows[0]["family_extra_layers"]),
                "n_seeds": len(rows),
                "mean_branch_rewrites": float(np.mean([float(row["n_branch_rewrites"]) for row in rows])),
                "rewrite_whitelist_union": "none" if not whitelist_counter else ",".join(sorted(whitelist_counter.keys(), key=str)),
                "rewrite_whitelist_consensus": "none" if not whitelist_counter else ",".join(
                    whitelist for whitelist, count in sorted(whitelist_counter.items()) if count == len(rows)
                ),
                "mean_rewrite_extra_threshold": float(np.mean(collect_float_values(rows, "mean_rewrite_extra_threshold")))
                if collect_float_values(rows, "mean_rewrite_extra_threshold")
                else np.nan,
            }
        )
    return out


def render_report(seed_rows: Sequence[Dict[str, object]], family_rows: Sequence[Dict[str, object]], out_dir: Path) -> str:
    lines: List[str] = []
    lines.append("# Init-Only Boundary Family Report")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("本报告只看 boundary seeds，把它们按 `prune signature -> fixed shadow signature` 归成 family，检查同型 boundary 是否共享同一组可触发 rewrite 的早期层。")
    lines.append("")
    lines.append("## 2. Family 汇总")
    lines.append("")
    lines.append("| family | extra layers | seeds | mean rewrites | rewrite union | rewrite consensus | mean rewrite extra-th |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | ---: |")
    for row in family_rows:
        lines.append(
            f"| {row['boundary_family']} | {row['family_extra_layers']} | {row['n_seeds']} | {fmt(float(row['mean_branch_rewrites']))} | {row['rewrite_whitelist_union']} | {row['rewrite_whitelist_consensus']} | {fmt(float(row['mean_rewrite_extra_threshold']))} |"
        )
    lines.append("")
    lines.append("## 3. Seed 级细节")
    lines.append("")
    lines.append("| seed | family | extra layers | rewrites | rewrite whitelists | mean rewrite extra-th |")
    lines.append("| ---: | --- | --- | ---: | --- | ---: |")
    for row in seed_rows:
        lines.append(
            f"| {row['init_seed']} | {row['boundary_family']} | {row['family_extra_layers']} | {row['n_branch_rewrites']} | {row['rewrite_whitelists']} | {fmt(float(row['mean_rewrite_extra_threshold']))} |"
        )
    lines.append("")
    lines.append("## 4. 当前判断")
    lines.append("")
    lines.append("- 如果同一个 family 里多个 seeds 共享相同的 rewrite consensus，就说明这不是单个 seed 的偶然现象，而是同型 boundary 的稳定因果结构。")
    lines.append("- 如果不同 family 的 rewrite consensus 明显不同，就说明 boundary 不是单一机制，而是至少包含多种 family-specific feedback route。")
    lines.append("")
    lines.append("## 5. 生成产物")
    lines.append("")
    lines.append(f"- boundary seed 表：`{out_dir / 'boundary_seed_summary.csv'}`")
    lines.append(f"- boundary family 表：`{out_dir / 'boundary_family_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path("results/init_only_lth_20260401/boundary_family_overview")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(args.layer_feedback_csv)
    seed_rows = build_seed_rows(rows)
    family_rows = build_family_rows(seed_rows)

    write_csv(out_dir / "boundary_seed_summary.csv", seed_rows)
    write_csv(out_dir / "boundary_family_summary.csv", family_rows)
    report = render_report(seed_rows=seed_rows, family_rows=family_rows, out_dir=out_dir)
    (out_dir / "boundary_family_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
