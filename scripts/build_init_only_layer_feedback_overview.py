import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cross-seed overview for layer-targeted init-only feedback interventions."
    )
    parser.add_argument("--run-roots", nargs="+", type=Path, required=True)
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


def infer_branch_label(fixed_signature: str, prune_signature: str) -> str:
    if fixed_signature != prune_signature:
        return "boundary_mismatch"
    if fixed_signature.endswith("+3") or fixed_signature == "3":
        return "stable_positive"
    return "stable_negative"


def summarize_run(run_root: Path) -> List[Dict[str, object]]:
    config = load_json(run_root / "run_config.json")
    rows = load_csv_rows(run_root / "analysis" / "layer_feedback_summary.csv")
    by_label = {str(row["run_label"]): row for row in rows}

    fixed_signature = str(by_label["fixed_shadow_baseline"]["final_shadow_signature"])
    prune_signature = str(by_label["prune_only_baseline"]["final_actual_signature"])
    branch_label = infer_branch_label(fixed_signature=fixed_signature, prune_signature=prune_signature)
    family_extra_layers = str(by_label["fixed_shadow_baseline"].get("family_extra_layers", "none"))

    out: List[Dict[str, object]] = []
    for row in rows:
        run_label = str(row["run_label"])
        final_shadow_signature = str(row["final_shadow_signature"])
        is_targeted = int(run_label not in {"fixed_shadow_baseline", "prune_only_baseline"})
        branch_rewrite = int(is_targeted and final_shadow_signature != fixed_signature)
        match_prune = ""
        match_fixed = ""
        if is_targeted:
            match_prune = int(final_shadow_signature == prune_signature)
            match_fixed = int(final_shadow_signature == fixed_signature)
        out.append(
            {
                "init_seed": int(config["init_seed"]),
                "run_root": str(run_root),
                "branch_label": branch_label,
                "run_label": run_label,
                "whitelist_label": str(row["whitelist_label"]),
                "family_extra_layers": family_extra_layers,
                "total_pruned": int(row["total_pruned"]),
                "first_actual_signature": str(row["first_actual_signature"]),
                "final_actual_signature": str(row["final_actual_signature"]),
                "final_shadow_signature": final_shadow_signature,
                "final_shadow_extra_threshold_mean": row.get("final_shadow_extra_threshold_mean"),
                "final_shadow_extra_would_prune_sum": row.get("final_shadow_extra_would_prune_sum"),
                "is_targeted": is_targeted,
                "branch_rewrite": branch_rewrite,
                "match_fixed": match_fixed,
                "match_prune": match_prune,
            }
        )
    return out


def build_seed_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_seed: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["init_seed"]), []).append(row)

    out: List[Dict[str, object]] = []
    for seed, seed_rows in sorted(by_seed.items()):
        branch_label = str(seed_rows[0]["branch_label"])
        fixed_signature = next(str(row["final_shadow_signature"]) for row in seed_rows if str(row["run_label"]) == "fixed_shadow_baseline")
        prune_signature = next(str(row["final_actual_signature"]) for row in seed_rows if str(row["run_label"]) == "prune_only_baseline")
        targeted_rows = [row for row in seed_rows if int(row["is_targeted"]) == 1]
        rewrite_rows = [row for row in targeted_rows if int(row["branch_rewrite"]) == 1]
        nonzero_rows = [row for row in targeted_rows if int(row["total_pruned"]) > 0]
        out.append(
            {
                "init_seed": seed,
                "branch_label": branch_label,
                "fixed_signature": fixed_signature,
                "prune_signature": prune_signature,
                "family_extra_layers": str(seed_rows[0]["family_extra_layers"]),
                "n_targeted_runs": len(targeted_rows),
                "n_nonzero_targeted_runs": len(nonzero_rows),
                "n_branch_rewrites": len(rewrite_rows),
                "rewrite_whitelists": "none" if not rewrite_rows else ",".join(str(row["whitelist_label"]) for row in rewrite_rows),
                "nonzero_whitelists": "none" if not nonzero_rows else ",".join(str(row["whitelist_label"]) for row in nonzero_rows),
            }
        )
    return out


def render_report(
    detailed_rows: Sequence[Dict[str, object]],
    seed_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Init-Only Layer Feedback Overview")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("这个总览把稳定正支、边界支和稳定负支的 layer-targeted intervention 放在一起，回答两个问题：")
    lines.append("")
    lines.append("- branch rewrite 是否只出现在 boundary seed")
    lines.append("- rewrite 是否只由早期真正发生 pruning 的活跃层触发")
    lines.append("")
    lines.append("## 2. Seed 级汇总")
    lines.append("")
    lines.append("| seed | branch | fixed sig | prune sig | extra layers | nonzero targeted | branch rewrites | rewrite whitelists |")
    lines.append("| ---: | --- | --- | --- | --- | ---: | ---: | --- |")
    for row in seed_rows:
        lines.append(
            f"| {row['init_seed']} | {row['branch_label']} | {row['fixed_signature']} | {row['prune_signature']} | {row['family_extra_layers']} | {row['n_nonzero_targeted_runs']} | {row['n_branch_rewrites']} | {row['rewrite_whitelists']} |"
        )
    lines.append("")
    lines.append("## 3. Targeted 细表")
    lines.append("")
    lines.append("| seed | branch | whitelist | total pruned | first actual | final shadow | rewrite | match prune |")
    lines.append("| ---: | --- | --- | ---: | --- | --- | ---: | ---: |")
    for row in sorted(
        [row for row in detailed_rows if int(row["is_targeted"]) == 1],
        key=lambda item: (int(item["init_seed"]), str(item["whitelist_label"])),
    ):
        lines.append(
            f"| {row['init_seed']} | {row['branch_label']} | {row['whitelist_label']} | {row['total_pruned']} | {row['first_actual_signature']} | {row['final_shadow_signature']} | {row['branch_rewrite']} | {row['match_prune']} |"
        )
    lines.append("")
    lines.append("## 4. 当前判断")
    lines.append("")
    lines.append("- 如果稳定正支和稳定负支在所有 targeted runs 下都没有 rewrite，而 boundary seed 只在早期活跃层 whitelist 下发生 rewrite，就可以把结构反馈敏感性收缩成 boundary-only 机制。")
    lines.append("- 如果某些 whitelist 没有产生真实 pruning，同时也没有 rewrite，这说明关键不是“切到 prune 模式”，而是“该层真的发生了结构更新”。")
    lines.append("- family extra-layer threshold mean 在 rewrite 条件下回到 prune baseline，而在非 rewrite 条件下维持 dense baseline，是更通用的 gate 证据。")
    lines.append("")
    lines.append("## 5. 生成产物")
    lines.append("")
    lines.append(f"- 细表：`{out_dir / 'layer_feedback_overview.csv'}`")
    lines.append(f"- seed 汇总：`{out_dir / 'layer_feedback_seed_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path("results/init_only_lth_20260401/layer_feedback_overview")
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows: List[Dict[str, object]] = []
    for run_root in args.run_roots:
        detailed_rows.extend(summarize_run(run_root))
    seed_rows = build_seed_summary(detailed_rows)

    write_csv(out_dir / "layer_feedback_overview.csv", detailed_rows)
    write_csv(out_dir / "layer_feedback_seed_summary.csv", seed_rows)
    report = render_report(detailed_rows=detailed_rows, seed_rows=seed_rows, out_dir=out_dir)
    (out_dir / "layer_feedback_overview_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
