import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cross-seed overview for init-only feedback intervention runs."
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(run_root: Path) -> List[Dict[str, object]]:
    config = load_json(run_root / "run_config.json")
    path_rows = load_csv_rows(run_root / "analysis" / "intervention_path_summary.csv")
    agg_rows = load_csv_rows(run_root / "aggregate_summary.csv")
    agg_by_label = {str(row["run_label"]): row for row in agg_rows}

    out: List[Dict[str, object]] = []
    for path_row in path_rows:
        run_label = str(path_row["run_label"])
        agg_row = agg_by_label[run_label]
        out.append(
            {
                "init_seed": int(config["init_seed"]),
                "run_root": str(run_root),
                "run_label": run_label,
                "selected_test_acc": float(agg_row["selected_test_acc"]),
                "final_test_acc": float(agg_row["final_test_acc"]),
                "total_pruned": int(agg_row["total_pruned"]),
                "actual_path": str(path_row["actual_path"]),
                "shadow_path": str(path_row["shadow_path"]),
                "final_actual_signature": str(path_row["final_actual_signature"]),
                "final_shadow_signature": str(path_row["final_shadow_signature"]),
            }
        )
    return out


def render_report(rows: Sequence[Dict[str, object]], out_dir: Path) -> str:
    by_seed: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["init_seed"]), []).append(row)

    lines: List[str] = []
    lines.append("# Init-Only Feedback Overview")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("这个总览把多个 seed 的 feedback intervention 放在一起，用于区分四类情况：")
    lines.append("")
    lines.append("- 稳定正 branch：真实 pruning 后仍保留 late layer event")
    lines.append("- 稳定负 branch：真实 pruning 后仍不出现 late layer event")
    lines.append("- 边界 mismatch branch：真实 pruning 会改写 late layer event")
    lines.append("- timing-only mismatch：最终 branch 不变，但真实 pruning 会提前或推迟 commit 时机")
    lines.append("")
    lines.append("## 2. 总表")
    lines.append("")
    lines.append("| seed | run label | selected acc | final acc | total pruned | actual path | shadow path |")
    lines.append("| ---: | --- | ---: | ---: | ---: | --- | --- |")
    for row in sorted(rows, key=lambda item: (int(item["init_seed"]), str(item["run_label"]))):
        lines.append(
            f"| {row['init_seed']} | {row['run_label']} | {float(row['selected_test_acc']):.4f} | {float(row['final_test_acc']):.4f} | {row['total_pruned']} | {row['actual_path']} | {row['shadow_path']} |"
        )
    lines.append("")
    lines.append("## 3. 判断")
    lines.append("")
    for seed, seed_rows in sorted(by_seed.items()):
        row_by_label = {str(row["run_label"]): row for row in seed_rows}
        fixed_shadow_path = str(row_by_label["fixed_shadow_baseline"]["shadow_path"])
        prune_only_path = str(row_by_label["prune_only_baseline"]["actual_path"])
        e2_shadow_path = str(row_by_label["prune_until_e2_then_shadow"]["shadow_path"])
        fixed_shadow_final = row_by_label["fixed_shadow_baseline"]["final_shadow_signature"]
        prune_e4_final = row_by_label["prune_until_e4_then_shadow"]["final_shadow_signature"]
        if fixed_shadow_final == prune_e4_final:
            if fixed_shadow_path != prune_only_path:
                if e2_shadow_path == prune_only_path:
                    lines.append(f"- `seed {seed}`：最终 branch 保持不变，但真实 pruning 从 `epoch 2` 起就把 commit 时机提前到了 prune baseline，说明这是 timing-only mismatch。")
                else:
                    lines.append(f"- `seed {seed}`：最终 branch 保持不变，但 baseline shadow path 与 prune baseline path 不同，说明结构反馈改写了 commit timing。")
            else:
                lines.append(f"- `seed {seed}`：真实 pruning 到 `epoch 4` 后，最终 shadow signature 仍保持不变，说明该 branch 对结构反馈稳定。")
        else:
            lines.append(f"- `seed {seed}`：真实 pruning 到 `epoch 4` 后，最终 shadow signature 改变，说明该 seed 位于结构反馈边界。")
    lines.append("")
    lines.append("## 4. 结论")
    lines.append("")
    lines.append("- 这一总览把结构反馈效应进一步分成两种：一种改写最终 branch，另一种只改写 commit timing。")
    lines.append("- 边界 branch 的存在意味着：phase 不是纯观察现象，它也会被真实结构更新重新塑形。")
    lines.append("- timing-only mismatch 说明：即使最终 coarse phase 不变，真实 pruning 仍能改变 phase 形成的时间结构。")
    lines.append("")
    lines.append("## 5. 生成产物")
    lines.append("")
    lines.append(f"- 汇总表：`{out_dir / 'feedback_overview.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path("results/init_only_lth_20260401/feedback_overview")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for run_root in args.run_roots:
        rows.extend(summarize_run(run_root))

    write_csv(out_dir / "feedback_overview.csv", rows)
    report = render_report(rows, out_dir)
    (out_dir / "feedback_overview_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
