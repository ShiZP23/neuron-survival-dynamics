import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_RESULTS_ROOT = Path("results/init_only_lth_20260401")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cross-run overview for init-only structured paired pilots."
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


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def summarize_run(run_root: Path) -> Dict[str, object]:
    config = load_json(run_root / "run_config.json")
    mode_rows = load_csv_rows(run_root / "analysis" / "mode_summary.csv")
    paired_rows = load_csv_rows(run_root / "analysis" / "paired_seed_summary.csv")

    mode_lookup = {str(row["run_label"]): row for row in mode_rows}
    prune_signatures = sorted({str(row["prune_signature"]) for row in paired_rows})
    shadow_match_rate = float(np.mean([float(row["shadow_matches_prune_signature"]) for row in paired_rows])) if paired_rows else float("nan")
    prune_total_pruned = np.asarray([float(row["prune_total_pruned"]) for row in paired_rows], dtype=float) if paired_rows else np.asarray([], dtype=float)

    fixed = mode_lookup.get("fixed", {})
    prune = mode_lookup.get("prune_only", {})

    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "run_root": str(run_root),
        "n_seeds": len(paired_rows),
        "epochs": config["epochs"],
        "update_interval": config["update_interval"],
        "unique_prune_signatures": len(prune_signatures),
        "prune_signature_list": "; ".join(prune_signatures),
        "shadow_match_rate": shadow_match_rate,
        "fixed_selected_test_acc_mean": float(fixed.get("selected_test_acc_mean", float("nan"))),
        "prune_selected_test_acc_mean": float(prune.get("selected_test_acc_mean", float("nan"))),
        "fixed_final_test_acc_mean": float(fixed.get("final_test_acc_mean", float("nan"))),
        "prune_final_test_acc_mean": float(prune.get("final_test_acc_mean", float("nan"))),
        "prune_total_pruned_mean": float(prune_total_pruned.mean()) if prune_total_pruned.size else float("nan"),
    }


def build_report(rows: Sequence[Dict[str, object]], out_dir: Path) -> str:
    by_diversity = sorted(rows, key=lambda row: int(row["unique_prune_signatures"]), reverse=True)
    lines: List[str] = []
    lines.append("# Init-Only Structured Overview")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("这个总览把已经完成的 `init-only paired structured` pilots 放在一起，用于判断：在哪些任务 / 模型上，shadow alignment 已经成立，以及 phase diversity 从哪里开始出现。")
    lines.append("")
    lines.append("## 2. 运行摘要")
    lines.append("")
    lines.append("| dataset | model | seeds | signatures | shadow match | fixed selected acc | prune selected acc | mean pruned |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_seeds']} | {row['unique_prune_signatures']} | {fmt(float(row['shadow_match_rate']))} | {fmt(float(row['fixed_selected_test_acc_mean']))} | {fmt(float(row['prune_selected_test_acc_mean']))} | {fmt(float(row['prune_total_pruned_mean']))} |"
        )
    lines.append("")
    lines.append("## 3. 初步结论")
    lines.append("")
    most_diverse = by_diversity[0]
    lines.append(
        f"- 当前 phase diversity 最强的是 `{most_diverse['dataset']} + {most_diverse['model']}`，已观察到 `{most_diverse['unique_prune_signatures']}` 种粗结构签名。"
    )
    for row in rows:
        lines.append(
            f"- `{row['dataset']} + {row['model']}`：shadow match=`{fmt(float(row['shadow_match_rate']))}`，签名集合为 `{row['prune_signature_list']}`。"
        )
    lines.append("")
    lines.append("## 4. 判断")
    lines.append("")
    lines.append("- 如果某个 pilot 只有单一签名，但 shadow match 已经很高，它更像是“单相 regime”而不是机制失败。")
    lines.append("- 如果更难任务开始出现多签名，同时 shadow 仍完全对齐，这就说明 `latent phase` 正在跨到更真实的数据集。")
    lines.append("- 现阶段最合理的扩展方向是扩大 `CIFAR-10 structured` 的 seed 数，而不是继续在 `Fashion-MNIST` 上细调。")
    lines.append("")
    lines.append("## 5. 相关文件")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['dataset']} + {row['model']}`：`{row['run_root']}`")
    lines.append(f"- 汇总表：`{out_dir / 'structured_overview.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (DEFAULT_RESULTS_ROOT / "structured_overview")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(run_root) for run_root in args.run_roots]
    write_csv(out_dir / "structured_overview.csv", rows)
    report = build_report(rows, out_dir)
    (out_dir / "structured_overview_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
