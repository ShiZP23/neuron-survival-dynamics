import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_ROOT = Path("results/init_only_lth_20260401")
INITIAL_HIDDEN_SIZES = {
    "lenet300100": [300, 100],
    "smallconv": [64, 64, 128, 128],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an init-only paired structured report from a completed run root."
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


def parse_hidden_sizes(text: str) -> List[int]:
    return [int(value) for value in str(text).split(",") if str(value).strip()]


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def summarize_mode(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for run_label in sorted({str(row["run_label"]) for row in rows}):
        subset = [row for row in rows if str(row["run_label"]) == run_label]
        selected = np.asarray([float(row["selected_test_acc"]) for row in subset], dtype=float)
        final = np.asarray([float(row["final_test_acc"]) for row in subset], dtype=float)
        final_gap = np.asarray([float(row["final_minus_selected_loss"]) for row in subset], dtype=float)
        pruned = np.asarray([float(row["total_pruned"]) for row in subset], dtype=float)
        out.append(
            {
                "run_label": run_label,
                "n_seeds": len(subset),
                "selected_test_acc_mean": float(selected.mean()),
                "selected_test_acc_std": float(selected.std(ddof=0)),
                "final_test_acc_mean": float(final.mean()),
                "final_test_acc_std": float(final.std(ddof=0)),
                "final_minus_selected_loss_mean": float(final_gap.mean()),
                "total_pruned_mean": float(pruned.mean()),
            }
        )
    return out


def classify_prune_signature(hidden_sizes: List[int], initial_hidden_sizes: List[int]) -> str:
    hit_layers = [idx for idx, (initial, final) in enumerate(zip(initial_hidden_sizes, hidden_sizes)) if final < initial]
    if not hit_layers:
        return "no_prune"
    return "+".join(f"prune{idx}" for idx in hit_layers)


def summarize_shadow_run(shadow_run_dir: Path) -> Dict[str, object]:
    metrics_path = shadow_run_dir / "metrics.csv"
    shadow_path = shadow_run_dir / "shadow_prune_snapshots.csv"
    if not shadow_path.exists():
        return {
            "shadow_total_unique": 0,
            "shadow_signature": "no_prune",
            "shadow_last_update_epoch": None,
        }

    metrics_rows = load_csv_rows(metrics_path)
    snapshot_rows = load_csv_rows(shadow_path)
    update_epochs = [int(row["epoch"]) for row in metrics_rows if int(row["is_shadow_update_epoch"]) == 1]
    by_layer: Dict[int, set] = {}
    for row in snapshot_rows:
        if int(row["would_prune"]) != 1:
            continue
        layer_idx = int(row["layer_idx"])
        neuron_idx = int(row["neuron_idx"])
        by_layer.setdefault(layer_idx, set()).add(neuron_idx)

    signature_layers = [layer_idx for layer_idx, values in sorted(by_layer.items()) if values]
    signature = "no_prune" if not signature_layers else "+".join(f"prune{layer_idx}" for layer_idx in signature_layers)
    return {
        "shadow_total_unique": int(sum(len(values) for values in by_layer.values())),
        "shadow_signature": signature,
        "shadow_last_update_epoch": max(update_epochs) if update_epochs else None,
    }


def build_paired_rows(run_root: Path, aggregate_rows: Sequence[Dict[str, object]], model_name: str) -> List[Dict[str, object]]:
    by_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    for row in aggregate_rows:
        init_seed = int(row["init_seed"])
        by_seed.setdefault(init_seed, {})[str(row["run_label"])] = dict(row)

    initial_hidden_sizes = INITIAL_HIDDEN_SIZES[model_name]
    paired_rows: List[Dict[str, object]] = []
    for seed in sorted(by_seed):
        per_seed = by_seed[seed]
        fixed = per_seed.get("fixed")
        prune = per_seed.get("prune_only")
        shadow = per_seed.get("fixed_shadow")
        if fixed is None or prune is None:
            continue

        prune_signature = classify_prune_signature(
            hidden_sizes=parse_hidden_sizes(str(prune["final_hidden_sizes"])),
            initial_hidden_sizes=initial_hidden_sizes,
        )
        shadow_stats = {
            "shadow_total_unique": None,
            "shadow_signature": None,
            "shadow_last_update_epoch": None,
        }
        if shadow is not None:
            shadow_stats = summarize_shadow_run(run_root / "fixed_shadow" / f"init_seed_{seed}")

        paired_rows.append(
            {
                "init_seed": seed,
                "fixed_selected_test_acc": float(fixed["selected_test_acc"]),
                "prune_selected_test_acc": float(prune["selected_test_acc"]),
                "delta_selected_test_acc_prune_minus_fixed": float(prune["selected_test_acc"]) - float(fixed["selected_test_acc"]),
                "fixed_final_test_acc": float(fixed["final_test_acc"]),
                "prune_final_test_acc": float(prune["final_test_acc"]),
                "delta_final_test_acc_prune_minus_fixed": float(prune["final_test_acc"]) - float(fixed["final_test_acc"]),
                "fixed_final_minus_selected_loss": float(fixed["final_minus_selected_loss"]),
                "prune_final_minus_selected_loss": float(prune["final_minus_selected_loss"]),
                "delta_final_minus_selected_loss_prune_minus_fixed": float(prune["final_minus_selected_loss"]) - float(fixed["final_minus_selected_loss"]),
                "prune_total_pruned": int(prune["total_pruned"]),
                "prune_final_hidden_sizes": str(prune["final_hidden_sizes"]),
                "prune_signature": prune_signature,
                "shadow_total_unique": shadow_stats["shadow_total_unique"],
                "shadow_signature": shadow_stats["shadow_signature"],
                "shadow_last_update_epoch": shadow_stats["shadow_last_update_epoch"],
                "shadow_matches_prune_signature": int(
                    shadow_stats["shadow_signature"] == prune_signature
                ) if shadow_stats["shadow_signature"] is not None else None,
            }
        )
    return paired_rows


def plot_selected_scatter(paired_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    x = np.asarray([float(row["fixed_selected_test_acc"]) for row in paired_rows], dtype=float)
    y = np.asarray([float(row["prune_selected_test_acc"]) for row in paired_rows], dtype=float)
    seeds = [int(row["init_seed"]) for row in paired_rows]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(x, y, color="#1f77b4")
    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray", linewidth=1.0)
    for x_value, y_value, seed in zip(x, y, seeds):
        ax.annotate(str(seed), (x_value, y_value), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Fixed selected test acc")
    ax.set_ylabel("Prune selected test acc")
    ax.set_title("Paired fixed vs prune_only")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(
    run_root: Path,
    config: Dict[str, object],
    mode_rows: Sequence[Dict[str, object]],
    paired_rows: Sequence[Dict[str, object]],
    out_dir: Path,
) -> str:
    mode_lookup = {str(row["run_label"]): row for row in mode_rows}
    fixed = mode_lookup.get("fixed")
    prune = mode_lookup.get("prune_only")
    shadow = mode_lookup.get("fixed_shadow")
    shadow_match_rate = None
    if paired_rows and all(row["shadow_matches_prune_signature"] is not None for row in paired_rows):
        shadow_match_rate = float(np.mean([int(row["shadow_matches_prune_signature"]) for row in paired_rows]))

    lines: List[str] = []
    lines.append("# Init-Only Paired Structured Pilot Report")
    lines.append("")
    lines.append("## 1. 研究目的")
    lines.append("")
    lines.append("本轮实验把 `fixed`、`prune_only` 和 `fixed+shadow` 放在同一组初始化 seed 上配对比较，用于回答三个问题：")
    lines.append("")
    lines.append("- 只改初始化 seed 时，`prune_only` 是否会系统性偏离 `fixed`。")
    lines.append("- `prune_only` 是否已经出现可命名的粗结构签名。")
    lines.append("- `fixed+shadow` 是否能在不改结构的情况下复现 `prune_only` 的筛选签名。")
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集 / 模型：`{config['dataset']}` / `{config['model']}`")
    lines.append(f"- init seeds：`{config['init_seeds']}`")
    lines.append(f"- run labels：`{config['run_labels']}`")
    lines.append(f"- epochs / update interval：`{config['epochs']}` / `{config['update_interval']}`")
    lines.append(f"- batch size / lr：`{config['batch_size']}` / `{config['lr']}`")
    lines.append(f"- 固定随机性：split=`{config['split_seed']}`，train order=`{config['train_order_seed']}`，runtime=`{config['runtime_seed']}`")
    lines.append("")
    lines.append("## 3. 模式摘要")
    lines.append("")
    lines.append("| run label | seeds | selected acc mean±std | final acc mean±std | final-selected loss mean | mean total pruned |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in mode_rows:
        lines.append(
            f"| {row['run_label']} | {row['n_seeds']} | {fmt(float(row['selected_test_acc_mean']))} ± {fmt(float(row['selected_test_acc_std']))} | {fmt(float(row['final_test_acc_mean']))} ± {fmt(float(row['final_test_acc_std']))} | {fmt(float(row['final_minus_selected_loss_mean']))} | {fmt(float(row['total_pruned_mean']))} |"
        )
    lines.append("")
    lines.append("## 4. 配对观察")
    lines.append("")
    if fixed is not None and prune is not None:
        selected_delta = float(prune["selected_test_acc_mean"]) - float(fixed["selected_test_acc_mean"])
        final_delta = float(prune["final_test_acc_mean"]) - float(fixed["final_test_acc_mean"])
        lines.append(
            f"- `prune_only - fixed` 的平均 selected test acc 差为 `{fmt(selected_delta)}`，平均 final test acc 差为 `{fmt(final_delta)}`。"
        )
    if shadow is not None and shadow_match_rate is not None:
        lines.append(f"- `fixed+shadow` 与 `prune_only` 的结构签名匹配率为 `{fmt(shadow_match_rate)}`。")
    if paired_rows:
        best_gain = max(paired_rows, key=lambda row: float(row["delta_selected_test_acc_prune_minus_fixed"]))
        worst_gain = min(paired_rows, key=lambda row: float(row["delta_selected_test_acc_prune_minus_fixed"]))
        lines.append(
            f"- 对 selected test acc 来说，`prune_only` 相比 `fixed` 受益最大的 seed 是 `{best_gain['init_seed']}`，增益 `{fmt(float(best_gain['delta_selected_test_acc_prune_minus_fixed']))}`。"
        )
        lines.append(
            f"- 受损最大的 seed 是 `{worst_gain['init_seed']}`，差值 `{fmt(float(worst_gain['delta_selected_test_acc_prune_minus_fixed']))}`。"
        )
    lines.append("")
    lines.append("## 5. 种子级配对表")
    lines.append("")
    lines.append("| seed | fixed selected acc | prune selected acc | delta | prune signature | shadow signature | shadow match |")
    lines.append("| ---: | ---: | ---: | ---: | --- | --- | ---: |")
    for row in paired_rows:
        lines.append(
            f"| {row['init_seed']} | {fmt(float(row['fixed_selected_test_acc']))} | {fmt(float(row['prune_selected_test_acc']))} | {fmt(float(row['delta_selected_test_acc_prune_minus_fixed']))} | {row['prune_signature']} | {row['shadow_signature']} | {row['shadow_matches_prune_signature']} |"
        )
    lines.append("")
    lines.append("## 6. 当前结论")
    lines.append("")
    lines.append("- 这轮 paired pilot 的价值不在于最终精度绝对值，而在于确认 `init-only` 设定下，结构筛选与 dense shadow 是否已经能形成稳定对应。")
    lines.append("- 如果 `prune_signature` 和 `shadow_signature` 已经高度一致，下一轮就应该直接扩大 seed 数，而不是继续调模型。")
    lines.append("- 如果匹配率低，则说明图像域的 screening 机制还不够稳，需要先调 `update_interval`、`epsilon` 和 `max_candidates_per_layer`。")
    lines.append("")
    lines.append("## 7. 生成产物")
    lines.append("")
    lines.append(f"- 模式摘要：`{out_dir / 'mode_summary.csv'}`")
    lines.append(f"- 种子级配对：`{out_dir / 'paired_seed_summary.csv'}`")
    lines.append(f"- 配对散点图：`{out_dir / 'fixed_vs_prune_selected_acc.png'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.run_root / "run_config.json")
    aggregate_rows = load_csv_rows(args.run_root / "aggregate_summary.csv")
    mode_rows = summarize_mode(aggregate_rows)
    paired_rows = build_paired_rows(args.run_root, aggregate_rows, model_name=str(config["model"]))

    write_csv(out_dir / "mode_summary.csv", mode_rows)
    write_csv(out_dir / "paired_seed_summary.csv", paired_rows)
    if paired_rows:
        plot_selected_scatter(paired_rows, out_dir / "fixed_vs_prune_selected_acc.png")
    report = render_report(
        run_root=args.run_root,
        config=config,
        mode_rows=mode_rows,
        paired_rows=paired_rows,
        out_dir=out_dir,
    )
    (out_dir / "init_only_paired_report_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
