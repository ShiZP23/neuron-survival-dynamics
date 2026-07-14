import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_ROOT = Path("results/init_only_lth_20260401")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Phase-0 dense init-only report from a completed sweep."
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="lenet300100")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def latest_run_root(results_root: Path, dataset: str, model: str) -> Path:
    base = results_root / dataset / model / "dense_init_only"
    if not base.exists():
        raise FileNotFoundError(f"No run root found under {base}")
    candidates = sorted(path for path in base.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No timestamped runs found under {base}")
    return candidates[-1]


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


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None or np.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def safe_corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def aggregate_epoch_histories(metric_rows_by_seed: Dict[int, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    metric_names = ["train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
    epoch_to_rows: Dict[int, List[Dict[str, object]]] = {}
    for rows in metric_rows_by_seed.values():
        for row in rows:
            epoch = int(row["epoch"])
            epoch_to_rows.setdefault(epoch, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for epoch in sorted(epoch_to_rows):
        out_row: Dict[str, object] = {"epoch": epoch, "n_seeds": len(epoch_to_rows[epoch])}
        for name in metric_names:
            values = np.asarray([float(row[name]) for row in epoch_to_rows[epoch]], dtype=float)
            out_row[f"{name}_mean"] = float(values.mean())
            out_row[f"{name}_std"] = float(values.std(ddof=0))
            out_row[f"{name}_min"] = float(values.min())
            out_row[f"{name}_max"] = float(values.max())
        summary_rows.append(out_row)
    return summary_rows


def build_seed_rankings(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = [dict(row) for row in summary_rows]
    rows.sort(
        key=lambda row: (
            float(row["selected_test_acc"]),
            -float(row["selected_test_loss"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["selected_test_acc_rank"] = rank
    final_sorted = sorted(rows, key=lambda row: float(row["final_test_acc"]), reverse=True)
    final_rank_by_seed = {int(row["init_seed"]): rank for rank, row in enumerate(final_sorted, start=1)}
    for row in rows:
        row["final_test_acc_rank"] = final_rank_by_seed[int(row["init_seed"])]
    return rows


def build_core_statistics(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = [
        "selected_test_acc",
        "final_test_acc",
        "selected_test_loss",
        "final_test_loss",
        "selected_minus_final_acc",
        "final_minus_selected_loss",
        "selected_epoch",
        "best_val_loss",
    ]
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        stats = summarize_distribution([float(row[metric]) for row in summary_rows])
        rows.append({"metric": metric, **stats})
    return rows


def plot_curves(epoch_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in epoch_rows], dtype=int)
    val_loss_mean = np.asarray([float(row["val_loss_mean"]) for row in epoch_rows], dtype=float)
    val_loss_std = np.asarray([float(row["val_loss_std"]) for row in epoch_rows], dtype=float)
    test_acc_mean = np.asarray([float(row["test_acc_mean"]) for row in epoch_rows], dtype=float)
    test_acc_std = np.asarray([float(row["test_acc_std"]) for row in epoch_rows], dtype=float)
    train_acc_mean = np.asarray([float(row["train_acc_mean"]) for row in epoch_rows], dtype=float)
    train_acc_std = np.asarray([float(row["train_acc_std"]) for row in epoch_rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, val_loss_mean, label="val loss", color="#1f77b4")
    axes[0].fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2)
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_acc_mean, label="train acc", color="#2ca02c")
    axes[1].fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, alpha=0.15)
    axes[1].plot(epochs, test_acc_mean, label="test acc", color="#d62728")
    axes[1].fill_between(epochs, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std, alpha=0.15)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_seed_scatter(summary_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    selected = np.asarray([float(row["selected_test_acc"]) for row in summary_rows], dtype=float)
    final = np.asarray([float(row["final_test_acc"]) for row in summary_rows], dtype=float)
    seeds = [int(row["init_seed"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(selected, final, color="#1f77b4")
    lower = min(selected.min(), final.min())
    upper = max(selected.max(), final.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray", linewidth=1.0)
    for x_value, y_value, seed in zip(selected, final, seeds):
        ax.annotate(str(seed), (x_value, y_value), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Selected test acc")
    ax.set_ylabel("Final test acc")
    ax.set_title("Init-seed selected vs final accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(
    run_root: Path,
    out_dir: Path,
    config: Dict[str, object],
    summary_rows: Sequence[Dict[str, object]],
    core_stats: Sequence[Dict[str, object]],
    epoch_rows: Sequence[Dict[str, object]],
    rankings: Sequence[Dict[str, object]],
) -> str:
    selected_test_acc = [float(row["selected_test_acc"]) for row in summary_rows]
    final_test_acc = [float(row["final_test_acc"]) for row in summary_rows]
    selected_minus_final_acc = [float(row["selected_minus_final_acc"]) for row in summary_rows]
    final_minus_selected_loss = [float(row["final_minus_selected_loss"]) for row in summary_rows]
    corr = safe_corr(selected_test_acc, final_test_acc)
    mean_selected_minus_final_acc = float(np.mean(selected_minus_final_acc))
    mean_final_minus_selected_loss = float(np.mean(final_minus_selected_loss))

    best_seed = max(rankings, key=lambda row: float(row["selected_test_acc"]))
    worst_seed = min(rankings, key=lambda row: float(row["selected_test_acc"]))

    lines: List[str] = []
    lines.append("# Init-Only Phase-0 Pilot Report")
    lines.append("")
    lines.append("## 1. 研究目的")
    lines.append("")
    lines.append(
        "本轮 pilot 只固定数据划分、训练顺序和优化协议，只改变模型初始化 seed，先验证 `dense init-only` 设定是否能稳定跑通，并估计初始化单独造成的性能波动范围。"
    )
    lines.append("")
    lines.append("## 2. 运行配置")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append(f"- 数据集：`{config['dataset']}`")
    lines.append(f"- 模型：`{config['model']}`")
    lines.append(f"- init seeds：`{config['init_seeds']}`")
    lines.append(f"- epochs：`{config['epochs']}`")
    lines.append(f"- batch size：`{config['batch_size']}`")
    lines.append(f"- lr / weight decay：`{config['lr']}` / `{config['weight_decay']}`")
    lines.append(
        f"- train / val / test 样本数：`{config['train_size']}` / `{config['val_size']}` / `{config['test_size']}`"
    )
    lines.append(
        f"- 固定种子：split=`{config['split_seed']}`，train order=`{config['train_order_seed']}`，runtime=`{config['runtime_seed']}`"
    )
    lines.append("")
    lines.append("## 3. 首轮观察")
    lines.append("")
    lines.append("- 这里的 `selected` epoch 定义为 `val loss` 最低的 epoch，而不是 `test acc` 最高的 epoch。")
    lines.append(
        f"- 本次 sweep 共运行 `{len(summary_rows)}` 个 init seeds；selected test acc 均值为 `{fmt_float(float(np.mean(selected_test_acc)))}`，标准差为 `{fmt_float(float(np.std(selected_test_acc)))}`。"
    )
    lines.append(
        f"- final test acc 均值为 `{fmt_float(float(np.mean(final_test_acc)))}`，标准差为 `{fmt_float(float(np.std(final_test_acc)))}`。"
    )
    if mean_selected_minus_final_acc >= 0.0:
        lines.append(
            f"- `selected - final acc` 均值为 `{fmt_float(mean_selected_minus_final_acc)}`，按测试精度看，终点平均略低于 selected 点。"
        )
    else:
        lines.append(
            f"- `selected - final acc` 均值为 `{fmt_float(mean_selected_minus_final_acc)}`，按测试精度看，终点平均反而略高于 selected 点。"
        )
    if mean_final_minus_selected_loss >= 0.0:
        lines.append(
            f"- `final - selected loss` 均值为 `{fmt_float(mean_final_minus_selected_loss)}`，按测试损失看，终点平均略差于 selected 点。"
        )
    else:
        lines.append(
            f"- `final - selected loss` 均值为 `{fmt_float(mean_final_minus_selected_loss)}`，按测试损失看，终点平均略优于 selected 点。"
        )
    if corr is None:
        lines.append("- `selected` 与 `final` 测试精度之间的相关性当前不可稳定估计。")
    else:
        lines.append(f"- `selected` 与 `final` 测试精度的 Pearson 相关为 `{fmt_float(corr)}`。")
    lines.append(
        f"- 当前最佳 seed 为 `{best_seed['init_seed']}`，selected test acc=`{fmt_float(float(best_seed['selected_test_acc']))}`；最弱 seed 为 `{worst_seed['init_seed']}`，selected test acc=`{fmt_float(float(worst_seed['selected_test_acc']))}`。"
    )
    lines.append("")
    lines.append("## 4. 指标摘要")
    lines.append("")
    lines.append("| metric | mean | std | min | p25 | median | p75 | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in core_stats:
        lines.append(
            "| {metric} | {mean} | {std} | {min_} | {p25} | {median} | {p75} | {max_} |".format(
                metric=row["metric"],
                mean=fmt_float(float(row["mean"])),
                std=fmt_float(float(row["std"])),
                min_=fmt_float(float(row["min"])),
                p25=fmt_float(float(row["p25"])),
                median=fmt_float(float(row["median"])),
                p75=fmt_float(float(row["p75"])),
                max_=fmt_float(float(row["max"])),
            )
        )
    lines.append("")
    lines.append("## 5. 种子排名")
    lines.append("")
    lines.append("| rank(selected) | seed | selected test acc | final test acc | selected-final acc | selected epoch |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rankings:
        lines.append(
            "| {rank} | {seed} | {selected_acc} | {final_acc} | {gap} | {epoch} |".format(
                rank=row["selected_test_acc_rank"],
                seed=row["init_seed"],
                selected_acc=fmt_float(float(row["selected_test_acc"])),
                final_acc=fmt_float(float(row["final_test_acc"])),
                gap=fmt_float(float(row["selected_minus_final_acc"])),
                epoch=int(row["selected_epoch"]),
            )
        )
    lines.append("")
    lines.append("## 6. 结论与下一步")
    lines.append("")
    lines.append("- 这轮结果首先验证了 `init-only` 研究协议在当前仓库内已经可运行、可复现、可批量汇总。")
    lines.append("- 如果 MNIST 上波动很小，不应把它当作负结果；这更像是后续迁移到 CIFAR-10 前的 deterministic sanity check。")
    lines.append("- 下一步应优先做两件事：")
    lines.append("  1. 扩展到 `Fashion-MNIST` 或 `CIFAR-10`，看初始化波动是否显著放大。")
    lines.append("  2. 在同一协议上接入 `ticket / pruning / shadow-prune`，把当前 dense 基线升级为真正的 LTH-style paired study。")
    lines.append("")
    lines.append("## 7. 生成产物")
    lines.append("")
    lines.append(f"- 聚合统计：`{out_dir / 'core_statistics.csv'}`")
    lines.append(f"- 每 epoch 汇总：`{out_dir / 'epoch_curve_summary.csv'}`")
    lines.append(f"- 种子排名：`{out_dir / 'seed_rankings.csv'}`")
    lines.append(f"- 曲线图：`{out_dir / 'accuracy_and_loss_curves.png'}`")
    lines.append(f"- 散点图：`{out_dir / 'selected_vs_final_accuracy.png'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_root = args.run_root or latest_run_root(args.results_root, args.dataset, args.model)
    out_dir = args.out_dir or (run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(run_root / "run_config.json")
    summary_rows = load_csv_rows(run_root / "aggregate_summary.csv")
    metric_rows_by_seed: Dict[int, List[Dict[str, object]]] = {}
    for seed_dir in sorted(path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("init_seed_")):
        seed = int(seed_dir.name.split("_")[-1])
        metric_rows_by_seed[seed] = load_csv_rows(seed_dir / "metrics.csv")

    epoch_rows = aggregate_epoch_histories(metric_rows_by_seed)
    core_stats = build_core_statistics(summary_rows)
    rankings = build_seed_rankings(summary_rows)

    write_csv(out_dir / "epoch_curve_summary.csv", epoch_rows)
    write_csv(out_dir / "core_statistics.csv", core_stats)
    write_csv(out_dir / "seed_rankings.csv", rankings)
    plot_curves(epoch_rows, out_dir / "accuracy_and_loss_curves.png")
    plot_seed_scatter(summary_rows, out_dir / "selected_vs_final_accuracy.png")

    report_text = render_report(
        run_root=run_root,
        out_dir=out_dir,
        config=config,
        summary_rows=summary_rows,
        core_stats=core_stats,
        epoch_rows=epoch_rows,
        rankings=rankings,
    )
    report_path = out_dir / "init_only_phase0_report_zh.md"
    report_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
