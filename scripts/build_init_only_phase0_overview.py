import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from neuron_survival_dynamics.init_only.models import MODEL_DEFAULTS


DEFAULT_RESULTS_ROOT = Path("results/init_only_lth_20260401")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cross-dataset Phase-0 overview for init-only dense pilots."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist"])
    parser.add_argument(
        "--dataset-model-specs",
        nargs="+",
        default=None,
        help="Optional explicit specs formatted as dataset:model, e.g. mnist:lenet300100 cifar10:smallconv",
    )
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
        return list(reader)


def to_float(row: Dict[str, object], key: str) -> float:
    return float(row[key])


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
    rows = load_csv_rows(run_root / "aggregate_summary.csv")

    selected_acc = np.asarray([to_float(row, "selected_test_acc") for row in rows], dtype=float)
    final_acc = np.asarray([to_float(row, "final_test_acc") for row in rows], dtype=float)
    selected_minus_final_acc = np.asarray([to_float(row, "selected_minus_final_acc") for row in rows], dtype=float)
    final_minus_selected_loss = np.asarray([to_float(row, "final_minus_selected_loss") for row in rows], dtype=float)

    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "run_root": str(run_root),
        "n_seeds": len(rows),
        "epochs": config["epochs"],
        "train_size": config["train_size"],
        "val_size": config["val_size"],
        "test_size": config["test_size"],
        "selected_test_acc_mean": float(selected_acc.mean()),
        "selected_test_acc_std": float(selected_acc.std(ddof=0)),
        "selected_test_acc_min": float(selected_acc.min()),
        "selected_test_acc_max": float(selected_acc.max()),
        "final_test_acc_mean": float(final_acc.mean()),
        "final_test_acc_std": float(final_acc.std(ddof=0)),
        "final_test_acc_min": float(final_acc.min()),
        "final_test_acc_max": float(final_acc.max()),
        "selected_minus_final_acc_mean": float(selected_minus_final_acc.mean()),
        "final_minus_selected_loss_mean": float(final_minus_selected_loss.mean()),
    }


def build_markdown(rows: Sequence[Dict[str, object]], out_dir: Path) -> str:
    by_variability = sorted(rows, key=lambda row: float(row["selected_test_acc_std"]), reverse=True)
    by_final_variability = sorted(rows, key=lambda row: float(row["final_test_acc_std"]), reverse=True)
    by_degradation = sorted(rows, key=lambda row: float(row["final_minus_selected_loss_mean"]), reverse=True)
    lines: List[str] = []
    lines.append("# Init-Only Phase-0 Cross-Dataset Overview")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("这个总览把同一套 `fixed-data / fixed-order / init-only` 协议在不同数据集上的首轮 dense pilot 放在一起，先看初始化单独带来的性能波动是否随任务难度上升。")
    lines.append("")
    lines.append("## 2. 数据集摘要")
    lines.append("")
    lines.append("| dataset | model | seeds | epochs | selected acc mean±std | final acc mean±std | selected acc range |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        selected_text = f"{fmt(float(row['selected_test_acc_mean']))} ± {fmt(float(row['selected_test_acc_std']))}"
        final_text = f"{fmt(float(row['final_test_acc_mean']))} ± {fmt(float(row['final_test_acc_std']))}"
        spread_text = f"{fmt(float(row['selected_test_acc_min']))} – {fmt(float(row['selected_test_acc_max']))}"
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['n_seeds']} | {row['epochs']} | {selected_text} | {final_text} | {spread_text} |"
        )
    lines.append("")
    lines.append("## 3. 初步判断")
    lines.append("")
    most_variable = by_variability[0]
    most_final_variable = by_final_variability[0]
    most_degraded = by_degradation[0]
    lines.append(
        f"- 当前波动最大的已完成数据集是 `{most_variable['dataset']}`，selected test acc 标准差为 `{fmt(float(most_variable['selected_test_acc_std']))}`。"
    )
    lines.append(
        f"- 如果更关注训练终点，`{most_final_variable['dataset']}` 的 final test acc 标准差最大，为 `{fmt(float(most_final_variable['final_test_acc_std']))}`。"
    )
    lines.append(
        f"- 如果更关注 late instability，`{most_degraded['dataset']}` 的 `final-selected loss mean` 最大，为 `{fmt(float(most_degraded['final_minus_selected_loss_mean']))}`。"
    )
    for row in rows:
        if float(row["selected_minus_final_acc_mean"]) >= 0.0:
            acc_note = "终点测试精度平均低于 selected 点"
        else:
            acc_note = "终点测试精度平均高于 selected 点"
        if float(row["final_minus_selected_loss_mean"]) >= 0.0:
            loss_note = "终点测试损失平均更差"
        else:
            loss_note = "终点测试损失平均更优"
        lines.append(
            f"- `{row['dataset']}`：`selected-final acc mean={fmt(float(row['selected_minus_final_acc_mean']))}`，{acc_note}；`final-selected loss mean={fmt(float(row['final_minus_selected_loss_mean']))}`，{loss_note}。"
        )
    lines.append("")
    lines.append("## 4. 下一步")
    lines.append("")
    lines.append("- Phase 0 的目标不是立刻找到 phase，而是先验证当只有初始化变化时，性能波动是否足够大到值得做结构分析。")
    lines.append("- 如果 `Fashion-MNIST` 仍然很稳，下一步应直接进入 `CIFAR-10`；如果它已明显放大波动，则说明 Phase 1 的图像任务已经有可研究信号。")
    lines.append("- 一旦 dense 波动被确认，下一层就应接入 `ticket / pruning / shadow-prune`，把当前 baseline 升级为 paired 机制研究。")
    lines.append("")
    lines.append("## 5. 相关文件")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['dataset']}` run：`{row['run_root']}`")
    lines.append(f"- 汇总表：`{out_dir / 'dataset_phase0_summary.csv'}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.results_root / "phase0_overview")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    dataset_model_pairs: List[tuple[str, str]] = []
    if args.dataset_model_specs:
        for spec in args.dataset_model_specs:
            if ":" not in spec:
                raise ValueError(f"Invalid dataset-model spec: {spec!r}")
            dataset, model = spec.split(":", 1)
            dataset_model_pairs.append((dataset, model))
    else:
        for dataset in args.datasets:
            dataset_model_pairs.append((dataset, MODEL_DEFAULTS[dataset]))

    for dataset, model in dataset_model_pairs:
        run_root = latest_run_root(args.results_root, dataset, model)
        summary_rows.append(summarize_run(run_root))

    write_csv(out_dir / "dataset_phase0_summary.csv", summary_rows)
    report = build_markdown(summary_rows, out_dir)
    (out_dir / "phase0_overview_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
