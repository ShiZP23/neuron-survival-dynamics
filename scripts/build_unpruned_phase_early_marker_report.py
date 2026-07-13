import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ANALYSIS_ROOT = Path(
    "results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis"
)
ALIGNMENT_CSV = ANALYSIS_ROOT / "phase_seed_alignment.csv"
OUT_DIR = ANALYSIS_ROOT / "early_markers"
DEFAULT_CHECKPOINTS = [50, 100, 200, 500, 1000]
MAIN3_PHASES = ["prune0+2", "prune0_only", "no_prune"]
BINARY_PHASES = ["prune0+2", "prune0_only"]
RISK_TARGETS = {
    "degradation_gt_2e-4": ("fixed_final_minus_selected", 2e-4),
    "final_loss_gt_4e-4": ("fixed_final_test_loss", 4e-4),
    "final_gap_gt_1.5e-4": ("fixed_final_gap", 1.5e-4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an early-marker report for the unpruned phase differentiation study."
    )
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Epoch checkpoints to extract fixed-run trajectory features from.",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--cv-seed", type=int, default=17)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: Path) -> Dict[int, Dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["epoch"]): row for row in rows}


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted = [1.0] * n
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        value = p_values[idx] * n / rank
        running = min(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}g}"
    return str(value)


def extract_checkpoint_features(
    alignment: pd.DataFrame, checkpoints: Sequence[int]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in alignment.iterrows():
        metrics_by_epoch = read_metrics(Path(row["fixed_run"]) / "metrics.csv")
        min_epoch = min(metrics_by_epoch.keys())
        previous = metrics_by_epoch[min_epoch]
        record: Dict[str, object] = {
            "seed": int(row["seed"]),
            "phase": row["prune_only_phase"],
            "fixed_final_test_loss": float(row["fixed_final_test_loss"]),
            "fixed_final_minus_selected": float(row["fixed_final_minus_selected"]),
            "fixed_final_gap": float(row["fixed_final_gap"]),
        }
        for checkpoint in checkpoints:
            current = metrics_by_epoch[checkpoint]
            active = np.array([int(current[f"active_{layer_idx}"]) for layer_idx in range(3)], dtype=float)
            active_prev = np.array([int(previous[f"active_{layer_idx}"]) for layer_idx in range(3)], dtype=float)
            share = active / max(float(active.sum()), 1.0)
            share_prev = active_prev / max(float(active_prev.sum()), 1.0)

            record[f"e{checkpoint}_train_loss"] = float(current["train_loss"])
            record[f"e{checkpoint}_val_loss"] = float(current["val_loss"])
            record[f"e{checkpoint}_test_loss"] = float(current["test_loss"])
            record[f"e{checkpoint}_generalization_gap"] = float(current["test_loss"]) - float(
                current["train_loss"]
            )
            for layer_idx in range(3):
                record[f"e{checkpoint}_active_{layer_idx}"] = int(active[layer_idx])
                record[f"e{checkpoint}_share_{layer_idx}"] = float(share[layer_idx])
                record[f"e{checkpoint}_delta_active_{layer_idx}"] = int(active[layer_idx] - active_prev[layer_idx])
                record[f"e{checkpoint}_delta_share_{layer_idx}"] = float(share[layer_idx] - share_prev[layer_idx])
            previous = current
        rows.append(record)
    return pd.DataFrame(rows)


def evaluate_binary_cv(
    feature_df: pd.DataFrame,
    checkpoints: Sequence[int],
    cv_splits: int,
    cv_repeats: int,
    cv_seed: int,
) -> List[Dict[str, object]]:
    subset = feature_df[feature_df["phase"].isin(BINARY_PHASES)].copy()
    y = (subset["phase"] == "prune0+2").astype(int).to_numpy()
    rows: List[Dict[str, object]] = []
    splitter = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=cv_seed)
    for checkpoint in checkpoints:
        feature_cols = [column for column in subset.columns if column.startswith(f"e{checkpoint}_")]
        X = subset[feature_cols].to_numpy(dtype=float)
        fold_balanced_accuracy: List[float] = []
        fold_auc: List[float] = []
        for train_idx, test_idx in splitter.split(X, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            )
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_score = model.predict_proba(X[test_idx])[:, 1]
            fold_balanced_accuracy.append(balanced_accuracy_score(y[test_idx], y_pred))
            fold_auc.append(roc_auc_score(y[test_idx], y_score))
        rows.append(
            {
                "task": "binary_prune0+2_vs_prune0_only",
                "checkpoint": checkpoint,
                "n_samples": int(len(subset)),
                "n_features": int(len(feature_cols)),
                "mean_balanced_accuracy": float(np.mean(fold_balanced_accuracy)),
                "std_balanced_accuracy": float(np.std(fold_balanced_accuracy)),
                "mean_roc_auc": float(np.mean(fold_auc)),
                "std_roc_auc": float(np.std(fold_auc)),
            }
        )
    return rows


def evaluate_main3_cv(
    feature_df: pd.DataFrame,
    checkpoints: Sequence[int],
    cv_splits: int,
    cv_repeats: int,
    cv_seed: int,
) -> List[Dict[str, object]]:
    subset = feature_df[feature_df["phase"].isin(MAIN3_PHASES)].copy()
    y = subset["phase"].to_numpy()
    rows: List[Dict[str, object]] = []
    splitter = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=cv_seed)
    for checkpoint in checkpoints:
        feature_cols = [column for column in subset.columns if column.startswith(f"e{checkpoint}_")]
        X = subset[feature_cols].to_numpy(dtype=float)
        fold_balanced_accuracy: List[float] = []
        fold_macro_f1: List[float] = []
        for train_idx, test_idx in splitter.split(X, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, class_weight="balanced"),
            )
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            fold_balanced_accuracy.append(balanced_accuracy_score(y[test_idx], y_pred))
            fold_macro_f1.append(f1_score(y[test_idx], y_pred, average="macro"))
        rows.append(
            {
                "task": "main3_prune0+2_vs_prune0_only_vs_no_prune",
                "checkpoint": checkpoint,
                "n_samples": int(len(subset)),
                "n_features": int(len(feature_cols)),
                "mean_balanced_accuracy": float(np.mean(fold_balanced_accuracy)),
                "std_balanced_accuracy": float(np.std(fold_balanced_accuracy)),
                "mean_macro_f1": float(np.mean(fold_macro_f1)),
                "std_macro_f1": float(np.std(fold_macro_f1)),
            }
        )
    return rows


def evaluate_risk_cv(
    feature_df: pd.DataFrame,
    checkpoints: Sequence[int],
    cv_splits: int,
    cv_repeats: int,
    cv_seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    splitter = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=cv_seed)
    for risk_name, (column, threshold) in RISK_TARGETS.items():
        y = (feature_df[column] > threshold).astype(int).to_numpy()
        for checkpoint in checkpoints:
            feature_cols = [value for value in feature_df.columns if value.startswith(f"e{checkpoint}_")]
            X = feature_df[feature_cols].to_numpy(dtype=float)
            fold_balanced_accuracy: List[float] = []
            fold_auc: List[float] = []
            for train_idx, test_idx in splitter.split(X, y):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=5000, class_weight="balanced"),
                )
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])
                y_score = model.predict_proba(X[test_idx])[:, 1]
                fold_balanced_accuracy.append(balanced_accuracy_score(y[test_idx], y_pred))
                fold_auc.append(roc_auc_score(y[test_idx], y_score))
            rows.append(
                {
                    "risk_target": risk_name,
                    "source_column": column,
                    "threshold": threshold,
                    "checkpoint": checkpoint,
                    "risk_rate": float(y.mean()),
                    "mean_balanced_accuracy": float(np.mean(fold_balanced_accuracy)),
                    "std_balanced_accuracy": float(np.std(fold_balanced_accuracy)),
                    "mean_roc_auc": float(np.mean(fold_auc)),
                    "std_roc_auc": float(np.std(fold_auc)),
                }
            )
    return rows


def build_univariate_binary_rows(feature_df: pd.DataFrame, checkpoints: Sequence[int]) -> List[Dict[str, object]]:
    subset = feature_df[feature_df["phase"].isin(BINARY_PHASES)].copy()
    rows: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        checkpoint_rows: List[Dict[str, object]] = []
        for feature_name in [column for column in subset.columns if column.startswith(f"e{checkpoint}_")]:
            values_a = subset.loc[subset["phase"] == "prune0+2", feature_name].to_numpy(dtype=float)
            values_b = subset.loc[subset["phase"] == "prune0_only", feature_name].to_numpy(dtype=float)
            stat, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
            auc = stat / (len(values_a) * len(values_b))
            checkpoint_rows.append(
                {
                    "checkpoint": checkpoint,
                    "feature_name": feature_name,
                    "median_prune0+2": float(np.median(values_a)),
                    "median_prune0_only": float(np.median(values_b)),
                    "median_diff_prune0+2_minus_prune0_only": float(np.median(values_a) - np.median(values_b)),
                    "mannwhitney_u": float(stat),
                    "auc_prune0+2_greater": float(auc),
                    "p_value": float(p_value),
                }
            )
        adjusted = benjamini_hochberg([row["p_value"] for row in checkpoint_rows]) if checkpoint_rows else []
        for row, p_adjusted in zip(checkpoint_rows, adjusted):
            row["p_value_bh"] = p_adjusted
            rows.append(row)
    return rows


def build_logistic_coefficient_rows(feature_df: pd.DataFrame, checkpoints: Sequence[int]) -> List[Dict[str, object]]:
    subset = feature_df[feature_df["phase"].isin(BINARY_PHASES)].copy()
    y = (subset["phase"] == "prune0+2").astype(int).to_numpy()
    rows: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        feature_cols = [column for column in subset.columns if column.startswith(f"e{checkpoint}_")]
        X = subset[feature_cols].to_numpy(dtype=float)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        )
        model.fit(X, y)
        coefficients = model.named_steps["logisticregression"].coef_[0]
        for feature_name, coefficient in zip(feature_cols, coefficients):
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "feature_name": feature_name,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                }
            )
    return rows


def build_report(
    out_path: Path,
    binary_cv: pd.DataFrame,
    main3_cv: pd.DataFrame,
    risk_cv: pd.DataFrame,
    univariate_rows: pd.DataFrame,
    coefficient_rows: pd.DataFrame,
) -> None:
    binary_lookup = {int(row["checkpoint"]): row for _, row in binary_cv.iterrows()}
    main3_lookup = {int(row["checkpoint"]): row for _, row in main3_cv.iterrows()}
    risk_lookup = {(row["risk_target"], int(row["checkpoint"])): row for _, row in risk_cv.iterrows()}

    def top_features(checkpoint: int) -> List[str]:
        group = univariate_rows[univariate_rows["checkpoint"] == checkpoint].sort_values(
            ["p_value_bh", "p_value", "feature_name"]
        )
        return [
            f"{row['feature_name']} (BH={fmt(row['p_value_bh'])}, diff={fmt(row['median_diff_prune0+2_minus_prune0_only'])})"
            for _, row in group.head(5).iterrows()
        ]

    def top_coefs(checkpoint: int) -> List[str]:
        group = coefficient_rows[coefficient_rows["checkpoint"] == checkpoint].sort_values(
            "abs_coefficient", ascending=False
        )
        return [
            f"{row['feature_name']} (coef={fmt(row['coefficient'])})"
            for _, row in group.head(5).iterrows()
        ]

    lines = [
        "# 未剪枝网络分化的早期标记报告",
        "",
        f"生成日期：{date.today().isoformat()}",
        "",
        "## 研究问题",
        "",
        "- 如果 `fixed` 的 latent phase differentiation 真的存在，那么 paired phase label 是否能在 `fixed` 的早期轨迹里被预测出来？",
        "- 进一步地，早期粗指标是否也能直接预测后续的 severe late degradation？",
        "",
        "## 主要结论",
        "",
        f"- 只用 `fixed` 的 epoch 50 特征，就能在 `prune0+2` vs `prune0_only` 二分类上达到 balanced accuracy {fmt(binary_lookup[50]['mean_balanced_accuracy'])}，ROC-AUC {fmt(binary_lookup[50]['mean_roc_auc'])}。",
        f"- 这个结果在 epoch 100 与 epoch 200 仍然稳定，balanced accuracy 分别为 {fmt(binary_lookup[100]['mean_balanced_accuracy'])} 与 {fmt(binary_lookup[200]['mean_balanced_accuracy'])}。",
        f"- 但三分类 `prune0+2 / prune0_only / no_prune` 的 balanced accuracy 只有 {fmt(main3_lookup[50]['mean_balanced_accuracy'])} 到 {fmt(main3_lookup[100]['mean_balanced_accuracy'])}，说明 `no_prune` 小样本与弱分离仍然限制了早期可预测性。",
        f"- 对 late degradation 风险本身，epoch 50/100/200 的粗特征几乎没有稳定预测力；例如 `degradation_gt_2e-4` 在 epoch 50 的 ROC-AUC 只有 {fmt(risk_lookup[('degradation_gt_2e-4', 50)]['mean_roc_auc'])}。",
        "",
        "## 如何理解这个结果",
        "",
        "- 这说明 phase identity 比 catastrophic late failure 更早可见。换句话说，dense trajectory 很早就已经带着“会走向哪一类结构命运”的信息，但后期是否真的退化到严重程度，还需要更细的中介机制才能解释。",
        "- 这和我们前一轮 deep dive 的发现是一致的：phase 差异是早期定轨、后期显化，但当前 aggregate drift proxy 还不足以解释最终退化。",
        "",
        "## 最强的早期标记",
        "",
        f"- epoch 50 最显著的单变量标记包括：{'; '.join(top_features(50))}。",
        f"- epoch 100 的主标记包括：{'; '.join(top_features(100))}。",
        f"- epoch 200 的主标记包括：{'; '.join(top_features(200))}。",
        "",
        "这些结果有一个稳定方向：",
        "",
        "- `prune0+2` 对应的 `fixed` 在早期就表现出更高的 layer-2 active count / share。",
        "- 与之配套，layer-0 share 更低，且 epoch 50 的 generalization gap 与 test loss 也偏高。",
        "- 这提示 paired `prune0+2` 相并不是在真正剪枝发生后才出现，而是 dense network 在 very early stage 就已经偏向更强的上层参与和更不同的表征分工。",
        "",
        "## 全模型可解释方向",
        "",
        f"- epoch 50 的逻辑回归绝对系数前几项是：{'; '.join(top_coefs(50))}。",
        f"- epoch 100 的前几项是：{'; '.join(top_coefs(100))}。",
        f"- epoch 200 的前几项是：{'; '.join(top_coefs(200))}。",
        "",
        "这些系数只能当描述性方向，不应被解读成因果效应，但它们与单变量结果高度一致：layer-2 active/share 和早期 gap 是最稳定的候选前兆。",
        "",
        "## 为什么 tail-risk 预测几乎失败",
        "",
        "- 这并不意味着早期不重要，而更像意味着“我们现在记录到的早期 summary 太粗”。",
        "- 目前只记录了 loss、active count、share 这种聚合量；它们足以捕捉粗 phase，但不足以捕捉哪些具体 neurons 会在后面成为 instability source。",
        "- 因此，下一步最合理的不是继续加更多 logistic regression，而是采集更细粒度状态，例如 would-be-pruned neuron identity、importance EMA、candidate set 稳定性、以及早期 representation drift。",
        "",
        "## 与文献的关系",
        "",
        "- Frankle, Schwab, Morcos 2020《The Early Phase of Neural Network Training》强调训练最初阶段会快速决定后续轨迹。我们的结果把这一点具体化为：在完全不剪枝的 `fixed` 中，paired phase 信息在 epoch 50 左右就已有可预测信号。链接：https://arxiv.org/abs/2002.10365",
        "- Achille, Rovere, Soatto 2017《Critical Learning Periods in Deep Neural Networks》提出 early critical period 与后续可塑性下降。我们的 early-marker 结果与这种“早期定轨、后续更难改写”的叙述是相容的。链接：https://arxiv.org/abs/1711.08856",
        "- Tian et al. 2019《Luck Matters》指出少数“lucky nodes”会更快对齐并主导训练。这里 layer-2 active count 的早期分化，和“seed 决定哪些单元早期占优”这类机制直觉是吻合的。链接：https://arxiv.org/abs/1905.13405",
        "- Kwok et al. 2025《The Butterfly Effect》进一步强调训练轨迹对初始条件高度敏感，而且这种敏感性集中在 very early phase。我们的结果与其方向一致：同样的 dense training，在 very early checkpoint 就已经带着后续结构 phase 的信息。链接：https://openreview.net/forum?id=L1Bm396P0X",
        "",
        "## 研究判断",
        "",
        "作为下一步研究判断，我认为：",
        "",
        "1. 现在已经可以把“latent phase 在早期可见”写成一个更强的经验命题。",
        "2. 但当前 coarse early markers 还不足以解释“为什么有些 seed 最后真的会严重退化”。",
        "3. 因此最值得做的下一步仍然是 `shadow prune`，并把 early-marker 分析和 neuron-level fate tracing 结合起来。",
        "",
        "## 文件导航",
        "",
        "- 本报告：`early_marker_report_zh.md`",
        "- per-seed 早期特征：`early_marker_feature_rows.csv`",
        "- 二分类 / 三分类可预测性：`early_phase_cv_binary.csv`、`early_phase_cv_main3.csv`",
        "- late-risk 可预测性：`early_risk_cv.csv`",
        "- 早期单变量差异：`early_binary_univariate.csv`",
        "- 描述性 logistic 系数：`early_binary_logistic_coefficients.csv`",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    alignment = pd.read_csv(args.alignment_csv)
    feature_df = extract_checkpoint_features(alignment, args.checkpoints)
    binary_cv = pd.DataFrame(
        evaluate_binary_cv(feature_df, args.checkpoints, args.cv_splits, args.cv_repeats, args.cv_seed)
    )
    main3_cv = pd.DataFrame(
        evaluate_main3_cv(feature_df, args.checkpoints, args.cv_splits, args.cv_repeats, args.cv_seed)
    )
    risk_cv = pd.DataFrame(
        evaluate_risk_cv(feature_df, args.checkpoints, args.cv_splits, args.cv_repeats, args.cv_seed)
    )
    univariate_rows = pd.DataFrame(build_univariate_binary_rows(feature_df, [50, 100, 200]))
    coefficient_rows = pd.DataFrame(build_logistic_coefficient_rows(feature_df, [50, 100, 200]))

    write_csv(args.out_dir / "early_marker_feature_rows.csv", feature_df.to_dict("records"))
    write_csv(args.out_dir / "early_phase_cv_binary.csv", binary_cv.to_dict("records"))
    write_csv(args.out_dir / "early_phase_cv_main3.csv", main3_cv.to_dict("records"))
    write_csv(args.out_dir / "early_risk_cv.csv", risk_cv.to_dict("records"))
    write_csv(args.out_dir / "early_binary_univariate.csv", univariate_rows.to_dict("records"))
    write_csv(
        args.out_dir / "early_binary_logistic_coefficients.csv",
        coefficient_rows.to_dict("records"),
    )
    build_report(
        args.out_dir / "early_marker_report_zh.md",
        binary_cv,
        main3_cv,
        risk_cv,
        univariate_rows,
        coefficient_rows,
    )


if __name__ == "__main__":
    main()
