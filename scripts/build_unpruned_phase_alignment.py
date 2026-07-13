import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


PHASE_ORDER = {
    "prune0+2": 0,
    "prune0_only": 1,
    "prune2_only": 2,
    "no_prune": 3,
    "other": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align hard-task fixed runs to the prune_only phase labels of the same seed."
    )
    parser.add_argument(
        "--prune-root",
        type=Path,
        default=Path("results/publishable_pilot_20260313/hard/prune_only"),
        help="Root directory for hard-task prune_only runs.",
    )
    parser.add_argument(
        "--fixed-root",
        type=Path,
        default=Path("results/publishable_pilot_20260313/hard/fixed"),
        help="Root directory for hard-task fixed runs.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path(
            "research/analysis/studies/unpruned_phase_differentiation_20260316/phase_seed_alignment.csv"
        ),
        help="CSV path for the seed-to-phase alignment table.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def classify_phase(rows: List[Dict[str, str]]) -> Tuple[str, List[int]]:
    updates = [row for row in rows if int(row["is_update_epoch"]) == 1]
    totals = [sum(int(row[f"pruned_{idx}"]) for row in updates) for idx in range(3)]
    if totals[0] > 0 and totals[2] > 0:
        return "prune0+2", totals
    if totals[0] > 0 and totals[2] == 0:
        return "prune0_only", totals
    if totals[0] == 0 and totals[2] > 0:
        return "prune2_only", totals
    if sum(totals) == 0:
        return "no_prune", totals
    return "other", totals


def summarize_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    best_row = min(rows, key=lambda row: float(row["val_loss"]))
    final_row = rows[-1]
    return {
        "selected_test_loss": float(best_row["test_loss"]),
        "final_test_loss": float(final_row["test_loss"]),
        "final_minus_selected": float(final_row["test_loss"]) - float(best_row["test_loss"]),
        "best_epoch": int(best_row["epoch"]),
        "final_gap": float(final_row["test_loss"]) - float(final_row["train_loss"]),
        "active_0_best": int(best_row["active_0"]),
        "active_1_best": int(best_row["active_1"]),
        "active_2_best": int(best_row["active_2"]),
        "active_0_final": int(final_row["active_0"]),
        "active_1_final": int(final_row["active_1"]),
        "active_2_final": int(final_row["active_2"]),
    }


def latest_metrics_by_seed(root: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for seed_dir in sorted(root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates = sorted(seed_dir.glob("*/metrics.csv"))
        if not candidates:
            continue
        out[seed] = candidates[-1]
    return out


def build_rows(prune_root: Path, fixed_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    prune_latest = latest_metrics_by_seed(prune_root)
    fixed_latest = latest_metrics_by_seed(fixed_root)
    for seed in sorted(prune_latest.keys()):
        prune_metrics = prune_latest[seed]
        if seed not in fixed_latest:
            continue
        fixed_metrics = fixed_latest[seed]

        prune_rows = load_metrics(prune_metrics)
        fixed_rows = load_metrics(fixed_metrics)
        phase, totals = classify_phase(prune_rows)
        prune_summary = summarize_metrics(prune_rows)
        fixed_summary = summarize_metrics(fixed_rows)

        row: Dict[str, object] = {
            "phase_order": PHASE_ORDER[phase],
            "prune_only_phase": phase,
            "seed": seed,
            "prune_only_run": prune_metrics.parent.as_posix(),
            "fixed_run": fixed_metrics.parent.as_posix(),
            "prune_only_total_pruned": sum(totals),
            "prune_only_pruned_0_total": totals[0],
            "prune_only_pruned_1_total": totals[1],
            "prune_only_pruned_2_total": totals[2],
        }
        for key, value in prune_summary.items():
            row[f"prune_only_{key}"] = value
        for key, value in fixed_summary.items():
            row[f"fixed_{key}"] = value
        row["fixed_active_0_drift"] = row["fixed_active_0_final"] - row["fixed_active_0_best"]
        row["fixed_active_1_drift"] = row["fixed_active_1_final"] - row["fixed_active_1_best"]
        row["fixed_active_2_drift"] = row["fixed_active_2_final"] - row["fixed_active_2_best"]
        rows.append(row)

    rows.sort(key=lambda row: (int(row["phase_order"]), int(row["seed"])))
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows collected for phase alignment")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.prune_root, args.fixed_root)
    write_csv(args.out_path, rows)


if __name__ == "__main__":
    main()
