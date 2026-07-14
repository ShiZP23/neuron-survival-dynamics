import argparse
import csv
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = Path("results/followup_20260317/unpruned_phase_seed_sweep")
DEFAULT_TASK = "hard"
DEFAULT_MODES = ["prune_only", "fixed"]
PHASE_ORDER = ["prune0+2", "prune0_only", "prune2_only", "no_prune", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and analyze a large seed sweep for the unpruned-phase study."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["fixed", "prune_only"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=49, help="Inclusive end of the seed range.")
    parser.add_argument(
        "--seed-list",
        type=str,
        default=None,
        help="Comma-separated explicit seed list. Overrides --seed-start/--seed-end.",
    )
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--rerun-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--update-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-neurons", type=int, default=16)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--ema-z-threshold", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-layer", type=int, default=8)
    parser.add_argument("--ablation-epsilon-ratio", type=float, default=0.01)
    parser.add_argument("--active-threshold", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--shadow-prune",
        action="store_true",
        help="Pass --shadow-prune to fixed runs so they log would-be-pruned neurons.",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def parse_seed_values(args: argparse.Namespace) -> List[int]:
    if args.seed_list:
        return [int(token.strip()) for token in args.seed_list.split(",") if token.strip()]
    return list(range(args.seed_start, args.seed_end + 1))


def load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def classify_phase(metrics_rows: List[Dict[str, str]]) -> Dict[str, object]:
    updates = [row for row in metrics_rows if int(row["is_update_epoch"]) == 1]
    pruned = [sum(int(row[f"pruned_{idx}"]) for row in updates) for idx in range(3)]
    if pruned[0] > 0 and pruned[2] > 0:
        phase = "prune0+2"
    elif pruned[0] > 0 and pruned[2] == 0:
        phase = "prune0_only"
    elif pruned[0] == 0 and pruned[2] > 0:
        phase = "prune2_only"
    elif sum(pruned) == 0:
        phase = "no_prune"
    else:
        phase = "other"
    losses = [float(row["test_loss"]) for row in metrics_rows]
    best_row = min(metrics_rows, key=lambda row: float(row["val_loss"]))
    final_row = metrics_rows[-1]
    return {
        "phase": phase,
        "total_pruned": sum(pruned),
        "pruned_0_total": pruned[0],
        "pruned_1_total": pruned[1],
        "pruned_2_total": pruned[2],
        "selected_test_loss": float(best_row["test_loss"]),
        "final_test_loss": float(final_row["test_loss"]),
        "final_minus_selected": float(final_row["test_loss"]) - float(best_row["test_loss"]),
        "best_epoch": int(best_row["epoch"]),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def existing_run_metrics(results_root: Path, task: str, mode: str, seed: int) -> List[Path]:
    return sorted((results_root / task / mode / f"seed_{seed}").glob("*/metrics.csv"))


def build_run_command(args: argparse.Namespace, mode: str, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        "run.py",
        "--task",
        args.task,
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--results-dir",
        str(args.results_root),
        "--epochs",
        str(args.epochs),
        "--update-interval",
        str(args.update_interval),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--min-neurons",
        str(args.min_neurons),
        "--ema-beta",
        str(args.ema_beta),
        "--ema-z-threshold",
        str(args.ema_z_threshold),
        "--max-candidates-per-layer",
        str(args.max_candidates_per_layer),
        "--ablation-epsilon-ratio",
        str(args.ablation_epsilon_ratio),
        "--active-threshold",
        str(args.active_threshold),
        "--n-train",
        str(args.n_train),
        "--n-val",
        str(args.n_val),
        "--n-test",
        str(args.n_test),
        "--noise",
        str(args.noise),
        "--device",
        args.device,
    ]
    if args.shadow_prune and mode == "fixed":
        cmd.append("--shadow-prune")
    return cmd


def run_sweep(args: argparse.Namespace, seeds: Iterable[int]) -> None:
    for seed in seeds:
        for mode in args.modes:
            existing = existing_run_metrics(args.results_root, args.task, mode, seed)
            if existing and args.skip_existing:
                print(f"[skip] seed={seed} mode={mode} existing_runs={len(existing)}")
                continue
            cmd = build_run_command(args, mode, seed)
            print(f"[run] {shlex.join(cmd)}")
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_prune_inventory(prune_root: Path, out_dir: Path) -> Optional[Path]:
    latest = latest_metrics_by_seed(prune_root)
    if not latest:
        return None
    rows: List[Dict] = []
    counts = Counter()
    seeds_by_phase: Dict[str, List[int]] = {phase: [] for phase in PHASE_ORDER}
    for seed, metrics_path in sorted(latest.items()):
        summary = classify_phase(load_metrics(metrics_path))
        phase = str(summary["phase"])
        counts[phase] += 1
        seeds_by_phase.setdefault(phase, []).append(seed)
        rows.append(
            {
                "seed": seed,
                "phase": phase,
                "run_path": str(metrics_path.parent),
                "total_pruned": summary["total_pruned"],
                "pruned_0_total": summary["pruned_0_total"],
                "pruned_1_total": summary["pruned_1_total"],
                "pruned_2_total": summary["pruned_2_total"],
                "selected_test_loss": summary["selected_test_loss"],
                "final_test_loss": summary["final_test_loss"],
                "final_minus_selected": summary["final_minus_selected"],
                "best_epoch": summary["best_epoch"],
            }
        )
    write_csv(out_dir / "prune_only_phase_inventory.csv", rows)
    count_rows = [
        {
            "phase": phase,
            "count": counts.get(phase, 0),
            "seed_list": ",".join(str(seed) for seed in seeds_by_phase.get(phase, [])),
        }
        for phase in PHASE_ORDER
    ]
    write_csv(out_dir / "prune_only_phase_counts.csv", count_rows)
    lines = [
        "# Prune-Only Phase Sweep Summary",
        "",
        f"Source root: `{prune_root}`",
        "",
        "Phase counts:",
    ]
    for row in count_rows:
        lines.append(f"- {row['phase']}: {row['count']} seeds ({row['seed_list'] or 'none'})")
    if rows:
        lines.extend(
            [
                "",
                f"Median final test loss across discovered seeds: {median([row['final_test_loss'] for row in rows]):.6g}",
            ]
        )
    summary_path = out_dir / "prune_only_phase_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def run_builder(cmd: List[str]) -> None:
    print(f"[analyze] {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_analysis(results_root: Path, task: str) -> Path:
    analysis_dir = results_root / task / "unpruned_phase_seed_sweep_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    prune_root = results_root / task / "prune_only"
    fixed_root = results_root / task / "fixed"

    build_prune_inventory(prune_root, analysis_dir)

    if not prune_root.exists() or not fixed_root.exists():
        return analysis_dir

    prune_latest = latest_metrics_by_seed(prune_root)
    fixed_latest = latest_metrics_by_seed(fixed_root)
    paired_seeds = sorted(set(prune_latest.keys()) & set(fixed_latest.keys()))
    if not paired_seeds:
        return analysis_dir

    alignment_csv = analysis_dir / "phase_seed_alignment.csv"
    run_builder(
        [
            sys.executable,
            "scripts/build_unpruned_phase_alignment.py",
            "--prune-root",
            str(prune_root),
            "--fixed-root",
            str(fixed_root),
            "--out-path",
            str(alignment_csv),
        ]
    )
    run_builder(
        [
            sys.executable,
            "scripts/build_unpruned_phase_browser.py",
            "--alignment-csv",
            str(alignment_csv),
            "--out-dir",
            str(analysis_dir / "by_prune_phase"),
        ]
    )
    run_builder(
        [
            sys.executable,
            "scripts/build_unpruned_phase_comparison_study.py",
            "--alignment-csv",
            str(alignment_csv),
            "--out-dir",
            str(analysis_dir / "phase_comparison"),
        ]
    )

    lines = [
        "# Unpruned Phase Seed Sweep Analysis",
        "",
        f"Results root: `{results_root}`",
        f"Task: `{task}`",
        "",
        "Generated files:",
        "- `prune_only_phase_inventory.csv`: one latest prune_only run per seed with its discovered phase",
        "- `prune_only_phase_counts.csv`: current phase counts and seed lists",
        "- `phase_seed_alignment.csv`: paired fixed/prune_only alignment for seeds where both modes exist",
        "- `by_prune_phase/`: browse tree grouped by paired prune phase",
        "- `phase_comparison/`: phase comparison plots for fixed runs",
    ]
    (analysis_dir / "README.md").write_text("\n".join(lines) + "\n")
    return analysis_dir


def main() -> None:
    args = parse_args()
    seeds = parse_seed_values(args)
    if not args.analysis_only:
        run_sweep(args, seeds)
    analysis_dir = build_analysis(args.results_root, args.task)
    print(analysis_dir)


if __name__ == "__main__":
    main()
