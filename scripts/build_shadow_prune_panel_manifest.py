import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


ALIGNMENT_CSV = Path(
    "results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis/phase_seed_alignment.csv"
)
OUT_PATH = Path(
    "results/followup_20260318/shadow_prune_fixed_panel/shadow_prune_panel_manifest.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified seed panel for shadow-prune follow-up runs."
    )
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _stratified_even_sample(seeds: Sequence[int], n_samples: int) -> List[int]:
    if len(seeds) <= n_samples:
        return list(seeds)
    if n_samples <= 1:
        return [seeds[len(seeds) // 2]]
    positions = [round(idx * (len(seeds) - 1) / (n_samples - 1)) for idx in range(n_samples)]
    ordered: List[int] = []
    seen = set()
    for pos in positions:
        seed = int(seeds[pos])
        if seed not in seen:
            ordered.append(seed)
            seen.add(seed)
    for seed in seeds:
        if len(ordered) >= n_samples:
            break
        if int(seed) not in seen:
            ordered.append(int(seed))
            seen.add(int(seed))
    return ordered


def choose_common_phase_panel(
    alignment: pd.DataFrame, phase: str, target_per_phase: int, must_include: Sequence[int]
) -> List[Dict[str, object]]:
    phase_rows = alignment[alignment["prune_only_phase"] == phase].sort_values("fixed_final_minus_selected").copy()
    if phase_rows.empty:
        return []

    cuts = [0, len(phase_rows) // 3, 2 * len(phase_rows) // 3, len(phase_rows)]
    stratum_names = ["low_degradation", "mid_degradation", "high_degradation"]
    selected: List[Dict[str, object]] = []
    selected_seeds = set()
    stratum_target = target_per_phase // 3

    for idx, stratum_name in enumerate(stratum_names):
        stratum = phase_rows.iloc[cuts[idx] : cuts[idx + 1]].copy()
        seeds = stratum["seed"].astype(int).tolist()
        chosen = _stratified_even_sample(seeds, stratum_target)
        for seed in chosen:
            selected_seeds.add(seed)
            selected.append(
                {
                    "phase": phase,
                    "seed": int(seed),
                    "selection_group": stratum_name,
                    "selection_reason": "stratified_by_fixed_final_minus_selected",
                }
            )

    for seed in must_include:
        if seed not in selected_seeds:
            selected.append(
                {
                    "phase": phase,
                    "seed": int(seed),
                    "selection_group": "continuity_exemplar",
                    "selection_reason": "carry_over_from_shadow_prune_pilot",
                }
            )
            selected_seeds.add(seed)

    remaining_rows = phase_rows[~phase_rows["seed"].isin(selected_seeds)]
    while len(selected) < target_per_phase and not remaining_rows.empty:
        middle = remaining_rows.iloc[len(remaining_rows) // 2]
        selected.append(
            {
                "phase": phase,
                "seed": int(middle["seed"]),
                "selection_group": "fill_to_target",
                "selection_reason": "median_fill_after_stratification",
            }
        )
        selected_seeds.add(int(middle["seed"]))
        remaining_rows = remaining_rows[~remaining_rows["seed"].isin(selected_seeds)]

    return sorted(selected, key=lambda row: row["seed"])


def main() -> None:
    args = parse_args()
    alignment = pd.read_csv(args.alignment_csv)

    rows: List[Dict[str, object]] = []
    rows.extend(choose_common_phase_panel(alignment, "prune0+2", target_per_phase=12, must_include=[0]))
    rows.extend(choose_common_phase_panel(alignment, "prune0_only", target_per_phase=12, must_include=[1]))

    for phase, reason in [("no_prune", "rare_phase_exhaustive"), ("prune2_only", "rare_phase_exhaustive")]:
        subset = alignment[alignment["prune_only_phase"] == phase].sort_values("seed")
        for _, row in subset.iterrows():
            rows.append(
                {
                    "phase": phase,
                    "seed": int(row["seed"]),
                    "selection_group": "all_seeds",
                    "selection_reason": reason,
                }
            )

    rows = sorted(rows, key=lambda row: (row["phase"], row["seed"]))
    write_csv(args.out_path, rows)


if __name__ == "__main__":
    main()
