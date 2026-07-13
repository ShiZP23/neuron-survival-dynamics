from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/followup_20260319/mode_phase_overlap_hard"

PRUNE_ONLY_SWEEP = (
    ROOT
    / "results/followup_20260317/unpruned_phase_seed_sweep/hard/"
    / "unpruned_phase_seed_sweep_analysis/prune_only_phase_inventory.csv"
)
TASK_ATLAS_ROWS = ROOT / "results/publishable_pilot_20260313/task_atlas/task_atlas_rows.csv"
SPLIT_PATTERN_ROWS = ROOT / "results/publishable_pilot_20260313/seed_pattern_review/pattern_rows.csv"
LEGACY_HARD_ROOT = ROOT / "former results/3.12results，128/hard"

CANONICAL_LEGACY_RUNS = {
    "prune_grow_split": {
        0: "seed0",
        1: "20260311_191005",
        2: "20260311_191130",
        3: "20260311_191256",
        4: "20260311_191422",
        5: "20260311_191546",
        6: "20260311_191711",
        7: "20260311_191837",
        8: "20260311_192002",
        9: "20260311_192126",
    },
    "prune_grow_random": {
        0: "20260311_192315",
        1: "20260311_192440",
        2: "20260311_192604",
        3: "20260311_192730",
        4: "20260311_192855",
        5: "20260311_193019",
        6: "20260311_193143",
        7: "20260311_193309",
        8: "20260311_193435",
        9: "20260311_193559",
    },
}

MODE_LABEL = {
    "prune_only": "prune_only",
    "prune_grow_split": "grow_split",
    "prune_grow_random": "grow_random",
}


def classify_phase(pruned_0_total: int, pruned_1_total: int, pruned_2_total: int) -> str:
    if pruned_0_total > 0 and pruned_1_total == 0 and pruned_2_total > 0:
        return "prune0+2"
    if pruned_0_total > 0 and pruned_1_total == 0 and pruned_2_total == 0:
        return "prune0_only"
    if pruned_0_total == 0 and pruned_1_total == 0 and pruned_2_total > 0:
        return "prune2_only"
    if pruned_0_total == 0 and pruned_1_total == 0 and pruned_2_total == 0:
        return "no_prune"
    return "other"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_prune_only_sweep_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for row in read_csv_dicts(PRUNE_ONLY_SWEEP):
        rows.append(
            {
                "dataset": "hard_prune_only_seed_sweep_200",
                "mode": "prune_only",
                "seed": int(row["seed"]),
                "run_name": Path(row["run_path"]).name,
                "run_path": row["run_path"],
                "phase": row["phase"],
                "pruned_0_total": int(row["pruned_0_total"]),
                "pruned_1_total": int(row["pruned_1_total"]),
                "pruned_2_total": int(row["pruned_2_total"]),
                "source_note": "followup_20260317 hard prune_only sweep",
            }
        )
    return rows


def build_publishable_shared_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for row in read_csv_dicts(TASK_ATLAS_ROWS):
        if row["task"] != "hard":
            continue
        if row["mode"] not in MODE_LABEL:
            continue
        pruned_0_total = int(row["pruned_0_total"])
        pruned_1_total = int(row["pruned_1_total"])
        pruned_2_total = int(row["pruned_2_total"])
        rows.append(
            {
                "dataset": "hard_publishable_shared5",
                "mode": row["mode"],
                "seed": int(row["seed"]),
                "run_name": row["run_name"],
                "run_path": row["run_dir"],
                "phase": classify_phase(pruned_0_total, pruned_1_total, pruned_2_total),
                "pruned_0_total": pruned_0_total,
                "pruned_1_total": pruned_1_total,
                "pruned_2_total": pruned_2_total,
                "source_note": "publishable_pilot_20260313 task_atlas hard pool",
            }
        )
    return rows


def build_legacy_canonical_rows(mode: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for seed, run_name in sorted(CANONICAL_LEGACY_RUNS[mode].items()):
        metrics_path = LEGACY_HARD_ROOT / mode / f"seed_{seed}" / run_name / "metrics.csv"
        metrics_rows = read_csv_dicts(metrics_path)
        pruned_0_total = sum(int(row["pruned_0"]) for row in metrics_rows)
        pruned_1_total = sum(int(row["pruned_1"]) for row in metrics_rows)
        pruned_2_total = sum(int(row["pruned_2"]) for row in metrics_rows)
        rows.append(
            {
                "dataset": "hard_legacy_canonical10",
                "mode": mode,
                "seed": seed,
                "run_name": run_name,
                "run_path": str(metrics_path.parent.relative_to(ROOT)),
                "phase": classify_phase(pruned_0_total, pruned_1_total, pruned_2_total),
                "pruned_0_total": pruned_0_total,
                "pruned_1_total": pruned_1_total,
                "pruned_2_total": pruned_2_total,
                "source_note": "legacy_3_12_results_128 canonical hard seed pool",
            }
        )
    return rows


def build_split_full_pool_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for row in read_csv_dicts(SPLIT_PATTERN_ROWS):
        pattern = row["pattern"]
        phase = {
            "p0+p2": "prune0+2",
            "p0_only": "prune0_only",
            "none": "no_prune",
        }.get(pattern, pattern)
        rows.append(
            {
                "dataset": "hard_legacy_split_full_pool",
                "mode": "prune_grow_split",
                "seed": int(row["seed_dir"].split("_")[1]),
                "run_name": row["run_name"],
                "run_path": row["run_path"],
                "phase": phase,
                "pruned_0_total": int(row["pruned_0_total"]),
                "pruned_1_total": int(row["pruned_1_total"]),
                "pruned_2_total": int(row["pruned_2_total"]),
                "source_note": "seed_pattern_review full split pool",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_counts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    dataset_totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        dataset = str(row["dataset"])
        mode = str(row["mode"])
        phase = str(row["phase"])
        counts[(dataset, mode, phase)] += 1
        dataset_totals[(dataset, mode)] += 1
    out: list[dict[str, object]] = []
    for dataset, mode, phase in sorted(counts):
        total = dataset_totals[(dataset, mode)]
        count = counts[(dataset, mode, phase)]
        out.append(
            {
                "dataset": dataset,
                "mode": mode,
                "phase": phase,
                "count": count,
                "share": count / total if total else 0.0,
                "dataset_mode_total": total,
            }
        )
    return out


def build_shared_alignment(
    prune_only_rows: list[dict[str, str | int]],
    legacy_rows: list[dict[str, str | int]],
    publishable_rows: list[dict[str, str | int]],
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for row in prune_only_rows:
        seed = int(row["seed"])
        if 0 <= seed <= 9:
            by_key[("prune_only_sweep", seed)] = row
    for row in legacy_rows:
        seed = int(row["seed"])
        by_key[(f"{row['mode']}_legacy", seed)] = row
    for row in publishable_rows:
        seed = int(row["seed"])
        if 0 <= seed <= 4:
            by_key[(f"{row['mode']}_publishable", seed)] = row

    rows: list[dict[str, object]] = []
    for seed in range(10):
        prune_only = by_key[("prune_only_sweep", seed)]
        grow_split = by_key[("prune_grow_split_legacy", seed)]
        grow_random = by_key[("prune_grow_random_legacy", seed)]
        legacy_match_all = (
            prune_only["phase"] == grow_split["phase"] == grow_random["phase"]
        )
        out_row = {
            "seed": seed,
            "prune_only_phase": prune_only["phase"],
            "grow_split_phase": grow_split["phase"],
            "grow_random_phase": grow_random["phase"],
            "legacy_match_all": legacy_match_all,
            "prune_only_run_path": prune_only["run_path"],
            "grow_split_run_path": grow_split["run_path"],
            "grow_random_run_path": grow_random["run_path"],
        }
        if seed <= 4:
            publishable_prune = by_key[("prune_only_publishable", seed)]
            publishable_split = by_key[("prune_grow_split_publishable", seed)]
            publishable_random = by_key[("prune_grow_random_publishable", seed)]
            out_row.update(
                {
                    "publishable_prune_only_phase": publishable_prune["phase"],
                    "publishable_grow_split_phase": publishable_split["phase"],
                    "publishable_grow_random_phase": publishable_random["phase"],
                    "publishable_match_all": (
                        publishable_prune["phase"]
                        == publishable_split["phase"]
                        == publishable_random["phase"]
                    ),
                    "publishable_prune_only_run_path": publishable_prune["run_path"],
                    "publishable_grow_split_run_path": publishable_split["run_path"],
                    "publishable_grow_random_run_path": publishable_random["run_path"],
                }
            )
        else:
            out_row.update(
                {
                    "publishable_prune_only_phase": "",
                    "publishable_grow_split_phase": "",
                    "publishable_grow_random_phase": "",
                    "publishable_match_all": "",
                    "publishable_prune_only_run_path": "",
                    "publishable_grow_split_run_path": "",
                    "publishable_grow_random_run_path": "",
                }
            )
        rows.append(out_row)
    return rows


def build_presence_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    phases_by_mode_dataset: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        phases_by_mode_dataset[(str(row["dataset"]), str(row["mode"]))].add(str(row["phase"]))
    out: list[dict[str, object]] = []
    for (dataset, mode), phases in sorted(phases_by_mode_dataset.items()):
        out.append(
            {
                "dataset": dataset,
                "mode": mode,
                "phases_present": ",".join(sorted(phases)),
                "has_prune0_plus_2": "prune0+2" in phases,
                "has_prune0_only": "prune0_only" in phases,
                "has_prune2_only": "prune2_only" in phases,
                "has_no_prune": "no_prune" in phases,
                "has_other": "other" in phases,
            }
        )
    return out


def write_readme(
    all_rows: list[dict[str, object]],
    counts_rows: list[dict[str, object]],
    alignment_rows: list[dict[str, object]],
) -> None:
    def lookup(dataset: str, mode: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in counts_rows:
            if row["dataset"] == dataset and row["mode"] == mode:
                out[str(row["phase"])] = int(row["count"])
        return out

    legacy_split_counts = lookup("hard_legacy_canonical10", "prune_grow_split")
    legacy_random_counts = lookup("hard_legacy_canonical10", "prune_grow_random")
    sweep_counts = lookup("hard_prune_only_seed_sweep_200", "prune_only")
    split_full_counts = lookup("hard_legacy_split_full_pool", "prune_grow_split")

    legacy_all_match = all(bool(row["legacy_match_all"]) for row in alignment_rows)
    publishable_rows = [row for row in alignment_rows if row["publishable_match_all"] != ""]
    publishable_all_match = all(bool(row["publishable_match_all"]) for row in publishable_rows)

    lines = [
        "# Hard Mode Phase Overlap: prune_only vs grow_split vs grow_random",
        "",
        "## 结论",
        "",
        (
            "- 三种模式在 `hard` 上确实出现了同一套粗 phase："
            "`prune0+2`、`prune0_only`、`no_prune`。"
        ),
        (
            "- 在当前已检查的数据里，`grow_split` 和 `grow_random` 都没有出现"
            "`layer1` 剪枝相。"
        ),
        (
            "- `prune2_only` 目前只在大规模 `prune_only` sweep 中出现；"
            "当前 `grow_split` / `grow_random` 的 canonical hard 池里没有看到。"
        ),
        "",
        "## 最强证据",
        "",
        (
            f"- 共享 seed 的 `publishable hard 5-seed` 池里，"
            f"`prune_only`、`grow_split`、`grow_random` 的 phase 映射完全一致："
            f"`match_all = {publishable_all_match}`。"
        ),
        (
            f"- `legacy canonical hard 10-seed` 池里，三种模式在 `seed 0-9` 上的"
            f" phase 映射仍然完全一致：`match_all = {legacy_all_match}`。"
        ),
        (
            f"- `grow_split` canonical 10-seed 计数：`{legacy_split_counts}`；"
            f"`grow_random` canonical 10-seed 计数：`{legacy_random_counts}`。"
        ),
        (
            f"- `prune_only` 200-seed sweep 计数：`{sweep_counts}`。"
            " 这说明 `prune_only` 的 phase 空间更大，额外出现了少量 `prune2_only`。"
        ),
        (
            f"- `grow_split` 的历史 full pool 也仍只落在 `prune0+2 / prune0_only / no_prune`："
            f"`{split_full_counts}`。"
        ),
        "",
        "## 文件",
        "",
        "- `phase_rows.csv`: 统一 phase 行表。",
        "- `mode_phase_counts.csv`: 各数据池各模式的 phase 计数。",
        "- `shared_seed_phase_alignment.csv`: 同 seed 跨模式的直接对照。",
        "- `phase_presence_summary.csv`: 每个数据池/模式出现过哪些 phase。",
        "",
        "## 命名对齐",
        "",
        "- 历史 `p0+p2` 对应当前 `prune0+2`。",
        "- 历史 `p0_only` 对应当前 `prune0_only`。",
        "- 历史 `none` 对应当前 `no_prune`。",
        "",
        "## 口径说明",
        "",
        (
            "- `grow_split` / `grow_random` 的 canonical hard 10-seed 来自"
            " `former results/3.12results，128/hard/...`。"
        ),
        (
            "- `grow_split` 的 full pool 额外包含 `seed_0` 的 data/factor 变体，"
            "因此它用于回答“是否出现相同分相”，不用于与其他模式做严格频率比较。"
        ),
        (
            "- `prune_only` 使用 `results/followup_20260317/unpruned_phase_seed_sweep`"
            " 的 200-seed 大 sweep 作为主口径。"
        ),
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prune_only_rows = build_prune_only_sweep_rows()
    publishable_rows = build_publishable_shared_rows()
    legacy_split_rows = build_legacy_canonical_rows("prune_grow_split")
    legacy_random_rows = build_legacy_canonical_rows("prune_grow_random")
    split_full_rows = build_split_full_pool_rows()

    all_rows = (
        prune_only_rows
        + publishable_rows
        + legacy_split_rows
        + legacy_random_rows
        + split_full_rows
    )
    all_rows.sort(key=lambda row: (str(row["dataset"]), str(row["mode"]), int(row["seed"]), str(row["run_name"])))

    phase_rows_fields = [
        "dataset",
        "mode",
        "seed",
        "run_name",
        "run_path",
        "phase",
        "pruned_0_total",
        "pruned_1_total",
        "pruned_2_total",
        "source_note",
    ]
    write_csv(OUT_DIR / "phase_rows.csv", all_rows, phase_rows_fields)

    counts_rows = build_counts(all_rows)
    write_csv(
        OUT_DIR / "mode_phase_counts.csv",
        counts_rows,
        ["dataset", "mode", "phase", "count", "share", "dataset_mode_total"],
    )

    alignment_rows = build_shared_alignment(
        prune_only_rows,
        legacy_split_rows + legacy_random_rows,
        publishable_rows,
    )
    write_csv(
        OUT_DIR / "shared_seed_phase_alignment.csv",
        alignment_rows,
        [
            "seed",
            "prune_only_phase",
            "grow_split_phase",
            "grow_random_phase",
            "legacy_match_all",
            "prune_only_run_path",
            "grow_split_run_path",
            "grow_random_run_path",
            "publishable_prune_only_phase",
            "publishable_grow_split_phase",
            "publishable_grow_random_phase",
            "publishable_match_all",
            "publishable_prune_only_run_path",
            "publishable_grow_split_run_path",
            "publishable_grow_random_run_path",
        ],
    )

    presence_rows = build_presence_summary(all_rows)
    write_csv(
        OUT_DIR / "phase_presence_summary.csv",
        presence_rows,
        [
            "dataset",
            "mode",
            "phases_present",
            "has_prune0_plus_2",
            "has_prune0_only",
            "has_prune2_only",
            "has_no_prune",
            "has_other",
        ],
    )

    write_readme(all_rows, counts_rows, alignment_rows)


if __name__ == "__main__":
    main()
