import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from json import JSONDecodeError
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "research"
TASK_NAMES = {"simple", "medium", "hard"}
MODE_NAMES = {"fixed", "prune_only", "prune_grow_random", "prune_grow_split"}
CONFIG_KEYS = [
    "epochs",
    "n_train",
    "n_val",
    "n_test",
    "lr",
    "update_interval",
    "min_neurons",
    "ablation_epsilon_ratio",
    "active_threshold",
]
COLLECTIONS = [
    {
        "section": "data/raw",
        "slug": "legacy_3_12_results_128",
        "title": "Raw Data: Legacy 3.12 Results 128",
        "kind": "raw",
        "description": "Historical exploratory runs used for legacy seed-pattern and structural-phase analysis.",
        "sources": [
            ("hard_fixed", "former results/3.12results，128/hard/fixed"),
            ("hard_prune_grow_random", "former results/3.12results，128/hard/prune_grow_random"),
            ("hard_prune_grow_split", "former results/3.12results，128/hard/prune_grow_split"),
            ("medium_prune_grow_random", "former results/3.12results，128/medium/prune_grow_random"),
            ("medium_prune_grow_split", "former results/3.12results，128/medium/prune_grow_split"),
        ],
        "notes": [
            "Legacy source; validation-loss checkpoint selection is not available.",
            "Use this collection for historical diagnosis, not for the main protocol table.",
        ],
        "builder_scripts": [],
    },
    {
        "section": "data/raw",
        "slug": "legacy_3000_20_seed_sweeps",
        "title": "Raw Data: Legacy 3000-20 Seed Sweeps",
        "kind": "raw",
        "description": "Older seed sweeps kept as an untouched raw archive.",
        "sources": [
            ("prune_grow_random", "former results/3000:20/prune_grow_random"),
            ("prune_grow_split", "former results/3000:20/prune_grow_split"),
        ],
        "notes": [
            "This archive has incomplete metadata and may not contain config files for every run.",
            "Treat it as historical reference only.",
        ],
        "builder_scripts": [],
    },
    {
        "section": "data/raw",
        "slug": "legacy_full_grid_results",
        "title": "Raw Data: Legacy Full Grid Results",
        "kind": "raw",
        "description": "Older full-grid result pool preserved as a raw archive before later protocol cleanup.",
        "sources": [
            ("simple", "former results/results/simple"),
            ("medium", "former results/results/medium"),
            ("hard", "former results/results/hard"),
        ],
        "notes": [
            "Legacy archive with mixed provenance.",
            "Keep separate from the validation-selected mainline protocol.",
        ],
        "builder_scripts": [],
    },
    {
        "section": "data/raw",
        "slug": "mainline_publishable_pilot_20260313_runs",
        "title": "Raw Data: Mainline Publishable Pilot 2026-03-13",
        "kind": "raw",
        "description": "Current validation-selected mainline run pool used for cross-task comparisons.",
        "sources": [
            ("simple", "results/publishable_pilot_20260313/simple"),
            ("medium", "results/publishable_pilot_20260313/medium"),
            ("hard_fixed", "results/publishable_pilot_20260313/hard/fixed"),
            ("hard_prune_only", "results/publishable_pilot_20260313/hard/prune_only"),
            ("hard_prune_grow_random", "results/publishable_pilot_20260313/hard/prune_grow_random"),
            ("hard_prune_grow_split", "results/publishable_pilot_20260313/hard/prune_grow_split"),
        ],
        "notes": [
            "This is the mainline protocol source with validation-selected checkpoints.",
            "Derived summaries and paper-oriented figures are filed under analysis/project_level.",
        ],
        "builder_scripts": ["scripts/summarize_protocol_runs.py", "scripts/build_task_atlas.py"],
    },
    {
        "section": "data/raw",
        "slug": "followup_hard_20260313_runs",
        "title": "Raw Data: Hard Follow-Up 2026-03-13",
        "kind": "raw",
        "description": "Targeted hard-task follow-up runs used to probe prune-only and split behavior.",
        "sources": [
            ("hard_prune_only", "results/followup_20260313/hard/prune_only"),
            ("hard_prune_grow_split", "results/followup_20260313/hard/prune_grow_split"),
        ],
        "notes": [
            "Focused exploratory follow-up on the hard task.",
            "Follow-up analysis outputs are filed under analysis/project_level.",
        ],
        "builder_scripts": ["scripts/analyze_followup_results.py"],
    },
    {
        "section": "data/raw",
        "slug": "phase_freeze_interventions_pilot_runs_20260314",
        "title": "Raw Data: Phase Freeze Interventions Pilot 2026-03-14",
        "kind": "raw",
        "description": "Pilot intervention reruns that freeze structural updates after baseline commit or best epochs.",
        "sources": [
            ("raw_runs", "results/followup_20260314/phase_freeze_interventions_pilot_runs"),
        ],
        "notes": [
            "Representative pilot reruns for causal follow-up on structural-phase degradation.",
            "Treat this as a focused intervention pool rather than a broad seed sweep.",
        ],
        "builder_scripts": ["scripts/run_phase_freeze_interventions.py"],
    },
    {
        "section": "data/raw",
        "slug": "adhoc_partial_runs_20260312",
        "title": "Raw Data: Ad Hoc Partial Runs 2026-03-12",
        "kind": "raw",
        "description": "Small exploratory run fragments kept separate from both legacy archives and the mainline protocol.",
        "sources": [
            ("simple_prune_only", "results/simple/prune_only"),
            ("hard_prune_only", "results/hard/prune_only"),
            ("hard_prune_grow_split", "results/hard/prune_grow_split"),
        ],
        "notes": [
            "Partial exploratory runs with uneven coverage.",
            "Do not mix these with the mainline pilot or formal studies.",
        ],
        "builder_scripts": [],
    },
    {
        "section": "data/curated",
        "slug": "curated_legacy_views",
        "title": "Curated Data: Legacy Views",
        "kind": "curated",
        "description": "Non-destructive curated symlink view over the legacy archives.",
        "sources": [
            ("legacy_3_12_results_128", "curated_results/3.12results，128"),
            ("manifest", "curated_results/manifest_all.csv"),
        ],
        "notes": [
            "Use this when you want cleaned access paths without touching the original legacy folders.",
        ],
        "builder_scripts": ["scripts/curate_results.py"],
    },
    {
        "section": "data/curated",
        "slug": "curated_current_views",
        "title": "Curated Data: Current Result Views",
        "kind": "curated",
        "description": "Curated symlink view over current protocol and follow-up runs.",
        "sources": [
            ("current_results", "curated_results/results"),
            ("manifest", "curated_results/results/manifest.csv"),
        ],
        "notes": [
            "This keeps seed-sweep and canonical runs grouped without moving the source outputs.",
        ],
        "builder_scripts": ["scripts/curate_results.py"],
    },
    {
        "section": "analysis/project_level",
        "slug": "mainline_core_summary_20260313",
        "title": "Project Analysis: Mainline Core Summary 2026-03-13",
        "kind": "analysis",
        "description": "Top-level summary tables and figures for the validation-selected mainline pilot.",
        "sources": [
            ("summary_three_task", "results/publishable_pilot_20260313/summary_three_task.csv"),
            ("summary_three_task_stats", "results/publishable_pilot_20260313/summary_three_task_stats.csv"),
            ("summary_three_task_partial", "results/publishable_pilot_20260313/summary_three_task_partial.csv"),
            ("selected_loss_plot", "results/publishable_pilot_20260313/three_task_selected_loss.png"),
            ("selected_params_plot", "results/publishable_pilot_20260313/three_task_selected_params.png"),
            ("figure_catalog", "results/publishable_pilot_20260313/figure_catalog.md"),
        ],
        "notes": [
            "These are derived summaries, not raw runs.",
        ],
        "builder_scripts": ["scripts/summarize_protocol_runs.py"],
    },
    {
        "section": "analysis/project_level",
        "slug": "mainline_hard_diagnostics_20260313",
        "title": "Project Analysis: Mainline Hard Diagnostics 2026-03-13",
        "kind": "analysis",
        "description": "Hard-task diagnostic summaries kept separate from the core summary tables.",
        "sources": [
            ("analysis", "results/publishable_pilot_20260313/hard/analysis"),
            ("analysis_partial", "results/publishable_pilot_20260313/hard/analysis_partial"),
        ],
        "notes": [
            "Diagnostic-only package for hard-task deep dives.",
        ],
        "builder_scripts": ["scripts/analyze_legacy_128_results.py"],
    },
    {
        "section": "analysis/project_level",
        "slug": "task_atlas_20260313",
        "title": "Project Analysis: Task Atlas 2026-03-13",
        "kind": "analysis",
        "description": "Task-level figure atlas with primary, supplementary, and archive panels.",
        "sources": [
            ("task_atlas", "results/publishable_pilot_20260313/task_atlas"),
        ],
        "notes": [
            "Primary project-level figure package for task-by-task comparison.",
        ],
        "builder_scripts": ["scripts/build_task_atlas.py"],
    },
    {
        "section": "analysis/project_level",
        "slug": "seed_pattern_review_20260313",
        "title": "Project Analysis: Seed Pattern Review 2026-03-13",
        "kind": "analysis",
        "description": "Pattern-focused package for the legacy prune-grow split phase taxonomy.",
        "sources": [
            ("seed_pattern_review", "results/publishable_pilot_20260313/seed_pattern_review"),
        ],
        "notes": [
            "Keep separate from the formal structural-phase study because this package is older and narrower.",
        ],
        "builder_scripts": ["scripts/build_seed_pattern_review.py"],
    },
    {
        "section": "analysis/project_level",
        "slug": "experiment_review_20260313",
        "title": "Project Analysis: Experiment Review 2026-03-13",
        "kind": "analysis",
        "description": "Broad review package that integrates mainline and legacy summary views.",
        "sources": [
            ("review", "results/publishable_pilot_20260313/review"),
        ],
        "notes": [
            "Narrative review package; not a raw data source.",
        ],
        "builder_scripts": ["scripts/build_experiment_review.py"],
    },
    {
        "section": "analysis/project_level",
        "slug": "followup_hard_analysis_20260313",
        "title": "Project Analysis: Hard Follow-Up Analysis 2026-03-13",
        "kind": "analysis",
        "description": "Derived figures and summary tables for the hard-task follow-up experiments.",
        "sources": [
            ("analysis", "results/followup_20260313/hard/analysis"),
        ],
        "notes": [
            "Companion analysis package for the hard follow-up raw runs.",
        ],
        "builder_scripts": ["scripts/analyze_followup_results.py"],
    },
    {
        "section": "analysis/studies",
        "slug": "prune_only_vs_fixed_final_loss_20260314",
        "title": "Study Package: Prune-Only vs Fixed Final Loss 2026-03-14",
        "kind": "study",
        "description": "Focused study package combining the study spec and the generated artifacts.",
        "sources": [
            ("study_spec", "studies/prune_only_vs_fixed_final_loss"),
            ("artifacts", "results/studies/prune_only_vs_fixed_final_loss_20260314"),
        ],
        "notes": [
            "Final-state focused study; keep separate from the main validation-selected protocol tables.",
        ],
        "builder_scripts": ["scripts/build_prune_only_vs_fixed_study.py"],
    },
    {
        "section": "analysis/studies",
        "slug": "prune_only_vs_grow_final_loss_20260314",
        "title": "Study Package: Prune-Only vs Grow Final Loss 2026-03-14",
        "kind": "study",
        "description": "Focused study package comparing prune-only against grow variants under a final-loss lens.",
        "sources": [
            ("study_spec", "studies/prune_only_vs_grow_final_loss"),
            ("artifacts", "results/studies/prune_only_vs_grow_final_loss_20260314"),
        ],
        "notes": [
            "Secondary focused study kept for comparison.",
        ],
        "builder_scripts": ["scripts/build_prune_only_vs_grow_study.py"],
    },
    {
        "section": "analysis/studies",
        "slug": "structural_phase_effects_20260314",
        "title": "Study Package: Structural Phase Effects 2026-03-14",
        "kind": "study",
        "description": "Focused study package for prune0+2, prune0_only, and no_prune structural phases.",
        "sources": [
            ("study_spec", "studies/structural_phase_effects"),
            ("artifacts", "results/studies/structural_phase_effects_20260314"),
        ],
        "notes": [
            "Legacy-source structural-phase study with explicit seed and data-realization analysis.",
        ],
        "builder_scripts": ["scripts/build_structural_phase_study.py"],
    },
    {
        "section": "analysis/studies",
        "slug": "phase_freeze_interventions_pilot_20260314",
        "title": "Study Package: Phase Freeze Interventions Pilot 2026-03-14",
        "kind": "study",
        "description": "Focused pilot package testing freeze-after-commit and freeze-after-best interventions.",
        "sources": [
            ("study_spec", "studies/phase_freeze_interventions_pilot"),
            ("artifacts", "results/studies/phase_freeze_interventions_pilot_20260314"),
        ],
        "notes": [
            "Intervention follow-up kept separate from the observational structural-phase study.",
        ],
        "builder_scripts": ["scripts/run_phase_freeze_interventions.py"],
    },
]


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_symlink(target: Path, link_path: Path) -> None:
    ensure_dir(link_path.parent)
    if link_path.exists() or link_path.is_symlink():
        remove_path(link_path)
    link_path.symlink_to(target.resolve())


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str] = ()) -> None:
    if not rows and not fieldnames:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        resolved_fieldnames = list(fieldnames) if fieldnames else list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=resolved_fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def iter_metric_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("metrics*.csv"))


def primary_metrics_file(run_dir: Path) -> Path:
    standard = run_dir / "metrics.csv"
    if standard.exists():
        return standard
    metric_files = iter_metric_files(run_dir)
    return metric_files[0]


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_run_dirs(source_paths: Sequence[Path]) -> List[Path]:
    seen = set()
    run_dirs: List[Path] = []
    for source_path in source_paths:
        if not source_path.exists() or source_path.is_file():
            continue
        for config_path in source_path.rglob("config.json"):
            run_dir = config_path.parent.resolve()
            if not iter_metric_files(run_dir):
                continue
            if run_dir not in seen:
                seen.add(run_dir)
                run_dirs.append(run_dir)
    return sorted(run_dirs)


def has_val_loss(metrics_path: Path) -> bool:
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
    return first is not None and "val_loss" in first


def _format_values(values: Iterable[object]) -> str:
    uniq = sorted({str(value) for value in values if value is not None})
    if not uniq:
        return "N/A"
    if len(uniq) <= 5:
        return ", ".join(uniq)
    return ", ".join(uniq[:5]) + f", ... ({len(uniq)} values)"


def load_json_if_possible(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except (OSError, JSONDecodeError):
        return {}


def infer_path_label(run_dir: Path, candidates: Iterable[str]) -> str:
    for part in run_dir.parts:
        if part in candidates:
            return part
    return ""


def infer_seed_dir(run_dir: Path) -> str:
    for part in reversed(run_dir.parts[:-1]):
        if part.startswith("seed_"):
            return part
    return ""


def rel_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def nested_results_penalty(path_str: str) -> int:
    parts = Path(path_str).parts
    return 1 if "results" in parts[4:] else 0


def safe_component(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in text)


def run_link_name(record: Dict[str, object]) -> str:
    return "__".join(
        [
            f"seed_{safe_component(record['seed_label'])}",
            f"data_{safe_component(record['data_seed_label'])}",
            f"model_{safe_component(record['model_seed_label'])}",
            f"shuffle_{safe_component(record['shuffle_seed_label'])}",
            f"struct_{safe_component(record['structure_seed_label'])}",
            f"run_{safe_component(record['run_name'])}",
        ]
    )


def collection_sort_key(record: Dict[str, object]) -> Tuple:
    return (
        int(record["collection_order"]),
        0 if record["metrics_file_name"] == "metrics.csv" else 1,
        0 if record["has_model"] else 1,
        nested_results_penalty(str(record["rel_run_path"])),
        len(Path(str(record["rel_run_path"])).parts),
        str(record["rel_run_path"]),
    )


def build_raw_run_records(raw_collections: Sequence[Dict]) -> Tuple[Dict[str, List[Dict]], List[Dict], Dict[str, List[Path]]]:
    records_by_collection: Dict[str, List[Dict]] = {}
    archive_files_by_collection: Dict[str, List[Path]] = {}
    all_records: List[Dict] = []
    for collection_order, collection in enumerate(raw_collections):
        slug = collection["slug"]
        source_paths = [REPO_ROOT / rel_path_str for _, rel_path_str in collection["sources"]]
        run_records: List[Dict] = []
        archive_files: List[Path] = []
        for source_path in source_paths:
            if source_path.exists() and source_path.is_dir():
                archive_files.extend(sorted(source_path.rglob("*.zip")))
        for run_dir in discover_run_dirs(source_paths):
            config_path = run_dir / "config.json"
            config = load_json_if_possible(config_path)
            metrics_path = primary_metrics_file(run_dir)
            resolved_run_dir = run_dir.resolve()
            task = str(config.get("task") or infer_path_label(resolved_run_dir, TASK_NAMES) or "unknown_task")
            mode = str(config.get("mode") or infer_path_label(resolved_run_dir, MODE_NAMES) or "unknown_mode")
            seed_dir = infer_seed_dir(resolved_run_dir)
            seed_value = config.get("seed")
            data_seed_value = config.get("data_seed")
            model_seed_value = config.get("model_seed")
            shuffle_seed_value = config.get("shuffle_seed")
            structure_seed_value = config.get("structure_seed")
            has_model = (resolved_run_dir / "model.pt").exists()
            has_best_model = (resolved_run_dir / "best_model.pt").exists()
            record = {
                "collection_slug": slug,
                "collection_order": collection_order,
                "run_path": str(resolved_run_dir),
                "rel_run_path": rel_path(resolved_run_dir),
                "task": task,
                "mode": mode,
                "seed_dir": seed_dir or "N/A",
                "run_name": resolved_run_dir.name,
                "seed": seed_value if seed_value is not None else "",
                "data_seed": data_seed_value if data_seed_value is not None else "",
                "model_seed": model_seed_value if model_seed_value is not None else "",
                "shuffle_seed": shuffle_seed_value if shuffle_seed_value is not None else "",
                "structure_seed": structure_seed_value if structure_seed_value is not None else "",
                "seed_label": seed_value if seed_value is not None else (seed_dir or resolved_run_dir.name),
                "data_seed_label": data_seed_value if data_seed_value is not None else "na",
                "model_seed_label": model_seed_value if model_seed_value is not None else "na",
                "shuffle_seed_label": shuffle_seed_value if shuffle_seed_value is not None else "na",
                "structure_seed_label": structure_seed_value if structure_seed_value is not None else "na",
                "config_valid": bool(config),
                "metrics_file_name": metrics_path.name,
                "metrics_rel_path": rel_path(metrics_path),
                "has_model": has_model,
                "has_best_model": has_best_model,
                "has_val_loss": has_val_loss(metrics_path),
                "nonstandard_metrics_name": metrics_path.name != "metrics.csv",
                "missing_checkpoint": not has_model and not has_best_model,
                "metrics_hash": sha256_file(metrics_path),
                "config_hash": sha256_file(config_path),
                "model_hash": sha256_file(resolved_run_dir / "model.pt"),
                "best_model_hash": sha256_file(resolved_run_dir / "best_model.pt"),
                "_strict_signature": (
                    sha256_file(metrics_path),
                    sha256_file(config_path),
                    sha256_file(resolved_run_dir / "model.pt"),
                    sha256_file(resolved_run_dir / "best_model.pt"),
                ),
                "_config_metrics_signature": (
                    sha256_file(metrics_path),
                    sha256_file(config_path),
                ),
            }
            run_records.append(record)
            all_records.append(record)
        records_by_collection[slug] = sorted(run_records, key=lambda record: str(record["rel_run_path"]))
        archive_files_by_collection[slug] = sorted({path.resolve() for path in archive_files}, key=lambda path: str(path))
    return records_by_collection, all_records, archive_files_by_collection


def annotate_duplicate_status(all_records: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    strict_groups: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    config_metrics_groups: Dict[Tuple[str, str], List[Dict]] = {}
    for record in all_records:
        strict_groups.setdefault(record["_strict_signature"], []).append(record)
        config_metrics_groups.setdefault(record["_config_metrics_signature"], []).append(record)

    strict_rows: List[Dict] = []
    for idx, group in enumerate(sorted([group for group in strict_groups.values() if len(group) > 1], key=lambda group: min(str(item["rel_run_path"]) for item in group)), start=1):
        group_id = f"strict_dup_{idx:03d}"
        canonical = min(group, key=collection_sort_key)
        for member in group:
            member["dedup_status"] = "canonical_duplicate" if member is canonical else "duplicate_alias"
            member["strict_duplicate_group"] = group_id
            member["canonical_rel_run_path"] = canonical["rel_run_path"]
            member["canonical_collection_slug"] = canonical["collection_slug"]
        for member in group:
            strict_rows.append(
                {
                    "strict_duplicate_group": group_id,
                    "canonical_rel_run_path": canonical["rel_run_path"],
                    "canonical_collection_slug": canonical["collection_slug"],
                    "member_rel_run_path": member["rel_run_path"],
                    "member_collection_slug": member["collection_slug"],
                    "member_status": member["dedup_status"],
                    "metrics_file_name": member["metrics_file_name"],
                }
            )
    for record in all_records:
        if "dedup_status" not in record:
            record["dedup_status"] = "unique"
            record["strict_duplicate_group"] = ""
            record["canonical_rel_run_path"] = record["rel_run_path"]
            record["canonical_collection_slug"] = record["collection_slug"]

    conflict_rows: List[Dict] = []
    for idx, group in enumerate(
        sorted(
            [
                group
                for group in config_metrics_groups.values()
                if len(group) > 1 and len({member["_strict_signature"] for member in group}) > 1
            ],
            key=lambda group: min(str(item["rel_run_path"]) for item in group)),
        start=1,
    ):
        group_id = f"same_logs_{idx:03d}"
        canonical = min(group, key=collection_sort_key)
        for member in group:
            member["same_config_metrics_conflict_group"] = group_id
        for member in group:
            conflict_rows.append(
                {
                    "same_config_metrics_conflict_group": group_id,
                    "canonical_rel_run_path": canonical["rel_run_path"],
                    "member_rel_run_path": member["rel_run_path"],
                    "member_collection_slug": member["collection_slug"],
                    "member_dedup_status": member["dedup_status"],
                    "model_hash": member["model_hash"] or "MISSING",
                    "has_model": member["has_model"],
                    "metrics_file_name": member["metrics_file_name"],
                }
            )
    for record in all_records:
        if "same_config_metrics_conflict_group" not in record:
            record["same_config_metrics_conflict_group"] = ""
    return strict_rows, conflict_rows


def summarize_raw_collection(run_records: Sequence[Dict], archive_files: Sequence[Path]) -> Dict[str, str]:
    tasks = set()
    modes = set()
    seed_dirs = set()
    val_flags = []
    configs: List[Dict] = []
    invalid_config_count = 0
    for record in run_records:
        if record["task"]:
            tasks.add(str(record["task"]))
        if record["mode"]:
            modes.add(str(record["mode"]))
        if record["seed_dir"] and record["seed_dir"] != "N/A":
            seed_dirs.add(str(record["seed_dir"]))
        val_flags.append(bool(record["has_val_loss"]))
        config_path = REPO_ROOT / str(record["rel_run_path"]) / "config.json"
        if config_path.exists():
            cfg = load_json_if_possible(config_path)
            if cfg:
                configs.append(cfg)
            else:
                invalid_config_count += 1
    if not val_flags:
        selection_protocol = "unknown"
    elif all(val_flags):
        selection_protocol = "validation checkpoints available"
    elif not any(val_flags):
        selection_protocol = "legacy final-test only"
    else:
        selection_protocol = "mixed"
    config_summary = {}
    for key in CONFIG_KEYS:
        config_summary[key] = _format_values(cfg.get(key) for cfg in configs if key in cfg)
    return {
        "run_count": str(len(run_records)),
        "canonical_run_count": str(sum(1 for record in run_records if record["dedup_status"] != "duplicate_alias")),
        "duplicate_alias_count": str(sum(1 for record in run_records if record["dedup_status"] == "duplicate_alias")),
        "strict_duplicate_group_count": str(len({record["strict_duplicate_group"] for record in run_records if record["strict_duplicate_group"]})),
        "same_log_conflict_count": str(len({record["same_config_metrics_conflict_group"] for record in run_records if record["same_config_metrics_conflict_group"]})),
        "nonstandard_metrics_count": str(sum(1 for record in run_records if record["nonstandard_metrics_name"])),
        "missing_checkpoint_count": str(sum(1 for record in run_records if record["missing_checkpoint"])),
        "archive_file_count": str(len(archive_files)),
        "task_coverage": _format_values(tasks),
        "mode_coverage": _format_values(modes),
        "seed_dir_count": str(len(seed_dirs)) if seed_dirs else "N/A",
        "selection_protocol": selection_protocol,
        "config_count": str(len(configs)),
        "invalid_config_count": str(invalid_config_count),
        **{f"cfg_{key}": value for key, value in config_summary.items()},
    }


def write_markdown(path: Path, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_collection_readme(
    collection_dir: Path,
    collection: Dict,
    source_paths: Sequence[Path],
    run_records: Sequence[Dict] = (),
    archive_files: Sequence[Path] = (),
) -> None:
    lines = [
        f"# {collection['title']}",
        "",
        f"Section: `{collection['section']}`",
        "",
        "Purpose:",
        f"- {collection['description']}",
        "",
        "Canonical links in this folder:",
    ]
    for label, rel_source_path in collection["sources"]:
        target = REPO_ROOT / rel_source_path
        suffix = "/" if target.is_dir() else ""
        lines.append(f"- `links/{label}{suffix}` -> `{rel_source_path}`")
    if collection["kind"] == "raw":
        summary = summarize_raw_collection(run_records, archive_files)
        lines.extend(
            [
                "",
                "Automatically extracted experimental summary:",
                f"- run_count: {summary['run_count']}",
                f"- canonical_run_count: {summary['canonical_run_count']}",
                f"- duplicate_alias_count: {summary['duplicate_alias_count']}",
                f"- strict_duplicate_group_count: {summary['strict_duplicate_group_count']}",
                f"- same_log_conflict_count: {summary['same_log_conflict_count']}",
                f"- nonstandard_metrics_count: {summary['nonstandard_metrics_count']}",
                f"- missing_checkpoint_count: {summary['missing_checkpoint_count']}",
                f"- archive_file_count: {summary['archive_file_count']}",
                f"- task_coverage: {summary['task_coverage']}",
                f"- mode_coverage: {summary['mode_coverage']}",
                f"- seed_dir_count: {summary['seed_dir_count']}",
                f"- selection_protocol: {summary['selection_protocol']}",
                f"- config_files_found: {summary['config_count']}",
                f"- invalid_config_files: {summary['invalid_config_count']}",
                "",
                "Browsable organization inside this folder:",
                "- `canonical_runs/`: one representative symlink per unique or canonical run",
                "- `duplicate_alias_runs/`: exact duplicate run copies that point to a canonical run elsewhere",
                "- `flagged_runs/`: runs with nonstandard metrics naming, missing checkpoints, or same-log conflicts",
                "- `run_inventory.csv`: full per-run manifest",
                "- `strict_duplicate_groups.csv`: exact duplicate groups in this collection",
                "- `same_config_metrics_conflicts.csv`: same config+metrics but non-identical checkpoint cases",
                "",
                "Common configuration values found in config.json:",
            ]
        )
        for key in CONFIG_KEYS:
            lines.append(f"- {key}: {summary[f'cfg_{key}']}")
        lines.extend(
            [
                "",
                "Condition note:",
                "- These values are extracted from available config files under the linked raw runs.",
                "- Nonstandard metric filenames such as `metrics_seed*.csv` are included in this audit.",
                "- Duplicate handling is based on file-content hashes, not only on folder names.",
            ]
        )
        if archive_files:
            lines.extend(["", "Archive files discovered in this collection:"])
            for archive_path in archive_files:
                lines.append(f"- `{rel_path(archive_path)}`")
    if collection["builder_scripts"]:
        lines.extend(["", "Builder scripts:"])
        for script in collection["builder_scripts"]:
            lines.append(f"- `{script}`")
    if collection["notes"]:
        lines.extend(["", "Notes:"])
        for note in collection["notes"]:
            lines.append(f"- {note}")
    write_markdown(collection_dir / "README.md", lines)


def inventory_rows(run_records: Sequence[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in sorted(run_records, key=lambda item: str(item["rel_run_path"])):
        issues = []
        if record["nonstandard_metrics_name"]:
            issues.append("nonstandard_metrics_name")
        if record["missing_checkpoint"]:
            issues.append("missing_checkpoint")
        if record["same_config_metrics_conflict_group"]:
            issues.append("same_config_metrics_conflict")
        rows.append(
            {
                "collection_slug": record["collection_slug"],
                "dedup_status": record["dedup_status"],
                "strict_duplicate_group": record["strict_duplicate_group"],
                "same_config_metrics_conflict_group": record["same_config_metrics_conflict_group"],
                "task": record["task"],
                "mode": record["mode"],
                "seed_dir": record["seed_dir"],
                "run_name": record["run_name"],
                "seed": record["seed"],
                "data_seed": record["data_seed"],
                "model_seed": record["model_seed"],
                "shuffle_seed": record["shuffle_seed"],
                "structure_seed": record["structure_seed"],
                "metrics_file_name": record["metrics_file_name"],
                "has_val_loss": record["has_val_loss"],
                "has_model": record["has_model"],
                "has_best_model": record["has_best_model"],
                "canonical_rel_run_path": record["canonical_rel_run_path"],
                "rel_run_path": record["rel_run_path"],
                "issues": " | ".join(issues),
            }
        )
    return rows


def strict_rows_for_collection(run_records: Sequence[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in sorted(run_records, key=lambda item: (str(item["strict_duplicate_group"]), str(item["rel_run_path"]))):
        if not record["strict_duplicate_group"]:
            continue
        rows.append(
            {
                "strict_duplicate_group": record["strict_duplicate_group"],
                "member_status": record["dedup_status"],
                "canonical_rel_run_path": record["canonical_rel_run_path"],
                "rel_run_path": record["rel_run_path"],
                "task": record["task"],
                "mode": record["mode"],
                "metrics_file_name": record["metrics_file_name"],
            }
        )
    return rows


def conflict_rows_for_collection(run_records: Sequence[Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in sorted(run_records, key=lambda item: (str(item["same_config_metrics_conflict_group"]), str(item["rel_run_path"]))):
        if not record["same_config_metrics_conflict_group"]:
            continue
        rows.append(
            {
                "same_config_metrics_conflict_group": record["same_config_metrics_conflict_group"],
                "dedup_status": record["dedup_status"],
                "rel_run_path": record["rel_run_path"],
                "canonical_rel_run_path": record["canonical_rel_run_path"],
                "model_hash": record["model_hash"] or "MISSING",
                "metrics_file_name": record["metrics_file_name"],
            }
        )
    return rows


def archive_rows_for_collection(archive_files: Sequence[Path]) -> List[Dict[str, object]]:
    return [{"archive_rel_path": rel_path(path)} for path in archive_files]


def write_raw_collection_assets(collection_dir: Path, run_records: Sequence[Dict], archive_files: Sequence[Path]) -> None:
    canonical_dir = collection_dir / "canonical_runs"
    duplicate_dir = collection_dir / "duplicate_alias_runs"
    flagged_dir = collection_dir / "flagged_runs"
    archive_dir = collection_dir / "archive_files"
    ensure_dir(canonical_dir)
    ensure_dir(duplicate_dir)
    ensure_dir(flagged_dir)
    if archive_files:
        ensure_dir(archive_dir)

    for record in run_records:
        seed_bucket = f"seed_{safe_component(record['seed_label'])}"
        link_name = run_link_name(record)
        task_dir = safe_component(record["task"])
        mode_dir = safe_component(record["mode"])
        target = REPO_ROOT / str(record["rel_run_path"])
        if record["dedup_status"] != "duplicate_alias":
            make_symlink(target, canonical_dir / task_dir / mode_dir / seed_bucket / link_name)
        else:
            make_symlink(target, duplicate_dir / task_dir / mode_dir / seed_bucket / link_name)
        if record["nonstandard_metrics_name"] or record["missing_checkpoint"] or record["same_config_metrics_conflict_group"]:
            make_symlink(target, flagged_dir / task_dir / mode_dir / seed_bucket / link_name)

    for archive_path in archive_files:
        make_symlink(archive_path, archive_dir / safe_component(archive_path.name))

    write_csv_rows(
        collection_dir / "run_inventory.csv",
        inventory_rows(run_records),
        fieldnames=[
            "collection_slug",
            "dedup_status",
            "strict_duplicate_group",
            "same_config_metrics_conflict_group",
            "task",
            "mode",
            "seed_dir",
            "run_name",
            "seed",
            "data_seed",
            "model_seed",
            "shuffle_seed",
            "structure_seed",
            "metrics_file_name",
            "has_val_loss",
            "has_model",
            "has_best_model",
            "canonical_rel_run_path",
            "rel_run_path",
            "issues",
        ],
    )
    write_csv_rows(
        collection_dir / "strict_duplicate_groups.csv",
        strict_rows_for_collection(run_records),
        fieldnames=[
            "strict_duplicate_group",
            "member_status",
            "canonical_rel_run_path",
            "rel_run_path",
            "task",
            "mode",
            "metrics_file_name",
        ],
    )
    write_csv_rows(
        collection_dir / "same_config_metrics_conflicts.csv",
        conflict_rows_for_collection(run_records),
        fieldnames=[
            "same_config_metrics_conflict_group",
            "dedup_status",
            "rel_run_path",
            "canonical_rel_run_path",
            "model_hash",
            "metrics_file_name",
        ],
    )
    write_csv_rows(
        collection_dir / "archive_files.csv",
        archive_rows_for_collection(archive_files),
        fieldnames=["archive_rel_path"],
    )


def write_top_level_readmes(by_section: Dict[str, List[Dict]]) -> None:
    top_lines = [
        "# Research Workspace",
        "",
        "This is the canonical, non-destructive research view for the repository.",
        "",
        "Use this tree when browsing data and analysis.",
        "Old scattered paths are kept only for compatibility with existing scripts.",
        "",
        "Top-level layout:",
        "- `data/raw/`: raw experiment outputs grouped by collection",
        "- `data/curated/`: curated symlink views and manifests",
        "- `analysis/project_level/`: mainline analysis packages and broad summaries",
        "- `analysis/studies/`: focused study packages",
        "- `metadata/`: machine-readable catalog of the organized view",
        "",
        "Rebuild command:",
        "- `python3 scripts/build_research_layout.py`",
        "",
        "Collections:",
    ]
    ordered_sections = ["data/raw", "data/curated", "analysis/project_level", "analysis/studies"]
    for section in ordered_sections:
        top_lines.append(f"- `{section}/`")
        for collection in by_section.get(section, []):
            top_lines.append(f"  - `{collection['slug']}`: {collection['description']}")
    write_markdown(OUT_ROOT / "README.md", top_lines)

    section_titles = {
        "data": "Data",
        "data/raw": "Raw Data",
        "data/curated": "Curated Data",
        "analysis": "Analysis",
        "analysis/project_level": "Project-Level Analysis",
        "analysis/studies": "Study Packages",
    }
    section_children = {
        "data": ["data/raw", "data/curated"],
        "analysis": ["analysis/project_level", "analysis/studies"],
    }
    for section, title in section_titles.items():
        lines = [f"# {title}", ""]
        if section in section_children:
            lines.extend(["Subsections:"])
            for child in section_children[section]:
                child_name = child.split("/")[-1]
                lines.append(f"- `{child_name}/`")
        else:
            lines.extend(["Collections:"])
            for collection in by_section.get(section, []):
                lines.append(f"- `{collection['slug']}`: {collection['description']}")
        write_markdown(OUT_ROOT / section / "README.md", lines)


def write_raw_overview(raw_collections: Sequence[Dict], records_by_collection: Dict[str, List[Dict]], all_records: Sequence[Dict]) -> None:
    lines = [
        "# Raw Data",
        "",
        "This section is now organized at the run level.",
        "",
        "How to read each raw collection:",
        "- `links/`: original source folders kept untouched",
        "- `canonical_runs/`: one canonical symlink per unique or deduplicated run",
        "- `duplicate_alias_runs/`: exact duplicate copies removed from the canonical view",
        "- `flagged_runs/`: runs that need extra caution because of nonstandard metrics files, missing checkpoints, or same-log conflicts",
        "- `run_inventory.csv`: full manifest for every run in that collection",
        "",
        "Global manifests in this folder:",
        "- `run_inventory_all.csv`",
        "- `strict_duplicate_groups_all.csv`",
        "- `same_config_metrics_conflicts_all.csv`",
        "",
        "Global counts:",
        f"- discovered_runs: {len(all_records)}",
        f"- canonical_runs: {sum(1 for record in all_records if record['dedup_status'] != 'duplicate_alias')}",
        f"- duplicate_alias_runs: {sum(1 for record in all_records if record['dedup_status'] == 'duplicate_alias')}",
        f"- strict_duplicate_groups: {len({record['strict_duplicate_group'] for record in all_records if record['strict_duplicate_group']})}",
        f"- same_config_metrics_conflict_groups: {len({record['same_config_metrics_conflict_group'] for record in all_records if record['same_config_metrics_conflict_group']})}",
        f"- nonstandard_metrics_runs: {sum(1 for record in all_records if record['nonstandard_metrics_name'])}",
        f"- missing_checkpoint_runs: {sum(1 for record in all_records if record['missing_checkpoint'])}",
        "",
        "Collections:",
    ]
    for collection in raw_collections:
        slug = collection["slug"]
        run_records = records_by_collection[slug]
        lines.append(
            f"- `{slug}`: runs={len(run_records)}, "
            f"canonical={sum(1 for record in run_records if record['dedup_status'] != 'duplicate_alias')}, "
            f"duplicate_aliases={sum(1 for record in run_records if record['dedup_status'] == 'duplicate_alias')}, "
            f"flags={sum(1 for record in run_records if record['nonstandard_metrics_name'] or record['missing_checkpoint'] or record['same_config_metrics_conflict_group'])}"
        )
    write_markdown(OUT_ROOT / "data" / "raw" / "README.md", lines)


def write_raw_global_manifests(all_records: Sequence[Dict]) -> None:
    raw_root = OUT_ROOT / "data" / "raw"
    write_csv_rows(
        raw_root / "run_inventory_all.csv",
        inventory_rows(all_records),
        fieldnames=[
            "collection_slug",
            "dedup_status",
            "strict_duplicate_group",
            "same_config_metrics_conflict_group",
            "task",
            "mode",
            "seed_dir",
            "run_name",
            "seed",
            "data_seed",
            "model_seed",
            "shuffle_seed",
            "structure_seed",
            "metrics_file_name",
            "has_val_loss",
            "has_model",
            "has_best_model",
            "canonical_rel_run_path",
            "rel_run_path",
            "issues",
        ],
    )
    write_csv_rows(
        raw_root / "strict_duplicate_groups_all.csv",
        [
            {
                "strict_duplicate_group": record["strict_duplicate_group"],
                "member_status": record["dedup_status"],
                "canonical_rel_run_path": record["canonical_rel_run_path"],
                "rel_run_path": record["rel_run_path"],
                "collection_slug": record["collection_slug"],
            }
            for record in sorted(all_records, key=lambda item: (str(item["strict_duplicate_group"]), str(item["rel_run_path"])))
            if record["strict_duplicate_group"]
        ],
        fieldnames=[
            "strict_duplicate_group",
            "member_status",
            "canonical_rel_run_path",
            "rel_run_path",
            "collection_slug",
        ],
    )
    write_csv_rows(
        raw_root / "same_config_metrics_conflicts_all.csv",
        [
            {
                "same_config_metrics_conflict_group": record["same_config_metrics_conflict_group"],
                "dedup_status": record["dedup_status"],
                "canonical_rel_run_path": record["canonical_rel_run_path"],
                "rel_run_path": record["rel_run_path"],
                "collection_slug": record["collection_slug"],
                "model_hash": record["model_hash"] or "MISSING",
            }
            for record in sorted(all_records, key=lambda item: (str(item["same_config_metrics_conflict_group"]), str(item["rel_run_path"])))
            if record["same_config_metrics_conflict_group"]
        ],
        fieldnames=[
            "same_config_metrics_conflict_group",
            "dedup_status",
            "canonical_rel_run_path",
            "rel_run_path",
            "collection_slug",
            "model_hash",
        ],
    )


def write_catalog(collection_rows: Sequence[Dict[str, str]]) -> None:
    ensure_dir(OUT_ROOT / "metadata")
    with (OUT_ROOT / "metadata" / "collection_catalog.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "section",
                "slug",
                "title",
                "kind",
                "organized_path",
                "source_paths",
                "builder_scripts",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerows(collection_rows)

    lines = [
        "# Compatibility Note",
        "",
        "This organized workspace is non-destructive.",
        "",
        "That means:",
        "- all canonical links under `research/` point back to the original repository locations",
        "- old scattered folders remain valid for existing scripts",
        "- new browsing work should start from `research/README.md`",
    ]
    write_markdown(OUT_ROOT / "metadata" / "compatibility_note.md", lines)


def main() -> None:
    remove_path(OUT_ROOT)
    ensure_dir(OUT_ROOT)

    by_section: Dict[str, List[Dict]] = {}
    catalog_rows: List[Dict[str, str]] = []
    raw_collections = [collection for collection in COLLECTIONS if collection["kind"] == "raw"]
    records_by_collection, all_raw_records, archive_files_by_collection = build_raw_run_records(raw_collections)
    annotate_duplicate_status(all_raw_records)
    for collection in COLLECTIONS:
        by_section.setdefault(collection["section"], []).append(collection)
        collection_dir = OUT_ROOT / collection["section"] / collection["slug"]
        link_dir = collection_dir / "links"
        ensure_dir(link_dir)
        source_paths = []
        for label, rel_path in collection["sources"]:
            source_path = REPO_ROOT / rel_path
            source_paths.append(source_path)
            make_symlink(source_path, link_dir / label)
        if collection["kind"] == "raw":
            run_records = records_by_collection[collection["slug"]]
            archive_files = archive_files_by_collection[collection["slug"]]
            write_raw_collection_assets(collection_dir, run_records, archive_files)
            write_collection_readme(collection_dir, collection, source_paths, run_records, archive_files)
        else:
            write_collection_readme(collection_dir, collection, source_paths)
        catalog_rows.append(
            {
                "section": collection["section"],
                "slug": collection["slug"],
                "title": collection["title"],
                "kind": collection["kind"],
                "organized_path": str(collection_dir.resolve()),
                "source_paths": " | ".join(str((REPO_ROOT / rel_path).resolve()) for _, rel_path in collection["sources"]),
                "builder_scripts": " | ".join(collection["builder_scripts"]),
                "description": collection["description"],
            }
        )

    write_top_level_readmes(by_section)
    write_raw_global_manifests(all_raw_records)
    write_raw_overview(raw_collections, records_by_collection, all_raw_records)
    write_catalog(catalog_rows)
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
