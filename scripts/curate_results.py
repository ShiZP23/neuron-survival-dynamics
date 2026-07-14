import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-destructive curated views of experiment runs")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["former results/3.12results，128", "results"],
        help="Experiment roots to scan",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("curated_results"),
        help="Directory where curated symlink views and manifests will be written",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def iter_runs(root: Path) -> Iterable[Path]:
    for config_path in root.rglob("config.json"):
        run_dir = config_path.parent
        if (run_dir / "metrics.csv").exists():
            yield run_dir


def has_val_loss(metrics_path: Path) -> bool:
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
    return first is not None and "val_loss" in first


def expected_seeds(base_seed: int) -> Dict[str, int]:
    return {
        "data_seed": base_seed + 11,
        "model_seed": base_seed + 23,
        "shuffle_seed": base_seed + 37,
        "structure_seed": base_seed + 53,
    }


def classify_run(run_dir: Path, cfg: Dict) -> Dict[str, str]:
    seed = int(cfg["seed"])
    expected = expected_seeds(seed)
    varying = [
        key
        for key, expected_value in expected.items()
        if int(cfg.get(key, expected_value)) != expected_value
    ]

    if not varying:
        group = "canonical"
        alias = f"seed_{seed}"
    elif varying == ["data_seed"]:
        group = "data_seed_sweep"
        alias = f"seed_{seed}__data_seed_{cfg['data_seed']}"
    elif varying == ["model_seed"]:
        group = "model_seed_sweep"
        alias = f"seed_{seed}__model_seed_{cfg['model_seed']}"
    elif varying == ["shuffle_seed"]:
        group = "shuffle_seed_sweep"
        alias = f"seed_{seed}__shuffle_seed_{cfg['shuffle_seed']}"
    elif varying == ["structure_seed"]:
        group = "structure_seed_sweep"
        alias = f"seed_{seed}__structure_seed_{cfg['structure_seed']}"
    else:
        suffix = "__".join(f"{key}_{cfg[key]}" for key in varying)
        group = "mixed_control"
        alias = f"seed_{seed}__{suffix}"

    if run_dir.name.startswith("data"):
        group = "data_seed_sweep"
    elif run_dir.name.startswith("seed") and run_dir.name != f"seed{seed}":
        group = "model_seed_sweep"

    return {"group": group, "alias": alias}


def safe_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.resolve() == target.resolve():
                return
        except FileNotFoundError:
            pass
        link_path.unlink()
    os.symlink(target.resolve(), link_path)


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def curate_root(root: Path, out_root: Path) -> List[Dict[str, str]]:
    manifest_rows: List[Dict[str, str]] = []
    root_label = root.name.replace(" ", "_")
    curated_root = out_root / root_label
    for run_dir in sorted(iter_runs(root)):
        cfg = load_json(run_dir / "config.json")
        parts = run_dir.relative_to(root).parts
        if len(parts) < 4:
            continue
        task, mode, seed_dir = parts[0], parts[1], parts[2]
        classification = classify_run(run_dir, cfg)
        selection_protocol = (
            "val_checkpoint_available"
            if has_val_loss(run_dir / "metrics.csv")
            else "legacy_final_test_only"
        )
        alias = classification["alias"]
        link_path = curated_root / task / mode / classification["group"] / alias
        safe_symlink(run_dir, link_path)
        manifest_rows.append(
            {
                "source_root": str(root),
                "task": task,
                "mode": mode,
                "seed_dir": seed_dir,
                "run_name": run_dir.name,
                "group": classification["group"],
                "alias": alias,
                "selection_protocol": selection_protocol,
                "seed": str(cfg.get("seed")),
                "data_seed": str(cfg.get("data_seed")),
                "model_seed": str(cfg.get("model_seed")),
                "shuffle_seed": str(cfg.get("shuffle_seed")),
                "structure_seed": str(cfg.get("structure_seed")),
                "epochs": str(cfg.get("epochs")),
                "path": str(run_dir.resolve()),
                "curated_link": str(link_path.resolve()),
            }
        )
    write_manifest(curated_root / "manifest.csv", manifest_rows)
    return manifest_rows


def main() -> None:
    args = parse_args()
    all_rows: List[Dict[str, str]] = []
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            continue
        all_rows.extend(curate_root(root, args.out_root))
    write_manifest(args.out_root / "manifest_all.csv", all_rows)
    print(args.out_root)


if __name__ == "__main__":
    main()
