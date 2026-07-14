import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Sequence

from neuron_survival_dynamics.init_only.data import IMAGE_DATASETS, create_data_loaders, load_image_dataset
from neuron_survival_dynamics.init_only.prunable_models import PRUNABLE_MODEL_NAMES
from neuron_survival_dynamics.init_only.structured_train import train_structured_classifier_run
from neuron_survival_dynamics.utils import get_device, save_json


RUN_LABELS = ["fixed", "prune_only", "fixed_shadow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired init-only fixed / prune_only / fixed+shadow pilots on image datasets."
    )
    parser.add_argument("--dataset", choices=IMAGE_DATASETS, default="fashion_mnist")
    parser.add_argument("--model", choices=PRUNABLE_MODEL_NAMES, default="lenet300100")
    parser.add_argument("--run-labels", nargs="+", choices=RUN_LABELS, default=RUN_LABELS)
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--update-interval", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-order-seed", type=int, default=0)
    parser.add_argument("--runtime-seed", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--min-neurons", type=int, default=16)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--ema-z-threshold", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-layer", type=int, default=8)
    parser.add_argument("--ablation-epsilon-ratio", type=float, default=0.01)
    parser.add_argument("--active-threshold", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-root", type=str, default="~/.cache/torchvision")
    parser.add_argument("--results-dir", type=str, default="results/init_only_lth_20260401")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    return parser.parse_args()


def _make_run_root(results_dir: str, dataset: str, model: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(results_dir, dataset, model, "structured_init_only", stamp)
    os.makedirs(run_root, exist_ok=True)
    return run_root


def _write_aggregate_summary(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_mode(run_label: str) -> Dict[str, object]:
    if run_label == "fixed":
        return {"mode": "fixed", "shadow_prune": False}
    if run_label == "prune_only":
        return {"mode": "prune_only", "shadow_prune": False}
    if run_label == "fixed_shadow":
        return {"mode": "fixed", "shadow_prune": True}
    raise ValueError(f"Unsupported run_label: {run_label}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    bundle = load_image_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        val_size=args.val_size,
        split_seed=args.split_seed,
        download=not args.no_download,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        subset_seed=args.subset_seed,
    )

    run_root = _make_run_root(args.results_dir, args.dataset, args.model)
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["train_size"] = len(bundle.train_dataset)
    config["val_size"] = len(bundle.val_dataset)
    config["test_size"] = len(bundle.test_dataset)
    config["loader_reinitialized_per_seed"] = True
    config["loader_reinitialized_per_mode"] = True
    save_json(os.path.join(run_root, "run_config.json"), config)

    summary_rows: List[Dict[str, object]] = []
    for init_seed in args.init_seeds:
        for run_label in args.run_labels:
            run_mode = _resolve_mode(run_label)
            train_loader, val_loader, test_loader = create_data_loaders(
                bundle=bundle,
                batch_size=args.batch_size,
                train_order_seed=args.train_order_seed,
                num_workers=args.num_workers,
            )
            run_dir = os.path.join(run_root, run_label, f"init_seed_{init_seed}")
            summary = train_structured_classifier_run(
                dataset_name=args.dataset,
                model_name=args.model,
                mode=run_mode["mode"],
                init_seed=init_seed,
                runtime_seed=args.runtime_seed,
                split_seed=args.split_seed,
                train_order_seed=args.train_order_seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                input_shape=(bundle.channels, *bundle.image_size),
                num_classes=bundle.num_classes,
                run_dir=run_dir,
                device=device,
                epochs=args.epochs,
                update_interval=args.update_interval,
                lr=args.lr,
                weight_decay=args.weight_decay,
                min_neurons=args.min_neurons,
                ema_beta=args.ema_beta,
                ema_z_threshold=args.ema_z_threshold,
                max_candidates_per_layer=args.max_candidates_per_layer,
                ablation_epsilon_ratio=args.ablation_epsilon_ratio,
                active_threshold=args.active_threshold,
                shadow_prune=bool(run_mode["shadow_prune"]),
                deterministic_algorithms=args.deterministic_algorithms,
                save_checkpoints=not args.no_save_checkpoints,
            )
            row = dict(summary.__dict__)
            row["run_label"] = run_label
            row["shadow_prune"] = int(bool(run_mode["shadow_prune"]))
            summary_rows.append(row)

    _write_aggregate_summary(os.path.join(run_root, "aggregate_summary.csv"), summary_rows)


if __name__ == "__main__":
    main()
