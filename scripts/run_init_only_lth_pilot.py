import argparse
import csv
import os
from datetime import datetime
from typing import List

from neuron_survival_dynamics.init_only.data import IMAGE_DATASETS, create_data_loaders, load_image_dataset
from neuron_survival_dynamics.init_only.models import MODEL_DEFAULTS, MODEL_NAMES
from neuron_survival_dynamics.init_only.train import train_dense_classifier_run
from neuron_survival_dynamics.utils import get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-0 init-only dense pilot on Lottery-Ticket-style image datasets."
    )
    parser.add_argument("--dataset", choices=IMAGE_DATASETS, default="mnist")
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-order-seed", type=int, default=0)
    parser.add_argument("--runtime-seed", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
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
    run_root = os.path.join(results_dir, dataset, model, "dense_init_only", stamp)
    os.makedirs(run_root, exist_ok=True)
    return run_root


def _write_aggregate_summary(run_root: str, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(os.path.join(run_root, "aggregate_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model_name = args.model or MODEL_DEFAULTS[args.dataset]
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
    run_root = _make_run_root(args.results_dir, args.dataset, model_name)
    config = vars(args).copy()
    config["model"] = model_name
    config["device_resolved"] = str(device)
    config["train_size"] = len(bundle.train_dataset)
    config["val_size"] = len(bundle.val_dataset)
    config["test_size"] = len(bundle.test_dataset)
    config["loader_reinitialized_per_seed"] = True
    save_json(os.path.join(run_root, "run_config.json"), config)

    summary_rows: List[dict] = []
    for init_seed in args.init_seeds:
        train_loader, val_loader, test_loader = create_data_loaders(
            bundle=bundle,
            batch_size=args.batch_size,
            train_order_seed=args.train_order_seed,
            num_workers=args.num_workers,
        )
        run_dir = os.path.join(run_root, f"init_seed_{init_seed}")
        summary = train_dense_classifier_run(
            dataset_name=args.dataset,
            model_name=model_name,
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
            lr=args.lr,
            weight_decay=args.weight_decay,
            deterministic_algorithms=args.deterministic_algorithms,
            save_checkpoints=not args.no_save_checkpoints,
        )
        summary_rows.append(summary.__dict__)

    _write_aggregate_summary(run_root, summary_rows)


if __name__ == "__main__":
    main()
