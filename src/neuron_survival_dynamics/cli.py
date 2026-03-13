import argparse
import os

from neuron_survival_dynamics.data import TASKS
from neuron_survival_dynamics.aggregate import summarize_results, write_summary_csv, plot_summary
from neuron_survival_dynamics.plots import (
    plot_active_neurons,
    plot_losses,
    plot_param_count,
    plot_sizes,
    plot_surface,
    plot_turnover,
)
from neuron_survival_dynamics.train import train_one_run
from neuron_survival_dynamics.utils import get_device, make_run_dir, save_json, set_seed

MODES = ["fixed", "prune_only", "prune_grow_random", "prune_grow_split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neuron prune/regrow toy experiment")
    parser.add_argument("--task", choices=TASKS, default="medium")
    parser.add_argument("--mode", choices=MODES, default="fixed")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--structure-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--all", action="store_true", help="run all tasks and modes")
    return parser.parse_args()


def run_single(args: argparse.Namespace, task: str, mode: str, seed: int) -> None:
    set_seed(seed)
    device = get_device(args.device)
    run_dir = make_run_dir(args.results_dir, task, mode, seed)
    data_seed = args.data_seed if args.data_seed is not None else seed + 11
    model_seed = args.model_seed if args.model_seed is not None else seed + 23
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else seed + 37
    structure_seed = args.structure_seed if args.structure_seed is not None else seed + 53

    config = {
        "task": task,
        "mode": mode,
        "epochs": args.epochs,
        "update_interval": args.update_interval,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_neurons": args.min_neurons,
        "ema_beta": args.ema_beta,
        "ema_z_threshold": args.ema_z_threshold,
        "max_candidates_per_layer": args.max_candidates_per_layer,
        "ablation_epsilon_ratio": args.ablation_epsilon_ratio,
        "active_threshold": args.active_threshold,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "noise": args.noise,
        "seed": seed,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "shuffle_seed": shuffle_seed,
        "structure_seed": structure_seed,
        "device": str(device),
    }
    save_json(os.path.join(run_dir, "config.json"), config)

    model, history = train_one_run(
        task=task,
        mode=mode,
        run_dir=run_dir,
        seed=seed,
        device=device,
        epochs=args.epochs,
        update_interval=args.update_interval,
        batch_size=args.batch_size,
        lr=args.lr,
        min_neurons=args.min_neurons,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        noise=args.noise,
        data_seed=data_seed,
        model_seed=model_seed,
        shuffle_seed=shuffle_seed,
        structure_seed=structure_seed,
        ema_beta=args.ema_beta,
        ema_z_threshold=args.ema_z_threshold,
        max_candidates_per_layer=args.max_candidates_per_layer,
        ablation_epsilon_ratio=args.ablation_epsilon_ratio,
        active_threshold=args.active_threshold,
    )

    plot_losses(history, os.path.join(run_dir, "loss.png"))
    plot_sizes(history, os.path.join(run_dir, "sizes.png"))
    plot_active_neurons(history, os.path.join(run_dir, "active_neurons.png"))
    plot_param_count(history, os.path.join(run_dir, "params.png"))
    plot_turnover(history, os.path.join(run_dir, "turnover.png"))
    plot_surface(model, task, os.path.join(run_dir, "surface.png"), device)


def main() -> None:
    args = parse_args()

    if args.all:
        for idx, task in enumerate(TASKS):
            for jdx, mode in enumerate(MODES):
                run_single(args, task, mode, args.seed)
        records = summarize_results(args.results_dir)
        if records:
            write_summary_csv(records, os.path.join(args.results_dir, "summary.csv"))
            plot_summary(records, os.path.join(args.results_dir, "summary.png"))
        return

    run_single(args, args.task, args.mode, args.seed)


if __name__ == "__main__":
    main()
