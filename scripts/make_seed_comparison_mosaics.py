import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create seed comparison mosaics for run plots")
    parser.add_argument("root", type=Path, help="Root directory such as results/hard/prune_grow_split")
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["active_neurons.png", "loss_log.png", "turnover.png"],
        help="Plot filenames to combine",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <root>/seed_comparison",
    )
    parser.add_argument("--cols", type=int, default=2, help="Number of mosaic columns")
    return parser.parse_args()


def latest_seed_runs(root: Path) -> List[Tuple[str, Path]]:
    selected: List[Tuple[str, Path]] = []
    seed_dirs = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("seed_"):
            continue
        suffix = path.name.split("_", 1)[1]
        if suffix.isdigit():
            seed_dirs.append(path)

    seed_dirs.sort(key=lambda path: int(path.name.split("_", 1)[1]))
    for seed_dir in seed_dirs:
        runs = sorted(path for path in seed_dir.iterdir() if path.is_dir())
        if runs:
            selected.append((seed_dir.name, runs[-1]))
    return selected


def make_mosaic(
    root: Path,
    plot_name: str,
    selected_runs: List[Tuple[str, Path]],
    out_dir: Path,
    cols: int,
) -> Optional[Path]:
    available = [(seed_name, run_dir, run_dir / plot_name) for seed_name, run_dir in selected_runs if (run_dir / plot_name).exists()]
    if not available:
        return None

    rows = (len(available) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5.5))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    elif rows == 1:
        axes_list = list(axes)
    elif cols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row_axes in axes for ax in row_axes]

    for ax in axes_list:
        ax.axis("off")

    for ax, (seed_name, run_dir, img_path) in zip(axes_list, available):
        image = mpimg.imread(img_path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{seed_name}  {run_dir.name}", fontsize=10)

    fig.suptitle(f"{root.as_posix()} / {plot_name}", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / f"{plot_name[:-4]}_all_seeds.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "seed_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_runs = latest_seed_runs(root)
    for plot_name in args.plots:
        out_path = make_mosaic(root, plot_name, selected_runs, out_dir, args.cols)
        if out_path is not None:
            print(out_path)


if __name__ == "__main__":
    main()
