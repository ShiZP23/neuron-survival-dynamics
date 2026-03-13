import argparse
from pathlib import Path
from typing import List, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neuron_survival_dynamics.model import MLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an MLP checkpoint as a layer diagram")
    parser.add_argument("checkpoint", type=Path, help="Path to model.pt checkpoint")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path; defaults to checkpoint directory / network_diagram.png",
    )
    parser.add_argument(
        "--edge-threshold-ratio",
        type=float,
        default=0.15,
        help="Hide edges with |w| below this fraction of the global max |w|",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=0.25,
        help="Minimum linewidth for visible edges",
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=2.2,
        help="Maximum linewidth for visible edges",
    )
    parser.add_argument(
        "--node-size",
        type=float,
        default=18.0,
        help="Node marker size",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> Tuple[MLP, List[torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    hidden_sizes = checkpoint["hidden_sizes"]
    input_dim = state_dict["layers.0.weight"].shape[1]
    output_dim = state_dict["out.weight"].shape[0]

    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim)
    model.load_state_dict(state_dict)
    model.eval()

    weights = [layer.weight.detach().cpu() for layer in model.layers]
    weights.append(model.out.weight.detach().cpu())
    return model, weights


def layer_sizes_from_weights(weights: Sequence[torch.Tensor]) -> List[int]:
    sizes = [weights[0].shape[1]]
    for weight in weights:
        sizes.append(weight.shape[0])
    return sizes


def layer_labels(layer_sizes: Sequence[int]) -> List[List[str]]:
    labels: List[List[str]] = []
    input_dim = layer_sizes[0]
    labels.append([f"x{i + 1}" for i in range(input_dim)])

    for hidden_size in layer_sizes[1:-1]:
        labels.append([""] * hidden_size)

    output_dim = layer_sizes[-1]
    if output_dim == 1:
        labels.append(["y"])
    else:
        labels.append([f"y{i + 1}" for i in range(output_dim)])
    return labels


def node_positions(layer_sizes: Sequence[int]) -> List[List[Tuple[float, float]]]:
    n_layers = len(layer_sizes)
    x_margin = 0.08
    y_margin = 0.08
    positions: List[List[Tuple[float, float]]] = []

    for layer_idx, size in enumerate(layer_sizes):
        y = 1.0 - y_margin - layer_idx * (1.0 - 2.0 * y_margin) / max(n_layers - 1, 1)
        if size == 1:
            xs = [0.5]
        else:
            xs = [
                x_margin + idx * (1.0 - 2.0 * x_margin) / (size - 1)
                for idx in range(size)
            ]
        positions.append([(x, y) for x in xs])
    return positions


def draw_edges(
    ax: plt.Axes,
    weights: Sequence[torch.Tensor],
    positions: Sequence[Sequence[Tuple[float, float]]],
    edge_threshold_ratio: float,
    min_width: float,
    max_width: float,
) -> None:
    global_max = max(float(weight.abs().max()) for weight in weights)
    threshold = edge_threshold_ratio * global_max

    positive_segments = []
    positive_widths = []
    negative_segments = []
    negative_widths = []

    for layer_idx, weight in enumerate(weights):
        src_positions = positions[layer_idx]
        dst_positions = positions[layer_idx + 1]
        src_count = weight.shape[1]
        dst_count = weight.shape[0]

        for dst_idx in range(dst_count):
            for src_idx in range(src_count):
                value = float(weight[dst_idx, src_idx])
                abs_value = abs(value)
                if abs_value < threshold:
                    continue

                width = min_width + (max_width - min_width) * (abs_value / global_max)
                segment = [src_positions[src_idx], dst_positions[dst_idx]]
                if value >= 0.0:
                    positive_segments.append(segment)
                    positive_widths.append(width)
                else:
                    negative_segments.append(segment)
                    negative_widths.append(width)

    if positive_segments:
        ax.add_collection(
            LineCollection(
                positive_segments,
                colors="#c73d3d",
                linewidths=positive_widths,
                alpha=0.5,
                zorder=1,
            )
        )
    if negative_segments:
        ax.add_collection(
            LineCollection(
                negative_segments,
                colors="#3c66c7",
                linewidths=negative_widths,
                alpha=0.5,
                zorder=1,
            )
        )


def draw_nodes(
    ax: plt.Axes,
    positions: Sequence[Sequence[Tuple[float, float]]],
    labels: Sequence[Sequence[str]],
    node_size: float,
) -> None:
    for layer_idx, layer_positions in enumerate(positions):
        xs = [x for x, _ in layer_positions]
        ys = [y for _, y in layer_positions]
        ax.scatter(xs, ys, s=node_size, color="black", zorder=2)

        for (x, y), label in zip(layer_positions, labels[layer_idx]):
            if not label:
                continue
            dx = -0.03 if layer_idx == 0 else 0.015
            ha = "right" if layer_idx == 0 else "left"
            ax.text(x + dx, y, label, fontsize=10, ha=ha, va="center")


def draw_layer_titles(ax: plt.Axes, positions: Sequence[Sequence[Tuple[float, float]]]) -> None:
    titles = ["Input"] + [f"Hidden {idx}" for idx in range(1, len(positions) - 1)] + ["Output"]
    for title, layer_positions in zip(titles, positions):
        center_x = sum(x for x, _ in layer_positions) / len(layer_positions)
        y = layer_positions[0][1] + 0.045
        ax.text(center_x, y, title, fontsize=11, ha="center", va="bottom")


def plot_network(
    checkpoint_path: Path,
    out_path: Path,
    edge_threshold_ratio: float,
    min_width: float,
    max_width: float,
    node_size: float,
) -> None:
    _, weights = load_checkpoint(checkpoint_path)
    layer_sizes = layer_sizes_from_weights(weights)
    labels = layer_labels(layer_sizes)
    positions = node_positions(layer_sizes)

    fig_w = max(8.0, min(22.0, 3.0 + max(layer_sizes) * 0.12))
    fig_h = max(6.0, 2.2 + len(layer_sizes) * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    draw_edges(ax, weights, positions, edge_threshold_ratio, min_width, max_width)
    draw_nodes(ax, positions, labels, node_size)
    draw_layer_titles(ax, positions)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.04)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    out_path = args.out or (checkpoint_path.parent / "network_diagram.png")
    plot_network(
        checkpoint_path=checkpoint_path,
        out_path=out_path,
        edge_threshold_ratio=args.edge_threshold_ratio,
        min_width=args.min_width,
        max_width=args.max_width,
        node_size=args.node_size,
    )
    print(out_path)


if __name__ == "__main__":
    main()
