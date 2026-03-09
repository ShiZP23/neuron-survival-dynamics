# Neuron Kill: Prune + Regrow Toy Study

## Project overview

This project is a small, research-oriented PyTorch experiment that explores whether a dense MLP can self-organize into a more efficient architecture when we periodically prune weak neurons and regrow new ones during training.

The focus is on interpretability, structural evolution, and reproducibility rather than production performance.

## Experiment goal

Investigate whether periodic neuron death and regrowth can reshape a dense network into a structure that preserves performance with fewer effective parameters.

## Task definitions (2D regression)

Inputs are 2D points in `[-1, 1]^2` and the target is a scalar.

- `simple`: `y = x1 + x2`
- `medium`: `y = sin(2πx1) + 0.3*x2^2 + 0.2*x1*x2`
- `hard`: `y = sin(4πx1)*cos(2πx2) + 0.2*x1^3`

## Modes

- `fixed`: standard MLP, no structure change
- `prune_only`: remove weak neurons every update interval
- `prune_grow_random`: prune then add random neurons
- `prune_grow_split`: prune then split strong neurons

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example commands

Single run:

```bash
python run.py --task medium --mode prune_grow_split --seed 0
```

Run all tasks and modes (also generates a summary plot):

```bash
python run.py --all
```

Aggregate existing results manually:

```bash
PYTHONPATH=src python -m neuron_kill.aggregate --results-dir results
```

## Outputs

Each run writes to `results/<task>/<mode>/seed_<seed>/<timestamp>/`:

- `config.json`: run configuration
- `metrics.csv`: per-epoch losses, parameter counts, and layer sizes
- `model.pt`: checkpoint dict (`state_dict`, final `hidden_sizes`, and metadata)
- `loss.png`: train/test curves
- `sizes.png`: layer sizes over time
- `params.png`: parameter count over time
- `surface.png`: target vs predicted surface

When running `--all`, a summary is also written to:

- `results/summary.csv`: final test loss and parameter count per run
- `results/summary.png`: final parameter count vs final test loss

## Engineering notes

- Structural edits preserve all surviving weights and biases.
- Pruning removes neuron rows from a layer and the corresponding columns in the next layer.
- Random growth appends new rows/columns with fresh initialization.
- Split growth copies a strong neuron into a new unit with small Gaussian noise.
- Optimizer state is reset after architecture changes because parameter shapes change.

## Experimental meaning

The structural update is a simplified stand-in for neuron death and regrowth. The core question is whether the network can maintain accuracy while reorganizing its hidden structure, and whether some modes lead to more compact solutions.

## Expected qualitative findings

These trends are common but not guaranteed:

- `fixed`: stable performance with no structural change
- `prune_only`: shrinking capacity can harm the hardest task but may mildly regularize easier tasks
- `prune_grow_random`: can recover capacity but often behaves like mild noise injection
- `prune_grow_split`: tends to preserve performance while still reshaping structure

## Limitations

- Small toy tasks; conclusions may not generalize to larger problems
- Uses a simple importance score and deterministic update interval
- Optimizer state reset can confound comparisons with continuous training

## Next-step ideas

- Track neuron importance over time and visualize survival dynamics
- Try different activations or sparsity penalties
- Compare with magnitude-based weight pruning baselines
- Add multiple seeds per mode and report aggregates
