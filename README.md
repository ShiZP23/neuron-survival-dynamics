# Neuron Kill: Survival-Based Prune + Regrow

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

## Conceptual change: fixed fraction → survival-based pruning

Old approach (fixed ratio):
- Every update pruned a fixed fraction of neurons, regardless of functional impact.

New approach (survival-based):
- Neurons die only if they are persistently weak and functionally unimportant.
- The number of deaths is variable and can be zero.
- Different layers can prune different counts in the same update.

## Neuron death rule (current version)

1. **Current importance** per neuron:
   `importance_j = mean(abs(activation_j)) * norm(outgoing_weight_j)`
2. **EMA memory** of importance across updates:
   `ema_j = beta * ema_j_old + (1 - beta) * current_importance_j`
   EMA is initialized from the first measured importance at the first structure update.
3. **Candidate screening** by low EMA within each layer:
   `ema_j < mean(layer_ema) - z * std(layer_ema)`
   Only the `K` weakest candidates per layer are tested.
4. **Ablation test** on the validation set:
   A neuron dies only if the loss increase is tiny:
   `delta_loss < ablation_epsilon_ratio * baseline_val_loss`
5. **Split growth parent selection** uses current importance among surviving neurons.

## Data splits

- Train: used for parameter updates.
- Val: used for ablation-based death decisions (to avoid test leakage).
- Test: used only for evaluation/logging.

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

## New CLI arguments

- `--ema-beta`: EMA decay factor
- `--ema-z-threshold`: z-score threshold for low-EMA candidates
- `--max-candidates-per-layer`: max ablation candidates per layer
- `--ablation-epsilon-ratio`: relative loss tolerance for neuron death

## Outputs

Each run writes to `results/<task>/<mode>/seed_<seed>/<timestamp>/`:

- `config.json`: run configuration
- `metrics.csv`: per-epoch losses, parameter counts, and layer stats
- `model.pt`: checkpoint dict (`state_dict`, final `hidden_sizes`, EMA state, metadata)
- `loss.png`: train/test curves
- `sizes.png`: layer sizes over time
- `params.png`: parameter count over time
- `surface.png`: target vs predicted surface

When running `--all`, a summary is also written to:

- `results/summary.csv`: final test loss and parameter count per run
- `results/summary.png`: final parameter count vs final test loss

### Metrics notes

- `is_update_epoch` is `1` only on structure update epochs.
- `candidate_*`, `ema_mean_*`, and `ema_std_*` are meaningful only on update epochs.
- On non-update epochs, those fields are recorded as zeros for simplicity.

## Engineering notes

- Structural edits preserve all surviving weights and biases.
- Pruning removes neuron rows from a layer and the corresponding columns in the next layer.
- Random growth appends new rows/columns with fresh initialization.
- Split growth copies a strong neuron into a new unit with small Gaussian noise.
- Optimizer state is reset only if the architecture actually changes.

## Experimental meaning

The structural update is a simplified stand-in for neuron death and regrowth. The key change is that neurons now die only when both long-term weakness (EMA) and functional irrelevance (ablation) agree. This makes structural evolution more data-dependent and less forced.

## Expected qualitative findings

- Some updates may prune nothing.
- Some layers may prune more than others.
- Structure evolution may become less symmetric than before.
- Performance may be more stable than fixed-fraction pruning.

## Limitations

- Ablation testing is more expensive than simple pruning rules.
- This is still a toy experiment on small synthetic tasks.
- EMA alignment after structural edits depends on careful bookkeeping.

## Next-step ideas

- Track neuron survival time and visualize life/death dynamics.
- Compare with magnitude-based weight pruning baselines.
- Try different activations or sparsity penalties.
- Run multiple seeds and report aggregate statistics.
