# Neuron Survival Dynamics: Survival-Based Prune + Regrow

## Project overview

This project is a small, research-oriented PyTorch experiment that explores whether a dense MLP can self-organize into a more efficient architecture when we periodically prune weak neurons and regrow new ones during training.

The focus is on interpretability, structural evolution, and reproducibility rather than production performance.

The current default model is a `2 -> 64 -> 64 -> 1` ReLU MLP. Structural updates operate on hidden neurons only.

## Repository structure

- `run.py`: entrypoint script
- `src/neuron_survival_dynamics/`: package source
- `scripts/`: standalone visualization / post-processing utilities
- `requirements.txt`: Python dependencies
- `results/`: generated experiment outputs (git-ignored)

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

## Turnover-focused analysis

For `prune_grow_random` and `prune_grow_split`, an important analysis target is not only the surviving widths (`size_*`) but also neuron turnover at each structure update:

- how many neurons are pruned per update
- how many neurons are regrown per update
- how turnover varies by layer and by seed

## Physical Width vs Functional Activity

- `size_*` tracks physical neuron count (how many units exist in each layer).
- `active_*` tracks functional participation (how many units have mean absolute activation above a threshold on the validation set).
- In grow modes, active-neuron dynamics can be more informative than raw width because some newly added units may exist physically but remain weakly active.

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
- Val: used for ablation-based death decisions and active-neuron measurement (to avoid test leakage).
- Test: used only for evaluation/logging.

## Reproducibility controls

- Dataset generation uses `data_seed`.
- Model initialization uses `model_seed`.
- Training batch order uses `shuffle_seed`.
- Structural prune/grow randomness uses `structure_seed`.

By default these seeds are deterministically derived from `--seed`, but they can also be set independently for controlled ablations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example commands

Single run:

```bash
python3 run.py --task medium --mode prune_grow_split --seed 0
```

Run all tasks and modes (also generates a summary plot):

```bash
python3 run.py --all
```

Aggregate existing results manually:

```bash
PYTHONPATH=src python3 -m neuron_survival_dynamics.aggregate --results-dir results
```

Render a checkpoint as a network diagram:

```bash
python3 scripts/visualize_network.py results/hard/prune_only/seed_1/<timestamp>/model.pt
```

Build seed-wise mosaics from the latest run under each `seed_*` directory:

```bash
python3 scripts/make_seed_comparison_mosaics.py results/hard/prune_grow_split
```

## Key CLI arguments

- `--seed`: base seed (for convenience)
- `--data-seed`: seed for synthetic dataset generation
- `--model-seed`: seed for model weight initialization
- `--shuffle-seed`: seed for train loader shuffling
- `--structure-seed`: seed for prune/grow structural RNG
- `--update-interval`: epochs between structural updates (default `50`)
- `--ema-beta`: EMA decay factor
- `--ema-z-threshold`: z-score threshold for low-EMA candidates
- `--max-candidates-per-layer`: max ablation candidates per layer
- `--ablation-epsilon-ratio`: relative loss tolerance for neuron death
- `--active-threshold`: threshold used to mark neurons as active

By default, `data_seed/model_seed/shuffle_seed/structure_seed` are deterministically derived from `--seed`, but you can set them explicitly to decouple reproducibility experiments.

## Outputs

Each run writes to `results/<task>/<mode>/seed_<seed>/<timestamp>/`:

- `config.json`: run configuration
- `metrics.csv`: per-epoch losses, parameter counts, and layer stats
- `model.pt`: checkpoint dict (`state_dict`, final `hidden_sizes`, EMA state, metadata)
- `loss.png`: train/test curves
- `sizes.png`: layer sizes over time
- `active_neurons.png`: active neuron counts over time
- `params.png`: parameter count over time
- `turnover.png`: total and layer-wise neuron turnover over epochs
- `surface.png`: target vs predicted surface

Additional post-processing can produce:

- `network_diagram.png`: topology visualization rendered from a saved checkpoint
- `seed_comparison/*.png`: mosaics that compare the latest run for each seed within a task/mode folder

When running `--all`, a summary is also written to:

- `results/summary.csv`: final test loss and parameter count per run
- `results/summary.png`: final parameter count vs final test loss

### Metrics notes

- `is_update_epoch` is `1` only on structure update epochs.
- `total_pruned` and `total_grown` are explicit per-epoch turnover totals.
- `candidate_i`, `ema_mean_i`, `ema_std_i`, `size_i`, `pruned_i`, `grown_i`, and `active_i` are created dynamically for each hidden layer `i`.
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
