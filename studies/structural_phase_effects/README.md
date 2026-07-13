# Structural Phase Effects

## Question

Hypothesis:

`Under the same prune-grow rule, training can fall into a small number of discrete structural phases, and those phases are sensitive to random realization while carrying different endpoint behavior.`

This study isolates that branch of the project.

## Structural Phases

Current taxonomy:

- `prune0+2`
- `prune0_only`
- `no_prune`

Definitions:

- `prune0+2`: pruning happens in hidden layers 0 and 2
- `prune0_only`: pruning happens only in hidden layer 0
- `no_prune`: no hidden layer ever prunes

## Scope

Current data source:

- `former results/3.12results，128/hard/prune_grow_split`

What this legacy sweep already contains:

- canonical `base seed` sweep
- controlled `model_seed` sweep with data/shuffle/structure fixed
- controlled `data_seed` sweep with model/shuffle/structure fixed

What it does not yet contain:

- controlled `structure_seed` sweep
- validation-selected checkpoints
- non-hard tasks for this exact phase analysis

## Questions To Answer

- Are the structural phases strongly associated with base seed, initialization, or data realization?
- Do the phases differ systematically in final performance?
- Do they differ in active-neuron evolution, training speed, and generalization gap?

## Endpoints

Primary endpoints for this branch:

- `final_test_loss`
- `final_minus_best`
- `final_test_train_gap`

Secondary endpoints:

- `best_test_loss`
- `best_after_commit_lag`
- `post_best_drift_rate`
- first epoch reaching fixed test-loss thresholds
- `best_epoch`
- active-neuron trajectories by layer
- active-share drift after best by layer

Explicit caveat:

- because the legacy sweep has no validation loss, `best_test_loss` here is diagnostic only

## Outputs

Generated artifacts:

- `results/studies/structural_phase_effects_20260314/`

Expected primary figures:

- random-factor phase assignment map
- phase commit vs best epoch
- best-after-commit lag by phase
- post-best drift rate by phase
- best-to-final degradation by phase
- degradation vs layer-2 share drift
- active drift from best to final for layer 1 and layer 2
- active-share drift after best
- active share dynamics by layer
- phase-stratified endpoint comparison
- phase-stratified speed comparison
- phase-stratified active-neuron dynamics

Expected tables:

- per-run phase rows
- per-phase summary
- random-factor phase-count summary
- `prune0_only` vs `prune0+2` pairwise statistics

## Planned Follow-Up

After the legacy readout, the next controlled sweep should move to the current validation-selected protocol and vary one randomness source at a time:

Initialization-only:

```bash
python3 run.py --task hard --mode prune_grow_split --seed 0 --data-seed 11 --model-seed <k> --shuffle-seed 37 --structure-seed 53
```

Data-only:

```bash
python3 run.py --task hard --mode prune_grow_split --seed 0 --data-seed <k> --model-seed 23 --shuffle-seed 37 --structure-seed 53
```

Structure-only:

```bash
python3 run.py --task hard --mode prune_grow_split --seed 0 --data-seed 11 --model-seed 23 --shuffle-seed 37 --structure-seed <k>
```

That follow-up is the clean way to connect this branch to the mainline paper protocol.
