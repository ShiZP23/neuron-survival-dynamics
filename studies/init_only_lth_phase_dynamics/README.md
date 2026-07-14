# Initialization-Only LTH Phase Dynamics

## Question

If we hold the dataset, train/val/test split, batch order, optimizer, and full training protocol fixed, and change only the model initialization seed, does initialization alone induce:

- stable ticket-performance differences
- stable structural or functional phases
- latent dense-run phase differentiation
- shadow-prune gate differences

## Why this study exists

This study extends the current repository in two directions at once:

1. it moves from toy regression tasks toward Lottery-Ticket-style benchmark datasets
2. it tightens causal control by fixing every randomness source except model initialization

The central goal is to separate:

- `initialization-only effects`

from

- `initialization × data/order interaction effects`

## Phase structure

Planned execution order:

1. `Phase 0`: deterministic dense baseline on MNIST / Fashion-MNIST / CIFAR-10
2. `Phase 1`: small init-only pilot sweep on CIFAR-10
3. `Phase 2`: main 100–200 seed dense + prune + shadow sweep
4. `Phase 3`: interventions such as rewind, freeze, and mask/init/sign swap

## Current implementation status

Implemented in this repository now:

- deterministic image-classification data pipeline with fixed split and fixed batch order
- init-only dense-run training entrypoint
- result layout for repeated initialization sweeps
- per-run dense sweep report builder
- cross-dataset Phase-0 overview builder

Completed pilots:

- corrected `MNIST`, `LeNet-300-100`, 10 init seeds, 8 epochs
- corrected `Fashion-MNIST`, `LeNet-300-100`, 10 init seeds, 8 epochs
- `Fashion-MNIST`, `LeNet-300-100`, 5 init seeds, paired `fixed` / `prune_only` / `fixed+shadow`
- `CIFAR-10`, `smallconv`, 3 init seeds, subset-based dense pilot
- `CIFAR-10`, `smallconv`, 3 init seeds, paired `fixed` / `prune_only` / `fixed+shadow`
- `CIFAR-10`, `smallconv`, 8 init seeds, multi-update paired `fixed` / `prune_only` / `fixed+shadow`

Current empirical takeaway:

- the protocol is operational and deterministic under fixed data/order
- `MNIST` shows small but non-zero init-only spread
- `Fashion-MNIST` shows a slightly larger selected-point spread and supports paired structured pruning
- the first paired structured pilot shows full shadow/prune signature alignment, but not yet phase diversity
- the first harder-dataset `CIFAR-10` dense pilot already shows larger init-only spread than the corrected `MNIST` / `Fashion-MNIST` pilots
- the first `CIFAR-10 structured` pilot now shows both full shadow/prune signature alignment and the emergence of more than one coarse prune signature
- the expanded `CIFAR-10 structured` sweep now shows:
  - 3 coarse phases
  - 6 fine-grained update paths
  - mostly preserved shadow alignment, but not perfect alignment
  - one mismatch seed where dense shadow over-predicts a late layer event, suggesting pruning itself can feed back on later screening
- the 16-seed `CIFAR-10 structured` cohort now shows:
  - 5 coarse phases
  - 11 fine-grained update paths
  - 3 mismatch seeds out of 16
  - two qualitatively different boundary types: `0+2+3 -> 0+1+2+3` and `0 -> 0+2+3`
- the 24-seed `CIFAR-10 structured` cohort now shows:
  - the same 5 coarse phases and 11 fine-grained update paths remain stable
  - shadow coarse/fine match rates remain high at `0.8333 / 0.7917`
  - mismatch seeds expand to `5 / 24`, adding one timing-only seed (`17`) and one new boundary family seed (`18`)
- the 32-seed `CIFAR-10 structured` cohort now shows:
  - phase diversity expands again to `7` coarse phases and `15` fine paths
  - shadow coarse/fine match remains high at `0.8438 / 0.7812`
  - mismatch seeds expand to `7 / 32`, adding a new boundary family seed (`30`) and a second timing-only family seed (`31`)
- the 40-seed `CIFAR-10 structured` cohort now shows:
  - phase diversity expands further to `8` coarse phases and `18` fine paths
  - shadow coarse/fine match remains high at `0.8750 / 0.8250`
  - mismatch seeds stay fixed at `7 / 40`, meaning the newest 8-seed extension adds phase diversity but no new boundary seeds
- the new mechanism analyses now show:
  - `epoch 2` shadow signature already predicts final coarse/fine phase with about `0.875` modal accuracy
  - one important prefix (`0+1+2` at epochs 2 and 4) remains ambiguous until epoch 6, revealing a late-commit branch
  - the only mismatch seed diverges only at the final update, strengthening the hypothesis that real pruning can alter later screening rather than shadow being wrong from the start
  - targeted intervention on the mismatch seed shows that pruning only once at `epoch 2` is already enough to remove the later `layer 1` shadow event
  - within the ambiguous `0+1+2 -> 0+1+2` branch, the final split is controlled by whether `layer 3` threshold turns positive at epoch 6
  - layer-targeted interventions on the mismatch seed now show that a single real prune event in either early active layer (`layer 0` or `layer 2`) is sufficient to suppress the later `layer 1` shadow event
  - negative-control interventions restricted to inactive early layers (`layer 1` or `layer 3`) produce zero actual pruning and preserve the dense baseline late mismatch unchanged
  - cross-seed layer-feedback overview now shows that this rewrite behavior is boundary-specific: stable positive and stable negative branches remain invariant under all tested single-layer interventions, while the boundary seed rewrites only when an early active layer actually gets pruned
  - the newly analyzed `seed 8` shows that the original `seed 3` mechanism is reproducible: for the same `0+2+3 -> 0+1+2+3` boundary type, one real `epoch 2` prune already removes the late `layer 1` event, and any active early layer (`0`, `2`, or `3`) is individually sufficient
  - the newly discovered `seed 9` boundary confirms this generalizes beyond the original mismatch type: a single real `epoch 2` prune of `layer 0` is enough to suppress the dense run's late `0+2+3` shadow branch
  - `seed 17` establishes a second kind of feedback effect: real pruning can advance commit timing without changing the final coarse phase, moving the `layer 3` event from `epoch 6` up to `epoch 4`
  - `seed 18` adds a third boundary family: `0+1+2 -> 0+1+2+3`; one real prune at `epoch 2` already removes the late `layer 3` event
  - layer-targeted interventions on `seed 18` show that any early active layer (`0`, `1`, or `2`) is individually sufficient to suppress that late `layer 3` branch, while inactive `layer 3` remains a clean negative control
  - `seed 30` adds a fourth boundary family: `0+1 -> 0+1+2`; its late `layer 2` event survives a single real prune at `epoch 2` and disappears only after pruning continues through `epoch 4`
  - layer-targeted interventions on `seed 30` show that no tested `epoch 2` single-layer prefix rewrites this family, implying a second-update feedback route rather than a one-shot early trigger
  - a new `epoch 4` targeted follow-up on `seed 30` sharpens that claim: the late `layer 2` event is shut off by real pruning in `layer 0`, `layer 1`, or `0+1` at `epoch 4`, but not by `layer 2` or `layer 3`; the causal route is therefore second-update and upstream-layer driven
  - `seed 31` establishes a second timing-only family: `0 -> 0+3`; one real prune at `epoch 2` advances the `layer 3` commit to `epoch 4`, but can also transiently induce an extra late `layer 2`
  - boundary-family analysis now separates at least four mechanism families: `0+2+3 -> 0+1+2+3`, `0 -> 0+2+3`, `0+1+2 -> 0+1+2+3`, and `0+1 -> 0+1+2`
  - the new 40-seed prefix-branch analysis shows that most ambiguous prefixes do not end in boundary rewrites: within `0+1 -> 0+1`, five seeds are stable late-commit and only one is boundary; within `0+1+2 -> 0+1+2`, three are stable late-commit and only one is boundary
  - this separates `late-commit` from `feedback rewrite`: late-commit is often a stable latent phase, while feedback rewrite remains a minority boundary phenomenon
  - the new threshold-persistence cohort analysis pushes this one step further: late-layer events now split cleanly into three mechanism types
    - `stable_late_commit`: modal target-layer sign trajectory is `--+ / --+`
    - `boundary_absent_in_actual`: modal target-layer sign trajectory is `--- / --+`
    - `boundary_timing_gap`: modal target-layer sign trajectory is `-++ / --+`
  - this means `threshold-sign persistence` is already sufficient to separate stable branches, absence-type boundary seeds, and timing-gap boundary seeds at cohort scale
  - new timing-gap targeted interventions now show that timing-gap itself is not a single family:
    - `seed 17`: any early active layer (`0`, `1`, `2`, `0+2`) is sufficient to advance the `layer 3` commit from `epoch 6` to `epoch 4`, with no extra late layers
    - `seed 31`: only `layer 0`-containing prefixes (`0`, `0+2`) advance the same `layer 3` commit, and both simultaneously induce a new late `layer 2`
  - so timing-gap seeds should be treated as a separate intervention family with at least two subtypes: a distributed upstream-trigger subtype and a layer-0-coupled subtype with side effects
- this justifies treating `CIFAR-10` as the main next-step dataset for broader seed sweeps

Pending later phases:

- ticket extraction / IMP-style pruning
- dense shadow-prune instrumentation
- phase discovery reports
- broader seed-level intervention sweeps

## Main protocol rule

For this study, the only seed that should vary across the core sweep is:

- `model_init_seed`

The following should remain fixed within a sweep:

- dataset split
- train sample order
- optimizer and schedule
- training budget
- augmentation policy
- pruning schedule

## Result root

Outputs are expected under:

- `results/init_only_lth_20260401/`

Current analysis artifacts include:

- `results/init_only_lth_20260401/mnist/lenet300100/dense_init_only/20260401_152851/analysis/init_only_phase0_report_zh.md`
- `results/init_only_lth_20260401/fashion_mnist/lenet300100/dense_init_only/20260401_153246/analysis/init_only_phase0_report_zh.md`
- `results/init_only_lth_20260401/fashion_mnist/lenet300100/structured_init_only/20260401_153803/analysis/init_only_paired_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/dense_init_only/20260401_155624/analysis/init_only_phase0_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_162714/analysis/init_only_paired_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_163849/analysis/init_only_paired_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_163849/analysis/phase_taxonomy_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_163849/analysis/early_marker_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_163849/analysis/mismatch_case_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_163849/analysis/late_commit_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_163849/analysis/threshold_sign_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_195400/analysis/phase_taxonomy_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_195400/analysis/mismatch_case_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_195400/analysis/threshold_sign_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_211125/analysis/init_only_paired_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_211125/analysis/phase_taxonomy_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_211125/analysis/mismatch_case_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_211125/analysis/late_commit_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260401_211125/analysis/threshold_sign_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_095410/analysis/init_only_paired_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_095410/analysis/phase_taxonomy_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_095410/analysis/mismatch_case_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_095410/analysis/late_commit_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_095410/analysis/threshold_sign_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_125209/analysis/init_only_paired_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_125209/analysis/phase_taxonomy_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_125209/analysis/early_marker_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_125209/analysis/mismatch_case_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_125209/analysis/late_commit_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/structured_init_only/20260402_125209/analysis/threshold_sign_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/feedback_interventions/seed_17/20260401_221042/analysis/feedback_intervention_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_17/20260402_162719/analysis/timing_gap_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/feedback_interventions/seed_18/20260401_221805/analysis/feedback_intervention_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/feedback_interventions/seed_30/20260402_101423/analysis/feedback_intervention_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/feedback_interventions/seed_31/20260402_101423/analysis/feedback_intervention_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_18/20260401_222230/analysis/layer_feedback_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_30/20260402_101937/analysis/layer_feedback_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/epoch4_layer_feedback_interventions/seed_30/20260402_113120/analysis/layer_feedback_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_31/20260402_101937/analysis/layer_feedback_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_31/20260402_101937/analysis/timing_gap_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/feedback_interventions/seed_3/20260401_181753/analysis/feedback_intervention_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/feedback_interventions/seed_8/20260401_202049/analysis/feedback_intervention_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/feedback_interventions/seed_9/20260401_201415/analysis/feedback_intervention_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_3/20260401_192206/analysis/layer_feedback_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_8/20260401_202516/analysis/layer_feedback_report_zh.md`
- `results/init_only_lth_20260401/cifar10/smallconv/layer_feedback_interventions/seed_9/20260401_204621/analysis/layer_feedback_report_zh.md`
- `results/init_only_lth_20260401/structured_cohort_overview/structured_cohort_overview_zh.md`
- `results/init_only_lth_20260401/structured_cohort_overview/prefix_branch_stability_report_zh.md`
- `results/init_only_lth_20260401/structured_cohort_overview/threshold_persistence_cohort_report_zh.md`
- `results/init_only_lth_20260401/timing_gap_overview/timing_gap_family_overview_zh.md`
- `results/init_only_lth_20260401/layer_feedback_overview/layer_feedback_overview_zh.md`
- `results/init_only_lth_20260401/boundary_family_overview/boundary_family_report_zh.md`
- `results/init_only_lth_20260401/phase0_overview/phase0_overview_zh.md`
- `results/init_only_lth_20260401/structured_overview/structured_overview_zh.md`
