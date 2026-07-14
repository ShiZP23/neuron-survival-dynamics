# Phase Freeze Interventions Pilot

## Question

Can the late degradation of a structural phase be reduced by freezing structural updates after the phase is already formed?

## Interventions

- `freeze_after_commit`: allow the run to reach its baseline structural commit epoch, then stop all later prune/grow updates while continuing weight optimization
- `freeze_after_best`: allow the run to proceed until its baseline best epoch, then stop all later prune/grow updates while continuing weight optimization

## Source Pool

- baseline rows from `results/studies/structural_phase_effects_20260314/phase_rows.csv`
- pilot restricted to the `base_seed` family for a cleaner first causal read

## Selection Rule

- choose representative runs within each source phase
- rank candidates by closeness to the phase medians in:
  - `log10(final_minus_best)`
  - `share_2_drift`

## Readout

Primary pilot readouts:

- `final_test_loss`
- `final_minus_best`
- `share_2_drift`
- `post_best_drift_rate`

## Output Layout

Raw reruns:

- `results/followup_20260314/phase_freeze_interventions_pilot_runs/`

Derived study package:

- `results/studies/phase_freeze_interventions_pilot_20260314/`

## Interpretation Goal

- if `freeze_after_commit` already removes most degradation, then late structural motion is likely a major causal driver
- if only `freeze_after_best` helps, then the damage may happen after the best checkpoint rather than during the whole post-commit window
- if neither helps much, the phase may already encode the bad endpoint before late drift becomes visible
