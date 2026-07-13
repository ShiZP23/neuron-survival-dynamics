# Prune-Only vs Fixed Final-Loss Study

## Question

Hypothesis:

`On harder tasks, prune_only is more likely than fixed-width training to finish with a lower final test loss under the same training budget.`

This is the current primary focused study.

## Scope

Primary comparison:
- `prune_only`
- `fixed`

Secondary context:
- earlier `prune_only vs grow` analysis is retained as a secondary study, not the main claim path

Tasks:
- `simple`
- `medium`
- `hard`

Current data source:
- `results/publishable_pilot_20260313/summary_three_task.csv`

## Endpoints

Primary endpoint:
- `final_test_loss`

Secondary control endpoint:
- `test_loss_at_best_val`

Supporting quantities:
- `best_param_count`
- seed-matched win rate
- paired per-seed log-ratio in loss

Explicitly excluded:
- `best_test_loss`

## Interpretation rule

This study is about `end-of-training behavior`, not checkpoint oracle performance.

Therefore:
- use `final_test_loss` for the headline claim
- use `test_loss_at_best_val` only as a control to separate stability effects from peak attainable performance

## Success criterion

The hypothesis is supported on `hard` only if the evidence points in the same direction across:
- lower mean or median `final_test_loss`
- favorable seed-matched win rate
- paired-seed comparisons that do not rely only on group averages

If `prune_only` wins on `selected` but not on `final`, the claim is false for this branch.
If `prune_only` loses on both, the hypothesis should be rejected rather than softened.

## Outputs

Generated artifacts live in:
- `results/studies/prune_only_vs_fixed_final_loss_20260314/`

Expected primary figures:
- final-loss distributions by task
- paired seed slopegraphs for `prune_only` vs `fixed`
- task-by-task win-rate summary
- hard-task final-loss / parameter frontier

Expected control figures:
- validation-selected loss distributions by task
- task-by-task win-rate summary using `test_loss_at_best_val`
