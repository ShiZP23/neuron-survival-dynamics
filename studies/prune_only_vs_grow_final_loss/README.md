# Prune-Only vs Grow Final-Loss Study

## Question

Hypothesis:

`On harder tasks, prune_only is more likely than grow-based strategies to finish training with a lower final test loss under the same training budget.`

This study isolates that claim from the broader project.

## Scope

Primary comparison:
- `prune_only`
- `prune_grow_random`
- `prune_grow_split`

Reference baseline:
- `fixed`

Tasks:
- `simple`
- `medium`
- `hard`

Current data source:
- `results/publishable_pilot_20260313/summary_three_task.csv`

## Endpoints

Primary endpoint:
- `final_test_loss`

Secondary endpoints:
- `test_loss_at_best_val`
- `best_param_count`
- seed-wise win rate of `prune_only` against each grow strategy
- paired per-seed log-ratio in final test loss

Explicitly excluded:
- `best_test_loss`

Rationale:
- `best_test_loss` is a post-hoc oracle-on-test quantity and is not needed for this study.
- `test_loss_at_best_val` is the correct control metric because it remains consistent with the main evaluation protocol.

## Why final test loss here

This study is intentionally narrower than the main evaluation protocol.

The main paper protocol should still use validation-selected checkpoints as the primary model-selection rule.
This study instead focuses on a different question:

`When training is allowed to run to the scheduled end, does prune_only more reliably finish in a good state than grow-based strategies, especially on harder tasks?`

That makes `final_test_loss` the right primary endpoint for this branch of the investigation.

## Required evidence

To support the hypothesis, the study should show all of the following on `hard`:
- lower mean or median `final_test_loss` for `prune_only` than both grow strategies
- favorable per-seed win rate for `prune_only` against each grow strategy
- paired seed comparisons that do not rely only on group averages

To strengthen the “harder tasks” framing, the study should also show that:
- the advantage is weaker or absent on `simple`
- the trend changes monotonically or at least directionally from `simple -> medium -> hard`

## Outputs

Generated artifacts live in:
- `results/studies/prune_only_vs_grow_final_loss_20260314/`

Expected primary figures:
- final-loss distributions by task
- paired seed slopegraphs for `prune_only` vs each grow strategy
- task-by-task win-rate summary
- hard-task final-loss vs parameter frontier

Expected control figures:
- validation-selected loss distributions by task
- task-by-task win-rate summary using `test_loss_at_best_val`

## Notes

- This study does not modify the main training protocol.
- It is designed to test a concrete claim using the results already produced.
