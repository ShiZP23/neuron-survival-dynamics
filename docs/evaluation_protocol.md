# Evaluation Protocol

## Goal

This project now distinguishes between:

- model quality under a fixed training budget
- training stability late in optimization

Those two questions should not be collapsed into a single `final test loss` number when the training trajectory shows late degradation.

## Main reporting rule

For new experiments, the primary comparison metric is:

- `test_loss_at_best_val`

This is defined as the test loss at the epoch where validation loss reaches its minimum during a fixed-budget training run.

This follows the standard pattern:

- keep the optimization budget fixed across methods
- use validation data for checkpoint selection
- evaluate the selected checkpoint on the test set

## Secondary reporting rule

The following quantities should be reported alongside the primary metric:

- `final_test_loss`
- `best_val_loss`
- `best_epoch`
- `final_test_loss - test_loss_at_best_val`
- selected parameter count and final parameter count

These quantities expose whether a method learns well but becomes unstable later.

## Why not use `best test loss` directly

`best test loss` over the whole trajectory is useful for diagnosis, but it should not be the main comparison metric because it uses the test set for checkpoint selection.

In this repository, `best test loss` may still be plotted as an oracle diagnostic, but it should not be used for the main table.

## Why not rely only on `final test loss`

For stable methods, final performance may be a reasonable summary.
For dynamic-structure methods, especially when structure updates continue late into training, the final checkpoint may understate the method's attainable generalization quality.

Therefore:

- `test_loss_at_best_val` is the main performance metric
- `final_test_loss` is the main stability metric

## Legacy results

Older runs in `former results/` do not contain epoch-level validation loss.
They should therefore be treated as:

- `legacy_final_test_only`

These runs remain useful for exploratory analysis, seed sensitivity studies, and qualitative comparisons, but they are not directly comparable to new validation-selected runs in a publication-quality main table.

## Sources

- RigL evaluates sparse training under fixed training budgets and reports repeated experiments under a controlled protocol:
  https://proceedings.mlr.press/v119/evci20a.html
- Rice et al. distinguish between best and final robust performance and use validation-based early stopping as the practical model-selection rule:
  https://proceedings.mlr.press/v119/rice20a.html
- "Show Your Work" argues that reporting should include more than a single test-set number and should expose experimental procedure more transparently:
  https://arxiv.org/abs/1909.03004
- "Don’t stop me now" studies checkpoint selection criteria directly and supports validation-loss-based parameter selection as a strong default:
  https://arxiv.org/abs/2602.22107
