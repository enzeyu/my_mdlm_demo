# Best 10000-step CoDraft-Diff Evaluation

Checkpoint: `results/codraft_diff_gpu1_extended/long_train/checkpoints_10000/draft_refine_adapter/checkpoint.pt`

## Experiments

- `best_10000_eval.csv/json`: ratio=0.20 method comparison.
- `ratio_sweep_10000.csv/json`: refinement ratio sweep.
- `gate_threshold_sweep.csv/json`: adapter/utility gate threshold sweep.
- `selector_ablation.csv/json`: selector feature ablation.
- `next_step_decision.md`: recommendation for whether to run 30000-step training.

## Plots

- `accuracy_vs_ratio_10000.png`: accuracy across refinement ratios.
- `wrong_edit_vs_ratio_10000.png`: harmful edit rate across ratios.
- `edit_gain_vs_ratio_10000.png`: W->C minus C->W across ratios.
- `precision_recall_tradeoff_10000.png`: correction precision/recall tradeoff.
- `final_acc_vs_gate_threshold.png`, `wrong_edit_vs_gate_threshold.png`, `edit_gain_vs_gate_threshold.png`, `wtc_ctw_vs_gate_threshold.png`: gate sweep plots.
- `selector_ablation_acc.png`, `selector_ablation_edit_gain.png`, `selector_ablation_wrong_edit.png`: selector feature ablation plots.

Current best non-oracle: `error_aware_adapter_gate` final_acc=0.3327.
Recommendation: Start 30000-step training.
