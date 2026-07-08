# CoDraft-Diff GPU1 Extended Experiments

This directory contains extended evaluation outputs for AR-Diffusion edge collaboration.

## Scripts

- `scripts/run_codraft_ratio_sweep.sh`: runs refinement-ratio sweep and writes `ratio_sweep/`.
- `scripts/run_codraft_multiseed.sh`: runs seed aggregation and writes `multiseed/`.
- `scripts/run_codraft_long_train.sh`: trains/evaluates longer checkpoints when requested.
- `scripts/analyze_codraft_results.py`: runs confidence-bin and edit-transition analysis.
- `scripts/plot_codraft_results.py`: regenerates plots from saved CSV files.

## Inputs

- Config: `configs/codraft_diff_wikitext2_gpu1.yaml`
- Adapter checkpoint: `results/codraft_diff_gpu1/draft_refine_adapter/checkpoint.pt`

## Outputs

- `ratio_sweep/ratio_sweep_eval.csv` and `.json`
- `multiseed/multiseed_eval.csv` and `.json`
- `analysis/confidence_bins.csv` and `.json`
- `analysis/edit_transitions.csv` and `.json`
- plot PNG files in corresponding subdirectories

All outputs are written under `results/codraft_diff_gpu1_extended/`; the original `results/codraft_diff_gpu1/` directory is not modified.
