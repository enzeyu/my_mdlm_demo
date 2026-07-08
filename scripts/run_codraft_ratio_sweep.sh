#!/usr/bin/env bash
# Runs CoDraft-Diff refinement-ratio sweep on GPU 1.
# Input: configs/codraft_diff_wikitext2_gpu1.yaml and the 3000-step adapter checkpoint.
# Output: results/codraft_diff_gpu1_extended/ratio_sweep/
# Resume behavior: if ratio_sweep_eval.csv exists, the script exits unless FORCE=1 is set.

set -euo pipefail

GPU_ID="${GPU_ID:-1}"
OUT_DIR="${OUT_DIR:-results/codraft_diff_gpu1_extended}"
CONFIG="${CONFIG:-configs/codraft_diff_wikitext2_gpu1.yaml}"
CKPT="${CKPT:-results/codraft_diff_gpu1/draft_refine_adapter/checkpoint.pt}"
EVAL_STEPS="${EVAL_STEPS:-200}"

if [[ -f "${OUT_DIR}/ratio_sweep/ratio_sweep_eval.csv" && "${FORCE:-0}" != "1" ]]; then
  echo "Found existing ${OUT_DIR}/ratio_sweep/ratio_sweep_eval.csv; set FORCE=1 to rerun."
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python codraft_extended_eval.py ratio_sweep \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --eval_steps "${EVAL_STEPS}"
