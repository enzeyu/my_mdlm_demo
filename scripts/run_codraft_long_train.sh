#!/usr/bin/env bash
# Evaluates available long-training checkpoints and optionally launches missing 10000-step adapter training.
# Input: configs/codraft_diff_wikitext2_gpu1.yaml.
# Output: results/codraft_diff_gpu1_extended/long_train/
# GPU: set GPU_ID, default physical GPU 1.
# Resume behavior: existing outputs are skipped unless FORCE=1 is set.

set -euo pipefail

GPU_ID="${GPU_ID:-1}"
OUT_DIR="${OUT_DIR:-results/codraft_diff_gpu1_extended}"
CONFIG="${CONFIG:-configs/codraft_diff_wikitext2_gpu1.yaml}"
EVAL_STEPS="${EVAL_STEPS:-200}"
BASE_CKPT="${BASE_CKPT:-results/codraft_diff_gpu1/draft_refine_adapter/checkpoint.pt}"

if [[ "${TRAIN_10000:-0}" == "1" ]]; then
  LONG_DIR="${OUT_DIR}/long_train/checkpoints_10000"
  LONG_CKPT="${LONG_DIR}/draft_refine_adapter/checkpoint.pt"
  if [[ ! -f "${LONG_CKPT}" || "${FORCE:-0}" == "1" ]]; then
    mkdir -p "${LONG_DIR}"
    python - <<'PY'
from pathlib import Path
import yaml
base = yaml.safe_load(Path("configs/codraft_diff_wikitext2_gpu1.yaml").read_text())
base["save_dir"] = "results/codraft_diff_gpu1_extended/long_train/checkpoints_10000"
base["training"]["max_steps"] = 10000
base["train_steps"] = 10000
Path("configs/codraft_diff_wikitext2_gpu1_10000.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
PY
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_codraft_diff.py \
      --config configs/codraft_diff_wikitext2_gpu1_10000.yaml \
      --mode train_draft_refine_adapter
  fi
  STEP_CKPTS="3000:${BASE_CKPT},10000:${LONG_CKPT}"
else
  STEP_CKPTS="3000:${BASE_CKPT}"
fi

if [[ -f "${OUT_DIR}/long_train/long_train_eval.csv" && "${FORCE:-0}" != "1" ]]; then
  echo "Found existing ${OUT_DIR}/long_train/long_train_eval.csv; set FORCE=1 to rerun."
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python codraft_extended_eval.py long_train_eval \
  --config "${CONFIG}" \
  --checkpoint "${BASE_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --eval_steps "${EVAL_STEPS}" \
  --step_checkpoints "${STEP_CKPTS}"
