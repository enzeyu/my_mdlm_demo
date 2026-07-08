# DART: Draft-aware LoRA Fine-tuning for Edge MDLM

## 1. Project Overview

This project implements a collaborative edge refinement framework between a device-side GPT-2 draft model and an edge-side MDLM adapted with draft-aware LoRA.

Core pipeline:

```text
Device GPT-2 draft
-> low-confidence token selection
-> Edge MDLM LoRA adaptation
-> learned accept gate
-> selective refinement
```

Key constraints:

- GPT-2 is frozen and only generates drafts, token-level confidence, and low-confidence masks.
- The MDLM backbone is frozen.
- Only LoRA parameters and the accept gate are trained.
- GPT-2 draft context is used as structured training noise.
- The goal is to adapt edge MDLM refinement to the actual device GPT-2 draft context.

## 2. Method

### 2.1 AR Draft-induced Corruption

Device GPT-2 runs teacher-forced draft prediction. Token confidence, entropy, and margin are computed from GPT-2 logits, and low-confidence tokens are selected for refinement.

### 2.2 Draft-aware LoRA Adaptation

The edge MDLM receives GPT-2 draft context with selected low-confidence positions masked. The MDLM backbone stays frozen and only LoRA adapters are trained to reconstruct clean tokens from the draft-aware corruption pattern.

### 2.3 Utility-aware Accept Gate

The learned accept gate predicts whether accepting the LoRA-MDLM candidate is useful. Positive examples are tokens where GPT-2 was wrong and LoRA-MDLM fixes the token. Negative examples are tokens where GPT-2 was correct and LoRA-MDLM would regress it.

## 3. Model Preparation

Device GPT-2:

```text
/mnt/data/enzeyu/hf_downloads/models/gpt2
```

Edge MDLM:

```text
/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
```

The project uses these local pretrained models and does not depend on any extra full MDLM checkpoint.

## 4. Training Commands

```bash
python train_draft_aware_lora.py \
  --config configs/wikitext2_random_mask_lora.yaml

python train_draft_aware_lora.py \
  --config configs/wikitext2_draft_aware_lora.yaml

python train_accept_gate.py \
  --config configs/wikitext2_accept_gate.yaml
```

## 5. Evaluation Commands

```bash
python eval_final_refinement.py \
  --config configs/wikitext2_dart_final.yaml
```

## 6. Ablation Commands

```bash
python run_block_size_ablation.py \
  --base_config configs/wikitext2_draft_aware_lora.yaml \
  --block_sizes 1 2 4 8
```

## 7. Analysis

```bash
python analyze_results.py
```

This generates:

```text
results/final_report.md
```

## 8. Metrics Explanation

- Standard Top1 / Top5: accuracy under standard clean/random-mask MDLM-style evaluation; this is mainly a sanity check.
- Draft Top1 / Draft Top5: accuracy under GPT-2 draft context; this is the core edge-device collaboration metric.
- Draft PPL: perplexity under GPT-2 draft-context refinement.
- Correction: fraction of selected GPT-2-wrong tokens corrected by MDLM refinement.
- Regression: fraction of selected GPT-2-correct tokens changed into wrong predictions.
- Net Correction: correction rate minus regression rate.
- Accepted Ratio: fraction of selected refinement candidates accepted by the gate.
- Selected Token Ratio: fraction of valid tokens selected for possible refinement.

Standard metrics are sanity checks. Draft metrics are the main target for the edge-device collaboration setting.

## 9. Recommended Running Order

Step 1: Train random-mask LoRA baseline

Step 2: Train draft-aware LoRA

Step 3: Train accept gate

Step 4: Run final evaluation

Step 5: Run block-size ablation

Step 6: Generate final report

## 10. Main Findings

1. Pretrained MDLM is strong under standard random-mask denoising but degrades under GPT-2 draft context.
2. Draft-aware LoRA outperforms random-mask LoRA under GPT-2 draft context.
3. block_size=1 is currently best, while fixed large blocks perform worse.
4. The learned accept gate reduces regression.
5. GPT-2 + draft-aware LoRA-MDLM + learned gate exceeds GPT-2-only.
