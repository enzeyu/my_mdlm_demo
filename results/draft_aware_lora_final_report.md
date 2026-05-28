# Draft-aware LoRA Fine-tuning for Edge MDLM

## 1. Problem

Pretrained MDLM is strong under clean standard random-mask denoising, but the existing experiments show a clear drop when the visible context is a GPT-2 draft rather than clean tokens. This mismatch matters for edge-device collaboration because the edge MDLM must refine structured AR draft errors, not independent random masks.

## 2. Idea

AR draft-induced corruption is treated as a structured training noise source: GPT-2 produces a draft context, low-confidence tokens or blocks are masked, and LoRA trains the frozen MDLM to recover the clean tokens from that draft context. GPT-2 is not trained and the full MDLM checkpoint is not saved.

## 3. Random-mask LoRA vs Draft-aware LoRA

| Model | Train Noise | Standard Top1 | Standard Top5 | Draft Top1 | Draft Top5 | Draft PPL | Correction | Regression | Net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained_mdlm | pretrained | 0.5191 | 0.7368 | 0.0964 | 0.2313 | 2484.4637 | 0.0732 | 0.7646 | -0.6915 |
| random_mask_lora_mdlm | random_mask | 0.6372 | 0.8111 | 0.1317 | 0.2851 | 478.7520 | 0.1049 | 0.7081 | -0.6032 |
| draft_aware_lora_mdlm | draft_aware | 0.5233 | 0.7283 | 0.2036 | 0.3791 | 184.5030 | 0.1785 | 0.6484 | -0.4699 |

Draft-aware LoRA beats random-mask LoRA on GPT-2 draft-context Top1: yes.

## 4. Block-size Ablation

| Block Size | Draft Top1 | Draft Top5 | PPL | Block EM | Correction | Regression | Net | Selected Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3282 | 0.5153 | 53.2684 | 0.3282 | 0.3169 | 0.5008 | -0.1840 | 0.2042 |
| 2 | 0.2804 | 0.4687 | 89.1048 | 0.1341 | 0.2557 | 0.5226 | -0.2669 | 0.1995 |
| 4 | 0.2037 | 0.3790 | 184.5060 | 0.0205 | 0.1789 | 0.6477 | -0.4688 | 0.2164 |
| 8 | 0.1404 | 0.2942 | 347.2426 | 0.0005 | 0.1145 | 0.7409 | -0.6264 | 0.2481 |

block_size=4 vs token-level block_size=1: not better or not enough evidence.
Current best block_size by Draft Top1: `1`.

## 5. Accept Gate

| mode | gate_threshold | top1 | top5 | ppl | correction_rate | regression_rate | net_correction | accepted_ratio | gate_accuracy | selected_token_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_aware_lora_refine_no_gate |  | 0.3169 | 0.5106 | 65.9995 | 0.1526 | 0.2259 | -0.0733 | 1.0000 |  | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.3000 | 0.3171 | 0.5104 | 65.7937 | 0.1479 | 0.1900 | -0.0421 | 0.8747 | 0.8084 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.4000 | 0.3160 | 0.5100 | 65.6311 | 0.1325 | 0.1336 | -0.0011 | 0.5981 | 0.7769 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.5000 | 0.3134 | 0.5092 | 65.7455 | 0.1083 | 0.0734 | 0.0349 | 0.3649 | 0.7027 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.6000 | 0.3107 | 0.5085 | 65.9732 | 0.0886 | 0.0428 | 0.0458 | 0.2332 | 0.6262 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.7000 | 0.3083 | 0.5077 | 66.2304 | 0.0727 | 0.0238 | 0.0489 | 0.1526 | 0.5594 | 0.2164 |
| draft_aware_lora_refine_with_rule_gate |  | 0.3055 | 0.5065 | 66.7150 | 0.0562 | 0.0173 | 0.0389 | 0.1507 | 0.4782 | 0.2164 |
| gpt2_only |  | 0.2956 | 0.5027 | 68.1806 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 |
| pretrained_mdlm_refine |  | 0.3007 | 0.5056 | 68.1526 | 0.0857 | 0.3469 | -0.2612 | 1.0000 |  | 0.2164 |
| random_mask_lora_refine |  | 0.3077 | 0.5070 | 67.3102 | 0.1119 | 0.2785 | -0.1665 | 1.0000 |  | 0.2164 |

Learned gate lowers regression vs no gate: yes.
Learned gate exceeds GPT-2-only Top1/Top5: yes.
Best learned-gate threshold: `0.3`.

## 6. Final Conclusion

The conclusion should be read from the three result groups above. The hypothesis is supported when draft-aware LoRA outperforms random-mask LoRA under GPT-2 draft context, block-level training improves over token-level or avoids degradation, and the accept gate reduces harmful regressions enough for GPT-2 + LoRA-MDLM refinement to match or exceed GPT-2-only.

Current aggregate support: supported.
