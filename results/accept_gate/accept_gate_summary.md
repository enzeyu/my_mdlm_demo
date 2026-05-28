# Accept Gate Evaluation

| Mode | Threshold | Top1 | Top5 | PPL | Correction | Regression | Net | Accepted | Gate Acc | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| draft_aware_lora_refine_no_gate | 0.00 | 0.3169 | 0.5106 | 65.9995 | 0.1526 | 0.2259 | -0.0733 | 1.0000 | 0.0000 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.30 | 0.3171 | 0.5104 | 65.7937 | 0.1479 | 0.1900 | -0.0421 | 0.8747 | 0.8084 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.40 | 0.3160 | 0.5100 | 65.6311 | 0.1325 | 0.1336 | -0.0011 | 0.5981 | 0.7769 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.50 | 0.3134 | 0.5092 | 65.7455 | 0.1083 | 0.0734 | 0.0349 | 0.3649 | 0.7027 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.60 | 0.3107 | 0.5085 | 65.9732 | 0.0886 | 0.0428 | 0.0458 | 0.2332 | 0.6262 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.70 | 0.3083 | 0.5077 | 66.2304 | 0.0727 | 0.0238 | 0.0489 | 0.1526 | 0.5594 | 0.2164 |
| draft_aware_lora_refine_with_rule_gate | 0.00 | 0.3055 | 0.5065 | 66.7150 | 0.0562 | 0.0173 | 0.0389 | 0.1507 | 0.4782 | 0.2164 |
| gpt2_only | 0.00 | 0.2956 | 0.5027 | 68.1806 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pretrained_mdlm_refine | 0.00 | 0.3007 | 0.5056 | 68.1526 | 0.0857 | 0.3469 | -0.2612 | 1.0000 | 0.0000 | 0.2164 |
| random_mask_lora_refine | 0.00 | 0.3077 | 0.5070 | 67.3102 | 0.1119 | 0.2785 | -0.1665 | 1.0000 | 0.0000 | 0.2164 |

## Answers

1. learned gate 是否降低 regression？是。
2. learned gate 是否让 net_correction 接近 0 或转正？否；net=-0.042102。
3. GPT-2 + draft-aware LoRA-MDLM + gate 是否超过 GPT-2-only？是。
4. gate 是否优于 rule-based gate？否或证据不足。
5. 当前最佳 threshold 是多少？`0.30`。
