# LoRA Training Mode Comparison

| Model | Train Noise | Standard Top1 | Standard Top5 | Draft Top1 | Draft Top5 | Draft PPL | Correction | Regression | Net |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained_mdlm | pretrained | 0.5191 | 0.7368 | 0.0964 | 0.2313 | 2484.4637 | 0.0732 | 0.7646 | -0.6915 |
| random_mask_lora_mdlm | random_mask | 0.6372 | 0.8111 | 0.1317 | 0.2851 | 478.7520 | 0.1049 | 0.7081 | -0.6032 |
| draft_aware_lora_mdlm | draft_aware | 0.5233 | 0.7283 | 0.2036 | 0.3791 | 184.5030 | 0.1785 | 0.6484 | -0.4699 |

## Answer

1. draft-aware LoRA 在 GPT-2 draft context 下是否优于 random-mask LoRA？是。
2. draft-aware LoRA 的 refinement net correction 是否优于 random-mask LoRA？是。
3. 该结果是否支持 AR draft 作为结构化训练噪声源？支持。
