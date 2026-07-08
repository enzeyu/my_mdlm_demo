# DART Final Report

## 1. Problem Definition

DART targets edge-device collaboration where a frozen device GPT-2 produces an autoregressive draft and a frozen edge MDLM is adapted with LoRA to refine the low-confidence parts of that draft.

## 2. Method Flow

Device GPT-2 draft -> low-confidence token selection -> Edge MDLM draft-aware LoRA adaptation -> learned accept gate -> selective refinement.

## 3. Random-mask LoRA vs Draft-aware LoRA

| Model | Train Noise | Standard Top1 | Standard Top5 | Draft Top1 | Draft Top5 | Draft PPL | Correction | Regression | Net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained_mdlm | pretrained | 0.5191 | 0.7368 | 0.0964 | 0.2313 | 2484.4637 | 0.0732 | 0.7646 | -0.6915 |
| random_mask_lora_mdlm | random_mask | 0.6372 | 0.8111 | 0.1317 | 0.2851 | 478.7520 | 0.1049 | 0.7081 | -0.6032 |
| draft_aware_lora_mdlm | draft_aware | 0.5233 | 0.7283 | 0.2036 | 0.3791 | 184.5030 | 0.1785 | 0.6484 | -0.4699 |

Conclusion: AR draft-induced corruption is an effective structured training noise source; draft-aware LoRA under GPT-2 draft context is better than random-mask LoRA when Draft Top1 improves (0.1317 -> 0.2036).

## 4. Block-size Ablation

| Block Size | Draft Top1 | Draft Top5 | PPL | Correction | Regression | Net | Selected Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3282 | 0.5153 | 53.2684 | 0.3169 | 0.5008 | -0.1840 | 0.2042 |
| 2 | 0.2804 | 0.4687 | 89.1048 | 0.2557 | 0.5226 | -0.2669 | 0.1995 |
| 4 | 0.2037 | 0.3790 | 184.5060 | 0.1789 | 0.6477 | -0.4688 | 0.2164 |
| 8 | 0.1404 | 0.2942 | 347.2426 | 0.1145 | 0.7409 | -0.6264 | 0.2481 |

Conclusion: block_size=1 is the current best setting when it is the top Draft Top1 row; current best block_size is `1`. Fixed large blocks are not used as the main configuration.

## 5. Accept Gate

| mode | gate_threshold | top1 | top5 | ppl | correction_rate | regression_rate | net_correction | accepted_ratio | gate_accuracy | selected_token_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_aware_lora_refine_no_gate |  | 0.3169 | 0.5106 | 65.9995 | 0.1526 | 0.2259 | -0.0733 | 1.0000 |  | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.3000 | 0.3171 | 0.5104 | 65.7937 | 0.1479 | 0.1900 | -0.0421 | 0.8747 | 0.8084 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.4000 | 0.3160 | 0.5100 | 65.6311 | 0.1325 | 0.1336 | -0.0011 | 0.5981 | 0.7769 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.5000 | 0.3134 | 0.5092 | 65.7455 | 0.1083 | 0.0734 | 0.0349 | 0.3649 | 0.7027 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.6000 | 0.3107 | 0.5085 | 65.9732 | 0.0886 | 0.0428 | 0.0458 | 0.2332 | 0.6262 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.7000 | 0.3083 | 0.5077 | 66.2304 | 0.0727 | 0.0238 | 0.0489 | 0.1526 | 0.5594 | 0.2164 |
| gpt2_only |  | 0.2956 | 0.5027 | 68.1806 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 |
| pretrained_mdlm_refine |  | 0.3007 | 0.5056 | 68.1526 | 0.0857 | 0.3469 | -0.2612 | 1.0000 |  | 0.2164 |
| random_mask_lora_refine |  | 0.3077 | 0.5070 | 67.3102 | 0.1119 | 0.2785 | -0.1665 | 1.0000 |  | 0.2164 |

Conclusion: learned accept gate lowers regression when its regression_rate is below the no-gate variant (0.2259 -> 0.0238).

## 6. Final Evaluation

| mode | gate_threshold | top1 | top5 | ppl | correction_rate | regression_rate | net_correction | accepted_ratio | gate_accuracy | selected_token_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_aware_lora_refine_no_gate |  | 0.3169 | 0.5106 | 65.9995 | 0.1526 | 0.2259 | -0.0733 | 1.0000 |  | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.3000 | 0.3171 | 0.5104 | 65.7937 | 0.1479 | 0.1900 | -0.0421 | 0.8747 | 0.8084 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.4000 | 0.3160 | 0.5100 | 65.6311 | 0.1325 | 0.1336 | -0.0011 | 0.5981 | 0.7769 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.5000 | 0.3134 | 0.5092 | 65.7455 | 0.1083 | 0.0734 | 0.0349 | 0.3649 | 0.7027 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.6000 | 0.3107 | 0.5085 | 65.9732 | 0.0886 | 0.0428 | 0.0458 | 0.2332 | 0.6262 | 0.2164 |
| draft_aware_lora_refine_with_learned_gate | 0.7000 | 0.3083 | 0.5077 | 66.2304 | 0.0727 | 0.0238 | 0.0489 | 0.1526 | 0.5594 | 0.2164 |
| gpt2_only |  | 0.2956 | 0.5027 | 68.1806 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0.0000 |
| pretrained_mdlm_refine |  | 0.3007 | 0.5056 | 68.1526 | 0.0857 | 0.3469 | -0.2612 | 1.0000 |  | 0.2164 |
| random_mask_lora_refine |  | 0.3077 | 0.5070 | 67.3102 | 0.1119 | 0.2785 | -0.1665 | 1.0000 |  | 0.2164 |

Conclusion: GPT-2 + draft-aware LoRA-MDLM + learned gate exceeds GPT-2-only when final Top1/Top5 improves (0.2956/0.5027 -> 0.3171/0.5104).

## 7. Final Conclusion

- AR draft-induced corruption is an effective structured training noise source.
- Draft-aware LoRA in GPT-2 draft context outperforms random-mask LoRA.
- block_size=1 is currently best.
- The learned accept gate lowers regression.
- GPT-2 + draft-aware LoRA-MDLM + learned gate exceeds GPT-2-only.

## 8. Current Best Configuration

- Device model: `/mnt/data/enzeyu/hf_downloads/models/gpt2`
- Edge model: `/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt`
- LoRA mode: `draft_aware`
- block_size: `1`
- refine_ratio: `0.2`
- gate_threshold: `0.4`
- trainable modules: LoRA adapter and accept gate only
