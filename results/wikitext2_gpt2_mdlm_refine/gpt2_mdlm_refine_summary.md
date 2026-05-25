# Diffusion-Assisted Autoregressive Refinement

- Device GPT-2: `/mnt/data/enzeyu/hf_downloads/models/gpt2`
- Edge MDLM: `/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt`
- Uncertainty score: `inverse_confidence`
- Refine ratios: `[0.05, 0.1, 0.2, 0.3]`

| Mode | Refine Ratio | Top1 | Top5 | PPL | Correction | Regression | Error Detect Precision | Error Detect Recall | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_only | 0.00 | 0.2975 | 0.5040 | 66.9593 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0229 |
| mdlm_only | 1.00 | 0.0529 | 0.0756 | 4261.1763 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 |
| random_refine | 0.05 | 0.2943 | 0.5004 | 70.7891 | 0.1643 | 0.6128 | 0.7061 | 0.0505 | 0.0426 |
| gpt2_mdlm_refine | 0.05 | 0.3032 | 0.5128 | 65.5699 | 0.1384 | 0.7578 | 0.9717 | 0.0696 | 0.0428 |
| random_refine | 0.10 | 0.2895 | 0.4954 | 75.3932 | 0.1543 | 0.6343 | 0.7028 | 0.1005 | 0.0428 |
| gpt2_mdlm_refine | 0.10 | 0.3073 | 0.5178 | 68.2869 | 0.1320 | 0.7435 | 0.9606 | 0.1373 | 0.0428 |
| random_refine | 0.20 | 0.2796 | 0.4828 | 88.6797 | 0.1458 | 0.6453 | 0.7023 | 0.2003 | 0.0428 |
| gpt2_mdlm_refine | 0.20 | 0.3152 | 0.5265 | 85.4274 | 0.1431 | 0.7569 | 0.9392 | 0.2679 | 0.0429 |
| random_refine | 0.30 | 0.2649 | 0.4640 | 108.5064 | 0.1289 | 0.6688 | 0.7023 | 0.3004 | 0.0428 |
| gpt2_mdlm_refine | 0.30 | 0.3202 | 0.5264 | 92.9190 | 0.1427 | 0.7275 | 0.9227 | 0.3947 | 0.0428 |

## Questions

1. MDLM 是否能有效修正 GPT-2 的低置信 token？否，或证据不足。最佳 refine ratio=0.20，Top5 gain=0.022452。
2. uncertainty selection 是否优于 random selection？是。同 ratio 下 Top5 gain=0.043667。
3. 只 refine 少量 token 是否能获得明显质量提升？是。最佳 refine 的 estimated compute saved=0.7996。
4. 当前方向是否比 GPT-2 assist MDLM 更有希望？是。这个方向至少把强 MDLM 用在弱 GPT-2 的错误位置，建模方向更合理。
5. 下一步建议：先优化 error detection，尝试 threshold/gating、校准 GPT-2 confidence、加入 right-context verifier 特征；若离 full MDLM 仍有差距，再训练一个轻量 selector 或 distill MDLM refinement policy。

Full MDLM Top5=0.075638；最佳 refine Top5=0.526455；是否接近 full MDLM：是。
