# Diffusion-Assisted Autoregressive Refinement

- Device GPT-2: `/mnt/data/enzeyu/hf_downloads/models/gpt2`
- Edge MDLM: `/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt`
- Eval steps: `1000`
- Mask ratio: `0.15`
- Uncertainty score: `inverse_confidence`
- Refine ratios: `[0.05, 0.1, 0.2, 0.3]`

| Mode | Refine Ratio | Top1 | Top5 | PPL | Correction | Regression | Net Correction | Error Detect Precision | Error Detect Recall | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_only | 0.00 | 0.2956 | 0.5027 | 68.1806 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0211 |
| mdlm_only | 1.00 | 0.5208 | 0.7381 | 14.9206 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0198 |
| random_refine | 0.05 | 0.2915 | 0.4982 | 72.3433 | 0.1604 | 0.6456 | -0.4852 | 0.6995 | 0.0499 | 0.0410 |
| gpt2_mdlm_refine | 0.05 | 0.3011 | 0.5113 | 66.8935 | 0.1349 | 0.7478 | -0.6129 | 0.9720 | 0.0694 | 0.0412 |
| random_refine | 0.10 | 0.2883 | 0.4944 | 76.7482 | 0.1567 | 0.6198 | -0.4632 | 0.7047 | 0.1005 | 0.0411 |
| gpt2_mdlm_refine | 0.10 | 0.3053 | 0.5163 | 69.4400 | 0.1300 | 0.7460 | -0.6160 | 0.9614 | 0.1371 | 0.0411 |
| random_refine | 0.20 | 0.2768 | 0.4809 | 90.3066 | 0.1437 | 0.6521 | -0.5083 | 0.7017 | 0.1996 | 0.0410 |
| gpt2_mdlm_refine | 0.20 | 0.3134 | 0.5252 | 86.9716 | 0.1429 | 0.7567 | -0.6138 | 0.9399 | 0.2674 | 0.0411 |
| random_refine | 0.30 | 0.2628 | 0.4629 | 111.8546 | 0.1301 | 0.6817 | -0.5516 | 0.7054 | 0.3009 | 0.0410 |
| gpt2_mdlm_refine | 0.30 | 0.3182 | 0.5253 | 94.8575 | 0.1421 | 0.7311 | -0.5889 | 0.9234 | 0.3939 | 0.0410 |

## Diagnostics

- mdlm_standard_top1=0.520824
- mdlm_standard_top5=0.738147
- mdlm_draft_context_top1=0.151833
- mdlm_draft_context_top5=0.298576
- best_gpt2_mdlm_refine_ratio=0.30
- best_gpt2_error_detection_precision=0.923418
- best_gpt2_error_detection_recall=0.393897

## Questions

1. MDLM-only baseline 为什么异常低？旧逻辑把整段序列全部 mask，并在 full valid sequence 上评估，和 edge_only 的随机 masked recovery 不一致；当前已改为按照 mask_ratio 构造 target_mask，并只在 masked positions 上统计。
2. 是 evaluation bug，还是 GPT-2 draft context 导致 MDLM 失效？standard recovery 明显强于 draft-context recovery，主要问题是 GPT-2 draft context 误导 MDLM。
3. eval_steps 从 200 增加到 1000 后，结果是否稳定？本次配置 eval_steps=1000；稳定性需和 200-step 旧结果对比，重点看 Top1/Top5 排序是否保持。
4. gpt2_mdlm_refine 是否稳定优于 gpt2_only？是。最佳 refine Top5 gain=0.022615，Top1 gain=0.022611。
5. uncertainty selection 是否稳定优于 random refine？是。同 ratio 下 Top5 gain=0.062358。
6. 下一步是否应该加入 accept gate，而不是直接替换 GPT-2 token？是。best correction_rate=0.142121，regression_rate=0.731052，net_correction=-0.588931。
