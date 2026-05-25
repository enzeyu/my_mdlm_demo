# AR-Guided Diffusion Verification

- Edge MDLM: `/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt`
- Device GPT-2: `/mnt/data/enzeyu/hf_downloads/models/gpt2`
- Uncertainty score: `entropy`
- Hard token ratio: `0.3`
- Device top-k: `20`
- Assist alpha: `0.5`

| Mode | Loss | PPL | Top1 | Top5 | Hard Top1 | Hard Top5 | Correction | Regression | Query Ratio | Comm MB | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_only | 2.7707 | 15.9699 | 0.5167 | 0.7369 | 0.1728 | 0.3969 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.0248 |
| gpt2_only | 6.9235 | 1015.8877 | 0.1699 | 0.3045 | 0.0385 | 0.0965 | 0.0000 | 0.0000 | 1.0000 | 4.633179 | 0.3597 |
| random_assist | 2.7710 | 15.9741 | 0.5167 | 0.7368 | 0.1728 | 0.3967 | 0.0000 | 0.0000 | 0.3032 | 1.404724 | 0.1354 |
| gpt2_assist | 2.8204 | 16.7844 | 0.5108 | 0.7271 | 0.1532 | 0.3647 | 0.0094 | 0.0203 | 0.3032 | 1.404724 | 0.1354 |

| Comparison | Top1 Gain | Top5 Gain | Hard Top1 Gain | Hard Top5 Gain | Extra Comm MB | Extra Latency | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| gpt2_assist vs edge_only | -0.005961 | -0.009781 | -0.019661 | -0.032262 | 1.404724 | 0.110654 | not_clear |
| gpt2_assist vs random_assist | -0.005961 | -0.009715 | -0.019661 | -0.032044 | 0.000000 | 0.000000 | not_clear |

## Questions

1. gpt2_assist 是否优于 edge_only？否，当前指标不足以支持。Top5 gain=-0.009781，Hard Top5 gain=-0.032262，PPL delta=0.814518。
2. gpt2_assist 是否优于 random_assist？否。Top5 gain=-0.009715，Hard Top5 gain=-0.032044。
3. 提升是否主要发生在 hard tokens 上？是。
4. correction_rate 是否高于 regression_rate？否。correction=0.009404，regression=0.020331。
5. 通信开销是否可接受？需要结合系统预算判断。当前 assist 通信量约 1.404724 MB。
6. 是否支持“端侧 AR 小模型辅助边侧扩散模型”的研究假设？暂不支持，证据还不充分。
7. 如果效果不明显，可能原因包括：GPT-2 只能看左上下文而 MDLM 使用双向 masked context；GPT-2 top-k 未覆盖真实 token；log-prob bias 的 alpha 未调优；候选只改变 logits 不改变 MDLM hidden state；WikiText masked token 中存在需要右上下文的歧义；GPT-2 与 MDLM 训练目标和数据分布不同。
