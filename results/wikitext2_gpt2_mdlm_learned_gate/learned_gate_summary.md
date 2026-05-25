# Learned Accept Gate Evaluation

- Device GPT-2: `/mnt/data/enzeyu/hf_downloads/models/gpt2`
- Edge MDLM: `/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt`
- refine_window: `0`
- refine_ratios: `[0.2, 0.3]`
- candidate_top_k: `20`

## Results

| Mode | Ratio | Top1 | Top5 | PPL | Correction | Regression | Net | Accepted | Gate Acc | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_only | 0.00 | 0.2956 | 0.5026 | 68.1912 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mdlm_only | 1.00 | 0.5217 | 0.7420 | 14.7485 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_refine | 0.20 | 0.3027 | 0.5067 | 67.8336 | 0.1385 | 0.2094 | -0.0709 | 1.0000 | 0.0000 | 0.6538 |
| candidate_rerank_no_gate | 0.20 | 0.3098 | 0.5098 | 66.6783 | 0.0974 | 0.3449 | -0.2474 | 1.0000 | 0.0000 | 0.3373 |
| candidate_rerank_rule_gate | 0.20 | 0.3026 | 0.5064 | 67.3023 | 0.0424 | 0.0773 | -0.0349 | 0.2879 | 0.4983 | 0.3373 |
| candidate_rerank_learned_gate | 0.20 | 0.3098 | 0.5098 | 66.6542 | 0.0970 | 0.3342 | -0.2372 | 0.9740 | 0.8178 | 0.3373 |
| random_refine | 0.30 | 0.3038 | 0.5081 | 67.9943 | 0.1311 | 0.2189 | -0.0878 | 1.0000 | 0.0000 | 0.6567 |
| candidate_rerank_no_gate | 0.30 | 0.3176 | 0.5133 | 65.9331 | 0.1099 | 0.3685 | -0.2586 | 1.0000 | 0.0000 | 0.3984 |
| candidate_rerank_rule_gate | 0.30 | 0.3074 | 0.5084 | 66.6774 | 0.0498 | 0.0865 | -0.0368 | 0.2793 | 0.5207 | 0.3984 |
| candidate_rerank_learned_gate | 0.30 | 0.3177 | 0.5132 | 65.8958 | 0.1095 | 0.3597 | -0.2502 | 0.9776 | 0.7847 | 0.3984 |

## Answers

1. learned gate 是否降低 regression？是。
2. learned gate 是否保持或提升 Top1/Top5？是。
3. learned gate 是否优于 rule-based gate？否或证据不足。
4. learned gate 是否优于 no-gate candidate rerank？是。
5. candidate coverage 是否足够？否或需要尝试 candidate_top_k=50；coverage=0.398444。
6. 当前结果是否支持“端侧 GPT-2 draft + 边侧 MDLM verifier/refiner”的研究假设？暂不充分支持。
