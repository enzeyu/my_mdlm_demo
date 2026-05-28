# Block-size Ablation

| Block Size | Draft Top1 | Draft Top5 | PPL | Block EM | Correction | Regression | Net | Selected Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3282 | 0.5153 | 53.2684 | 0.3282 | 0.3169 | 0.5008 | -0.1840 | 0.2042 |
| 2 | 0.2804 | 0.4687 | 89.1048 | 0.1341 | 0.2557 | 0.5226 | -0.2669 | 0.1995 |
| 4 | 0.2037 | 0.3790 | 184.5060 | 0.0205 | 0.1789 | 0.6477 | -0.4688 | 0.2164 |
| 8 | 0.1404 | 0.2942 | 347.2426 | 0.0005 | 0.1145 | 0.7409 | -0.6264 | 0.2481 |

## Answers

1. block_size=4 是否优于 token-level block_size=1？否或证据不足。
2. block-level training 是否缓解 GPT-2 draft context 对 MDLM 的误导？是。
3. block_size 过大是否导致任务变难？是。
4. 当前最优 block_size 是多少？`1`。
