# DART：面向边缘 MDLM 的草稿感知 LoRA 微调

本项目实现了一个设备端 GPT-2 草稿模型与边缘端 MDLM 协同的文本 refinement 框架。核心目标是：设备端先用冻结的 GPT-2 生成自回归草稿，边缘端冻结 MDLM 主干，只训练少量 LoRA / adapter 参数和接受门控，对 GPT-2 低置信 token 做选择性修正。

核心流程：

```text
设备端 GPT-2 草稿
-> 基于置信度 / 熵 / margin 选择低置信 token
-> 边缘端 MDLM 进行 draft-aware LoRA / adapter refinement
-> 接受门控判断是否采纳 MDLM 候选
-> 输出选择性修正后的最终序列
```

## 方法概览

### 1. AR 草稿诱导扰动

GPT-2 以 teacher-forcing 方式产生逐 token 草稿预测，同时计算 token confidence、entropy 和 margin。系统根据这些不确定性信号选择需要边缘端进一步 refinement 的 token。

### 2. Draft-aware LoRA / Adapter

边缘 MDLM 接收 GPT-2 草稿上下文，并只在低置信位置进行 mask / renoise。MDLM 主干保持冻结，训练 LoRA 或 draft refine adapter，使其适配真实 GPT-2 草稿带来的结构化噪声，而不是只适配随机 mask 噪声。

### 3. Utility-aware Accept Gate

接受门控用于判断是否采纳 MDLM 的修正候选。正例是 GPT-2 原本错误且 MDLM 修正正确的 token；负例是 GPT-2 原本正确但 MDLM 会改错的 token。门控的目标是降低 regression，同时保留有效 correction。

## 关键约束

- GPT-2 作为设备端 draft model，默认冻结，只负责生成草稿、不确定性分数和低置信 mask。
- MDLM 主干默认冻结。
- 可训练部分主要是 LoRA / draft adapter 和 accept gate。
- 训练噪声来自 GPT-2 draft context，而不是单纯随机 mask。
- 主要评价目标是 GPT-2 草稿上下文下的协同 refinement 效果。

## 本地模型与数据

项目配置默认使用本地 Hugging Face 缓存：

```text
GPT-2 设备端模型:
/mnt/data/enzeyu/hf_downloads/models/gpt2

MDLM 边缘端模型:
/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt

数据缓存:
/mnt/data/enzeyu/hf_downloads/datasets
```

默认数据集是 `wikitext-2`，主要配置位于 `configs/`。

## 主要代码结构

```text
train_draft_aware_lora.py      # 训练 random-mask / draft-aware LoRA
train_accept_gate.py           # 训练 learned accept gate
eval_final_refinement.py       # DART 最终 refinement 评估
run_block_size_ablation.py     # block size 消融
analyze_results.py             # 汇总实验结果并生成 final_report.md

draft_utils.py                 # GPT-2 / MDLM 加载与模型接口校验
data_real.py                   # WikiText-2 数据加载与 mask 构造
lora_utils.py                  # LoRA / 参数冻结相关工具
metrics.py                     # 指标、计时、显存统计等工具
```

## 推荐运行顺序

### 1. 训练随机 mask LoRA baseline

```bash
python train_draft_aware_lora.py \
  --config configs/wikitext2_random_mask_lora.yaml
```

### 2. 训练 draft-aware LoRA

```bash
python train_draft_aware_lora.py \
  --config configs/wikitext2_draft_aware_lora.yaml
```

### 3. 训练接受门控

```bash
python train_accept_gate.py \
  --config configs/wikitext2_accept_gate.yaml
```

### 4. 最终评估

```bash
python eval_final_refinement.py \
  --config configs/wikitext2_dart_final.yaml
```

### 5. block size 消融

```bash
python run_block_size_ablation.py \
  --base_config configs/wikitext2_draft_aware_lora.yaml \
  --block_sizes 1 2 4 8
```

### 6. 生成结果报告

```bash
python analyze_results.py
```

输出文件：

```text
results/final_report.md
```

## 指标解释

- `Standard Top1 / Top5`：标准 clean / random-mask MDLM 评估指标，主要作为 sanity check。
- `Draft Top1 / Top5`：GPT-2 draft context 下的 refinement 准确率，是本项目核心指标。
- `Draft PPL / PPL`：草稿上下文 refinement 的困惑度。
- `Correction`：被选中的 GPT-2 错误 token 中，被 MDLM 修正正确的比例。
- `Regression`：被选中的 GPT-2 正确 token 中，被 MDLM 改错的比例。
- `Net Correction`：`Correction - Regression`。
- `Accepted Ratio / Accept Rate`：接受门控采纳 refinement 候选的比例。
- `Selected Token Ratio / Refine Ratio`：有效 token 中被选择进入 refinement 的比例。

## 当前主要结果总结

### Random-mask LoRA vs Draft-aware LoRA

| 模型 | 训练噪声 | Standard Top1 | Draft Top1 | Draft PPL | Correction | Regression | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrained_mdlm | pretrained | 0.5191 | 0.0964 | 2484.4637 | 0.0732 | 0.7646 | -0.6915 |
| random_mask_lora_mdlm | random_mask | 0.6372 | 0.1317 | 478.7520 | 0.1049 | 0.7081 | -0.6032 |
| draft_aware_lora_mdlm | draft_aware | 0.5233 | 0.2036 | 184.5030 | 0.1785 | 0.6484 | -0.4699 |

结论：在 GPT-2 草稿上下文下，draft-aware LoRA 明显优于 random-mask LoRA。Draft Top1 从 `0.1317` 提升到 `0.2036`，Draft PPL 从 `478.7520` 降到 `184.5030`。

### Block-size 消融

| Block Size | Draft Top1 | Draft Top5 | PPL | Correction | Regression | Net | Selected Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3282 | 0.5153 | 53.2684 | 0.3169 | 0.5008 | -0.1840 | 0.2042 |
| 2 | 0.2804 | 0.4687 | 89.1048 | 0.2557 | 0.5226 | -0.2669 | 0.1995 |
| 4 | 0.2037 | 0.3790 | 184.5060 | 0.1789 | 0.6477 | -0.4688 | 0.2164 |
| 8 | 0.1404 | 0.2942 | 347.2426 | 0.1145 | 0.7409 | -0.6264 | 0.2481 |

结论：当前 `block_size=1` 最好；固定大 block 会降低 GPT-2 草稿上下文下的 refinement 效果。

### 接受门控

| 模式 | Gate Threshold | Top1 | Top5 | PPL | Correction | Regression | Net | Accepted Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| draft_aware_lora_refine_no_gate | - | 0.3169 | 0.5106 | 65.9995 | 0.1526 | 0.2259 | -0.0733 | 1.0000 |
| learned_gate | 0.3000 | 0.3171 | 0.5104 | 65.7937 | 0.1479 | 0.1900 | -0.0421 | 0.8747 |
| learned_gate | 0.4000 | 0.3160 | 0.5100 | 65.6311 | 0.1325 | 0.1336 | -0.0011 | 0.5981 |
| learned_gate | 0.5000 | 0.3134 | 0.5092 | 65.7455 | 0.1083 | 0.0734 | 0.0349 | 0.3649 |
| learned_gate | 0.6000 | 0.3107 | 0.5085 | 65.9732 | 0.0886 | 0.0428 | 0.0458 | 0.2332 |
| learned_gate | 0.7000 | 0.3083 | 0.5077 | 66.2304 | 0.0727 | 0.0238 | 0.0489 | 0.1526 |

结论：learned accept gate 能显著降低 regression。阈值提高时采纳比例下降，regression 也下降；阈值 `0.4` 附近在 Top1、PPL 和 regression 之间较均衡。

### 最终对比

| 模式 | Top1 | Top5 | PPL | Correction | Regression | Net | Accepted Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2_only | 0.2956 | 0.5027 | 68.1806 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| pretrained_mdlm_refine | 0.3007 | 0.5056 | 68.1526 | 0.0857 | 0.3469 | -0.2612 | 1.0000 |
| random_mask_lora_refine | 0.3077 | 0.5070 | 67.3102 | 0.1119 | 0.2785 | -0.1665 | 1.0000 |
| draft_aware_lora_refine_no_gate | 0.3169 | 0.5106 | 65.9995 | 0.1526 | 0.2259 | -0.0733 | 1.0000 |
| draft_aware_lora_refine_with_learned_gate, threshold=0.4 | 0.3160 | 0.5100 | 65.6311 | 0.1325 | 0.1336 | -0.0011 | 0.5981 |

结论：`GPT-2 + draft-aware LoRA-MDLM + learned gate` 优于 GPT-2-only。最终 Top1 / Top5 从 `0.2956 / 0.5027` 提升到约 `0.3160 / 0.5100`，同时 regression 明显低于无门控版本。

## 当前推荐配置

```text
Device model: /mnt/data/enzeyu/hf_downloads/models/gpt2
Edge model: /mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
LoRA mode: draft_aware
block_size: 1
refine_ratio: 0.2
gate_threshold: 0.4
trainable modules: LoRA / draft adapter + accept gate
```

## 总结

1. 预训练 MDLM 在标准 random-mask denoising 下较强，但直接放到 GPT-2 草稿上下文中会明显退化。
2. GPT-2 草稿诱导的结构化扰动是有效训练信号。
3. Draft-aware LoRA / adapter 在草稿上下文下优于 random-mask LoRA。
4. 当前 `block_size=1` 是最佳设置，大 block 不适合作为主配置。
5. Learned accept gate 可以降低把 GPT-2 正确 token 改错的 regression。
6. 完整系统相较 GPT-2-only 有稳定提升，说明边缘 MDLM refinement 对设备端草稿有实际增益。

## 论文化前最需要解决的问题

当前最大问题不是实验不够多，而是最终提升幅度还偏小：`GPT-2-only` Top1 为 `0.2956`，完整系统约 `0.3160`，绝对提升只有约 `+2.0%`。如果要写成论文，优先目标应该是把这个提升做大，而不是先补很多外围实验。

下一步重点：

1. 优先提高 correction，同时控制 regression：改进低置信 token selector、refine ratio 和 accept gate，让 MDLM 少改对的 token、多修错的 token。
2. 重点调 `block_size=1` 下的主配置：更长训练、更合适 LoRA rank / learning rate / refine ratio，而不是继续投入大 block。
3. 做 confidence / entropy 分桶分析，找出 GPT-2 哪些错误最容易被 MDLM 修正，再把 selector 对准这些位置。
4. 如果提升仍然小，尝试更强的 draft-aware 训练目标或更强的 draft/refine 模型组合；否则论文主张会比较弱。
5. 只有当最终提升明显扩大后，再补多 seed、更多数据集、效率指标和完整消融。
