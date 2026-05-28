# Draft-aware LoRA Fine-tuning for MDLM

本项目用于验证一个核心假设：

端侧 GPT-2 生成的 draft context 可以作为一种结构化训练噪声源，用 LoRA 微调边侧 pretrained MDLM 后，MDLM 会更擅长在 GPT-2 draft context 中恢复真实 clean tokens，并进一步提升 GPT-2 draft refinement。

当前实验不训练 GPT-2，不全量训练 MDLM，也不依赖额外 MDLM checkpoint。所有实验都直接加载本地 pretrained MDLM，然后只训练 LoRA 或 accept gate。

## 1. 本地模型路径

需要确认以下路径存在：

```text
/mnt/data/enzeyu/hf_downloads/models/gpt2
/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
```

配置中默认使用：

```yaml
tokenizer_name: /mnt/data/enzeyu/hf_downloads/models/gpt2
device_model_name_or_path: /mnt/data/enzeyu/hf_downloads/models/gpt2
edge_model_name_or_path: /mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
pretrained_edge_path: /mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
mdlm_ckpt: null
```

## 2. 推荐验证顺序

建议按下面顺序跑：

1. 训练 random-mask LoRA baseline
2. 训练 draft-aware LoRA
3. 比较两种 LoRA 训练噪声
4. 做 block-size ablation
5. 训练 accept gate
6. 评估最终 GPT-2 + LoRA-MDLM + gate
7. 生成总报告

## 3. 训练 random-mask LoRA baseline

这个 baseline 对应传统 MDLM 风格训练：

clean context + random mask -> 恢复被随机 mask 的 clean tokens。

运行：

```bash
python train_draft_aware_lora.py \
  --config configs/wikitext2_random_mask_lora.yaml
```

输出目录：

```text
results/wikitext2_random_mask_lora/
```

关键输出：

```text
results/wikitext2_random_mask_lora/lora_adapter/
results/wikitext2_random_mask_lora/train_metrics.csv
results/wikitext2_random_mask_lora/eval_metrics.csv
```

## 4. 训练 draft-aware LoRA

这是当前方法：

GPT-2 draft context + uncertain block mask -> 恢复 GPT-2 低置信 block 中的真实 clean tokens。

运行：

```bash
python train_draft_aware_lora.py \
  --config configs/wikitext2_draft_aware_lora.yaml
```

输出目录：

```text
results/wikitext2_draft_aware_lora/
```

关键输出：

```text
results/wikitext2_draft_aware_lora/lora_adapter/
results/wikitext2_draft_aware_lora/train_metrics.csv
results/wikitext2_draft_aware_lora/eval_metrics.csv
```

## 5. 比较 random-mask LoRA 和 draft-aware LoRA

两种 LoRA 都训练完成后运行：

```bash
python run_lora_training_mode_comparison.py \
  --eval_config configs/wikitext2_draft_aware_lora.yaml
```

输出目录：

```text
results/lora_training_mode_comparison/
```

重点看：

```text
results/lora_training_mode_comparison/training_mode_comparison_summary.md
```

核心指标：

```text
Draft Top1
Draft Top5
Correction
Regression
Net
```

判断标准：

如果 `draft_aware_lora_mdlm` 的 `Draft Top1` / `Draft Top5` 明显高于 `random_mask_lora_mdlm`，说明 GPT-2 draft context 作为结构化训练噪声源是有效的。

## 6. Block-size ablation

用于比较 token-level 和 block-level draft-aware training。

运行：

```bash
python run_block_size_ablation.py \
  --base_config configs/wikitext2_draft_aware_lora.yaml \
  --block_sizes 1 2 4 8
```

输出目录：

```text
results/block_size_ablation/
```

重点看：

```text
results/block_size_ablation/block_size_ablation_summary.md
```

核心问题：

1. `block_size=4` 是否优于 `block_size=1`
2. block-level training 是否缓解 GPT-2 draft context 对 MDLM 的误导
3. `block_size=8` 是否因为 block 太大导致任务变难
4. 当前最优 block size 是多少

## 7. 训练 accept gate

如果 LoRA-MDLM 无条件替换 GPT-2 token，可能 correction 增加，但 regression 也很高。accept gate 的目标是只接受有收益的 MDLM refinement。

运行：

```bash
python train_accept_gate.py \
  --config configs/wikitext2_accept_gate.yaml
```

输出：

```text
results/accept_gate/learned_gate.pt
results/accept_gate/accept_gate_train_metrics.csv
results/accept_gate/accept_gate_train_metrics.json
```

训练目标使用 utility-aware weighted BCE：

```yaml
positive_weight: 1.0
negative_weight: 3.0
```

也就是更重惩罚 harmful regression。

## 8. 评估 accept gate 和最终系统

运行：

```bash
python eval_accept_gate.py \
  --config configs/wikitext2_accept_gate.yaml
```

输出：

```text
results/accept_gate/accept_gate_eval.csv
results/accept_gate/accept_gate_eval.json
results/accept_gate/accept_gate_summary.md
results/accept_gate/accept_gate_best_config.json
```

重点比较这些模式：

```text
gpt2_only
pretrained_mdlm_refine
random_mask_lora_refine
draft_aware_lora_refine_no_gate
draft_aware_lora_refine_with_rule_gate
draft_aware_lora_refine_with_learned_gate
```

最终目标：

```text
draft_aware_lora_refine_with_learned_gate 的 Top1 / Top5 > gpt2_only
draft_aware_lora_refine_with_learned_gate 的 regression_rate < draft_aware_lora_refine_no_gate
draft_aware_lora_refine_with_learned_gate 的 net_correction 接近 0 或转正
```

## 9. 生成总报告

前三组实验跑完后运行：

```bash
python analyze_draft_aware_lora_results.py
```

输出：

```text
results/draft_aware_lora_final_report.md
```

报告会汇总：

1. random-mask LoRA vs draft-aware LoRA
2. block-size ablation
3. accept gate 是否降低 regression
4. 最终 GPT-2 + draft-aware LoRA-MDLM + gate 是否超过 GPT-2-only

## 10. 指标解释

`Standard Top1 / Standard Top5`：
MDLM 在标准 random mask context 下恢复 token 的准确率。

`Draft Top1 / Draft Top5`：
MDLM 在 GPT-2 draft context 下恢复被 mask token 的准确率。这是验证 draft-aware LoRA 的核心指标。

`PPL`：
在评估 token 上的 perplexity，越低越好。

`Correction`：
GPT-2 原本预测错，MDLM refinement 后预测对的比例。

`Regression`：
GPT-2 原本预测对，MDLM refinement 后改错的比例。

`Net`：

```text
Correction - Regression
```

越高越好。当前阶段如果能接近 0 或转正，说明 refinement 不再明显伤害 GPT-2。

`Selected Ratio`：
被选择进入 MDLM refinement 的 token 比例。

`Accepted Ratio`：
gate 最终接受 MDLM refinement 的比例。

`Gate Accuracy`：
gate 在 correction / regression utility label 上的判断准确率。

## 11. 快速 sanity check

如果只是确认脚本能启动，可以临时跑很小步数：

```bash
python train_draft_aware_lora.py \
  --config configs/wikitext2_random_mask_lora.yaml \
  --train_steps 1
```

```bash
python train_draft_aware_lora.py \
  --config configs/wikitext2_draft_aware_lora.yaml \
  --train_steps 1
```

```bash
python eval_accept_gate.py \
  --config configs/wikitext2_accept_gate.yaml \
  --eval_steps 1
```

正式验证时不要使用这些小步数结果作为结论。

## 12. 当前最重要的结论判断

验证方法是否成立，主要看三件事：

1. draft-aware LoRA 是否在 GPT-2 draft context 下优于 random-mask LoRA
2. block-level draft-aware training 是否优于 token-level training
3. accept gate 是否降低 regression，并让最终系统超过 GPT-2-only

如果这三点同时成立，就能支持：

端侧 AR draft 可以作为扩散语言模型训练噪声源，提升边侧 MDLM 对端侧草稿上下文的修正能力。
