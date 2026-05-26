# 当前方法流程总结：Diffusion-Assisted Autoregressive Refinement

## 1. 方法定位与研究目标

当前代码实现的是一种“端侧自回归草稿生成 + 边侧扩散语言模型验证/修正”的两阶段文本预测框架。系统将本地 GPT-2 small 视为端侧 autoregressive draft model，将预训练 MDLM masked diffusion language model 视为边侧 verifier/refiner。核心思想不是重新训练 GPT-2 或 MDLM，而是在 GPT-2 低置信度位置上调用 MDLM 进行候选重排序，并用一个轻量级 learned accept gate 决定是否接受 MDLM 辅助后的修正。

从论文角度看，该方法可以表述为：在资源受限端侧模型已给出初始 token 预测的前提下，利用边侧扩散式双向上下文建模能力对不确定位置进行局部验证，从而在尽量限制额外计算范围的同时提升预测质量或降低错误修正带来的 regression 风险。

当前实验的关键假设是：

> GPT-2 的低置信度位置包含较高比例的潜在错误；MDLM 可在这些位置上提供互补的上下文证据；学习式接收门控能够过滤有害修正，使端边协同 refinement 相比无门控或规则门控更稳定。

## 2. 数据与模型设置

### 2.1 数据处理

数据集由 `data_real.py` 构建，当前配置使用 WikiText-2。文本首先通过 GPT-2 tokenizer 编码，并在必要时添加 `[MASK]` token。随后所有 token 被拼接并切分为固定长度 block，默认序列长度为 `max_length=128`。训练和验证 DataLoader 分别返回形状为 `[batch_size, max_length]` 的 token block。

MDLM-only baseline 使用随机 mask corruption：对非 padding token 按 `mask_ratio=0.15` 进行遮蔽，并保证每条序列至少有一个被 mask 的监督位置。该部分主要用于评估预训练 MDLM 的 masked-token 恢复能力。

### 2.2 端侧 Draft Model：GPT-2

端侧模型由 `refine_utils.load_gpt2` 加载，当前路径为：

```text
/mnt/data/enzeyu/hf_downloads/models/gpt2
```

GPT-2 在所有训练和评测流程中均被冻结。代码采用 teacher-forced 方式构造每个位置的左上下文预测：输入序列整体右移一位，首位填入 EOS/PAD token，然后得到每个位置的 next-token logits。设输入 token 序列为 \(x_{1:L}\)，GPT-2 对第 \(i\) 个位置输出：

\[
p_{\mathrm{AR}}(x_i \mid x_{<i})
\]

其 top-1 预测形成 draft token：

\[
\hat{x}^{\mathrm{GPT}}_i = \arg\max_v p_{\mathrm{AR}}(v \mid x_{<i})
\]

### 2.3 边侧 Verifier/Refiner：MDLM

边侧模型由 `models_mdlm_wrapper.py` 加载，当前配置优先使用本地预训练 MDLM：

```text
/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
```

MDLM 同样被冻结。代码会检查 tokenizer 兼容性，并在 GPT-2 tokenizer 额外加入 `[MASK]` 后调整 MDLM 词表表面，使 MDLM 能处理 mask token。MDLM forward 接收 masked draft 和 diffusion timestep，输出每个位置的 masked-token logits：

\[
p_{\mathrm{MDLM}}(x_i \mid \tilde{x}, t)
\]

其中 \(\tilde{x}\) 是将待 refinement 位置替换为 `[MASK]` 后的 draft 序列。

## 3. 整体流程

当前主流程可以概括为以下步骤：

1. GPT-2 对完整序列进行 teacher-forced token 预测，得到每个位置的 logits、top-1 draft 以及置信度。
2. 根据 GPT-2 的不确定性分数选择待修正位置，默认使用 inverse confidence：

\[
u_i = 1 - \max_v p_{\mathrm{AR}}(v \mid x_{<i})
\]

3. 按 `refine_ratio` 选择不确定性最高的一部分有效 token。当前评测使用 `refine_ratios=[0.2, 0.3]`，训练 gate 时默认使用 `refine_ratio=0.3`。
4. 将选中位置在 GPT-2 draft 中替换为 `[MASK]`，得到 MDLM 输入。当前 `refine_window=0`，即只 mask 被选中的 token，不额外 mask 邻域。
5. GPT-2 在每个选中位置提供 top-k 候选集合，当前 `candidate_top_k=20`。
6. MDLM 对 masked draft 进行一次 forward，给出选中位置上的 token 分布。
7. 在 GPT-2 top-k 候选集合内部融合 GPT-2 和 MDLM 分数，得到 rerank 后的候选 token。
8. learned accept gate 根据 GPT-2、MDLM 和 rerank 特征判断是否接受 rerank token。
9. 若门控接受，则用 rerank token 替换原 GPT-2 top-1；否则保留 GPT-2 原预测。

整体路径可写为：

```text
GPT-2 draft
-> uncertainty-based position selection
-> GPT-2 top-k candidate extraction
-> mask selected draft positions
-> MDLM masked-token scoring
-> candidate reranking within GPT-2 top-k
-> learned accept gate
-> final refined prediction
```

## 4. Candidate Reranking 机制

对每个待 refinement 位置，系统只在 GPT-2 的 top-k 候选集合内搜索。设候选集合为：

\[
\mathcal{C}_i = \operatorname{TopK}(p_{\mathrm{AR}}(\cdot \mid x_{<i}))
\]

对任一候选 token \(c \in \mathcal{C}_i\)，代码使用 GPT-2 log-probability 与 MDLM log-probability 的线性组合进行打分：

\[
s_i(c) =
\lambda_{\mathrm{GPT}} \log p_{\mathrm{AR}}(c \mid x_{<i})
+ \lambda_{\mathrm{MDLM}} \log p_{\mathrm{MDLM}}(c \mid \tilde{x}, t)
\]

当前默认：

```yaml
lambda_gpt2: 0.5
lambda_mdlm: 0.5
candidate_top_k: 20
```

最终 rerank token 为：

\[
\hat{x}^{\mathrm{rerank}}_i = \arg\max_{c \in \mathcal{C}_i} s_i(c)
\]

该设计有两个重要约束：第一，MDLM 只负责重排 GPT-2 已提出的候选，而不是在完整词表上自由替换；第二，candidate coverage 会成为上限因素。如果真实 token 不在 GPT-2 top-k 内，即使 MDLM 能在完整词表中恢复正确 token，当前 rerank 模块也无法选中它。

## 5. Learned Accept Gate

### 5.1 门控结构

`refine_gate.py` 中的 `AcceptGateMLP` 是一个小型 MLP，输入维度为 10，默认隐藏层大小为 64，层数为 2，输出一个标量 logit。经 sigmoid 后得到接受 rerank token 的概率：

\[
a_i = \sigma(g_\theta(f_i))
\]

若 \(a_i > \tau\)，则接受 rerank token；否则保留 GPT-2 token。当前阈值为：

```yaml
accept_threshold: 0.5
```

### 5.2 门控特征

当前 gate 使用以下 10 个特征：

1. `gpt2_confidence`：GPT-2 top-1 概率。
2. `gpt2_entropy`：GPT-2 归一化熵。
3. `mdlm_confidence`：MDLM top-1 概率。
4. `mdlm_margin`：MDLM top-1 与 top-2 概率差。
5. `rerank_score_gap`：rerank 第一名和第二名融合分数差。
6. `gpt2_mdlm_agree`：rerank token 是否等于 GPT-2 top-1 token。
7. `gpt2_top1_in_mdlm_topk`：GPT-2 top-1 是否出现在 MDLM top-k 内。
8. `rerank_candidate_rank`：rerank token 在 GPT-2 top-k 候选中的归一化 rank。
9. `gpt2_candidate_logprob`：rerank token 的 GPT-2 log-probability。
10. `mdlm_candidate_logprob`：rerank token 的 MDLM log-probability。

这些特征共同描述三类信息：GPT-2 自身不确定性、MDLM 对候选的置信程度，以及 rerank 决策的稳定性。

### 5.3 监督标签构造

训练 gate 时，GPT-2 和 MDLM 均被冻结，只优化 MLP 参数。标签来自 ground truth token，但只在 GPT-2 与 rerank 的正确性发生分歧时构造训练样本：

\[
y_i =
\begin{cases}
1, & \hat{x}^{\mathrm{rerank}}_i = x_i \ \text{and}\ \hat{x}^{\mathrm{GPT}}_i \ne x_i \\
0, & \hat{x}^{\mathrm{GPT}}_i = x_i \ \text{and}\ \hat{x}^{\mathrm{rerank}}_i \ne x_i
\end{cases}
\]

也就是说，gate 只学习“何时接受会带来真实纠错”以及“何时接受会造成 regression”。当 GPT-2 和 rerank 同对或同错时，该位置不参与 gate 的二分类训练。

训练损失为 binary cross entropy with logits：

\[
\mathcal{L}_{\mathrm{gate}}
=
- y_i \log a_i - (1-y_i)\log(1-a_i)
\]

当前训练脚本 `train_refine_gate.py` 只训练 learned accept gate，并保存：

```text
results/wikitext2_gpt2_mdlm_learned_gate/learned_gate.pt
results/wikitext2_gpt2_mdlm_learned_gate/gate_train_metrics.csv
results/wikitext2_gpt2_mdlm_learned_gate/gate_train_metrics.json
results/wikitext2_gpt2_mdlm_learned_gate/best_gate_config.json
```

## 6. 推理与评测模式

`eval_refine_gate.py` 当前评估以下模式：

1. `gpt2_only`：只使用 GPT-2 左上下文预测。
2. `mdlm_only`：随机 mask 输入后评估 MDLM masked-token 恢复能力。
3. `random_refine`：随机选择 refinement 位置，并始终接受 candidate rerank。
4. `candidate_rerank_no_gate`：按 GPT-2 不确定性选择位置，rerank 后无条件接受。
5. `candidate_rerank_rule_gate`：使用规则阈值判断是否接受。
6. `candidate_rerank_learned_gate`：使用训练得到的 MLP gate 判断是否接受。

规则门控当前使用三个阈值：

```yaml
tau_gpt2: 0.5
tau_mdlm: 0.4
tau_margin: 0.05
```

其逻辑是：只有当 GPT-2 置信度低、MDLM 置信度高，且 MDLM 相比 GPT-2 有足够 margin 时才接受修正。

## 7. 评价指标

当前评测输出包括：

- `top1`：最终 logits top-1 是否命中真实 token。
- `top5`：真实 token 是否位于最终 logits top-5。
- `ppl`：由交叉熵计算的 perplexity。
- `correction_rate`：在被选中的 GPT-2 错误位置中，被 refinement 修正的比例。
- `regression_rate`：在被选中的 GPT-2 正确位置中，被 refinement 改错的比例。
- `net_correction`：`correction_rate - regression_rate`。
- `accepted_ratio`：被选中位置中最终接受修正的比例。
- `gate_accuracy`：gate 在可监督分歧样本上的二分类准确率。
- `error_detection_precision`：选中位置中 GPT-2 错误所占比例。
- `error_detection_recall`：GPT-2 全部错误中被选中的比例。
- `candidate_coverage`：真实 token 是否出现在 GPT-2 top-k 候选集合中。
- `latency` 与 `tokens_per_sec`：批平均延迟和吞吐。

论文中尤其建议突出 `correction_rate`、`regression_rate`、`net_correction` 和 `candidate_coverage`。其中 `candidate_coverage` 是当前候选受限 rerank 的理论上限指标。

## 8. 当前实验结果概览

当前已有评测结果保存在：

```text
results/wikitext2_gpt2_mdlm_learned_gate/learned_gate_summary.md
```

主要结果如下：

| Mode | Ratio | Top1 | Top5 | PPL | Correction | Regression | Net | Accepted | Gate Acc | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2_only | 0.00 | 0.2956 | 0.5026 | 68.1912 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mdlm_only | 1.00 | 0.5217 | 0.7420 | 14.7485 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| candidate_rerank_no_gate | 0.20 | 0.3098 | 0.5098 | 66.6783 | 0.0974 | 0.3449 | -0.2474 | 1.0000 | 0.0000 | 0.3373 |
| candidate_rerank_rule_gate | 0.20 | 0.3026 | 0.5064 | 67.3023 | 0.0424 | 0.0773 | -0.0349 | 0.2879 | 0.4983 | 0.3373 |
| candidate_rerank_learned_gate | 0.20 | 0.3098 | 0.5098 | 66.6542 | 0.0970 | 0.3342 | -0.2372 | 0.9740 | 0.8178 | 0.3373 |
| candidate_rerank_no_gate | 0.30 | 0.3176 | 0.5133 | 65.9331 | 0.1099 | 0.3685 | -0.2586 | 1.0000 | 0.0000 | 0.3984 |
| candidate_rerank_rule_gate | 0.30 | 0.3074 | 0.5084 | 66.6774 | 0.0498 | 0.0865 | -0.0368 | 0.2793 | 0.5207 | 0.3984 |
| candidate_rerank_learned_gate | 0.30 | 0.3177 | 0.5132 | 65.8958 | 0.1095 | 0.3597 | -0.2502 | 0.9776 | 0.7847 | 0.3984 |

从这些结果可以得到几条阶段性结论：

1. 相比 GPT-2-only，candidate rerank 系列通常提升 Top1/Top5 并降低 PPL，说明 MDLM 分数对 GPT-2 候选排序具有一定补充作用。
2. 无门控 rerank 虽能提升整体 top-k 指标，但 regression_rate 很高，说明直接接受 MDLM 辅助 rerank 会频繁破坏 GPT-2 原本正确的位置。
3. rule gate 显著降低 regression_rate，但 accepted_ratio 较低，也压缩了 correction_rate。
4. learned gate 当前 gate_accuracy 较高，但 accepted_ratio 接近无门控，导致 regression_rate 仍然偏高；它相对 no-gate 略有改善，但尚未充分解决 regression 问题。
5. `candidate_coverage` 在 `candidate_top_k=20` 下仅约 0.34-0.40，说明真实 token 经常不在 GPT-2 top-k 候选内，候选集合限制是当前上限瓶颈。

## 9. 可写入论文的方法描述

可以将当前方法命名为：

> Diffusion-Assisted Autoregressive Refinement with Learned Acceptance Gate

论文方法部分可按以下逻辑组织：

1. **Autoregressive Drafting**：端侧 GPT-2 以低延迟方式产生每个位置的初始分布与 top-1 draft。
2. **Uncertainty-Aware Token Selection**：根据 GPT-2 置信度选择最值得调用边侧模型的位置，控制边侧计算预算。
3. **Masked Diffusion Verification**：将 draft 中的待修正位置替换为 `[MASK]`，利用 MDLM 的双向上下文能力估计这些位置的 token 分布。
4. **Candidate-Constrained Reranking**：在 GPT-2 top-k 候选内融合 AR 与 MDLM log-probability，避免完全开放词表搜索带来的不稳定替换。
5. **Learned Acceptance Control**：训练轻量级 MLP 判断 rerank 是否应被接受，以降低错误修正导致的 regression。

论文中的方法图可以画成三段式：

```text
Device side:
  input context -> GPT-2 logits -> draft tokens + uncertainty + top-k candidates

Edge side:
  masked draft -> MDLM logits -> candidate reranking scores

Decision layer:
  rerank features -> accept gate -> final token prediction
```

## 10. 当前方法的局限与下一步实验建议

当前代码已经形成完整的训练和评测闭环，但从论文结果角度仍有几个明显问题需要后续实验支撑：

1. `candidate_coverage` 偏低，建议尝试 `candidate_top_k=50`，观察 coverage、Top1/Top5、latency 和 regression 的变化。
2. learned gate 当前过于倾向接受 rerank，建议调高 `accept_threshold` 或在训练中处理正负样本不平衡。
3. rule gate 虽然更保守，但 regression 明显更低，可作为 learned gate 的强 baseline。
4. 当前 `net_correction` 仍为负，说明 refinement 的有效纠错比例还不足以抵消 regression；论文结论需要谨慎表述为“初步验证候选重排序可提升整体预测指标，但可靠接收机制仍是关键挑战”。
5. 由于 MDLM-only 的 masked-token 指标明显强于 GPT-2-only，后续值得探索让 MDLM 候选不完全受限于 GPT-2 top-k，或引入 MDLM top-k 与 GPT-2 top-k 的并集。

## 11. 一句话总结

当前方法实现了一个端边协同的 refinement 框架：端侧 GPT-2 负责快速产生 draft 和候选，边侧 MDLM 对低置信度位置提供双向上下文验证，候选重排序模块融合两类模型分数，最后由 learned accept gate 控制是否接受修正；实验表明该框架具备提升 Top1/Top5 与降低 PPL 的潜力，但当前主要瓶颈在于候选覆盖率不足和修正 regression 偏高。
