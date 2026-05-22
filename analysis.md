# WikiText-2 MDLM 边端协作实验分析

## 1. 实验目标与当前设置

本实验想验证的是：端侧小扩散语言模型先在 masked input 上产生 token 级细粒度中间表示，边缘侧较大的 MDLM 利用这个表示进行 denoising，从而在通信开销可控的前提下提升 masked token recovery 质量。

当前结果来自 `results/wikitext2_mdlm_medium`，核心设置如下：

- 后端模型：`mdlm`
- 边缘模型：预训练 MDLM，`pretrained_edge_loaded=True`
- 协作方式：`coarse_to_fine`
- 端侧细粒度表示维度：`coarse_dim=128`
- 边缘 hidden size：`768`
- 压缩比：`6.0x`
- 每 batch 通信量：`0.25 MB`
- 训练步数：`20000`
- 评估任务：单步 masked token recovery，不是多步 iterative diffusion sampling

这里的“细粒度空间”是 token-level latent space：每个 token 位置都传递一个 128 维表示，而不是只传句向量或全局条件。因此它具备细粒度位置条件能力，但通信量仍比传 768 维 edge hidden 小 6 倍。

## 2. 训练过程观察

训练曲线整体是收敛的。前 100 step 到最后 100 step 的均值变化如下：

| 指标 | 前 100 step 均值 | 后 100 step 均值 | 变化 |
| --- | ---: | ---: | ---: |
| edge loss | 2.2980 | 1.5365 | 下降 |
| device loss | 63.1021 | 7.9711 | 大幅下降 |
| align loss | 916.6542 | 225.8723 | 大幅下降 |
| distill loss | 60.8049 | 6.4066 | 大幅下降 |
| top-1 acc | 0.5571 | 0.6714 | 上升 |
| top-5 acc | 0.7474 | 0.8397 | 上升 |

最终训练 step 的主要指标：

- total loss：15.2612
- edge loss：1.5051
- device loss：7.9177
- align loss：230.1623
- distill loss：6.6445
- train top-1：0.6667
- train top-5：0.8491

这说明训练没有完全失败：端侧模型、边缘 denoising、端边对齐和蒸馏目标都在改善。尤其是 device loss 从 60 以上降到 8 左右，说明端侧小模型确实学到了一部分 masked recovery 能力。

但是，端侧模型仍然非常弱。最终评估中 `device_only` 的 top-1 只有 0.0327，perplexity 高达 2994.66。这意味着端侧细粒度表示虽然经过 alignment 约束，但其自身语言恢复能力和边缘预训练 MDLM 相比差距极大。

## 3. 验证集曲线

验证集每 1000 step 评估一次，共 20 个点。

最佳验证 loss 出现在 step 13000：

- val loss：1.5409
- val perplexity：4.6687
- val top-1：0.6713
- val top-5：0.8297

最佳验证 top-1 出现在 step 17000：

- val loss：1.5480
- val perplexity：4.7021
- val top-1：0.6752
- val top-5：0.8305

最终 step 20000：

- val loss：1.5501
- val perplexity：4.7120
- val top-1：0.6672
- val top-5：0.8333

结论是：模型在 13000 到 17000 step 附近已经达到较好状态，继续训练到 20000 step 没有显著提升 top-1，但 top-5 略有改善。整体验证曲线存在波动，不是单调下降。

## 4. 三种推理模式对比

最终 benchmark 结果如下：

| 模式 | loss | PPL | top-1 | top-5 | latency/s | tokens/s | GPU MB | 通信 MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| device_only | 8.0046 | 2994.6559 | 0.0327 | 0.1702 | 0.0043 | 34770.86 | 4171.28 | 0.00 |
| edge_only | 1.7387 | 5.6898 | 0.6434 | 0.8143 | 0.0226 | 6682.21 | 4215.02 | 0.00 |
| coarse_to_fine | 1.7510 | 5.7601 | 0.6368 | 0.8109 | 0.0301 | 5021.67 | 4965.25 | 0.25 |

`coarse_to_fine` 相比 `edge_only`：

- top-1 下降：0.0066
- top-5 下降：0.0034
- PPL 增加：0.0703
- latency 增加：0.0075 s
- tokens/s 从 6682.21 降到 5021.67，约下降 24.85%
- GPU memory 从 4215.02 MB 增到 4965.25 MB，增加约 750.23 MB
- 额外通信量：0.25 MB/batch
- quality gain per MB：-0.0264

因此，当前结果不支持“端侧细粒度空间协作能够提升边缘 MDLM denoising 质量”这个假设。benchmark summary 中也明确给出：

- `coarse_to_fine_better_than_edge_only_top1: False`
- `perplexity_decreased: False`
- `hypothesis_supported: False`

## 5. 对边端协作假设的解释

当前实验更像是“强 edge + 弱 device 条件注入”，而不是“端侧补充边缘侧缺失信息”。

`edge_only` 已经有很强的 masked token recovery 能力，top-1 为 0.6434，top-5 为 0.8143。相比之下，`device_only` top-1 只有 0.0327。端侧表示如果没有携带足够可靠的 token 语义，注入到边缘模型后更容易成为噪声。

当前协作机制对预训练 edge 的使用也比较受限。由于 Hugging Face MDLM forward 没有稳定 hidden-state injection point，代码对预训练 edge 的 coarse conditioning 是通过 `coarse_to_logits(conditioning)` 加到 edge logits 上。这是一种后验 logits residual，而不是在扩散 denoising 主干内部进行深层条件融合。因此它可能干扰边缘模型原本已经较好的输出分布。

alignment loss 规模也偏大。最终 align loss 仍有 230 左右，训练早期甚至达到 900 以上。虽然实际乘了 `lambda_align=0.05`，但它仍然对 total loss 有明显贡献。端侧表示被拉向 edge hidden 投影空间，不代表它对最终 token logits 一定有正贡献。

另外，当前评估是单步 masked recovery。扩散模型协作的优势可能更适合在多步 denoising 过程中体现，例如端侧先提供粗到细的轨迹、置信度或局部候选集合，而不是只在单个 mask ratio 下做一次恢复。

## 6. 当前结果的价值

虽然主假设没有被支持，但结果仍然有价值：

1. 预训练边缘 MDLM 已经成功接入并稳定训练到 20000 step。
2. token-level 128 维细粒度表示通信量可控，每 batch 为 0.25 MB，相当于 6 倍压缩。
3. 端侧模型训练目标有效，device loss、distill loss 和 align loss 都显著下降。
4. 实验清楚暴露了当前协作瓶颈：端侧表示质量不足，且 logits-level residual conditioning 不够理想。

换句话说，当前结果不是证明“边端细粒度协作无效”，而是证明“当前这种弱端侧表示 + logits residual 注入方式，无法超过强预训练 edge-only baseline”。

## 7. 建议的后续实验

### 7.1 增强端侧细粒度表示质量

优先提高 device model 的表示能力，而不是直接调大协作权重。可以尝试：

- 增大 `device_hidden_size` 或 `device_layers`
- 延长端侧预训练或先单独训练 device denoiser
- 增大 `lambda_device`，但需要观察是否牺牲 edge loss
- 使用更强蒸馏，例如 feature-level distillation，而不仅是 logits KL

目标是先让 `device_only` top-1 不再接近 0。否则端侧表示很难对强 edge 模型产生正增益。

### 7.2 改进注入位置

当前对预训练 MDLM 的条件注入发生在 logits residual 层。更合理的方式是把 128 维细粒度表示注入 edge backbone 内部，例如：

- 在每层 block 前加入 cross-attention 或 adapter
- 将 coarse representation 映射为 AdaLN 条件
- 在 embedding 后、Transformer block 前注入 token-level residual
- 只在 masked positions 注入，避免污染未 mask 上下文

如果要验证“细粒度空间协作”，注入点最好位于 hidden state 或 diffusion transition 内部，而不是最终 logits。

### 7.3 做 coarse_dim 消融

当前只有 `coarse_dim=128`。建议至少比较：

| coarse_dim | 压缩比 | 预期作用 |
| ---: | ---: | --- |
| 32 | 24x | 极低通信，检验信息瓶颈 |
| 64 | 12x | 平衡通信和表示 |
| 128 | 6x | 当前设置 |
| 256 | 3x | 检验更高维细粒度空间是否开始有增益 |

如果 256 维仍然不能超过 edge-only，问题更可能在注入机制；如果高维有效、低维无效，问题更可能在通信瓶颈。

### 7.4 加入多步 diffusion sampling 评估

当前评估只覆盖单步 masked token recovery。建议增加多步 denoising：

- 不同 mask ratio 的逐步恢复
- 每一步由端侧提供 token-level condition
- 比较 edge-only iterative sampling 和 coarse-to-fine iterative sampling
- 记录每一步的 entropy、top-k hit、最终文本质量

这会更贴近“扩散模型协作”的原始动机。

### 7.5 分析端侧表示是否真的有信息

建议额外加入 probing：

- 用 `coarse` 直接训练一个线性 probe 预测 clean token
- 计算 masked positions 上 coarse 与 edge hidden 的 cosine similarity
- 比较正确恢复和错误恢复位置的 coarse norm / entropy
- 可视化 coarse embedding 的 token cluster

如果 probe 也很弱，说明细粒度空间本身信息不足；如果 probe 有效但协作无效，说明主要问题在 edge 注入方式。

## 8. 总结结论

当前实验成功跑通了预训练 MDLM 边缘模型、端侧小模型、token-level 细粒度表示传输和端边联合训练流程。训练指标整体改善，说明系统工程路径是可行的。

但从最终评估看，`coarse_to_fine` 没有超过 `edge_only`。它在 top-1、top-5、PPL、延迟、吞吐和显存上都弱于 edge-only，同时还引入 0.25 MB/batch 的通信量。因此当前结果不能作为边端细粒度扩散协作有效性的正证据。

最可能的原因是：edge-only baseline 已经很强，device-only 太弱，而当前预训练 MDLM 的协作注入只是在 logits 层做 residual，无法充分利用 token-level 细粒度空间。下一步应该优先改进 hidden-level 条件注入和端侧表示质量，再做 coarse_dim 与多步 diffusion sampling 消融。
