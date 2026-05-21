# Coarse-to-Fine 边端协同实验结果分析

## 1. 实验结果读取情况

本次分析读取了目录 `results/wikitext2_mdlm_medium` 下的以下文件：

- `train_metrics.csv`
- `train_metrics.json`
- `train_metrics.jsonl`
- `eval_metrics.csv`
- `eval_metrics.json`
- `benchmark_results.csv`
- `benchmark_results.json`
- `benchmark_summary.txt`
- `ppl_curve.png`
- `top1_acc_curve.png`
- `top5_acc_curve.png`
- `quality_vs_comm.png`

训练日志三种格式一致：`train_metrics.csv`、`train_metrics.json`、`train_metrics.jsonl` 均包含 20000 条训练记录，最终 step 为 20000。结果中记录 `model_backend=mdlm` 且 `pretrained_edge_loaded=True`，说明本轮实验已经接入本地预训练 MDLM-style edge checkpoint，而不是随机初始化 edge model。

图像文件均存在，曲线分析主要基于训练日志中的数值序列完成。

## 2. 训练与验证曲线趋势

### 2.1 训练指标

| Metric | Initial | Final | Best / Min / Max | Last-100 Mean | Trend |
|---|---:|---:|---:|---:|---|
| Total Loss | 64.7016 | 15.3239 | min 13.5949 | 15.1468 | 明显下降，但受 align/distill 项影响较大 |
| Edge Loss | 2.8295 | 1.5134 | min 0.7206 | 1.5433 | 下降，edge denoising 能力改善 |
| Device Loss | 69.8842 | 7.8698 | min 6.8478 | 7.9725 | 明显下降，但 device 仍远弱于 edge |
| Align Loss | 820.4493 | 231.5384 | min 196.3388 | 227.3844 | 下降但绝对值仍高 |
| Distill Loss | 68.7279 | 6.5961 | min 5.5871 | 6.3975 | 明显下降 |
| Train Top-1 | 0.4323 | 0.6667 | max 0.8163 | 0.6692 | 提升 |
| Train Top-5 | 0.6581 | 0.8616 | max 0.9478 | 0.8396 | 提升 |
| Train PPL | 16.9372 | 4.5422 | min 2.0557 | 4.8049 | 下降 |
| Step Time | 1.1547 s | 0.1277 s | min 0.0873 s | 0.1044 s | warmup 后稳定 |
| GPU Memory | 3260.69 MB | 5917.46 MB | max 5917.46 MB | 5917.46 MB | 后期稳定 |

训练本身是正常的：edge loss、device loss、distill loss、train PPL 均有明显下降，train top-1/top-5 均有提升。需要注意的是，align loss 虽然下降，但最终仍为 231.5384，说明 device coarse space 与 edge projected coarse space 的对齐并不充分，可能影响 coarse conditioning 的有效性。

### 2.2 验证指标

验证曲线每 1000 step 记录一次。最终验证结果：

- final val loss: 1.5544
- final val PPL: 4.7324
- final val top-1: 0.6682
- final val top-5: 0.8333

最佳验证点：

- best val loss: 1.5455 at step 13000
- best val PPL: 4.6901 at step 13000
- best val top-1: 0.6764 at step 17000
- best val top-5: 0.8333 at step 20000

这说明预训练 edge 接入后，masked token recovery 质量明显强于之前随机初始化实验；但后期曲线存在小幅波动，PPL 和 top-1 的最优点并不在最终 step。

### 2.3 曲线图对应结论

![Perplexity Curve](ppl_curve.png)

PPL 曲线整体处于较低区间，最终 train PPL 为 4.5422，final val PPL 为 4.7324。相比随机初始化 edge 时数百级 PPL，本轮预训练 edge 的质量显著更好。

![Top-1 Accuracy Curve](top1_acc_curve.png)

Top-1 曲线显示模型训练后具备较强 masked token recovery 能力，final val top-1 为 0.6682，best val top-1 为 0.6764。

![Top-5 Accuracy Curve](top5_acc_curve.png)

Top-5 曲线最终达到 0.8333，说明正确 token 大多数情况下能进入高概率候选集合。

![Quality vs Communication](quality_vs_comm.png)

Quality vs Communication 图中，`coarse_to_fine` 引入 0.25 MB / batch 通信，但 benchmark 质量没有超过 `edge_only`，因此当前通信-质量折中不成立。

## 3. Benchmark 三种模式对比

| Mode | Loss | PPL | Top-1 Acc | Top-5 Acc | Latency | Tokens/s | GPU Mem MB | Comm MB | Compression Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| device_only | 8.0158 | 3028.2828 | 0.0312 | 0.1665 | 0.0044 | 34607.64 | 4171.28 | 0.00 | 6.00 |
| edge_only | 1.7496 | 5.7521 | 0.6411 | 0.8118 | 0.0232 | 6518.51 | 4216.02 | 0.00 | 6.00 |
| coarse_to_fine | 1.7636 | 5.8335 | 0.6384 | 0.8116 | 0.0293 | 5207.88 | 4965.25 | 0.25 | 6.00 |

### 3.1 device_only vs edge_only

`device_only` 明显弱于 `edge_only`：

- PPL: 3028.2828 vs 5.7521
- top-1: 0.0312 vs 0.6411
- top-5: 0.1665 vs 0.8118

这说明端侧轻量模型单独建模能力很弱，预训练 edge model 是主要质量来源。该结果符合边端异构设定：端侧模型负责轻量表征，边缘侧模型负责高质量 denoising。

### 3.2 coarse_to_fine vs edge_only

| Comparison | PPL Change | Top-1 Gain | Top-5 Gain | Extra Comm | Extra Latency | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| coarse_to_fine vs edge_only | +0.0814 | -0.0026 | -0.0002 | +0.25 MB | +0.0061 s | 未优于 edge_only |

更具体地说：

- `coarse_to_fine` 的 PPL 为 5.8335，高于 `edge_only` 的 5.7521，PPL 变差 0.0814，约 +1.42%；
- `coarse_to_fine` top-1 为 0.6384，低于 `edge_only` 的 0.6411，下降 0.00264；
- `coarse_to_fine` top-5 为 0.8116，低于 `edge_only` 的 0.8118，下降 0.00018；
- `coarse_to_fine` latency 为 0.0293 s，高于 `edge_only` 的 0.0232 s，增加 0.0061 s，约 +26.5%；
- `coarse_to_fine` tokens/sec 为 5207.88，低于 `edge_only` 的 6518.51，下降约 20.1%；
- `coarse_to_fine` GPU memory 为 4965.25 MB，高于 `edge_only` 的 4216.02 MB，增加约 749.23 MB；
- `coarse_to_fine` 通信量为 0.25 MB / batch，compression ratio 为 6.0；
- `quality_gain_per_MB = -0.01056`，为负值。

因此，在当前 benchmark 下，`coarse_to_fine` 没有在质量指标上超过 `edge_only`。

## 4. 对核心研究问题的回答

核心问题是：端侧轻量模型生成的 coarse representation 是否能够改善边缘侧扩散语言模型的效果，并在较低通信成本下提升整体效率？

### 4.1 端侧模型是否对边侧模型有正向帮助？

当前结果不支持。`coarse_to_fine` 相比 `edge_only`：

- loss 从 1.7496 增加到 1.7636；
- PPL 从 5.7521 增加到 5.8335；
- top-1 从 0.6411 降到 0.6384；
- top-5 从 0.8118 降到 0.8116。

因此，当前 coarse representation 没有对预训练 edge denoising 产生可观测的正向帮助。

### 4.2 coarse_to_fine 是否优于 edge_only？

不优于。三个核心质量指标均略差于 `edge_only`：

- PPL 更高；
- top-1 更低；
- top-5 更低。

### 4.3 这种提升是否值得通信开销？

当前不存在质量提升，因此不能说值得通信开销。虽然 0.25 MB / batch 的通信量本身较低，compression ratio 为 6.0，但 `quality_gain_per_MB = -0.01056`，单位通信收益为负。

### 4.4 是否体现出边端协同相比单独边侧模型的效率优势？

当前单设备测得的 latency 和 tokens/sec 不支持效率优势：

- latency 增加 0.0061 s，约 +26.5%；
- tokens/sec 下降约 20.1%；
- GPU memory 增加约 749.23 MB。

但需要谨慎解释：本实验是在一个设备上串行测量 `device + edge` 路径，不能代表真实端边异构部署。在真实系统中，端侧 coarse encoder 可能运行在低功耗设备，边缘侧只接收低维 coarse representation，潜在目标可能是降低端侧上传 token/hidden/logit 的通信量或降低端侧能耗。不过当前日志没有记录能耗、端边并行时延、网络传输时延或端侧功耗，因此“能耗降低”目前只是合理假设，尚未被本实验直接验证。

### 4.5 当前结果是否足以支撑 CCF-A 论文方向？

作为“预实验流程可行性”可以支撑；作为“coarse-to-fine 明显优于 edge-only”的核心实验证据暂不支持。

本轮实验已经验证：

- pretrained MDLM-style edge checkpoint 可以接入；
- masked diffusion LM 训练和评估流程可跑通；
- `device_only`、`edge_only`、`coarse_to_fine` 三种模式可以统一 benchmark；
- coarse representation 通信量可控，当前为 0.25 MB / batch。

但还没有验证：

- coarse representation 能提升预训练 edge quality；
- coarse-to-fine 能在质量、通信、延迟或能耗之间形成 Pareto 优势；
- 端边协同机制相较强 edge-only baseline 的稳定收益。

## 5. 当前目标是否达成

结论选择：

## C. 当前实验暂不支持该目标

依据如下：

1. `coarse_to_fine` 没有优于 `edge_only`：PPL 5.8335 高于 5.7521，top-1 0.6384 低于 0.6411，top-5 0.8116 低于 0.8118。
2. 通信成本虽低，但没有换来质量收益：communication 为 0.25 MB / batch，compression ratio 为 6.0，但 `quality_gain_per_MB = -0.01056`。
3. 单设备测量下效率也未提升：latency 增加 0.0061 s，tokens/sec 下降约 20.1%。
4. 当前训练流程和预训练 edge 接入是成功的，但核心假设“端侧 coarse representation 改善边缘侧 denoising”没有被本次数值支持。

更准确的表述是：当前实验支持“系统流程可行”，但暂不支持“coarse-to-fine 协作优于强 edge-only baseline”。

## 6. 为什么会这样

### 6.1 预训练 edge model 已经足够强，coarse 信息边际收益小

`edge_only` 已达到 PPL 5.7521、top-1 0.6411、top-5 0.8118。相比之下，`device_only` 只有 top-1 0.0312、top-5 0.1665。端侧模型本身能力很弱，其 coarse representation 很可能无法提供超出 edge model 自身上下文建模能力的有效信息。

### 6.2 device coarse representation 可能较弱

虽然 device loss 从 69.8842 降到 7.8698，但 benchmark 中 `device_only` PPL 仍为 3028.2828，top-1 仅 0.0312。这说明端侧 token-level 预测能力很弱。若 coarse representation 与 device token prediction 共享表征，其语义质量可能不足以帮助强 edge model。

### 6.3 alignment loss 没有形成足够有效的互补空间

align loss 从 820.4493 降至 231.5384，但绝对值仍高。当前 alignment 可能只是在数值上拉近 coarse projection，并未形成对 edge denoising 真正有用的语义条件。

### 6.4 coarse adapter 过于简单

当前 coarse conditioning 主要通过线性映射注入 edge hidden/logit 路径。对于强预训练 MDLM 来说，简单 additive 或 residual-style conditioning 可能扰动原有表示空间，导致 PPL、top-1、top-5 均轻微下降。

### 6.5 训练目标仍是单步 masked token recovery

当前评估主要是单步 masked token recovery，不是真正 iterative diffusion sampling。coarse representation 可能更适合在多步 denoising 中提供全局语义方向，但当前实验没有覆盖这种机制。

### 6.6 端侧和边侧没有形成真正互补

当前结果显示 edge model 单独已经能很好恢复 masked token，而 device model 单独较弱。二者之间不是“互补强弱项”，更像是“强 edge + 弱 device 信息注入”。如果注入方式不够精细，弱 coarse 信号可能成为噪声。

## 7. 后续改进方向

### Priority 1: 必须先做的实验

1. 重复 `coarse_to_fine vs edge_only` 公平实验，至少 3 个随机种子，报告均值和标准差。
2. 做 `with / without coarse conditioning` 消融，验证当前负增益是否来自 coarse 注入本身。
3. 做 `with / without alignment loss` 消融，判断 alignment 是否真的帮助 edge refinement。
4. 做 `coarse_dim` 消融：64 / 128 / 256，建立通信量、压缩率、质量之间的关系。
5. 分别报告 best checkpoint 和 final checkpoint，避免训练后期波动影响结论。
6. 记录稳定性指标：PPL、top-1、top-5、latency、tokens/sec 的多次评估方差。

### Priority 2: 方法层面改进

1. 使用更强的 coarse encoder，提升端侧 coarse representation 的语义质量。
2. 将简单线性 additive conditioning 替换为 cross-attention conditioning 或 prefix-token conditioning。
3. 使用 confidence-driven collaboration：只对 edge 不确定的位置注入或上传 coarse representation。
4. 做 adaptive token selection：只上传高不确定 token 的 coarse representation，进一步降低通信量。
5. 改进 alignment objective，使 coarse space 更贴近 edge denoising 所需的语义结构，而不是单纯 MSE 对齐。
6. 引入 edge-to-device distillation，让端侧模型学习强 edge 的中间语义，而不仅是 token logits。

### Priority 3: 论文级升级

1. 从单步 masked recovery 扩展到多步 iterative diffusion denoising / sampling。
2. 使用更大语料和更真实的 benchmark，避免只在 WikiText-2 小规模设置上判断。
3. 建立通信-质量-延迟-能耗 Pareto trade-off，而不仅是质量和单设备 latency。
4. 在真实或模拟端边异构系统中测量端侧能耗、边缘侧计算、网络传输时延和整体 wall-clock latency。
5. 设计端边协作调度策略：何时上传 coarse representation、上传哪些 token、上传多少维度。

## 8. 汇报版总结

### 当前发现

- 本轮实验已成功接入预训练 edge：训练日志显示 `pretrained_edge_loaded=True`，final val PPL 为 4.7324，final val top-1 为 0.6682，final val top-5 为 0.8333。
- `edge_only` 显著强于 `device_only`：PPL 5.7521 vs 3028.2828，top-1 0.6411 vs 0.0312，top-5 0.8118 vs 0.1665。
- `coarse_to_fine` 没有超过 `edge_only`：PPL 5.8335 高于 5.7521，top-1 0.6384 低于 0.6411，top-5 0.8116 低于 0.8118。
- `coarse_to_fine` 引入 0.25 MB / batch 通信，compression ratio 为 6.0，但 `quality_gain_per_MB = -0.01056`。
- 单设备串行测量下，`coarse_to_fine` latency 从 0.0232 s 增至 0.0293 s；但该 latency 不代表真实端边异构部署，也不能直接否定潜在能耗收益。

### 当前结论

当前实验尚未证明端侧模型能够改善边侧模型效果。更准确地说，本轮实验证明了预训练 MDLM edge 接入和 coarse-to-fine 流程可行，但在强 `edge_only` baseline 下，当前 coarse representation 没有带来质量收益，反而造成 PPL、top-1 和 top-5 的轻微下降。因此当前结果属于“系统可行但核心效果暂不成立”的预实验阶段。

### 下一步工作

- 优先做 `coarse_to_fine vs edge_only` 多随机种子重复实验，并加入 `with / without coarse conditioning` 和 `with / without alignment loss` 消融。
- 做 `coarse_dim = 64 / 128 / 256` 消融，建立通信量、压缩率和质量变化之间的 Pareto 曲线。
- 改进 coarse 注入方式，从线性 additive conditioning 升级为 cross-attention 或 prefix conditioning，减少对预训练 edge 表示空间的扰动。
- 引入 confidence-driven / token-selection 协作，只上传 edge 高不确定位置的 coarse representation。
- 从单步 masked recovery 扩展到多步 diffusion sampling，并补充真实端边部署下的通信、延迟和能耗评估。
