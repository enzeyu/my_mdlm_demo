# Coarse-to-Fine Edge-Device Diffusion LM Prototype

This is a minimal real PyTorch experiment for coarse-to-fine masked diffusion
language modeling. It trains on WikiText-2 with a GPT-2 tokenizer and runs on
CUDA automatically when available.

## Idea

The device model is a small Transformer masked diffusion LM. It receives a
masked token sequence and emits:

- low-dimensional `coarse representation`
- optional device token logits for the `device_only` baseline

The edge model is a larger Transformer masked diffusion LM. It receives the same
masked token sequence and optionally adds a projected coarse representation into
its token hidden states before refinement.

The main research question is:

> Can a low-dimensional coarse representation reduce communication while
> preserving or improving token-level denoising quality after edge refinement?

## Files

- `configs/wikitext2_coarse.yaml`: default runnable experiment config.
- `data_real.py`: WikiText-2 loading, GPT-2 tokenizer, fixed-length token blocks,
  and masked diffusion corruption.
- `model_coarse_to_fine.py`: device Transformer, edge Transformer, coarse
  conditioning, losses, and communication-size helpers.
- `train_coarse_to_fine.py`: CUDA-aware training script.
- `eval_coarse_to_fine.py`: evaluates `device_only`, `edge_only`, and
  `coarse_to_fine`.
- `metrics.py`: metric saving and table formatting.

## Train

```bash
python train_coarse_to_fine.py --config configs/wikitext2_coarse.yaml
```

The script prints:

- training loss
- edge token loss
- device token loss
- coarse alignment loss
- masked token accuracy
- step time
- GPU memory

It saves:

- `results/wikitext2_coarse/checkpoint.pt`
- `results/wikitext2_coarse/train_metrics.csv`
- `results/wikitext2_coarse/train_metrics.json`
- `results/wikitext2_coarse/train_metrics.jsonl`

## Evaluate

```bash
python eval_coarse_to_fine.py --config configs/wikitext2_coarse.yaml
```

Or with an explicit checkpoint:

```bash
python eval_coarse_to_fine.py \
  --config configs/wikitext2_coarse.yaml \
  --ckpt results/wikitext2_coarse/checkpoint.pt
```

## Metrics

- `validation_loss`: masked-token cross entropy on validation batches.
- `masked_token_accuracy`: accuracy only on tokens corrupted by `[MASK]`.
- `denoising_accuracy`: same as masked-token accuracy in this minimal prototype.
- `refinement_latency`: average forward/loss latency per validation batch.
- `tokens_per_sec`: masked denoising targets processed per second.
- `gpu_memory_MB`: peak allocated CUDA memory.
- `coarse_comm_MB`: size of transmitted coarse representation per batch.
- `compression_ratio`: edge hidden size divided by coarse dimension.
- `refinement_gain_over_device_only`: accuracy improvement over device-only.

## Next Research Extensions

1. Replace additive conditioning with cross-attention or prefix tokens.
2. Sweep `coarse_dim` over 32/64/128 and report quality/communication Pareto
   curves.
3. Add iterative MDLM sampling schedules instead of single-step denoising eval.
4. Pretrain or freeze a stronger edge model to study whether coarse semantics
   transfer across heterogeneous model sizes.
5. Compare against transmitting edge hidden states, logits, or selected token
   positions under equal communication budgets.
