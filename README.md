# Coarse-to-Fine Edge-Device Diffusion LM Prototype

This is a minimal real PyTorch experiment for coarse-to-fine masked diffusion
language modeling. It trains on WikiText-2 with a GPT-2 tokenizer and runs on
CUDA automatically when available.

The project now supports two backends:

- `internal_toy`: the original small Transformer sanity-check baseline.
- `mdlm`: an MDLM-style masked diffusion backend. With
  `require_pretrained_edge: true`, it must load a real Hugging Face or local
  pretrained MDLM checkpoint; otherwise the run fails instead of silently using
  a randomly initialized edge model.

## Idea

The device model is a lightweight masked diffusion LM. It receives a masked
token sequence and emits:

- low-dimensional `coarse representation`
- optional device token logits for the `device_only` baseline

The edge model is either the legacy larger Transformer baseline or an MDLM-style
edge denoiser. In `model_backend: mdlm`, the code uses the same tokenizer,
absorbing `[MASK]` noising process, masked denoising objective, and checkpoint
for `device_only`, `edge_only`, and `coarse_to_fine`.

The main research question is:

> Can a low-dimensional coarse representation reduce communication while
> preserving or improving token-level denoising quality after edge refinement?

## Files

- `configs/wikitext2_coarse.yaml`: default runnable experiment config.
- `configs/wikitext2_mdlm_medium.yaml`: MDLM backend experiment config.
- `data_real.py`: WikiText-2 loading, GPT-2 tokenizer, fixed-length token blocks,
  and masked diffusion corruption.
- `model_coarse_to_fine.py`: device Transformer, edge Transformer, coarse
  conditioning, backend dispatch, losses, and communication-size helpers.
- `models_mdlm_wrapper.py`: MDLM-style DiT denoiser, Hugging Face/local
  checkpoint loading, lightweight device model, edge wrapper, and coarse adapter.
- `train_coarse_to_fine.py`: CUDA-aware training script.
- `eval_coarse_to_fine.py`: evaluates `device_only`, `edge_only`, and
  `coarse_to_fine`.
- `run_coarse_dim_ablation.py`: optional coarse-dimension ablation runner.
- `metrics.py`: metric saving and table formatting.

## Train

```bash
python train_coarse_to_fine.py --config configs/wikitext2_coarse.yaml
```

MDLM backend:

```bash
python train_coarse_to_fine.py \
  --config configs/wikitext2_mdlm_medium.yaml
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

For `configs/wikitext2_mdlm_medium.yaml`, outputs are written under
`results/wikitext2_mdlm_medium/`.

## Evaluate

```bash
python eval_coarse_to_fine.py --config configs/wikitext2_coarse.yaml
```

Or with an explicit checkpoint:

```bash
python eval_coarse_to_fine.py \
  --config configs/wikitext2_mdlm_medium.yaml \
  --ckpt results/wikitext2_mdlm_medium/checkpoint.pt
```

Evaluation always reports `device_only`, `edge_only`, and `coarse_to_fine`.

## MDLM Checkpoints and hf-mirror

The code sets `HF_ENDPOINT=https://hf-mirror.com` by default before importing
`transformers`. The medium config uses
`kuleshov-group/mdlm-no_flashattn-fp32-owt`, which avoids the flash-attn
dependency required by the original `kuleshov-group/mdlm-owt` remote code.

The medium config currently points to local downloads:

```yaml
tokenizer_name: /mnt/data/enzeyu/hf_downloads/models/gpt2
hf_local_files_only: true
pretrained_edge_path: /mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
```

Run this first to confirm the edge model is truly pretrained:

```bash
HF_ENDPOINT=https://huggingface.co \
python validate_pretrained_mdlm.py \
  --config configs/wikitext2_mdlm_medium.yaml \
  --forward-check
```

Expected validation output should include:

```text
model_backend=mdlm
pretrained_edge_loaded=True
```

After validation succeeds, start pretrained MDLM training from the project root:

```bash
cd /mnt/data/enzeyu/my_researchs/mydllm/my_mdlm_demo

python train_coarse_to_fine.py \
  --config configs/wikitext2_mdlm_medium.yaml
```

If you want to keep the old randomly initialized pre-experiment results, change
`save_dir` in `configs/wikitext2_mdlm_medium.yaml` before training, for example:

```yaml
save_dir: results/wikitext2_mdlm_medium_pretrained
```

Then evaluate the new pretrained run with:

```bash
python eval_coarse_to_fine.py \
  --config configs/wikitext2_mdlm_medium.yaml \
  --ckpt results/wikitext2_mdlm_medium_pretrained/checkpoint.pt
```

If you keep the default `save_dir`, use:

```bash
python eval_coarse_to_fine.py \
  --config configs/wikitext2_mdlm_medium.yaml \
  --ckpt results/wikitext2_mdlm_medium/checkpoint.pt
```

To use a manually downloaded checkpoint, set:

```yaml
use_pretrained_edge: true
require_pretrained_edge: true
pretrained_edge_path: /path/to/local/mdlm/checkpoint_or_snapshot
```

With `require_pretrained_edge: true`, if neither Hugging Face nor the local path
is available, training stops with an explicit error. The script prints:

- whether `model_backend` is `internal_toy` or `mdlm`;
- whether pretrained edge MDLM was loaded;
- whether the current edge model is still the toy model.

## Metrics

- `validation_loss`: masked-token cross entropy on validation batches.
- `masked_token_accuracy`: accuracy only on tokens corrupted by `[MASK]`.
- `denoising_accuracy`: same as masked-token accuracy in this minimal prototype.
- `refinement_latency`: average forward/loss latency per validation batch.
- `tokens_per_sec`: masked denoising targets processed per second.
- `gpu_memory_MB`: peak allocated CUDA memory.
- `coarse_comm_MB`: size of transmitted coarse representation per batch.
- `compression_ratio`: edge hidden size divided by coarse dimension.
- `gain_over_edge_only`: top-1 accuracy improvement over edge-only.
- `quality_gain_per_MB`: `gain_over_edge_only / communication_MB`.

`eval_coarse_to_fine.py` saves:

- `benchmark_results.csv`
- `benchmark_results.json`
- `benchmark_summary.txt`
- `eval_metrics.csv`
- `eval_metrics.json`

The CSV schema is:

```text
mode,model_backend,loss,perplexity,top1_acc,top5_acc,latency,tokens_per_sec,gpu_memory_MB,communication_MB,compression_ratio,gain_over_edge_only,quality_gain_per_MB
```

## Coarse-Dim Ablation

```bash
python run_coarse_dim_ablation.py \
  --base_config configs/wikitext2_mdlm_medium.yaml \
  --coarse_dims 64 128 256
```

This writes `coarse_dim_ablation.csv` and `coarse_dim_ablation.json` next to the
configured result directory.

## Next Research Extensions

1. Replace additive conditioning with cross-attention or prefix tokens for HF
   checkpoints whose forward API exposes stable hidden-state injection.
2. Add iterative MDLM sampling schedules instead of single-step denoising eval.
3. Pretrain or freeze a stronger edge model to study whether coarse semantics
   transfer across heterogeneous model sizes.
4. Compare against transmitting edge hidden states, logits, or selected token
   positions under equal communication budgets.
