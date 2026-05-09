# Coarse-to-Fine Edge-Device Diffusion LM Demo

This repository is a runnable research prototype for coarse-to-fine collaborative
masked diffusion language modeling. It keeps the code lightweight enough to run
in a CPU-only environment, while following the organization of the Arche MDLM
baseline: real-text dataset hooks, GPT-2 tokenizer support, masked denoising
training, confidence-based iterative generation, baseline comparisons, and CSV
/ JSON experiment tables.

## Method

The generation process is split into two spaces:

1. The device model denoises a low-dimensional coarse semantic representation.
2. The edge model receives that coarse representation and refines masked tokens
   in fine-grained token space.

This targets the main bottleneck on resource-limited devices: not only many
diffusion steps, but high per-step hidden-state compute, memory traffic, and
state storage. The prototype measures whether sending compact coarse semantics
can reduce device-side cost while preserving enough information for edge
refinement.

## Files

- `data.py`: arche-like dataset loader with TinyStories / WikiText-2 / local
  file hooks and automatic fallback to a small real-text corpus.
- `coarse_space.py`: linear, segment pooling, and placeholder VQ coarse semantic
  spaces.
- `coarse_device_model.py`: small Transformer-style coarse denoiser.
- `fine_edge_model.py`: larger bidirectional masked token refiner with coarse
  adapter conditioning.
- `collaboration.py`: coarse transfer and distillation communication accounting.
- `trainer.py`: `device_only`, `edge_only`, `vanilla_collaborative_distillation`,
  and `coarse_to_fine` training/evaluation.
- `run_ablation.py`: automated baseline and coarse-space ablations.

## Training Objective

The implemented objective is:

```text
L = L_device_coarse + L_edge_fine + lambda_align * L_align + lambda_distill * L_distill
```

The first version uses direct parameter updates for lightweight NumPy modules:

- `L_device_coarse`: MSE between predicted and clean coarse representations.
- `L_edge_fine`: masked-token cross entropy for edge refinement.
- `L_align`: MSE alignment between device coarse output and clean coarse target.
- `L_distill`: optional KL from edge logits to device coarse-token logits.

## Run

```bash
python run_train.py --config configs/arche_like.yaml --mode coarse_to_fine
python run_eval.py --config configs/arche_like.yaml
python run_ablation.py --config configs/arche_like.yaml
```

Other modes:

```bash
python run_train.py --config configs/arche_like.yaml --mode device_only
python run_train.py --config configs/arche_like.yaml --mode edge_only
python run_train.py --config configs/arche_like.yaml --mode vanilla_collaborative_distillation
```

Results are saved to:

- `results/coarse_to_fine_results.csv`
- `results/coarse_to_fine_results.json`

## Metrics

The CSV table includes:

```text
mode,dataset,coarse_dim,compression_method,device_steps,edge_steps,
train_time,step_time,memory_MB,comm_MB,sampling_latency,tokens_per_sec,
token_acc,val_loss,compression_ratio,refinement_gain
```

The JSON rows also include parameter counts, approximate FLOPs, device/edge
generation time, coarse representation bytes, hidden-state baseline bytes,
communication per batch, communication per generated sequence, masked token
accuracy, perplexity-like surrogate, alignment loss, distinct-n, and repetition
rates.

## Research Questions

1. How should the coarse representation space be designed: linear projection,
   segment pooling, learned codebook, or another semantic bottleneck?
2. How should a heterogeneous edge model consume coarse semantics: adapter
   addition, cross-attention, or a stronger refinement interface?
3. What coarse dimension gives the best quality/communication trade-off?
4. Can coarse-to-fine collaboration reduce device training and generation cost
   while keeping token recovery quality acceptable?
5. Is there a Pareto frontier between communication, latency, memory, and
   generation quality that can be optimized for edge deployment?

## Notes

This is a first runnable prototype, not a SOTA implementation. The code avoids
PyTorch because the current environment does not provide it. The model classes
still preserve the intended research structure: device-side coarse denoising,
edge-side token refinement, coarse-to-fine conditioning, baseline organization,
and metrics needed for follow-up experiments.
