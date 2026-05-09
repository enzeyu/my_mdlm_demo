# Edge-Device Collaborative Training for Diffusion Language Models

This directory contains a minimal PyTorch research prototype for edge-device
collaborative training of masked diffusion language models. It is intentionally
small: the toy config runs on CPU without downloading datasets.

## Run

```bash
python edge_device_training/run_train.py --config edge_device_training/configs/toy.yaml --mode device_only
python edge_device_training/run_train.py --config edge_device_training/configs/toy.yaml --mode edge_only
python edge_device_training/run_train.py --config edge_device_training/configs/toy.yaml --mode collaborative
python edge_device_training/run_eval.py --config edge_device_training/configs/toy.yaml
```

Results are saved to:

- `edge_device_training/results/results.json`
- `edge_device_training/results/results.csv`

## Design

- `data.py`: built-in toy corpus, simple tokenizer, fixed-length token chunks,
  and hooks for local files or WikiText-2.
- `models.py`: two configurable Transformer masked diffusion LMs.
- `diffusion_utils.py`: MDLM-style token masking corruption and denoising loss.
- `collaboration.py`: error-driven token selection, edge-to-device logit
  distillation, and communication byte accounting.
- `trainer.py`: `device_only`, `edge_only`, and `collaborative` training modes.
- `run_eval.py`: baseline comparison table.

## Metrics

The prototype records training loss, validation denoising loss, token recovery
accuracy, wall-clock training time, time per step, GPU memory, communication MB,
sampling latency, and generated tokens per second.

## Potential Research Insights

This prototype is meant to support controlled experiments on:

1. Which low-cost signal is most valuable to transmit: uncertain token ids,
   dense logits, hidden prototypes, or partial parameter updates.
2. How to select token positions under a communication budget using loss,
   entropy, or confidence.
3. How synchronization frequency trades off against device convergence and
   communication volume.
4. Whether edge-guided training improves the small device model's denoising
   accuracy per communicated MB.
5. Whether collaborative training can reduce generation-time denoising steps for
   a fixed quality target.
6. How to build an optimization objective over quality, latency, and
   communication cost.

