# AR-Guided Diffusion Verification

This repository now focuses on one evaluation-only idea:

> Use a local GPT-2 small model as an autoregressive candidate generator, and use
> a local MDLM masked diffusion model as the parallel verifier/reranker.

The first implementation does not train a new module. It evaluates whether
GPT-2 top-k candidates help MDLM recover high-uncertainty masked tokens.

## Local Assets

Keep Hugging Face assets under `/mnt/data/enzeyu/hf_downloads`:

- Models: `/mnt/data/enzeyu/hf_downloads/models`
- Datasets: `/mnt/data/enzeyu/hf_downloads/datasets`

Expected local models:

```text
/mnt/data/enzeyu/hf_downloads/models/gpt2
/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
```

The evaluation config sets:

```yaml
tokenizer_name: /mnt/data/enzeyu/hf_downloads/models/gpt2
dataset_cache_dir: /mnt/data/enzeyu/hf_downloads/datasets
hf_local_files_only: true
pretrained_edge_path: /mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
device_model_name_or_path: /mnt/data/enzeyu/hf_downloads/models/gpt2
```

If WikiText is not available under the dataset cache and the environment cannot
download it, the evaluator writes a skipped result instead of falling back to
`/home/enzeyu/.cache`.

## Files

- `eval_gpt2_assist.py`: main evaluation entrypoint.
- `configs/wikitext2_mdlm_gpt2_assist.yaml`: AR-guided verification config.
- `data_real.py`: tokenizer, WikiText loading, fixed-length blocks, and masked
  token corruption.
- `models_mdlm_wrapper.py`: local/Hugging Face MDLM edge verifier loader.
- `metrics.py`: table formatting and small metric helpers.
- `validate_pretrained_mdlm.py`: optional smoke test for the local MDLM.

Old coarse-hidden injection training/evaluation scripts have been removed.

## Evaluation Modes

`eval_gpt2_assist.py` reports four modes:

- `edge_only`: MDLM predicts masked tokens from bidirectional masked context.
- `gpt2_only`: GPT-2 predicts each masked position from left context only.
- `random_assist`: MDLM logits are biased using random candidate ids.
- `gpt2_assist`: GPT-2 proposes top-k candidates for high-uncertainty masked
  positions; MDLM logits rerank them through an additive candidate bias.

For hard token selection, `uncertainty_score` supports:

- `entropy`
- `inverse_confidence`
- `margin`

Important config fields:

```yaml
uncertainty_score: entropy
hard_token_ratio: 0.3
device_top_k: 20
gpt2_assist_alpha: 0.5
eval_steps: 200
```

## Run

Main command:

```bash
python eval_gpt2_assist.py \
  --config configs/wikitext2_mdlm_gpt2_assist.yaml \
  --ckpt results/wikitext2_mdlm_medium/checkpoint.pt
```

The checkpoint argument is optional if you only want to use the configured local
pretrained MDLM. When present, matching `edge_model.*` weights are loaded and old
non-edge keys are ignored.

Optional MDLM smoke test:

```bash
python validate_pretrained_mdlm.py \
  --config configs/wikitext2_mdlm_gpt2_assist.yaml \
  --forward-check
```

## Outputs

Results are written to:

```text
results/wikitext2_mdlm_gpt2_assist/
```

Files:

- `gpt2_assist_eval.csv`
- `gpt2_assist_eval.json`
- `gpt2_assist_summary.md`

The benchmark table includes:

```text
mode,loss,perplexity,top1_acc,top5_acc,hard_top1_acc,hard_top5_acc,correction_rate,regression_rate,query_ratio,communication_MB,quality_gain_per_MB,latency,tokens_per_sec
```

The summary answers whether `gpt2_assist` improves over `edge_only` and
`random_assist`, whether gains concentrate on hard tokens, whether correction
beats regression, and whether the result supports the AR-guided assistance
hypothesis.
