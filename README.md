# Diffusion-Assisted Autoregressive Refinement

This repository currently evaluates one idea:

> Use a local GPT-2 small model as the device-side autoregressive draft model,
> use a local pretrained MDLM masked diffusion model as the edge-side verifier,
> rerank GPT-2 top-k candidates at low-confidence positions, and train a small
> learned accept gate to decide whether to accept each reranked correction.

The current experiment does not require any local MDLM training checkpoint. By
default it loads the local pretrained MDLM directly from:

```text
/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
```

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
device_model_name_or_path: /mnt/data/enzeyu/hf_downloads/models/gpt2
edge_model_name_or_path: /mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
pretrained_edge_path: /mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt
mdlm_ckpt: null
```

## Files

- `train_refine_gate.py`: trains only the learned accept gate MLP. GPT-2 and
  MDLM are frozen.
- `eval_refine_gate.py`: evaluates GPT-2-only, MDLM-only, random refine,
  candidate rerank without gate, rule gate, and learned gate.
- `refine_gate.py`: learned gate module plus candidate rerank feature extraction.
- `refine_utils.py`: shared loading, tokenizer compatibility, uncertainty
  selection, and metric helpers for the current learned-gate path.
- `configs/wikitext2_gpt2_mdlm_learned_gate.yaml`: current GPT-2 + pretrained
  MDLM + learned gate config.
- `data_real.py`: tokenizer, WikiText loading, fixed-length blocks, and masked
  token corruption.
- `models_mdlm_wrapper.py`: local/Hugging Face MDLM edge verifier loader.
- `metrics.py`: table formatting and small metric helpers.
- `validate_pretrained_mdlm.py`: optional smoke test for the local MDLM.

## Current Experiment

The current path is:

```text
GPT-2 draft
-> select low-confidence positions by inverse GPT-2 confidence
-> take GPT-2 top-k candidates
-> mask selected draft positions
-> score candidates with MDLM
-> rerank candidates inside GPT-2 top-k only
-> learned accept gate chooses rerank token or original GPT-2 token
```

Fixed defaults:

```yaml
refine_window: 0
refine_ratios: [0.2, 0.3]
refine_ratio: 0.3
refinement_method: candidate_rerank
candidate_top_k: 20
lambda_gpt2: 0.5
lambda_mdlm: 0.5
accept_threshold: 0.5
gate_hidden_size: 64
gate_layers: 2
gate_lr: 0.0003
```

The learned gate uses these features:

```text
gpt2_confidence, gpt2_entropy, mdlm_confidence, mdlm_margin,
rerank_score_gap, gpt2_mdlm_agree, gpt2_top1_in_mdlm_topk,
rerank_candidate_rank, gpt2_candidate_logprob, mdlm_candidate_logprob
```

Training labels are supervised from ground truth only where GPT-2 and rerank
disagree in correctness:

```text
accept_label = 1 if rerank is correct and GPT-2 is wrong
accept_label = 0 if GPT-2 is correct and rerank is wrong
```

GPT-2 and MDLM are never trained by this script.

## Evaluation Modes

`eval_refine_gate.py` reports:

- `gpt2_only`: GPT-2 predicts each masked position from left context only.
- `mdlm_only`: pretrained MDLM masked-token recovery baseline.
- `random_refine`: random position selection baseline.
- `candidate_rerank_no_gate`: GPT-2 top-k candidates are reranked with MDLM
  scores and always accepted.
- `candidate_rerank_rule_gate`: original threshold-based rule gate.
- `candidate_rerank_learned_gate`: learned MLP gate.

The evaluation table includes:

```text
mode,refine_ratio,candidate_top_k,top1,top5,ppl,correction_rate,regression_rate,net_correction,accepted_ratio,gate_accuracy,error_detection_precision,error_detection_recall,candidate_coverage,latency,tokens_per_sec
```

## Run Current Idea

Use the learned-gate config:

```bash
configs/wikitext2_gpt2_mdlm_learned_gate.yaml
```

### 1. Quick Sanity Check

Run a tiny gate training pass:

```bash
python train_refine_gate.py \
  --config configs/wikitext2_gpt2_mdlm_learned_gate.yaml \
  --train_steps 2
```

Then run a tiny evaluation:

```bash
python eval_refine_gate.py \
  --config configs/wikitext2_gpt2_mdlm_learned_gate.yaml \
  --gate_ckpt results/wikitext2_gpt2_mdlm_learned_gate/learned_gate.pt \
  --eval_steps 1
```

Expected logs include:

```text
Loaded edge MDLM from pretrained path
Loaded device GPT-2 from pretrained path
No MDLM checkpoint provided; using pretrained edge model only
```

### 2. Full Learned-Gate Training

```bash
python train_refine_gate.py \
  --config configs/wikitext2_gpt2_mdlm_learned_gate.yaml
```

This writes:

```text
results/wikitext2_gpt2_mdlm_learned_gate/learned_gate.pt
results/wikitext2_gpt2_mdlm_learned_gate/gate_train_metrics.csv
results/wikitext2_gpt2_mdlm_learned_gate/gate_train_metrics.json
results/wikitext2_gpt2_mdlm_learned_gate/best_gate_config.json
```

### 3. Full Evaluation

```bash
python eval_refine_gate.py \
  --config configs/wikitext2_gpt2_mdlm_learned_gate.yaml \
  --gate_ckpt results/wikitext2_gpt2_mdlm_learned_gate/learned_gate.pt
```

This writes:

```text
results/wikitext2_gpt2_mdlm_learned_gate/learned_gate_eval.csv
results/wikitext2_gpt2_mdlm_learned_gate/learned_gate_eval.json
results/wikitext2_gpt2_mdlm_learned_gate/learned_gate_summary.md
```

Read `learned_gate_summary.md` first. It answers:

```text
1. learned gate 是否降低 regression？
2. learned gate 是否保持或提升 Top1/Top5？
3. learned gate 是否优于 rule-based gate？
4. learned gate 是否优于 no-gate candidate rerank？
5. candidate coverage 是否足够？
6. 当前结果是否支持“端侧 GPT-2 draft + 边侧 MDLM verifier/refiner”的研究假设？
```

### 4. Candidate Top-k Follow-up

Start with `candidate_top_k: 20`. If `candidate_coverage` is low, test top-k 50
without changing the research direction:

```bash
python train_refine_gate.py \
  --config configs/wikitext2_gpt2_mdlm_learned_gate.yaml
```

after editing:

```yaml
candidate_top_k: 50
save_dir: results/wikitext2_gpt2_mdlm_learned_gate_top50
```

then evaluate:

```bash
python eval_refine_gate.py \
  --config configs/wikitext2_gpt2_mdlm_learned_gate.yaml \
  --gate_ckpt results/wikitext2_gpt2_mdlm_learned_gate_top50/learned_gate.pt
```

Keep `refine_window: 0`.

Optional MDLM smoke test:

```bash
python validate_pretrained_mdlm.py \
  --config configs/wikitext2_gpt2_mdlm_learned_gate.yaml \
  --forward-check
```

`--mdlm_ckpt` is optional and should only be passed when an extra MDLM checkpoint
exists. The current learned-gate experiment does not require it.

## Outputs

Current learned-gate results are written to:

```text
results/wikitext2_gpt2_mdlm_learned_gate/
```

Files:

- `gate_train_metrics.csv`
- `gate_train_metrics.json`
- `learned_gate_eval.csv`
- `learned_gate_eval.json`
- `learned_gate_summary.md`
- `learned_gate.pt`
- `best_gate_config.json`

The summary states that this round uses the local pretrained MDLM without an
extra training checkpoint, reports the current learned-gate result, and answers
whether the results support edge-side MDLM verification/refinement of
device-side GPT-2 drafts.
