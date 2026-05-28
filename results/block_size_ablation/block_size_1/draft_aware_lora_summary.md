# MDLM LoRA Training Summary

- lora_training_mode: `draft_aware`
- Device GPT-2: `/mnt/data/enzeyu/hf_downloads/models/gpt2`
- Edge MDLM: `/mnt/data/enzeyu/hf_downloads/models/mdlm-no_flashattn-fp32-owt`
- train_steps: `10000`
- block_size: `1`
- refine_ratio: `0.2`
- LoRA rank/alpha/dropout: `8/16/0.05`
- total_parameters: `209405778`
- trainable_parameters: `1179648`
- trainable_ratio: `0.00563331`

## LoRA Target Modules

- `edge_model.model.backbone.blocks.0.attn_qkv`
- `edge_model.model.backbone.blocks.0.attn_out`
- `edge_model.model.backbone.blocks.0.mlp.0`
- `edge_model.model.backbone.blocks.0.mlp.2`
- `edge_model.model.backbone.blocks.1.attn_qkv`
- `edge_model.model.backbone.blocks.1.attn_out`
- `edge_model.model.backbone.blocks.1.mlp.0`
- `edge_model.model.backbone.blocks.1.mlp.2`
- `edge_model.model.backbone.blocks.2.attn_qkv`
- `edge_model.model.backbone.blocks.2.attn_out`
- `edge_model.model.backbone.blocks.2.mlp.0`
- `edge_model.model.backbone.blocks.2.mlp.2`
- `edge_model.model.backbone.blocks.3.attn_qkv`
- `edge_model.model.backbone.blocks.3.attn_out`
- `edge_model.model.backbone.blocks.3.mlp.0`
- `edge_model.model.backbone.blocks.3.mlp.2`
- `edge_model.model.backbone.blocks.4.attn_qkv`
- `edge_model.model.backbone.blocks.4.attn_out`
- `edge_model.model.backbone.blocks.4.mlp.0`
- `edge_model.model.backbone.blocks.4.mlp.2`
- `edge_model.model.backbone.blocks.5.attn_qkv`
- `edge_model.model.backbone.blocks.5.attn_out`
- `edge_model.model.backbone.blocks.5.mlp.0`
- `edge_model.model.backbone.blocks.5.mlp.2`
- `edge_model.model.backbone.blocks.6.attn_qkv`
- `edge_model.model.backbone.blocks.6.attn_out`
- `edge_model.model.backbone.blocks.6.mlp.0`
- `edge_model.model.backbone.blocks.6.mlp.2`
- `edge_model.model.backbone.blocks.7.attn_qkv`
- `edge_model.model.backbone.blocks.7.attn_out`
- `edge_model.model.backbone.blocks.7.mlp.0`
- `edge_model.model.backbone.blocks.7.mlp.2`
- `edge_model.model.backbone.blocks.8.attn_qkv`
- `edge_model.model.backbone.blocks.8.attn_out`
- `edge_model.model.backbone.blocks.8.mlp.0`
- `edge_model.model.backbone.blocks.8.mlp.2`
- `edge_model.model.backbone.blocks.9.attn_qkv`
- `edge_model.model.backbone.blocks.9.attn_out`
- `edge_model.model.backbone.blocks.9.mlp.0`
- `edge_model.model.backbone.blocks.9.mlp.2`
- `edge_model.model.backbone.blocks.10.attn_qkv`
- `edge_model.model.backbone.blocks.10.attn_out`
- `edge_model.model.backbone.blocks.10.mlp.0`
- `edge_model.model.backbone.blocks.10.mlp.2`
- `edge_model.model.backbone.blocks.11.attn_qkv`
- `edge_model.model.backbone.blocks.11.attn_out`
- `edge_model.model.backbone.blocks.11.mlp.0`
- `edge_model.model.backbone.blocks.11.mlp.2`

## Last Metrics

- train_loss: `4.599401950836182`
- train_token_acc: `0.32692310214042664`
- train_top5_acc: `0.4711538553237915`
- eval_draft_context_loss: `3.8497444668120084`
- eval_token_acc: `0.3317238135809256`
- eval_top5_acc: `0.5316272480282905`

Run `eval_draft_aware_lora.py` for the full pretrained-vs-LoRA refinement comparison.
