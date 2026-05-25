"""Smoke-test that the configured MDLM edge model is truly pretrained."""

from __future__ import annotations

import argparse

import torch
import yaml

from data_real import load_tokenizer
from models_mdlm_wrapper import build_edge_mdlm_model


def choose_device() -> torch.device:
    """Use CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--forward-check", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    tokenizer = load_tokenizer(
        config["tokenizer_name"],
        local_files_only=bool(config.get("hf_local_files_only", False)),
    )
    model = build_edge_mdlm_model(
        config,
        len(tokenizer),
        int(tokenizer.pad_token_id),
        int(tokenizer.mask_token_id),
    )

    print(f"model_backend={getattr(model, 'backend_name', 'unknown')}")
    print(f"pretrained_edge_loaded={getattr(model, 'pretrained_loaded', False)}")
    print(f"edge_hidden_size={getattr(model, 'edge_hidden_size', 'unknown')}")
    print(f"edge_model_status={getattr(model, 'load_message', 'unknown')}")

    if not getattr(model, "pretrained_loaded", False):
        raise RuntimeError("Configured edge model is not a loaded pretrained MDLM checkpoint.")

    if args.forward_check:
        device = choose_device()
        model.to(device).eval()
        seq_len = min(16, int(config["max_length"]))
        input_ids = torch.full((1, seq_len), int(tokenizer.eos_token_id), dtype=torch.long, device=device)
        input_ids[:, seq_len // 2] = int(tokenizer.mask_token_id)
        timesteps = torch.full((1,), float(config["mask_ratio"]), device=device)
        with torch.no_grad():
            outputs = model(input_ids, timesteps)
        print(f"forward_logits_shape={tuple(outputs['logits'].shape)}")


if __name__ == "__main__":
    main()
