#!/usr/bin/env python3
"""
Quick CPU test for running a Transformers causal LM end-to-end using the local
SmolLM cache, while keeping torch==2.2.2. This verifies whether we can
instantiate AutoModelForCausalLM and generate a short continuation without GPU
or quantization.

It does NOT modify project code; it's an isolated sanity test.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only Transformers generate test")
    parser.add_argument(
        "--model_dir",
        default="memories/kernel/HuggingFaceTB_SmolLM_360M",
        help="Local directory for the model/tokenizer (default: local SmolLM cache)",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, how are you?",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="Number of new tokens to generate (default: 40)",
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as exc:
        print(f"[error] Missing deps: {exc}")
        return 1

    is_local_dir = os.path.isdir(args.model_dir)
    model_ref = os.path.abspath(args.model_dir) if is_local_dir else args.model_dir

    print(f"[info] torch: {getattr(torch, '__version__', 'unknown')}")
    print(f"[info] loading tokenizer from: {model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=is_local_dir)
    # Ensure pad token id exists for CPU open-end generation warnings
    if tokenizer.pad_token_id is None:
        # Many chat LMs reuse eos as pad
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[info] set tokenizer.pad_token -> eos ({tokenizer.pad_token!r})")

    print(f"[info] loading model from: {model_ref} (CPU)")
    t0 = time.time()
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=is_local_dir,
            torch_dtype=torch.float32,
            device_map=None,
        )
        .to("cpu")
        .eval()
    )
    t1 = time.time()
    print(f"[info] model loaded in {t1 - t0:.2f}s")
    # Align model pad token id if missing
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"[info] set model.config.pad_token_id = {model.config.pad_token_id}")

    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    print(f"[info] input_ids len: {input_ids.shape[-1]}")

    # Generate (short, deterministic)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print("\n===== GENERATION OUTPUT =====")
    print(text)
    print("===== END OUTPUT =====\n")

    # Optional: memory footprint
    if hasattr(model, "get_memory_footprint"):
        try:
            mb = model.get_memory_footprint() / 1e6
            print(f"[info] memory footprint: {mb:.2f} MB")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
