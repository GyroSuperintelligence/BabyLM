#!/usr/bin/env python3
"""
Test running HuggingFace tokenizers (AutoTokenizer + chat template) without
loading any transformer model, on environments with torch 2.2.2.

This script verifies that we can:
- Use AutoTokenizer.from_pretrained with a local cache (or remote fallback)
- Apply the chat template to build prompts
- Encode and decode with skip_special_tokens

It does NOT load any model or require a specific transformers-torch coupling.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Any


def print_torch_info() -> None:
    try:
        import torch  # type: ignore

        version = getattr(torch, "__version__", "unknown")
        print(f"[torch] version: {version}")
        # Optional: warn if not 2.2.2
        if version != "2.2.2":
            print("[warn] torch is not 2.2.2; tokenizer-only usage should still work.")
    except Exception as exc:
        print(f"[info] torch not available or failed to import: {exc}")


def load_tokenizer(path_or_repo: str):
    from transformers import AutoTokenizer  # import only tokenizer API

    # Prefer local files if present
    local_dir = os.path.expanduser(path_or_repo)
    if os.path.isdir(local_dir):
        print(f"[tokenizer] loading from local dir: {local_dir}")
        return AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

    # Fallback: remote
    print(f"[tokenizer] loading from hub: {path_or_repo}")
    return AutoTokenizer.from_pretrained(path_or_repo)


def apply_chat_template(tokenizer: Any, user_prompt: str) -> List[int]:
    # Prefer tokenizer's built-in chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        conversation = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_prompt},
        ]
        try:
            out = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            if isinstance(out, list):
                return out
            if hasattr(out, "input_ids"):
                ids = out.input_ids
                return ids if isinstance(ids, list) else list(ids)
        except Exception as exc:
            print(f"[warn] apply_chat_template failed, falling back: {exc}")

    # Minimal fallback if template not available
    templated = (
        "You are a helpful AI assistant.\n<|im_start|>user\n" + user_prompt +
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    return tokenizer.encode(templated)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test tokenizer-only usage with chat template")
    parser.add_argument(
        "--tokenizer",
        default="memories/kernel/HuggingFaceTB_SmolLM_360M",
        help="Local tokenizer dir or HF repo id (default: local SmolLM cache)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="User prompt to encode",
    )
    args = parser.parse_args()

    print_torch_info()

    try:
        tokenizer = load_tokenizer(args.tokenizer)
    except Exception as exc:
        print(f"[error] failed to load tokenizer: {exc}")
        return 1

    # Report special tokens & chat template availability
    try:
        specials = getattr(tokenizer, "special_tokens_map", {})
        print(f"[tokenizer] vocab size: {tokenizer.vocab_size}")
        print(f"[tokenizer] special tokens map: {specials}")
        template = getattr(tokenizer, "chat_template", None)
        print(f"[tokenizer] chat template present: {bool(template)}")
    except Exception:
        pass

    # Build tokens via chat template
    token_ids = apply_chat_template(tokenizer, args.prompt)
    print(f"[encode] token count: {len(token_ids)} (first 20): {token_ids[:20]}")

    # Decode a fake generated sequence (echo) skipping specials
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"[decode] (skip_special_tokens) preview: {decoded[:200]!r}")
    except Exception as exc:
        print(f"[warn] decode failed: {exc}")

    print("[success] tokenizer-only path works without loading a model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


