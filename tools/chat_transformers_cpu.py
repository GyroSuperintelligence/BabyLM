#!/usr/bin/env python3
"""
Interactive CPU chat with a Transformers Causal LM (e.g., SmolLM-360M), while
still using torch==2.2.2. This script:
- Loads tokenizer and model from a local directory or the HF hub
- Uses the tokenizer chat template if present, else a safe fallback
- Handles pad_token/attention_mask correctly
- Lets you talk to the model in a simple REPL (type /exit to quit)

Example:
  .venv/bin/python tools/chat_transformers_cpu.py \
      --model_dir HuggingFaceTB/SmolLM-360M \
      --system "You are a helpful AI assistant."
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Dict


def ensure_pad(tokenizer: Any, model: Any) -> None:
    # Many chat LMs reuse eos as pad
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[info] set tokenizer.pad_token -> eos ({tokenizer.pad_token!r})")
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"[info] set model.config.pad_token_id = {model.config.pad_token_id}")


def has_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def build_inputs(
    tokenizer: Any,
    conversation: List[Dict[str, str]],
    add_generation_prompt: bool = True,
):
    # Prefer tokenizer's built-in chat template
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )

    # Fallback to simple SmolLM-style formatting
    text = ""
    for turn in conversation:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "system":
            # Put system text as a plain prologue
            text += content + "\n"
        else:
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return tokenizer(text, return_tensors="pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU chat with Transformers Causal LM")
    parser.add_argument(
        "--model_dir",
        default="HuggingFaceTB/SmolLM-360M",
        help="Local dir or HF repo id (default: SmolLM-360M)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful AI assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (set <= 0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (ignored if greedy)",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as exc:
        print(f"[error] missing deps: {exc}")
        return 1

    is_local_dir = os.path.isdir(args.model_dir)
    model_ref = os.path.abspath(args.model_dir) if is_local_dir else args.model_dir

    print(f"[info] torch: {getattr(torch, '__version__', 'unknown')}")
    print(f"[info] loading tokenizer from: {model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=is_local_dir)

    print(f"[info] loading model from: {model_ref} (CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=is_local_dir,
        torch_dtype=torch.float32,
        device_map=None,
    ).to("cpu").eval()

    ensure_pad(tokenizer, model)

    # Conversation buffer
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": args.system},
    ]

    print("\n[chat] Type your message. Use /exit to quit.\n")
    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user == "" or user.lower() == "/exit":
            break

        conversation.append({"role": "user", "content": user})

        # Build inputs (append assistant generation prompt)
        inputs = build_inputs(tokenizer, conversation, add_generation_prompt=True)
        if hasattr(inputs, "input_ids"):
            input_ids = inputs.input_ids
            attention_mask = getattr(inputs, "attention_mask", None)
        else:
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

        # Generate
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.temperature and args.temperature > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            })
        else:
            gen_kwargs.update({
                "do_sample": False,
            })

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids.to("cpu"),
                attention_mask=attention_mask.to("cpu") if attention_mask is not None else None,
                **gen_kwargs,
            )

        # Decode only the newly generated part
        new_tokens = gen_ids[0][input_ids.shape[-1]:]
        assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"assistant> {assistant_text}")

        # Append assistant turn to conversation
        conversation.append({"role": "assistant", "content": assistant_text})

    return 0


if __name__ == "__main__":
    sys.exit(main())


