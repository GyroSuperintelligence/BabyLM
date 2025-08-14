#!/usr/bin/env python3
"""
Rebuild a HuggingFace-compatible model.safetensors from our cached chunked
NPZ weights (weights_chunk_*.npz) so AutoModelForCausalLM can load entirely
from the local directory (no network).

Usage:
  .venv/bin/python tools/rebuild_safetensors_from_chunks.py \
      --model_dir memories/kernel/HuggingFaceTB_SmolLM_360M

It will write model.safetensors into the same directory.
Optionally, pass --verify to attempt an AutoModelForCausalLM load afterwards.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict


def load_chunks(model_dir: str) -> Dict[str, "np.ndarray"]:
    import numpy as np

    pattern = os.path.join(model_dir, "weights_chunk_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No chunk files found at {pattern}")
    tensors: Dict[str, "np.ndarray"] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as zf:
            for k in zf.files:
                tensors[k] = zf[k]
    return tensors


def save_safetensors(model_dir: str, tensors: Dict[str, "np.ndarray"]) -> str:
    from safetensors.numpy import save_file

    out_path = os.path.join(model_dir, "model.safetensors")
    save_file(tensors, out_path)
    return out_path


def verify_load(model_dir: str) -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=None,
            device_map=None,
        )
        # Avoid heavy allocate on GPU, keep on CPU and dispose
        del m
        print("[verify] AutoModelForCausalLM.from_pretrained() succeeded.")
    except Exception as exc:
        print(f"[verify] load failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild model.safetensors from cached chunks")
    parser.add_argument(
        "--model_dir",
        default="memories/kernel/HuggingFaceTB_SmolLM_360M",
        help="Directory containing weights_chunk_*.npz and config.json",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Attempt to load the rebuilt model with Transformers",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"[error] model_dir not found: {model_dir}")
        return 1

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"[warn] config.json missing in {model_dir}; Transformers load may fail.")
    else:
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            arch = cfg.get("architectures")
            print(f"[info] config.json architectures: {arch}")
        except Exception:
            pass

    print(f"[info] loading chunked tensors from: {model_dir}")
    tensors = load_chunks(model_dir)
    print(f"[info] loaded {len(tensors)} tensors from chunks")

    out_path = save_safetensors(model_dir, tensors)
    print(f"[info] wrote {out_path}")

    if args.verify:
        verify_load(model_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
