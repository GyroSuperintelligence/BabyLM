#!/usr/bin/env python3

import os
from pathlib import Path


def _abs(path: str, base: Path) -> str:
    """Return absolute path, expanding user and joining with base if needed."""
    if path is None:
        raise ValueError("Path must not be None")
    p = Path(os.path.expanduser(str(path)))
    return str(p if p.is_absolute() else base / p)


# Test the path resolution
base_path = Path(__file__).resolve().parents[0]  # Project root
print(f"Base path: {base_path}")

test_path = "toys/training/Archive/wikipedia_simple.bin"
resolved_path = _abs(test_path, base_path)
print(f"Test path: {test_path}")
print(f"Resolved path: {resolved_path}")
print(f"File exists: {os.path.exists(resolved_path)}")

# Also test with the baby directory as base
baby_base = Path(__file__).resolve().parents[0] / "baby"
print(f"\nBaby base path: {baby_base}")
resolved_path_baby = _abs(test_path, baby_base)
print(f"Resolved path with baby base: {resolved_path_baby}")
print(f"File exists: {os.path.exists(resolved_path_baby)}")
