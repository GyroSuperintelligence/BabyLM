#!/usr/bin/env python3
"""
Build the universal GyroSI harmonics matrix (resonance mask + operator vector).
Output layout (all byte counts are fixed):
    32  bytes : SHA-256(Gene)                     – integrity anchor
    48*256 b : resonance mask   M  (48 rows × 256 bits → 1536 bytes)
    48  bytes : operator vector O (one byte / phase)
   ---------------------------------------------------------------
    1616 bytes total
"""
import os
import sys
import hashlib
import numpy as np
import torch
from core.g1 import _OP_CODES


# ------------------------------------------------------------------ #
# Immutable Gene (copied verbatim from g1)
# ------------------------------------------------------------------ #
def gene_const() -> dict:
    pattern = [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ]
    base = torch.tensor(pattern, dtype=torch.int8)
    return {"id_0": base.clone(), "id_1": base.clone()}


# ------------------------------------------------------------------ #
# Helper – given phase, return ±1 pair stored in the immutable Gene
# ------------------------------------------------------------------ #
def slice_at_phase(phase: int) -> tuple:
    g = gene_const()
    tid = phase % 2
    pos = (phase // 2) % 24
    out, inn, sp = pos // 6, (pos // 3) % 2, pos % 3
    sl = g[f"id_{tid}"][out][inn][sp]
    return int(sl[0]), int(sl[1])


# ------------------------------------------------------------------ #
# Build resonance mask M[48,256]  (bit-packed rows: uint8 per column)
# ------------------------------------------------------------------ #
def build_mask() -> np.ndarray:
    mask = np.zeros((48, 256), dtype=np.uint8)
    for p in range(48):
        s0, s1 = slice_at_phase(p)
        for b in range(256):
            hi, lo = (b >> 4) & 0xF, b & 0xF
            if (1 if hi >= 8 else -1) == s0 and (1 if lo >= 8 else -1) == s1:
                mask[p, b] = 1
    return mask


# ------------------------------------------------------------------ #
# Build operator vector O[48]  (one byte per phase, packed as id₁<<4|id₀)
# ------------------------------------------------------------------ #
def build_operator_vector() -> np.ndarray:
    operator_vector = np.zeros(48, dtype=np.uint8)
    for p in range(48):
        if p % 12 == 0:  # CS
            op0, op1 = _OP_CODES["IDENTITY"], _OP_CODES["INVERSE"]
        elif p % 12 in (3, 9):  # UNA/ONA
            base = _OP_CODES["FORWARD"] if p % 24 < 12 else _OP_CODES["BACKWARD"]
            op0 = op1 = base
        elif p % 6 == 0:  # Nesting
            op0 = op1 = _OP_CODES["BACKWARD"]
        else:  # unreachable (no resonance)
            op0 = op1 = 0
        operator_vector[p] = (op1 << 4) | (op0 & 0x0F)
    return operator_vector


# ------------------------------------------------------------------ #
# Main builder
# ------------------------------------------------------------------ #
def main(out_path: str):
    mask = build_mask()
    operator_vector = build_operator_vector()
    h = hashlib.sha256()
    g = gene_const()
    h.update(g["id_0"].numpy().tobytes())
    h.update(g["id_1"].numpy().tobytes())
    gene_digest = h.digest()  # 32 B
    payload = gene_digest + mask.tobytes() + operator_vector.tobytes()  # 1616 B
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(payload)
    print(f"✓ Universal harmonics matrix written → {out_path}")
    print(f"  Size: {len(payload)} bytes")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: build_operator_matrix.py <output_file>")
        sys.exit(1)
    main(sys.argv[1])


"""
tool_g1_alignment.py - Tools for G1 (Genetic Memory) Alignment
"""

class G1AlignmentTool:
    """Placeholder for G1 alignment tool logic."""
    pass
