"""
s1_governance.py - GyroSI Baby LM Governance Module

Defines immutable tensor mechanics and byte↔operation mapping.
No storage, no inference logic - pure definitions and transformations.
"""

import torch
import numpy as np
import hashlib
import os
from typing import Dict, Tuple


# Gyration operator codes
_OP_CODES = {
    "IDENTITY": 0,  # Left Identity: no transformation
    "INVERSE": 1,  # Left Inverse: global sign flip
    "FORWARD": 2,  # Forward Gyration: flip rows 0 and 2
    "BACKWARD": 3,  # Backward Gyration: flip rows 1 and 3
}


def get_gene_anchor() -> bytes:
    """
    Return the constant 32-byte anchor (SHA-256 of immutable gene tensors).
    This is used for pack headers and should never change.

    Returns:
        32-byte SHA-256 hash of id_0 and id_1 tensors
    """
    gene = get_gene_tensors()
    hasher = hashlib.sha256()
    hasher.update(gene["id_0"].numpy().tobytes())
    hasher.update(gene["id_1"].numpy().tobytes())
    return hasher.digest()


def get_gene_constant() -> Dict[str, torch.Tensor]:
    """
    Return the immutable Gene tensors (id_0 and id_1).
    Each is a 4×2×3×2 tensor with entries ∈ {-1, +1}.
    """
    gene_pattern = [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ]
    base_tensor = torch.tensor(gene_pattern, dtype=torch.int8)
    return {"id_0": base_tensor.clone(), "id_1": base_tensor.clone()}


def get_gene_tensors() -> Dict[str, torch.Tensor]:
    """
    Return a copy of the gene tensors for initialization.
    This is an alias for get_gene_constant() for compatibility.
    """
    return get_gene_constant()


def get_baseline_epigenome() -> torch.Tensor:
    """
    Return the baseline Epigenome mask: a 48×8-bit tensor (384 bits total).
    Initialized to zeros - will be populated by the epigenome projection.
    """
    return torch.zeros(48, 8, dtype=torch.uint8)


def compute_bit_index(phase: int, op_index: int) -> int:
    """
    Compute the bit index for a given phase and operation index.

    Args:
        phase: Phase value (0-47)
        op_index: Operation index (0-7)

    Returns:
        Bit index in the flattened 384-bit space
    """
    return phase * 8 + op_index


def gyration_op(tensor: torch.Tensor, code: int, clone: bool = True) -> torch.Tensor:
    """
    Apply a gyration transformation to the given tensor.

    Parameters:
        tensor: The 4×2×3×2 gene tensor to transform
        code: Gyration operator code (0-3)
        clone: If True, operate on a copy. If False, mutate in place

    Returns:
        Transformed tensor
    """
    result = tensor.clone() if clone else tensor

    if code == 0:
        # Left Identity Operator: no transformation
        pass
    elif code == 1:
        # Left Inverse Operator: global sign flip
        result *= -1
    elif code == 2:
        # Forward Gyration Operator: flip rows 0 and 2
        result[0] *= -1
        result[2] *= -1
    elif code == 3:
        # Backward Gyration Operator: flip rows 1 and 3
        result[1] *= -1
        result[3] *= -1
    else:
        raise ValueError(f"Unsupported gyration code: {code}")

    return result


def byte_to_gyrations(byte_val: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Convert a byte value to two gyration operations (one per tensor).

    Args:
        byte_val: Byte value (0-255)

    Returns:
        Tuple of two (op_code, tensor_id) pairs
    """
    # High nibble for first operation
    hi_nibble = (byte_val >> 4) & 0xF
    hi_op = (hi_nibble >> 1) & 0x7  # 3 bits for op code
    hi_tensor = hi_nibble & 0x1  # 1 bit for tensor id

    # Low nibble for second operation
    lo_nibble = byte_val & 0xF
    lo_op = (lo_nibble >> 1) & 0x7  # 3 bits for op code
    lo_tensor = lo_nibble & 0x1  # 1 bit for tensor id

    # Ensure op codes are in valid range (0-3)
    hi_op = min(hi_op, 3)
    lo_op = min(lo_op, 3)

    return ((hi_op, hi_tensor), (lo_op, lo_tensor))


def _extract_slice(tensor: torch.Tensor, phase: int) -> torch.Tensor:
    """
    Extract the specific tensor slice for a given phase.

    Args:
        tensor: The gene tensor to slice
        phase: Phase value (0-47)

    Returns:
        The 2-element slice at the specified phase position
    """
    pos = phase % 24  # Position within a single tensor
    outer_idx = pos // 6
    inner_idx = (pos // 3) % 2
    spatial_idx = pos % 3

    return tensor[outer_idx][inner_idx][spatial_idx]


def _find_alignment_operator(phase: int, target_alignment: Tuple[int, int]) -> int:
    """
    Identify which operator code aligns the tensor slice with the target.
    If no perfect match exists, select the best approximation.
    """
    gene = get_gene_tensors()
    tensor_id = phase % 2
    tensor_key = f"id_{tensor_id}"

    # Try each operator and track how many positions match the target
    best_match_score = -1
    best_op_code = 0

    for op_code in range(4):  # Four operations: IDENTITY, INVERSE, FORWARD, BACKWARD
        # Apply operation
        temp = gyration_op(gene[tensor_key].clone(), op_code, clone=False)
        transformed_slice = _extract_slice(temp, phase)
        transformed_list = transformed_slice.tolist()
        match_score = sum(1 for i in range(2) if transformed_list[i] == target_alignment[i])
        if match_score > best_match_score:
            best_match_score = match_score
            best_op_code = op_code
            if match_score == 2:
                return best_op_code
    return best_op_code  # Return best available match


def build_epigenome_projection(
    output_path: str = "s2_information/agency/g2_information/g2_information.dat",
) -> None:
    """
    Build and save the Epigenome projection table (48×256).
    The output file contains a 32-byte SHA-256 header followed by 12,288 bytes of table data.

    Args:
        output_path: Path where the epigenome projection will be saved
    """
    print("Building Epigenome projection...")
    table = np.zeros((48, 256), dtype=np.uint8)

    for phase in range(48):
        for byte_val in range(256):
            # Convert byte to hi/lo alignment values
            hi = 1 if (byte_val >> 4) & 0xF >= 8 else -1
            lo = 1 if byte_val & 0xF >= 8 else -1
            # Find which operator produces this alignment
            table[phase, byte_val] = _find_alignment_operator(phase, (hi, lo))

    # Create integrity header from gene tensors
    gene = get_gene_tensors()
    hasher = hashlib.sha256()
    hasher.update(gene["id_0"].numpy().tobytes())
    hasher.update(gene["id_1"].numpy().tobytes())
    header = hasher.digest()

    # Write file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(table.tobytes())

    size = len(header) + table.nbytes
    print(f"Epigenome projection saved to {output_path} ({size} bytes)")


# Module constants for external access
GENE_TENSORS = get_gene_tensors()
BASELINE_EPIGENOME = get_baseline_epigenome()
OP_CODES = _OP_CODES.copy()


if __name__ == "__main__":
    # Generate the epigenome projection file when run directly
    build_epigenome_projection()
