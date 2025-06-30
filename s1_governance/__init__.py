"""
s1_governance/__init__.py - GyroSI Baby LM Governance Module

Defines immutable tensor mechanics and byte↔operation mapping.
No storage, no inference logic - pure definitions and transformations.

Device logic: All tensors are created on the selected device (GPU if available, else CPU).
"""

import torch

# Select device for all tensors and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import hashlib
import os
from typing import Dict, Tuple, Optional


# Gyration operator codes
_OP_CODES = {
    "IDENTITY": 0,  # Left Identity: no transformation
    "INVERSE": 1,  # Left Inverse: global sign flip
    "FORWARD": 2,  # Forward Gyration: flip rows 0 and 2
    "BACKWARD": 3,  # Backward Gyration: flip rows 1 and 3
}

# Global immutable gene tensors - NEVER CHANGE THESE
_GENE_TENSORS: Optional[Dict[str, torch.Tensor]] = None


def _initialize_gene_tensors():
    """Initialize the immutable gene tensors once"""
    global _GENE_TENSORS
    if _GENE_TENSORS is None:
        gene_pattern = [
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        ]
        base_tensor = torch.tensor(
            gene_pattern, dtype=torch.int8, requires_grad=False, device=device
        )
        _GENE_TENSORS = {
            "id_0": base_tensor.clone().requires_grad_(False).to(device),
            "id_1": base_tensor.clone().requires_grad_(False).to(device),
        }
        # Make tensors truly immutable
        for tensor in _GENE_TENSORS.values():
            tensor.requires_grad_(False)


def get_gene_anchor() -> bytes:
    """
    Return the constant 32-byte anchor (SHA-256 of immutable gene tensors).
    This is used for pack headers and should never change.
    """
    gene = get_gene_tensors()
    hasher = hashlib.sha256()
    hasher.update(gene["id_0"].numpy().tobytes())
    hasher.update(gene["id_1"].numpy().tobytes())
    return hasher.digest()


def get_gene_constant() -> Dict[str, torch.Tensor]:
    """
    Return READ-ONLY copies of the immutable Gene tensors.
    These define the invariant topological space.
    """
    _initialize_gene_tensors()
    if _GENE_TENSORS is None:
        # This branch should be logically unreachable if _initialize_gene_tensors works.
        raise RuntimeError("Fatal error: Gene tensors did not initialize.")
    return {
        "id_0": _GENE_TENSORS["id_0"].clone().to(device),
        "id_1": _GENE_TENSORS["id_1"].clone().to(device),
    }


def get_gene_tensors() -> Dict[str, torch.Tensor]:
    """
    Return a copy of the gene tensors for read-only access.
    NEVER mutate these - they define the invariant topology.
    """
    return get_gene_constant()


def get_baseline_epigenome() -> torch.Tensor:
    """
    Return the baseline Epigenome mask: a 48×8-bit tensor (384 bits total).
    Initialized to zeros - will be populated by the epigenome projection.
    """
    return torch.zeros(48, 8, dtype=torch.uint8, device=device)


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
    Apply a gyration transformation to a tensor.

    IMPORTANT: This function ALWAYS operates on copies when clone=True.
    The original Gene tensors should NEVER be mutated.

    Parameters:
        tensor: The 4×2×3×2 tensor to transform
        code: Gyration operator code (0-3)
        clone: If True, operate on a copy (RECOMMENDED). If False, mutate in place

    Returns:
        Transformed tensor (new tensor if clone=True, modified original if clone=False)
    """
    if clone:
        result = tensor.clone()
    else:
        result = tensor

    if code == 0:
        # Left Identity Operator: no transformation
        pass
    elif code == 1:
        # Left Inverse Operator: global sign flip
        result = result * -1
    elif code == 2:
        # Forward Gyration Operator: flip rows 0 and 2
        result[0] = result[0] * -1
        result[2] = result[2] * -1
    elif code == 3:
        # Backward Gyration Operator: flip rows 1 and 3
        result[1] = result[1] * -1
        result[3] = result[3] * -1
    else:
        raise ValueError(f"Unsupported gyration code: {code}")

    return result


def byte_to_gyrations(byte_val: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Convert a byte value to two gyration operations.
    This is the universal codec entry point.
    """
    if not 0 <= byte_val <= 255:
        raise ValueError("byte_val must be 0-255")

    # High nibble for first operation
    hi_nibble = (byte_val >> 4) & 0xF
    hi_op = (hi_nibble >> 1) & 0x7
    hi_tensor = hi_nibble & 0x1

    # Low nibble for second operation
    lo_nibble = byte_val & 0xF
    lo_op = (lo_nibble >> 1) & 0x7
    lo_tensor = lo_nibble & 0x1

    # Ensure op codes are in valid range (0-3)
    hi_op = min(hi_op, 3)
    lo_op = min(lo_op, 3)

    return ((hi_op, hi_tensor), (lo_op, lo_tensor))


def gyrations_to_byte(op_pair1: Tuple[int, int], op_pair2: Tuple[int, int]) -> int:
    """
    Convert two gyration operation pairs back to a byte.
    This is the inverse of byte_to_gyrations.
    """
    hi_op, hi_tensor = op_pair1
    lo_op, lo_tensor = op_pair2

    # Reconstruct the nibbles
    # High nibble: op is bits 1,2,3; tensor is bit 0
    hi_nibble = ((hi_op & 0x7) << 1) | (hi_tensor & 0x1)

    # Low nibble: op is bits 1,2,3; tensor is bit 0
    lo_nibble = ((lo_op & 0x7) << 1) | (lo_tensor & 0x1)

    # Combine nibbles to form the byte
    byte_val = (hi_nibble << 4) | lo_nibble

    return byte_val


def _extract_slice(tensor: torch.Tensor, phase: int) -> torch.Tensor:
    """Extract tensor slice for phase without mutation"""
    pos = phase % 24
    outer_idx = pos // 6
    inner_idx = (pos // 3) % 2
    spatial_idx = pos % 3
    return tensor[outer_idx][inner_idx][spatial_idx].clone().to(device)


def _find_alignment_operator(phase: int, target_alignment: Tuple[int, int]) -> int:
    """Find which operator achieves target alignment at this phase"""
    gene = get_gene_tensors()
    tensor_id = phase % 2
    tensor_key = f"id_{tensor_id}"

    best_match_score = -1
    best_op_code = 0

    for op_code in range(4):
        # Apply operation to a COPY - never mutate the gene
        temp = gyration_op(gene[tensor_key], op_code, clone=True)
        transformed_slice = _extract_slice(temp, phase)
        transformed_list = transformed_slice.tolist()
        match_score = sum(1 for i in range(2) if transformed_list[i] == target_alignment[i])
        if match_score > best_match_score:
            best_match_score = match_score
            best_op_code = op_code
            if match_score == 2:
                return best_op_code
    return best_op_code


def build_epigenome_projection(
    output_path: str = "s2_information/agency/g2_information/g2_information.dat",
) -> None:
    """Build and save the Epigenome projection table"""
    print("Building Epigenome projection...")
    table = np.zeros((48, 256), dtype=np.uint8)

    for phase in range(48):
        for byte_val in range(256):
            hi = 1 if (byte_val >> 4) & 0xF >= 8 else -1
            lo = 1 if byte_val & 0xF >= 8 else -1
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


# Module constants
OP_CODES = _OP_CODES.copy()

if __name__ == "__main__":
    build_epigenome_projection()
