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
from typing import Dict, Tuple, Optional, List, Union, cast

# Gyration operator codes
_OP_CODES = {
    "IDENTITY": 0,  # Left Identity: no transformation
    "INVERSE": 1,  # Left Inverse: global sign flip
    "FORWARD": 2,  # Forward Gyration: flip rows 0 and 2
    "BACKWARD": 3,  # Backward Gyration: flip rows 1 and 3
}

# Module constants
OP_CODES = _OP_CODES.copy()

# Canonical op-pairs
IDENTITY_OP_PAIR: Tuple[int, int] = (0, 0)  # Status-quo identity op-pair
VOID_OP_PAIR: Tuple[int, int] = (7, 0)  # Canonical padding/void op-pair (op_code=7, tensor_id=0)

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

    The transformation applies one of four operations to the tensor:
    - 0: Left Identity - no transformation
    - 1: Left Inverse - global sign flip
    - 2: Forward Gyration - flip rows 0 and 2
    - 3: Backward Gyration - flip rows 1 and 3

    Parameters:
        tensor: The 4×2×3×2 tensor to transform
        code: Gyration operator code (0-3)
        clone: If True, operate on a copy (RECOMMENDED). If False, mutate in place

    Returns:
        Transformed tensor (new tensor if clone=True, modified original if clone=False)
        
    Raises:
        ValueError: If code is not 0-3
    """
    if not 0 <= code <= 3:
        raise ValueError(f"Unsupported gyration code: {code}")
        
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

    return result


def byte_to_gyrations(byte_val: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Map one byte to two op-pairs for CGM processing.

    Each byte is split into high and low nibbles, each encoding an operation pair:
    - High nibble (bits 7-4) → first op-pair
    - Low nibble (bits 3-0) → second op-pair

    Within each nibble:
    - Bits 3-1: operation code (0-3, clamped if higher)
    - Bit 0: tensor ID (0 or 1)

    Args:
        byte_val: A single byte value (0-255)

    Returns:
        Tuple of two op-pairs: ((op_code1, tensor_id1), (op_code2, tensor_id2))

    Raises:
        ValueError: If byte_val is outside the valid range
        TypeError: If byte_val is not an integer
    """
    if not isinstance(byte_val, int):
        raise TypeError(f"Expected integer byte value, got {type(byte_val)}")

    if not 0 <= byte_val <= 255:
        raise ValueError(f"byte_val must be 0-255, got {byte_val}")

    # Extract high and low nibbles
    high_nibble = (byte_val >> 4) & 0xF
    low_nibble = byte_val & 0xF

    # Extract operation code (bits 3-1) and tensor ID (bit 0)
    op_code1 = (high_nibble >> 1) & 0x7
    tensor_id1 = high_nibble & 0x1

    op_code2 = (low_nibble >> 1) & 0x7
    tensor_id2 = low_nibble & 0x1

    # Ensure op codes are valid (0-3)
    op_code1 = min(op_code1, 3)
    op_code2 = min(op_code2, 3)

    return ((op_code1, tensor_id1), (op_code2, tensor_id2))


def gyrations_to_byte(op_pair1: Tuple[int, int], op_pair2: Tuple[int, int]) -> int:
    """
    Convert two operation pairs back to a byte (inverse of byte_to_gyrations).

    Args:
        op_pair1: First op-pair (op_code, tensor_id)
        op_pair2: Second op-pair (op_code, tensor_id)

    Returns:
        Byte value (0-255)

    Raises:
        ValueError: If operation codes or tensor IDs are outside valid ranges
        TypeError: If op_pairs have incorrect format
    """
    # Validate input
    for idx, op_pair in enumerate([op_pair1, op_pair2]):
        if not isinstance(op_pair, tuple) or len(op_pair) != 2:
            raise TypeError(f"op_pair{idx+1} must be a tuple of (op_code, tensor_id)")

        op_code, tensor_id = op_pair
        # Accept op_code 0-7 (0-3 are real, 4-7 are reserved/padding)
        if not 0 <= op_code <= 7:
            raise ValueError(f"op_code must be 0-7, got {op_code}")
        if not 0 <= tensor_id <= 1:
            raise ValueError(f"tensor_id must be 0-1, got {tensor_id}")

    # Extract components
    op_code1, tensor_id1 = op_pair1
    op_code2, tensor_id2 = op_pair2

    # Assemble nibbles
    high_nibble = ((op_code1 & 0x7) << 1) | (tensor_id1 & 0x1)
    low_nibble = ((op_code2 & 0x7) << 1) | (tensor_id2 & 0x1)

    # Combine into byte
    return (high_nibble << 4) | low_nibble


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


def is_void(pair: Tuple[int, int]) -> bool:
    """Return True if this op-pair is the canonical padding/void."""
    return pair == VOID_OP_PAIR


def apply_gyrations_to_tensor(
    tensor: torch.Tensor, gyration_sequence: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Apply a sequence of gyration operations to a tensor.

    This is a convenience function for applying multiple operations in sequence.

    Args:
        tensor: The 4×2×3×2 gene tensor to transform
        gyration_sequence: List of (op_code, tensor_id) pairs to apply

    Returns:
        Transformed tensor with all gyrations applied
    """
    # Ensure tensor is on the correct device
    result = tensor.clone().to(device)

    for op_code, tensor_id in gyration_sequence:
        # Only apply operation if this is the correct tensor
        if tensor_id == 0:  # Only apply if this is id_0
            result = gyration_op(result, op_code, clone=False)

    return result


def batch_gyrations(tensor: torch.Tensor, op_codes: torch.Tensor) -> torch.Tensor:
    """
    Apply gyrations to a tensor in batch mode (vectorized).

    This optimized version uses vectorized operations for better performance.

    Args:
        tensor: The 4×2×3×2 gene tensor to transform
        op_codes: Tensor of operation codes (0-3)

    Returns:
        Batch of transformed tensors
    """
    # Ensure tensor and op_codes are on the correct device
    tensor = tensor.to(device)
    op_codes = op_codes.to(device)
    batch_size = op_codes.size(0)
    result = tensor.repeat(batch_size, 1, 1, 1, 1)

    # Apply operations using vectorized logic
    # Identity (0): do nothing

    # Left Inverse (1): global sign flip
    inverse_mask = op_codes == 1
    if inverse_mask.any():
        result[inverse_mask] = -result[inverse_mask]

    # Forward Gyration (2): flip rows 0 and 2
    forward_mask = op_codes == 2
    if forward_mask.any():
        result[forward_mask, 0] = -result[forward_mask, 0]
        result[forward_mask, 2] = -result[forward_mask, 2]

    # Backward Gyration (3): flip rows 1 and 3
    backward_mask = op_codes == 3
    if backward_mask.any():
        result[backward_mask, 1] = -result[backward_mask, 1]
        result[backward_mask, 3] = -result[backward_mask, 3]

    return result


def build_epigenome_projection(
    output_path: str = "s2_information/agency/g2_information/g2_information.dat",
) -> None:
    """Build and save the Epigenome projection table (48x256).
    The output file contains only 12,288 bytes of table data (no SHA-256 header).
    """
    print("Building Epigenome projection...")
    table = np.zeros((48, 256), dtype=np.uint8)

    for phase in range(48):
        for byte_val in range(256):
            hi = 1 if (byte_val >> 4) & 0xF >= 8 else -1
            lo = 1 if byte_val & 0xF >= 8 else -1
            table[phase, byte_val] = _find_alignment_operator(phase, (hi, lo))

    # Write file (no SHA header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(table.tobytes())

    size = table.nbytes
    print(f"Epigenome projection saved to {output_path} ({size} bytes)")


if __name__ == "__main__":
    build_epigenome_projection()