"""
g2_intelligence_eg.py - Gyration Primitives

Pure computational helpers for gyration operations.
No storage or side-effects.
"""

from typing import Tuple
import torch
import s1_governance

# Import the canonical operation codes
OP_CODES = s1_governance.OP_CODES


def byte_to_gyrations(byte_val: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Map one byte to two op-pairs.
    Each op-pair consists of (op_code, tensor_id).
    The byte is split into two 4-bit segments:
    - High nibble (bits 7-4) → first op-pair
    - Low nibble (bits 3-0) → second op-pair
    Each 4-bit segment encodes:
    - Bits 3-1: operation code (0-3)
    - Bit 0: tensor ID (0 or 1)
    Returns:
        Tuple of two op-pairs: ((op_code1, tensor_id1), (op_code2, tensor_id2))
    """
    if not 0 <= byte_val <= 255:
        raise ValueError("byte_val must be 0-255")
    high_nibble = (byte_val >> 4) & 0xF
    low_nibble = byte_val & 0xF
    op_code1 = (high_nibble >> 1) & 0x7
    tensor_id1 = high_nibble & 0x1
    op_code2 = (low_nibble >> 1) & 0x7
    tensor_id2 = low_nibble & 0x1
    op_code1 = min(op_code1, 3)
    op_code2 = min(op_code2, 3)
    return ((op_code1, tensor_id1), (op_code2, tensor_id2))


def gyrations_to_byte(op_pair1: Tuple[int, int], op_pair2: Tuple[int, int]) -> int:
    """
    Convert two op-pairs back to a byte.
    Args:
        op_pair1: First op-pair (op_code, tensor_id)
        op_pair2: Second op-pair (op_code, tensor_id)
    Returns:
        Byte value (0-255)
    """
    op_code1, tensor_id1 = op_pair1
    op_code2, tensor_id2 = op_pair2
    high_nibble = ((op_code1 & 0x7) << 1) | (tensor_id1 & 0x1)
    low_nibble = ((op_code2 & 0x7) << 1) | (tensor_id2 & 0x1)
    return (high_nibble << 4) | low_nibble


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
    return s1_governance.gyration_op(tensor, code, clone=clone)
