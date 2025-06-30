"""
g2_intelligence_eg.py - Gyration Primitives

Pure computational helpers for gyration operations.
No storage or side-effects - these functions transform data without mutation.
"""

from typing import Tuple, Union, List, Dict, cast
import torch
import numpy as np
from s1_governance import gyration_op as s1_gyration_op, OP_CODES

# Re-export the canonical operation codes
OP_CODES = OP_CODES.copy()


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
    """
    # Validate input
    for idx, op_pair in enumerate([op_pair1, op_pair2]):
        if not isinstance(op_pair, tuple) or len(op_pair) != 2:
            raise TypeError(f"op_pair{idx+1} must be a tuple of (op_code, tensor_id)")
        
        op_code, tensor_id = op_pair
        if not 0 <= op_code <= 3:
            raise ValueError(f"op_code must be 0-3, got {op_code}")
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


def gyration_op(tensor: torch.Tensor, code: int, clone: bool = True) -> torch.Tensor:
    """
    Apply a gyration transformation to a tensor.
    
    This is a pass-through to the canonical implementation in s1_governance.
    The transformation applies one of four operations to the tensor:
    
    - 0: Left Identity - no transformation
    - 1: Left Inverse - global sign flip
    - 2: Forward Gyration - flip rows 0 and 2
    - 3: Backward Gyration - flip rows 1 and 3
    
    Args:
        tensor: The 4×2×3×2 gene tensor to transform
        code: Gyration operator code (0-3)
        clone: If True, operate on a copy. If False, mutate in place
        
    Returns:
        Transformed tensor
        
    Raises:
        ValueError: If code is not 0-3
    """
    if not 0 <= code <= 3:
        raise ValueError(f"Gyration code must be 0-3, got {code}")
    
    return s1_gyration_op(tensor, code, clone=clone)


def apply_gyrations_to_tensor(
    tensor: torch.Tensor, 
    gyration_sequence: List[Tuple[int, int]]
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
    result = tensor.clone()
    
    for op_code, tensor_id in gyration_sequence:
        # Only apply operation if this is the correct tensor
        if tensor_id == 0:  # Only apply if this is id_0
            result = gyration_op(result, op_code, clone=False)
    
    return result


def batch_gyrations(
    tensor: torch.Tensor,
    op_codes: torch.Tensor
) -> torch.Tensor:
    """
    Apply gyrations to a tensor in batch mode (vectorized).
    
    This optimized version uses vectorized operations for better performance.
    
    Args:
        tensor: The 4×2×3×2 gene tensor to transform
        op_codes: Tensor of operation codes (0-3)
        
    Returns:
        Batch of transformed tensors
    """
    # Create batch of tensors
    batch_size = op_codes.size(0)
    result = tensor.repeat(batch_size, 1, 1, 1, 1)
    
    # Apply operations using vectorized logic
    # Identity (0): do nothing
    
    # Left Inverse (1): global sign flip
    inverse_mask = (op_codes == 1)
    if inverse_mask.any():
        result[inverse_mask] = -result[inverse_mask]
    
    # Forward Gyration (2): flip rows 0 and 2
    forward_mask = (op_codes == 2)
    if forward_mask.any():
        result[forward_mask, 0] = -result[forward_mask, 0]
        result[forward_mask, 2] = -result[forward_mask, 2]
    
    # Backward Gyration (3): flip rows 1 and 3
    backward_mask = (op_codes == 3)
    if backward_mask.any():
        result[backward_mask, 1] = -result[backward_mask, 1]
        result[backward_mask, 3] = -result[backward_mask, 3]
    
    return result