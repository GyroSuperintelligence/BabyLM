"""
governance.py - S1 Core tensor operations for GyroSI Baby LM

This module implements the foundational tensor structures and operations for the
GyroSI Baby LM, representing the Governance (S1) layer of the Common Governance Model.
It contains immutable constants and pure functions with no state.
"""

import numpy as np
from typing import List, Tuple

# 3.1.1 Governance Identity (Gene Com)
gene_com = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.int8)  # Shape: [3, 2]

# 3.1.2 Information Structure (Gene Nest)
gene_nest = np.array(
    [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]], dtype=np.int8  # Frame 1  # Frame 2
)  # Shape: [2, 3, 2]

# 3.1.3 Intelligence Projection (Gene Add)
gene_add = np.array(
    [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ],
    dtype=np.int8,
)  # Shape: [4, 2, 3, 2]

# Global invariant for inference
gene_stateless = 0xAA  # 10101010 in binary

_L0 = [0, 7]  # Identity
_LI = [1, 6]  # Inverse
_FG = [2, 5]  # Forward Gyration
_BG = [3, 4]  # Backward Gyration


def apply_operation(T: np.ndarray, bit_index: int) -> np.ndarray:
    """Apply a discrete operation to the tensor T based on the bit index."""
    T_new = T.copy()
    if bit_index in _LI:
        T_new *= -1
    elif bit_index in _FG:
        T_new[[0, 2]] *= -1
    elif bit_index in _BG:
        T_new[[1, 3]] *= -1
    # No action needed for _L0 (identity)
    return T_new


def gyrodistance(T1: np.ndarray, T2: np.ndarray) -> float:
    """
    Calculate gyroscopic distance between two tensors

    Args:
        T1: First tensor
        T2: Second tensor

    Returns:
        float: Distance value (smaller means closer match)
    """
    # Flatten tensors to make comparison easier
    flat_T1 = T1.flatten()
    flat_T2 = T2.flatten()

    # Calculate dot product (correlation)
    dot_product = np.dot(flat_T1, flat_T2)

    # Normalize by tensor size
    normalized_distance = dot_product / flat_T1.size

    # Convert to angular distance (0 to pi)
    # Perfect match: normalized_distance = 1 → angular_distance = 0
    # Perfect mismatch: normalized_distance = -1 → angular_distance = pi
    angular_distance = np.arccos(np.clip(normalized_distance, -1.0, 1.0))

    return angular_distance


def derive_canonical_patterns() -> Tuple[np.ndarray, List[str]]:
    """
    Derive the 256 canonical patterns representing all possible operation combinations

    Returns:
        Tuple containing:
        - patterns: Array of shape [256, 48] containing all canonical patterns
        - gyration_features: List of 256 class labels for each pattern
    """
    patterns = []
    gyration_features = []

    # Start with base tensor
    base_tensor = gene_add.copy().astype(np.float32)

    # Generate all 256 possible operation combinations (2^8)
    for mask in range(256):
        # Make a fresh copy of the base tensor
        T = base_tensor.copy()

        # Apply operations according to bits in the mask
        for i in range(8):
            if mask & (1 << i):
                T = apply_operation(T, i)

        # Store the flattened pattern
        patterns.append(T.flatten())

        # Classify the pattern's resonance
        gyration_feature = classify_pattern_resonance(mask)
        gyration_features.append(gyration_feature)

    # Convert list to numpy array
    patterns_array = np.array(patterns, dtype=np.float32)

    return patterns_array, gyration_features


def classify_pattern_resonance(mask: int) -> str:
    """
    Classify a pattern's resonance based on its bit mask

    Args:
        mask: Integer representation of the bit pattern (0-255)

    Returns:
        str: Resonance class ("identity", "inverse", "forward", or "backward")
    """
    # Count bits for each operation type
    l0_count = bin(mask & 0b10000001).count("1")  # bits 0,7
    li_count = bin(mask & 0b01000010).count("1")  # bits 1,6
    fg_count = bin(mask & 0b00100100).count("1")  # bits 2,5
    bg_count = bin(mask & 0b00011000).count("1")  # bits 3,4

    counts = [l0_count, li_count, fg_count, bg_count]
    max_count = max(counts)

    # Classify based on which operation is most frequent
    if l0_count == max_count:
        return "identity"  # Governance Traceability
    elif li_count == max_count:
        return "inverse"  # Information Variety
    elif fg_count == max_count:
        return "forward"  # Inference Accountability
    else:
        return "backward"  # Intelligence Integrity
