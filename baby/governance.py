"""
S1: Governance - Physics & Primitives

This module contains the fundamental constants and pure functions that define
the physical laws of the GyroSI system. No engine class is needed here;
all operations are stateless functions.
"""

import numpy as np
from functools import reduce
from typing import List, Tuple


# Core genetic constants - Section 4.1 & 4.2
GENE_Mic_S = 0xAA  # 10101010 binary, stateless constant

# GENE_Mac_S: The archetypal 48-byte tensor [4, 2, 3, 2]
GENE_Mac_S = np.array(
    [
        # Layer 0: 0° phase
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        # Layer 1: 180° phase
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        # Layer 2: 360° phase
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        # Layer 3: 540° phase
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ],
    dtype=np.int8,
)


def build_masks_and_constants() -> Tuple[int, int, int, List[int]]:
    """Pre-computes transformation masks based on layer-based physics.

    Returns:
        Tuple of (FG_MASK, BG_MASK, FULL_MASK, INTRON_BROADCAST_MASKS)
    """
    FG, BG = 0, 0

    # Tensor is flattened in C-order (row-major)
    for layer in range(4):
        # Each layer has 2 frames * 3 rows * 2 cols = 12 elements
        for frame in range(2):
            for row in range(3):
                for col in range(2):
                    bit_index = ((layer * 2 + frame) * 3 + row) * 2 + col

                    # FG flips all bits in layers 0 & 2
                    if layer in (0, 2):
                        FG |= 1 << bit_index

                    # BG flips all bits in layers 1 & 3
                    if layer in (1, 3):
                        BG |= 1 << bit_index

    FULL_MASK = (1 << 48) - 1

    # Create broadcast patterns for each possible intron
    INTRON_BROADCAST_MASKS = []
    for i in range(256):
        mask = 0
        for j in range(6):
            mask |= i << (8 * j)
        INTRON_BROADCAST_MASKS.append(mask)

    return FG, BG, FULL_MASK, INTRON_BROADCAST_MASKS


# Pre-compute the masks at module load time
FG_MASK, BG_MASK, FULL_MASK, INTRON_BROADCAST_MASKS = build_masks_and_constants()
INTRON_BROADCAST_MASKS = np.array(INTRON_BROADCAST_MASKS, dtype=np.uint64)

# Precompute transformation masks for all 256 introns
XFORM_MASK = np.empty(256, dtype=np.uint64)
for i in range(256):
    m = 0
    if i & 0b01000010:  # LI
        m ^= FULL_MASK
    if i & 0b00100100:  # FG
        m ^= FG_MASK
    if i & 0b00011000:  # BG
        m ^= BG_MASK
    XFORM_MASK[i] = m
PATTERN_MASK = INTRON_BROADCAST_MASKS  # semantic alias, now a NumPy array


def apply_gyration_and_transform(state_int: int, intron: int) -> int:
    """
    Applies the complete gyroscopic physics transformation.

    This implements gyro-addition followed by Thomas gyration:
    1. Apply transformational forces based on intron bit patterns (using precomputed mask)
    2. Apply path-dependent memory/carry term

    Args:
        state_int: Current 48-bit state as integer
        intron: 8-bit instruction mask

    Returns:
        New 48-bit state after transformation
    """
    # Ensure Python int types for bitwise operations
    state_int = int(state_int)
    intron = int(intron)
    # Step 1: Gyro-addition (applying transformational forces) using precomputed mask
    temp_state = state_int ^ XFORM_MASK[intron & 0xFF]

    # Step 2: Thomas Gyration (path-dependent memory)
    intron_pattern = int(INTRON_BROADCAST_MASKS[intron])
    gyration = temp_state & intron_pattern
    final_state = temp_state ^ gyration

    return final_state


def apply_gyration_and_transform_batch(states: np.ndarray, intron: int) -> np.ndarray:
    """
    Vectorised transform for a batch of states (uint64).
    Semantics identical to apply_gyration_and_transform per element.
    """
    mask = XFORM_MASK[intron]
    pattern = PATTERN_MASK[intron]
    temp = states ^ mask
    return temp ^ (temp & pattern)


def apply_gyration_and_transform_all_introns(states: np.ndarray) -> np.ndarray:
    """
    Returns an array shape (states.size, 256) of successor states.
    Memory-heavy; prefer intron loop for large batches.
    """
    temp = states[:, None] ^ XFORM_MASK[None, :]
    return temp ^ (temp & PATTERN_MASK[None, :])


# Optional: test helper for equivalence (not run by default)
def _test_vector_equivalence():
    rng = np.random.default_rng(0)
    sample_states = rng.integers(0, 1 << 48, size=4096, dtype=np.uint64)
    for intron in range(256):
        scalar = np.array([apply_gyration_and_transform(int(s), intron) for s in sample_states], dtype=np.uint64)
        batch = apply_gyration_and_transform_batch(sample_states, intron)
        assert np.array_equal(scalar, batch)


def transcribe_byte(byte: int) -> int:
    """
    Transforms input byte through XOR with holographic topology.

    Args:
        byte: Input byte (0-255)

    Returns:
        Transcribed intron instruction
    """
    return byte ^ GENE_Mic_S


def coadd(a: int, b: int) -> int:
    """
    Performs true gyrogroup coaddition (a ⊞ b) on two 8-bit integers.

    The operation is: a ⊞ b = a ⊕ gyr[a, ¬b](b)
    where gyr[a, b](c) = c ⊕ (a AND b)

    Note:
        This operation is intentionally non-commutative and non-associative.
        The order matters (a ⊞ b ≠ b ⊞ a), ensuring sequence is preserved.

    Args:
        a: First 8-bit integer
        b: Second 8-bit integer

    Returns:
        Result of gyrogroup coaddition
    """
    not_b = b ^ 0xFF
    gyration_of_b = b ^ (a & not_b)
    return a ^ gyration_of_b


def batch_introns_coadd_ordered(introns: List[int]) -> int:
    """
    Reduces a list of introns into a single representative using ordered
    coaddition. Preserves path-dependence of learning.

    Args:
        introns: List of 8-bit intron values

    Returns:
        Single representative intron from ordered reduction
    """
    if not introns:
        return 0
    return reduce(coadd, introns)


def validate_tensor_consistency() -> bool:
    """
    Validates that GENE_Mac_S has the correct structure and properties.

    Returns:
        True if tensor passes all consistency checks
    """
    # Check shape
    if GENE_Mac_S.shape != (4, 2, 3, 2):
        return False

    # Check data type
    if GENE_Mac_S.dtype != np.int8:
        return False

    # Check that all values are ±1
    unique_vals = np.unique(GENE_Mac_S)
    if not np.array_equal(unique_vals, np.array([-1, 1])):
        return False

    # Check alternating pattern consistency
    expected_pattern = np.array(
        [
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        ],
        dtype=np.int8,
    )

    return np.array_equal(GENE_Mac_S, expected_pattern)


# Validate on module load
if not validate_tensor_consistency():
    raise RuntimeError("GENE_Mac_S failed consistency validation")
