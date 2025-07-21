"""
S1: Governance - Physics & Primitives

This module contains the fundamental constants and pure functions that define
the physical laws of the GyroSI system. No engine class is needed here;
all operations are stateless functions.
"""

import numpy as np
from functools import reduce
from typing import List, Tuple, Any, cast


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


# These masks operate on the expressed exon_mask (i.e., exons)
EXON_LI_MASK = 0b01000010  # UNA   bits (Parity / Reflection)
EXON_FG_MASK = 0b00100100  # ONA   bits (Forward Gyration)
EXON_BG_MASK = 0b00011000  # BU‑Eg bits (Backward Gyration)
EXON_DYNAMIC_MASK = EXON_LI_MASK | EXON_FG_MASK | EXON_BG_MASK  # All active bits

EXON_BROADCAST_MASKS = {"li": EXON_LI_MASK, "fg": EXON_FG_MASK, "bg": EXON_BG_MASK, "dynamic": EXON_DYNAMIC_MASK}


def compute_governance_signature(mask: int) -> tuple[int, int, int, int, int]:
    """
    Returns an immutable 5‑tuple:

      (neutral_reserve, li_bits, fg_bits, bg_bits, dynamic_population)

    – neutral_reserve : 6 − (# set dynamic bits)
    – li_bits         : # set bits in LI group   (0‑2)
    – fg_bits         : # set bits in FG group   (0‑2)
    – bg_bits         : # set bits in BG group   (0‑2)
    – dynamic_population = li_bits + fg_bits + bg_bits  (0‑6)
    """
    m = mask & 0xFF
    li = (m & EXON_LI_MASK).bit_count()
    fg = (m & EXON_FG_MASK).bit_count()
    bg = (m & EXON_BG_MASK).bit_count()
    dyn = li + fg + bg
    return (6 - dyn, li, fg, bg, dyn)


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
    intron_broadcast_masks_list: List[int] = []
    for i in range(256):
        mask = 0
        for j in range(6):
            mask |= i << (8 * j)
        intron_broadcast_masks_list.append(mask)

    return FG, BG, FULL_MASK, intron_broadcast_masks_list


# Pre-compute the masks at module load time
FG_MASK, BG_MASK, FULL_MASK, INTRON_BROADCAST_MASKS_LIST = build_masks_and_constants()
INTRON_BROADCAST_MASKS: np.ndarray = np.array(INTRON_BROADCAST_MASKS_LIST, dtype=np.uint64)

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
    intron &= 0xFF  # Defensive masking
    # Step 1: Gyro-addition (applying transformational forces) using precomputed mask
    temp_state = state_int ^ int(XFORM_MASK[intron])

    # Step 2: Thomas Gyration (path-dependent memory)
    intron_pattern = int(INTRON_BROADCAST_MASKS[intron])
    gyration = temp_state & intron_pattern
    final_state = temp_state ^ gyration

    return final_state


def apply_gyration_and_transform_batch(states: np.ndarray, intron: int) -> "np.ndarray[np.uint64, Any]":
    """
    Vectorised transform for a batch of states (uint64).
    Semantics identical to apply_gyration_and_transform per element.
    """
    intron &= 0xFF  # Defensive masking
    mask = XFORM_MASK[intron]
    pattern = PATTERN_MASK[intron]
    temp = states ^ mask
    return cast("np.ndarray[np.uint64, Any]", (temp ^ (temp & pattern)).astype(np.uint64))


def apply_gyration_and_transform_all_introns(states: np.ndarray) -> "np.ndarray[np.uint64, Any]":
    """
    Returns an array shape (states.size, 256) of successor states.
    Memory-heavy; prefer intron loop for large batches.
    """
    temp = states[:, np.newaxis] ^ XFORM_MASK[np.newaxis, :]
    return cast("np.ndarray[np.uint64, Any]", (temp ^ (temp & PATTERN_MASK[np.newaxis, :])).astype(np.uint64))


# Optional: test helper for equivalence (not run by default)
def _test_vector_equivalence() -> None:
    rng = np.random.default_rng(0)
    sample_states = rng.integers(0, 1 << 48, size=4096, dtype=np.uint64)
    for intron in range(256):
        scalar = np.array(
            [apply_gyration_and_transform(int(s), intron) for s in sample_states],
            dtype=np.uint64,
        )
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


MASK = 0xFF  # 8-bit mask


def fold(a: int, b: int) -> int:
    """
    The Monodromic Fold (⋄), the single, unified learning operator for BU.
    Formula: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))

    This operation is the algebraic expression of the BU stage's dual monodromy.
    It is fundamentally non-associative and non-commutative, preserving the
    path-dependence required by the Common Source axiom.

    Its algebraic properties, discovered empirically, are:
    - Left Identity (CS Emergence):   fold(0, b) = b
    - Right Absorber (Return to CS):  fold(a, 0) = 0
    - Self-Annihilation (BU Closure): fold(a, a) = 0
    """
    a &= MASK
    b &= MASK
    # This is the formula from your successful experiments: a ^ (b ^ (a & ~b))
    gyration_of_b = b ^ (a & (~b & MASK))
    return (a ^ gyration_of_b) & MASK


def fold_sequence(introns: List[int], start_state: int = 0) -> int:
    """
    Performs an ordered reduction of a sequence of introns using the
    Monodromic Fold. This is the only valid form of batching.

    Args:
        introns: A list of 8-bit intron values.
        start_state: The initial state to begin the fold from.

    Returns:
        The final state after the entire sequence has been folded.
    """
    if not introns:
        return start_state
    # The reduce function correctly applies the fold sequentially:
    # fold(fold(fold(start, i1), i2), i3)...
    return reduce(fold, introns, start_state)


def dual(x: int) -> int:
    """
    The Global Duality Operator (¬), corresponding to the 'Fifth Element'.
    It reflects a state through the origin, enabling the return path.
    """
    return (x ^ 0xFF) & MASK


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
