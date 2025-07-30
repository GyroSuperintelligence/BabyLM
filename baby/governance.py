"""
Governance operations for GyroSI - the core physics of recursive structural alignment.

This module implements the fundamental operations of the Common Governance Model (CGM):
- Monodromic Fold (path-dependent learning)
- Gyration operations (LI, FG, BG)
- Exon product computation
- Byte transcription and transformation
"""

import math
from functools import reduce
from typing import List, Tuple, cast

import numpy as np
from numpy.typing import NDArray

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
INTRON_BROADCAST_MASKS: NDArray[np.uint64] = np.array(INTRON_BROADCAST_MASKS_LIST, dtype=np.uint64)

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

# Note: Full activation (LI + FG + BG) results in mask = 0 due to cancellation:
# LI applies FULL_MASK, FG applies FG_MASK, BG applies BG_MASK
# Since FG_MASK ^ BG_MASK = FULL_MASK, the result is FULL_MASK ^ FULL_MASK = 0


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

    # Ensure result stays within 48-bit limit
    final_state &= (1 << 48) - 1

    return final_state


def apply_gyration_and_transform_batch(states: NDArray[np.uint64], intron: int) -> NDArray[np.uint64]:
    """
    Vectorised transform for a batch of states (uint64).
    Semantics identical to apply_gyration_and_transform per element.
    """
    intron &= 0xFF  # Defensive masking
    mask = XFORM_MASK[intron]
    pattern = INTRON_BROADCAST_MASKS[intron]
    temp = states ^ mask
    return cast("NDArray[np.uint64]", (temp ^ (temp & pattern)).astype(np.uint64))


def apply_gyration_and_transform_all_introns(states: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """
    Returns an array shape (states.size, 256) of successor states.
    Memory-heavy; prefer intron loop for large batches.
    """
    temp = states[:, np.newaxis] ^ XFORM_MASK[np.newaxis, :]
    return cast("NDArray[np.uint64]", (temp ^ (temp & INTRON_BROADCAST_MASKS[np.newaxis, :])).astype(np.uint64))


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


def exon_product_from_metadata(mask: int, confidence: float, orbit_v: int, v_max: int) -> int:
    """
    Compress the minimal phenotype metadata into an 8‑bit exon‑product.

    Returns an 8‑bit int using the same LI/FG/BG family layout
    as exon_mask but **never persisted**.

    Guarantee: never return 0 so that PAD is not emitted spuriously.
    """
    # Extract LI/FG/BG components from the mask
    li = (mask >> 6) & 0x03  # bits 6-7
    fg = (mask >> 4) & 0x03  # bits 4-5
    bg = (mask >> 2) & 0x03  # bits 2-3
    neutral = mask & 0x03  # bits 0-1

    tau = li - bg
    eta = fg - neutral

    # v_max must be positive (it's the maximum orbit size)
    if v_max <= 0:
        raise ValueError("v_max must be positive")

    scale = confidence * math.sqrt(orbit_v / v_max)

    A = int(round(tau * scale)) & 0x0F  # high nibble
    B = int(round(eta * scale)) & 0x0F  # low  nibble
    p = ((A << 4) | B) & 0xFF

    # Fallback: inject minimal chirality if result would be 0
    if p == 0:
        p = 0b01000010  # LI bits set (0x42) == cooling intron

    return p


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
