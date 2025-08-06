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


def exon_product_from_state(state_index: int, theta: float, orbit_size: int) -> int:
    """
    Project the 48-bit state tensor to an 8-bit exon product using theta and orbit size.

    This provides the baseline mask when no specific learning exists for a (state, token) pair.
    The exon product encodes the state's intrinsic resonance characteristics.

    Args:
        state_index: Canonical index of the physical state
        theta: Angular divergence from the archetypal state (radians)
        orbit_size: Cardinality of the state's phenomenological orbit

    Returns:
        8-bit exon product encoding state's baseline characteristics
    """
    # Theta normalization: map [0, π] to [0, 1]
    theta_norm = min(1.0, theta / math.pi)

    # Orbit variety normalization: larger orbits have lower specificity
    orbit_norm = min(1.0, orbit_size / 1000.0)  # Assume max orbit size ~1000

    # Extract bit families from state_index for deterministic projection
    li_component = (state_index >> 6) & 0x03
    fg_component = (state_index >> 4) & 0x03
    bg_component = (state_index >> 2) & 0x03
    neutral_component = state_index & 0x03

    # Apply theta modulation (cooling effect)
    cooling_factor = 1.0 - theta_norm
    heating_factor = theta_norm

    # Compute weighted components
    tau = int((li_component - bg_component) * cooling_factor) & 0x0F
    eta = int((fg_component - neutral_component) * heating_factor) & 0x0F

    # Apply orbit size modulation
    orbit_factor = int(orbit_norm * 15) & 0x0F

    # Combine into 8-bit product with bit family structure
    exon_product = ((tau & 0x0F) << 4) | ((eta ^ orbit_factor) & 0x0F)

    # Ensure non-zero result
    if exon_product == 0:
        exon_product = 0b01000010  # LI bits set (cooling intron)

    return exon_product & 0xFF


def propose_resonant_introns(exon_product: int, max_candidates: int = 3) -> List[int]:
    """
    Convert exon product to candidate intron bytes for tokenizer trie lookup.

    Uses bit family coherence rules to generate 1-3 intron candidates that are
    most likely to reduce divergence based on the current exon product state.

    Args:
        exon_product: 8-bit exon product from state physics or learned mask
        max_candidates: Maximum number of intron candidates to return

    Returns:
        List of 1-3 intron bytes (0-255) for tokenizer trie lookup
    """
    candidates = []

    # Extract bit families from exon product
    high_nibble = (exon_product >> 4) & 0x0F
    low_nibble = exon_product & 0x0F

    # Primary candidate: direct resonance (minimal perturbation)
    primary = exon_product
    candidates.append(primary)

    if max_candidates > 1:
        # Secondary candidate: LI coherence (flip parity bits)
        li_flip = exon_product ^ EXON_LI_MASK
        candidates.append(li_flip & 0xFF)

    if max_candidates > 2:
        # Tertiary candidate: FG/BG stress relief (flip gyration bits with highest stress)
        if high_nibble > low_nibble:
            # High stress in tau component: apply FG correction
            fg_correction = exon_product ^ EXON_FG_MASK
        else:
            # High stress in eta component: apply BG correction
            fg_correction = exon_product ^ EXON_BG_MASK

        candidates.append(fg_correction & 0xFF)

    # Remove duplicates while preserving order
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    return unique_candidates[:max_candidates]


def token_last_intron(token_id: int) -> int:
    """
    Get the last intron byte for a token ID using the ψ isomorphism.

    Converts token_id to LEB128 bytes, applies XOR 0xAA transformation,
    and returns the final byte which encodes the token's decisive action.

    Args:
        token_id: Token ID from the tokenizer

    Returns:
        Last intron byte (0-255) for this token
    """
    if token_id < 0:
        raise ValueError("Token ID must be non-negative")

    # Convert to LEB128 bytes
    leb_bytes = []
    val = token_id
    while True:
        byte = val & 0x7F
        val >>= 7
        if val == 0:
            leb_bytes.append(byte)
            break
        else:
            leb_bytes.append(byte | 0x80)

    # Apply ψ isomorphism (XOR with 0xAA) to get introns
    introns = [b ^ GENE_Mic_S for b in leb_bytes]

    # Return the last (decisive) intron
    return introns[-1] if introns else 0





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
