"""
gyro_core.py - Tier 3: The Pure Navigation Engine

This module contains the GyroEngine, the pure computational heart of the GyroSI
system. It is responsible for executing the atomic steps of the navigation cycle.

Key characteristics:
- It is completely stateless regarding session or knowledge context.
- It has no awareness of files, I/O, or external state.
- Its operations are deterministic and computationally pure.
- It holds only the invariant Gene and the transient session phase.

References:
- CORE-SPEC-01: Architecture and Principles
- CORE-SPEC-02: Foundations (Core Mechanics)
- CORE-SPEC-07: Baseline Implementation Specifications
- CORE-SPEC-08: Operational Walkthrough
"""

import torch
from typing import Optional, Tuple, Dict


# ============================================================================
# CONSTANTS
# ============================================================================

# Phase boundaries for operator resonance, per CORE-SPEC-08.
# These are fundamental to the engine's selection logic.
_CS_BOUNDARIES = frozenset({0, 12, 24, 36})
_UNA_ONA_BOUNDARIES = frozenset({3, 9, 15, 21, 27, 33, 39, 45})
_NESTING_BOUNDARIES = frozenset({6, 18, 30, 42})

# Operator codes, per CORE-SPEC-02 and CORE-SPEC-08.
# [3:1] bits for operator, [0] bit for tensor ID.
_OP_CODES = {
    "IDENTITY": 0,  # 000
    "INVERSE": 1,  # 001
    "FORWARD": 2,  # 010
    "BACKWARD": 3,  # 011
}


# ============================================================================
# THE PURE NAVIGATION ENGINE
# ============================================================================


class GyroEngine:
    """
    A pure navigation engine. It holds no direct file handles or complex state,
    only the invariant Gene and the current session's phase.

    This class implements the core computational mechanics of GyroSI:
    - Phase advancement (CS→UNA transition)
    - Structural resonance checking (UNA→ONA transition)
    - Operator code selection (BU_In→BU_Eg transition)

    All operations are deterministic and side-effect free.
    """

    def __init__(self):
        """
        Initializes the engine with its immutable structural components.

        The Gene is created once and never modified. It represents the
        invariant tensor substrate through which all navigation occurs.
        """
        # The Gene is an immutable constant, part of the engine's identity.
        # It is defined once and never changed.
        self.gene: Dict[str, torch.Tensor] = self._get_gene_constant()

        # The phase is the ONLY piece of mutable state the engine tracks directly.
        # It represents the current position in the 48-step navigation cycle.
        self.phase: int = 0

    def load_phase(self, phase: int) -> None:
        """
        Loads the minimal required state for a session's execution context.
        This is called by the ExtensionManager when a session is initialized.

        Args:
            phase: The starting phase (0-47) for the session.

        Raises:
            ValueError: If phase is outside the valid range [0, 47].
        """
        if not (0 <= phase < 48):
            raise ValueError(f"Phase must be between 0 and 47, got {phase}")
        self.phase = phase

    def _get_gene_constant(self) -> Dict[str, torch.Tensor]:
        """
        Returns the invariant Gene structure per CORE-SPEC-02.
        This is the fixed, immutable tensor substrate of the system.

        The Gene consists of two 4×2×3×2 int8 tensors (id_0 and id_1),
        each containing values in {-1, 1}. The pattern represents the
        fundamental coordination topology through which all navigation occurs.

        Returns:
            Dict with 'id_0' and 'id_1' tensors, each of shape (4, 2, 3, 2).
        """
        # This is the full, authoritative definition of the Gene per CORE-SPEC-02.
        gene_pattern = [
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        ]

        # Create the base tensor with the specified pattern
        base_tensor = torch.tensor(gene_pattern, dtype=torch.int8)

        # Verify the shape is correct
        assert base_tensor.shape == (
            4,
            2,
            3,
            2,
        ), f"Gene tensor has incorrect shape: {base_tensor.shape}"

        # Return both tensors as independent clones
        return {"id_0": base_tensor.clone(), "id_1": base_tensor.clone()}

    def _structural_resonance(self, input_byte: int) -> bool:
        """
        Pure computational check for structural alignment per CORE-SPEC-08.
        Determines if an input byte's bit pattern aligns with the Gene
        topology at the current phase.

        The resonance check maps the input byte to a specific position in
        the Gene tensor based on the current phase, then tests if the bit
        patterns align with the tensor values at that position.

        Args:
            input_byte: The input byte (0-255) to test.

        Returns:
            True if resonance occurs, False otherwise.
        """
        # Validate input
        if not (0 <= input_byte <= 255):
            return False

        # Phase-to-tensor mapping: determine which tensor and position to check
        tensor_id = self.phase % 2  # Alternates between id_0 and id_1
        position_in_tensor = (self.phase // 2) % 24  # 24 positions per tensor

        # Map position to tensor coordinates
        outer_idx = position_in_tensor // 6  # 4 outer positions
        inner_idx = (position_in_tensor // 3) % 2  # 2 inner positions
        spatial_idx = position_in_tensor % 3  # 3 spatial dimensions (X, Y, Z)

        # Access the specific slice of the immutable Gene
        tensor_key = f"id_{tensor_id}"
        current_slice = self.gene[tensor_key][outer_idx][inner_idx][spatial_idx]

        # Bit pattern alignment test per CORE-SPEC-08
        # High nibble (bits 7-4) maps to first element
        # Low nibble (bits 3-0) maps to second element
        high_nibble = (input_byte >> 4) & 0x0F
        low_nibble = input_byte & 0x0F

        # Alignment: nibble >= 8 maps to 1, < 8 maps to -1
        high_alignment = 1 if high_nibble >= 8 else -1
        low_alignment = 1 if low_nibble >= 8 else -1

        # Check if both alignments match the tensor values
        return (
            high_alignment == current_slice[0].item() and low_alignment == current_slice[1].item()
        )

    def _select_operator_codes(self) -> Optional[Tuple[int, int]]:
        """
        Pure computational selection of operator codes based on phase.
        It does NOT call the operators; it only determines which ones should fire.

        The selection follows the precise phase boundaries defined in CORE-SPEC-08:
        - CS boundaries (0,12,24,36): Stable operator (Identity/Inverse)
        - UNA/ONA transitions (3,9,15,21,27,33,39,45): Unstable operator (Forward/Backward)
        - Nesting boundaries (6,18,30,42): Neutral operator (Backward)

        Returns:
            A tuple of (op_code_id0, op_code_id1) if an operator resonates,
            otherwise None.
        """
        op_code_0 = None
        op_code_1 = None

        if self.phase in _CS_BOUNDARIES:
            # Stable Operator (gyro_curation) resonance
            # id_0 gets Identity, id_1 gets Inverse
            op_code_0 = (_OP_CODES["IDENTITY"] << 1) | 0  # Bits: 000|0
            op_code_1 = (_OP_CODES["INVERSE"] << 1) | 1  # Bits: 001|1

        elif self.phase in _UNA_ONA_BOUNDARIES:
            # Unstable Operator (gyro_interaction) resonance
            # Alternates between Forward and Backward based on position in cycle
            if (self.phase % 24) < 12:
                base_op = _OP_CODES["FORWARD"]
            else:
                base_op = _OP_CODES["BACKWARD"]

            op_code_0 = (base_op << 1) | 0  # Tensor id 0
            op_code_1 = (base_op << 1) | 1  # Tensor id 1

        elif self.phase in _NESTING_BOUNDARIES:
            # Neutral Operator (gyro_cooperation) resonance
            # Both tensors get Backward gyration
            op_code_0 = (_OP_CODES["BACKWARD"] << 1) | 0  # Bits: 011|0
            op_code_1 = (_OP_CODES["BACKWARD"] << 1) | 1  # Bits: 011|1

        # Return the codes if any operator resonated
        if op_code_0 is not None:
            return (op_code_0, op_code_1)

        return None

    def execute_cycle(self, input_byte: int) -> Optional[Tuple[int, int]]:
        """
        Executes one full, atomic navigation cycle step. This is the primary
        method called by the ExtensionManager.

        The cycle implements the complete CS→UNA→ONA→BU navigation path:
        1. Phase advance (CS→UNA): Moves to the next position in the 48-step cycle
        2. Structural resonance (UNA→ONA): Tests input alignment with Gene
        3. Operator selection (ONA→BU): Determines which operators should fire

        Args:
            input_byte: The byte being processed (0-255).

        Returns:
            A tuple of (op_code_id0, op_code_id1) if a navigation event was
            generated, otherwise None.
        """
        # 1. Advance Phase (CS→UNA transition)
        # The engine is the only component that can modify the phase
        self.phase = (self.phase + 1) % 48

        # 2. Structural Resonance Check (UNA→ONA transition)
        if not self._structural_resonance(input_byte):
            return None

        # 3. Operator Selection (ONA→BU transition)
        operator_codes = self._select_operator_codes()

        # Return the generated operator codes to the caller (ExtensionManager)
        # These will be recorded in the navigation log if not None
        return operator_codes


# ============================================================================
# CORE TRANSFORMATION PRIMITIVE
# ============================================================================


def gyration_op(tensor: torch.Tensor, code: int, clone: bool = True) -> torch.Tensor:
    """
    Applies a gyration transformation to a 4×2×3×2 int8 tensor per CORE-SPEC-02.

    This primitive is pure and used by other parts of the system (like the
    storage extension to decode the genome) but is defined here as it is
    fundamental to the engine's mechanics.

    The four gyration operators are:
    - Identity (0): No transformation
    - Inverse (1): Global sign flip (multiply all elements by -1)
    - Forward (2): Flip signs of rows 0 and 2
    - Backward (3): Flip signs of rows 1 and 3

    Args:
        tensor: The 4×2×3×2 gene tensor to transform.
        code: The 3-bit gyration operator code (0-3).
        clone: If True, operates on a copy. If False, mutates in place.

    Returns:
        The transformed tensor.

    Raises:
        ValueError: If tensor shape is incorrect or code is invalid.
    """
    # Validate tensor shape
    if tensor.shape != (4, 2, 3, 2):
        raise ValueError(f"Invalid tensor shape: {tensor.shape}. Must be (4, 2, 3, 2).")

    # Validate operator code
    if code not in _OP_CODES.values():
        raise ValueError(
            f"Invalid gyration code: {code}. Must be one of {list(_OP_CODES.values())}."
        )

    # Clone if requested to avoid mutation
    result = tensor.clone() if clone else tensor

    # Apply the transformation based on the operator code
    if code == _OP_CODES["IDENTITY"]:
        # Identity: no transformation
        pass

    elif code == _OP_CODES["INVERSE"]:
        # Inverse: global sign flip
        result *= -1

    elif code == _OP_CODES["FORWARD"]:
        # Forward gyration: flip rows 0 and 2
        result[0] *= -1
        result[2] *= -1

    elif code == _OP_CODES["BACKWARD"]:
        # Backward gyration: flip rows 1 and 3
        result[1] *= -1
        result[3] *= -1

    return result


# ============================================================================
# PUBLIC API of this Module
# ============================================================================
# Only the engine class and the primitive operation are exposed.
__all__ = [
    "GyroEngine",
    "gyration_op",
]
