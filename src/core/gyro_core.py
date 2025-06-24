"""
gyro_core.py - Tier 3: The Pure Navigation Engine

This module contains the GyroEngine, the pure computational heart of the GyroSI
system. It is responsible for executing the atomic steps of the navigation cycle.

Key characteristics:
- It is completely stateless regarding session or knowledge context.
- It has no awareness of files, I/O, or external state.
- Its operations are traceable and computationally pure.
- It holds only the invariant Gene and the transient session phase.

References:
- CORE-SPEC-01: Architecture and Principles
- CORE-SPEC-02: Foundations (Core Mechanics)
- CORE-SPEC-07: Baseline Implementation Specifications
- CORE-SPEC-08: Operational Walkthrough
"""

import os
import torch
import numpy as np
import hashlib
from typing import Tuple, Dict, Optional
from .gyro_errors import GyroIntegrityError


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
    Provably mechanical navigation engine.
    Every input byte ALWAYS produces operator codes via harmonics mask lookup.
    No heuristics, no branching, no None returns.
    """

    def __init__(self, harmonics_path: Optional[str] = None):
        """Initialize with harmonics matrix and Gene validation"""
        self.gene: Dict[str, torch.Tensor] = self._get_gene_constant()
        self.phase: int = 0
        if harmonics_path is None:
            harmonics_path = os.path.join(os.path.dirname(__file__), "gyro_harmonics.dat")
        self._load_and_validate_harmonics(harmonics_path)

    def _load_and_validate_harmonics(self, harmonics_path: str) -> None:
        """Load harmonics matrix, validate Gene, and parse mask/operator vector"""
        try:
            with open(harmonics_path, "rb") as f:
                payload = f.read()
        except FileNotFoundError:
            raise GyroIntegrityError(
                f"CRITICAL: gyro_harmonics.dat not found at {harmonics_path}. "
                f"Run: python gyro_tools/build_operator_matrix.py <output_file>"
            )
        if len(payload) < 1616:
            raise GyroIntegrityError("Harmonics matrix is corrupted or incomplete.")
        digest = payload[:32]
        mask = np.frombuffer(payload[32 : 32 + 48 * 256], dtype=np.uint8).reshape(48, 256)
        opvec = np.frombuffer(payload[32 + 48 * 256 : 32 + 48 * 256 + 48], dtype=np.uint8)
        current_digest = self._compute_gene_checksum()
        if digest != current_digest:
            raise GyroIntegrityError(
                "Harmonics matrix Gene checksum mismatch. "
                "Matrix was built for different Gene version. "
                "Rebuild with: python gyro_tools/build_operator_matrix.py <output_file>"
            )
        self._harmonics_mask = mask
        self._operator_vector = opvec

    def execute_cycle(self, input_byte: int) -> Tuple[int, int]:
        """
        Execute one mechanical navigation cycle using harmonics mask and operator vector.
        Returns operator codes if resonance, else raises.
        """
        self.phase = (self.phase + 1) % 48
        clamped_byte = max(0, min(255, input_byte))
        if self._harmonics_mask[self.phase, clamped_byte] == 0:
            raise GyroIntegrityError(f"No resonance at phase {self.phase} for byte {input_byte}")
        op_pair = self._operator_vector[self.phase]
        op_code_0 = op_pair & 0x0F  # lower 4 bits
        op_code_1 = op_pair >> 4  # upper 4 bits
        return (op_code_0, op_code_1)

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

    def _compute_gene_checksum(self) -> bytes:
        """Compute SHA-256 checksum of current Gene"""
        hasher = hashlib.sha256()
        hasher.update(self.gene["id_0"].numpy().tobytes())
        hasher.update(self.gene["id_1"].numpy().tobytes())
        return hasher.digest()

    def load_phase(self, phase: int) -> None:
        """Set the engine's phase to the provided value."""
        self.phase = phase


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
