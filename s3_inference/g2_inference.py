"""
g2_inference.py - Information Engine

Resonance tagging engine that classifies op-pairs based on the epigenome mask.
No bit-flipping or mask mutation - purely classificatory.

Device logic: All tensors are created on the selected device (GPU if available, else CPU).
"""

import torch

# Select device for all tensors and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import os
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from s4_intelligence.g2_intelligence_eg import VOID_OP_PAIR, is_void


@dataclass
class ResonanceEvent:
    """
    Event emitted for each processed op-pair with resonance flag.

    Attributes:
        phase: Current phase (0-47) when this op-pair was processed
        op_pair: The operation pair as (op_code, tensor_id)
        resonance_flag: Whether this op-pair resonates with the epigenome at this phase
        bit_index: Calculated index in the 384-bit epigenome space (phase*8 + op_index)
    """

    phase: int
    op_pair: Tuple[int, int]  # (op_code, tensor_id)
    resonance_flag: bool
    bit_index: int


class InformationEngine:
    """
    Resonance classification engine using the immutable epigenome mask.
    Tags each op-pair as resonant or non-resonant based on phase alignment.

    The epigenome is a 48×256 table mapping each (phase, byte) combination
    to an expected operation code. Resonance occurs when an op-pair's code
    matches the expected code for the current phase and input byte.
    """

    def __init__(
        self, epigenome_path: str = "s2_information/agency/g2_information/g2_information.dat"
    ):
        """
        Initialize with the epigenome projection table.

        Args:
            epigenome_path: Path to the epigenome projection file
        """
        self.resonance_mask = self._load_epigenome(epigenome_path)
        self.resonance_counts = {"total": 0, "resonant": 0}

    def _load_epigenome(self, path: str) -> np.ndarray:
        """
        Load the epigenome projection table from disk.

        Args:
            path: Path to the epigenome file

        Returns:
            48x256 numpy array of resonance values
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Epigenome file not found: {path}")

        try:
            with open(path, "rb") as f:
                # Skip 32-byte SHA-256 header
                header = f.read(32)
                if len(header) != 32:
                    raise ValueError(f"Invalid epigenome header size: {len(header)}")

                # Read 48x256 table
                data = f.read(48 * 256)
                if len(data) != 48 * 256:
                    raise ValueError(f"Invalid epigenome data size: {len(data)}")

                return np.frombuffer(data, dtype=np.uint8).reshape(48, 256)
        except Exception as e:
            # Fall back to a default empty mask if file loading fails
            print(f"Error loading epigenome, using empty mask: {e}")
            return np.zeros((48, 256), dtype=np.uint8)

    def process_accepted_op_pair(
        self, phase: int, op_pair: Tuple[int, int], byte_val: int
    ) -> ResonanceEvent:
        """
        Process an accepted op-pair and determine its resonance.

        Args:
            phase: Current phase (0-47)
            op_pair: The accepted operation pair (op_code, tensor_id)
            byte_val: The original input byte that generated this op_pair

        Returns:
            ResonanceEvent with classification result
        """
        # Validate inputs
        if not 0 <= phase < 48:
            raise ValueError(f"Phase must be 0-47, got {phase}")

        if not isinstance(op_pair, tuple) or len(op_pair) != 2:
            raise ValueError(f"Invalid op_pair format: {op_pair}")

        if not 0 <= byte_val <= 255:
            raise ValueError(f"Byte value must be 0-255, got {byte_val}")

        # Extract operation components
        op_code, tensor_id = op_pair

        # Range checks
        if is_void(op_pair):
            # This is padding—emit a no-resonance event but skip mask lookup
            return ResonanceEvent(phase, op_pair, False, -1)
        if not 0 <= op_code <= 3:
            raise ValueError(f"Op code must be 0-3, got {op_code}")
        if not 0 <= tensor_id <= 1:
            raise ValueError(f"Tensor ID must be 0-1, got {tensor_id}")

        # Get the expected operator from the epigenome table for this phase/byte
        expected_op_code = self.resonance_mask[phase, byte_val]

        # Resonance occurs if the op-pair's code matches the expected code
        resonance_flag = op_code == expected_op_code

        # Update statistics
        self.resonance_counts["total"] += 1
        if resonance_flag:
            self.resonance_counts["resonant"] += 1

        # Compute bit index (used for Hebbian learning markers)
        bit_index = phase * 8 + op_code * 2 + tensor_id

        return ResonanceEvent(
            phase=phase, op_pair=op_pair, resonance_flag=resonance_flag, bit_index=bit_index
        )

    def get_state(self) -> Dict[str, Any]:
        """Return current engine state."""
        return {
            "mask_shape": self.resonance_mask.shape,
            "mask_loaded": self.resonance_mask is not None,
            "resonance_ratio": self.resonance_counts["resonant"]
            / max(1, self.resonance_counts["total"]),
            "resonance_counts": self.resonance_counts,
        }

    def reset(self) -> None:
        """Reset the engine statistics."""
        self.resonance_counts = {"total": 0, "resonant": 0}
