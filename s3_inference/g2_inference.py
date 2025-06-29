"""
g2_inference.py - Information Engine

Resonance tagging engine that classifies op-pairs based on the epigenome mask.
No bit-flipping or mask mutation - purely classificatory.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class ResonanceEvent:
    """Event emitted for each processed op-pair with resonance flag."""

    phase: int
    op_pair: Tuple[int, int]  # (op_code, tensor_id)
    resonance_flag: bool
    bit_index: int


class InformationEngine:
    """
    Resonance classification engine using the immutable epigenome mask.
    Tags each op-pair as resonant or non-resonant based on phase alignment.
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

    def _load_epigenome(self, path: str) -> np.ndarray:
        """
        Load the epigenome projection table from disk.

        Args:
            path: Path to the epigenome file

        Returns:
            48x256 numpy array of resonance values
        """
        with open(path, "rb") as f:
            # Skip 32-byte SHA-256 header
            f.read(32)
            # Read 48x256 table
            data = f.read(48 * 256)
            return np.frombuffer(data, dtype=np.uint8).reshape(48, 256)

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
        # Extract operation components
        op_code, tensor_id = op_pair

        # Compute operation index and bit index
        op_index = op_code * 2 + tensor_id
        bit_index = phase * 8 + op_index

        # Look up resonance from the mask
        resonance_bit = self.resonance_mask[phase, byte_val]
        resonance_flag = resonance_bit == op_code

        return ResonanceEvent(
            phase=phase, op_pair=op_pair, resonance_flag=resonance_flag, bit_index=bit_index
        )

    def get_state(self) -> Dict:
        """Return current engine state."""
        return {
            "mask_shape": self.resonance_mask.shape,
            "mask_loaded": self.resonance_mask is not None,
        }
