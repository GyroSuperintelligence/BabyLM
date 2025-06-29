"""
g1_inference.py - Governance Engine

Pure CGM processing for alignment and canonical gating.
Operates entirely in memory with no storage logic.
"""

import torch
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class AcceptedOpPair:
    """Event emitted for each accepted operation pair."""

    phase: int
    op_pair: Tuple[int, int]  # Single op_pair: (op_code, tensor_id)
    cycle_position: int


@dataclass
class CycleComplete:
    """Event emitted when a 48-step cycle completes."""

    cycle_number: int
    op_pairs: List[Tuple[int, int]]  # List of 48 op_pairs
    resonance_flags: List[bool]  # List of 48 resonance flags for pruning analysis


class GovernanceEngine:
    """
    Phase-driven acceptance engine for operation pairs.
    Maintains a circular buffer of the last 48 accepted op-pairs.
    """

    def __init__(self):
        self.phase = 0
        self.cycle_count = 0
        self.buffer = deque(maxlen=48)
        self.current_cycle_ops = []
        self.current_cycle_resonance = []  # Track resonance flags for current cycle

    def process_op_pair(self, op_pair: Tuple[int, int], resonance_flag: bool = False) -> List:
        """
        Process a single operation pair.

        Args:
            op_pair: Tuple of (op_code, tensor_id)
            resonance_flag: Whether this op-pair was resonant

        Returns:
            List of emitted events
        """
        events = []

        # Accept the op-pair unconditionally
        self.buffer.append(op_pair)
        self.current_cycle_ops.append(op_pair)
        self.current_cycle_resonance.append(resonance_flag)

        # Emit accepted event
        events.append(
            AcceptedOpPair(
                phase=self.phase, op_pair=op_pair, cycle_position=len(self.current_cycle_ops) - 1
            )
        )

        # Advance phase
        self.phase = (self.phase + 1) % 48

        # Check for cycle completion
        if self.phase == 0:
            events.append(
                CycleComplete(
                    cycle_number=self.cycle_count,
                    op_pairs=self.current_cycle_ops.copy(),
                    resonance_flags=self.current_cycle_resonance.copy(),
                )
            )
            self.cycle_count += 1
            self.current_cycle_ops = []
            self.current_cycle_resonance = []

        return events

    def get_state(self) -> Dict:
        """Return current engine state."""
        return {
            "phase": self.phase,
            "cycle_count": self.cycle_count,
            "buffer_size": len(self.buffer),
            "current_cycle_size": len(self.current_cycle_ops),
        }
