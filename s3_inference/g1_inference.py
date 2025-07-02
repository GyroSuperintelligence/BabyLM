"""
g1_inference.py - Governance Engine

Pure CGM processing for alignment and canonical gating.
Operates entirely in memory with no storage logic.

Device logic: All tensors are created on the selected device (GPU if available, else CPU).
"""

import torch

# Select device for all tensors and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque

# Import gene mechanics from s1_governance
from s1_governance import get_gene_tensors, gyration_op

# Import VOID_OP_PAIR and is_void from s1_governance
from s1_governance import VOID_OP_PAIR, is_void


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
    Maintains a circular buffer of the last 48 accepted op-pairs and tracks the phase.

    Key responsibilities:
    1. Accept operation pairs and advance the phase
    2. Maintain a fixed-size buffer of the most recent 48 op-pairs
    3. Emit events when cycles complete (every 48 steps)
    4. Track resonance flags for pattern analysis
    """

    def __init__(self):
        """Initialize the engine with empty state."""
        self.phase = 0  # Current phase (0-47)
        self.cycle_index = 0  # Number of completed cycles
        self.buffer = deque(maxlen=48)  # Circular buffer of recent op-pairs
        self.current_cycle_ops = []  # Accumulating buffer for current cycle
        self.current_cycle_resonance = []  # Resonance flags for current cycle

    def process_op_pair(
        self, op_pair: Tuple[int, int], resonance_flag: bool, padding: bool = False
    ) -> List[Any]:
        """
        Process a single operation pair through the governance cycle.

        Args:
            op_pair: Tuple of (op_code, tensor_id)
            resonance_flag: Whether this op-pair was resonant
            padding: Whether this op-pair is a padding op-pair

        Returns:
            List of emitted events (AcceptedOpPair and possibly CycleComplete)
        """
        # Validate inputs
        if not isinstance(op_pair, tuple) or len(op_pair) != 2:
            raise ValueError(
                f"Invalid op_pair format: {op_pair}. Expected (op_code, tensor_id) tuple."
            )

        op_code, tensor_id = op_pair
        if not padding:
            if not (0 <= op_code <= 3 and 0 <= tensor_id <= 1):
                if not is_void(op_pair):
                    raise ValueError(f"Invalid op_pair: {op_pair}")

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
            # Make deep copies to ensure immutability
            cycle_ops = self.current_cycle_ops.copy()
            cycle_res = self.current_cycle_resonance.copy()

            # Emit cycle completion event
            events.append(
                CycleComplete(
                    cycle_number=self.cycle_index,
                    op_pairs=cycle_ops,
                    resonance_flags=cycle_res,
                )
            )

            # Increment cycle index and reset accumulators
            self.cycle_index += 1
            self.current_cycle_ops = []
            self.current_cycle_resonance = []

        return events

    def get_state(self) -> Dict[str, Any]:
        """
        Return the current engine state as a format.
        Useful for monitoring and debugging.
        """
        return {
            "phase": self.phase,
            "cycle_index": self.cycle_index,
            "buffer_size": len(self.buffer),
            "current_cycle_size": len(self.current_cycle_ops),
            "current_cycle_buffer": list(self.buffer),
        }

    def reset(self) -> None:
        """
        Reset the engine to its initial state.
        Useful for testing or when starting a new session.
        """
        self.phase = 0
        self.cycle_index = 0
        self.buffer.clear()
        self.current_cycle_ops = []
        self.current_cycle_resonance = []
