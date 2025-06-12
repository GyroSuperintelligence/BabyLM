"""
GyroTensor: Core tensor object for GyroSI.

This module implements the fundamental tensor structure that undergoes
the canonical CS→UNA→ONA→BU_In→BU_En→ONA→UNA→CS cycle. The GyroTensor
orchestrates its own lifecycle by calling into the appropriate stage
modules while maintaining strict adherence to CGM constants and
structural accountability.

All tensor operations preserve memory of origin and maintain continuous
coherence between emergence and recollection.
"""

import math
import time
import hashlib
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

# Absolute imports
from gyro_si.gyro_constants import ALPHA, BETA, GAMMA, M_P, HALF_HORIZON
from gyro_si.gyro_errors   import StructuralViolation, QuantizationDefect, HelicalCoherenceError
from gyro_si.gyro_comm     import send_message, MessageTypes, create_tensor_context


logger = logging.getLogger(__name__)


class TensorStage(Enum):
    """Canonical tensor stages in the recursive cycle."""
    CS = "CS"           # Closure/Identity stage
    UNA = "UNA"         # Unary/Normalization stage
    ONA = "ONA"         # Binary/Correlation stage
    BU_IN = "BU_In"     # Integrative quantization
    BU_EN = "BU_En"     # Generative quantization


class OperationType(Enum):
    """Types of operations between stages."""
    GENERATION = "generation"     # CS→UNA, BU_In→BU_En, BU_En→ONA, UNA→CS
    INTEGRATION = "integration"   # UNA→ONA, ONA→BU_In


@dataclass
class StageTransition:
    """Defines a valid stage transition."""
    from_stage: TensorStage
    to_stage: TensorStage
    operation_type: OperationType


# Canonical stage transition table - unified for forward and return paths
# Key: (current_stage, is_forward_path) -> StageTransition
STAGE_TRANSITIONS = {
    # Forward path transitions
    (TensorStage.CS, True): StageTransition(TensorStage.CS, TensorStage.UNA, OperationType.GENERATION),
    (TensorStage.UNA, True): StageTransition(TensorStage.UNA, TensorStage.ONA, OperationType.INTEGRATION),
    (TensorStage.ONA, True): StageTransition(TensorStage.ONA, TensorStage.BU_IN, OperationType.INTEGRATION),
    (TensorStage.BU_IN, True): StageTransition(TensorStage.BU_IN, TensorStage.BU_EN, OperationType.GENERATION),
    (TensorStage.BU_EN, True): StageTransition(TensorStage.BU_EN, TensorStage.ONA, OperationType.GENERATION),

    # Return path transitions
    (TensorStage.ONA, False): StageTransition(TensorStage.ONA, TensorStage.UNA, OperationType.INTEGRATION),
    (TensorStage.UNA, False): StageTransition(TensorStage.UNA, TensorStage.CS, OperationType.GENERATION),
}


class GyroTensor:
    """
    Core tensor object implementing the canonical CS→UNA→ONA→BU→CS cycle.

    The GyroTensor maintains its own state and orchestrates transitions
    through the five canonical stages. All operations are structurally
    accountable and preserve helical trajectory information.

    Note: All tensor transitions must occur under an active asyncio event loop,
    as status updates and algedonic signals are sent asynchronously.
    """

    def __init__(self, tensor_id: int, parent_id: Optional[int] = None):
        """
        Initialize a new GyroTensor in CS stage.

        Args:
            tensor_id: Unique identifier for this tensor
            parent_id: ID of parent tensor if this is spawned
        """
        # Identity and lineage
        self.tensor_id = tensor_id
        self.parent_id = parent_id
        self.creation_time = time.time()

        # Stage and cycle tracking
        self.stage = TensorStage.CS
        self.cycle_index = 0
        self.transition_count = 0

        # Helical trajectory tracking
        self.cumulative_phase = 0.0      # [0, 4π) - Total helical progress
        self.chirality_phase = 0.0       # [0, 2π) - Forward/return position
        self.helical_position = 0.0      # [0, 1) - Normalized fraction
        self.spinor_cycle_count = 0      # Number of completed 720° revolutions

        # Amplitude and quantization
        self.amplitude = 0.0             # Bounded by ±mₚ
        self.last_epsilon = 0.0          # Previous quantization error

        # CSR storage (allocated post-CS)
        self.indptr: Optional[List[int]] = None
        self.indices: Optional[List[int]] = None
        self.data: Optional[List[int]] = None  # Values in {0, 1, -1} encoded as {0, 1, 2}

        # State tracking
        self.is_forward_path = True      # True for 0→2π, False for 2π→4π
        self.spawn_ready = False         # True when |amplitude| ≥ mₚ and cumulative_phase ≥ 4π

        # Audit and logging
        self.state_checksum = self._compute_checksum()
        self.trace_buffer: List[Dict[str, Any]] = []

        self._log("tensor_created",
                 tensor_id=tensor_id,
                 parent_id=parent_id,
                 stage=self.stage.value)

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of current tensor state."""
        state_data = (
            self.tensor_id,
            self.stage.value,
            tuple(self.indptr) if self.indptr else (),
            tuple(self.indices) if self.indices else (),
            tuple(self.data) if self.data else (),
            self.amplitude,
            self.cumulative_phase,
            self.chirality_phase
        )
        state_str = str(state_data).encode('utf-8')
        return hashlib.sha256(state_str).hexdigest()

    def _log(self, event_type: str, **kwargs) -> None:
        """Log an event to the trace buffer."""
        entry = {
            "event_type": event_type,
            "tensor_id": self.tensor_id,
            "stage": self.stage.value,
            "cycle_index": self.cycle_index,
            "timestamp": time.time(),
            "checksum": self.state_checksum,
            **kwargs
        }
        self.trace_buffer.append(entry)
        logger.debug(f"Tensor {self.tensor_id}: {event_type}", extra=kwargs)

    def _validate_stage_transition(self, target_stage: TensorStage) -> bool:
        """
        Validate that a stage transition is permitted.

        Args:
            target_stage: The stage to transition to

        Returns:
            True if transition is valid

        Raises:
            StructuralViolation: If transition is not permitted
        """
        transition_key = (self.stage, self.is_forward_path)

        if transition_key not in STAGE_TRANSITIONS:
            raise StructuralViolation(
                f"No valid transition from stage {self.stage.value} "
                f"(forward_path={self.is_forward_path})"
            )

        expected_transition = STAGE_TRANSITIONS[transition_key]
        if target_stage != expected_transition.to_stage:
            raise StructuralViolation(
                f"Invalid transition: {self.stage.value} → {target_stage.value}, "
                f"expected {expected_transition.to_stage.value} "
                f"(forward_path={self.is_forward_path})"
            )

        return True

    def _update_helical_trajectory(self, phase_delta: float) -> None:
        """
        Update helical trajectory tracking.

        Args:
            phase_delta: Change in phase for this transition (must be > 0)
        """
        if phase_delta <= 0:
            raise ValueError(f"phase_delta must be positive, got {phase_delta}")

        # Update cumulative phase
        old_cumulative = self.cumulative_phase
        old_chirality = old_cumulative % (2 * math.pi)

        self.cumulative_phase = (self.cumulative_phase + phase_delta) % (4 * math.pi)

        # Update chirality phase
        self.chirality_phase = self.cumulative_phase % (2 * math.pi)

        # Update helical position (normalized)
        self.helical_position = self.cumulative_phase / (4 * math.pi)

        # Check for spinor cycle completion (720° = 4π)
        if old_cumulative < 4 * math.pi <= self.cumulative_phase:
            self.spinor_cycle_count += 1
            self._log("spinor_cycle_completed",
                     cycle_count=self.spinor_cycle_count,
                     cumulative_phase=self.cumulative_phase)

        # Check for chirality flip (360° = 2π)
        # Simplified logic: if new chirality is less than old, we've wrapped past 2π
        if self.chirality_phase < old_chirality:
            self.is_forward_path = not self.is_forward_path
            self._log("chirality_flip",
                     is_forward=self.is_forward_path,
                     chirality_phase=self.chirality_phase)

        # Check spawn readiness
        if abs(self.amplitude) >= M_P and self.cumulative_phase >= 4 * math.pi:
            if not self.spawn_ready:
                self.spawn_ready = True
                self._log("spawn_ready",
                         amplitude=self.amplitude,
                         cumulative_phase=self.cumulative_phase)

    def _allocate_csr_storage(self, stage: TensorStage) -> None:
        """
        Allocate CSR storage arrays for the given stage.

        Args:
            stage: Target stage requiring CSR allocation
        """
        if stage == TensorStage.CS:
            # CS stage has no CSR storage
            self.indptr = None
            self.indices = None
            self.data = None
            return

        # Stage-specific CSR dimensions
        if stage == TensorStage.UNA:
            # 3×2 tensor, 6 non-zeros
            rows, cols, nnz = 3, 2, 6
        elif stage == TensorStage.ONA:
            # 2×3×2 tensor, 24 non-zeros
            rows, cols, nnz = 2, 6, 24  # Flattened to 2×6
        elif stage in (TensorStage.BU_IN, TensorStage.BU_EN):
            # 2×2×3×2 tensor, 48 non-zeros
            rows, cols, nnz = 4, 12, 48  # Flattened to 4×12
        else:
            raise StructuralViolation(f"Unknown stage for CSR allocation: {stage}")

        # Allocate CSR arrays
        self.indptr = [0] * (rows + 1)
        self.indices = [0] * nnz
        self.data = [1] * nnz  # Initialize to +1 (encoded as 1)

        # Set up indptr for uniform distribution
        nnz_per_row = nnz // rows
        for i in range(rows + 1):
            self.indptr[i] = i * nnz_per_row

        # Set up indices for uniform column distribution
        cols_per_row = cols
        for row in range(rows):
            start_idx = self.indptr[row]
            for j in range(nnz_per_row):
                self.indices[start_idx + j] = j % cols_per_row

        self._log("csr_allocated",
                 stage=stage.value,
                 rows=rows,
                 cols=cols,
                 nnz=nnz)

    def transition_to_stage(self, target_stage: TensorStage, phi: float) -> None:
        """
        Transition tensor to the target stage.

        Args:
            target_stage: Stage to transition to
            phi: Phase input for quantization - REQUIRED for all transitions
                 to ensure deterministic helical progression

        Raises:
            StructuralViolation: If transition is invalid
            QuantizationDefect: If quantization produces invalid results
            ValueError: If phi is not provided
        """
        # Validate transition
        self._validate_stage_transition(target_stage)

        old_stage = self.stage
        old_checksum = self.state_checksum

        # Allocate CSR storage if needed
        if target_stage != TensorStage.CS and self.indptr is None:
            self._allocate_csr_storage(target_stage)

        # Perform quantization if transitioning to/from BU stages
        if target_stage in (TensorStage.BU_IN, TensorStage.BU_EN) or old_stage in (TensorStage.BU_IN, TensorStage.BU_EN):
            epsilon = self._quantize_phase(phi)
            self._update_amplitude(epsilon)

        # Update stage
        self.stage = target_stage
        self.cycle_index += 1
        self.transition_count += 1

        # Update helical trajectory with explicit phase delta
        # Use absolute value to ensure positive phase progression
        phase_delta = abs(phi)
        self._update_helical_trajectory(phase_delta)

        # Update checksum
        self.state_checksum = self._compute_checksum()

        # Log transition
        self._log("stage_transition",
                 from_stage=old_stage.value,
                 to_stage=target_stage.value,
                 phi=phi,
                 phase_delta=phase_delta,
                 old_checksum=old_checksum,
                 new_checksum=self.state_checksum)

        # Send status update if significant transition
        if target_stage in (TensorStage.BU_IN, TensorStage.BU_EN, TensorStage.CS):
            try:
                # Get the running event loop safely
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._send_status_update())
                else:
                    logger.warning(
                        f"No running event loop for tensor {self.tensor_id} status update"
                    )
            except RuntimeError:
                logger.warning(
                    f"No event loop found for tensor {self.tensor_id} status update"
                )

    def _quantize_phase(self, phi: float) -> float:
        """
        Quantize continuous phase to discrete values.

        Args:
            phi: Continuous phase input

        Returns:
            Quantization error ε = φ - φ_q

        Raises:
            QuantizationDefect: If quantization error exceeds bounds
        """
        # Canonical quantization rule from G1 specification
        if phi < -M_P / 2:
            phi_q = -M_P
        elif phi >= M_P / 2:
            phi_q = M_P
        else:
            phi_q = 0.0

        # Calculate quantization error
        epsilon = phi - phi_q

        # Validate error bounds
        if abs(epsilon) > M_P:
            raise QuantizationDefect(
                f"Quantization error |ε| = {abs(epsilon)} exceeds mₚ = {M_P}"
            )

        # Store for next iteration
        self.last_epsilon = epsilon

        self._log("quantization_event",
                 phi=phi,
                 phi_q=phi_q,
                 epsilon=epsilon)

        return epsilon

    def _update_amplitude(self, epsilon: float) -> None:
        """
        Update tensor amplitude based on quantization error.

        Args:
            epsilon: Quantization error from _quantize_phase
        """
        old_amplitude = self.amplitude

        # Canonical amplitude update: clip to ±mₚ bounds
        self.amplitude = max(-M_P, min(M_P, self.amplitude + epsilon))

        self._log("amplitude_update",
                 old_amplitude=old_amplitude,
                 epsilon=epsilon,
                 new_amplitude=self.amplitude)

        # Check for algedonic signal generation
        if abs(epsilon) > M_P / 2:
            signal_type = "pain" if epsilon > 0 else "pleasure"
            try:
                # Get the running event loop safely
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._generate_algedonic_signal(signal_type, epsilon))
                else:
                    logger.warning(
                        f"No running event loop for tensor {self.tensor_id} algedonic signal"
                    )
            except RuntimeError:
                logger.warning(
                    f"No event loop found for tensor {self.tensor_id} algedonic signal"
                )

    async def _generate_algedonic_signal(self, signal_type: str, epsilon: float) -> None:
        """
        Generate algedonic signal when quantization error exceeds threshold.

        Args:
            signal_type: "pain" or "pleasure"
            epsilon: Quantization error that triggered the signal
        """
        tensor_context = create_tensor_context(
            cumulative_phase=self.cumulative_phase,
            chirality_phase=self.chirality_phase,
            helical_position=self.helical_position,
            spinor_cycle_count=self.spinor_cycle_count,
            tensor_id=self.tensor_id,
            amplitude=self.amplitude
        )

        message = {
            "type": MessageTypes.ALGEDONIC_SIGNAL,
            "source": "G1",
            "destination": "G2",  # Route through G2 to G4
            "cycle_index": self.cycle_index,
            "tensor_context": tensor_context,
            "payload": {
                "signal_type": signal_type,
                "epsilon": epsilon,
                "tensor_id": self.tensor_id,
                "stage": self.stage.value,
                "threshold_exceeded": abs(epsilon) > M_P / 2
            },
            "timestamp": time.time()
        }

        # Actually send the message
        await send_message(message)

        self._log("algedonic_signal_generated",
                 signal_type=signal_type,
                 epsilon=epsilon,
                 threshold=M_P / 2)

    async def _send_status_update(self) -> None:
        """Send status update to G2 for coordination."""
        tensor_context = create_tensor_context(
            cumulative_phase=self.cumulative_phase,
            chirality_phase=self.chirality_phase,
            helical_position=self.helical_position,
            spinor_cycle_count=self.spinor_cycle_count,
            tensor_id=self.tensor_id,
            amplitude=self.amplitude
        )

        message = {
            "type": MessageTypes.STATUS_UPDATE,
            "source": "G1",
            "destination": "G2",
            "cycle_index": self.cycle_index,
            "tensor_context": tensor_context,
            "payload": {
                "tensor_id": self.tensor_id,
                "stage": self.stage.value,
                "amplitude": self.amplitude,
                "spawn_ready": self.spawn_ready,
                "transition_count": self.transition_count,
                "helical_metrics": {
                    "cumulative_phase": self.cumulative_phase,
                    "chirality_phase": self.chirality_phase,
                    "spinor_cycles": self.spinor_cycle_count,
                    "is_forward_path": self.is_forward_path
                }
            },
            "timestamp": time.time()
        }

        # Actually send the message
        await send_message(message)

        self._log("status_update_sent",
                 stage=self.stage.value,
                 spawn_ready=self.spawn_ready)

    def get_collision_factor(self) -> float:
        """
        Calculate collision factor for this tensor.

        Returns:
            Collision factor as cs_memory / max_discrete
        """
        cs_memory = ALPHA  # π/2
        max_discrete = 3 * M_P
        return cs_memory / max_discrete

    async def execute_full_cycle(self, phi_values: List[float]) -> None:
        """
        Execute a complete CS→UNA→ONA→BU→ONA→UNA→CS cycle.

        This method orchestrates the tensor through its complete lifecycle,
        calling the appropriate stage modules for each transition.

        Args:
            phi_values: List of 7 phase values for each transition
                        (CS→UNA, UNA→ONA, ONA→BU_In, BU_In→BU_En,
                         BU_En→ONA, ONA→UNA, UNA→CS)

        Raises:
            ValueError: If phi_values doesn't contain exactly 7 values
            StructuralViolation: If not starting from CS stage
        """
        if self.stage != TensorStage.CS:
            raise StructuralViolation(
                f"Full cycle must start from CS stage, currently in {self.stage.value}"
            )

        if len(phi_values) != 7:
            raise ValueError(f"Expected 7 phi values for full cycle, got {len(phi_values)}")

        self._log("full_cycle_start", cycle_index=self.cycle_index)

        try:
            # Forward path: CS → UNA → ONA → BU_In → BU_En
            self.transition_to_stage(TensorStage.UNA, phi_values[0])      # Generation
            self.transition_to_stage(TensorStage.ONA, phi_values[1])      # Integration
            self.transition_to_stage(TensorStage.BU_IN, phi_values[2])    # Integration (anomalous)
            self.transition_to_stage(TensorStage.BU_EN, phi_values[3])    # Generation

            # Return path: BU_En → ONA → UNA → CS
            self.transition_to_stage(TensorStage.ONA, phi_values[4])      # Generation (return)
            self.transition_to_stage(TensorStage.UNA, phi_values[5])      # Integration (return)
            self.transition_to_stage(TensorStage.CS, phi_values[6])       # Generation (closure)

            self._log("full_cycle_complete",
                     cycle_index=self.cycle_index,
                     spinor_cycles=self.spinor_cycle_count)

        except Exception as e:
            self._log("full_cycle_error",
                     error=str(e),
                     stage=self.stage.value)
            raise

    def get_tensor_context(self) -> Dict[str, Any]:
        """
        Get current tensor context for message passing.

        Returns:
            Dictionary with helical trajectory and tensor state
        """
        return create_tensor_context(
            cumulative_phase=self.cumulative_phase,
            chirality_phase=self.chirality_phase,
            helical_position=self.helical_position,
            spinor_cycle_count=self.spinor_cycle_count,
            tensor_id=self.tensor_id,
            parent_id=self.parent_id,
            stage=self.stage.value,
            amplitude=self.amplitude,
            spawn_ready=self.spawn_ready
        )

    def get_trace_buffer(self) -> List[Dict[str, Any]]:
        """
        Get copy of trace buffer for audit purposes.

        Returns:
            List of trace entries
        """
        return self.trace_buffer.copy()

    def clear_trace_buffer(self, before_timestamp: Optional[float] = None) -> int:
        """
        Clear trace buffer entries.

        Args:
            before_timestamp: Only clear entries before this timestamp

        Returns:
            Number of entries cleared
        """
        if before_timestamp is None:
            count = len(self.trace_buffer)
            self.trace_buffer.clear()
            return count

        original_count = len(self.trace_buffer)
        self.trace_buffer = [
            entry for entry in self.trace_buffer
            if entry["timestamp"] >= before_timestamp
        ]
        return original_count - len(self.trace_buffer)

    def validate_structural_integrity(self) -> bool:
        """
        Validate tensor structural integrity.

        Returns:
            True if tensor structure is valid

        Raises:
            StructuralViolation: If structure is invalid
        """
        # Validate stage-specific CSR structure
        if self.stage == TensorStage.CS:
            if self.indptr is not None or self.indices is not None or self.data is not None:
                raise StructuralViolation("CS stage must not have CSR storage")
        else:
            if self.indptr is None or self.indices is None or self.data is None:
                raise StructuralViolation(f"Stage {self.stage.value} requires CSR storage")

            # Validate CSR dimensions
            expected_nnz = {
                TensorStage.UNA: 6,
                TensorStage.ONA: 24,
                TensorStage.BU_IN: 48,
                TensorStage.BU_EN: 48
            }

            if self.stage in expected_nnz:
                if len(self.data) != expected_nnz[self.stage]:
                    raise StructuralViolation(
                        f"Stage {self.stage.value} expects {expected_nnz[self.stage]} "
                        f"non-zeros, got {len(self.data)}"
                    )

        # Validate helical coherence
        if not (0 <= self.cumulative_phase < 4 * math.pi):
            raise HelicalCoherenceError(
                f"cumulative_phase {self.cumulative_phase} outside [0, 4π)"
            )

        if not (0 <= self.chirality_phase < 2 * math.pi):
            raise HelicalCoherenceError(
                f"chirality_phase {self.chirality_phase} outside [0, 2π)"
            )

        if not (0 <= self.helical_position < 1):
            raise HelicalCoherenceError(
                f"helical_position {self.helical_position} outside [0, 1)"
            )

        # Validate amplitude bounds
        if abs(self.amplitude) > M_P:
            raise QuantizationDefect(
                f"Amplitude {self.amplitude} exceeds bounds ±{M_P}"
            )

        return True

    def __str__(self) -> str:
        """String representation of tensor."""
        return (
            f"GyroTensor(id={self.tensor_id}, stage={self.stage.value}, "
            f"cycle={self.cycle_index}, amplitude={self.amplitude:.6f}, "
            f"phase={self.cumulative_phase:.3f})"
        )

    def __repr__(self) -> str:
        """Developer representation of tensor."""
        return (
            f"GyroTensor(tensor_id={self.tensor_id}, parent_id={self.parent_id}, "
            f"stage={self.stage.value}, cycle_index={self.cycle_index}, "
            f"amplitude={self.amplitude}, cumulative_phase={self.cumulative_phase}, "
            f"spawn_ready={self.spawn_ready})"
        )


class TensorFamily:
    """
    Manages phase-locked family members and spawning relationships.

    This class tracks lineage relationships and coordinates spawning
    when amplitude and phase conditions are met.
    """

    def __init__(self, root_tensor_id: int):
        """
        Initialize tensor family with root tensor.

        Args:
            root_tensor_id: ID of the root tensor for this family
        """
        self.root_tensor_id = root_tensor_id
        self.members: Dict[int, GyroTensor] = {}
        self.lineage_graph: Dict[int, List[int]] = {}  # parent_id -> [child_ids]
        self.creation_time = time.time()

    def add_tensor(self, tensor: GyroTensor) -> None:
        """Add tensor to family and update lineage."""
        self.members[tensor.tensor_id] = tensor
        if tensor.parent_id is not None:
            self.lineage_graph.setdefault(tensor.parent_id, []).append(tensor.tensor_id)

    def get_spawn_candidates(self) -> List[GyroTensor]:
        """Get tensors ready for spawning."""
        return [t for t in self.members.values() if t.spawn_ready]

    def get_family_metrics(self) -> Dict[str, Any]:
        """Get metrics for the entire tensor family."""
        if not self.members:
            return {
                "family_size": 0,
                "total_amplitude": 0.0,
                "average_phase": 0.0,
                "spawn_ready_count": 0,
                "lineage_depth": 0
            }

        total_amplitude   = sum(t.amplitude for t in self.members.values())
        average_phase     = sum(t.cumulative_phase for t in self.members.values()) / len(self.members)
        spawn_ready_count = len(self.get_spawn_candidates())

        # Compute maximum lineage depth via DFS
        lineage_depth = 0
        visited = set()

        def dfs(node_id: int, depth: int):
            nonlocal lineage_depth
            visited.add(node_id)
            lineage_depth = max(lineage_depth, depth)
            for child in self.lineage_graph.get(node_id, []):
                if child not in visited:
                    dfs(child, depth + 1)

        # Start DFS from root
        dfs(self.root_tensor_id, 0)

        return {
            "family_size":        len(self.members),
            "total_amplitude":    total_amplitude,
            "average_phase":      average_phase,
            "spawn_ready_count":  spawn_ready_count,
            "lineage_depth":      lineage_depth
        }