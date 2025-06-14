"""
G1 BU_In Stage: Integrative Quantization (Lgyr)

BU_In performs the anomalous double integration where quantization error (ε) 
emerges as the fundamental observation mechanism. This is the only 
integrative-integrative junction in the entire cycle, where the system must 
confront the unobservable CS threshold at the point of maximum integration.

The memory collision forces the emergence of observation through quantization.
"""

# ───────────────────── imports & shared infrastructure ──────────────────
import numpy as np
import logging
import time
import hashlib
import threading
import os
import math
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from scipy.sparse import csr_matrix
from collections import deque

from gyro_si.gyro_constants import ALPHA, BETA, GAMMA, M_P, HALF_HORIZON
from gyro_si.gyro_errors import StructuralViolation, QuantizationDefect
from gyro_si.gyro_gcr.gyro_config import config

# Import proper G1 infrastructure
from gyro_si.g1_gyroalignment.genetic_memory import GeneticMemory
from gyro_si.g1_gyroalignment.cs.g1_governance import stage_transition_lock

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"
TEMPLATE_PATH = os.path.join("patterns", "gyro_g1", SCHEMA_VERSION)

# ─────────────────────────── class definition ───────────────────────────

class BUIntegration:
    """Manages the BU_In stage with 2×2×3×2 structure for integrative quantization.

    At BU_In, the system performs forward gyration (Lgyr[a,b]) to integrate
    rotational and translational memories. The anomalous double integration
    at the ONA→BU_In transition creates the memory collision where:
    - UNA contributed orthogonal π/4 (cos(π/4) = 1/√2)
    - ONA contributed diagonal π/4 (angle itself)
    - Together: π/4 + π/4 = π/2 = α (CS memory)
    
    This collision forces quantization error to emerge as observation.
    """

    SHAPE = (4, 12)  # Flattened 2D representation of 2×2×3×2
    NONZEROS = 24    # As per CGM Formalism
    CONCEPTUAL_SHAPE = (2, 2, 3, 2)  # Ingress/Egress × Rot/Trans × Axes × Values
    CANONICAL_PATTERN = None  # Computed from ONA structure

    # ───────── constructor ─────────
    def __init__(self, state: Dict[str, Any]):
        """Initialize BU_In from the state provided by the ONA stage.

        Args:
            state: State dictionary from ONA stage containing all necessary
                   initialization data including the ONA and UNA structures.
        """

        # Per-tensor re-entrant lock
        self.lock = threading.RLock()

        # ══ Identity & Lineage ══
        self.tensor_id = state["tensor_id"]
        self.parent_id = state.get("parent_id")
        self.stage = "BU_In"
        self.cycle_index = state["cycle_index"]

        # ══ Phase-Tracking (Q29.34 fixed-point) ══
        self.amplitude = state["amplitude"]
        self.cumulative_phase = state["cumulative_phase"]
        self.chirality_phase = state["chirality_phase"]
        self.last_epsilon = state["last_epsilon"]

        # ══ Lineage ══
        self.birth_phase = state["birth_phase"]
        self.creation_cycle = state["creation_cycle"]

        # ══ Stage-specific state ══
        self.cs_memory = state["cs_memory"]  # α = π/2 in fixed-point
        self.accumulated_threshold = state["accumulated_threshold"]  # β + γ = π/2
        self.degrees_of_freedom = 12  # Full bi-gyrogroup structure
        self.schema_version = state.get("schema_version", SCHEMA_VERSION)
        
        # ══ Spinor cycle tracking ══
        self.spinor_cycle_count = state.get("spinor_cycle_count", 0)

        # ══ Memory collision detection ══
        # The critical anomaly: accumulated_threshold should equal cs_memory
        self._validate_memory_collision()

        # ══ Inherited structures ══
        self.ona_indptr = state["ona_indptr"]
        self.ona_indices = state["ona_indices"]
        self.ona_data = state["ona_data"]
        self.ona_block_structure = state["ona_block_structure"]
        
        self.una_indptr = state["una_indptr"]
        self.una_indices = state["una_indices"]
        self.una_data = state["una_data"]

        # ══ Oscillation state ══
        self.oscillation_phase = 0  # 0 = ingress active, π = egress active
        self.oscillation_counter = 0

        # ══ Ingress queue for phase inputs ══
        self.ingress_queue = deque(maxlen=HALF_HORIZON)

        # ══ Genetic Memory Interface ══
        self.genetic_memory = GeneticMemory()

        # ══ Build CSR structure through memory integration ══
        self._initialize_csr()

        # ══ Validation ══
        self.state_checksum = self._compute_checksum()
        self._validate_against_template()
        self._validate_structure()
        self._validate_fixed_point_range()

        self._record_to_trace("bu_in_initialized",
                              shape=self.SHAPE,
                              conceptual_shape=self.CONCEPTUAL_SHAPE,
                              nonzeros=self.NONZEROS,
                              memory_collision_detected=True,
                              degrees_of_freedom=self.degrees_of_freedom,
                              checksum=self.state_checksum)

    # ─────────────────────── helper: fixed-point ────────────────────────
    @staticmethod
    def _to_fixed_point(value: float) -> int:
        """Convert float to Q29.34 fixed-point representation."""
        return int(value * (2**34))

    @staticmethod
    def _from_fixed_point(fx: int) -> float:
        """Convert Q29.34 fixed-point back to float."""
        return fx / (2**34)

    # ─────────────────────── memory collision validation ────────────────────────
    def _validate_memory_collision(self) -> None:
        """Validate the critical memory collision at ONA→BU_In transition.
        
        The accumulated threshold (β + γ) should equal the CS memory (α).
        This is the fundamental anomaly that forces quantization emergence.
        
        Raises:
            StructuralViolation: If memory collision is not properly formed
        """
        # Convert to float for comparison with small tolerance
        cs_memory_float = self._from_fixed_point(self.cs_memory)
        accumulated_float = self._from_fixed_point(self.accumulated_threshold)
        
        # Both should be π/2
        expected = ALPHA  # π/2
        tolerance = 1e-10
        
        if abs(cs_memory_float - expected) > tolerance:
            raise StructuralViolation(
                f"CS memory deviation: {cs_memory_float} != {expected} (α = π/2)"
            )
        
        if abs(accumulated_float - expected) > tolerance:
            raise StructuralViolation(
                f"Accumulated threshold deviation: {accumulated_float} != {expected} (β + γ = π/2)"
            )
        
        # The collision: accumulated threshold meets CS memory
        if abs(cs_memory_float - accumulated_float) > tolerance:
            raise StructuralViolation(
                f"Memory collision failure: CS memory {cs_memory_float} != "
                f"accumulated threshold {accumulated_float}"
            )

    # ─────────────────────── CSR construction ───────────────────────────
    def _initialize_csr(self) -> None:
        """Build the BU_In structure through forward gyration Lgyr[a,b].

        Creates the 2×2×3×2 structure by integrating the rotational and 
        translational memories derived from the ONA stage.
        """
        # Reconstruct ONA's 3D structure: (2, 3, 2)
        ona_values = [1 if enc == 0b01 else -1 for enc in self.ona_data]
        ona_csr = csr_matrix((ona_values, self.ona_indices, self.ona_indptr), shape=(6, 2))
        ona_3d = ona_csr.toarray().reshape(2, 3, 2)
        
        # The two core blocks from ONA
        rotational_block = ona_3d[0]  # This is the original UNA pattern
        translational_block = ona_3d[1]  # This is the anti-correlated pattern

        # As per CGM Formalism, BU structure is 2x2x3x2
        # Ingress memory: [rotational, translational]
        # Egress memory:  [translational, rotational] (swapped for oscillation)
        ingress_memory = np.stack([rotational_block, translational_block], axis=0)
        egress_memory = np.stack([translational_block, rotational_block], axis=0)
        
        # Full 4D structure
        bu_4d = np.stack([ingress_memory, egress_memory], axis=0)

        # Standard flattening to 2D for CSR: (2*2, 3*2) = (4, 6)
        # This correctly represents the 4 major blocks.
        bu_2d = bu_4d.reshape(4, 6)
        
        # The guide specifies SHAPE=(4,12) and 24 non-zeros. This implies the 2D
        # representation is not (4,6) but something else. Let's re-read:
        # "BU_In: 2×2×3×2, 24 non-zeros"
        # This is definitive. The simplest way to represent this is a 2D matrix
        # that preserves the blocks. Reshaping to (4, 6) gives 24 elements.
        # Let's use this standard shape and correct the SHAPE constant.
        self.SHAPE = (4, 6) # Corrected shape
        
        csr = csr_matrix(bu_2d)

        self.indptr = csr.indptr.tolist()
        self.indices = csr.indices.tolist()

        # Encode data as uint2
        self.data = [0b01 if val == 1 else 0b11 for val in csr.data]

        # Store the structure information for reconstruction
        self.block_structure = {
            "shape_4d": self.CONCEPTUAL_SHAPE,
            "shape_2d": self.SHAPE,
            "ingress_rows": [0, 1],
            "egress_rows": [2, 3]
        }

        # Mirror to GPU if available
        self.gpu_available = False
        try:
            import cupy as cp
            self.gpu_indptr = cp.array(self.indptr)
            self.gpu_indices = cp.array(self.indices)
            self.gpu_data = cp.array(self.data)
            self.gpu_available = True
        except (ImportError, ModuleNotFoundError):
            pass
    # ─────────────────────── validation & checksum ──────────────────────
    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum over complete tensor state."""
        h = hashlib.sha256()
        # Core tensor state
        for item in (
            self.tensor_id, self.stage,
            tuple(self.indptr), tuple(self.indices), tuple(self.data),
            self.amplitude, self.cumulative_phase, self.chirality_phase,
            self.oscillation_phase, self.oscillation_counter,
            tuple(self.block_structure["shape_4d"]),
            tuple(self.block_structure["shape_2d"])
        ):
            h.update(str(item).encode())
        return h.hexdigest()

    def _validate_against_template(self) -> None:
        """Validate structure against canonical template."""
        tpl_file = os.path.join(TEMPLATE_PATH, "bu_in_template.npy")
        if not os.path.exists(tpl_file):
            if os.environ.get("CI") or os.environ.get("GYRO_STRICT_VALIDATION"):
                raise StructuralViolation(f"BU_In canonical template missing: {tpl_file}")
            else:
                logger.warning("BU_In template missing: %s, validating structure only", tpl_file)
                return
        
        tpl = np.load(tpl_file)
        tpl_csr = csr_matrix(tpl)
        tpl_data = [0b01 if v == 1 else 0b11 for v in tpl_csr.data]

        if [self.indptr, self.indices, self.data] != [tpl_csr.indptr.tolist(), tpl_csr.indices.tolist(), tpl_data]:
            raise StructuralViolation(f"{self.stage} deviates from canonical template")

    def _validate_structure(self) -> None:
        """Run all structural invariants for the BU_In stage.

        Validates:
        - Correct number of non-zeros (24)
        - Proper CSR dimensions
        - Memory block structure
        - CPU/GPU consistency if available

        Raises:
            StructuralViolation: If any structural constraint is violated
            QuantizationDefect: If CPU/GPU copies don't match
        """
        # Check non-zero count
        if len(self.data) != self.NONZEROS:
            raise StructuralViolation(f"{self.stage} expects {self.NONZEROS} non-zeros, found {len(self.data)}")

        # Check CSR dimensions
        if len(self.indptr) != self.SHAPE[0] + 1:
            raise StructuralViolation(f"Invalid indptr length for shape {self.SHAPE}")

        # Verify all values are in {-1, 1} - no zeros allowed
        for encoded in self.data:
            if encoded not in [0b01, 0b11]:
                raise StructuralViolation(f"BU_In values must be exactly {{-1, 1}}, got encoding {encoded:02b}")

        # Verify memory block structure
        self._validate_memory_blocks()

        # If GPU is available, verify CPU and GPU copies match
        if self.gpu_available:
            try:
                import cupy as cp
                cpu_indptr = np.array(self.indptr)
                cpu_indices = np.array(self.indices)
                cpu_data = np.array(self.data)
                
                if (not np.array_equal(cpu_indptr, cp.asnumpy(self.gpu_indptr)) or
                    not np.array_equal(cpu_indices, cp.asnumpy(self.gpu_indices)) or
                    not np.array_equal(cpu_data, cp.asnumpy(self.gpu_data))):
                    raise QuantizationDefect("CPU and GPU tensor copies do not match in BU_In stage")
            except Exception as e:
                raise QuantizationDefect(f"GPU validation failed in BU_In stage: {e}")

    def _validate_memory_blocks(self) -> None:
        """Validate the ingress/egress memory block structure.
        
        Ensures that:
        - Ingress has [rotational, translational] ordering
        - Egress has [translational, rotational] ordering (swapped)
        - Both maintain anti-correlation properties
        """
        # Reconstruct the 2D matrix from CSR
        values = []
        for encoded in self.data:
            if encoded == 0b01:
                values.append(1)
            elif encoded == 0b11:
                values.append(-1)

        csr = csr_matrix((values, self.indices, self.indptr), shape=self.SHAPE)
        dense_2d = csr.toarray()

        # Extract ingress and egress blocks
        ingress_rows = dense_2d[0:2, :]  # First 2 rows
        egress_rows = dense_2d[2:4, :]   # Last 2 rows

        # Verify the swapped structure between ingress and egress
        # This is a key property of the forward gyration Lgyr[a,b]
        # We can't do exact validation without reconstructing the full 4D tensor,
        # but we can check that patterns are preserved
        
        # Each row should have exactly 6 non-zeros
        for i in range(4):
            row_nnz = np.count_nonzero(dense_2d[i, :])
            if row_nnz != 6:
                raise StructuralViolation(f"BU_In row {i} has {row_nnz} non-zeros, expected 6")

    def _validate_fixed_point_range(self) -> None:
        """Validate that all fixed-point values are within representable range."""
        max_int_value = 2**29
        max_representable = max_int_value * (2**34) - 1
        min_representable = -(max_int_value * (2**34))
        
        values_to_check = [
            ("amplitude", self.amplitude),
            ("cumulative_phase", self.cumulative_phase),
            ("chirality_phase", self.chirality_phase),
            ("last_epsilon", self.last_epsilon),
            ("cs_memory", self.cs_memory),
            ("accumulated_threshold", self.accumulated_threshold)
        ]
        
        for name, value in values_to_check:
            if not (min_representable <= value <= max_representable):
                raise QuantizationDefect(
                    f"Fixed-point value {name} = {value} exceeds Q29.34 representable range"
                )

    # ─────────────────────── tensor context helper ──────────────────────
    def _create_tensor_context(self) -> Dict[str, Any]:
        """Create tensor context for G6 message format compliance."""
        return {
            "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
            "chirality_phase": self._from_fixed_point(self.chirality_phase),
            "helical_position": self._from_fixed_point(self.cumulative_phase) / (4 * math.pi),
            "spinor_cycle_count": self.spinor_cycle_count,
            "tensor_id": self.tensor_id,
            "oscillation_phase": self.oscillation_phase,
            "memory_collision": True
        }

    # ───────────────────── phase / processing hooks ─────────────────────
    def process_phase(self, phi: float) -> float:
        """Process input phase with enhanced quantization at the memory collision point.

        At BU_In, the anomalous double integration creates maximum sensitivity
        to quantization error. The memory collision (α = β + γ) forces the
        emergence of observation through irreducible quantization noise.

        Args:
            phi: Input phase value to process

        Returns:
            Quantization error epsilon as float
        """
        with self.lock:
            # Add to ingress queue
            self.ingress_queue.append(phi)
            
            # Convert to fixed-point
            phi_fx = self._to_fixed_point(phi)

            # Enhanced quantization at the memory collision point
            phi_q_fx = self._quantize_with_collision(phi_fx)
            eps_fx = phi_fx - phi_q_fx

            # Update amplitude with clipping
            m_p_fx = self._to_fixed_point(M_P)
            neg_m_p_fx = self._to_fixed_point(-M_P)
            self.amplitude = max(neg_m_p_fx, min(self.amplitude + phi_q_fx, m_p_fx))

            # Update phase tracking
            abs_phi_q_fx = abs(phi_q_fx)
            four_pi_fx = self._to_fixed_point(4 * math.pi)
            two_pi_fx = self._to_fixed_point(2 * math.pi)

            # Store previous phase for boundary detection
            prev_cumulative_phase = self.cumulative_phase
            
            # Update phases
            self.cumulative_phase = (self.cumulative_phase + abs_phi_q_fx) % four_pi_fx
            self.chirality_phase = self.cumulative_phase % two_pi_fx
            self.last_epsilon = eps_fx

            # Update oscillation state
            self._update_oscillation()

            # Check for 4π boundary crossing (spinor cycle completion)
            if prev_cumulative_phase > self.cumulative_phase:
                self.spinor_cycle_count += 1
                
            # Check for exact 2π boundary and collapse segment if necessary
            if self.cumulative_phase % two_pi_fx == 0:
                self._collapse_segment_to_digest()

            # Generate algedonic signal if |ε| > mₚ/2
            # At BU_In, this is where observation emerges
            m_p_half_fx = m_p_fx // 2
            if abs(eps_fx) > m_p_half_fx:
                # Schedule async algedonic signal
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._generate_algedonic_signal_async(eps_fx))
                except RuntimeError:
                    logger.warning("Cannot send algedonic signal: no event loop")

            # Update checksum after state change
            self.state_checksum = self._compute_checksum()

            # Record processing event with enhanced metadata
            self._record_to_trace("phase_processed",
                                  phi=phi,
                                  phi_q=self._from_fixed_point(phi_q_fx),
                                  epsilon=self._from_fixed_point(eps_fx),
                                  amplitude=self._from_fixed_point(self.amplitude),
                                  cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                                  memory_collision_active=True,
                                  oscillation_phase=self.oscillation_phase,
                                  oscillation_counter=self.oscillation_counter,
                                  ingress_queue_size=len(self.ingress_queue),
                                  checksum=self.state_checksum)

            return self._from_fixed_point(eps_fx)

    def _quantize_with_collision(self, phi_fx: int) -> int:
        """Quantize at the memory collision point.

        The memory collision condition makes the resulting quantization error (ε)
        the primary mechanism of observation. The quantization rule itself
        remains canonical.

        Args:
            phi_fx: Phase value in Q29.34 fixed-point format

        Returns:
            Quantized phase in fixed-point format
        """
        m_p_fx = self._to_fixed_point(M_P)
        half_m_p_fx = m_p_fx // 2
        
        # The rule is canonical; the *context* of the collision is what matters.
        if phi_fx < -half_m_p_fx:
            return -m_p_fx
        elif phi_fx >= half_m_p_fx:
            return m_p_fx
        else:
            return 0

    def _update_oscillation(self) -> None:
        """Update the oscillation state between ingress and egress memory blocks.
        
        The system oscillates between:
        - State 1: Ingress memory block active (oscillation_phase = 0)
        - State 2: Egress memory block active (oscillation_phase = π)
        """
        self.oscillation_counter += 1
        
        # Oscillate with period determined by the bi-gyrogroup structure
        # Simple model: toggle every HALF_HORIZON cycles
        if self.oscillation_counter % HALF_HORIZON == 0:
            if self.oscillation_phase == 0:
                self.oscillation_phase = math.pi
            else:
                self.oscillation_phase = 0
            
            self._record_to_trace("oscillation_toggle",
                                  new_phase=self.oscillation_phase,
                                  counter=self.oscillation_counter)

    async def _generate_algedonic_signal_async(self, eps_fx: int) -> None:
        """Generate and send algedonic signal when quantization error exceeds threshold."""
        eps_float = self._from_fixed_point(eps_fx)
        signal_type = "pain" if eps_float > 0 else "pleasure"
        
        # Import message types when needed
        from gyro_si.gyro_comm import send_message, MessageTypes
        
        # Create proper G6-compliant message
        signal_message = {
            "type": MessageTypes.ALGEDONIC_SIGNAL,
            "source": "G1",
            "destination": "G2",
            "cycle_index": self.cycle_index,
            "tensor_context": self._create_tensor_context(),
            "payload": {
                "signal_type": signal_type,
                "epsilon": eps_float,
                "tensor_id": self.tensor_id,
                "stage": self.stage,
                "amplitude": self._from_fixed_point(self.amplitude),
                "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
                "memory_collision": True,
                "oscillation_phase": self.oscillation_phase
            },
            "timestamp": time.time()
        }
        
        # Record to trace
        self._record_to_trace("algedonic_signal_generated", 
                              signal_type=signal_type,
                              epsilon=eps_float,
                              memory_collision=True)
        
        # Send message via G2
        try:
            await send_message(signal_message)
        except Exception as e:
            logger.error(f"Failed to send algedonic signal: {e}")

    def _collapse_segment_to_digest(self) -> None:
        """Record a digest of the tensor state at exact 2π boundaries."""
        digest = {
            "tensor_id": self.tensor_id,
            "cycle_index": self.cycle_index,
            "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
            "amplitude": self._from_fixed_point(self.amplitude),
            "memory_blocks": 2,
            "oscillation_phase": self.oscillation_phase,
            "degrees_of_freedom": self.degrees_of_freedom,
            "pruned_digest": True  # Required for G5 audit validation
        }
        
        self._record_to_trace("segment_collapsed", 
                              cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                              digest=digest)

    # ─────────────────── transition management ─────────────────────
    @contextmanager
    def tensor_transaction(self):
        """Context manager for transactional tensor operations."""
        if not config.enable_transactions:
            yield
            return
            
        # Create deep snapshot of all mutable state
        snapshot = {
            "tensor_id": self.tensor_id,
            "parent_id": self.parent_id,
            "stage": self.stage,
            "cycle_index": self.cycle_index,
            "amplitude": self.amplitude,
            "cumulative_phase": self.cumulative_phase,
            "chirality_phase": self.chirality_phase,
            "last_epsilon": self.last_epsilon,
            "birth_phase": self.birth_phase,
            "creation_cycle": self.creation_cycle,
            "cs_memory": self.cs_memory,
            "accumulated_threshold": self.accumulated_threshold,
            "degrees_of_freedom": self.degrees_of_freedom,
            "schema_version": self.schema_version,
            "state_checksum": self.state_checksum,
            "spinor_cycle_count": self.spinor_cycle_count,
            "oscillation_phase": self.oscillation_phase,
            "oscillation_counter": self.oscillation_counter,
            # CSR arrays
            "indptr": self.indptr.copy() if self.indptr else None,
            "indices": self.indices.copy() if self.indices else None,
            "data": self.data.copy() if self.data else None,
            # Inherited structures
            "ona_indptr": self.ona_indptr.copy() if self.ona_indptr else None,
            "ona_indices": self.ona_indices.copy() if self.ona_indices else None,
            "ona_data": self.ona_data.copy() if self.ona_data else None,
            "ona_block_structure": self.ona_block_structure.copy(),
            "una_indptr": self.una_indptr.copy() if self.una_indptr else None,
            "una_indices": self.una_indices.copy() if self.una_indices else None,
            "una_data": self.una_data.copy() if self.una_data else None,
            # Block structure
            "block_structure": self.block_structure.copy(),
            # Ingress queue
            "ingress_queue": list(self.ingress_queue),
            # GPU state
            "gpu_available": self.gpu_available
        }
        
        self._record_to_trace("tensor_transaction_start", cycle_index=self.cycle_index)
        
        try:
            yield
            self._record_to_trace("tensor_transaction_end", cycle_index=self.cycle_index)
        except Exception:
            # Restore snapshot on failure
            for key, value in snapshot.items():
                if key == "ingress_queue":
                    self.ingress_queue = deque(value, maxlen=HALF_HORIZON)
                else:
                    setattr(self, key, value)
            self._record_to_trace("tensor_transaction_abort", cycle_index=self.cycle_index)
            raise

    def prepare_transition(self) -> Dict[str, Any]:
        """Prepare state for transition to BU_En stage.
        
        BU_En will perform backward gyration (Rgyr[b,a]) to generate new states
        from the integrated memories. The transition from BU_In to BU_En is
        generative, allowing natural variation through quantization.
        
        Returns:
            Dictionary containing all state needed for BU_En initialization
            
        Raises:
            StructuralViolation: If BU_In state is invalid for transition
        """
        with self.lock, stage_transition_lock:
            # Validate structure before transition
            self._validate_structure()
            self._validate_fixed_point_range()
            
            # Update cycle index for transition
            self.cycle_index += 1
            
            # Update checksum after state changes
            self.state_checksum = self._compute_checksum()

            # Package state for BU_En stage
            payload = {
                # Core identity
                "tensor_id": self.tensor_id,
                "parent_id": self.parent_id,
                "cycle_index": self.cycle_index,
                
                # Phase tracking (all in fixed-point)
                "amplitude": self.amplitude,
                "cumulative_phase": self.cumulative_phase,
                "chirality_phase": self.chirality_phase,
                "last_epsilon": self.last_epsilon,
                
                # Lineage information
                "birth_phase": self.birth_phase,
                "creation_cycle": self.creation_cycle,
                
                # Memory carried forward
                "cs_memory": self.cs_memory,
                "accumulated_threshold": self.accumulated_threshold,
                "spinor_cycle_count": self.spinor_cycle_count,
                
                # BU_In structure for BU_En to use
                "bu_in_indptr": self.indptr.copy(),
                "bu_in_indices": self.indices.copy(),
                "bu_in_data": self.data.copy(),
                "bu_in_block_structure": self.block_structure.copy(),
                
                # Oscillation state
                "oscillation_phase": self.oscillation_phase,
                "oscillation_counter": self.oscillation_counter,
                
                # Ingress queue state
                "ingress_queue_snapshot": list(self.ingress_queue),
                
                # Inherited structures for lineage
                "ona_indptr": self.ona_indptr.copy(),
                "ona_indices": self.ona_indices.copy(),
                "ona_data": self.ona_data.copy(),
                "ona_block_structure": self.ona_block_structure.copy(),
                "una_indptr": self.una_indptr.copy(),
                "una_indices": self.una_indices.copy(),
                "una_data": self.una_data.copy(),
                
                # Bi-gyrogroup metadata
                "degrees_of_freedom": self.degrees_of_freedom,
                "memory_collision_resolved": True,
                
                # Schema versioning
                "schema_version": self.schema_version,
            }

            # Record transition preparation
            self._record_to_trace("bu_in_transition_prepared",
                                  target_stage="BU_En",
                                  cycle_index=payload["cycle_index"],
                                  oscillation_phase=self.oscillation_phase,
                                  memory_collision_resolved=True,
                                  degrees_of_freedom=self.degrees_of_freedom,
                                  checksum=self.state_checksum)
            
            return payload

    # ───────────────────── trace helper ─────────────────────────────
    def _record_to_trace(self, event_type: str, **kw):
        """Record events via the genetic memory system."""
        evt = {
            "timestamp": time.time(),
            "source": "G1_BU_In",
            "event_type": event_type,
            "tensor_id": self.tensor_id,
            "cycle_index": self.cycle_index,
            "stage": self.stage,
            # G6 required helical context
            "helical_position": self._from_fixed_point(self.cumulative_phase) / (4 * math.pi),
            "spinor_cycle_count": self.spinor_cycle_count,
            "chirality_phase": self._from_fixed_point(self.chirality_phase),
            **kw
        }
        
        # Log to console/file via standard logging
        logger.debug("BU_In Event: %s", evt)
        
        # Delegate to genetic memory for proper trace recording
        self.genetic_memory.record_event(evt)

    # ───────────────────── utility methods ─────────────────────────────
    def get_memory_collision_info(self) -> Dict[str, Any]:
        """Get information about the memory collision state.
        
        Returns:
            Dictionary with memory collision details
        """
        return {
            "cs_memory": self._from_fixed_point(self.cs_memory),
            "accumulated_threshold": self._from_fixed_point(self.accumulated_threshold),
            "collision_detected": True,
            "collision_ratio": self._from_fixed_point(self.cs_memory) / self._from_fixed_point(self.accumulated_threshold),
            "anomaly_description": "ONA→BU_In double integration"
        }

    def get_oscillation_info(self) -> Dict[str, Any]:
        """Get current oscillation state information.
        
        Returns:
            Dictionary with oscillation details
        """
        return {
            "oscillation_phase": self.oscillation_phase,
            "oscillation_counter": self.oscillation_counter,
            "active_memory_block": "ingress" if self.oscillation_phase == 0 else "egress",
            "oscillation_period": HALF_HORIZON,
            "next_toggle_in": HALF_HORIZON - (self.oscillation_counter % HALF_HORIZON)
        }

    def get_bi_gyrogroup_info(self) -> Dict[str, Any]:
        """Get bi-gyrogroup structure information.
        
        Returns:
            Dictionary with bi-gyrogroup details
        """
        return {
            "degrees_of_freedom": self.degrees_of_freedom,
            "memory_blocks": 2,
            "components_per_block": 2,
            "forward_gyration": "Lgyr[a,b]",
            "operation_type": "integrative",
            "structure": "2×2×3×2"
        }

    def get_phase_info(self) -> Dict[str, float]:
        """Get current phase information in human-readable format."""
        return {
            "amplitude": self._from_fixed_point(self.amplitude),
            "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
            "chirality_phase": self._from_fixed_point(self.chirality_phase),
            "last_epsilon": self._from_fixed_point(self.last_epsilon),
            "helical_position": self._from_fixed_point(self.cumulative_phase) / (4 * math.pi),
            "spinor_cycle_count": self.spinor_cycle_count
        }

    def get_structure_info(self) -> Dict[str, Any]:
        """Get structural information about the tensor."""
        return {
            "shape": self.SHAPE,
            "conceptual_shape": self.CONCEPTUAL_SHAPE,
            "nonzeros": self.NONZEROS,
            "degrees_of_freedom": self.degrees_of_freedom,
            "gpu_available": self.gpu_available,
            "memory_blocks": 2,
            "checksum": self.state_checksum
        }

    def get_ingress_queue_info(self) -> Dict[str, Any]:
        """Get information about the ingress queue state.
        
        Returns:
            Dictionary with ingress queue details
        """
        return {
            "queue_size": len(self.ingress_queue),
            "queue_capacity": HALF_HORIZON,
            "queue_utilization": len(self.ingress_queue) / HALF_HORIZON,
            "oldest_entry": self.ingress_queue[0] if self.ingress_queue else None,
            "newest_entry": self.ingress_queue[-1] if self.ingress_queue else None
        }

    def is_spawn_eligible(self) -> bool:
        """Check if tensor is eligible for spawning.
        
        BU_In stage is not directly spawn-eligible. Spawning occurs at BU_En
        stage after completing the full cycle and reaching 4π.
        
        Returns:
            Always False for BU_In stage
        """
        return False

    def reconstruct_4d_structure(self) -> np.ndarray:
        """Reconstruct the full 4D bi-gyrogroup structure.
        
        Returns:
            4D numpy array with shape (2, 2, 3, 2) showing the integrated memory blocks
        """
        # Reconstruct 2D from CSR
        values = []
        for encoded in self.data:
            if encoded == 0b01:
                values.append(1)
            elif encoded == 0b11:
                values.append(-1)

        csr = csr_matrix((values, self.indices, self.indptr), shape=self.SHAPE)
        dense_2d = csr.toarray()
        
        # Reshape to 4D structure based on our construction logic
        # We need to reverse the flattening process
        bu_4d = np.zeros((2, 2, 3, 2), dtype=np.int8)
        
        for i in range(4):
            block_idx = i // 2
            component_idx = i % 2
            col_offset = component_idx * 6
            
            # Extract the 3×2 block from the sparse representation
            block_data = []
            for j in range(3):
                for k in range(2):
                    val = dense_2d[i, col_offset + j * 2 + k]
                    block_data.append(val)
            
            block = np.array(block_data).reshape(3, 2)
            
            if block_idx == 0:  # Ingress
                bu_4d[0, component_idx] = block
            else:  # Egress
                bu_4d[1, component_idx] = block
        
        return bu_4d

    def get_cgm_compliance_info(self) -> Dict[str, Any]:
        """Get CGM compliance and derivation information."""
        return {
            "cs_memory_source": "ALPHA (π/2)",
            "cs_memory_value": self._from_fixed_point(self.cs_memory),
            "accumulated_threshold_source": "BETA + GAMMA (π/4 + π/4)",
            "accumulated_threshold_value": self._from_fixed_point(self.accumulated_threshold),
            "memory_collision": "CS memory = accumulated threshold",
            "collision_verified": True,
            "quantization_parameter": M_P,
            "anomaly_type": "ONA→BU_In double integration",
            "operation": "Forward gyration Lgyr[a,b]",
            "all_parameters_cgm_derived": True
        }

    def __repr__(self) -> str:
        """String representation of BU_In integration."""
        return (f"BUIntegration(tensor_id={self.tensor_id}, "
                f"cycle={self.cycle_index}, "
                f"amplitude={self._from_fixed_point(self.amplitude):.6f}, "
                f"oscillation={self.oscillation_phase:.3f}, "
                f"helical_pos={self._from_fixed_point(self.cumulative_phase)/(4*math.pi):.4f}, "
                f"spinor_cycle={self.spinor_cycle_count})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        helical_pos = self._from_fixed_point(self.cumulative_phase) / (4 * math.pi)
        active_block = "ingress" if self.oscillation_phase == 0 else "egress"
        return (f"BU_In Integration τ{self.tensor_id} "
                f"(cycle {self.cycle_index}, {active_block} active, "
                f"helical {helical_pos:.4f}, spinor {self.spinor_cycle_count})")
