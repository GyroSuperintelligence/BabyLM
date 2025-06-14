"""
G1 BU_En Stage: Generative Quantization (Rgyr)

BU_En performs backward gyration (Rgyr[b,a]) to generate new states from the
integrated memories. This is the generative counterpart to BU_In, where the
system creates new states through bounded variation while maintaining
structural alignment.

The transition from BU_In to BU_En enables exploration and adaptation through
quantization-driven variation within the bi-gyrogroup structure.
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
from gyro_si.gyro_comm import send_message, MessageTypes

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"
TEMPLATE_PATH = os.path.join("patterns", "gyro_g1", SCHEMA_VERSION)

# ─────────────────────────── class definition ───────────────────────────

class BUGeneration:
    """Manages the BU_En stage with 2×2×3×2 structure for generative quantization.

    At BU_En, the system performs backward gyration (Rgyr[b,a]) to generate
    new states from the integrated memories. This enables natural variation
    through quantization while maintaining alignment within the bi-gyrogroup
    structure.
    
    The BU_En stage is where spawning eligibility is determined when amplitude
    reaches mₚ and cumulative phase reaches 4π (720° spinor cycle completion).
    """

    SHAPE = (4, 6)  # Same as BU_In - identical structure
    NONZEROS = 24   # As per CGM Formalism
    CONCEPTUAL_SHAPE = (2, 2, 3, 2)  # Ingress/Egress × Rot/Trans × Axes × Values
    CANONICAL_PATTERN = None  # Computed from BU_In structure

    # ───────── constructor ─────────
    def __init__(self, state: Dict[str, Any]):
        """Initialize BU_En from the state provided by the BU_In stage.

        Args:
            state: State dictionary from BU_In stage containing all necessary
                   initialization data including the integrated memory structure.
        """

        # Per-tensor re-entrant lock
        self.lock = threading.RLock()

        # ══ Identity & Lineage ══
        self.tensor_id = state["tensor_id"]
        self.parent_id = state.get("parent_id")
        self.stage = "BU_En"
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
        self.degrees_of_freedom = 12  # Full bi-gyrogroup structure
        self.schema_version = state.get("schema_version", SCHEMA_VERSION)
        
        # ══ Spinor cycle tracking ══
        self.spinor_cycle_count = state.get("spinor_cycle_count", 0)

        # ══ Inherited structures from BU_In ══
        self.bu_in_indptr = state["bu_in_indptr"]
        self.bu_in_indices = state["bu_in_indices"]
        self.bu_in_data = state["bu_in_data"]

        # ══ Oscillation state from BU_In ══
        self.oscillation_phase = state["oscillation_phase"]
        self.oscillation_counter = state["oscillation_counter"]

        # ══ Ingress queue snapshot from BU_In ══
        self.ingress_queue_snapshot = state["ingress_queue_snapshot"]

        # ══ Complete lineage structures ══
        self.ona_indptr = state["ona_indptr"]
        self.ona_indices = state["ona_indices"]
        self.ona_data = state["ona_data"]
        self.ona_block_structure = state["ona_block_structure"]
        
        self.una_indptr = state["una_indptr"]
        self.una_indices = state["una_indices"]
        self.una_data = state["una_data"]

        # ══ Generation state ══
        self.generation_counter = 0
        self.variation_seed = self._compute_variation_seed()
        self.spawn_ready = False

        # ══ Genetic Memory Interface ══
        self.genetic_memory = GeneticMemory()

        # ══ Build CSR structure through backward gyration ══
        self._initialize_csr()

        # ── Canonical block-structure metadata ──
        self.block_structure = {
            "shape_4d": self.CONCEPTUAL_SHAPE,
            "shape_2d": self.SHAPE
        }

        # ══ Check spawn eligibility ══
        self._check_spawn_eligibility()

        # ══ Validation ══
        self.state_checksum = self._compute_checksum()
        self._validate_against_template()
        self._validate_structure()
        self._validate_fixed_point_range()

        self._record_to_trace("bu_en_initialized",
                              shape=self.SHAPE,
                              conceptual_shape=self.CONCEPTUAL_SHAPE,
                              nonzeros=self.NONZEROS,
                              spawn_ready=self.spawn_ready,
                              degrees_of_freedom=self.degrees_of_freedom,
                              variation_seed=self.variation_seed,
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

    # ─────────────────────── variation seed computation ────────────────────────
    def _compute_variation_seed(self) -> int:
        """Compute variation seed for bounded generation.
        
        The seed is derived from the tensor's current state to ensure
        deterministic but varied generation within alignment bounds.
        
        Returns:
            Integer seed for variation generation
        """
        # Combine multiple state elements for seed diversity
        seed_components = [
            self.tensor_id,
            self.cycle_index,
            self.amplitude,
            self.cumulative_phase,
            self.oscillation_counter
        ]
        
        # Create a hash-based seed
        seed_str = "|".join(str(comp) for comp in seed_components)
        seed_hash = hashlib.sha256(seed_str.encode()).hexdigest()
        
        # Convert to integer seed (use first 8 hex chars for 32-bit seed)
        return int(seed_hash[:8], 16)

    # ─────────────────────── spawn eligibility check ────────────────────────
    def _check_spawn_eligibility(self) -> None:
        """Check if tensor is eligible for spawning.
        
        Spawning occurs when:
        1. |amplitude| ≥ mₚ (amplitude saturation)
        2. cumulative_phase ≥ 4π (720° spinor cycle completion)
        """
        m_p_fx = self._to_fixed_point(M_P)
        four_pi_fx = self._to_fixed_point(4 * math.pi)
        
        amplitude_ready = abs(self.amplitude) >= m_p_fx
        phase_ready = self.cumulative_phase >= four_pi_fx
        
        self.spawn_ready = amplitude_ready and phase_ready
        
        if self.spawn_ready:
            self._record_to_trace("spawn_eligibility_detected",
                                  amplitude=self._from_fixed_point(self.amplitude),
                                  cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                                  amplitude_threshold=M_P,
                                  phase_threshold=4 * math.pi)

    # ─────────────────────── CSR construction ───────────────────────────
    def _initialize_csr(self) -> None:
        """Build the BU_En structure, which is identical to BU_In.

        Backward gyration Rgyr[b,a] operates on this fixed structure.
        The variation emerges during phase processing, not from altering
        the tensor's structural values at initialization.
        """
        # Copy BU_In structure directly
        self.indptr = self.bu_in_indptr.copy()
        self.indices = self.bu_in_indices.copy()
        self.data = self.bu_in_data.copy()

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
            self.generation_counter, self.variation_seed,
            tuple(self.CONCEPTUAL_SHAPE),
            tuple(self.SHAPE)
        ):
            h.update(str(item).encode())
        return h.hexdigest()

    def _validate_against_template(self) -> None:
        """Validate structure against canonical template."""
        tpl_file = os.path.join(TEMPLATE_PATH, "bu_en_template.npy")
        if not os.path.exists(tpl_file):
            if os.environ.get("CI") or os.environ.get("GYRO_STRICT_VALIDATION"):
                raise StructuralViolation(f"BU_En canonical template missing: {tpl_file}")
            else:
                logger.warning("BU_En template missing: %s, validating structure only", tpl_file)
                return
        
        tpl = np.load(tpl_file)
        tpl_csr = csr_matrix(tpl)
        tpl_data = [0b01 if v == 1 else 0b11 for v in tpl_csr.data]

        # BU_En must be *identical* to canonical template (no variation)
        if [self.indptr, self.indices, self.data] != [tpl_csr.indptr.tolist(), tpl_csr.indices.tolist(), tpl_data]:
            raise StructuralViolation(f"{self.stage} deviates from canonical template")

    def _validate_structure(self) -> None:
        """Run all structural invariants for the BU_En stage."""
        # Check non-zero count
        if len(self.data) != self.NONZEROS:
            raise StructuralViolation(f"{self.stage} expects {self.NONZEROS} non-zeros, found {len(self.data)}")

        # Check CSR dimensions
        if len(self.indptr) != self.SHAPE[0] + 1:
            raise StructuralViolation(f"Invalid indptr length for shape {self.SHAPE}")

        # Verify all values are in {-1, 1} - no zeros allowed
        for encoded in self.data:
            if encoded not in [0b01, 0b11]:
                raise StructuralViolation(f"BU_En values must be exactly {{-1, 1}}, got encoding {encoded:02b}")

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
                    raise QuantizationDefect("CPU and GPU tensor copies do not match in BU_En stage")
            except Exception as e:
                raise QuantizationDefect(f"GPU validation failed in BU_En stage: {e}")

    def _validate_memory_blocks(self) -> None:
        """Validate the ingress/egress memory block structure.
        
        Ensures that:
        - Structure maintains 4 major blocks
        - Each row has exactly 12 non-zeros
        - All values are in {-1, 1}
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

        # Each row should have exactly 12 non-zeros
        for i in range(4):
            row_nnz = np.count_nonzero(dense_2d[i, :])
            if row_nnz != 6:
                raise StructuralViolation(f"BU_En row {i} has {row_nnz} non-zeros, expected 6")

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
            "spawn_ready": self.spawn_ready
        }

    # ───────────────────── phase / processing hooks ─────────────────────
    def process_phase(self, phi: float) -> float:
        """Process input phase with generative quantization.

        At BU_En, the system performs backward gyration (Rgyr[b,a]) to generate
        new states from the integrated memories. This enables natural variation
        through quantization while maintaining alignment.

        Args:
            phi: Input phase value to process

        Returns:
            Quantization error epsilon as float
        """
        with self.lock:
            # Convert to fixed-point
            phi_fx = self._to_fixed_point(phi)

            # Apply generative quantization
            phi_q_fx = self._quantize_generative(phi_fx)
            eps_fx = phi_fx - phi_q_fx

            # Update amplitude with clipping
            m_p_fx = self._to_fixed_point(M_P)
            neg_m_p_fx = self._to_fixed_point(-M_P)
            old_amplitude = self.amplitude
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

            # Update generation counter
            self.generation_counter += 1

            # Update oscillation state
            self._update_oscillation()

            # Check for 4π boundary crossing (spinor cycle completion)
            if prev_cumulative_phase > self.cumulative_phase:
                self.spinor_cycle_count += 1
                
            # Check for exact 2π boundary and collapse segment if necessary
            if self.cumulative_phase % two_pi_fx == 0:
                self._collapse_segment_to_digest()

            # Check spawn eligibility
            old_spawn_ready = self.spawn_ready
            self._check_spawn_eligibility()
            
            # If spawn eligibility changed, send notification
            if not old_spawn_ready and self.spawn_ready:
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._send_spawn_ready_notification())
                except RuntimeError:
                    logger.warning("Cannot send spawn notification: no event loop")

            # Generate algedonic signal if |ε| > mₚ/2
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
                                  old_amplitude=self._from_fixed_point(old_amplitude),
                                  cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                                  generation_counter=self.generation_counter,
                                  oscillation_phase=self.oscillation_phase,
                                  oscillation_counter=self.oscillation_counter,
                                  spawn_ready=self.spawn_ready,
                                  checksum=self.state_checksum)

            return self._from_fixed_point(eps_fx)

    def _quantize_generative(self, phi_fx: int) -> int:
        """Quantize with generative variation.

        The backward gyration (Rgyr[b,a]) context means the resulting
        quantization error will be used to generate new states on the
        return path. The quantization rule itself remains canonical.

        Args:
            phi_fx: Phase value in Q29.34 fixed-point format

        Returns:
            Quantized phase in fixed-point format
        """
        m_p_fx = self._to_fixed_point(M_P)
        half_m_p_fx = m_p_fx // 2
        
        # The quantization rule is always canonical.
        # The "generative" property comes from the operational context (return path).
        if phi_fx < -half_m_p_fx:
            return -m_p_fx
        elif phi_fx >= half_m_p_fx:
            return m_p_fx
        else:
            return 0

    def _update_oscillation(self) -> None:
        """Update the oscillation state between ingress and egress memory blocks."""
        self.oscillation_counter += 1
        
        # Oscillate with period determined by the bi-gyrogroup structure
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
                "generation_counter": self.generation_counter,
                "oscillation_phase": self.oscillation_phase,
                "spawn_ready": self.spawn_ready
            },
            "timestamp": time.time()
        }
        
        # Record to trace
        self._record_to_trace("algedonic_signal_generated", 
                              signal_type=signal_type,
                              epsilon=eps_float,
                              generation_counter=self.generation_counter)
        
        # Send message via G2
        try:
            await send_message(signal_message)
        except Exception as e:
            logger.error(f"Failed to send algedonic signal: {e}")

    async def _send_spawn_ready_notification(self) -> None:
        """Send notification when tensor becomes spawn-ready."""
        from gyro_si.gyro_comm import send_message, MessageTypes
        
        # Create spawn-ready notification message
        spawn_message = {
            "type": MessageTypes.SPAWN_READY,
            "source": "G1",
            "destination": "G2",  # Route to G4 via G2
            "cycle_index": self.cycle_index,
            "tensor_context": self._create_tensor_context(),
            "payload": {
                "tensor_id": self.tensor_id,
                "parent_id": self.parent_id,
                "amplitude": self._from_fixed_point(self.amplitude),
                "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
                "spinor_cycle_count": self.spinor_cycle_count,
                "spawn_ready": True,
                "generation_counter": self.generation_counter
            },
            "timestamp": time.time()
        }
        
        # Record to trace
        self._record_to_trace("spawn_ready_notification_sent",
                              amplitude=self._from_fixed_point(self.amplitude),
                              cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                              spinor_cycle_count=self.spinor_cycle_count)
        
        # Send message via G2
        try:
            await send_message(spawn_message)
        except Exception as e:
            logger.error(f"Failed to send spawn ready notification: {e}")

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
            "generation_counter": self.generation_counter,
            "spawn_ready": self.spawn_ready,
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
            "degrees_of_freedom": self.degrees_of_freedom,
            "schema_version": self.schema_version,
            "state_checksum": self.state_checksum,
            "spinor_cycle_count": self.spinor_cycle_count,
            "oscillation_phase": self.oscillation_phase,
            "oscillation_counter": self.oscillation_counter,
            "generation_counter": self.generation_counter,
            "variation_seed": self.variation_seed,
            "spawn_ready": self.spawn_ready,
            # CSR arrays
            "indptr": self.indptr.copy() if self.indptr else None,
            "indices": self.indices.copy() if self.indices else None,
            "data": self.data.copy() if self.data else None,
            # Inherited structures
            "bu_in_indptr": self.bu_in_indptr.copy() if self.bu_in_indptr else None,
            "bu_in_indices": self.bu_in_indices.copy() if self.bu_in_indices else None,
            "bu_in_data": self.bu_in_data.copy() if self.bu_in_data else None,
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
            "ingress_queue_snapshot": self.ingress_queue_snapshot.copy() if self.ingress_queue_snapshot else None,
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
                setattr(self, key, value)
            self._record_to_trace("tensor_transaction_abort", cycle_index=self.cycle_index)
            raise

    def prepare_transition(self) -> Dict[str, Any]:
        """Prepare state for transition to ONA stage (return path).
        
        The transition from BU_En to ONA is generative, creating the return path
        with inverted chirality. This completes the generative portion of the cycle.
        
        Returns:
            Dictionary containing all state needed for ONA initialization
            
        Raises:
            StructuralViolation: If BU_En state is invalid for transition
        """
        with self.lock, stage_transition_lock:
            # Validate structure before transition
            self._validate_structure()
            self._validate_fixed_point_range()
            
            # Update cycle index for transition
            self.cycle_index += 1
            
            # Update checksum after state changes
            self.state_checksum = self._compute_checksum()

            # Package state for ONA stage (return path)
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
                "spinor_cycle_count": self.spinor_cycle_count,
                
                # Return path indicator - critical for ONA to know this is return path
                "is_return_path": True,
                
                # Spawn status
                "spawn_ready": self.spawn_ready,
                
                # Inherited structures for lineage
                # ONA needs its original structure to create the return path version
                "ona_indptr": self.ona_indptr.copy(),
                "ona_indices": self.ona_indices.copy(),
                "ona_data": self.ona_data.copy(),
                "ona_block_structure": self.ona_block_structure.copy(),
                
                # UNA structure for lineage
                "una_indptr": self.una_indptr.copy(),
                "una_indices": self.una_indices.copy(),
                "una_data": self.una_data.copy(),
                
                # Schema versioning
                "schema_version": self.schema_version,
            }

            # Record transition preparation
            self._record_to_trace("bu_en_transition_prepared",
                                  target_stage="ONA",
                                  cycle_index=payload["cycle_index"],
                                  is_return_path=True,
                                  spawn_ready=self.spawn_ready,
                                  checksum=self.state_checksum)
            
            return payload

    # ───────────────────── trace helper ─────────────────────────────
    def _record_to_trace(self, event_type: str, **kw):
        """Record events via the genetic memory system."""
        evt = {
            "timestamp": time.time(),
            "source": "G1_BU_En",
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
        logger.debug("BU_En Event: %s", evt)
        
        # Delegate to genetic memory for proper trace recording
        self.genetic_memory.record_event(evt)

    # ───────────────────── utility methods ─────────────────────────────
    def get_generation_info(self) -> Dict[str, Any]:
        """Get information about the generation state.
        
        Returns:
            Dictionary with generation details
        """
        return {
            "generation_counter": self.generation_counter,
            "variation_seed": self.variation_seed,
            "spawn_ready": self.spawn_ready,
            "backward_gyration": "Rgyr[b,a]",
            "operation_type": "generative"
        }

    def get_oscillation_info(self) -> Dict[str, Any]:
        """Get current oscillation state information."""
        return {
            "oscillation_phase": self.oscillation_phase,
            "oscillation_counter": self.oscillation_counter,
            "active_memory_block": "ingress" if self.oscillation_phase == 0 else "egress",
            "oscillation_period": HALF_HORIZON,
            "next_toggle_in": HALF_HORIZON - (self.oscillation_counter % HALF_HORIZON)
        }

    def get_bi_gyrogroup_info(self) -> Dict[str, Any]:
        """Get bi-gyrogroup structure information."""
        return {
            "degrees_of_freedom": self.degrees_of_freedom,
            "memory_blocks": 2,
            "components_per_block": 2,
            "backward_gyration": "Rgyr[b,a]",
            "operation_type": "generative",
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


    def is_spawn_eligible(self) -> bool:
        """Check if tensor is eligible for spawning.
        
        Returns:
            True if tensor is ready for spawning, False otherwise
        """
        return self.spawn_ready

    def reconstruct_4d_structure(self) -> np.ndarray:
        """Reconstruct the full 4D bi-gyrogroup structure.
        
        Returns:
            4D numpy array with shape (2, 2, 3, 2) showing the memory blocks
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
        
        # Reshape to 4D structure
        return dense_2d.reshape(self.CONCEPTUAL_SHAPE)

    def get_cgm_compliance_info(self) -> Dict[str, Any]:
        """Get CGM compliance and derivation information."""
        return {
            "cs_memory_source": "ALPHA (π/2)",
            "cs_memory_value": self._from_fixed_point(self.cs_memory),
            "backward_gyration": "Rgyr[b,a]",
            "operation_type": "generative",
            "quantization_parameter": M_P,
            "spawn_threshold": {
                "amplitude": M_P,
                "phase": 4 * math.pi
            },
            "all_parameters_cgm_derived": True
        }

    def get_spawn_info(self) -> Dict[str, Any]:
        """Get information about spawn eligibility.
        
        Returns:
            Dictionary with spawn details
        """
        m_p_fx = self._to_fixed_point(M_P)
        four_pi_fx = self._to_fixed_point(4 * math.pi)
        
        return {
            "spawn_ready": self.spawn_ready,
            "amplitude": self._from_fixed_point(self.amplitude),
            "amplitude_threshold": M_P,
            "amplitude_ratio": self._from_fixed_point(self.amplitude) / M_P,
            "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
            "phase_threshold": 4 * math.pi,
            "phase_ratio": self._from_fixed_point(self.cumulative_phase) / (4 * math.pi),
            "spinor_cycle_count": self.spinor_cycle_count,
            "amplitude_ready": abs(self.amplitude) >= m_p_fx,
            "phase_ready": self.cumulative_phase >= four_pi_fx
        }

    def __repr__(self) -> str:
        """String representation of BU_En generation."""
        return (f"BUGeneration(tensor_id={self.tensor_id}, "
                f"cycle={self.cycle_index}, "
                f"amplitude={self._from_fixed_point(self.amplitude):.6f}, "
                f"gen_counter={self.generation_counter}, "
                f"spawn_ready={self.spawn_ready}, "
                f"helical_pos={self._from_fixed_point(self.cumulative_phase)/(4*math.pi):.4f}, "
                f"spinor_cycle={self.spinor_cycle_count})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        helical_pos = self._from_fixed_point(self.cumulative_phase) / (4 * math.pi)
        active_block = "ingress" if self.oscillation_phase == 0 else "egress"
        spawn_status = "spawn-ready" if self.spawn_ready else "not-spawn-ready"
        return (f"BU_En Generation τ{self.tensor_id} "
                f"(cycle {self.cycle_index}, {active_block} active, "
                f"{spawn_status}, helical {helical_pos:.4f}, "
                f"spinor {self.spinor_cycle_count})")
                                          