"""
G1 ONA Stage: Observation Non-Absolute

ONA creates anti-correlation through exact sign inversion of the UNA pattern.
The structure becomes 2×3×2 with 12 non-zeros, representing the first emergence
of observational structure through anti-correlated block extension.

This is where the tensor gains its first capacity for observation through
the anti-correlation mechanism that will later enable quantization at BU.
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
from typing import Dict, Any
from contextlib import contextmanager
from scipy.sparse import csr_matrix

from gyro_si.gyro_constants import GAMMA, M_P
from gyro_si.gyro_errors import StructuralViolation, QuantizationDefect
from gyro_si.gyro_gcr.gyro_config import config

# Import proper G1 infrastructure
from gyro_si.g1_gyroalignment.genetic_memory import GeneticMemory
from gyro_si.g1_gyroalignment.cs.g1_governance import stage_transition_lock

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"
TEMPLATE_PATH = os.path.join("patterns", "gyro_g1", SCHEMA_VERSION)

# ─────────────────────────── class definition ───────────────────────────

class ONAObservation:
    """Manages the ONA stage with anti-correlated 2×3×2 structure.

    Creates observational capacity through exact sign inversion of the UNA pattern.
    The anti-correlation mechanism enables the quantization that will emerge at BU_In.
    Six rotational degrees of freedom emerge through the extended structure.
    """

    SHAPE = (2, 3, 2)
    NONZEROS = 12  # 2×3×2 = 12 entries, all non-zero
    # ONA pattern is computed from UNA inversion, not stored as canonical pattern
    CANONICAL_PATTERN = None

    # ───────── constructor ─────────
    def __init__(self, state: Dict[str, Any]):
        """Initialize ONA from the state provided by the UNA stage.

        Args:
            state: State dictionary from UNA stage containing all necessary
                   initialization data including the UNA structure to invert.
        """

        # Per-tensor re-entrant lock
        self.lock = threading.RLock()

        # ══ Identity & Lineage ══
        self.tensor_id = state["tensor_id"]
        self.parent_id = state.get("parent_id")
        self.stage = "ONA"
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
        self.cs_memory = state["cs_memory"]  # π/2 chirality seed in fixed-point
        self.accumulated_threshold = state["accumulated_threshold"]  # β from UNA
        self.degrees_of_freedom = 6
        self.threshold_angle = self._to_fixed_point(GAMMA)  # GAMMA (π/4) in fixed-point
        self.schema_version = state.get("schema_version", SCHEMA_VERSION)
        
        # ══ Spinor cycle tracking ══
        self.spinor_cycle_count = state.get("spinor_cycle_count", 0)

        # ══ UNA structure for anti-correlation ══
        self.una_indptr = state["una_indptr"]
        self.una_indices = state["una_indices"]
        self.una_data = state["una_data"]

        # ══ Genetic Memory Interface ══
        self.genetic_memory = GeneticMemory()

        # ══ Build CSR structure through anti-correlation ══
        self._initialize_csr()

        # ══ Validation ══
        self.state_checksum = self._compute_checksum()
        self._validate_against_template()
        self._validate_structure()
        self._validate_fixed_point_range()

        self._record_to_trace("ona_initialized",
                              shape=self.SHAPE,
                              nonzeros=self.NONZEROS,
                              threshold_angle=self._from_fixed_point(self.threshold_angle),
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

    # ─────────────────────── CSR construction ───────────────────────────
    def _initialize_csr(self) -> None:
        """Build the ONA structure through anti-correlated block extension.

        Creates the 2×3×2 structure by:
        1. Reconstructing the UNA pattern from CSR
        2. Creating anti-correlated blocks through sign inversion
        3. Flattening 3D→2D for CSR representation
        """
        # Reconstruct UNA pattern from CSR
        una_values = []
        for encoded in self.una_data:
            if encoded == 0b01:
                una_values.append(1)
            elif encoded == 0b11:
                una_values.append(-1)
            else:
                raise StructuralViolation(f"Invalid UNA encoding: {encoded:02b}")

        una_csr = csr_matrix((una_values, self.una_indices, self.una_indptr), shape=(3, 2))
        una_dense = una_csr.toarray()

        # Create anti-correlated blocks through exact sign inversion
        # Block 1: Original UNA pattern [[-1,1], [-1,1], [-1,1]]
        # Block 2: Anti-correlated (sign-inverted) pattern [[1,-1], [1,-1], [1,-1]]
        block1 = una_dense
        block2 = -una_dense

        # Create 2×3×2 structure by stacking the blocks
        # Shape interpretation: (block, row, col)
        ona_3d = np.stack([block1, block2], axis=0)  # Shape: (2, 3, 2)

        # Flatten to 2D for CSR representation: (2*3, 2) = (6, 2)
        # This preserves the block structure while creating a valid CSR matrix
        ona_2d = ona_3d.reshape(6, 2)

        # Convert to CSR with SIMD alignment
        csr = csr_matrix(ona_2d)

        self.indptr = csr.indptr.tolist()
        self.indices = csr.indices.tolist()

        # Encode data as uint2: 01→+1, 11→-1 (no zeros in ONA)
        self.data = []
        for val in csr.data:
            if val == 1:
                self.data.append(0b01)
            elif val == -1:
                self.data.append(0b11)
            else:
                raise StructuralViolation(f"ONA cannot contain value {val}")

        # Store the 3D shape information for reconstruction
        self.block_structure = {
            "shape_3d": (2, 3, 2),
            "shape_2d": (6, 2),
            "block1_rows": [0, 1, 2],
            "block2_rows": [3, 4, 5]
        }

        # Mirror to GPU if available for SIMD alignment checks
        self.gpu_available = False
        self.gpu_indptr = None
        self.gpu_indices = None
        self.gpu_data = None
        
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
        for item in (
            self.tensor_id,
            self.stage,
            tuple(self.indptr),
            tuple(self.indices),
            tuple(self.data),
            self.amplitude,
            self.cumulative_phase,
            self.chirality_phase,
            tuple(self.block_structure["shape_3d"]),
            tuple(self.block_structure["shape_2d"]),
            tuple(self.block_structure["block1_rows"]),
            tuple(self.block_structure["block2_rows"]),
            self.spinor_cycle_count,
        ):
            h.update(str(item).encode())
        return h.hexdigest()


    def _validate_against_template(self) -> None:
        """Validate structure against canonical template or fallback to anti-correlation."""
        tpl_file = os.path.join(TEMPLATE_PATH, "ona_template.npy")
        if not os.path.exists(tpl_file):
            logger.warning("ONA template missing at %s, using anti-correlation only", tpl_file)
            self._validate_anti_correlation()
            return

        tpl = np.load(tpl_file)
        tpl_csr = csr_matrix(tpl)
        tpl_data = [0b01 if v == 1 else 0b11 for v in tpl_csr.data]

        if [self.indptr, self.indices, self.data] != [
            tpl_csr.indptr.tolist(),
            tpl_csr.indices.tolist(),
            tpl_data
        ]:
            raise StructuralViolation(f"{self.stage} deviates from canonical template")

    def _validate_anti_correlation(self) -> None:
        """Validate that the ONA structure exhibits proper anti-correlation.

        Verifies that Block 2 is exactly the sign inverse of Block 1,
        maintaining the fundamental anti-correlation property.

        Raises:
            StructuralViolation: If anti-correlation is violated
        """
        # Reconstruct the 2D matrix from CSR
        values = []
        for encoded in self.data:
            if encoded == 0b01:
                values.append(1)
            elif encoded == 0b11:
                values.append(-1)

        csr = csr_matrix((values, self.indices, self.indptr), shape=(6, 2))
        dense_2d = csr.toarray()

        # Extract blocks
        block1 = dense_2d[0:3, :]  # First 3 rows
        block2 = dense_2d[3:6, :]  # Last 3 rows

        # Verify anti-correlation: Block2 = -Block1
        if not np.array_equal(block2, -block1):
            raise StructuralViolation("ONA blocks must be anti-correlated (Block2 = -Block1)")

    def _validate_structure(self) -> None:
        """Run all structural invariants for the ONA stage.

        Validates:
        - Correct number of non-zeros (12)
        - Proper CSR dimensions
        - Anti-correlation property
        - CPU/GPU consistency if available

        Raises:
            StructuralViolation: If any structural constraint is violated
            QuantizationDefect: If CPU/GPU copies don't match
        """
        # Check non-zero count
        if len(self.data) != self.NONZEROS:
            raise StructuralViolation(f"{self.stage} expects {self.NONZEROS} non-zeros, found {len(self.data)}")

        # Check CSR dimensions (flattened 2D representation)
        expected_2d_shape = (6, 2)
        if len(self.indptr) != expected_2d_shape[0] + 1:
            raise StructuralViolation(f"Invalid indptr length for flattened shape {expected_2d_shape}")

        # Verify all values are in {-1, 1} - no zeros allowed
        for encoded in self.data:
            if encoded not in [0b01, 0b11]:
                raise StructuralViolation(f"ONA values must be exactly {{-1, 1}}, got encoding {encoded:02b}")

        # Verify anti-correlation property
        self._validate_anti_correlation()

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
                    raise QuantizationDefect("CPU and GPU tensor copies do not match in ONA stage")
            except Exception as e:
                raise QuantizationDefect(f"GPU validation failed in ONA stage: {e}")

    def _validate_fixed_point_range(self) -> None:
        """Validate that all fixed-point values are within the Q29.34 representable range."""
        max_fx = (1 << 63) - 1
        min_fx = - (1 << 63)

        checks = [
            ("amplitude", self.amplitude),
            ("cumulative_phase", self.cumulative_phase),
            ("chirality_phase", self.chirality_phase),
            ("last_epsilon", self.last_epsilon),
            ("cs_memory", self.cs_memory),
            ("accumulated_threshold", self.accumulated_threshold),
            ("threshold_angle", self.threshold_angle),
        ]

        for name, value in checks:
            if not (min_fx <= value <= max_fx):
                raise QuantizationDefect(
                    f"Fixed-point {name}={value} outside representable range [{min_fx}, {max_fx}]"
                )


    # ─────────────────────── tensor context helper ──────────────────────
    def _create_tensor_context(self) -> Dict[str, Any]:
        """Create tensor context for G6 message format compliance.
        
        Returns:
            Dictionary with required tensor context fields
        """
        return {
            "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
            "chirality_phase": self._from_fixed_point(self.chirality_phase),
            "helical_position": self._from_fixed_point(self.cumulative_phase) / (4 * math.pi),
            "spinor_cycle_count": self.spinor_cycle_count
        }

    # ───────────────────── phase / processing hooks ─────────────────────
    def process_phase(self, phi: float) -> float:
        """Process input phase, updating trackers using fixed-point arithmetic.

        The anti-correlated structure enables enhanced quantization sensitivity
        that will be fully realized at the BU_In stage.

        Args:
            phi: Input phase value to process

        Returns:
            Quantization error epsilon as float
        """
        with self.lock, stage_transition_lock:

            # Convert to fixed-point
            phi_fx = self._to_fixed_point(phi)

            # Quantize using fixed-point arithmetic with enhanced sensitivity
            phi_q_fx = self._quantize_fixed(phi_fx)
            eps_fx = phi_fx - phi_q_fx

            # Update amplitude with clipping
            m_p_fx = self._to_fixed_point(M_P)
            neg_m_p_fx = self._to_fixed_point(-M_P)
            self.amplitude = max(neg_m_p_fx, min(self.amplitude + phi_q_fx, m_p_fx))

            # Update phase tracking - use math.pi consistently
            abs_phi_q_fx = abs(phi_q_fx)
            four_pi_fx = self._to_fixed_point(4 * math.pi)
            two_pi_fx = self._to_fixed_point(2 * math.pi)

            # Store previous phase for boundary detection
            prev_cumulative_phase = self.cumulative_phase
            
            # Update phases
            self.cumulative_phase = (self.cumulative_phase + abs_phi_q_fx) % four_pi_fx
            self.chirality_phase = self.cumulative_phase % two_pi_fx
            self.last_epsilon = eps_fx

            # Check for 4π boundary crossing (spinor cycle completion)
            if prev_cumulative_phase > self.cumulative_phase:
                self.spinor_cycle_count += 1
                
            # Check for exact 2π boundary and collapse segment if necessary
            if self.cumulative_phase % two_pi_fx == 0:
                self._collapse_segment_to_digest()

            # Generate algedonic signal if |ε| > mₚ/2
            m_p_half_fx = m_p_fx // 2
            if abs(eps_fx) > m_p_half_fx:
                # Schedule async algedonic signal
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._generate_algedonic_signal_async(eps_fx))
                except RuntimeError:
                    # No event loop running, log warning
                    logger.warning("Cannot send algedonic signal: no event loop")

            # Update checksum after state change
            self.state_checksum = self._compute_checksum()

            # Record processing event
            self._record_to_trace("phase_processed",
                                  phi=phi,
                                  phi_q=self._from_fixed_point(phi_q_fx),
                                  epsilon=self._from_fixed_point(eps_fx),
                                  amplitude=self._from_fixed_point(self.amplitude),
                                  cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                                  anti_correlation_active=True,
                                  checksum=self.state_checksum)

            return self._from_fixed_point(eps_fx)

    def _quantize_fixed(self, phi_fx: int) -> int:
        """Quantize fixed-point phase with anti-correlation sensitivity.

        The anti-correlated structure provides enhanced quantization sensitivity
        compared to UNA, preparing for the full quantization emergence at BU_In.

        Args:
            phi_fx: Phase value in Q29.34 fixed-point format

        Returns:
            Quantized phase in fixed-point format
        """
        m_p_fx = self._to_fixed_point(M_P)
        half_m_p_fx = m_p_fx // 2
        neg_m_p_fx = self._to_fixed_point(-M_P)
        neg_half_m_p_fx = neg_m_p_fx // 2

        # Enhanced sensitivity due to anti-correlation
        # Use GAMMA as sensitivity factor
        gamma_fx = self.threshold_angle  # GAMMA in fixed-point
        sensitivity_factor = gamma_fx // 4  # GAMMA/4 for enhanced sensitivity

        # Apply sensitivity adjustment
        adjusted_half_m_p_fx = half_m_p_fx - sensitivity_factor
        adjusted_neg_half_m_p_fx = neg_half_m_p_fx + sensitivity_factor

        if phi_fx < adjusted_neg_half_m_p_fx:
            return neg_m_p_fx
        elif phi_fx >= adjusted_half_m_p_fx:
            return m_p_fx
        else:
            return 0

    async def _generate_algedonic_signal_async(self, eps_fx: int) -> None:
        """Generate and send algedonic signal when quantization error exceeds threshold."""
        eps_float = self._from_fixed_point(eps_fx)
        signal_type = "pain" if eps_float > 0 else "pleasure"

        from gyro_si.gyro_comm import send_message, MessageTypes

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
                "cumulative_phase": self._from_fixed_point(self.cumulative_phase)
            },
            "timestamp": time.time()
        }

        self._record_to_trace("algedonic_signal_generated",
                            signal_type=signal_type,
                            epsilon=eps_float)
        try:
            await send_message(signal_message)
        except Exception as e:
            logger.error(f"Failed to send algedonic signal: {e}")


    def _collapse_segment_to_digest(self) -> None:
        """Record a digest of the tensor state at exact 2π boundaries.
        
        This implements the pruning assertion from the specification:
        every tensor snapshot at cumulative_phase % 2π == 0 must carry
        a pruned_digest flag for G5 audit validation.
        """
        digest = {
            "tensor_id": self.tensor_id,
            "cycle_index": self.cycle_index,
            "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
            "amplitude": self._from_fixed_point(self.amplitude),
            "anti_correlation_blocks": 2,
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
            "threshold_angle": self.threshold_angle,
            "schema_version": self.schema_version,
            "state_checksum": self.state_checksum,
            "spinor_cycle_count": self.spinor_cycle_count,
            # CSR arrays (deep copy for lists)
            "indptr": self.indptr.copy() if self.indptr else None,
            "indices": self.indices.copy() if self.indices else None,
            "data": self.data.copy() if self.data else None,
            # UNA structure
            "una_indptr": self.una_indptr.copy() if self.una_indptr else None,
            "una_indices": self.una_indices.copy() if self.una_indices else None,
            "una_data": self.una_data.copy() if self.una_data else None,
            # Block structure
            "block_structure": {k: v.copy() for k, v in self.block_structure.items()},
            # GPU state
            "gpu_available": self.gpu_available
        }
        
        self._record_to_trace("tensor_transaction_start", cycle_index=self.cycle_index)
        try:
            yield
            self._record_to_trace("tensor_transaction_end", cycle_index=self.cycle_index)
        except Exception:
            for key, value in snapshot.items():
                setattr(self, key, value)
            self._record_to_trace("tensor_transaction_abort", cycle_index=self.cycle_index)
            raise

    def prepare_transition(self) -> Dict[str, Any]:
        """Prepare state for transition to BU_In stage.
        
        BU_In will perform the anomalous double integration where quantization
        error (ε) emerges as the fundamental observation mechanism.
        This method validates the current state and packages all necessary
        information for the BU_In stage constructor.
        
        Returns:
            Dictionary containing all state needed for BU_In initialization
            
        Raises:
            StructuralViolation: If ONA state is invalid for transition
        """
        with self.lock, stage_transition_lock:
            # Validate structure before transition
            self._validate_structure()
            self._validate_fixed_point_range()
            
            # Update cycle index for transition
            self.cycle_index += 1
            
            # Update accumulated threshold (β from UNA + γ from ONA)
            self.accumulated_threshold = self.accumulated_threshold + self.threshold_angle
            
            # Update checksum after state changes
            self.state_checksum = self._compute_checksum()

            # Package state for BU_In stage
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
                
                # Memory carried forward - CGM-derived threshold accumulation
                "cs_memory": self.cs_memory,  # α preserved
                "accumulated_threshold": self.accumulated_threshold,  # β + γ
                "spinor_cycle_count": self.spinor_cycle_count,
                
                # ONA structure for BU_In to use
                "ona_indptr": self.indptr.copy(),
                "ona_indices": self.indices.copy(),
                "ona_data": self.data.copy(),
                "ona_block_structure": self.block_structure.copy(),
                
                # UNA structure preserved for lineage
                "una_indptr": self.una_indptr.copy(),
                "una_indices": self.una_indices.copy(),
                "una_data": self.una_data.copy(),
                
                # Anti-correlation metadata
                "anti_correlation_active": True,
                "degrees_of_freedom": self.degrees_of_freedom,
                
                # Schema versioning
                "schema_version": self.schema_version,
            }

            # Record transition preparation
            self._record_to_trace("ona_transition_prepared",
                                  target_stage="BU_In",
                                  cycle_index=payload["cycle_index"],
                                  accumulated_threshold=self._from_fixed_point(self.accumulated_threshold),
                                  anti_correlation_active=True,
                                  degrees_of_freedom=self.degrees_of_freedom,
                                  checksum=self.state_checksum)
            
            return payload

    # ───────────────────────── trace helper ─────────────────────────────
    def _record_to_trace(self, event_type: str, **kw):
        """Record events via the genetic memory system.
        
        Delegates to the proper G1 infrastructure for trace recording,
        which will handle buffer management, retention policy, and
        coordination with G5 for audit collection.
        
        Args:
            event_type: Type of event being recorded
            **kw: Additional event data
        """
        evt = {
            "timestamp": time.time(),
            "source": "G1",
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
        logger.debug("ONA Event: %s", evt)
        
        # Delegate to genetic memory for proper trace recording
        self.genetic_memory.record_event(evt)

    # ───────────────────────── utility methods ─────────────────────────────
    def get_anti_correlation_info(self) -> Dict[str, Any]:
        """Get anti-correlation structure information.
        
        Returns:
            Dictionary with anti-correlation details
        """
        return {
            "block_structure": self.block_structure,
            "anti_correlation_active": True,
            "block_count": 2,
            "degrees_of_freedom": self.degrees_of_freedom
        }

    def get_threshold_info(self) -> Dict[str, float]:
        """Get threshold information in human-readable format.
        
        Returns:
            Dictionary with threshold values converted to floats
        """
        return {
            "threshold_angle": self._from_fixed_point(self.threshold_angle),
            "accumulated_threshold": self._from_fixed_point(self.accumulated_threshold),
            "sensitivity_factor": self._from_fixed_point(self.threshold_angle) / 4,
            "cgm_gamma_value": self._from_fixed_point(self.threshold_angle)
        }

    def get_phase_info(self) -> Dict[str, float]:
        """Get current phase information in human-readable format.
        
        Returns:
            Dictionary with phase values converted to floats
        """
        return {
            "amplitude": self._from_fixed_point(self.amplitude),
            "cumulative_phase": self._from_fixed_point(self.cumulative_phase),
            "chirality_phase": self._from_fixed_point(self.chirality_phase),
            "last_epsilon": self._from_fixed_point(self.last_epsilon),
            "helical_position": self._from_fixed_point(self.cumulative_phase) / (4 * math.pi),
            "spinor_cycle_count": self.spinor_cycle_count
        }

    def get_structure_info(self) -> Dict[str, Any]:
        """Get structural information about the tensor.
        
        Returns:
            Dictionary with tensor structure details
        """
        return {
            "shape": self.SHAPE,
            "shape_2d": (6, 2),  # Flattened representation
            "nonzeros": self.NONZEROS,
            "degrees_of_freedom": self.degrees_of_freedom,
            "gpu_available": self.gpu_available,
            "anti_correlation_blocks": 2,
            "checksum": self.state_checksum
        }

    def is_spawn_eligible(self) -> bool:
        """Check if tensor is eligible for spawning.
        
        ONA stage is not directly spawn-eligible. Spawning occurs at BU_En
        stage after completing the full cycle and reaching 4π.
        
        Returns:
            Always False for ONA stage
        """
        return False

    def reconstruct_3d_structure(self) -> np.ndarray:
        """Reconstruct the full 3D anti-correlated structure.
        
        Returns:
            3D numpy array with shape (2, 3, 2) showing the anti-correlated blocks
        """
        # Reconstruct 2D from CSR
        values = []
        for encoded in self.data:
            if encoded == 0b01:
                values.append(1)
            elif encoded == 0b11:
                values.append(-1)

        csr = csr_matrix((values, self.indices, self.indptr), shape=(6, 2))
        dense_2d = csr.toarray()
        
        # Reshape to 3D structure
        return dense_2d.reshape(2, 3, 2)

    def get_cgm_compliance_info(self) -> Dict[str, Any]:
        """Get CGM compliance and derivation information.
        
        Returns:
            Dictionary with CGM-derived values and their sources
        """
        return {
            "cs_memory_source": "ALPHA (π/2)",
            "cs_memory_value": self._from_fixed_point(self.cs_memory),
            "threshold_angle_source": "GAMMA (π/4)",
            "threshold_angle_value": self._from_fixed_point(self.threshold_angle),
            "accumulated_threshold_derivation": "β + γ (UNA + ONA)",
            "accumulated_threshold_value": self._from_fixed_point(self.accumulated_threshold),
            "quantization_parameter": M_P,
            "sensitivity_derivation": "GAMMA/4",
            "all_parameters_cgm_derived": True
        }

    def __repr__(self) -> str:
        """String representation of ONA observation."""
        return (f"ONAObservation(tensor_id={self.tensor_id}, "
                f"cycle={self.cycle_index}, "
                f"amplitude={self._from_fixed_point(self.amplitude):.6f}, "
                f"anti_corr=True, "
                f"helical_pos={self._from_fixed_point(self.cumulative_phase)/(4*math.pi):.4f}, "
                f"spinor_cycle={self.spinor_cycle_count})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        helical_pos = self._from_fixed_point(self.cumulative_phase) / (4 * math.pi)
        return (f"ONA Observation τ{self.tensor_id} "
                f"(cycle {self.cycle_index}, anti-correlated, "
                f"helical {helical_pos:.4f}, spinor {self.spinor_cycle_count})")