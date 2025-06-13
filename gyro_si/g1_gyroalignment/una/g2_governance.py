"""
G1 UNA Stage: Unity Non-Absolute

UNA creates the first observable structure through the left inverse operation.
The exact pattern is [[-1, 1], [-1, 1], [-1, 1]] — no variation allowed.
Three rotational degrees of freedom emerge through position alone.
"""

# ───────────────────── imports & shared infrastructure ──────────────────
import numpy as np
import logging
import time
import hashlib
import threading
import os
from typing import Dict, Any
from contextlib import contextmanager
from scipy.sparse import csr_matrix

from gyro_si.gyro_constants import BETA, M_P
from gyro_si.gyro_errors import StructuralViolation, QuantizationDefect
from gyro_si.gyro_gcr.gyro_config import config

# Import proper G1 infrastructure
from gyro_si.g1_gyroalignment.genetic_memory import GeneticMemory
from gyro_si.g1_gyroalignment.cs.g1_governance import stage_transition_lock

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"
TEMPLATE_PATH = os.path.join("patterns", "gyro_g1", SCHEMA_VERSION)

# ─────────────────────────── class definition ───────────────────────────

class UNANormalization:
    """Manages the UNA stage with a fixed 3×2 structure.

    Three rotational degrees of freedom emerge through position alone. All values
    are a uniform [-1, 1] representing maximal left bias. The structure is
    immutable and validated against a canonical template.
    """

    SHAPE = (3, 2)
    NONZEROS = 6
    CANONICAL_PATTERN = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.int8)

    # ───────── constructor ─────────
    def __init__(self, state: Dict[str, Any]):
        """Initialize UNA from the state provided by the CS stage.

        Args:
            state: State dictionary from CS stage containing all necessary
                   initialization data including tensor identity, phase tracking,
                   and the CS memory (π/2 chirality seed).
        """

        # Per-tensor re-entrant lock
        self.lock = threading.RLock()

        # ══ Identity & Lineage ══
        self.tensor_id = state["tensor_id"]
        self.parent_id = state.get("parent_id")
        self.stage = "UNA"
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
        self.degrees_of_freedom = 3
        self.threshold_angle = self._to_fixed_point(BETA)  # π/4 in fixed-point
        self.threshold_ratio = self._to_fixed_point(1.0 / np.sqrt(2))  # cos(π/4)
        self.schema_version = state.get("schema_version", SCHEMA_VERSION)

        # ══ Genetic Memory Interface ══
        self.genetic_memory = GeneticMemory()

        # ══ Build CSR structure ══
        self._initialize_csr()

        # ══ Validation ══
        self.state_checksum = self._compute_checksum()
        self._validate_against_template()
        self._validate_structure()

        self._record_to_trace("una_initialized",
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
        """Build the stage-specific CSR representation from the canonical pattern.

        Creates the exact UNA pattern [[-1,1], [-1,1], [-1,1]] with proper
        CSR encoding and optional GPU mirroring for validation.
        """
        # Use the exact canonical pattern - no variations allowed
        dense = np.array(self.CANONICAL_PATTERN, dtype=np.int8)

        # Verify it matches the expected pattern exactly
        expected = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.int8)
        if not np.array_equal(dense, expected):
            raise StructuralViolation("UNA pattern deviation from canonical form")

        # Convert to CSR with SIMD alignment
        csr = csr_matrix(dense)

        self.indptr = csr.indptr.tolist()
        self.indices = csr.indices.tolist()

        # Encode data as uint2: 01→+1, 11→-1 (no zeros allowed in UNA)
        self.data = []
        for val in csr.data:
            if val == 1:
                self.data.append(0b01)
            elif val == -1:
                self.data.append(0b11)
            else:
                raise StructuralViolation(f"UNA cannot contain value {val}")

        # Mirror to GPU if available for SIMD alignment checks
        try:
            import cupy as cp
            self.gpu_indptr = cp.array(self.indptr)
            self.gpu_indices = cp.array(self.indices)
            self.gpu_data = cp.array(self.data)
            self.gpu_available = True
        except (ImportError, ModuleNotFoundError):
            self.gpu_available = False

    # ─────────────────────── validation & checksum ──────────────────────
    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum over complete tensor state.

        Returns:
            Hexadecimal SHA-256 checksum string
        """
        h = hashlib.sha256()
        for item in (
            self.tensor_id, self.stage, self.indptr, self.indices, self.data,
            self.amplitude, self.cumulative_phase, self.chirality_phase
        ):
            h.update(str(item).encode())
        return h.hexdigest()

    def _validate_against_template(self) -> None:
        """Validate structure against canonical template for current schema version.

        Performs byte-for-byte comparison against the versioned template file.

        Raises:
            StructuralViolation: If structure deviates from canonical template
        """
        tpl_file = os.path.join(TEMPLATE_PATH, "una_template.npy")
        if not os.path.exists(tpl_file):
            logger.warning("Template %s missing; skipping validation.", tpl_file)
            return

        tpl = np.load(tpl_file)
        tpl_csr = csr_matrix(tpl)
        tpl_data = [0b01 if v == 1 else 0b11 for v in tpl_csr.data]

        if [self.indptr, self.indices, self.data] != [tpl_csr.indptr.tolist(), tpl_csr.indices.tolist(), tpl_data]:
            raise StructuralViolation(f"{self.stage} deviates from canonical template")

    def _validate_structure(self) -> None:
        """Run all structural invariants for the UNA stage.

        Validates:
        - Correct number of non-zeros
        - Proper CSR dimensions
        - Exact canonical pattern
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
                raise StructuralViolation(f"UNA values must be exactly {{-1, 1}}, got encoding {encoded:02b}")

        # Verify the exact canonical pattern
        self._verify_exact_pattern()

        # If GPU is available, verify CPU and GPU copies match
        if self.gpu_available:
            try:
                import cupy as cp
                if (not np.array_equal(self.indptr, cp.asnumpy(self.gpu_indptr)) or
                    not np.array_equal(self.indices, cp.asnumpy(self.gpu_indices)) or
                    not np.array_equal(self.data, cp.asnumpy(self.gpu_data))):
                    raise QuantizationDefect("CPU and GPU tensor copies do not match in UNA stage")
            except Exception as e:
                raise QuantizationDefect(f"GPU validation failed in UNA stage: {e}")

    def _verify_exact_pattern(self) -> None:
        """Verify tensor has EXACTLY the canonical UNA pattern.

        Reconstructs the dense array and compares against the expected
        [[-1,1], [-1,1], [-1,1]] pattern with no tolerance for variation.

        Raises:
            StructuralViolation: If pattern doesn't match exactly
        """
        # Reconstruct dense array from CSR
        values = []
        for encoded in self.data:
            if encoded == 0b01:
                values.append(1)
            elif encoded == 0b11:
                values.append(-1)

        csr = csr_matrix((values, self.indices, self.indptr), shape=self.SHAPE)
        dense = csr.toarray()

        # Check EXACT pattern - no variations allowed
        expected = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.int8)
        if not np.array_equal(dense, expected):
            raise StructuralViolation("UNA pattern must be EXACTLY [[-1,1], [-1,1], [-1,1]]")

    # ───────────────────── phase / processing hooks ─────────────────────
    def process_phase(self, phi: float) -> float:
        """Process input phase, updating trackers using fixed-point arithmetic.

        Note: The tensor structure NEVER changes. Only phase tracking occurs.

        Args:
            phi: Input phase value to process

        Returns:
            Quantization error epsilon as float
        """
        with self.lock:
            # Convert to fixed-point
            phi_fx = self._to_fixed_point(phi)

            # Quantize using fixed-point arithmetic
            phi_q_fx = self._quantize_fixed(phi_fx)
            eps_fx = phi_fx - phi_q_fx

            # Update amplitude with clipping
            m_p_fx = self._to_fixed_point(M_P)
            neg_m_p_fx = self._to_fixed_point(-M_P)
            self.amplitude = max(neg_m_p_fx, min(self.amplitude + phi_q_fx, m_p_fx))

            # Update phase tracking
            abs_phi_q_fx = abs(phi_q_fx)
            four_pi_fx = self._to_fixed_point(4 * np.pi)
            two_pi_fx = self._to_fixed_point(2 * np.pi)

            self.cumulative_phase = (self.cumulative_phase + abs_phi_q_fx) % four_pi_fx
            self.chirality_phase = self.cumulative_phase % two_pi_fx
            self.last_epsilon = eps_fx

            # Check for exact 2π boundary and collapse segment if necessary
            if self.cumulative_phase % two_pi_fx == 0:
                self._collapse_segment_to_digest()

            # Update checksum after state change
            self.state_checksum = self._compute_checksum()

            # Record processing event
            self._record_to_trace("phase_processed",
                                  phi=phi,
                                  phi_q=self._from_fixed_point(phi_q_fx),
                                  epsilon=self._from_fixed_point(eps_fx),
                                  amplitude=self._from_fixed_point(self.amplitude),
                                  cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                                  checksum=self.state_checksum)

            return self._from_fixed_point(eps_fx)

    def _quantize_fixed(self, phi_fx: int) -> int:
        """Quantize fixed-point phase to discrete values.

        Args:
            phi_fx: Phase value in Q29.34 fixed-point format

        Returns:
            Quantized phase in fixed-point format
        """
        m_p_fx = self._to_fixed_point(M_P)
        half_m_p_fx = m_p_fx // 2
        neg_m_p_fx = self._to_fixed_point(-M_P)
        neg_half_m_p_fx = neg_m_p_fx // 2

        if phi_fx < neg_half_m_p_fx:
            return neg_m_p_fx
        elif phi_fx >= half_m_p_fx:
            return m_p_fx
        else:
            return 0

    @staticmethod
    def quantize(phi, m_p):
        """Vectorized quantization for batch processing.

        Can be used with numpy or cupy arrays for bulk operations.

        Args:
            phi: Phase array (numpy or cupy)
            m_p: Quantization parameter

        Returns:
            Quantized phase array
        """
        xp = np
        try:
            import cupy
            if isinstance(phi, cupy.ndarray):
                xp = cupy
        except ImportError:
            pass
        return xp.where(phi < -m_p/2, -m_p, xp.where(phi >= m_p/2, m_p, 0))

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
            "pruned_digest": True  # Required for G5 audit validation
        }
        
        self._record_to_trace("segment_collapsed", 
                              cumulative_phase=self._from_fixed_point(self.cumulative_phase),
                              digest=digest)

    # ─────────────────── transition management ─────────────────────
    @contextmanager
    def tensor_transaction(self):
        """Context manager for transactional tensor operations.
        
        Provides rollback capability for failed operations by creating
        a complete snapshot of the tensor state before the operation
        and restoring it if an exception occurs.
        
        Yields:
            None - the context for the transaction
            
        Raises:
            Any exception from the wrapped operation, after rollback
        """
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
            "threshold_angle": self.threshold_angle,
            "threshold_ratio": self.threshold_ratio,
            "schema_version": self.schema_version,
            "state_checksum": self.state_checksum,
            # CSR arrays (deep copy for lists)
            "indptr": self.indptr.copy() if self.indptr else None,
            "indices": self.indices.copy() if self.indices else None,
            "data": self.data.copy() if self.data else None
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
        """Prepare state for transition to ONA stage.
        
        ONA will create anti-correlation through exact sign inversion of the UNA pattern.
        This method validates the current state and packages all necessary information
        for the ONA stage constructor.
        
        Returns:
            Dictionary containing all state needed for ONA initialization
            
        Raises:
            StructuralViolation: If UNA state is invalid for transition
        """
        with self.lock, stage_transition_lock:
            # Validate structure before transition
            self._validate_structure()
            
            # Update cycle index for transition
            self.cycle_index += 1
            
            # Update checksum after cycle increment
            self.state_checksum = self._compute_checksum()

            # Package state for ONA stage
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
                
                # CS memory carried forward
                "cs_memory": self.cs_memory,
                
                # UNA structure for ONA to use
                "una_indptr": self.indptr.copy(),
                "una_indices": self.indices.copy(),
                "una_data": self.data.copy(),
                
                # Accumulated thresholds (β from UNA)
                "accumulated_threshold": self.threshold_angle,
                
                # Schema versioning
                "schema_version": self.schema_version,
            }

            # Record transition preparation
            self._record_to_trace("una_transition_prepared",
                                  target_stage="ONA",
                                  cycle_index=payload["cycle_index"],
                                  accumulated_threshold=self._from_fixed_point(self.threshold_angle),
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
            "source": "G1_UNA",
            "event_type": event_type,
            "tensor_id": self.tensor_id,
            "cycle_index": self.cycle_index,
            "stage": self.stage,
            **kw
        }
        
        # Log to console/file via standard logging
        logger.debug("UNA Event: %s", evt)
        
        # Delegate to genetic memory for proper trace recording
        self.genetic_memory.record_event(evt)

    # ───────────────────────── utility methods ─────────────────────────────
    def get_threshold_info(self) -> Dict[str, float]:
        """Get threshold information in human-readable format.
        
        Returns:
            Dictionary with threshold values converted to floats
        """
        return {
            "threshold_angle": self._from_fixed_point(self.threshold_angle),
            "threshold_ratio": self._from_fixed_point(self.threshold_ratio),
            "accumulated_threshold": self._from_fixed_point(self.threshold_angle)
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
            "last_epsilon": self._from_fixed_point(self.last_epsilon)
        }

    def get_structure_info(self) -> Dict[str, Any]:
        """Get structural information about the tensor.
        
        Returns:
            Dictionary with tensor structure details
        """
        return {
            "shape": self.SHAPE,
            "nonzeros": self.NONZEROS,
            "degrees_of_freedom": self.degrees_of_freedom,
            "gpu_available": self.gpu_available,
            "checksum": self.state_checksum
        }

    def is_spawn_eligible(self) -> bool:
        """Check if tensor is eligible for spawning.
        
        UNA stage is not directly spawn-eligible. Spawning occurs at BU_En
        stage after completing the full cycle and reaching 4π.
        
        Returns:
            Always False for UNA stage
        """
        return False

    def __repr__(self) -> str:
        """String representation of UNA normalization."""
        return (f"UNANormalization(tensor_id={self.tensor_id}, "
                f"cycle={self.cycle_index}, "
                f"amplitude={self._from_fixed_point(self.amplitude):.6f})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"UNA Normalization τ{self.tensor_id} (cycle {self.cycle_index})"