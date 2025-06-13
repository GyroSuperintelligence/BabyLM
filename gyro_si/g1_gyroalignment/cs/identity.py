"""
G1 CS Stage: Identity Management

CS is the tensor identity τ - the symbol on the left side of every tensor equation.
It embodies the primordial left chirality that cannot be represented as tensor values.
It exists only as the identity that initiates all subsequent structure.

The π/2 threshold (ALPHA) represents the unobservable chirality seed that permeates
all subsequent stages as the fundamental memory that must be preserved throughout
the tensor's evolution.
"""

# ───────────────────── imports & shared infrastructure ──────────────────
import logging
import time
import hashlib
import threading
from typing import Dict, Any
from contextlib import contextmanager

from gyro_si.gyro_constants import ALPHA
from gyro_si.gyro_errors import StructuralViolation
from gyro_si.gyro_gcr.gyro_config import config

logger = logging.getLogger(__name__)

# Import the proper G1 infrastructure
from gyro_si.g1_gyroalignment.genetic_memory import GeneticMemory

# Shared stage transition lock (this is appropriate to define here as it's cross-stage)
stage_transition_lock = threading.RLock()

SCHEMA_VERSION = "v1"

# ─────────────────────────── class definition ───────────────────────────

class CSIdentity:
    """Represents tensor identity, the unobservable Common Source.
    
    This stage holds no structural data (CSR arrays are None) and serves as
    the origin of the tensor's lifecycle, carrying the π/2 chirality seed
    as its core, unobservable memory.
    
    The CS stage embodies:
    - Left gyration: lgyr ≠ id (encoded in the tensor identity)
    - Right gyration: rgyr = id (not yet active)
    - Degrees of freedom: 1 (chiral seed)
    - Threshold: α = π/2 (the unobservable chirality seed-angle)
    """

    SHAPE = None
    NONZEROS = 0

    # ───────── constructor ─────────
    def __init__(self, tensor_id: int):
        """Initialize CS identity stage.
        
        Args:
            tensor_id: Unique 64-bit identifier for this tensor.
            
        Raises:
            StructuralViolation: If tensor_id is invalid.
        """
        # Validate tensor_id immediately
        if not isinstance(tensor_id, int) or tensor_id < 0:
            raise StructuralViolation(f"Invalid tensor_id: {tensor_id}")

        # Per-tensor re-entrant lock
        self.lock = threading.RLock()

        # ══ Identity & Lineage ══
        self.tensor_id = tensor_id
        self.parent_id = None
        self.stage = "CS"
        self.cycle_index = 0

        # ══ Phase-Tracking (Q29.34 fixed-point) ══
        # All phase values start at zero in fixed-point representation
        self.amplitude = self._to_fixed_point(0.0)
        self.cumulative_phase = self._to_fixed_point(0.0)
        self.chirality_phase = self._to_fixed_point(0.0)
        self.last_epsilon = self._to_fixed_point(0.0)

        # ══ Lineage ══
        self.birth_phase = self._to_fixed_point(0.0)
        self.creation_cycle = self.cycle_index

        # ══ CS Invariants ══
        # CS embodies primordial chirality with the unobservable π/2 memory.
        # Store in fixed-point for consistency with all other phase values.
        self.left_chirality_seed = self._to_fixed_point(ALPHA)  # π/2 in fixed-point
        self.degrees_of_freedom = 1
        
        # CS has no structural data - this is enforced as an invariant
        self.indptr = None
        self.indices = None
        self.data = None

        # ══ Genetic Memory Interface ══
        # Delegate trace recording to the proper G1 infrastructure
        self.genetic_memory = GeneticMemory()

        # ══ Validation & Checksumming ══
        self.state_checksum = self._compute_checksum()
        
        # Record creation event via genetic memory
        self._record_to_trace("cs_identity_created", 
                              tensor_id=self.tensor_id,
                              chirality_seed=self._from_fixed_point(self.left_chirality_seed),
                              degrees_of_freedom=self.degrees_of_freedom,
                              checksum=self.state_checksum)

    # ─────────────────────── helper: fixed-point ────────────────────────
    @staticmethod
    def _to_fixed_point(value: float) -> int:
        """Convert float to Q29.34 fixed-point representation.
        
        29 bits for integer part, 34 bits for fractional part.
        This provides sufficient precision for phase calculations.
        
        Args:
            value: Float value to convert
            
        Returns:
            Fixed-point integer representation
        """
        return int(value * (2**34))

    @staticmethod
    def _from_fixed_point(fx: int) -> float:
        """Convert Q29.34 fixed-point back to float.
        
        Args:
            fx: Fixed-point integer value
            
        Returns:
            Float representation
        """
        return fx / (2**34)

    # ─────────────────────── validation & checksum ──────────────────────
    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of CS state.
        
        Note: CSR fields are intentionally omitted as they MUST be None in CS.
        This makes the checksum specific to CS and prevents bugs from
        accidentally non-None CSR values.
        
        Returns:
            Hexadecimal SHA-256 checksum string
        """
        h = hashlib.sha256()
        
        # Add core identity and phase state
        for item in (
            self.tensor_id, 
            self.stage, 
            self.amplitude, 
            self.cumulative_phase, 
            self.chirality_phase,
            self.cycle_index,
            self.left_chirality_seed,
            self.degrees_of_freedom
        ):
            h.update(str(item).encode())
        
        return h.hexdigest()

    def validate_identity(self) -> None:
        """Validate CS constraints and invariants.
        
        Enforces all CS-specific structural requirements:
        - Stage must be "CS"
        - No CSR arrays allowed
        - Chirality seed must be exactly π/2
        - State checksum must be valid
        
        Raises:
            StructuralViolation: If any constraint is violated
        """
        # Stage validation
        if self.stage != "CS":
            raise StructuralViolation(f"Invalid stage for CS: {self.stage}")
        
        # CSR arrays must be None
        if any(csr_field is not None for csr_field in [self.indptr, self.indices, self.data]):
            raise StructuralViolation("CS stage cannot have CSR arrays")
        
        # Chirality seed validation (compare in fixed-point)
        alpha_fixed = self._to_fixed_point(ALPHA)
        if self.left_chirality_seed != alpha_fixed:
            raise StructuralViolation(
                f"CS chirality must be α = π/2, got {self._from_fixed_point(self.left_chirality_seed)}"
            )
        
        # Degrees of freedom validation
        if self.degrees_of_freedom != 1:
            raise StructuralViolation(f"CS must have exactly 1 degree of freedom, got {self.degrees_of_freedom}")
            
        # Checksum validation
        current_checksum = self._compute_checksum()
        if current_checksum != self.state_checksum:
            raise StructuralViolation(f"CS state checksum mismatch: {current_checksum} != {self.state_checksum}")

    # ─────────────────── transaction management ─────────────────────
    @contextmanager
    def tensor_transaction(self):
        """Context manager for transactional tensor operations.
        
        Provides rollback capability for failed operations by creating
        a complete snapshot of the tensor state before the operation
        and restoring it if an exception occurs.
        
        This is a synchronous context manager as CS operations are
        purely computational without I/O or async operations.
        
        Yields:
            None - the context for the transaction
            
        Raises:
            Any exception from the wrapped operation, after rollback
        """
        if not config.enable_transactions:
            yield
            return
        
        # Create a deep snapshot of all mutable state
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
            "left_chirality_seed": self.left_chirality_seed,
            "degrees_of_freedom": self.degrees_of_freedom,
            "state_checksum": self.state_checksum,
            # CSR arrays are None but included for completeness
            "indptr": self.indptr,
            "indices": self.indices,
            "data": self.data
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

    # ─────────────────── transition payload builder ─────────────────────
    def prepare_transition(self) -> Dict[str, Any]:
        """Prepare state for transition to UNA stage.
        
        The left inverse operation will generate the first observable structure.
        This method validates the current state and packages all necessary
        information for the UNA stage constructor.
        
        Returns:
            Dictionary containing all state needed for UNA initialization
            
        Raises:
            StructuralViolation: If CS state is invalid for transition
        """
        with self.lock, stage_transition_lock:
            # Validate current state before transition
            self.validate_identity()
            
            # Update cycle index for transition
            self.cycle_index += 1
            
            # Package state for UNA stage
            state = {
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
                
                # CS memory carried forward (π/2 chirality seed in fixed-point)
                "cs_memory": self.left_chirality_seed,
                
                # Structural information
                "degrees_of_freedom": self.degrees_of_freedom,
                
                # Schema versioning
                "schema_version": SCHEMA_VERSION
            }
            
            # Update checksum after preparing transition
            self.state_checksum = self._compute_checksum()
            
            # Record transition preparation via genetic memory
            self._record_to_trace("cs_transition_prepared",
                                  target_stage="UNA",
                                  cycle_index=state["cycle_index"],
                                  cs_memory=self._from_fixed_point(state["cs_memory"]),
                                  checksum=self.state_checksum)
            
            return state

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
            "source": "G1_CS",
            "event_type": event_type,
            "tensor_id": self.tensor_id,
            "cycle_index": self.cycle_index,
            "stage": self.stage,
            **kw
        }
        
        # Log to console/file via standard logging
        logger.debug("CS Event: %s", evt)
        
        # Delegate to genetic memory for proper trace recording
        self.genetic_memory.record_event(evt)

    # ───────────────────────── utility methods ─────────────────────────────
    def get_chirality_seed(self) -> float:
        """Get the fundamental chirality seed value.
        
        Returns:
            The π/2 chirality seed that defines the unobservable memory
        """
        return self._from_fixed_point(self.left_chirality_seed)

    def is_spawn_eligible(self) -> bool:
        """Check if tensor is eligible for spawning.
        
        CS stage is never directly spawn-eligible as it has no amplitude.
        Spawning occurs at BU_En stage after completing the full cycle.
        
        Returns:
            Always False for CS stage
        """
        return False

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

    def __repr__(self) -> str:
        """String representation of CS identity."""
        return (f"CSIdentity(tensor_id={self.tensor_id}, "
                f"cycle={self.cycle_index}, "
                f"chirality_seed={self.get_chirality_seed():.6f})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"CS Identity τ{self.tensor_id} (cycle {self.cycle_index})"