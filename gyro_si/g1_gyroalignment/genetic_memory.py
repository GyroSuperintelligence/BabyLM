"""
G1 Genetic Memory: Immutable Audit Log and State Management

This module provides the genetic memory infrastructure for the G1 GyroAlignment
system. It maintains an immutable, auditable log of all tensor operations and
manages state snapshots for recovery purposes.

Genetic memory represents the most fundamental form of memory in the CGM model,
encoding the structural patterns and operational history that define tensor
identity and evolution.
"""

import threading
import time
import logging
import hashlib
from collections import deque
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass

from gyro_si.gyro_constants import HALF_HORIZON
from gyro_si.gyro_errors import StructuralViolation

logger = logging.getLogger(__name__)

# Module-level lock for thread safety
trace_lock = threading.RLock()

# Critical event types that must be retained during pruning
CRITICAL_EVENT_TYPES = {
    "spawn_ready",
    "spawn_eligibility_detected",
    "spawn_ready_notification_sent",
    "stage_transition",
    "cs_transition_prepared",
    "una_transition_prepared",
    "ona_transition_prepared",
    "bu_in_transition_prepared",
    "bu_en_transition_prepared",
    "memory_collision_detected",
    "segment_collapsed",
    "tensor_created",
    "spinor_cycle_completed",
    "chirality_flip"
}

@dataclass
class TraceEvent:
    """Structured representation of a trace event."""
    timestamp: float
    source: str
    event_type: str
    tensor_id: int
    cycle_index: int
    stage: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "timestamp": self.timestamp,
            "source": self.source,
            "event_type": self.event_type,
            "tensor_id": self.tensor_id,
            "cycle_index": self.cycle_index,
            "stage": self.stage
        }
        # Flatten the data into the result
        result.update(self.data)
        return result

@dataclass
class Snapshot:
    """Represents a delta-encoded state snapshot."""
    snapshot_id: str
    timestamp: float
    tensor_id: int
    cycle_index: int
    cycle_index_base: int
    state_delta: Dict[str, Any]
    checksum: str
    is_full_state: bool = False  # Flag to indicate if this is a full state snapshot

class GeneticMemory:
    """
    Manages the genetic memory for the G1 system, including the trace buffer
    and state snapshots for all tensors.
    
    This class provides thread-safe operations for recording events, managing
    snapshots, and interfacing with G5 for audit collection.
    """
    
    def __init__(self, max_buffer_size: Optional[int] = None, 
                 retention_horizon: Optional[int] = None):
        """
        Initialize the GeneticMemory instance.
        
        Args:
            max_buffer_size: Maximum number of events in buffer (default: 10 * HALF_HORIZON)
            retention_horizon: Number of recent events to keep during pruning (default: HALF_HORIZON)
        """
        # Use canonical constants if not specified
        self.retention_horizon = retention_horizon or HALF_HORIZON
        self.max_buffer_size = max_buffer_size or (self.retention_horizon * 10)
        
        # Core data structures
        self.trace_buffer: deque = deque(maxlen=self.max_buffer_size)
        self.snapshots: Dict[str, Snapshot] = {}
        self.last_known_states: Dict[int, Dict[str, Any]] = {}
        
        # Tracking for pruning and collection
        self.total_events_recorded = 0
        self.total_events_pruned = 0
        self.last_collection_time = time.time()
        self.collection_count = 0
        
        # Full state snapshot interval (every N cycles)
        self.full_snapshot_interval = 100
        
        logger.info("GeneticMemory initialized with buffer_size=%d, retention=%d",
                    self.max_buffer_size, self.retention_horizon)
    
    def record_event(self, event: Dict[str, Any]) -> None:
        """
        Record a single event to the thread-safe trace buffer.
        
        This is the primary interface for all G1 stages. It validates the event,
        adds metadata if missing, and manages buffer size through pruning.
        
        Args:
            event: Dictionary containing event data with required fields:
                   source, event_type, tensor_id, cycle_index, stage
                   (timestamp is auto-filled if missing)
        
        Raises:
            ValueError: If required fields are missing
        """
        # Auto-fill timestamp if missing
        if "timestamp" not in event:
            event["timestamp"] = time.time()
            
        # Validate required fields
        required_fields = {"source", "event_type", "tensor_id", "cycle_index", "stage"}
        missing_fields = required_fields - set(event.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields in event: {missing_fields}")
        
        with trace_lock:
            # Create structured event
            trace_event = TraceEvent(
                timestamp=event["timestamp"],
                source=event["source"],
                event_type=event["event_type"],
                tensor_id=event["tensor_id"],
                cycle_index=event["cycle_index"],
                stage=event["stage"],
                data={k: v for k, v in event.items() if k not in {"timestamp", "source", 
                                                                   "event_type", "tensor_id", 
                                                                   "cycle_index", "stage"}}
            )
            
            # Add to buffer
            self.trace_buffer.append(trace_event)
            self.total_events_recorded += 1
            
            # Prune if necessary
            self._prune_buffer()
            
            # Log critical events
            if event["event_type"] in CRITICAL_EVENT_TYPES:
                logger.debug("Critical event recorded: %s for tensor %d at cycle %d",
                            event["event_type"], event["tensor_id"], event["cycle_index"])
    
    def get_collected_traces(self) -> List[Dict[str, Any]]:
        """
        Provide a copy of the entire trace buffer for G5 collection.
        
        This method is thread-safe and returns events in chronological order.
        
        Returns:
            List of event dictionaries ready for collection
        """
        with trace_lock:
            # Convert TraceEvent objects to dictionaries
            traces = [event.to_dict() for event in self.trace_buffer]
            
            logger.info("Trace collection requested: %d events available", len(traces))
            return traces
    
    def clear_collected_traces(self, before_timestamp: float) -> int:
        """
        Remove collected events from the buffer to free memory.
        
        This method is called by G5 after successfully storing traces.
        Only removes events older than the specified timestamp to avoid
        race conditions with concurrent event recording.
        
        Args:
            before_timestamp: Remove events with timestamp < this value
            
        Returns:
            Number of events cleared
        """
        with trace_lock:
            initial_size = len(self.trace_buffer)
            
            # Filter out old events while preserving critical ones
            retained_events = []
            cleared_count = 0
            
            for event in self.trace_buffer:
                if event.timestamp < before_timestamp and event.event_type not in CRITICAL_EVENT_TYPES:
                    cleared_count += 1
                else:
                    retained_events.append(event)
            
            # Replace buffer with retained events, preserving maxlen
            self.trace_buffer = deque(retained_events, maxlen=self.max_buffer_size)
            
            # Update collection tracking
            self.last_collection_time = time.time()
            self.collection_count += 1
            
            logger.info("Cleared %d events from trace buffer (retained %d)",
                       cleared_count, len(self.trace_buffer))
            
            return cleared_count
    
    def create_snapshot(self, tensor_id: int, current_state: Dict[str, Any]) -> str:
        """
        Create a delta-encoded snapshot of a tensor's state.
        
        Compares the current state to the last known state for the tensor,
        stores only the changed fields, and returns a unique snapshot ID.
        Periodically creates full state snapshots for robust recovery.
        
        Args:
            tensor_id: The ID of the tensor being snapshotted
            current_state: The complete current state of the tensor
            
        Returns:
            A unique identifier for the created snapshot
        """
        with trace_lock:
            # Get last known state
            last_state = self.last_known_states.get(tensor_id, {})
            cycle_index = current_state.get("cycle_index", 0)
            
            # Determine if we should create a full state snapshot
            is_full_state = (cycle_index % self.full_snapshot_interval == 0) or (tensor_id not in self.last_known_states)
            
            if is_full_state:
                # Store complete state
                state_delta = current_state.copy()
            else:
                # Compute delta (changed fields only)
                state_delta = {}
                for key, value in current_state.items():
                    if key not in last_state or last_state[key] != value:
                        state_delta[key] = value
            
            # Generate snapshot ID
            snapshot_id = f"{tensor_id}_{cycle_index}_{int(time.time() * 1000000)}"
            
            # Compute checksum for integrity
            checksum = self._compute_snapshot_checksum(tensor_id, cycle_index, state_delta)
            
            # Create snapshot object
            snapshot = Snapshot(
                snapshot_id=snapshot_id,
                timestamp=time.time(),
                tensor_id=tensor_id,
                cycle_index=cycle_index,
                cycle_index_base=last_state.get("cycle_index", 0) if not is_full_state else cycle_index,
                state_delta=state_delta,
                checksum=checksum,
                is_full_state=is_full_state
            )
            
            # Store snapshot
            self.snapshots[snapshot_id] = snapshot
            
            # Update last known state
            self.last_known_states[tensor_id] = current_state.copy()
            
            logger.debug("Created %s snapshot %s for tensor %d (delta size: %d fields)",
                        "full" if is_full_state else "delta",
                        snapshot_id, tensor_id, len(state_delta))
            
            return snapshot_id
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete tensor state from a snapshot ID.
        
        Applies the stored delta to the relevant base state to reconstruct
        the full state at the time of the snapshot.
        
        Args:
            snapshot_id: The unique identifier of the snapshot
            
        Returns:
            The reconstructed full tensor state, or None if not found
        """
        with trace_lock:
            snapshot = self.snapshots.get(snapshot_id)
            if not snapshot:
                logger.warning("Snapshot %s not found", snapshot_id)
                return None
            
            # Verify checksum
            computed_checksum = self._compute_snapshot_checksum(
                snapshot.tensor_id, snapshot.cycle_index, snapshot.state_delta
            )
            if computed_checksum != snapshot.checksum:
                raise StructuralViolation(f"Snapshot {snapshot_id} checksum mismatch")
            
            # If this is a full state snapshot, return it directly
            if snapshot.is_full_state:
                return snapshot.state_delta.copy()
            
            # Otherwise, reconstruct from the most recent full state
            # Find the most recent full state snapshot for this tensor
            base_state = {}
            relevant_snapshots = []
            
            # Collect all snapshots for this tensor
            for sid, snap in self.snapshots.items():
                if snap.tensor_id == snapshot.tensor_id and snap.cycle_index <= snapshot.cycle_index:
                    relevant_snapshots.append(snap)
            
            # Sort by cycle index to apply in chronological order
            relevant_snapshots.sort(key=lambda s: s.cycle_index)
            
            # Find the most recent full state
            full_state_found = False
            for snap in reversed(relevant_snapshots):
                if snap.is_full_state:
                    base_state = snap.state_delta.copy()
                    full_state_found = True
                    # Now apply all deltas after this full state
                    for delta_snap in relevant_snapshots:
                        if delta_snap.cycle_index > snap.cycle_index and delta_snap.cycle_index <= snapshot.cycle_index:
                            base_state.update(delta_snap.state_delta)
                    break
            
            # If no full state found, reconstruct from all deltas
            if not full_state_found:
                for snap in relevant_snapshots:
                    base_state.update(snap.state_delta)
            
            logger.debug("Retrieved snapshot %s (reconstructed %d fields)",
                        snapshot_id, len(base_state))
            
            return base_state
    
    def get_tensor_lineage(self, tensor_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve the complete event history for a specific tensor.
        
        Args:
            tensor_id: The tensor ID to get lineage for
            
        Returns:
            List of events for this tensor in chronological order
        """
        with trace_lock:
            lineage = [
                event.to_dict() for event in self.trace_buffer
                if event.tensor_id == tensor_id
            ]
            
            logger.debug("Retrieved lineage for tensor %d: %d events",
                        tensor_id, len(lineage))
            
            return lineage
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the genetic memory system.
        
        Returns:
            Dictionary containing memory usage and performance statistics
        """
        with trace_lock:
            critical_events = sum(1 for event in self.trace_buffer 
                                 if event.event_type in CRITICAL_EVENT_TYPES)
            
            full_snapshots = sum(1 for snap in self.snapshots.values() if snap.is_full_state)
            delta_snapshots = len(self.snapshots) - full_snapshots
            
            return {
                "buffer_size": len(self.trace_buffer),
                "max_buffer_size": self.max_buffer_size,
                "buffer_utilization": len(self.trace_buffer) / self.max_buffer_size,
                "critical_events": critical_events,
                "total_events_recorded": self.total_events_recorded,
                "total_events_pruned": self.total_events_pruned,
                "snapshot_count": len(self.snapshots),
                "full_snapshots": full_snapshots,
                "delta_snapshots": delta_snapshots,
                "tracked_tensors": len(self.last_known_states),
                "collection_count": self.collection_count,
                "last_collection_age": time.time() - self.last_collection_time,
                "retention_horizon": self.retention_horizon
            }
    
    def _prune_buffer(self) -> None:
        """
        Manage the trace buffer size according to retention policy.
        
        Ensures the buffer does not grow indefinitely while retaining a
        minimum number of recent events and all critical events.
        """
        # Only prune if we're over the retention horizon
        if len(self.trace_buffer) <= self.retention_horizon:
            return
        
        # Calculate how many events to potentially remove
        excess = len(self.trace_buffer) - self.retention_horizon
        
        # Build list of candidates for removal (oldest first, non-critical only)
        removal_candidates = []
        for i, event in enumerate(self.trace_buffer):
            if event.event_type not in CRITICAL_EVENT_TYPES:
                removal_candidates.append(i)
            if len(removal_candidates) >= excess:
                break
        
        # Remove events from the buffer (in reverse order to maintain indices)
        removed_count = 0
        for idx in reversed(removal_candidates[:excess]):
            del self.trace_buffer[idx]
            removed_count += 1
            self.total_events_pruned += 1
        
        if removed_count > 0:
            logger.debug("Pruned %d non-critical events from trace buffer", removed_count)
    
        def _compute_snapshot_checksum(self,
                                    tensor_id: int,
                                    cycle_index: int,
                                    state_delta: Dict[str, Any]) -> str:
            """
            Compute a checksum for snapshot integrity verification.

            Args:
                tensor_id: The tensor ID
                cycle_index: The cycle index
                state_delta: The state delta dictionary

            Returns:
                SHA-256 checksum as hex string
            """
        h = hashlib.sha256()
        
        # Add tensor identity
        h.update(str(tensor_id).encode())
        h.update(str(cycle_index).encode())
        
        # Add sorted state delta for deterministic hashing
        for key in sorted(state_delta.keys()):
            h.update(key.encode())
            h.update(str(state_delta[key]).encode())
        
        return h.hexdigest()

# Singleton instance
_genetic_memory_instance: Optional[GeneticMemory] = None

def get_genetic_memory() -> GeneticMemory:
    """
    Get the singleton instance of GeneticMemory.
    
    This ensures all G1 components share the same memory system.
    
    Returns:
        The shared GeneticMemory instance
    """
    global _genetic_memory_instance
    
    if _genetic_memory_instance is None:
        _genetic_memory_instance = GeneticMemory()
        logger.info("Initialized singleton GeneticMemory instance")
    
    return _genetic_memory_instance

# Convenience function for backward compatibility
def get_genetic_memory_instance() -> GeneticMemory:
    """
    Alias for get_genetic_memory() for backward compatibility.
    
    Returns:
        The shared GeneticMemory instance
    """
    return get_genetic_memory()
                