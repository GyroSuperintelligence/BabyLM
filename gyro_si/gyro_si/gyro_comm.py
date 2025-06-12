"""
Communication infrastructure for GyroSI.

This module provides the canonical inter-system communication API. All
G-systems must communicate exclusively through these interfaces. The
implementation ensures message ordering by cycle_index and maintains
full audit trails for all inter-system communication.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Awaitable, Union
from collections import defaultdict
from dataclasses import dataclass

# Absolute imports to enforce unambiguous module resolution as per the spec
from gyro_constants import STANDARD_FIELDS, HELICAL_CONTEXT_FIELDS
from gyro_errors import CommunicationError, RoutingError

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Fault detected, rejecting requests
    HALF_OPEN = auto()   # Testing recovery

@dataclass
class Message:
    """
    Canonical message structure for inter-system communication.
    All fields are required and must follow the STANDARD_FIELDS ordering.
    """
    type: str                    # Message type identifier
    source: str                  # Originating G-system
    destination: str             # Target G-system or "broadcast"
    cycle_index: int             # Monotonic cycle counter
    tensor_context: Dict[str, Any]  # Helical state information
    payload: Dict[str, Any]      # Message-specific data
    timestamp: str               # ISO 8601 timestamp

    def __post_init__(self):
        """Validate message structure."""
        if not isinstance(self.cycle_index, int) or self.cycle_index < 0:
            raise CommunicationError(f"Invalid cycle_index: {self.cycle_index}")

        if self.source not in {"G1", "G2", "G3", "G4", "G5", "G6"}:
            raise CommunicationError(f"Invalid source: {self.source}")

        if self.destination not in {"G1", "G2", "G3", "G4", "G5", "G6", "broadcast"}:
            raise CommunicationError(f"Invalid destination: {self.destination}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary preserving STANDARD_FIELDS order."""
        return {field: getattr(self, field) for field in STANDARD_FIELDS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create Message from dictionary, validating required fields."""
        missing = set(STANDARD_FIELDS) - set(data.keys())
        if missing:
            raise CommunicationError(f"Missing required fields: {missing}")

        return cls(**{field: data[field] for field in STANDARD_FIELDS})

@dataclass
class SystemEvent:
    """
    Base class for system-level events that aren't regular messages.
    This ensures all events have a consistent structure.
    """
    type: str
    source: str
    timestamp: str

    @classmethod
    def create(cls, type_name: str, source: str, **kwargs) -> 'SystemEvent':
        """Factory method to create a SystemEvent with current timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat()
        event_data = {"type": type_name, "source": source, "timestamp": timestamp}
        event_data.update(kwargs)
        return cls(**event_data)

@dataclass
class CircuitStateChangeEvent(SystemEvent):
    """Event generated when a circuit breaker state changes."""
    system: str
    old_state: CircuitState
    new_state: CircuitState

# Define the explicit type for all subscriber handlers
EventHandler = Callable[[Union[Message, SystemEvent]], Awaitable[None]]

class MessageRouter:
    """
    Central message routing system for GyroSI.
    Handles message delivery, subscription management, and maintains
    ordering guarantees based on cycle_index.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[EventHandler]] = defaultdict(set)
        self._queues: Dict[str, asyncio.Queue] = {}
        self._circuit_breakers: Dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self._message_buffer: List[Message] = []
        self._lock = asyncio.Lock()

        # Initialize queues for each G-system
        for system in ["G1", "G2", "G3", "G4", "G5", "G6"]:
            self._queues[system] = asyncio.Queue()

    async def send_message(self, message: Dict[str, Any]) -> None:
        """
        Send a message through the routing system.

        Messages are validated, enriched with timestamp if missing,
        and routed according to destination.

        Args:
            message: Dictionary containing message fields

        Raises:
            CommunicationError: If message is invalid
            RoutingError: If destination is unreachable
        """
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Validate and create Message object
        try:
            msg = Message.from_dict(message)
        except Exception as e:
            raise CommunicationError(f"Invalid message format: {e}")

        # Check circuit breaker
        if self._circuit_breakers[msg.destination] == CircuitState.OPEN:
            raise RoutingError(f"Circuit breaker OPEN for {msg.destination}")

        # Route message
        async with self._lock:
            if msg.destination == "broadcast":
                # Broadcast to all systems except source
                for system, queue in self._queues.items():
                    if system != msg.source:
                        await queue.put(msg)
            else:
                # Direct routing
                if msg.destination in self._queues:
                    await self._queues[msg.destination].put(msg)
                else:
                    raise RoutingError(f"Unknown destination: {msg.destination}")

            # Add to buffer for audit
            self._message_buffer.append(msg)

        # Log message
        logger.debug(f"Routed message: {msg.type} from {msg.source} to {msg.destination}")

    def subscribe(self, message_type: str, handler: EventHandler) -> None:
        """
        Subscribe to messages or events of a specific type.

        Args:
            message_type: Type of messages/events to receive
            handler: Async callable to handle messages/events
                     Must accept either Message or SystemEvent
        """
        self._subscribers[message_type].add(handler)
        logger.debug(f"Subscribed handler to message type: {message_type}")

    def unsubscribe(self, message_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe from messages of a specific type.

        Args:
            message_type: Type of messages to stop receiving
            handler: Previously subscribed handler
        """
        self._subscribers[message_type].discard(handler)

    async def get_messages(self, system: str, timeout: Optional[float] = None) -> List[Message]:
        """
        Retrieve pending messages for a system.

        Args:
            system: G-system identifier
            timeout: Maximum time to wait for messages

        Returns:
            List of messages ordered by cycle_index
        """
        if system not in self._queues:
            raise RoutingError(f"Unknown system: {system}")

        messages = []
        queue = self._queues[system]

        try:
            # Get all available messages up to timeout
            end_time = asyncio.get_event_loop().time() + (timeout or 0)

            while True:
                remaining = max(0, end_time - asyncio.get_event_loop().time()) if timeout else None

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=remaining)
                    messages.append(msg)
                except asyncio.TimeoutError:
                    break

                # Check if more messages available without blocking
                if queue.empty():
                    break

        except Exception as e:
            logger.error(f"Error retrieving messages for {system}: {e}")
            raise CommunicationError(f"Failed to retrieve messages: {e}")

        # Sort by cycle_index as required
        messages.sort(key=lambda m: m.cycle_index)

        # Dispatch to type-specific subscribers
        for msg in messages:
            for handler in self._subscribers.get(msg.type, []):
                asyncio.create_task(handler(msg))

        return messages

    def set_circuit_state(self, system: str, state: CircuitState) -> None:
        """Update circuit breaker state for a system."""
        old_state = self._circuit_breakers[system]
        self._circuit_breakers[system] = state

        if old_state != state:
            logger.info(f"Circuit breaker for {system}: {old_state.name} -> {state.name}")

            # Create a proper event object for state change
            event = CircuitStateChangeEvent.create(
                type_name=MessageTypes.CIRCUIT_STATE_CHANGE,
                source="G6",  # Circuit system is part of G6
                system=system,
                old_state=old_state,
                new_state=state
            )

            # Notify subscribers of state change
            for handler in self._subscribers.get(MessageTypes.CIRCUIT_STATE_CHANGE, []):
                asyncio.create_task(handler(event))

# ===== Global Router Instance =====
_router = MessageRouter()

# ===== Public API =====

async def send_message(message: Dict[str, Any]) -> None:
    """
    Send a message through the global routing system.

    This is the primary API for inter-system communication in GyroSI.
    All G-systems must use this function for message passing.

    Args:
        message: Dictionary containing required STANDARD_FIELDS

    Raises:
        CommunicationError: If message is invalid
        RoutingError: If routing fails
    """
    await _router.send_message(message)

def subscribe(message_type: str, handler: EventHandler) -> None:
    """
    Subscribe to messages or events of a specific type.

    Args:
        message_type: Type of messages/events to receive (from MessageTypes)
        handler: Async callable that accepts either Message or SystemEvent
    """
    _router.subscribe(message_type, handler)

def unsubscribe(message_type: str, handler: EventHandler) -> None:
    """
    Unsubscribe from messages of a specific type.

    Args:
        message_type: Type of messages to stop receiving
        handler: Previously subscribed handler
    """
    _router.unsubscribe(message_type, handler)

async def get_messages(system: str, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Retrieve pending messages for a system.

    Args:
        system: G-system identifier (e.g., "G1", "G2")
        timeout: Maximum time to wait for messages in seconds

    Returns:
        List of message dictionaries ordered by cycle_index

    Raises:
        RoutingError: If system is unknown
        CommunicationError: If retrieval fails
    """
    messages = await _router.get_messages(system, timeout)
    return [msg.to_dict() for msg in messages]

def set_circuit_state(system: str, state: CircuitState) -> None:
    """
    Update circuit breaker state for a system.

    Used by G4 and G5 for fault management and recovery coordination.

    Args:
        system: G-system identifier
        state: New circuit breaker state
    """
    _router.set_circuit_state(system, state)

# ===== Utility Functions =====

def create_tensor_context(
    cumulative_phase: float,
    chirality_phase: float,
    helical_position: float,
    spinor_cycle_count: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a properly formatted tensor_context dictionary.

    Args:
        cumulative_phase: Total helical progress [0, 4π)
        chirality_phase: Position within forward/return [0, 2π)
        helical_position: Normalized fraction [0, 1)
        spinor_cycle_count: Number of completed 720° revolutions
        **kwargs: Additional context fields (e.g., tensor_id, family_id)

    Returns:
        Dictionary with validated tensor context

    Raises:
        ValueError: If parameters are out of valid ranges
    """
    import math

    # Validate ranges
    if not 0 <= cumulative_phase < 4 * math.pi:
        raise ValueError(f"cumulative_phase must be in [0, 4π), got {cumulative_phase}")

    if not 0 <= chirality_phase < 2 * math.pi:
        raise ValueError(f"chirality_phase must be in [0, 2π), got {chirality_phase}")

    if not 0 <= helical_position < 1:
        raise ValueError(f"helical_position must be in [0, 1), got {helical_position}")

    if spinor_cycle_count < 0:
        raise ValueError(f"spinor_cycle_count must be non-negative, got {spinor_cycle_count}")

    context = {
        "cumulative_phase": cumulative_phase,
        "chirality_phase": chirality_phase,
        "helical_position": helical_position,
        "spinor_cycle_count": spinor_cycle_count
    }

    # Add any additional fields
    context.update(kwargs)

    return context

def validate_message_fields(message: Dict[str, Any]) -> bool:
    """
    Validate that a message contains all required STANDARD_FIELDS.

    Args:
        message: Message dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    return all(field in message for field in STANDARD_FIELDS)

# ===== Message Type Constants =====
# Common message types used across the system

class MessageTypes:
    """Standard message types for inter-system communication."""

    # Bootstrap and lifecycle
    GYRO_BOOTSTRAP = "gyro_bootstrap"
    SYSTEM_READY = "system_ready"
    SHUTDOWN_REQUEST = "shutdown_request"

    # Status and monitoring
    STATUS_UPDATE = "status_update"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRICS = "performance_metrics"

    # Tensor operations
    TENSOR_TRANSITION = "tensor_transition"
    SPAWN_READY = "spawn_ready"
    SPAWN_APPROVED = "spawn_approved"

    # Quantization and observation
    QUANTIZATION_EVENT = "quantization_event"
    OBSERVATION_LOGGED = "observation_logged"

    # Algedonic signals
    ALGEDONIC_SIGNAL = "algedonic_signal"
    PAIN_SIGNAL = "pain_signal"
    PLEASURE_SIGNAL = "pleasure_signal"
    RESOURCE_PRESSURE = "resource_pressure"

    # Policy and governance
    POLICY_UPDATE = "policy_update"
    ASSESSMENT_COMPLETE = "assessment_complete"
    ADVISORY_PROPOSAL = "advisory_proposal"

    # Memory operations
    MEMORY_CHECKPOINT = "memory_checkpoint"
    MEMORY_EXPORT = "memory_export"
    MEMORY_IMPORT = "memory_import"
    # Recovery and fault management
    CARDIAC_RECOVERED = "cardiac_recovered"
    CIRCUIT_STATE_CHANGE = "circuit_state_change"
    FAULT_DETECTED = "fault_detected"

    # Audit and compliance
    AUDIT_EVENT = "audit_event"
    DEVIATION_EVENT = "deviation_event"
    TRACE_COLLECTION = "trace_collection"

# ===== Priority Constants =====

MESSAGE_PRIORITY = {
    "G1": 1,  # Highest priority - core tensor operations
    "G2": 2,  # Information routing
    "G3": 3,  # User interaction
    "G4": 4,  # Environmental monitoring
    "G5": 5,  # Policy decisions
    "G6": 6   # System coordination
} 