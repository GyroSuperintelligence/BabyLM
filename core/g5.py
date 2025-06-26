"""
extension_manager.py - Tier 2: The Extension Manager & Unified Core API

This module contains the ExtensionManager, the central orchestrator of the
GyroSI system. It embodies the "Unified Core" by providing the canonical G1-G5
memory system interfaces.

Its responsibilities are:
- Instantiating the pure GyroEngine and all system/application extensions.
- Loading and managing session/knowledge state by delegating to extensions.
- Providing the G1-G5 `gyro_*_memory` functions as a stable internal API.
- Orchestrating the full `gyro_operation` cycle: calling the engine, then
  dispatching results to extensions for persistence, forking, and analysis.
"""

import hashlib
import os
import uuid
import importlib
from typing import Callable, cast, final, TypeVar
from collections.abc import Mapping

# Tier 3 - The pure engine
from core.g1 import GyroEngine, gyration_op

# Core data structures and errors
from core.gyro_errors import (
    GyroExtensionError,
    GyroIntegrityError,
    GyroSessionError,
    GyroTagError,
)
from core.gyro_tag_parser import parse_tag, validate_tag
from memory.agency.g3_memory import G3_Memory
from memory.agency.g2_memory import G2_Memory

T = TypeVar("T")

# Forward references to prevent import cycles
StorageManagerType = T
ForkManagerType = T
ErrorHandlerType = T
StateHelperType = T


@final
class GyroOperation :
    """
    The orchestrator for a single, active GyroSI session.

    This class provides the unified interface to all five memory systems (G1-G5)
    and orchestrates the interaction between the pure GyroEngine and the
    extension ecosystem.
    """

    def __init__(self, agent_uuid: str | None = None):
        """
        Initializes the full GyroSI stack for a session.

        Args:
            agent_uuid: The UUID of an existing session to resume. If None, creates new.
        """
        self.agent_uuid: str = agent_uuid or str(uuid.uuid4())
        self.extensions: dict[str, object] = {}
        self.hlog: G3_Memory | None = None
        self.navigation_log: G2_Memory | None = None

        self._output_handler: Callable[[str], None] | None = None
        self.engine: GyroEngine | None = None

        # Initialize packed_byte to ensure it's always defined
        self._packed_byte: int = 0

        try:
            # 1. Initialize system-critical extensions in dependency order
            self._initialize_system_extensions()

            # 2. Initialize application extensions
            self._initialize_application_extensions()

            # 3. Initialize the pure engine
            self.engine = GyroEngine(
                harmonics_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "memory", "g1_memory.dat")
            )

            # 4. Load session state through extensions
            self._load_session_state()

            # 5. Perform integrity validation
            self._validate_system_integrity()

        except Exception as e:
            # Clean up any partially initialized state
            self._cleanup_on_error()
            raise GyroSessionError(f"Failed to initialize session: {str(e)}") from e

    def _initialize_system_extensions(self) -> None:
        """Initialize system-critical extensions in proper dependency order."""
        # Use dynamic imports to avoid circular imports
        ext_cryptographer = importlib.import_module("extensions.ext_cryptographer")
        ext_error_handler = importlib.import_module("extensions.ext_error_handler")
        ext_fork_manager = importlib.import_module("extensions.ext_fork_manager")
        ext_state_helper = importlib.import_module("extensions.ext_state_helper")

        # Initialize cryptographer FIRST
        self.extensions["crypto"] = ext_cryptographer.ext_Cryptographer(self._get_user_key())

        # State helper manages session/knowledge state coordination
        self.extensions["state"] = ext_state_helper.ext_StateHelper()

        # Fork manager handles knowledge immutability
        self.extensions["fork"] = ext_fork_manager.ext_ForkManager()

        # Error handler for centralized error management
        self.extensions["error"] = ext_error_handler.ext_ErrorHandler()

        # Set manager references for all extensions that need it
        for ext in self.extensions.values():
            if hasattr(ext, "set_manager"):
                getattr(ext, "set_manager")(self)

        # Create HLog after all critical extensions are loaded
        try:
            # Get gene checksum for HLog
            gene_checksum = None
            if self.engine and hasattr(self.engine, "gene"):
                hasher = hashlib.sha256()
                hasher.update(self.engine.gene["id_0"].numpy().tobytes())
                hasher.update(self.engine.gene["id_1"].numpy().tobytes())
                gene_checksum = hasher.digest()

            # Create HLog
            hlog_path = getattr(self.extensions["state"], "get_hlog_path")(self.agent_uuid)
            self.hlog = G3_Memory(hlog_path, gene_checksum)
        except Exception as e:
            # Log error but continue - HLog is optional for now
            if "error" in self.extensions:
                error_handler = self.extensions["error"]
                if hasattr(error_handler, "handle_error"):
                    getattr(error_handler, "handle_error")(e)
            print(f"Warning: Could not initialize HLog: {e}")

    def _initialize_application_extensions(self) -> None:
        """Initialize application-level extensions for pattern analysis."""
        # Use dynamic imports to avoid circular imports
        ext_dynamic_codec = importlib.import_module("extensions.ext_dynamic_codec")

        # Initialize dynamic codec
        crypto = self.extensions.get("crypto")
        self.extensions["dynamic_codec"] = ext_dynamic_codec.ext_DynamicCodec(
            crypto=crypto
        )

        # Set manager references for all extensions that need it
        for ext in self.extensions.values():
            if hasattr(ext, "set_manager"):
                getattr(ext, "set_manager")(self)

    def _load_session_state(self) -> None:
        """Load session state through the state helper extension."""
        # Load phase and navigation log
        state_helper = self.extensions["state"]
        if hasattr(state_helper, "load_session_state"):
            state = getattr(state_helper, "load_session_state")()

            # Set engine phase
            if self.engine and "phase" in state:
                self.engine.load_phase(state["phase"])

            # Initialize navigation log
            self.navigation_log = G2_Memory(agent_uuid=self.agent_uuid)

            # Load extension states
            for ext in self.extensions.values():
                if hasattr(ext, "load_state"):
                    getattr(ext, "load_state")()

    def _validate_system_integrity(self) -> None:
        """Perform comprehensive system integrity validation."""
        try:
            # Validate Gene checksum
            if self.engine:
                if hasattr(self.extensions["state"], "validate_gene_checksum"):
                    if not getattr(self.extensions["state"], "validate_gene_checksum")(self.engine.gene):
                        raise GyroIntegrityError("Gene checksum validation failed")
        except Exception:
            raise GyroIntegrityError("Gene validation failed")

        # Validate extension footprints
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, "validate_footprint"):
                if not getattr(ext, "validate_footprint")():
                    raise GyroExtensionError(f"Extension {ext_name} footprint validation failed")

    def _cleanup_on_error(self) -> None:
        """Clean up any partially initialized state on error."""
        # Shutdown extensions in reverse order
        for ext in reversed(list(self.extensions.values())):
            if hasattr(ext, "shutdown"):
                try:
                    getattr(ext, "shutdown")()
                except Exception:
                    pass  # Best effort cleanup

    def get_extension(self, name: str) -> object:
        """
        Helper to retrieve a loaded extension instance.

        Args:
            name: The extension name (without 'ext_' prefix).

        Returns:
            The extension instance.

        Raises:
            GyroExtensionError: If extension not found.
        """
        if name not in self.extensions:
            raise GyroExtensionError(f"Extension '{name}' not found")
        return self.extensions[name]

    def get_session_id(self) -> str:
        """Returns the managed session ID."""
        return self.agent_uuid

    def get_knowledge_id(self) -> str:
        """Returns the current knowledge package ID."""
        return self.agent_uuid

    # ========================================================================
    # G1-G5 CANONICAL MEMORY INTERFACES
    # ========================================================================

    def gyro_genetic_memory(self, tag: str, data: object | None = None) -> object:
        """
        G1: GyroAlignment through GyroTensor Management (Genetic Memory)

        Provides unified TAG-based access to all five invariants across G1-G5.
        Routes queries to appropriate storage (knowledge vs session).

        Args:
            tag: TAG expression per CORE-SPEC-04 grammar
            data: Optional data for write Operation 

        Returns:
            Requested invariant data

        Raises:
            GyroTagError: Invalid TAG expression
        """
        # Validate TAG syntax
        if not validate_tag(tag):
            raise GyroTagError(f"Invalid TAG expression: {tag}")

        # Parse TAG components
        tag_parts = parse_tag(tag)
        temporal = tag_parts["temporal"]
        invariant = tag_parts["invariant"]

        # Route based on invariant type
        if invariant == "gyrotensor_id":
            # Phase counter (session-local)
            if self.engine:
                if temporal == "current":
                    return self.engine.phase
                elif temporal == "previous":
                    return (self.engine.phase - 1) % 48
                elif temporal == "next":
                    return (self.engine.phase + 1) % 48
            return 0  # Default if engine not initialized

        elif invariant == "gyrotensor_com":
            # Event log (2×3 tensor)
            return self._get_event_tensor(temporal)

        elif invariant == "gyrotensor_nest":
            # Nesting structure (2×2×3 tensor)
            return self._get_nest_tensor(temporal)

        elif invariant == "gyrotensor_add":
            # Gene (always returns constant)
            return self.engine.gene if self.engine else None

        elif invariant == "gyrotensor_quant":
            # Navigation log (decoded state)
            return self._get_decoded_gene_state(temporal)

        else:
            raise GyroTagError(f"Unknown invariant: {invariant}")

    def gyro_epigenetic_memory(self, tag: str, data: object | None = None) -> object:
        """
        G2: GyroInformation through GyroTensor Curation (Epigenetic Memory)

        Manages dual event streams:
        - Learning events → knowledge package (exportable)
        - Session events → session directory (non-exportable)

        Per CORE-SPEC-03 mapping:
        - G2_CS: All Data Schemas
        - G2_UNA: Backend Pipeline
        - G2_ONA: Frontend Data
        - G2_BU_In: Ingress Data & Directives
        - G2_BU_Eg: Egress Data & Events
        """
        if data is not None:
            # Writing event - classify and store
            classifier = self.extensions.get("classifier")
            if (
                classifier
                and hasattr(classifier, "is_learning_event")
                and hasattr(self.extensions["state"], "store_learning_event")
                and hasattr(self.extensions["state"], "store_session_event")
            ):

                if getattr(classifier, "is_learning_event")(data):
                    # High-value learning event
                    getattr(self.extensions["state"], "store_learning_event")(self.agent_uuid, data)
                else:
                    # Session-specific event
                    getattr(self.extensions["state"], "store_session_event")(self.agent_uuid, data)

        # Parse TAG for retrieval
        tag_parts = parse_tag(tag)
        temporal = tag_parts["temporal"]

        # Return appropriate event data
        if temporal == "current":
            return None
        elif temporal == "previous":
            return None
        else:
            return None

    def gyro_structural_memory(self, tag: str, data: object | None = None) -> object:
        """
        G3: GyroInference through GyroTensor Interaction (Structural Memory)
        Now supports language output through .output context.

        Manages session-local I/O boundaries and UI state.
        All G3 data is session-specific and non-exportable.

        Per CORE-SPEC-03 mapping:
        - G3_CS: Hardware Endpoints
        - G3_UNA: Data Endpoints
        - G3_ONA: Frontend Interface
        - G3_BU_In: User/System Input
        - G3_BU_Eg: System Output
        """
        # Parse TAG components
        tag_parts = parse_tag(tag)

        # Handle language output
        if tag.endswith(".output") and data is not None:
            # This is how the system speaks!
            text_data = str(data) if data is not None else ""
            self._handle_language_output(text_data)
            return None

        # Existing implementation for other G3 Operation 
        if tag_parts["invariant"] == "gyrotensor_nest":
            # UI state is stored in session
            if hasattr(self.extensions["state"], "load_ui_state"):
                return getattr(self.extensions["state"], "load_ui_state")(self.agent_uuid)

        return None

    def gyro_somatic_memory(self, tag: str, data: object | None = None) -> object:
        """
        G4: GyroIntelligence through GyroTensor Ingress Cooperation (Somatic Memory)

        Tracks navigation phase and implements structural resonance.
        Phase is session-local and resets when linking to new knowledge.

        Per CORE-SPEC-03 mapping:
        - G4_CS: Governance Traceability
        - G4_UNA: Information Variety
        - G4_ONA: Inference Accountability
        - G4_BU_In: Intelligence Integrity Ingress
        - G4_BU_Eg: Intelligence Integrity Egress
        """
        tag_parts = parse_tag(tag)

        # Most G4 queries are about phase
        if "phase" in tag or tag_parts["invariant"] == "gyrotensor_id":
            return self.engine.phase if self.engine else 0

        # G4 also handles structural resonance info
        if "resonance" in tag:
            return None

        return None

    def gyro_immunity_memory(self, tag: str, data: object | None = None) -> object:
        """
        G5: GyroIntelligence through GyroTensor Egress Operation (Immunity Memory)

        Manages navigation log with fork-on-write for knowledge immutability.
        Implements the three Genome operators per CORE-SPEC-03.

        Per CORE-SPEC-03 mapping:
        - G5_CS: Management through G1
        - G5_UNA: Curation through G2
        - G5_ONA: Interaction through G3
        - G5_BU_In: Cooperation through G4
        - G5_BU_Eg: Operation through G5
        """
        tag_parts = parse_tag(tag)

        if tag_parts["invariant"] == "gyrotensor_quant":
            # Return navigation log reference
            return self.navigation_log

        # G5 also handles operator info
        if "operator" in tag:
            return None

        return None

    # ========================================================================
    # CORE ORCHESTRATION
    # ========================================================================

    def gyro_operation(self, input_byte: int) -> tuple[int, int]:
        """
        Orchestrates one complete, atomic navigation cycle.
        Now ALWAYS returns operator codes (never None).
        """
        if not self.engine:
            return (0, 0)  # Default if engine not initialized

        try:
            # 1. Execute the pure cycle in the engine
            # This now ALWAYS returns operator codes
            ops = self.engine.execute_cycle(input_byte)

            # 2. Delegate fork-on-write management
            if self.navigation_log:
                fork_manager = self.extensions.get("fork")
                if fork_manager and hasattr(fork_manager, "ensure_writable"):
                    self.navigation_log = getattr(fork_manager, "ensure_writable")(
                        self.navigation_log
                    )

                # 3. Use new pipeline for language processing
                self._packed_byte = (ops[1] & 0x0F) << 4 | (ops[0] & 0x0F)
                self._coordinate_pipeline(self._packed_byte, input_byte)

            # 4. Notify other extensions (for backward compatibility)
            self._notify_extensions(self._packed_byte, input_byte)

            # 5. Persist critical state
            state_helper = self.extensions.get("state")
            if state_helper and hasattr(state_helper, "persist_phase") and self.engine:
                getattr(state_helper, "persist_phase")(self.engine.phase)

            return ops

        except Exception as e:
            # Let error handler manage the exception
            error_handler = self.extensions.get("error")
            if error_handler and hasattr(error_handler, "handle_error"):
                getattr(error_handler, "handle_error")(e)
            raise

    def _notify_extensions(self, packed_nav: int, input_byte: int) -> None:
        """
        Notify all extensions with navigation event handlers.

        Args:
            packed_nav: The packed navigation byte (two 4-bit codes).
            input_byte: The original input byte that caused this navigation.
        """
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, "ext_on_navigation_event"):
                try:
                    getattr(ext, "ext_on_navigation_event")(packed_nav, input_byte)
                except Exception as e:
                    # Log but don't fail the operation
                    error_handler = self.extensions.get("error")
                    if error_handler and hasattr(error_handler, "log_extension_error"):
                        getattr(error_handler, "log_extension_error")(ext_name, e)

    # ========================================================================
    # HIGH-LEVEL MANAGEMENT METHODS (Called by API)
    # ========================================================================

    def export_knowledge(self, output_path: str) -> None:
        """
        Exports the current knowledge package to a .gyro bundle.

        Args:
            output_path: The file path to save the bundle.
        """
        # Ensure navigation log is persisted
        # Comment out or stub: self.navigation_log.persist_to_disk()

        # Gather extension learning states
        extension_states = {}
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, "get_learning_state"):
                state = getattr(ext, "get_learning_state")()
                if state:  # Only include non-empty states
                    extension_states[ext_name] = state

        # Delegate to state helper
        if hasattr(self.extensions["state"], "build_export_bundle"):
            getattr(self.extensions["state"], "build_export_bundle")(
                knowledge_id=self.agent_uuid,
                output_path=output_path,
                extension_states=extension_states,
            )

    def import_knowledge(self, bundle_path: str) -> str:
        """
        Imports a knowledge package from a .gyro bundle.

        Args:
            bundle_path: Path to the .gyro bundle file.

        Returns:
            The UUID of the imported knowledge package.
        """
        # Unpack bundle
        if not hasattr(self.extensions["state"], "unpack_import_bundle"):
            raise GyroExtensionError("State helper missing or incomplete")

        new_knowledge_id = getattr(self.extensions["state"], "unpack_import_bundle")(bundle_path)

        # Link session to imported knowledge
        self.link_to_knowledge(new_knowledge_id)

        return new_knowledge_id

    def fork_knowledge(self, new_session: bool = False) -> str:
        """
        Forks the current knowledge package.

        Args:
            new_session: If True, creates a new session for the fork.
                        If False, links current session to the fork.

        Returns:
            The UUID of the forked knowledge package.
        """
        # Delegate to fork manager
        if not hasattr(self.extensions["fork"], "fork"):
            raise GyroExtensionError("Fork manager missing or incomplete")

        new_knowledge_id = getattr(self.extensions["fork"], "fork")()

        if not new_session:
            # Link current session to the fork
            self.link_to_knowledge(new_knowledge_id)

        return new_knowledge_id

    def link_to_knowledge(self, knowledge_id: str) -> None:
        """
        Links the current session to a different knowledge package.

        Args:
            knowledge_id: The UUID of the knowledge package to link to.
        """
        # Save current state before switching
        self.shutdown(persist_only=True)

        # Update knowledge reference
        self.agent_uuid = knowledge_id
        if hasattr(self.extensions["state"], "switch_knowledge_context"):
            getattr(self.extensions["state"], "switch_knowledge_context")(knowledge_id)

        # Reset phase to 0 per CORE-SPEC-05
        if self.engine:
            self.engine.load_phase(0)

        # Create and load new navigation log for the linked knowledge
        if hasattr(self.extensions["state"], "load_ui_state"):
            self.navigation_log = G2_Memory(agent_uuid=self.agent_uuid)

        # Update other extensions as needed
        state_helper = self.extensions.get("state")
        if state_helper and hasattr(state_helper, "update_knowledge_link"):
            getattr(state_helper, "update_knowledge_link")(knowledge_id)

        # Reload extension states
        self._load_extension_states()

        # This forces a reload of all knowledge-dependent state
        self._load_session_state()

    def shutdown(self, persist_only: bool = False) -> None:
        """
        Shuts down the session, persisting state.

        Args:
            persist_only: If True, only persist state without full shutdown.
        """
        # Persist all state through extensions
        if not hasattr(self.extensions["state"], "save_extension_data"):
            return

        # Create a final state bundle
        state_bundle = {"extensions": {}}
        for name, ext in self.extensions.items():
            if hasattr(ext, "get_learning_state"):
                state = getattr(ext, "get_learning_state")()
                if state:
                    state_bundle["extensions"][name] = state

        # Save state bundle
        getattr(self.extensions["state"], "save_extension_data")(
            self.agent_uuid, "learning_state.json", state_bundle
        )

        # Persist navigation log
        # Comment out or stub: self.navigation_log.persist_to_disk()

        if persist_only:
            return

        # Shutdown extensions in reverse order
        for ext in reversed(list(self.extensions.values())):
            if hasattr(ext, "shutdown"):
                try:
                    getattr(ext, "shutdown")()
                except Exception:
                    pass  # Best effort

    # ========================================================================
    # INTERNAL HELPER METHODS
    # ========================================================================

    def _get_event_tensor(self, temporal: str) -> list[object]:
        """Helper to get event tensor based on temporal specifier."""
        if temporal == "recent":
            # Use navigation log for recent event access
            if self.navigation_log:
                return [step for step, _ in self.navigation_log.iter_steps()]
            else:
                return []
        elif temporal == "complete":
            # This is expensive and should be used sparingly
            if self.navigation_log:
                return [step for step, _ in self.navigation_log.iter_steps()]
            return []
        else:
            raise GyroTagError(f"Invalid temporal specifier: {temporal}")

    def _get_nest_tensor(self, temporal: str) -> list[object]:
        """Helper to get nest tensor based on temporal specifier."""
        if temporal == "recent":
            # Use navigation log for recent event access
            if self.navigation_log:
                return [nest for _, nest in self.navigation_log.iter_steps()]
            else:
                return []
        elif temporal == "complete":
            if self.navigation_log:
                return [nest for _, nest in self.navigation_log.iter_steps()]
            return []
        else:
            raise GyroTagError(f"Invalid temporal specifier: {temporal}")

    def _get_decoded_gene_state(self, _temporal: str) -> Mapping[str, object]:
        """
        Decode navigation log to reconstruct current gene state.
        Updated to use new NavigationLog iterator API.
        """
        if not self.engine:
            return {}

        # Start with base gene - make a copy to avoid modifying original
        result = {}
        if hasattr(self.engine.gene["id_0"], "clone") and hasattr(
            self.engine.gene["id_1"], "clone"
        ):
            result = {
                "id_0": self.engine.gene["id_0"].clone(),
                "id_1": self.engine.gene["id_1"].clone(),
            }

        # Apply all navigation events using NEW iterator
        if self.navigation_log:
            for op_code_0, op_code_1 in self.navigation_log.iter_steps():
                # Apply transformations
                result["id_0"] = gyration_op(result["id_0"], op_code_0, clone=False)
                result["id_1"] = gyration_op(result["id_1"], op_code_1, clone=False)

        return result

    def get_recent_events(self, count: int, reverse: bool = False) -> list[tuple[int, int]]:
        """Returns a list of the most recent navigation event tuples."""
        if not self.navigation_log:
            return []
        return list(self.navigation_log.iter_steps(reverse=reverse))[:count]

    def _load_extension_states(self) -> None:
        """Load saved states for all extensions."""
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, "load_state"):
                try:
                    getattr(ext, "load_state")()
                except Exception:
                    if "error" in self.extensions:
                        error_handler = self.extensions["error"]
                        if hasattr(error_handler, "log_extension_error"):
                            getattr(error_handler, "log_extension_error")(
                                ext_name, "State load failed"
                            )

    def _get_user_key(self) -> bytes:
        """Get or generate user encryption key"""
        # Try to get from environment or config
        key_hex = os.environ.get("GYROSI_USER_KEY")

        if key_hex:
            try:
                return bytes.fromhex(key_hex)
            except ValueError:
                pass

        # Try to get from structural memory
        try:
            key = self.gyro_structural_memory("current.user_key")
            if isinstance(key, bytes) and len(key) >= 16:
                return key[:32]
        except Exception:
            pass

        # Generate default key (should be replaced with proper key management)
        default = b"default_user_key_replace_me"
        return hashlib.sha256(default).digest()

    def _handle_language_output(self, text: str) -> None:
        """Handle language output from the system"""
        # Log the output
        if self._output_handler:
            self._output_handler(text)
        else:
            # Default console output
            print(f"[GyroSI Output]: {text}")

        # Store in session for UI retrieval
        if hasattr(self.extensions["state"], "append_output"):
            getattr(self.extensions["state"], "append_output")(self.agent_uuid, text)

    def export_session(self, output_path: str) -> None:
        """
        Exports the current session to a .session.gyro bundle.
        Args:
            output_path: The file path to save the bundle.
        """
        if hasattr(self.extensions["state"], "export_session"):
            getattr(self.extensions["state"], "export_session")(self.agent_uuid, output_path)

    def import_session(self, bundle_path: str) -> str:
        """
        Imports a session from a .session.gyro bundle.
        Args:
            bundle_path: Path to the .session.gyro bundle file.
        Returns:
            The new session UUID.
        """
        if hasattr(self.extensions["state"], "import_session"):
            return getattr(self.extensions["state"], "import_session")(bundle_path)
        return ""

    def _coordinate_pipeline(self, nav_byte: int, input_byte: int) -> None:
        """Wire the dynamic codec and language pipeline together."""
        # Get required extensions
        dynamic_codec = self.extensions.get("dynamic_codec")

        # Process through dynamic codec extension (universal converter)
        if dynamic_codec and hasattr(dynamic_codec, "process_navigation_event"):
            getattr(dynamic_codec, "process_navigation_event")(nav_byte, input_byte)

    def replay_token_stream(self, token_stream: bytes) -> None:
        """
        Replay a token stream through the dynamic codec.

        Args:
            token_stream: Encrypted token stream from storage
        """
        # Get required extensions
        dynamic_codec = self.extensions.get("dynamic_codec")
        crypto = self.extensions.get("crypto")

        if not dynamic_codec or not crypto:
            raise RuntimeError("Dynamic codec or cryptographer missing")

        # Decrypt token stream
        if hasattr(crypto, "decrypt"):
            plain = getattr(crypto, "decrypt")(token_stream)

            # Decode into navigation cycles
            if hasattr(dynamic_codec, "decode_tokens"):
                cycles = getattr(dynamic_codec, "decode_tokens")(plain)

                # Process each cycle
                for cycle in cycles:
                    for nav_byte in cycle:
                        # Apply navigation event to engine state
                        # This reconstructs the original state
                        _ = self.gyro_operation(nav_byte)


# ============================================================================
# PUBLIC API of this Module
# ============================================================================
__all__ = [
    "GyroOperation ",
]
