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

References:
- CORE-SPEC-03: Architecture Mapping
- CORE-SPEC-04: Language & Grammar
- CORE-SPEC-05: Baby ML Structure
- CORE-SPEC-06: Baby ML Interface Definitions
- CORE-SPEC-07: Baseline Implementation Specifications
"""

import uuid
from typing import Optional, Any, Tuple, Dict, List
from pathlib import Path

# Tier 3 - The pure engine
from .gyro_core import GyroEngine, gyration_op
# Core data structures and errors
from .alignment_nav import NavigationLog
from .gyro_tag_parser import parse_tag, validate_tag
from .gyro_errors import (
    GyroError,
    GyroTagError,
    GyroIntegrityError,
    GyroSessionError,
    GyroExtensionError
)

# Import all required system extension classes from the extensions directory
from ..extensions import (
    ext_StorageManager,
    ext_ForkManager,
    ext_StateHelper,
    ext_EventClassifier,
    ext_ErrorHandler,
    ext_NavigationHelper,
    ext_SystemMonitor,
    ext_PerformanceTracker,
    # Application extensions
    ext_MultiResolution,
    ext_BloomFilter,
    ext_ParametricAnalyzer,
    ext_SpinPIV,
    ext_CosetKnowledge,
)


class ExtensionManager:
    """
    The orchestrator for a single, active GyroSI session.
    
    This class provides the unified interface to all five memory systems (G1-G5)
    and orchestrates the interaction between the pure GyroEngine and the
    extension ecosystem.
    """
    
    def __init__(self, session_id: Optional[str] = None, knowledge_id: Optional[str] = None):
        """
        Initializes the full GyroSI stack for a session.
        
        Args:
            session_id: The UUID of an existing session to resume. If None, creates new.
            knowledge_id: The UUID of a knowledge package to link to. If None, creates new.
            
        Raises:
            GyroIntegrityError: If system integrity checks fail during initialization.
            GyroSessionError: If session initialization fails.
        """
        # Generate IDs if not provided
        self._session_id = session_id or str(uuid.uuid4())
        self._knowledge_id = knowledge_id  # May remain None initially
        
        # Initialize extension registry
        self.extensions: Dict[str, Any] = {}
        
        try:
            # 1. Initialize system-critical extensions in dependency order
            self._initialize_system_extensions()
            
            # 2. Initialize application extensions
            self._initialize_application_extensions()
            
            # 3. Initialize the pure engine
            self.engine = GyroEngine()
            
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
        # Storage manager is the foundation - handles all I/O
        self.extensions['storage'] = ext_StorageManager(
            session_id=self._session_id,
            knowledge_id=self._knowledge_id
        )
        
        # Update knowledge_id if it was created by storage manager
        if not self._knowledge_id:
            self._knowledge_id = self.extensions['storage'].knowledge_id
        
        # State helper manages session/knowledge state coordination
        self.extensions['state'] = ext_StateHelper(
            storage_manager=self.extensions['storage']
        )
        
        # Fork manager handles knowledge immutability
        self.extensions['fork'] = ext_ForkManager(
            storage_manager=self.extensions['storage']
        )
        
        # Event classifier determines learning vs session events
        self.extensions['classifier'] = ext_EventClassifier()
        
        # Error handler for centralized error management
        self.extensions['error'] = ext_ErrorHandler()
        
        # Navigation helper for cycle utilities
        self.extensions['nav_helper'] = ext_NavigationHelper()
        
        # System monitor for health checks
        self.extensions['monitor'] = ext_SystemMonitor()
        
        # Performance tracker for metrics
        self.extensions['perf'] = ext_PerformanceTracker()
    
    def _initialize_application_extensions(self) -> None:
        """Initialize application-level extensions for pattern analysis."""
        # Multi-resolution linguistic boundary detection
        self.extensions['multi_res'] = ext_MultiResolution()
        
        # Bloom filter for pattern recognition
        self.extensions['bloom'] = ext_BloomFilter()
        
        # Parametric analyzer for advanced patterns
        self.extensions['parametric'] = ext_ParametricAnalyzer()
        
        # Spin-based PIV for cryptographic evolution
        self.extensions['spin_piv'] = ext_SpinPIV()
        
        # Coset knowledge for semantic compression
        self.extensions['coset'] = ext_CosetKnowledge()
    
    def _load_session_state(self) -> None:
        """Load session state through the state helper extension."""
        # Load phase and navigation log
        state = self.extensions['state'].load_session_state()
        
        # Set engine phase
        self.engine.load_phase(state['phase'])
        
        # Initialize navigation log
        self.navigation_log = NavigationLog(
            knowledge_id=self._knowledge_id,
            storage_manager=self.extensions['storage'],
            max_size=state.get('nav_log_max_size', 1_048_576)
        )
        self.navigation_log.load_from_disk()
        
        # Load extension states
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, 'load_state'):
                ext.load_state()
    
    def _validate_system_integrity(self) -> None:
        """Perform comprehensive system integrity validation."""
        # Validate Gene checksum
        if not self.extensions['storage'].validate_gene_checksum(self.engine.gene):
            raise GyroIntegrityError("Gene checksum validation failed")
        
        # Validate extension footprints
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, 'validate_footprint'):
                if not ext.validate_footprint():
                    raise GyroExtensionError(f"Extension {ext_name} footprint validation failed")
    
    def _cleanup_on_error(self) -> None:
        """Clean up any partially initialized state on error."""
        # Shutdown extensions in reverse order
        for ext in reversed(list(self.extensions.values())):
            if hasattr(ext, 'shutdown'):
                try:
                    ext.shutdown()
                except:
                    pass  # Best effort cleanup
    
    def get_extension(self, name: str) -> Any:
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
        return self._session_id
    
    def get_knowledge_id(self) -> str:
        """Returns the current knowledge package ID."""
        return self._knowledge_id
    
    # ========================================================================
    # G1-G5 CANONICAL MEMORY INTERFACES
    # ========================================================================
    
    def gyro_genetic_memory(self, tag: str, data: Any = None) -> Any:
        """
        G1: GyroAlignment through GyroTensor Management (Genetic Memory)
        
        Provides unified TAG-based access to all five invariants across G1-G5.
        Routes queries to appropriate storage (knowledge vs session).
        
        Args:
            tag: TAG expression per CORE-SPEC-04 grammar
            data: Optional data for write operations
            
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
        temporal = tag_parts['temporal']
        invariant = tag_parts['invariant']
        context = tag_parts.get('context')
        
        # Route based on invariant type
        if invariant == "gyrotensor_id":
            # Phase counter (session-local)
            if temporal == "current":
                return self.engine.phase
            elif temporal == "previous":
                return (self.engine.phase - 1) % 48
            elif temporal == "next":
                return (self.engine.phase + 1) % 48
                
        elif invariant == "gyrotensor_com":
            # Event log (2×3 tensor)
            return self._get_event_tensor(temporal)
            
        elif invariant == "gyrotensor_nest":
            # Nesting structure (2×2×3 tensor)
            return self._get_nest_tensor(temporal)
            
        elif invariant == "gyrotensor_add":
            # Gene (always returns constant)
            return self.engine.gene
            
        elif invariant == "gyrotensor_quant":
            # Navigation log (decoded state)
            return self._get_decoded_gene_state(temporal)
            
        else:
            raise GyroTagError(f"Unknown invariant: {invariant}")
    
    def gyro_epigenetic_memory(self, tag: str, data: Any = None) -> Any:
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
            if self.extensions['classifier'].is_learning_event(data):
                # High-value learning event
                self.extensions['storage'].store_learning_event(
                    self._knowledge_id, data
                )
            else:
                # Session-specific event
                self.extensions['storage'].store_session_event(
                    self._session_id, data
                )
        
        # Parse TAG for retrieval
        tag_parts = parse_tag(tag)
        temporal = tag_parts['temporal']
        
        # Return appropriate event data
        if temporal == "current":
            return self._get_recent_events(count=1)
        elif temporal == "previous":
            return self._get_recent_events(count=10)
        else:
            return None
    
    def gyro_structural_memory(self, tag: str, data: Any = None) -> Any:
        """
        G3: GyroInference through GyroTensor Interaction (Structural Memory)
        
        Manages session-local I/O boundaries and UI state.
        All G3 data is session-specific and non-exportable.
        
        Per CORE-SPEC-03 mapping:
        - G3_CS: Hardware Endpoints
        - G3_UNA: Data Endpoints
        - G3_ONA: Frontend Interface
        - G3_BU_In: User/System Input
        - G3_BU_Eg: System Output
        """
        # G3 primarily handles UI state through the storage manager
        tag_parts = parse_tag(tag)
        
        if tag_parts['invariant'] == "gyrotensor_nest":
            # UI state is stored in session
            return self.extensions['storage'].load_ui_state(self._session_id)
        
        return None
    
    def gyro_somatic_memory(self, tag: str, data: Any = None) -> Any:
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
        if "phase" in tag or tag_parts['invariant'] == "gyrotensor_id":
            return self.engine.phase
        
        # G4 also handles structural resonance info
        if "resonance" in tag:
            return self.extensions['nav_helper'].get_resonance_info()
        
        return None
    
    def gyro_immunity_memory(self, tag: str, data: Any = None) -> Any:
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
        
        if tag_parts['invariant'] == "gyrotensor_quant":
            # Return navigation log reference
            return self.navigation_log
        
        # G5 also handles operator info
        if "operator" in tag:
            return self.extensions['nav_helper'].get_operator_info()
        
        return None
    
    # ========================================================================
    # CORE ORCHESTRATION
    # ========================================================================
    
    def gyro_operation(self, input_byte: int) -> Optional[Tuple[int, int]]:
        """
        Orchestrates one complete, atomic navigation cycle.
        
        This is the heart of the system's operation, implementing the pipeline
        from ADR-001 and the complete CS→UNA→ONA→BU cycle from CORE-SPEC-01.
        
        Args:
            input_byte: The input byte to process (0-255).
            
        Returns:
            A tuple of (op_code_id0, op_code_id1) if navigation occurred, None otherwise.
        """
        try:
            # Track performance
            self.extensions['perf'].start_operation()
            
            # 1. Execute the pure cycle in the engine
            ops = self.engine.execute_cycle(input_byte)
            
            if not ops:
                # No resonance occurred
                self.extensions['perf'].end_operation(resonated=False)
                return None
            
            # 2. Delegate fork-on-write management
            # This ensures we always have a writable navigation log
            self.navigation_log = self.extensions['fork'].ensure_writable(
                self.navigation_log
            )
            
            # Update knowledge_id if fork occurred
            if self.navigation_log.knowledge_id != self._knowledge_id:
                self._knowledge_id = self.navigation_log.knowledge_id
                self.extensions['state'].update_knowledge_link(self._knowledge_id)
            
            # 3. Record the navigation event via the log object
            # Pack two 4-bit codes into one byte: [id_1:4][id_0:4]
            packed_byte = (ops[1] & 0x0F) << 4 | (ops[0] & 0x0F)
            self.navigation_log.append(packed_byte)
            
            # 4. Notify all listening extensions of the successful event
            self._notify_extensions(packed_byte, input_byte)
            
            # 5. Persist critical state
            self.extensions['state'].persist_phase(self.engine.phase)
            
            # Track successful operation
            self.extensions['perf'].end_operation(resonated=True)
            
            return ops
            
        except Exception as e:
            # Let error handler manage the exception
            self.extensions['error'].handle_error(e)
            raise
    
    def _notify_extensions(self, packed_nav: int, input_byte: int) -> None:
        """
        Notify all extensions with navigation event handlers.
        
        Args:
            packed_nav: The packed navigation byte (two 4-bit codes).
            input_byte: The original input byte that caused this navigation.
        """
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, 'ext_on_navigation_event'):
                try:
                    ext.ext_on_navigation_event(packed_nav, input_byte)
                except Exception as e:
                    # Log but don't fail the operation
                    self.extensions['error'].log_extension_error(ext_name, e)
    
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
        self.navigation_log.persist_to_disk()
        
        # Gather extension learning states
        extension_states = {}
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, 'get_learning_state'):
                state = ext.get_learning_state()
                if state:  # Only include non-empty states
                    extension_states[ext_name] = state
        
        # Delegate to storage manager
        self.extensions['storage'].build_export_bundle(
            knowledge_id=self._knowledge_id,
            output_path=output_path,
            extension_states=extension_states
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
        new_knowledge_id = self.extensions['storage'].unpack_import_bundle(
            bundle_path
        )
        
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
        new_knowledge_id = self.extensions['fork'].fork()
        
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
        self._knowledge_id = knowledge_id
        self.extensions['storage'].switch_knowledge_context(knowledge_id)
        
        # Reset phase to 0 per CORE-SPEC-05
        self.engine.load_phase(0)
        
        # Create new navigation log for the knowledge
        self.navigation_log = NavigationLog(
            knowledge_id=knowledge_id,
            storage_manager=self.extensions['storage']
        )
        self.navigation_log.load_from_disk()
        
        # Update session link
        self.extensions['state'].update_knowledge_link(knowledge_id)
        
        # Reload extension states
        self._load_extension_states()
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Returns comprehensive system health metrics.
        
        Returns:
            Dictionary of health metrics and status indicators.
        """
        return self.extensions['monitor'].get_health_report(
            phase=self.engine.phase,
            nav_log_size=self.navigation_log.step_count,
            knowledge_id=self._knowledge_id,
            session_id=self._session_id
        )
    
    def shutdown(self, persist_only: bool = False) -> None:
        """
        Ensures all state is persisted before the session is terminated.
        
        Args:
            persist_only: If True, only persists state without full shutdown.
        """
        # Persist navigation log
        if self.navigation_log:
            self.navigation_log.persist_to_disk()
        
        # Persist all extension states
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, 'persist_state'):
                try:
                    ext.persist_state()
                except Exception as e:
                    self.extensions['error'].log_extension_error(ext_name, e)
        
        # Persist session state
        self.extensions['state'].persist_all_state(
            phase=self.engine.phase,
            knowledge_id=self._knowledge_id
        )
        
        if not persist_only:
            # Full shutdown - cleanup extensions
            for ext in reversed(list(self.extensions.values())):
                if hasattr(ext, 'shutdown'):
                    try:
                        ext.shutdown()
                    except:
                        pass  # Best effort
    
    # ========================================================================
    # INTERNAL HELPER METHODS
    # ========================================================================
    
    def _get_event_tensor(self, temporal: str) -> Any:
        """Get gyrotensor_com (2×3 event tensor) for temporal reference."""
        # Implementation would construct tensor from recent events
        # For now, return placeholder
        import torch
        return torch.tensor([[-1, 1], [-1, 1], [-1, 1]], dtype=torch.int8)
    
    def _get_nest_tensor(self, temporal: str) -> Any:
        """Get gyrotensor_nest (2×2×3 nesting tensor) for temporal reference."""
        # Implementation would construct from UI state
        # For now, return placeholder
        import torch
        return torch.tensor([
            [[-1, 1], [-1, 1], [-1, 1]],
            [[1, -1], [1, -1], [1, -1]]
        ], dtype=torch.int8)
    
    def _get_decoded_gene_state(self, temporal: str) -> Dict[str, Any]:
        """
        Decode navigation log to reconstruct current gene state.
        
        This is computationally expensive and typically cached.
        """
        # Start with base gene
        result = {
            "id_0": self.engine.gene["id_0"].clone(),
            "id_1": self.engine.gene["id_1"].clone()
        }
        
        # Apply all navigation events
        for packed_byte in self.navigation_log.iter_steps():
            op_0 = packed_byte & 0x0F
            op_1 = (packed_byte >> 4) & 0x0F
            
            # Extract operator codes (bits 3:1)
            op_type_0 = (op_0 >> 1) & 0x07
            op_type_1 = (op_1 >> 1) & 0x07
            
            # Apply transformations
            result["id_0"] = gyration_op(result["id_0"], op_type_0, clone=False)
            result["id_1"] = gyration_op(result["id_1"], op_type_1, clone=False)
        
        return result
    
    def _get_recent_events(self, count: int) -> List[int]:
        """Get recent navigation events from the log."""
        events = []
        for event in self.navigation_log.iter_steps(reverse=True):
            events.append(event)
            if len(events) >= count:
                break
        return list(reversed(events))
    
    def _load_extension_states(self) -> None:
        """Load saved states for all extensions."""
        for ext_name, ext in self.extensions.items():
            if hasattr(ext, 'load_state'):
                try:
                    ext.load_state()
                except Exception as e:
                    self.extensions['error'].log_extension_error(ext_name, e)


# ============================================================================
# PUBLIC API of this Module
# ============================================================================
__all__ = [
    'ExtensionManager',
]