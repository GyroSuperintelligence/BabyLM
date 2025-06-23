- 🔒 [CORE-SPEC-06] GyroSI: Baby ML Interface Definitions
    
    This chapter defines the minimal mechanical implementation contracts for GyroSI Baby ML, establishing precise interface specifications that enforce theoretical fidelity.
    
    ## Core Mechanical Contracts
    
    ### Navigation Contract
    
    ```python
    from typing import Optional, Iterator, Tuple, Dict, Any
    import torch
    
    class NavigationLog:
        """Append-only navigation log implementing gyrotensor_quant mechanics."""
    
        def append(self, id0_code: int, id1_code: int, *, fork_ok: bool = True) -> None:
            """
            Append navigation event for both Gene tensors.
            Thread-safe with advisory file lock.
    
            Args:
                id0_code: Operation code (0-3) for id_0 tensor
                id1_code: Operation code (0-3) for id_1 tensor
                fork_ok: If False, raise GyroImmutabilityError on immutable knowledge
    
            Raises:
                GyroPhaseError: If append violates navigation cycle constraints
                GyroImmutabilityError: If fork_ok=False and knowledge is immutable
                ValueError: If codes not in range 0-3
            """
            if not (0 <= id0_code <= 3) or not (0 <= id1_code <= 3):
                raise ValueError(f"Invalid codes: {id0_code}, {id1_code}. Must be 0-3.")
    
        def iter_steps(self, start: int = 0, stop: Optional[int] = None, reverse: bool = False) -> Iterator[Tuple[int, int]]:
            """Bounded iteration with snapshot semantics."""
            pass
    
        def checksum(self) -> str:
            """SHA256 checksum of navigation log."""
            pass
    
        @property
        def step_count(self) -> int:
            """Current number of navigation steps."""
            pass
    
        def create_shard(self, host_id: str) -> str:
            """Create collision-free shard: nav_<sequence>_<host_id>.bin"""
            pass
    
        def fork(self, new_knowledge_id: str) -> 'NavigationLog':
            """Fork for new learning path (fork-on-write)."""
            pass
    
    ```
    
    ### Memory System Interface
    
    ```python
    def gyro_genetic_memory(tag: str) -> Union[int, torch.Tensor, Dict[str, torch.Tensor], NavigationLog]:
        """
        Unified TAG-based access to all five invariants across G1-G5.
        Routes queries to appropriate storage (knowledge vs session).
    
        Args:
            tag: TAG expression (e.g., 'current.gyrotensor_add')
    
        Returns:
            - gyrotensor_id: int (phase counter 0-47)
            - gyrotensor_com: torch.Tensor (2×3 int8)
            - gyrotensor_nest: torch.Tensor (2×2×3 int8)
            - gyrotensor_add: Dict[str, torch.Tensor] (Gene with id_0, id_1)
            - gyrotensor_quant: NavigationLog instance
    
        Raises:
            GyroTagError: Invalid TAG expression
            GyroIntegrityError: Gene checksum validation fails
        """
        pass
    
    ```
    
    ### Operator Contracts
    
    ```python
    def gyro_operation() -> None:
        """Execute one resonance step. Exactly one operator activates."""
        pass
    
    def gyro_curation(gyrotensor_quant):
        """Stable operator (Left Inverse)."""
        pass
    
    def gyro_interaction(gyrotensor_quant):
        """Unstable operator (Forward Gyration)."""
        pass
    
    def gyro_cooperation(gyrotensor_quant):
        """Neutral operator (Backward Gyration)."""
        pass
    
    def gyration_op(tensor: torch.Tensor, code: int, clone: bool = True) -> torch.Tensor:
        """
        Apply gyration transformation to 4×2×3×2 int8 tensor.
    
        Args:
            code: 0=Identity, 1=Inverse, 2=Forward, 3=Backward
        """
        if tensor.shape != (4, 2, 3, 2):
            raise ValueError(f"Invalid shape: {tensor.shape}. Must be (4, 2, 3, 2).")
    
    ```
    
    ### TAG Parser
    
    ```python
    def parse(tag: str) -> Dict[str, str]:
        """Parse TAG expression: <temporal>.<invariant>[.<context>]"""
        pass
    
    def is_valid(tag: str) -> bool:
        """Validate TAG syntax."""
        pass
    
    ```
    
    ## Extension Framework
    
    ```python
    from abc import ABC, abstractmethod
    
    class GyroExtension(ABC):
        """Base contract for extensions operating through invariant substrate."""
    
        @abstractmethod
        def get_extension_name(self) -> str:
            """Canonical name (must start with 'ext_')."""
            pass
    
        @abstractmethod
        def get_extension_version(self) -> str:
            """Semantic version (e.g., '1.0.0')."""
            pass
    
        @abstractmethod
        def get_footprint_bytes(self) -> int:
            """Current memory footprint."""
            pass
    
        @abstractmethod
        def get_learning_state(self) -> Dict[str, Any]:
            """State for knowledge package export."""
            pass
    
        @abstractmethod
        def get_session_state(self) -> Dict[str, Any]:
            """Session-local state (non-exportable)."""
            pass
    
        @abstractmethod
        def set_learning_state(self, state: Dict[str, Any]) -> None:
            """Restore learning state."""
            pass
    
        @abstractmethod
        def set_session_state(self, state: Dict[str, Any]) -> None:
            """Restore session state."""
            pass
    
        @abstractmethod
        def process_navigation_event(self, event: torch.Tensor) -> Optional[torch.Tensor]:
            """Process navigation event, return optional learning contribution."""
            pass
    
        @abstractmethod
        def get_pattern_filename(self) -> str:
            """Pattern filename: ext_<name>@<version>.<type>"""
            pass
    
    ```
    
    ## Knowledge Management
    
    ```python
    def export_knowledge(knowledge_id: str, output_path: str) -> None:
        """Export knowledge package to .gyro bundle."""
        pass
    
    def import_knowledge(bundle_path: str, create_session: bool = False) -> str:
        """Import knowledge package, return UUID."""
        pass
    
    def fork_knowledge(source_id: str, session_id: Optional[str] = None) -> str:
        """Fork knowledge for new learning path."""
        pass
    
    def link_session(session_id: str, knowledge_id: str) -> None:
        """Link session to knowledge package."""
        pass
    
    def create_session(knowledge_id: Optional[str] = None) -> str:
        """Create new session, return UUID."""
        pass
    ```
    
    ## **Other Extensions**
    
    ### **ext_error_handler**
    
    ```python
    class ext_ErrorHandler(GyroExtension):
        """
        Centralized error handling and recovery strategies.
        FOOTPRINT: 10-20 bytes (error state cache)
        MAPPING: Intercepts all GyroError exceptions for logging/recovery
        """
        # Manages error hierarchy from gyro_core
        # Provides error recovery strategies
        # Maintains error history for debugging
    ```
    
    ### **ext_storage_manager**
    
    ```python
    class ext_StorageManager(GyroExtension):
        """
        All file I/O operations for knowledge and session persistence.
        FOOTPRINT: Variable (based on active file handles)
        MAPPING: Manages data/knowledge/ and data/sessions/ directories
        """
        # Handles all file operations removed from core
        # Manages atomic writes and file locking
        # Provides storage abstraction layer
    ```
    
    ### **ext_state_helper**
    
    ```python
    class ext_StateHelper(GyroExtension):
        """
        State management utilities and helpers.
        FOOTPRINT: 50-100 bytes (state cache)
        MAPPING: Provides state access patterns for other extensions
        """
        # Manages GyroState access patterns
        # Provides state snapshots and rollback
        # Handles state synchronization
    ```
    
    ### **ext_navigation_helper**
    
    ```python
    class ext_NavigationHelper(GyroExtension):
        """
        Navigation cycle utilities and helpers.
        FOOTPRINT: 20-30 bytes (navigation cache)
        MAPPING: Provides navigation patterns and utilities
        """
        # Common navigation patterns
        # Cycle validation utilities
        # Navigation history tracking
    ```
    
    ### **ext_api_gateway**
    
    ```python
    class ext_APIGateway(GyroExtension):
        """
        External API interface and request handling.
        FOOTPRINT: 100-200 bytes (request queue)
        MAPPING: Translates external requests to navigation events
        """
        # RESTful API endpoints
        # Request validation and routing
        # Response formatting
    ```
    
    ### **ext_system_monitor**
    
    ```python
    class ext_SystemMonitor(GyroExtension):
        """
        System health monitoring and validation.
        FOOTPRINT: 50-100 bytes (monitoring cache)
        MAPPING: Tracks and reports system health and diagnostics
        """
        # Monitors resource usage and health
        # Reports anomalies
    ```
    
    ### **ext_performance_tracker**
    
    ```python
    class ext_PerformanceTracker(GyroExtension):
        """
        Tracks extension and system performance metrics.
        FOOTPRINT: 100-200 bytes (metrics cache)
        MAPPING: Captures and logs performance statistics
        """
        # Captures timing and throughput data
        # Provides metrics to the operator
    ```
    
    ### **ext_fork_manager**
    
    ```python
    class ext_ForkManager(GyroExtension):
        """
        Fork-on-write knowledge management.
        FOOTPRINT: 20-30 bytes (fork state)
        MAPPING: Manages isolated forks for knowledge editing
        """
        # Handles knowledge versioning
        # Ensures fork consistency
    ```
    
    ### **ext_event_classifier**
    
    ```python
    class ext_EventClassifier(GyroExtension):
        """
        Event classification and tagging.
        FOOTPRINT: 10-15 bytes (event tags)
        MAPPING: Identifies and classifies extension events
        """
        # Tags and sorts events
        # Supports event-based filtering
    ```
    
    ### **ext_phase_controller**
    
    ```python
    class ext_PhaseController(GyroExtension):
        """
        Phase advancement strategies.
        FOOTPRINT: 4 bytes (phase state)
        MAPPING: Controls transition logic between phases
        """
        # Alternative phase advancement logic
        # Customizable progression criteria
    ```
    
    ### **ext_resonance_processor**
    
    ```python
    class ext_ResonanceProcessor(GyroExtension):
        """
        Structural resonance processing.
        FOOTPRINT: 8-12 bytes (resonance cache)
        MAPPING: Detects and processes structural resonance events
        """
        # Analyzes cyclic patterns
        # Modulates resonance response
    ```
    
    ### **ext_bloom_filter**
    
    ```python
    class ext_BloomFilter(GyroExtension):
        """
        0-byte footprint gene substrate.
        FOOTPRINT: 0 bytes
        MAPPING: Substrate for lightweight gene presence/absence checks
        """
        # Bloom filter operations
        # Ultra-compact state representation
    ```
    
    ### **ext_coset_knowledge**
    
    ```python
    class ext_CosetKnowledge(GyroExtension):
        """
        Variable footprint, compression tracking.
        FOOTPRINT: Variable
        MAPPING: Compression and coset-based knowledge structuring
        """
        # Knowledge compression strategies
        # Coset decomposition logic
    ```
    
    ### **ext_multi_resolution**
    
    ```python
    class ext_MultiResolution(GyroExtension):
        """
        3-byte footprint, boundary detection.
        FOOTPRINT: 3 bytes
        MAPPING: Detects boundaries across multiple resolutions
        """
        # Multiscale analysis
        # Detects granular boundaries
    ```
    
    ### **ext_spin_piv**
    
    ```python
    class ext_SpinPIV(GyroExtension):
        """
        3-byte footprint, PIV evolution.
        FOOTPRINT: 3 bytes
        MAPPING: Particle image velocimetry evolution tracking
        """
        # PIV-based analysis and reporting
        # Tracks motion evolution
    ```
    
    ## Error Hierarchy
    
    ```python
    class GyroError(Exception):
        """Base exception for all GyroSI errors."""
        pass
    
    class GyroTagError(GyroError):
        """TAG expression violations."""
        pass
    
    class GyroPhaseError(GyroError):
        """Navigation cycle constraint violations."""
        pass
    
    class GyroNoResonanceError(GyroError):
        """No operator resonance."""
        pass
    
    class GyroImmutabilityError(GyroError):
        """Knowledge modification attempts."""
        pass
    
    class GyroIntegrityError(GyroError):
        """Structural integrity failures."""
        pass
    
    class GyroExtensionError(GyroError):
        """Extension operation failures."""
        pass
    
    class GyroNavigationError(GyroError):
        """Navigation log operation failures."""
        pass
    
    class GyroForkError(GyroError):
        """Knowledge forking failures."""
        pass
    
    ```
    
    ## Memory System Projections
    
    - **G1 (Genetic)**: Unified invariant access via `gyro_genetic_memory(tag)`
    - **G2 (Epigenetic)**: Dual event streams (learning → knowledge, session → session)
    - **G3 (Structural)**: Session-local I/O boundaries and UI state
    - **G4 (Somatic)**: Phase counter (0-47) and structural resonance
    - **G5 (Immunity)**: Navigation log with fork-on-write immutability
    
    ## Navigation Encoding
    
    Each step encodes two 4-bit operations in one uint8:
    
    ```
    Byte: [id1_op:4][id0_op:4]
    4-bit: [b3][b2][b1][b0] = [op:3][id:1]
    
    Operations: 0=Identity, 1=Inverse, 2=Forward, 3=Backward
    Tensors: 0=id_0, 1=id_1
    
    ```
    
    ## Compliance Requirements
    
    1. **Naming**: Core functions `gyro_*`, extensions `ext_*`
    2. **TAG**: Only `previous|current|next.invariant[.context]`
    3. **Navigation**: Append-only, 4-bit encoding, fork-on-write
    4. **Extensions**: Footprint tracked, state classified
    5. **Immutability**: Knowledge packages never modified in-place
    
    ## Summary
    
    These mechanical contracts enforce navigation-based operation through structural resonance, append-only knowledge accumulation with immutability, TAG-based invariant access, and extension boundaries with footprint tracking. All functionality emerges from these primitives through the recursive navigation cycle.