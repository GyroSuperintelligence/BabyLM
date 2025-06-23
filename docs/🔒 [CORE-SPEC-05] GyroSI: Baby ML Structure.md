- 🔒 [CORE-SPEC-05] GyroSI: Baby ML Structure
    
    ## Introduction:
    
    The on-disk layout of GyroSI Baby ML is the direct physical manifestation of the ontological architecture defined by the five universal structural invariants (`gyrotensor_id`, `gyrotensor_com`, `gyrotensor_nest`, `gyrotensor_add`, `gyrotensor_quant`) operating through the four-phase cycle (CS→UNA→ONA→BU). This structure reflects the principled separation between learned intelligence (exportable knowledge) and usage context (session-specific ephemera).
    
    The layout embodies the core principle: intelligence emerges from the oscillation between the emergent Gene (G4 - Somatic Memory) and the emergent Genome (G5 - Immunity Memory), with knowledge packages as immutable, shareable artifacts and sessions as mutable, personal contexts.
    
    ## Foundational Principles
    
    The structure is governed by the ontological reality of the five invariants with principled knowledge/session separation:
    
    1. **gyrotensor_id**: Phase index (session-local, non-exportable)
    2. **gyrotensor_com**: Event logging (session events vs. learning events)
    3. **gyrotensor_nest**: UI state (session-specific, non-exportable)
    4. **gyrotensor_add (Gene)**: Invariant structure (code constant, validated via checksum)
    5. **gyrotensor_quant (Genome)**: Navigation log (exportable knowledge core)
    
    Knowledge packages contain only the learned intelligence; sessions contain only the usage context.
    
    ## Complete Directory Structure
    
    https://github.com/GyroSuperintelligence/BabyLM
    
    ```
    https://github.com/GyroSuperintelligence/BabyLM/
    ├── .git/                                   # Git repository
    │
    ├── docs/                                   # Project documentation
    │   ├── README.md                           # Main documentation
    │   ├── CORE-SPEC-01.md                     # Architecture and Principles
    │   ├── CORE-SPEC-02.md                     # Foundations (Core Mechanics)
    │   ├── CORE-SPEC-03.md                     # Architecture Mapping
    │   ├── CORE-SPEC-04.md                     # Language & Grammar
    │   ├── implementation.md                   # Implementation guide
    │   ├── knowledge-management.md             # Knowledge export/import guide
    │   ├── api/                                # API documentation
    │   └── examples/                           # Usage examples
    │
    ├── src/                                    # Core implementation
    │   ├── __init__.py
    │   ├── core/                               # Unified core reflecting ontological reality
    │   │   ├── __init__.py
    │   │   ├── gyro_core.py                    # All five memory systems + complete cycle
    │   │   ├── alignment_nav.py                # NavigationLog class (navigation log management)
    │   │   └── gyro_tag_parser.py              # TAG expression validation
    │   │
    │   ├── extensions/                         # Self-regulating extension framework
    │   │   ├── __init__.py
    │   │   ├── base.py                         # GyroExtension with checkpoint/restore
    │   │   ├── ext_bloom_filter.py             # 0-byte footprint, Gene substrate
    │   │   ├── ext_coset_knowledge.py          # Variable footprint, compression tracking
    │   │   ├── ext_multi_resolution.py         # 3-byte footprint, boundary detection
    │   │   ├── ext_spin_piv.py                 # 3-byte footprint, PIV evolution
    │   │   ├── ext_error_handler.py            # Error hierarchy and handling
    │   │   ├── ext_storage_manager.py          # File I/O and persistence
    │   │   ├── ext_state_helper.py             # State management utilities
    │   │   ├── ext_navigation_helper.py        # Navigation cycle utilities
    │   │   ├── ext_api_gateway.py              # External API interface
    │   │   ├── ext_system_monitor.py           # Health and validation
    │   │   ├── ext_performance_tracker.py      # Performance metrics
    │   │   ├── ext_fork_manager.py             # Fork-on-write management
    │   │   ├── ext_event_classifier.py         # Event classification
    │   │   ├── ext_phase_controller.py         # Phase advancement strategies
    │   │   └── ext_resonance_processor.py      # Structural resonance
    │   │
    │   ├── frontend/                           # UI components (G3 I/O Transducer)
    │   │   ├── __init__.py
    │   │   ├── gyro_app.py                     # Main application
    │   │   ├── components/
    │   │   │   ├── __init__.py
    │   │   │   ├── gyro_threads_panel.py
    │   │   │   ├── gyro_chat_interface.py
    │   │   │   └── gyro_document_upload.py
    │   │   └── assets/                         # UI assets
    │   │       ├── icons/
    │   │       ├── styles/
    │   │       └── fonts/
    │   │
    │   └── main.py                             # Entry point
    │
    ├── tests/                                  # Test suite
    │   ├── __init__.py
    │   ├── conftest.py                         # Pytest configuration and fixtures
    │   ├── test_core/                          # Core system tests
    │   │   ├── __init__.py
    │   │   ├── test_gyro_core.py               # Test all memory systems
    │   │   ├── test_alignment_nav.py           # Test NavigationLog
    │   │   ├── test_tag_parser.py              # Test TAG validation
    │   │   └── test_integration.py             # Integration tests
    │   │
    │   ├── test_extensions/                    # Extension tests
    │   │   ├── __init__.py
    │   │   ├── test_bloom_filter.py
    │   │   ├── test_coset_knowledge.py
    │   │   ├── test_multi_resolution.py
    │   │   └── test_spin_piv.py
    │   │
    │   ├── test_knowledge/                     # Knowledge management tests
    │   │   ├── __init__.py
    │   │   ├── test_export_import.py
    │   │   ├── test_knowledge_forking.py
    │   │   └── test_session_linking.py
    │   │
    │   ├── test_frontend/                      # UI tests
    │   │   ├── __init__.py
    │   │   ├── test_gyro_app.py
    │   │   └── test_components.py
    │   │
    │   ├── test_tools/                         # Tool tests
    │   │   ├── __init__.py
    │   │   ├── test_cli.py
    │   │   └── test_knowledge_manager.py
    │   │
    │   └── fixtures/                           # Test data and fixtures
    │       ├── sample_knowledge/
    │       ├── sample_sessions/
    │       └── test_genomes/
    │
    ├── data/                                   # Separated knowledge and session data
    │   ├── .gitkeep                            # Keep directory in git
    │   │
    │   ├── knowledge/                          # EXPORTABLE: Pure learned intelligence
    │   │   └── <knowledge_uuid>/               # Immutable knowledge packages
    │   │       ├── navigation_log/             # Core learned experience (gyrotensor_quant)
    │   │       │   ├── genome.log              # Primary navigation log
    │   │       │   ├── shards/                 # Collision-free distributed sharding
    │   │       │   │   ├── nav_00000000_<host_id>.bin
    │   │       │   │   └── nav_00000001_<host_id>.bin
    │   │       │   └── manifest.json           # Navigation log metadata
    │   │       │
    │   │       ├── extensions/                 # Extension-learned patterns
    │   │       │   ├── ext_coset_knowledge@0.9.1.patterns
    │   │       │   ├── ext_bloom_filter@1.2.0.patterns
    │   │       │   ├── ext_multi_resolution@1.0.0.boundaries
    │   │       │   └── ext_spin_piv@0.8.8.evolution
    │   │       │
    │   │       └── knowledge.meta.json         # Knowledge package metadata
    │   │
    │   └── sessions/                           # NON-EXPORTABLE: Usage context
    │       └── <session_uuid>/                 # Session-specific ephemera
    │           ├── active_knowledge.link       # UUID text file pointing to knowledge
    │           ├── phase.bin                   # Current session phase (0-47)
    │           ├── events.log                  # Session-specific events
    │           │
    │           ├── ui_state/                   # UI state (gyrotensor_nest projection)
    │           │   ├── threads.sqlite3         # Conversation threads (WAL mode, 5s timeout)
    │           │   └── folders.sqlite3         # Thread organization (WAL mode, 5s timeout)
    │           │
    │           └── session.meta.json           # Session metadata
    │
    ├── gyro_tools/                             # Tools leveraging built-in capabilities
    │   ├── __init__.py
    │   ├── gyro_cli.py                         # Main CLI interface
    │   ├── gyro_knowledge_manager.py           # Knowledge export/import/fork operations
    │   ├── gyro_session_manager.py             # Session management
    │   └── gyro_integrity_check.py             # System validation (read-only)
    │
    ├── config/                                 # Configuration
    │   ├── gyro_config.yaml                    # Main configuration (default log sizes: 1MB)
    │   └── extensions.config.yaml              # Extension footprint validation config
    │
    ├── logs/                                   # System logs (separate from agent memory)
    │   ├── .gitkeep                            # Keep directory in git
    │   ├── system.log                          # General system events
    │   ├── integrity.log                       # Built-in validation results
    │   └── extensions.log                      # Extension compliance events
    │
    ├── scripts/                                # Development and deployment scripts
    │   ├── setup_dev.py                        # Development environment setup
    │   ├── run_tests.py                        # Test runner with coverage
    │   ├── build_release.py                    # Release build script
    │   ├── benchmark.py                        # Performance benchmarking
    │   └── migrate_sessions.py                 # Session migration utility
    │
    ├── .gitignore                              # Git ignore patterns
    ├── .gitattributes                          # Git attributes
    ├── .editorconfig                           # Editor configuration
    ├── .pre-commit-config.yaml                 # Pre-commit hooks
    ├── pyproject.toml                          # Python project configuration
    ├── requirements.txt                        # Main dependencies
    ├── README.md                               # Project overview and quick start
    ├── LICENSE                                 # Project license
    ├── CHANGELOG.md                            # Version history
    └── Makefile                                # Common development tasks
    ```
    
    ## Knowledge Package Structure (Immutable, Exportable)
    
    ### Navigation Log (gyrotensor_quant)
    
    **Location**: `data/knowledge/<uuid>/navigation_log/`**Characteristics**:
    
    - Core learned experience from navigation patterns
    - Immutable once created (new learning creates new package)
    - Collision-free sharding for distributed operation
    - Complete provenance and integrity tracking
    
    ### Extension Patterns (Learned Intelligence)
    
    **Location**: `data/knowledge/<uuid>/extensions/`**Format**: `ext_<name>@<version>.<type>`**Examples**:
    
    - `ext_coset_knowledge@0.9.1.patterns` - Semantic compression patterns
    - `ext_bloom_filter@1.2.0.patterns` - Pattern recognition state
    - `ext_multi_resolution@1.0.0.boundaries` - Linguistic boundary patterns
    - `ext_spin_piv@0.8.8.evolution` - Cryptographic evolution state
    
    ### Knowledge Metadata
    
    **Location**: `data/knowledge/<uuid>/knowledge.meta.json`**Format**:
    
    ```json
    {
      "knowledge_id": "550e8400-e29b-41d4-a716-446655440000",
      "gyro_version": "0.8.8",
      "gene_checksum": "sha256:c1a7b3d9e0f1...",
      "step_count": 123456,
      "created_ts": 1750531200.0,
      "source": "session 79e2... on host A4F9E1C8",
      "parent_knowledge_id": null,
      "learning_sources": ["document_hash_1", "document_hash_2"],
      "extension_versions": {
        "ext_coset_knowledge": "0.9.1",
        "ext_bloom_filter": "1.2.0"
      }
    }
    
    ```
    
    ## Session Structure (Mutable, Non-exportable)
    
    ### Active Knowledge Link
    
    **Location**: `data/sessions/<uuid>/active_knowledge.link`**Format**: Plain text file containing knowledge UUID
    **Purpose**: Points to which knowledge package this session uses
    
    ### Session Phase (gyrotensor_id)
    
    **Location**: `data/sessions/<uuid>/phase.bin`**Characteristics**:
    
    - 4-byte integer (0-47 range)
    - Session-local, resets per session
    - Tracks current position in navigation cycle
    
    ### Session Events (gyrotensor_com subset)
    
    **Location**: `data/sessions/<uuid>/events.log`**Characteristics**:
    
    - Session-specific events (UI interactions, system events)
    - NOT learning events (those go to knowledge package)
    - Automatic rotation, non-exportable
    
    ### UI State (gyrotensor_nest)
    
    **Location**: `data/sessions/<uuid>/ui_state/`**Characteristics**:
    
    - Personal conversation organization
    - Thread and folder structures
    - User interface preferences
    - Session-specific, non-exportable
    
    ## Knowledge Management Operations
    
    ### Export Knowledge
    
    ```bash
    gyro export-knowledge --knowledge-id <uuid> --output package.gyro
    
    ```
    
    **Creates**: Compressed bundle containing:
    
    - Complete navigation log with all shards
    - All extension patterns with version compatibility
    - Knowledge metadata with full provenance
    - Gene checksum for integrity validation
    
    ### Import Knowledge
    
    ```bash
    gyro import-knowledge --input package.gyro [--new-session]
    
    ```
    
    **Process**:
    
    1. Validates Gene checksum compatibility
    2. Checks extension version compatibility
    3. Creates new knowledge package with fresh UUID
    4. Optionally creates new session linked to knowledge
    5. Preserves complete provenance chain
    
    ### Fork Knowledge
    
    ```bash
    gyro fork-knowledge --source <knowledge_uuid> [--session <session_uuid>]
    
    ```
    
    **Process**:
    
    1. Hard-links existing navigation shards (immutable)
    2. Copies knowledge metadata with new UUID
    3. Sets parent_knowledge_id for provenance
    4. Creates fresh genome.log for new learning
    5. Links session to forked knowledge
    
    ### Link Session to Knowledge
    
    ```bash
    gyro link-session --session <session_uuid> --knowledge <knowledge_uuid>
    
    ```
    
    **Process**:
    
    1. Updates active_knowledge.link file
    2. Validates knowledge package exists
    3. Resets session phase to 0
    4. Preserves existing UI state
    
    ## Immutability Contract
    
    ### Knowledge Package Immutability
    
    - Every `data/knowledge/<uuid>` directory is **write-once**
    - New navigation events trigger automatic forking
    - Original packages remain unchanged for integrity
    - Forking creates new UUID with parent reference
    
    ### Fork-on-Write Behavior
    
    When a session with active knowledge receives new navigation events:
    
    1. System automatically forks the knowledge package
    2. Creates new UUID for the forked knowledge
    3. Hard-links existing immutable shards
    4. Starts fresh navigation log for new learning
    5. Updates session's `active_knowledge.link` to new UUID
    6. Preserves complete provenance chain
    
    ### Concurrent Learning Protection
    
    - Multiple sessions can read from same knowledge package
    - First session to learn triggers fork, others continue with original
    - Race conditions avoided through atomic directory operations
    - Each learning path gets unique knowledge UUID
    
    ## Memory System Projections with Knowledge/Session Separation
    
    ### G1 (Genetic Memory) - Unified Invariant Access
    
    **Role**: Provides unified access to all invariants across knowledge and session boundaries
    **Implementation**: Routes TAG queries to appropriate storage (knowledge vs session)
    **Gene Access**: Always returns code constant, validated against knowledge package checksum
    
    ### G2 (Epigenetic Memory) - Dual Event Streams
    
    **Role**: Manages both learning events (→ knowledge) and session events (→ session)
    **Learning Events**: High-value events that contribute to intelligence (stored in knowledge)
    **Session Events**: UI interactions, system events (stored in session)
    **Routing Logic**: Extensions declare which events are "learning-worthy"
    
    ### G3 (Structural Memory) - Session-Local I/O
    
    **Role**: Manages UI state and I/O boundaries (session-specific only)
    **Storage**: Always in session directory
    **Characteristics**: Personal, non-exportable, session-isolated
    
    ### G4 (Somatic Memory) - Session-Local Phase
    
    **Role**: Tracks navigation phase within current session
    **Storage**: Session-specific phase counter
    **Behavior**: Resets when session links to different knowledge
    
    ### G5 (Immunity Memory) - Knowledge-Centric Navigation
    
    **Role**: Manages navigation log in knowledge packages
    **Write Behavior**: Triggers fork-on-write for immutability
    **Read Behavior**: Accesses current knowledge package via session link
    
    ## Extension Integration with Knowledge/Session Split
    
    ### Extension State Classification
    
    Extensions must declare which state is:
    
    - **Learning State**: Patterns, models, accumulated intelligence (→ knowledge)
    - **Session State**: Temporary caches, UI preferences (→ session)
    
    ### Extension State Storage
    
    ```python
    class GyroExtension:
        def get_learning_state(self) -> dict:
            """State that should be exported with knowledge"""
            pass
    
        def get_session_state(self) -> dict:
            """State that stays with session"""
            pass
    
        def set_learning_state(self, state: dict):
            """Restore learning state from knowledge package"""
            pass
    
        def set_session_state(self, state: dict):
            """Restore session state"""
            pass
    
    ```
    
    ### Extension Pattern Files
    
    **Naming Convention**: `ext_<name>@<version>.<type>`**Storage**: Knowledge package for learning patterns, session for temporary state
    **Versioning**: Prevents incompatible state import
    
    ## File Format Specifications
    
    ### Knowledge Package Bundle (.gyro)
    
    **Format**: Compressed tar archive
    **Structure**:
    
    ```
    package.gyro/
    ├── knowledge.meta.json          # Package metadata
    ├── navigation_log/              # Complete navigation history
    │   ├── genome.log
    │   ├── shards/
    │   └── manifest.json
    ├── extensions/                  # Extension learning patterns
    │   ├── ext_coset_knowledge@0.9.1.patterns
    │   └── ext_bloom_filter@1.2.0.patterns
    └── integrity.sha256             # Complete package checksum
    
    ```
    
    ### Active Knowledge Link
    
    **Format**: Plain text file containing UUID
    **Example**: `550e8400-e29b-41d4-a716-446655440000`**Behavior**: Atomic updates for session switching
    
    ### Knowledge Metadata Schema
    
    ```json
    {
      "knowledge_id": "550e8400-e29b-41d4-a716-446655440000",
      "gyro_version": "0.8.8",
      "gene_checksum": "sha256:c1a7b3d9e0f1a2b3c4d5e6f7...",
      "step_count": 123456,
      "created_ts": 1750531200.0,
      "source": "session 79e2f1a8... on host A4F9E1C8",
      "parent_knowledge_id": "440e8400-e29b-41d4-a716-446655440001",
      "fork_reason": "new_learning_session",
      "learning_sources": [
        "document_sha256:abc123...",
        "conversation_thread:def456..."
      ],
      "extension_versions": {
        "ext_coset_knowledge": "0.9.1",
        "ext_bloom_filter": "1.2.0",
        "ext_multi_resolution": "1.0.0",
        "ext_spin_piv": "0.8.8"
      },
      "compatibility": {
        "min_gyro_version": "0.8.0",
        "max_gyro_version": "0.9.x"
      }
    }
    
    ```
    
    ## Operational Scenarios
    
    ### Individual Learning
    
    1. User starts new session with empty knowledge
    2. System creates initial knowledge package
    3. User interacts, system learns, navigation log grows
    4. Knowledge package remains linked to session
    5. User can export learned intelligence anytime
    
    ### Knowledge Sharing
    
    1. User A exports knowledge package after learning
    2. User B imports package, creates new session
    3. User B continues learning from User A's knowledge
    4. System forks knowledge on first new learning event
    5. Both users have independent learning paths with shared ancestry
    
    ### Collaborative Learning
    
    1. Team shares knowledge package as starting point
    2. Multiple team members import same knowledge
    3. Each member works in separate sessions
    4. Members can export and share incremental learning
    5. Knowledge packages maintain provenance chains
    
    ### Knowledge Evolution
    
    1. Base knowledge package created from initial learning
    2. Specialized versions forked for different domains
    3. Successful patterns merged back to base knowledge
    4. Complete evolution tree maintained through parent references
    5. Any point in evolution can be restored and continued
    
    ## Updated Development Infrastructure
    
    ### .gitignore
    
    ```
    # Python
    __pycache__/
    *.py[cod]
    *$py.class
    *.so
    .Python
    build/
    develop-eggs/
    dist/
    downloads/
    eggs/
    .eggs/
    lib/
    lib64/
    parts/
    sdist/
    var/
    wheels/
    *.egg-info/
    .installed.cfg
    *.egg
    
    # Virtual environments
    venv/
    env/
    ENV/
    .venv/
    
    # IDE
    .vscode/settings.json
    .idea/
    *.swp
    *.swo
    *~
    
    # GyroSI specific - sessions and logs only
    data/sessions/*/
    logs/*.log
    config/local.yaml
    
    # Keep knowledge packages in git for examples
    !data/knowledge/example_*/
    
    # OS
    .DS_Store
    Thumbs.db
    
    # Testing
    .pytest_cache/
    .coverage
    htmlcov/
    .tox/
    
    # Documentation
    docs/_build/
    
    ```
    
    ### Makefile
    
    ```makefile
    .PHONY: help install test lint format clean run dev export-example import-example
    
    help:
    	@echo "Available commands:"
    	@echo "  install         Install dependencies"
    	@echo "  test           Run tests"
    	@echo "  lint           Run linting"
    	@echo "  format         Format code"
    	@echo "  clean          Clean build artifacts"
    	@echo "  run            Run the application"
    	@echo "  dev            Run in development mode"
    	@echo "  export-example Export example knowledge"
    	@echo "  import-example Import example knowledge"
    
    install:
    	pip install -r requirements/dev.txt
    	pre-commit install
    
    test:
    	python -m pytest
    
    lint:
    	flake8 src tests
    	mypy src
    
    format:
    	black src tests
    	isort src tests
    
    clean:
    	find . -type d -name __pycache__ -delete
    	find . -type f -name "*.pyc" -delete
    	rm -rf build/ dist/ *.egg-info/
    	rm -rf .pytest_cache/ .coverage htmlcov/
    
    run:
    	python src/main.py
    
    dev:
    	python src/main.py --dev
    
    export-example:
    	python -m gyro_tools.gyro_knowledge_manager export --knowledge-id example --output examples/example_knowledge.gyro
    
    import-example:
    	python -m gyro_tools.gyro_knowledge_manager import --input examples/example_knowledge.gyro --new-session
    
    ```
    
    ## Conclusion: Knowledge-Centric Architecture
    
    This structure provides the complete on-disk organization for GyroSI Baby ML, directly translating the ontological architecture into a practical, robust file system layout. The unified core implementation reflects the true simplicity of the GyroSI model, while the knowledge/session separation enables clean export of learned intelligence independent of usage context.
    
    The structure accommodates all deployment scenarios from single-user development to distributed production environments, with built-in mechanisms for integrity validation, collision-free operation, and bounded resource growth. The complete development infrastructure ensures the project can be built, tested, and deployed following modern software engineering practices while preserving the theoretical foundations of the GyroSI architecture.