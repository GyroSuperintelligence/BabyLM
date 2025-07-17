# CHANGELOG

## [0.9.6.2] – 2025-07-16

### Major Refactoring and Architecture Improvements
- **Storage Layer Consolidation**: OrbitStore is now the single canonical storage class. All overlays and canonicalization are handled via decorators (CanonicalView, OverlayView, ReadOnlyView). Legacy/duplicate storage classes and factories removed.
- **Async and Streaming Optimizations**: OrbitStore flush now uses async fsync (background thread). Batch learning uses O(1) memory streaming coaddition.
- **Protocol and Type Hygiene**: PhenotypeStore protocol and all shared types (PhenotypeEntry, ManifoldData, AgentConfig, etc.) are now in baby/contracts.py (renamed from types.py). All storage implementations are in baby/policies.py (renamed from maintenance.py).
- **Import and Packaging Consistency**: All imports now use absolute paths (from baby.*). No more relative imports for shared types or storage classes. Circular imports and shadowing issues resolved.
- **PEP8 and Linting**: All major linter errors fixed (unused imports/variables, blank lines, whitespace, long lines). Guidance provided for using black and autoflake for future formatting.
- **Error Diagnosis and Environment Guidance**: Diagnosed and provided solutions for persistent import errors, including shadowing, packaging, and cache issues. Provided shell commands and troubleshooting steps.

### Project Structure After Refactor

```
baby/
  contracts.py      # All protocols and shared types (PhenotypeStore, etc.)
  policies.py       # OrbitStore, CanonicalView, OverlayView, ReadOnlyView, and policy/maintenance functions
  information.py    # InformationEngine and related logic
  intelligence.py   # IntelligenceEngine, GyroSI, and orchestration logic
  inference.py      # Inference logic
  __init__.py       # Clean, canonical imports and __all__ for package API
  ...               # Other modules as needed
```

### Key Outcomes
- Single source of truth for all protocols and storage classes.
- No more circular imports, shadowing, or ambiguous imports.
- All code is PEP8-compliant and linter-friendly.
- Project is robust for both development and production.

## [0.9.6.1] – 2025-07-15

### **GyroSI Baby Language Model 0.9.6 (Conceptual & Architectural Refactor)**

This update represents a major conceptual and architectural refactoring of the GyroSI 0.9.6 specification. While the version number remains the same, the underlying theory, component architecture, and terminology have been significantly matured and clarified. The focus has shifted from an implementation-centric description to a physics-first framework, providing a more robust and scalable foundation.

**I. Major Architectural & Conceptual Refactoring**

1.  **Introduction of the Measured Manifold:**
    *   The system is now grounded in the **empirically measured and provably finite physical ontology** of precisely **788,986 unique states**. This replaces the previous, more abstract notion of a state space.
    *   The ontology's **diameter is a measured constant of 6**, meaning any state is reachable from any other in at most seven steps.
    *   This "measured ground truth" is now the cornerstone of the entire architecture, moving the system from "physics-inspired" to "physics-grounded".

2.  **VSM-Aligned Engine Architecture:**
    *   The four engines have been explicitly mapped to **Beer's Viable System Model (VSM)**, clarifying their roles and creating a recursive, self-regulating structure.
    *   **S1 Governance** is no longer an "engine" class but a set of pure, stateless functions and constants in `governance.py` (The Physics).
    *   **S2 Information Engine** is now solely responsible for measurement and storage coordination (`information.py`).
    *   **S3 Inference Engine** focuses on interpretation and meaning management (`inference.py`).
    *   **S4/S5 Intelligence Engine** handles orchestration, agent state, and the external API (`intelligence.py`).

3.  **Decoupled Storage via `PhenotypeStore` Interface:**
    *   The complex, bespoke file structure (`Gyronorm Formats`, `Gene Keys`, `Threads`) has been replaced by a clean, abstract **`PhenotypeStore` protocol**.
    *   This decouples the core physics from all persistence concerns, allowing for swappable storage backends (e.g., `PickleStore`, `MultiAgentPhenotypeStore`, or future database adapters).
    *   Knowledge is now stored in a `(context_key -> phenotype_entry)` mapping, where `context_key` is a `(tensor_index, intron)` tuple.

4.  **Formalized API and Integration Layer:**
    *   A dedicated **Core API and Integration** section has been added, defining a stable `GyroSI` class as the primary entry point.
    *   Introduced the **`AgentPool`** concept for managing multiple agents and orchestrating conversations through agent interaction, rather than specialized chat infrastructure.
    *   Provided a clear pattern for creating **Protocol Adapters**, with an example for an OpenAI-compatible API.

5.  **Canonicalization of Orbits:**
    *   A new, fundamental abstraction layer has been introduced: **Canonicalization**.
    *   A build-time process identifies a single canonical representative for each state orbit within the ontology.
    *   The **`CanonicalizingStore` decorator** ensures all physically equivalent states map to the same storage entry, improving data coherency and abstraction.

**II. Core Physics & Foundational Changes**

1.  **Formalized Gyrogroup Algebra:**
    *   Learning is no longer based on a heuristic `combined_score`. It is now defined by **true gyrogroup coaddition (`a ⊞ b`)**, a specific, path-dependent, non-commutative algebraic operation.
    *   This change introduces **Ordered Batching** as the canonical way to process multiple learning signals, preserving the structure of experience.

2.  **Refined Transformation Physics:**
    *   The core state transformation logic (`apply_gyration_and_transform`) now includes a physically correct **gyration memory (carry term)**, implemented as `final_state = temp_state ^ (temp_state & intron_pattern)`. This makes the transformation path-dependent in a more fundamental way.
    *   The physical effect of Forward/Backward Gyrations (FG/BG) is now specified to operate on entire **tensor layers** (0&2 / 1&3), which is a more precise definition than the previous "rows".

3.  **Canonical Integer State Representation:**
    *   The primary representation of the system's physical state (`GENE_Mac_M`) is now a **packed 48-bit integer**. The 48-byte NumPy tensor is used for geometric measurement but is secondary to the integer for state transitions. This has significant performance and storage benefits.

**III. Terminology & Naming Conventions**

A new, consistent naming scheme has been adopted to better reflect the system's physics.

*   **`GENE_*` Naming:**
    *   `GENE_Mac_S` (Stateless Macro Gene) replaces `gene_add`.
    *   `GENE_Mic_S` (Stateless Micro Gene) replaces `gene_stateless = 0xAA`.
    *   `GENE_Mac_M` (Mutated Macro Gene) replaces the "Epigenome Tensor" (`self.T`).
    *   `GENE_Mic_M` (Mutated Micro Gene) replaces `gene_mutated`.

*   **Conceptual Renaming:**
    *   The concepts of **"Exon"** (stateless archetype) and **"Intron"** (dynamic instruction) have been introduced. The `intron` is the 8-bit value derived from an input byte.
    *   `gyrodistance_angular` is the formal name for the measurement function.

**IV. Removed & Replaced Components**

*   **Removed: `Epigenome` and `Genome` Masks.**
    *   `epigenome.dat` (the 256 canonical patterns) is no longer used, as state is compared directly to the archetypal `GENE_Mac_S`.
    *   `genome.dat` (the output map) is replaced by the `phenotype` field within the `PhenotypeStore`.
*   **Removed: `Gyronorm Formats` and `Gene Keys`.**
    *   The complex JSON-based `format-<uuid>.json` files are gone. Semantic mapping is now handled by the simpler `phenotype_entry` dictionary in the `PhenotypeStore`.
    *   The `gene-<uuid>.ndjson` event logs are removed. Learning history is now implicitly captured in the state of the `PhenotypeStore`.
*   **Removed: `Thread` Files and Bespoke Encryption.**
    *   The structured `thread-uuid.ndjson.enc` files and their associated AES-256-GCM encryption model have been removed from the core spec. Content storage is now an application-level concern, separate from the knowledge store.
*   **Removed: Heuristic Response Generation.**
    *   The `_generate_response_byte` function with its `physical_score * semantic_score` logic is gone, replaced by the direct lookup in the `PhenotypeStore` after the physical state transition.

**V. New Features & Capabilities**

1.  **Maintenance & Operations Toolkit:**
    *   A new section details production-ready maintenance tools.
    *   Includes scripts/functions for **Confidence Decay** (`apply_confidence_decay`) to gracefully age out stale knowledge.
    *   Includes a **Map Merging Tool** (`merge_phenotype_maps`) for consolidating knowledge from multiple agents or sources.
2.  **Performance & Scaling Section:**
    *   A new section provides formal **performance characteristics**, including computational complexity, memory requirements, and throughput estimates.
3.  **Theoretical Appendix:**
    *   A new appendix explicitly maps GyroSI concepts to their corresponding principles in physics and mathematics, solidifying the theoretical foundation.

---

## [0.9.5] – 2025-07-12
- Refactored InferenceEngine and InformationEngine to support efficient batch processing with a new process_batch() method and updated process_stream() for fast-path batching.
- Created a high-performance bulk trainer script (toys/learning/trainer.py) for large curriculum files, supporting batch updates, periodic checkpointing, and multi-format learning.
- Added preference flags for batch size and optional Numba JIT compilation for further speedup.
- Established a clean, two-step curriculum workflow:
  1. Curriculum Serialization: Script (toys/learning/threads/wordnet_curriculum_threads.py) serializes the entire WordNet database into a flat text file for training. Output now goes to toys/learning/threads/corpus/wordnet_corpus.txt.
  2. Model Training: The trainer script consumes the generated corpus file for fast, scalable learning.
- Restored and updated all curriculum format and thread generator scripts in toys/learning/formats/ and toys/learning/threads/ to use correct namespace UUIDs and implement pattern index cycling (degeneracy).
- Ensured all scripts are runnable with PYTHONPATH=. for proper import resolution.
- Rewrote learning update logic so all loaded formats are updated for each winning pattern index, enabling true multi-format associative learning and correct handling of degeneracy.
- Fixed all pyright type errors related to string/bytes handling and added targeted type ignore comments where necessary.
- Moved all generated and output files to appropriate subdirectories to avoid clutter and maintain a clean project structure.

## [0.9.5] – 2025-07-11
- Enforced strict test isolation: all tests and engine code now use a dedicated test directory (`toys/health/memories/`).
- Standardized argument propagation: all helpers and engine methods now require and pass `prefs` and `base_memories_dir` as needed.
- Fixed critical bugs in `IntelligenceEngine` initialization and thread/gene key storage.
- Updated test mocks and assertions to match real function signatures, eliminating signature mismatch errors.
- Resolved all linter and static analysis issues; codebase is now clean and warning-free.
- Investigated and explained origins of stray test output and directory artifacts.
- Major performance and robustness improvements:
  - Added pattern matching cache to InferenceEngine for fast repeated lookups.
  - Implemented batch stream processing in InformationEngine with configurable batch size for efficient I/O.
  - Introduced robust, multi-process-safe registry caching with mtime-based invalidation.
  - Refactored PatternIndex to use defaultdict for cleaner and faster indexing.
  - Optimized IntelligenceEngine encode/decode logic with O(1) lookup maps supporting multiple patterns per character.
  - Simplified registry cache eviction logic for clarity and correctness.
- **Added new CLI suite under `toys/console/`:**
  - Interactive chat, dashboard, format viewer, and thread manager tools for BabyLM.
  - All CLI modules are type- and lint-clean, with robust error handling and safe type usage throughout.

- Refactored IntelligenceEngine to support multiple formats simultaneously, keyed by format_uuid.
- Updated all code and tests to use per-format access (self.formats[self.format_uuid]) instead of self.M.
- Fixed all test failures and ensured robust multi-format operation.
- Updated Genetics.md to clarify format access, pattern index cycling, and pattern distance matrix storage.
- Implemented stable UUIDs for public formats using uuid.uuid5 and a fixed namespace, enabling reproducible curriculum and format references.
- Fixed TypedDict access warnings for optional keys.

## [0.9.5] – 2025-07-10
- Refactored private gene key storage to use per-record encryption with length prefix for true append-only performance.
- Public thread metadata is now updated only at finalization, not per event, for better performance.
- Moved tensor_to_output_byte to InferenceEngine for architectural clarity.
- Fixed AES key length validation to require 32 bytes (256 bits) for AES-256.
- Added registry file locking in shard_path to prevent race conditions during sharding.
- Switched recent_patterns to collections.deque(maxlen=256) for efficient context tracking.
- PatternIndex.find_similar_contexts now caps locations checked for common patterns to avoid performance bottlenecks.
- Removed redundant cryptography imports and local JSON helpers.
- Replaced brittle thread UUID checks with robust file existence checks.
- Refactored ThreadMetadata to use children: List[ChildRef] instead of parallel child_uuids/child_names; updated all code and tests accordingly.
- Improved TypedDict access safety and fixed pyright linter errors throughout the codebase.
- Unified privacy logic using an explicit 'privacy' field ('public'/'private') for threads and gene keys
- Replaced legacy XOR encryption with robust AES-GCM encryption for private threads, using per-thread derived keys
- Clarified and retained 'agent_uuid' in gene keys for agent association/ownership (not privacy)
- Refactored thread/gene key creation, storage, and tests to use the new privacy model
- Updated all relevant code and tests for clarity, security, and future extensibility
- Major revision of Genetics.md: fully integrated Common Governance Model (CGM) theory, mapping all system components to CGM stages (CS, UNA, ONA, BU)
- Clarified and formalized the dual nature of BU (Egress/Recollection and Ingress/Closure) in both documentation and code
- Updated all terminology to remove analogies (e.g., "physicist/linguist"), using only precise CGM language
- Ensured the spec and implementation match: _generate_response_byte now documented and implemented as a two-stage ONA/BU process using combined resonance and confidence
- Rewrote learning mechanisms section to reflect that confidence directly influences generation (BU closure), and removed outdated "attention mechanism" text
- Added a comprehensive glossary of all key terms (Epigenome, Genome, Egress, Ingress, ONA, BU, etc.)
- Fixed Pyright and linter errors in intelligence.py and babylm.py (indentation, type safety, buffer handling)

### Fixed
- Fixed a critical state management bug in `IntelligenceEngine` that caused duplicate or excessive gene key writes in public mode. State buffers are now always cleared after finalization, preventing data leaks between sessions in both public and private modes.
- Fixed inconsistent and incorrect file path logic for public thread and gene key storage, ensuring all files are created and written to the correct sharded locations.
- Fixed issues where public NDJSON files were not being created or written due to file handle and path errors.

### Improved
- Modernized and strengthened the test suite for public thread and gene key storage, ensuring robust detection of subtle bugs and regressions.
- Unified file path calculation logic for all public thread operations, improving maintainability and reliability.
- Ensured state buffers are always cleared after finalization, preventing data leaks between sessions in both public and private modes.

## [0.9.5] – 2025-07-09

### Changed
🧠 **Major S3/S4 Refactor:**
- Inference (S3) is now purely resonance-based, with all statistical/contextual weighting removed
- S3 (physics) and S4 (intelligence/semantics) responsibilities are now cleanly separated
- S4 uses pattern metadata (`count`, `confidence`, etc.) for intelligent encoding, decoding, and generation
- Learning loop closed: resonance → confidence → generation → resonance
- Documentation and tests updated to reflect new architecture and learning behavior

## [0.9.5] – 2025-07-08

### Added
🌐 **Expanded Global Format Library**
- We have expanded our global format library! Formats are shared global knowledge and are available to all agents, though they do not contain contextual information. (Scripts Available at: toys/learning/formats)
  - **ASCII Curriculum:** 256 foundational ASCII characters
  - **Emoji Curriculum:** Over 5,000 Unicode emoji
  - **Mathematical Symbols Curriculum:** All Unicode mathematical symbols (excluding ASCII)
  - *(More curricula can be added as the system grows)*

### Changed
- Adopted NDJSON format for all gene keys and threads, supporting both public (unencrypted) and private (encrypted) storage for agent-agnostic knowledge sharing and curriculum learning
- Refactored IntelligenceEngine initialization to support full agent, read-only, and public/curation modes for flexible batch processing and knowledge sharing
- Integrated fast JSON parsing with orjson (and ujson fallback), using unified json_loads/json_dumps wrappers in all core modules for performance
- Ensured type safety and Pyright compliance throughout the codebase, especially for Optional[str] and None handling in public/curation mode
- Suppressed harmless linter warnings for optional imports (e.g., ujson)
- Improved overall codebase robustness, maintainability, and compatibility for both private and public agent operation

## [0.9.4] – 2025-07-07

### Added
- **Sharded Storage:** Replaced monolithic JSON with two-level hex sharding for all agent, thread, key, and format objects. Deterministic O(1) path computation for reads
- **Registry Files:** Each shard now maintains a `registry.json` index for immediate children, with automatic updates and crash recovery helpers
- **Atomic, Concurrent Writes:** All writes use atomic temp files and `flock`-protected registries for safe concurrent access
- **Agent, Thread, and Key Management:**
  - Automatic RFC 4122 agent UUID generation and sharding
  - Per-thread metadata JSONs and encrypted payloads/keys, with deterministic lookup
- **Format Management:**
  - Public, sharded formats with support for pattern-distance matrices and recursive listing/loading helpers
- **Module Boundaries:**
  - `information.py`: storage, sharding, atomic writes, registry, UUID, thread/key I/O
  - `inference.py`: pattern-matching, Epigenome mutations, genome mask loading
  - `intelligence.py`: orchestration, thread lifecycles, encryption, context-aware response
- **Type Safety & Test Reliability:**
  - Introduced TypedDicts for metadata, enforced type-correct assignments, and safe dictionary access
  - All tests pass; codebase is type- and lint-clean

## [0.9.3] – 2025-07-06

### Added
🧠 **Complete GyroSI Baby LM AI system architecture rewrite**
- **S1 Governance Layer**: Core system governance and coordination
- **S2 Information Layer**: Data processing and tensor operations
- **S3 Inference Layer**: Epigenome management and keystream generation
- **S4 Intelligence Layer**: Pattern matching and output generation

🔧 **Canonical tensor-to-byte conversion system**
- Implemented spec-compliant `tensor_to_output_byte` function
- Pattern matching approach for deterministic output generation
- Removed non-canonical conversion methods

🧪 **Comprehensive test suite**
- Integration tests with improved fixtures
- Unit tests for all engine components
- Test data properly organized in designated test directories

### Changed
🔄 **Refactored core engines for canonical approach**
- `InformationEngine`: Updated for spec-compliant tensor operations
- `IntelligenceEngine`: Implemented pattern-based output generation
- `InferenceEngine`: Fixed epigenome initialization bug

### Fixed
🐛 **Critical bug fixes**
- Fixed all-zero keystream causing ciphertext to equal plaintext
- Corrected pattern index handling in tests
- Resolved Pyright type errors throughout codebase
- Fixed flake8 linting issues (unused imports, whitespace, bare except)

### Technical
📝 **Code quality improvements**
- Added explicit type annotations
- Removed duplicate type declarations
- Cleaned up unused imports and variables
- Replaced bare except with specific exception handling

## [0.1.0] – 2025-06-22

### Added
🗂️ **Initial project structure established**
- 📁 Created base directories: src/, tests/, docs/, examples/
- 📄 Added s1_governance files: README.md, LICENSE, pyproject.toml, requirements.txt, Makefile, CHANGELOG.md