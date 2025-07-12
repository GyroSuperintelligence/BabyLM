# Changelog

## [0.9.7] – 2024-07-12
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

## [0.9.6] – 2025-07-11
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

## [0.9.6] – 2025-07-10
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

## [0.9.6] – 2025-07-09

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