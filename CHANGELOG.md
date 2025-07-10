# Changelog

All notable changes to this project will be documented in this file.

## [0.9.6] – 2025-07-10
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