All notable changes to this project will be documented in this file.

[0.9.4] – 2025-Jul-07

- **Sharded Storage:** Replaced monolithic JSON with two-level hex sharding for all agent, thread, key, and format objects. Deterministic O(1) path computation for reads.
- **Registry Files:** Each shard now maintains a `registry.json` index for immediate children, with automatic updates and crash recovery helpers.
- **Atomic, Concurrent Writes:** All writes use atomic temp files and `flock`-protected registries for safe concurrent access.
- **Agent, Thread, and Key Management:**
  - Automatic RFC 4122 agent UUID generation and sharding.
  - Per-thread metadata JSONs and encrypted payloads/keys, with deterministic lookup.
- **Format Management:**
  - Public, sharded formats with support for pattern-distance matrices and recursive listing/loading helpers.
- **Module Boundaries:**
  - `information.py`: storage, sharding, atomic writes, registry, UUID, thread/key I/O.
  - `inference.py`: pattern-matching, Epigenome mutations, genome mask loading.
  - `intelligence.py`: orchestration, thread lifecycles, encryption, context-aware response.
- **Type Safety & Test Reliability:**
  - Introduced TypedDicts for metadata, enforced type-correct assignments, and safe dictionary access.
  - All tests pass; codebase is type- and lint-clean.

[0.9.4] – 2025-Jul-06
### Added
🧠 Complete GyroSI Baby LM AI system architecture rewrite
- **S1 Governance Layer**: Core system governance and coordination
- **S2 Information Layer**: Data processing and tensor operations
- **S3 Inference Layer**: Epigenome management and keystream generation
- **S4 Intelligence Layer**: Pattern matching and output generation

🔧 Canonical tensor-to-byte conversion system
- Implemented spec-compliant `tensor_to_output_byte` function
- Pattern matching approach for deterministic output generation
- Removed non-canonical conversion methods

🧪 Comprehensive test suite
- Integration tests with improved fixtures
- Unit tests for all engine components
- Test data properly organized in designated test directories

### Changed
🔄 Refactored core engines for canonical approach
- `InformationEngine`: Updated for spec-compliant tensor operations
- `IntelligenceEngine`: Implemented pattern-based output generation
- `InferenceEngine`: Fixed epigenome initialization bug

### Fixed
🐛 Critical bug fixes
- Fixed all-zero keystream causing ciphertext to equal plaintext
- Corrected pattern index handling in tests
- Resolved Pyright type errors throughout codebase
- Fixed flake8 linting issues (unused imports, whitespace, bare except)

### Technical
📝 Code quality improvements
- Added explicit type annotations
- Removed duplicate type declarations
- Cleaned up unused imports and variables
- Replaced bare except with specific exception handling

[0.1.0] – 2025-06-22
### Added
🗂️ Initial project structure established

📁 Created base directories: src/, tests/, docs/, examples/

📄 Added s1_governance files: README.md, LICENSE, pyproject.toml, requirements.txt, Makefile, CHANGELOG.md

