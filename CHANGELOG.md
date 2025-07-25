# CHANGELOG

Here is a focused and accurate **changelog summary** of all critical changes and confirmations you made:

---

## [0.9.6.4] – 2025-07-26

**New Auto-Pruning Functionality** 

An auto-pruning functionality has been fully implemented with the following components:

# Changelog for Auto‑Pruning Feature

## Added

* **Preferences schema**

  * Introduced a new `"pruning"` section in `memories/memory_preferences.json`:

    ```json
    "pruning": {
      "confidence_threshold": 0.05,
      "decay_factor": 0.995,
      "decay_interval_hours": 6,
      "enable_auto_decay": true
    }
    ```
* **`PreferencesConfig.pruning`**

  * Extended `baby/contracts.py` to include a `pruning: Dict[str, Any]` field so agents can read all pruning settings from their config.
* **Auto‑pruning hook**

  * In `IntelligenceEngine.__init__`, register `_auto_prune_hook` if `preferences["pruning"]["enable_auto_decay"]` is `true`.
  * Implemented `_auto_prune_hook()` to:

    1. Read `confidence_threshold` from preferences.
    2. Call `InferenceEngine.prune_low_confidence_entries(threshold)`.
    3. Gracefully handle append‑only and view‑layer stores (catch and ignore non‑deletable errors).
    4. If > 10 000 entries were “removed,” invoke `prune_and_compact_store()` in‑place to do a full compaction.
* **`AgentConfig.preferences`**

  * Extended `AgentConfig` to accept a `preferences` sub‑dict and pass it through `GyroSI → IntelligenceEngine`.
* **`CanonicalView.commit()`**

  * Added a `commit()` method to `CanonicalView` so the auto‑pruner’s initial `commit()` call always succeeds on view‑wrapped stores.

## Changed

* **`InferenceEngine.prune_low_confidence_entries()`**

  * Now always calls `store.commit()` first to flush pending writes.
  * Wraps `store.delete(key)` in `try/except NotImplementedError/RuntimeError` so overlay and read‑only views don’t crash.
  * Removed fallback `del store.data[key]` for non‑append‑only stores (views use their own `.delete()`).
* **`GyroSI` constructor**

  * Now reads `config["preferences"]` and passes it into `IntelligenceEngine`.
* **`AgentPool`**

  * Propagates its `preferences` into each newly created agent’s `AgentConfig`, ensuring hooks get wired automatically.

## Tests

* **Extended `test_inference.py`**

  * Verified `prune_low_confidence_entries()` behavior for both deletable and append‑only stores.
  * Added tests for:

    * Hook registration when `enable_auto_decay` is `true` vs. `false`.
    * Hook execution (ensuring it doesn’t blow up on append‑only or overlay stores).
    * Custom vs. default thresholds.
* **All existing pruning and compact tests** continue to pass unmodified.


---

## [0.9.6.4] – 2025-07-25

### **Phase 1**
**Scope:** Tests, Tests, Tests - Corrections, Corrections, Corrections... 

- Flake8, Pyright, Mypy Error Free
- Pyright Robust Agent Isolation finally achieved! No littering or polution to our main data (somehow this silly thing proved to be a heck of a challenge!)
- Pyright Pass

Here is a clean, structured changelog entry summarising the full refactor:

---

### **Phase 2**
**Scope:** Ontology / Epistemology / Phenomenology / Theta

#### ✅ **Summary**

We removed all runtime JSON parsing from the engine and replaced the entire memory model with compact `.npy` binaries using NumPy + `mmap`. This creates a single source of truth for each of the four internal maps and reduces startup time from \~140 s to < 0.3 s per engine instance.

### Major Refactor: Full Migration to Binary `.npy` Assets

**Core changes:**
- **Ontology, Epistemology, and Phenomenology assets** are now stored and loaded exclusively as `.npy` files (`ontology_keys.npy`, `epistemology.npy`, `phenomenology_map.npy`). All legacy `.json`-based logic, schema, and code paths have been removed.
- **InformationEngine** now requires four file paths (`keys_path`, `ep_path`, `phenomap_path`, `theta_path`) and only supports array-based indexing. All dict-based and JSON-based overloads, as well as `use_array_indexing` and `strict_validation` parameters, are gone.
- **All CLI, build, and workflow instructions** have been updated to reference the new `.npy` filenames and arguments.
- **All tests and fixtures** have been updated to use the new four-path constructor for `InformationEngine`. All references to removed features (`use_array_indexing`, `strict_validation`, `endogenous_modulus`, `ontology_diameter`) have been removed or replaced.
- **Validation and maintenance utilities** now operate on `.npy` files and check array properties, not JSON keys.
- **Early failure for missing/corrupt `theta.npy`:** `InformationEngine` now raises at construction if `theta.npy` is missing or corrupt, rather than deferring to the first divergence calculation.
- **All error messages, logs, and comments** referencing `.json` assets for ontology/phenomenology have been updated or removed.
- **Dead code and comments** (e.g., ujson/json import fallbacks) have been removed for clarity.
- **Type safety:** All code and tests have been updated to pass `mypy` and `pyright` with no ignores, including correct handling of optional types and array lengths.

**Other improvements:**
- **Test suite**: All tests now use the correct `.npy`-based API, and obsolete tests for removed features have been deleted.
- **CI and Git LFS**: Workflow and LFS tracking updated to only include `.npy` assets.
- **Documentation**: All build instructions and module-level banners now reference the correct binary asset workflow.

---

**Summary:**  
The codebase is now fully binary-asset based, with a modern, type-safe, and maintainable API. All legacy JSON, dict, and fallback logic is gone, and the developer experience is consistent and robust for both contributors and CI.

---

## [0.9.6.4] – 2025-07-24

**Scope:** Performance, Storage, and Runtime Optimizations

#### ✅ **Storage Architecture**

* Replaced **gzip-compressed multi‑file store** with a **single `.mpk` file** using **msgpack**:

  * Set `store_options = {"append_only": True}` in the GyroSI agent config.
  * Removed `use_msgpack`, `.log`, and `.idx` files — now obsolete.
  * All training knowledge is streamed into one compact, append‑only `.mpk` file.

#### ✅ **Batch Learning Performance**

* Integrated **Numba JIT** acceleration for hot learning loop:

  * Added `_jit_batch` method (Numba-compiled) to replace the slow Python loop.
  * Automatically invoked when the STT (epistemology map) is loaded.
  * Performance improved from **1–2 KB/sec → 40–60 MB/sec** on Intel Mac.
  * Training now behaves as originally theorized: a **fast, deterministic converter** from text to internal knowledge.

#### ✅ **Compiler Compatibility (macOS‑specific)**

* Verified Numba + LLVM compatibility for Intel MacBook Pro:

  * Uses Homebrew’s `llvm` to ensure full `llvmlite` support.
  * Explicit `CC` and `CXX` environment variables documented for stable builds.

#### ✅ **Filesystem and Checkpoints**

* All changes work seamlessly with **pause/resume**:

  * `Ctrl+Z` to suspend, `fg` to resume.
  * Async and atomic checkpoints preserved.
  * Checkpoint format unchanged.

#### ✅ **Requirements Updated**

* `requirements.txt` updated:

  * Pinned `numba==0.60.*`, `llvmlite==0.60.*` for macOS stability.
  * Replaced compression and pickle dependencies with `msgpack==1.1.*`.
  * Ensured Python 3.10 compatibility across packages.

---

**Net Effect:**
Training now hopefully will run at hardware‑limited throughput. Storage is portable, readable, and consistent with the GyroSI theory. No artificial bottlenecks remain between your ontology–phenomenology–epistemology pipeline and the disk.


---

## [0.9.6.3] – 2025-07-23

#### 🚀 Added

* **Mandatory 3‑mind bootstrap (`user`, `system`, `assistant`)**

  * `AgentPool.ensure_triad()` creates (and guarantees presence of) the canonical trio.
  * New `AgentPool.get()` to fetch an agent *without* creating it (raises `KeyError` if missing).
  * `AgentPool.create_agent()` explicit creation API when you *do* want a new id.

* **Creation policy controls**

  * `AgentPool.__init__` now accepts:

    * `allowed_ids: set[str]` – whitelist of ids permitted when `allow_auto_create=False`.
    * `allow_auto_create: bool` – gate automatic creation of arbitrary ids.
    * `private_agents_base_path: str` – base dir for private agent stores.

* **Path override plumbing**

  * `GyroSI._create_default_store()` honors `private_agents_base_path` and `base_path`.
  * All file/folder creation under `memories/private/agents/` can be redirected via config (great for tests).

#### 🔧 Changed

* **`orchestrate_turn`** now *requires* that agents already exist; otherwise it raises with a helpful message.

* **External FastAPI adapter (`toys/communication/external_adapter.py`)**

  * Uses the shared pool with `ensure_triad()`.
  * Maps all external users to the internal `"user"` id by default.
  * No silent auto-creation of “user2”, “assistant42”, etc.

* **Store construction**

  * Default CanonicalView enabling logic kept, but paths are fully configurable.
  * OverlayView still used for public/private knowledge, but private paths respect overrides.

#### 🧪 Tests

We authored and landed the full test suite below, and everything is green as of today. The codebase is also clean under `flake8`, `pyright`, and `mypy` (zero errors, zero warnings).

**toys/health/conftest.py**
Session‑scoped and per‑test fixtures to isolate all artefacts in temporary directories. Provides ready‑to‑use `GyroSI`, `AgentPool`, `OrbitStore`, and helper assertions for ontology/phenotype validity. Ensures no pollution of shared meta files and auto‑cleans temp state.

**toys/health/test\_governance.py**
Exhaustive checks for the physics layer (`governance.py`): constants, bit masks, and tensor structure; governance signature maths; Monodromic Fold properties (identity, absorber, annihilation, non‑commutativity/associativity); dual operator involution; 48‑bit transform bounds; batch consistency; transcription XOR/involution; tensor validation routines; and assorted edge cases.

**toys/health/test\_inference.py**
Covers the interpretation layer (`inference.py`). Verifies initialisation with real ontology data, phenotype creation/retrieval, confidence maths, governance signatures, learning via fold (single and batch), decay and pruning operations, integrity validation, and private utility behaviour. Includes error paths (bad indices, malformed entries) and store integration.

**toys/health/test\_information.py**
Targets `information.py`: tensor↔int conversion (round‑trips, boundaries, error handling), state/index lookups in both dict and array modes, gyrodistance/divergence calculations, orbit cardinality handling, phenomenology integration, mmap utilities, and data consistency of the ontology map.

**toys/health/test\_intelligence.py**
End‑to‑end and integration tests for `intelligence.py` and the external FastAPI adapter. Exercises batch learning, hook batching, agent lifecycle and persistence, multi‑agent isolation in `AgentPool`, tokenizer round‑trips and UTF‑8 fallback, OpenAI/HF compatible endpoints (including SSE streaming), concurrency safety, and full conversation pipelines.

**Result**

* All tests pass locally today (23 July 2025).
* Lint/static analysis: `flake8`, `pyright`, and `mypy` report no issues.

No further action required for this cycle.

---

## [0.9.6.3] – 2025-07-22

### Major Refactor, Optimization, and Cleanup

- **Core Measurement Optimization:**
  - Replaced tensor-based θ calculation with fast XOR+bit_count+LUT approach.
  - Auto-generate `theta.npy` if missing for robust operation.

- **OrbitStore and Storage:**
  - Changed `pending_writes` to dict for O(1) access.
  - Added time-based `fsync` fuse and `mark_dirty` for in-memory updates.
  - Optimized log replay and switched to `msgpack` serialization.
  - Implemented mmap remap gating to reduce overhead on frequent commits.

- **Conversational Loop and AgentPool:**
  - Batched egress/ingress in `GyroSI.respond` for fewer store writes.
  - Refactored `AgentPool` to use sharded pools for concurrency.
  - Cached public read-only store and ensured single close.

- **Tokenizer Bridge:**
  - Optimized LEB128 encoding, added module-level cache, documented overflow guard.

- **Knowledge Decay and Pruning:**
  - Buffered policy changes and wrote once at end, leveraging dict buffer.

- **Phenomenology and State Sync:**
  - Added singleton loader for phenomenology map.
  - Cached `state_int` for θ in `IntelligenceEngine`.

- **Hooks and Regulatory Logic:**
  - Batched hook processing with ring buffer, cached mean θ in `process_ingress`.

- **API and Streaming:**
  - Added HTTP keep-alive and streaming for `/v1/chat/completions`.

- **Naming and Config Clarity:**
  - Renamed `batch_size` in `AgentConfig` to `learn_batch_size` and in `PreferencesConfig` to `write_batch_size`.

- **Testing, Docs, and Cleanups:**
  - Ensured all store-like objects use efficient `iter_entries()`.
  - Removed deprecated aliases and unused code.
  - Fixed linter and indentation errors.
  - Updated test configs and documentation for clarity and consistency.

---

## [0.9.6.2] – 2025-07-21

### Added

* **Tokenizer integration layer** (`toys/communication/tokenizer.py`):
  Implements reversible text-to-byte encoding via HuggingFace tokenizers using LEB128 encoding. All tokens are encoded as ≤255 bytes to remain compatible with GyroSI’s physics layer.
  Supports encoding/decoding via pretrained models (e.g. `bert-base-uncased`) stored under `memories/public/tokenizers/`.

* **Tokenizer setup script** (`toys/communication/setup_tokenizers.py`):
  Downloads and installs HuggingFace tokenizers into the shared public memory path. Currently installs `bert-base-uncased`.

* **Tokenizer training stub** (`toys/communication/train_tokenizer.py`):
  Provides a scaffolding for training custom WordPiece tokenizers on domain-specific corpora. Outputs are saved in the same shared tokenizer directory structure.

* **Mandatory tokenizer protocol in orchestration** (`baby/intelligence.py`):
  `orchestrate_turn` now requires a `tokenizer_name` argument. All text processing is routed through the tokenizer bridge; UTF-8 fallback and mixed encoding are no longer supported. This enforces a clean, consistent protocol and prevents knowledge contamination.

* **Extended agent configuration support** (`baby/contracts.py`):
  `AgentConfig` now includes `tokenizer_name` and `tokenizer_mode` (values: `"input"`, `"io"`, or `None`) for agent-specific encoding strategies.

* **Adapter and REST integration** (`toys/communication/external_adapter.py`):
  - All text encoding uses the tokenizer bridge and the default tokenizer is configurable via environment variable.
  - All calls to `orchestrate_turn` pass the required `tokenizer_name`.
  - Adapter path logic now uses `Path(__file__).resolve().parents[1]` for robust, CWD-independent resolution.

* **Tokenizer and REST tests** (`toys/health/test_tokenizer.py`):
  - Validates round-trip encoding/decoding, byte safety, and vocabulary size.
  - Adds a REST façade integration test: starts the FastAPI app, POSTs to `/generate`, and asserts a 200 response, using fixtures to avoid data litter.

* **Maintenance utilities**:
  - **`prune_and_compact_store`** (`baby/policies.py`): Prunes and compacts an `OrbitStore` in one pass, with support for age/confidence thresholds, dry-run, and archiving pruned entries.
  - **CLI script** (`toys/maintenance/prune_compact.py`): Command-line tool to prune/compact stores, with flexible options.
  - **Exported** `prune_and_compact_store` in `baby/__init__.py` for easy import.

### Notes

* All text encoding is now strictly via the tokenizer bridge; UTF-8 fallback is removed from orchestration and adapter layers.
* All test and adapter data is isolated using fixtures and temporary directories to prevent data litter.
* Public tokenizers are shared under `memories/public/tokenizers/` and can be reused by any agent.
* The system retains full deterministic replay and physics compliance.

---

## [0.9.6.2] – 2025-07-20

### ✅ **Changelog: Algebra, Learning, and Structural Corrections**

**1. Finalised the Monodromic Fold (`fold`)**

* Canonicalised the learning operation as:
  `fold(a, b) = a ^ (b ^ (a & ~b))`
* This form satisfies:

  * Non-associativity
  * Non-commutativity
  * Left-identity (`0 ⋄ b = b`)
  * Right-absorption (`a ⋄ 0 = 0`)
* All prior variants (e.g. OR-based `coadd`) were removed as physically invalid.

**2. Unified Egress and Ingress under One Operator**

* Removed artificial distinction between input (`Egress`) and output (`Ingress`) operators.
* Both processes now use the **same Monodromic Fold**, applied in opposite directions.
* Path dependence and non-associativity are preserved in both directions.

**3. Phenotype Learning Logic Clarified**

* Confirmed that **repeated learning with the same intron toggles memory\_mask** (x ↔ 0).
* This behavior is intentional and expresses **monodromic closure**, not cumulative accretion.
* Docstrings updated to explain this self-annihilating mechanism clearly.

**4. CanonicalView Bug Fixed**

* `context_signature` was mistakenly stripped in phenotype entries.
* Fixed: `context_signature` is retained; only `_original_context` may be removed safely.
* Prevents `KeyError` during learning and inference.

**5. Storage Durability Improvement**

* Added optional `fsync.result()` wait in `commit()` for guaranteed flush during tests.
* Prevents race conditions when asserting durability after write.

**6. Confirmed Map Validity**

* The `epistemology` and `ontology` maps were checked and found internally consistent with the Monodromic Fold.
* No regeneration required.

**7. Designed Physical Non-Associativity Experiment**

* Prepared a plan to empirically test physical path dependence using your actual state transition maps.
* Confirms that associativity violations are not algebraic artifacts, but grounded in state evolution.

**8. Genetics**
* Introduced Exons and Refined our Genetics assosiations.
* Changed our Phenotype's metadata contracts.
* Refined our Semantic Framework.

**8. Pytest Corrections & Results**

* Passed all 132 tests

---

## [0.9.6.2] – 2025-07-17 to 19

- Wrote the code for all:
baby/contracts.py
baby/governance.py
baby/inference.py
baby/information.py
baby/intelligence.py
baby/policies.py

- wrote the tests:
toys/health/conftest.py
toys/health/test_governance.py
toys/health/test_inference.py
toys/health/test_information.py
toys/health/test_intelligence.py
toys/health/test_miscellaneous.py

Here's a concise changelog entry capturing the essence of that addition:

---

**Added**: `toys/communication/external_adapter.py` — a FastAPI-based external adapter exposing GyroSI through industry-standard REST interfaces.

* Implements **OpenAI-compatible** endpoints (`/v1/models`, `/v1/chat/completions`) and **HuggingFace-style** generation (`/generate`).
* Connects to the internal `baby.intelligence` engine without modification; operates via `AgentPool` and `orchestrate_turn`.
* Manages three distinct agents per session (system/user/assistant) with consistent ID handling and memory bootstrapping logic.
* Enables seamless external integration without altering core physics or learning logic.

---

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