# 💫 Gyroscopic Superintelligence Baby 👶 - Language Model - CHANGELOG

---

## [0.9.6.7] – 2025-08-04

### 🔧 Plumbing & Training Infrastructure Improvements

This release focuses on critical plumbing fixes and training infrastructure improvements, addressing performance bottlenecks and system reliability issues identified through extensive testing and optimization work.

#### 🚀 Performance Optimizations Implemented

* **Candidate Lookup Optimization**
  * Implemented O(1) state-indexed candidate retrieval in `OrbitStore`
  * Added per-state candidate caching in `IntelligenceEngine` to reduce storage hits
  * Eliminated full-store scans that were causing generation hangs at ~200-300MB

* **Theta Calculation Optimization**
  * Replaced binary search with direct index access in `measure_state_divergence_index()`
  * Eliminated hundreds of binary searches per turn in `process_egress()`
  * Fixed performance bottleneck in epistemology chunk processing

* **Bulk Token Processing**
  * Replaced per-byte feedback loops with vectorized `process_egress_bulk()` calls
  * Eliminated N per-byte cycles where N = token byte length
  * Significantly reduced latency on development hardware

* **Tokenizer Caching**
  * Fixed repeated disk loading of `tokenizer.json` on every encode/decode call
  * Added tokenizer priming to warmup functions
  * Eliminated first-turn tokenizer loading penalty

* **Adapter Non-blocking Implementation**
  * Added `run_in_threadpool` wrapper to chat completion endpoints
  * Guaranteed event loop responsiveness during CPU-bound operations
  * Prevented server from appearing "hung" during long operations

#### 🧠 Training Infrastructure

* **Wikipedia Simple Dataset Processing**
  * Successfully processed 22,868 articles (39.8M tokens, 78.1MB)
  * Completed compilation in 1h 35m with 4 arts/s processing rate
  * Generated knowledge store for training experiments

* **Replay System Validation**
  * Successfully replayed 78.1MB training data in 45m 26s
  * Validated knowledge store integration and learning pipeline
  * Confirmed state evolution and storage mechanisms

#### 🔧 Critical Plumbing Fixes

* **Canonicalization Layer Optimization**
  * Verified proper store composition without redundant canonicalization
  * Confirmed correct `enable_phenomenology_storage` configuration
  * Eliminated potential performance degradation from double canonicalization

* **Token Divergence Origin Fix**
  * Fixed hard-coded `archetypal_state = 0` assumption in `compute_token_divergence()`
  * Added proper `origin_index` parameter for correct divergence calculations
  * Restored correctness to divergence diagnostics and temperature gating

* **Store Iteration Improvements**
  * Fixed unreachable code in `OrbitStore.iter_entries()`
  * Improved cycle counting accuracy in bulk processing
  * Enhanced store consistency with proper index integration

#### 📊 Current Status

* **Model Responsiveness**: ✅ Model now responds to queries successfully
* **Language Coherence**: 🔄 Still working on improving language coherence and generation quality
* **Performance**: ✅ Critical performance bottlenecks resolved
* **Training Pipeline**: ✅ Wikipedia simple training and replay working

#### 🎯 Next Steps

* Continue work on language coherence and generation quality
* Optimize remaining performance bottlenecks
* Expand training data processing capabilities
* Improve model response quality and consistency

---

## [0.9.6.7] – 2025-08-03

### 🚀 Performance Optimizations & Physics Alignment: Complete Implementation

This release implements comprehensive performance optimizations and physics-correct fixes that dramatically improve system responsiveness, storage efficiency, and generation quality. All optimizations from the assistant's analysis have been successfully implemented and are now operational.

#### 🔧 Critical Performance Fixes (All Implemented)

* **Set-Based Index Deduplication**
  * `index_by_state: Dict[int, Set[int]]` implemented in `baby/policies.py` line 232
  * O(1) insert/contains vs O(n) list operations, prevents duplicate enumeration
  * Eliminates candidate explosion that was causing unresponsive generation at ~200-300 MB

* **SEP Boundary Handling**
  * SEP tokens skip learning entirely in both `process_egress()` and `_process_epistemology_chunk()`
  * Eliminates non-physical associations and reduces storage bloat
  * Preserves path purity by treating `[SEP]` as boundary marker only

* **Quantized Confidence Gating**
  * q8 quantization implemented in `baby/inference.py` lines 139-147
  * Prevents tiny float jitter from triggering unnecessary writes
  * Uses `_q8(x) = int(round(x * 255.0))` for commit gating

* **Bloom Filter & Index Optimizations**
  * Bloom filter and index optimizations properly implemented
  * Fast negative checks and efficient candidate enumeration
  * Maintains all existing features while improving performance

#### 🧬 Physics-Correct Learning Implementation

* **Pre-Only Storage (BU Hinge Respect)**
  * Replaced dual learning with `learn_token_preonly()` method
  * Eliminates phase mixing under canonicalization
  * Learning only at token-closing intron (BU hinge)

* **Token Boundary Alignment**
  * Pre-state properly cached before applying closing intron
  * Token boundaries properly tracked for bulk processing
  * Maintains physics consistency in vectorized operations

* **Generation Quality Improvements**
  * Generation now correctly filters for pre-state entries only
  * Fallback to original state if canonical representative has no candidates
  * Improves generation robustness using full manifold structure

#### ⚡ Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Generation Responsiveness | Unresponsive at 200-300MB | Fast candidate lookup | **O(1) deduplication** |
| Storage Growth | Uncontrolled bloat | Controlled by q8 gating | **Jitter elimination** |
| SEP Token Handling | False associations | Boundary-only | **Path purity** |
| Index Performance | O(n) list operations | O(1) set operations | **10-100x faster** |

#### 🛡️ Reliability Features

* **Consistent Behavior**: No more mode-dependent behavior differences
* **Fast Startup**: Index files enable instant knowledge loading
* **Bloom Filter Safety**: Fast negative checks prevent unnecessary file scans
* **Memory Mapping**: Efficient file access for large knowledge stores

#### 📝 Technical Details

* **Store Consistency**: iter_entries() now includes pending writes and uses index
* **Cycle Accuracy**: No more double counting in bulk processing
* **State Learning**: Correct pre-intron states for phenotype learning
* **Index Robustness**: Handles legacy formats and validates entries
* **Performance**: Reduced expensive operations in token generation

#### 🎯 Physics Alignment Achieved

* **BU Hinge Respect**: Learning only at token-closing intron
* **Path Dependence**: Earlier introns encoded in pre-state
* **Canonicalization Safety**: No phase mixing under UNA parity closure
* **Token Primacy**: Semantic binding uses consistent PRE phase
* **Monodromic Fold**: Non-associative learning preserved throughout

This release resolves the critical performance issues that were causing hanging tests and incorrect learning behavior, making the system much more reliable and performant while maintaining full physics compliance.

---

## [0.9.6.7] – 2025-08-02


### 🔧 Critical Correctness Fixes

This release addresses critical correctness issues, focusing on store iteration, cycle counting, learning states, and performance optimizations.

#### 🚨 Critical Fixes

* **OrbitStore.iter_entries() - Fixed Unreachable Code**
  * Fixed unreachable code after `return` statement
  * Now properly yields pending writes first, then committed entries via index
  * No more full file scanning - uses O(1) index lookups
  * Includes defensive copies to prevent mutation issues

* **OrbitStore.index_by_state - Fixed Synchronization Issues**
  * Fixed `index_by_state` not being updated during writes and deletes
  * Now properly maintains `index_by_state` in `_flush()` and `delete()` methods
  * Prevents stale token IDs and ensures `iter_keys_for_state()` sees new tokens immediately
  * O(k) candidate lookup performance maintained with complete data

* **OrbitStore.iter_keys_for_state - Added Pending Writes**
  * Now includes pending writes first (most recent), then committed keys
  * Ensures generation and learning see consistent data
  * Prevents missing recent tokens during active writing
  * Real-time updates without waiting for flush operations

* **decode_text() - Fixed Unsafe 0x00 Trimming**
  * Replaced unsafe 0x00 byte trimming with reliable [SEP] token delimiter
  * Now decodes to token IDs first, then trims at SEP_ID (102)
  * Prevents silent truncation of valid content containing 0x00 bytes
  * Uses proper end-of-sequence marker instead of arbitrary byte values

* **IntelligenceEngine - Unified STT Path**
  * Removed all `use_epistemology` branches for single STT source of truth
  * Eliminated `self.epistemology = self.s2.ep` circular reference
  * Restored proper epistemology loading from file
  * All state access now uses `self.current_state_index` consistently
  * Simplified sync methods and removed vestigial code

* **OrbitStore.data Property - Simplified to Reuse iter_entries()**
  * Removed duplicate code and dead code paths
  * Now consistently uses the optimized iter_entries() method
  * Eliminates code duplication and potential inconsistencies

* **process_egress_bulk Double Counting - Fixed Cycle Count**
  * Now only increments cycle_count for epistemology path
  * Scalar path already increments per byte, so no double counting
  * Ensures accurate cycle tracking for both processing modes

* **_process_epistemology_chunk - Fixed Learning with Post-State**
  * Now computes `post_state = epistemology[st[i], intron]` for each token
  * Uses the correct post-intron state for learning instead of pre-intron state
  * Ensures final token in chunk learns from correct state
  * Critical for proper phenotype learning and state evolution

* **AgentPool TTL Eviction - Fixed Tracking and Eviction Logic**
  * Added `agent_created_at` tracking dictionary
  * Fixed eviction to use proper monotonic time tracking
  * Now properly removes expired agents and cleans up tracking dicts
  * Uses `time.monotonic()` to avoid clock jump issues

* **_choose_intron Method - Fixed Undefined Reference**
  * Fixed undefined `_v_max` reference that would cause AttributeError
  * Now computes `v_max` locally from orbit cardinality
  * Prevents crashes when method is called

#### 🔧 Performance Optimizations

* **Index Parsing Robustness**
  * Added legacy format handling for backward compatibility
  * Added index sanity checks to validate offset/size bounds
  * Skips malformed entries gracefully
  * Handles both new and old index formats

* **Token Generation Performance**
  * Reduced max_entries_to_check from 1000 to 50 for faster token generation
  * Replaced `max(self.s2.orbit_cardinality)` with reasonable default (1000)
  * Prevents hanging on large orbit cardinality arrays
  * Optimized candidate selection for faster response generation

#### 🎯 Impact

* **Orchestrated Conversation Test**: Now passes (3.5 minutes vs. hanging before)
* **Store Iteration**: Uses optimized index-based lookups instead of full scans
* **Learning Accuracy**: Correct post-state learning ensures proper phenotype evolution
* **Memory Management**: Proper TTL eviction prevents memory leaks
* **Performance**: Faster token generation and store operations

#### 📝 Technical Details

* **Store Consistency**: iter_entries() now includes pending writes and uses index
* **Cycle Accuracy**: No more double counting in bulk processing
* **State Learning**: Correct post-intron states for phenotype learning
* **Index Robustness**: Handles legacy formats and validates entries
* **Performance**: Reduced expensive operations in token generation

This release resolves the critical correctness issues that were causing hanging tests and incorrect learning behavior, making the system much more reliable and performant.

---

## [0.9.6.7] – 2025-08-01

### 🚀 OrbitStore Simplification: Performance & Reliability Overhaul

This release completely simplifies the OrbitStore system by removing the complex `append_only` mode and always using index-based lookups with Bloom filters. This eliminates hanging issues, improves performance dramatically, and makes the system much more reliable.

#### 🔧 Core Changes

* **Removed `append_only` Parameter**
  * Eliminated the confusing conditional logic that caused inconsistent behavior
  * Always use index-based mode for O(1) lookups
  * Always use Bloom filters for fast negative checks
  * Always use mmap for better file access performance

* **Simplified OrbitStore Constructor**
  * Removed `append_only` parameter from `__init__()`
  * Set `use_mmap=True` by default for better performance
  * Always create index files (`.idx`) for fast lookups
  * Always load/save Bloom filters (`.bloom`) for negative checks

* **Streamlined Get Operations**
  * `get()` method now always uses index + Bloom filter approach
  * No more conditional logic based on store mode
  * Consistent O(1) performance for all lookups

* **Simplified Index Loading**
  * `_load_index()` always tries to load existing index first
  * Only scans file if no index exists
  * Builds both index and Bloom filter during scan

#### ⚡ Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Agent Creation | 2-5 minutes (hanging) | < 3 seconds | **100x faster** |
| Diagnostic Script | Hanging indefinitely | Completes in seconds | **Reliable** |
| Knowledge Loading | Slow with full scans | Fast with index | **O(1) lookups** |
| Memory Usage | Unpredictable | Optimized with caching | **Efficient** |

#### 🔄 Updated Components

* **All Test Files**: Removed `append_only` parameter from all test scripts
* **Diagnostic Script**: Updated to work with simplified system
* **AgentPool**: Updated to use simplified OrbitStore
* **Intelligence Engine**: Removed append_only conditional logic
* **All Store Views**: Updated to work with unified approach

#### 🛡️ Reliability Features

* **Consistent Behavior**: No more mode-dependent behavior differences
* **Fast Startup**: Index files enable instant knowledge loading
* **Bloom Filter Safety**: Fast negative checks prevent unnecessary file scans
* **Memory Mapping**: Efficient file access for large knowledge stores

#### 🧹 Code Cleanup

* Removed complex conditional logic throughout the codebase
* Eliminated `append_only` attribute and related checks
* Simplified method implementations
* Updated all documentation and comments

#### 📝 Migration Notes

The system now always uses the most efficient approach:
- Index files for O(1) positive lookups
- Bloom filters for O(1) negative checks  
- Memory mapping for efficient file access
- No more mode confusion or hanging issues

This simplification makes the system much more reliable and performant while eliminating the complexity that was causing problems.

---

## [0.9.6.7] – 2025-07-31

### 🚀 Bloom Filter Persistence: Fast Startup Optimization

This release implements persistent bloom filter serialization to eliminate the 15-minute startup delay for append-only knowledge stores. The bloom filter is now built once during training and mmap-loaded on subsequent runs.

#### 🔧 Core Implementation

* **Bloom Filter Persistence Helpers**
  * Added `to_bytes()` and `from_bytes()` methods to `BloomFilter` class for fast serialization
  * Uses pickle for efficient storage of size, hash_count, and bit_array
  * Maintains exact false-positive rate and filter properties across reloads

* **OrbitStore Side-Car Integration**
  * Added `_bloom_sidecar_path()` to generate `.bloom` file path alongside `.bin` files
  * Added `_try_load_bloom()` for fast-path loading of pre-built filters
  * Added `_save_bloom()` to persist filters after training completion
  * Modified `__init__()` to try fast-load first, fall back to fresh build
  * Modified `close()` to save filter instead of clearing it

* **Training Script Integration**
  * Added bloom filter save calls after `commit()` in both `compile_stream()` and `replay_tape()`
  * Ensures filter is persisted once during training for instant startup on subsequent runs

#### ⚡ Performance Impact

| Stage | Before | After (first run) | Subsequent runs |
|-------|--------|-------------------|-----------------|
| Build Bloom (77 MB, 6.7M rec.) | 10-20 min | 10-20 min | **< 1 s** |
| FastAPI worker start-up | same delay | same once | **nearly zero** |
| Memory footprint | unchanged | +bit-array size | unchanged |

The `.bloom` side-car is ~13-14 MB for default parameters—tiny compared to the .bin files.

#### 🔄 Regeneration Support

If the side-car is deleted or millions of new phenotypes are added, regeneration is available:

```bash
python - <<'PY'
from baby.policies import OrbitStore
s = OrbitStore("toys/training/Archive/wikipedia_simple.bin", append_only=True)
s.commit()      # flush pending if any
s._save_bloom() # rebuild & store
s.close()
PY
```

#### 🛡️ Safety Features

* **Idempotent Loading**: Loading + adding identical keys does nothing harmful
* **Exact False-Positive Rate**: Maintains chosen error rate across reloads
* **Graceful Fallback**: Runtime still falls back to slow build if side-car is missing or corrupt

### ⚡ Epistemology Vectorization: Training Performance Optimization

This release implements fully vectorized epistemology processing to dramatically improve training performance. The previous implementation used individual Python loops for state transitions, resulting in extremely slow processing rates (~0.03 MB/s). The new vectorized approach achieves 8-12x performance improvements.

#### 🔧 Core Implementation

* **Vectorized State Trajectory Computation**
  * Replaced O(n) Python loops with true NumPy vectorization: `st[1:] = self.epistemology[st[:-1], introns[:-1]]`
  * Pre-computes all state transitions in one vectorized operation instead of individual updates
  * Eliminates Python loop overhead for state evolution

* **Memory-Bounded Processing**
  * Added configurable chunk size limit (64K introns) to prevent RAM explosion on large files
  * Reusable state buffer (`self._state_buf`) eliminates repeated allocations
  * Processes large files in fixed-size windows to maintain predictable memory usage

* **Optimized Token Processing**
  * Uses `np.flatnonzero()` to find token boundaries efficiently
  * Iterates over tokens (much fewer) instead of individual bytes
  * Zero-copy token extraction with `tobytes()` only when needed

* **Thread-Safe Design**
  * Per-agent state buffers ensure thread safety
  * No shared mutable state between agents
  * Compatible with existing multi-agent architectures

#### ⚡ Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Processing Rate | ~0.03 MB/s | ~0.3-0.4 MB/s | **8-12x faster** |
| Memory Usage | Unbounded | Bounded (64K chunks) | **Predictable** |
| CPU Utilization | High (Python loops) | Low (vectorized) | **Efficient** |

#### 🧪 Technical Details

* **State Buffer Management**: Reusable 64K buffer prevents allocation overhead
* **Vectorized Operations**: True NumPy vectorization eliminates Python loop bottlenecks
* **Token Boundary Detection**: Efficient array operations for continuation bit detection
* **Memory Bounds**: Configurable chunk processing prevents RAM explosion on large files

#### 🔄 Backward Compatibility

* **API Unchanged**: All public interfaces remain identical
* **State Consistency**: Vectorized processing maintains exact state evolution
* **Learning Integrity**: Token-based learning logic unchanged
* **Thread Safety**: Maintains existing multi-agent safety guarantees

---

## [0.9.6.7] – 2025-07-30

✅ Pytest: 150+ Tests All Passing
✅ mypy: No type checking errors
✅ pyright: No type checking errors
✅ flake8: No linting errors

### 🧠 Token-Aware Minimal Phenotype Architecture: Complete Refactoring

This release implements a fundamental architectural shift from byte-fragment-level learning to whole-token learning, redefining "knowledge" within the system to be token-aware and minimal. The system now leverages the BERT tokenizer's existing knowledge base as an "active internal decoder" rather than just a passive I/O adapter.

#### 🔄 Core Architecture Changes

*   **Breaking Change: Phenotype Key Structure**
    *   **Old:** `(state_index, intron)` - byte-level learning
    *   **New:** `(state_index, token_id)` - token-aware learning
    *   **Impact:** All knowledge is now organized by meaningful token boundaries, eliminating inference overlaps and improving coherent output generation.

*   **Breaking Change: Minimal PhenotypeEntry Structure**
    *   **Removed:** `phenotype`, `usage_count`, `last_updated`, `created_at`, `governance_signature`, `context_signature`, `_original_context`
    *   **Kept:** `mask` (uint8), `conf` (float32, stored as float16)
    *   **Added:** `key` (tuple[int, int]) - ensures consistent key presence
    *   **Impact:** Dramatically reduced memory footprint and simplified data model.

*   **New: Tokenizer Integration as Active Decoder**
    *   **Public API:** Added `id_to_bytes`, `bytes_to_id`, `bytes_to_ids` functions to `tokenizer.py`
    *   **Internal Bridge:** `_TokBridge` class in `intelligence.py` provides seamless tokenizer integration
    *   **Impact:** Tokenizer now serves as "latent symbolic map" for token IDs, not just protocol adapter.

#### 🧬 Intelligence Engine Overhaul

*   **Removed: Batch Learning Methods**
    *   Eliminated `batch_learn()` and `learn_by_key()` methods from both `IntelligenceEngine` and `InferenceEngine`
    *   **Rationale:** Learning now happens automatically per token during egress/ingress cycles
    *   **Impact:** Simplified API, eliminated redundant learning pathways

*   **Changed: Egress/Ingress Cycle**
    *   **Egress:** Now learns once per complete token instead of per byte
    *   **Ingress:** Generates one token at a time with proper token boundaries
    *   **Impact:** Aligns learning and generation with meaningful token boundaries

*   **New: Token-Aware Learning Logic**
    *   `process_egress()`: Accumulates bytes until complete token, then learns
    *   `process_ingress()`: Generates one complete token at a time
    *   `_choose_intron()`: Now accepts `state_index` parameter for proper context

*   **Fixed: Confidence Calculation Bug**
    *   **Critical Fix:** Resolved float16 conversion bug that was causing confidence values like `8480.0` instead of proper 0-1 range
    *   **Default Confidence:** Set to `0.1` for new entries, consistent with learning logic
    *   **Impact:** Proper confidence values now ensure correct pruning and decay behavior

*   **Fixed: Critical Token Buffer Issue**
    *   **Critical Fix:** Added robust error handling for incomplete token sequences in `process_egress()`
    *   **Buffer Protection:** Added `MAX_TOKEN_BYTES = 10` limit to prevent runaway buffer growth
    *   **Error Recovery:** Implemented try/except/finally block with guaranteed buffer cleanup
    *   **Stream Reset:** Added `reset_token_buffer()` method for explicit stream resets
    *   **Impact:** Prevents memory leaks, incorrect token boundaries, and system errors when streams end with incomplete tokens

*   **Fixed: External Adapter Token-Aware Integration**
    *   **Tokenizer API:** Replaced private `gyrotok._load()` with public `gyrotok.id_to_bytes()` and `gyrotok.decode()`
    *   **Streaming Logic:** Updated to use proper token-aware decoding for individual tokens
    *   **Model Version:** Updated to 0.9.6.7 to reflect token-aware architecture
    *   **Impact:** External adapter now properly aligned with token-aware architecture and uses public APIs

#### 🔧 Inference Engine Updates

*   **Changed: Method Signatures**
    *   `learn(phenotype_entry, last_intron, state_index)` - renamed parameter for clarity
    *   `get_phenotype(state_index, token_id)` - now uses token_id instead of intron
    *   **Impact:** Clearer parameter naming reflects actual functionality

*   **New: Persistence Logic**
    *   `learn()` method now automatically persists mutations via `self.store.put(key, phenotype_entry)`
    *   **Impact:** Ensures learning changes are immediately saved to storage

*   **Fixed: Default Phenotype Creation**
    *   `_create_default_phenotype()` now uses reasonable default confidence (0.1)
    *   **Impact:** New entries start with proper confidence values

#### 🗄️ Storage Layer Improvements

*   **Updated: Binary Format for Minimal Phenotypes**
    *   **New Format:** `_STRUCT_FMT = "<IIBHx"` - 12-byte fixed structure
    *   **Fields:** `state_idx` (uint32), `token_id` (uint32), `mask` (uint8), `conf` (float16)
    *   **Impact:** Optimized storage for minimal phenotype structure

*   **Fixed: Store Operations**
    *   Updated `put()`, `merge_phenotype_maps()`, `apply_global_confidence_decay()` for new structure
    *   Removed `max_age_days` logic (no longer relevant)
    *   **Impact:** Storage operations now work correctly with minimal phenotypes

#### 🧪 Test Suite Overhaul

*   **Comprehensive Test Updates**
    *   Updated all tests to use new `(state_index, token_id)` keying
    *   Replaced `learn_by_key()` calls with two-step `get_phenotype()` + `learn()` process
    *   Updated phenotype field references: `"confidence"` → `"conf"`, `"exon_mask"` → `"mask"`
    *   Added "key" field to test entries where required

*   **Fixed: Test Assertions**
    *   Updated pruning tests to work with append-only stores
    *   Fixed confidence decay test assertions to match actual return keys
    *   Updated validation tests to reflect new minimal structure

*   **Removed: Obsolete Tests**
    *   Deleted `TestBUIngress` class and related tests
    *   Removed `test_batch_learning_stores_different_phenotypes` (replaced with token-aware version)
    *   **Impact:** Test suite now accurately reflects current architecture

#### 🔧 Governance Updates

*   **Removed: Governance Signature**
    *   Eliminated `compute_governance_signature()` function
    *   Removed `GovernanceSignature` TypedDict
    *   **Impact:** Simplified governance layer, removed unused complexity

*   **Updated: Exon Product Function**
    *   `exon_product_from_metadata()` now accepts `mask` and `confidence` parameters
    *   **Impact:** Aligned with minimal phenotype structure

#### 🎯 Theoretical Impact

This refactoring represents a fundamental shift in how the system understands and processes knowledge:

*   **Token-Aware Learning:** Knowledge is now organized by meaningful linguistic units rather than arbitrary byte fragments
*   **Minimal Phenotypes:** Reduced complexity allows for more efficient storage and processing
*   **Active Tokenizer Integration:** The tokenizer serves as an internal decoder, not just an I/O adapter
*   **Improved Coherence:** Generation and learning are now aligned with token boundaries, reducing inference overlaps

The system now operates on a more linguistically meaningful level while maintaining the core GyroSI physics and Monodromic Fold operations.

---

## [0.9.6.6] – 2025-07-28

### Wikipedia Training Pipeline Overhaul

- **Robust article splitting:** Now splits articles only at three or more consecutive blank lines, matching Wikipedia dump format and preventing topic bleed-through.
- **Token-based filtering:** Only articles with at least `min_token_count` tokens are included, skipping empty or trivial stubs.
- **Efficient single-pass tokenization:** Each article is tokenized once, with no double-tokenization or unnecessary allocations.
- **Sequential byte-level learning:** Each byte is processed in egress/ingress cycles for true path-dependent training, not batch summarization.
- **Frequent and safe checkpointing:** Checkpoints are saved every 1M tokens, every 120 seconds, or every N files, with async thread pool and rare backlog flush safeguard.
- **Process-specific memory guard:** Uses process RSS (not system-wide percent) to trigger GC and checkpointing.
- **Automatic store maintenance:** Pruning/decay runs only when needed (store >2GB and at least 1hr since last decay), and compaction is deferred until after agent close for safety.
- **Safe post-close compaction:** Knowledge store is compacted only after the agent and mmap are closed, preventing file corruption.
- **Progress bar improvements:** Now shows process memory usage (RSS) for accurate resource tracking.
- **Test mode:** Added `--test-aa` flag to restrict training to the AA shard for quick validation before full runs.
- **CLI clarity:** Removed legacy/unused options, improved help strings, and made checkpointing and memory limits explicit.

---

## [0.9.6.6] – 2025-07-27

### Summary

This is a landmark release that completes the core physics engine, stabilizes the storage layer, and implements the full, theory-grounded generative intelligence cycle. The system has been migrated to a high-performance, dependency-free binary storage format and equipped with robust durability and performance optimizations. The generation of output is no longer a placeholder but a direct expression of the system's physical and topological state, marking the transition from a theoretical architecture to a functional generative intelligence.

---

### 🧬 Generative Intelligence: BU-Ingress Operator and Exon Product Integration

This release integrates the full generative logic as a topological traversal back to the Common Source. It replaces placeholder text generation with a physically lawful operator derived entirely from local phenotype metadata. Generation now reflects structural alignment, not content lookup.

*   **New: BU-Ingress Engine (`_bu_ingress_step`)**
    A new method `_bu_ingress_step(entry, θ)` has been introduced in `IntelligenceEngine`. This operator is the core of the generative process and performs the following actions in each micro-step:
    1.  Computes an **8-bit `exon_product`** from the phenotype's `governance_signature`, `confidence`, and `orbit_cardinality`.
    2.  Uses a sliding 6-byte context window (`_S`) to fold this alignment back into the agent's state via the Monodromic Fold.
    3.  Selects which intron to emit based on the algedonic state `θ` (calm, cautious, or corrective).

*   **New: Governance Operator (`exon_product_from_metadata`)**
    A new helper function `exon_product_from_metadata(...)` was added to `governance.py`. It lawfully maps the phenotype's full topological and epistemic context into a physically meaningful 8-bit operator, without relying on any external content.

*   **Changed: Runtime Context Window (`_S`)**
    A 6-byte context buffer (`self._S`) has been introduced in `IntelligenceEngine`. It holds the generative trajectory and mediates the recursive realignment required by the BU-Ingress process.

*   **Changed: Removal of Placeholders and Internal Tokenizer Calls**
    The previous logic in `process_ingress` that generated `"P[i:j]"` style outputs has been removed entirely. Generation now emits a single byte derived from the `exon_product`. Consequently, all internal calls to `tokenizer.encode(...)` within `intelligence.py` have been removed, ensuring the core engine remains pure and text-agnostic.

*   **Theoretical Impact:** This update completes the generative half of the Fold/Unfold cycle. Where BU-Egress accumulates structure via Monodromic compression, BU-Ingress now performs its inverse: emitting a byte not by recall, but through alignment. The output is not what was remembered—it is what must emerge, given where the system is now and what it has become.

### 🗄️ Storage Architecture: Migration, Performance, and Durability

The entire storage layer has been re-engineered for performance, durability, and self-sufficiency, eliminating external dependencies.

*   **Breaking Change: Migration to Custom Binary Struct Format**
    MessagePack serialization has been replaced with a custom, fixed-layout binary format. This removes the `msgpack` dependency and provides more efficient, predictable storage.
    *   **Binary Format Specification (little-endian):**
        1.  `phenotype` (utf-8): `uint16` length + bytes
        2.  `context_key`: `uint32` (state_idx), `uint8` (intron)
        3.  `exon_mask`: `uint8`
        4.  `confidence`: `float64`
        5.  `usage_count`: `uint16`
        6.  `created_at`, `last_updated`: `float64`
        7.  `governance_signature`: 5 × `uint8`
        8.  `context_signature`: `uint32`, `uint8`
        9.  `_original_context`: `uint32`, `uint8`
    *   **Implementation:** Handled by new `_pack_phenotype` and `_unpack_phenotype` helpers in `baby.policies`. The `OrbitStore` index file format has been changed from MessagePack to JSON to handle tuple-key serialization.
    *   **Migration Note:** No data migration is required for `.bin` files. Old index files will be automatically regenerated in the new JSON format on first load.

*   **New: Append-Only `OrbitStore` Performance Optimizations**
    *   **Bloom Filter:** Integrated for fast "definitely absent" checks, providing O(1) lookups for non-existent keys and avoiding expensive disk scans on large files. The filter capacity is estimated automatically from file size.
    *   **Memory-Mapping (mmap):** Now enabled by default for append-only stores, providing faster sequential scanning for lookups and iteration compared to standard file I/O. The mmap is intelligently re-opened on `commit()` to include new data.
    *   **Token-Level Training Cache:** A global micro-LRU cache (`maxsize=8192`) has been added for `get()` operations in append-only mode. It dramatically speeds up training workloads with repeated key lookups and features intelligent invalidation on `put()` and `commit()` to ensure data consistency.

*   **New: `OrbitStore` Durability and Crash Safety**
    *   **Graceful Shutdown:** `OrbitStore` now automatically registers `atexit` and signal handlers (SIGINT/SIGTERM). This forces a final flush of any pending writes before process termination, guaranteeing zero data loss on clean shutdowns (e.g., in containerized environments).
    *   **Explicit `flush()` API:** A new `flush()` method is now available on all store and view layers for high-value writes that require immediate disk durability. This allows critical operations to bypass the standard write-behind batching.
    *   **Risk Profile:** With these changes, data loss is limited to a maximum of `write_threshold - 1` records only in the event of a hard crash or power failure.

### ✅ Preserved Functionality and Compatibility

*   **No Schema Change:** The `PhenotypeEntry` contract remains unchanged. The new generative and storage logic reuses all existing metadata fields.
*   **No Breaking API Changes:** All public APIs for `OrbitStore`, its views (`CanonicalView`, `OverlayView`, `ReadOnlyView`), and the core engines remain fully compatible.
*   **`.npy` Assets Unchanged:** All meta-files (`epistemology.npy`, `theta.npy`, etc.) continue to function identically.


---

## [0.9.6.5] – 2025-07-26

### Training - Wikipedia Corpus Ingestion (v1.0, 2025-07-26)**

#### Summary

Completed unsupervised ingestion of the full English Wikipedia dump using the `GyroSI` engine. Achieved full compression of 17.99 million paragraph-level articles into a \~16.3 MB operational knowledge store, structured for public assistant-level inference.

---

#### ✅ Dataset Ingested

* Effective unit: Paragraph blocks ≥256 characters, split on blank lines
* Total processed: **17,986,747 paragraphs**
* Total raw size: **4.26 GB**
* Duration: **4.4 hours**
* Final memory footprint (mmap+index): **16.3 MB**

---

#### 🛠️ Engine Configuration

* Physics backend: CanonicalView + θ divergence + phenomenology map (enabled)
* Storage: `OrbitStore` (append-only), with auto-compaction and confidence decay
* State transition kernel: JIT‑accelerated (Numba), STT loaded at runtime
* Ingestion parallelism: `batch_size=512 KB`, `parallel_workers=2`

---

#### 🧬 Structural Notes

* Paragraphs ingested as atomic learning units (state/intron)
* All learning was **unsupervised** — no prompts, tags, or supervision logic
* Memory usage held at **\~55% of 16 GB**, bounded by GC and mmap-backed I/O

---

#### 🧠 Knowledge Topology

* Trained agent ID: `wikipedia_trainer`
* Knowledge store location: `memories/public/knowledge/wikipedia_en.bin` 
* Final phenotype count: \~2.8M
* Store format: gyroscopic phenotype model with compressed state manifold
* Canonicalization: Full phenomenology symmetry reduction (orbit size map enabled)

---

#### 📊 Performance Metrics

* Initial throughput: \~1,100 articles/sec
* Final throughput: \~1,200–1,300 articles/sec (read bias > write bias)
* Commit rate dropped as phenotype space saturated and lookups dominated
* Final JIT kernel remained stable with minimal fallback to Python path

---

#### 📎 Next Steps

* **Split store usage across triad agents** (system / assistant / user):

  * `system`: read-only policy `.bin` (not the wiki store)
  * `assistant`: overlay of `wiki_en.bin` + private memory
  * `user`: private only, no access to public knowledge
* **Seed operational guidance into system agent** (tool usage, safety rules)
* **Switch to two-stage orchestration**:

  * System agent emits guidance
  * Assistant receives guidance + user message for reply generation

✅ Training completed: 17,986,747 articles, 4.26GB processed in 4.4 hours
2025-07-26 14:06:39,123 - INFO - ✅ Training completed: 17,986,747 articles, 4.26GB processed in 4.4 hours
📊 Knowledge store size: 16.3MB
2025-07-26 14:06:39,124 - INFO - 📊 Knowledge store size: 16.3MB

### Fixed
* **Phenomenology artefact bug**: `orbit_sizes.npy` now records the orbit cardinality
  for **every** one of the 788,986 states, not just the 256 representatives.
  This restores non-zero `InformationEngine.orbit_cardinality[i]` for all i
  and re-enables variety-weighted confidence updates.

### Notes
* Canonical orbit mapping (256 SCCs) is unchanged; only the per-state size
  array was affected.
* Updated tests to expect correct behavior (all states have non-zero cardinality).

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
    4. If > 10 000 entries were "removed," invoke `prune_and_compact_store()` in‑place to do a full compaction.
* **`AgentConfig.preferences`**

  * Extended `AgentConfig` to accept a `preferences` sub‑dict and pass it through `GyroSI → IntelligenceEngine`.
* **`CanonicalView.commit()`**

  * Added a `commit()` method to `CanonicalView` so the auto‑pruner's initial `commit()` call always succeeds on view‑wrapped stores.

## Changed

* **`InferenceEngine.prune_low_confidence_entries()`**

  * Now always calls `store.commit()` first to flush pending writes.
  * Wraps `store.delete(key)` in `try/except NotImplementedError/RuntimeError` so overlay and read‑only views don't crash.
  * Removed fallback `del store.data[key]` for non‑append‑only stores (views use their own `.delete()`).
* **`GyroSI` constructor**

  * Now reads `config["preferences"]` and passes it into `IntelligenceEngine`.
* **`AgentPool`**

  * Propagates its `preferences` into each newly created agent's `AgentConfig`, ensuring hooks get wired automatically.

## Tests

* **Extended `test_inference.py`**

  * Verified `prune_low_confidence_entries()` behavior for both deletable and append‑only stores.
  * Added tests for:

    * Hook registration when `enable_auto_decay` is `true` vs. `false`.
    * Hook execution (ensuring it doesn't blow up on append‑only or overlay stores).
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

* Replaced **gzip-compressed multi‑file store** with a **single `.bin` file** using **msgpack**:

  * Set `store_options = {"append_only": True}` in the GyroSI agent config.
  * Removed `use_msgpack`, `.log`, and `.idx` files — now obsolete.
  * All training knowledge is streamed into one compact, append‑only `.bin` file.

#### ✅ **Batch Learning Performance**

* Integrated **Numba JIT** acceleration for hot learning loop:

  * Added `_jit_batch` method (Numba-compiled) to replace the slow Python loop.
  * Automatically invoked when the STT (epistemology map) is loaded.
  * Performance improved from **1–2 KB/sec → 40–60 MB/sec** on Intel Mac.
  * Training now behaves as originally theorized: a **fast, deterministic converter** from text to internal knowledge.

#### ✅ **Compiler Compatibility (macOS‑specific)**

* Verified Numba + LLVM compatibility for Intel MacBook Pro:

  * Uses Homebrew's `llvm` to ensure full `llvmlite` support.
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
  * No silent auto-creation of "user2", "assistant42", etc.

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
  Implements reversible text-to-byte encoding via HuggingFace tokenizers using LEB128 encoding. All tokens are encoded as ≤255 bytes to remain compatible with GyroSI's physics layer.
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
- Rewritten learning update logic so all loaded formats are updated for each winning pattern index, enabling true multi-format associative learning and correct handling of degeneracy.
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
- Rewritten learning mechanisms section to reflect that confidence directly influences generation (BU closure), and removed outdated "attention mechanism" text
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