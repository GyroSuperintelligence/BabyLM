# Comprehensive Implementation Guide: Aligning GyroSI's Temporal Physics

## Executive Summary

The current implementation stores phenotypes at the **post-transition state** while generation queries the **pre-transition state**, creating a fundamental phase mismatch. This guide provides all necessary code changes to:

1. Store both pre- and post-state associations in the same 12-byte record using bit 0 as direction flag
2. Align all learning paths (live, bulk, replay) to the correct temporal frame
3. Enhance generation to leverage the full manifold structure through orbit-aware lookups
4. Remove confidence-based pruning in favor of physics-based trajectory preservation

---

## Core Fix: Dual-State Storage with Direction Bit

### 1. Modified Phenotype Storage Structure

The least significant bit (bit 0) of the mask will encode direction:
- `0` = pre-state entry (for generation)
- `1` = post-state entry (for reverse lookup/diagnostics)

**File: `baby/policies.py`**

```python
# Add after line 74 (after _STRUCT_SIZE definition)
# Direction bit constants
DIR_PRE = 0   # Pre-state entry (for generation)
DIR_POST = 1  # Post-state entry (for diagnostics/reverse)

# Modify _pack_phenotype function (around line 106)
def _pack_phenotype(entry: Dict[str, Any]) -> bytes:
    if "key" not in entry:
        raise KeyError("Entry must have 'key' field")
    state_idx, token_id = entry["key"]
    
    # Extract direction bit if present, default to DIR_PRE
    direction = entry.get("direction", DIR_PRE)
    raw_mask = entry["mask"] & 0xFE  # Clear bit 0
    mask = raw_mask | (direction & 0x01)  # Set direction bit
    
    conf = float(entry["conf"])
    conf_f16 = np.float16(conf)
    conf_u16 = conf_f16.view(np.uint16).item()
    return struct.pack(_STRUCT_FMT, state_idx, token_id, mask, conf_u16)

# Modify _unpack_phenotype function (around line 118)
def _unpack_phenotype(buf: memoryview, offset: int = 0) -> tuple[Dict[str, Any], int]:
    state_idx, token_id, mask, conf_u16 = struct.unpack_from(_STRUCT_FMT, buf, offset)
    conf_f16 = np.uint16(conf_u16).view(np.float16)
    raw_conf = float(conf_f16.item())
    normalized_conf = normalize_confidence(raw_conf)
    
    # Extract direction bit and clear it from mask
    direction = mask & 0x01
    clean_mask = mask & 0xFE
    
    entry = {
        "mask": clean_mask,
        "conf": normalized_conf,
        "key": (state_idx, token_id),
        "direction": direction
    }
    return entry, offset + _STRUCT_SIZE
```

### 2. Enhanced Inference Engine for Dual Learning

**File: `baby/inference.py`**

```python
# Modify learn() method (around line 116) to handle dual storage
def learn(self, phenotype_entry: PhenotypeEntry, last_intron: int, state_index: int) -> PhenotypeEntry:
    """
    Update memory via the Monodromic Fold.
    Now stores both pre-state and post-state entries for richer trajectory.
    """
    if state_index < 0 or state_index >= len(self.s2.orbit_cardinality):
        raise IndexError(f"state_index {state_index} out of bounds [0, {len(self.s2.orbit_cardinality)})")
    
    last_intron = last_intron & 0xFF
    old_mask = phenotype_entry.get("mask", 0) & 0xFE  # Clear direction bit
    
    # Use Monodromic Fold
    new_mask = governance.fold(old_mask, last_intron) & 0xFE  # Keep clean
    
    # Calculate learning rate and confidence
    novelty = bin(old_mask ^ new_mask).count("1") / 8.0
    v = self.s2.orbit_cardinality[state_index]
    alpha = (1 / 6) * math.sqrt(v / self._v_max)
    current_confidence = phenotype_entry.get("conf", 0.1)
    new_confidence = min(1.0, current_confidence + (1 - current_confidence) * alpha * novelty)
    
    if new_mask == old_mask and abs(round(new_confidence, 4) - round(current_confidence, 4)) < 1e-4:
        return phenotype_entry
    
    assert 0 <= new_mask <= 254  # Excluding direction bit
    assert 0 <= new_confidence <= 1
    
    phenotype_entry["mask"] = new_mask
    phenotype_entry["conf"] = new_confidence
    
    key = phenotype_entry["key"]
    storage_key = key
    self.store.put(storage_key, phenotype_entry)
    
    return phenotype_entry

# Add new method after learn_token (around line 195)
def learn_token_dual(self, token_id: int, state_index_pre: int, state_index_post: int, last_intron: int) -> None:
    """
    Learn both pre-state and post-state associations for a token.
    This creates a bidirectional trajectory map.
    """
    # Store pre-state entry (for generation)
    pre_key = (state_index_pre, token_id)
    pre_entry = self.store.get(pre_key)
    
    if pre_entry is None:
        pre_entry = self._create_default_phenotype(pre_key)
        pre_entry["direction"] = DIR_PRE
    else:
        pre_entry = dict(pre_entry)
        pre_entry["direction"] = DIR_PRE
    
    pre_entry["key"] = pre_key
    self.learn(cast(PhenotypeEntry, pre_entry), last_intron, state_index_pre)
    
    # Store post-state entry (for trajectory mapping)
    post_key = (state_index_post, token_id)
    post_entry = self.store.get(post_key)
    
    if post_entry is None:
        post_entry = self._create_default_phenotype(post_key)
        post_entry["direction"] = DIR_POST
    else:
        post_entry = dict(post_entry)
        post_entry["direction"] = DIR_POST
    
    post_entry["key"] = post_key
    # Use dual of last intron for post-state to encode reverse trajectory
    dual_intron = governance.dual(last_intron)
    self.learn(cast(PhenotypeEntry, post_entry), dual_intron, state_index_post)
```

### 3. Intelligence Engine Phase Alignment

**File: `baby/intelligence.py`**

```python
# Modify process_egress() method (around line 342)
def process_egress(self, input_byte: int) -> int:
    input_byte &= 0xFF
    intron = governance.transcribe_byte(input_byte)
    intron &= 0xFF
    
    if len(self._raw_leb_buf) >= self.MAX_TOKEN_BYTES:
        self._raw_leb_buf.clear()
        self._last_token_id = 0
        return intron
    
    self._raw_leb_buf.append(intron)
    
    # Cache pre-state before transition
    pre_state_index = self.current_state_index
    
    # Apply state transition
    self.current_state_index = self.epistemology[self.current_state_index, intron]
    self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
    self._cached_state_int = self.gene_mac_m_int
    
    assert self.gene_mac_m_int < (1 << 48)
    self.cycle_count += 1
    
    div = self.s2.measure_state_divergence(self._cached_state_int)
    self._θ_buf.append(div)
    
    if (intron & 0x80) == 0:  # Token complete
        try:
            tok_bytes = bytes(b ^ 0xAA for b in self._raw_leb_buf)
            token_id = TOK.bytes_to_id(tok_bytes)
            
            if token_id == SEP_ID:
                token_id = getattr(self, "_prev_token_id", 102)
            self._prev_token_id = token_id
            
            # Learn with dual storage: both pre and post states
            post_state_index = self.current_state_index
            self.operator.learn_token_dual(
                token_id, pre_state_index, post_state_index, intron
            )
            
            for hook in self.post_cycle_hooks:
                try:
                    hook(self, phenotype_entry, intron, token_id, pre_state_index)
                except Exception:
                    pass
            
            self._last_token_id = token_id
        except Exception:
            pass
        finally:
            self._raw_leb_buf.clear()
    
    return intron

# Modify _process_epistemology_chunk() (around line 450)
# In the token processing loop:
def _process_epistemology_chunk(self, introns: np.ndarray[Any, Any], masked_arr: Optional[np.ndarray[Any, Any]] = None) -> None:
    ep = self.epistemology
    n = introns.shape[0]
    if n == 0:
        return
    
    if self._state_buf.shape[0] < n:
        self._state_buf = np.empty(max(n, 65536), dtype=np.int32)
    
    st = self._state_buf[:n]
    st[0] = self.current_state_index
    
    epistemology_size = ep.shape[0]
    if st[0] >= epistemology_size:
        raise RuntimeError(f"Initial state index {st[0]} is out of bounds")
    
    # Build state trajectory
    for i in range(1, n):
        prev_state = st[i - 1]
        intron = introns[i - 1]
        if prev_state >= epistemology_size:
            raise RuntimeError(f"State index {prev_state} is out of bounds")
        new_state = ep[prev_state, intron]
        if new_state >= epistemology_size:
            raise RuntimeError(f"Transition to state {new_state} is out of bounds")
        st[i] = new_state
    
    # Process tokens with correct phase
    token_start_idx = 0
    for i, intron in enumerate(introns):
        if len(self._raw_leb_buf) >= self.MAX_TOKEN_BYTES:
            self._raw_leb_buf.clear()
            self._last_token_id = 0
            token_start_idx = i + 1  # Reset token start
        
        self._raw_leb_buf.append(intron)
        
        if (intron & 0x80) == 0:  # Token complete
            tok_bytes = bytes(b ^ 0xAA for b in self._raw_leb_buf)
            try:
                token_id = TOK.bytes_to_id(tok_bytes)
                
                if token_id == SEP_ID:
                    token_id = getattr(self, "_prev_token_id", 102)
                self._prev_token_id = token_id
                
                # Get pre-state (where token started) and post-state (where it ended)
                pre_state = int(st[token_start_idx])
                post_state = int(ep[st[i], intron])
                
                # Dual storage
                self.operator.learn_token_dual(
                    token_id, pre_state, post_state, int(intron)
                )
                
                for hook in self.post_cycle_hooks:
                    try:
                        hook(self, phenotype_entry, int(intron), token_id, pre_state)
                    except Exception:
                        pass
                
                self._last_token_id = token_id
            except Exception:
                pass
            finally:
                self._raw_leb_buf.clear()
                token_start_idx = i + 1  # Next token starts here
    
    # Update final state
    if n > 0:
        final_pre_state = int(st[n - 1])
        final_intron = int(introns[-1])
        new_state = self.epistemology[final_pre_state, final_intron]
        self.current_state_index = new_state
        
        try:
            final_state_int = self.s2.get_state_from_index(self.current_state_index)
            div = self.s2.measure_state_divergence(final_state_int)
            self._θ_buf.append(div)
        except Exception:
            pass
```

### 4. Enhanced Generation with Orbit-Aware Lookup

**File: `baby/intelligence.py`**

```python
# Modify generate_token_exon() method (around line 710)
def generate_token_exon(self, state_index: int, temperature: float = 1.0) -> int:
    """
    Generate next token using exon-product physics with orbit-aware fallback.
    """
    candidates = []
    
    # Primary lookup: canonicalized state
    rep_state_idx = state_index
    if self.phenomenology_map is not None:
        try:
            rep_state_idx = int(self.phenomenology_map[state_index])
        except Exception:
            pass
    
    fetch_limit = 512
    pulled = 0
    
    # First try: exact state match (pre-state entries only)
    for s_idx, token_id in getattr(self.operator.store, "iter_keys_for_state", lambda _s: [])(rep_state_idx):
        if pulled >= fetch_limit:
            break
        entry = self.operator.store.get((s_idx, token_id))
        if not entry or entry.get("direction", 0) != DIR_PRE:
            continue  # Skip post-state entries
        pulled += 1
        
        confidence = entry.get("conf", 0.1)
        mask = entry.get("mask", 0) & 0xFE  # Clear direction bit
        
        orbit_v = 1
        v_max = 1
        if hasattr(self, "s2") and hasattr(self.s2, "orbit_cardinality"):
            try:
                orbit_v = self.s2.orbit_cardinality[rep_state_idx]
                v_max = getattr(self.s2, "_v_max", 1000) or 1000
            except (IndexError, AttributeError):
                pass
        
        from baby.governance import exon_product_from_metadata
        exon_product = exon_product_from_metadata(mask, confidence, orbit_v, v_max)
        resonance = self._calculate_resonance(rep_state_idx, exon_product)
        
        orbit_factor = min(1.0, orbit_v / v_max) if v_max > 0 else 0.1
        score = (confidence * 0.4) + (resonance * 0.4) + (orbit_factor * 0.2)
        
        candidates.append((token_id, score, exon_product))
    
    # Second try: if no candidates and state differs from its representative
    if not candidates and state_index != rep_state_idx:
        # Try original state directly
        for s_idx, token_id in getattr(self.operator.store, "iter_keys_for_state", lambda _s: [])(state_index):
            if pulled >= fetch_limit:
                break
            entry = self.operator.store.get((s_idx, token_id))
            if not entry or entry.get("direction", 0) != DIR_PRE:
                continue
            pulled += 1
            
            confidence = entry.get("conf", 0.1)
            mask = entry.get("mask", 0) & 0xFE
            
            orbit_v = self.s2.orbit_cardinality[state_index]
            v_max = getattr(self.s2, "_v_max", 1000) or 1000
            
            from baby.governance import exon_product_from_metadata
            exon_product = exon_product_from_metadata(mask, confidence, orbit_v, v_max)
            resonance = self._calculate_resonance(state_index, exon_product)
            
            orbit_factor = min(1.0, orbit_v / v_max) if v_max > 0 else 0.1
            score = (confidence * 0.4) + (resonance * 0.4) + (orbit_factor * 0.2)
            
            candidates.append((token_id, score, exon_product))
    
    if not candidates:
        # Fallback but track it
        if not hasattr(self, "_fallback_count"):
            self._fallback_count = 0
        if self._fallback_count < 5:
            print(f"[gen] Fallback: no candidates for state={state_index} (rep={rep_state_idx})")
        self._fallback_count += 1
        return self._generate_random_token()
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Apply smooth temperature function
    if temperature < 0.1 or len(candidates) <= 3:
        return int(candidates[0][0])
    else:
        scores = np.array([score for _, score, _ in candidates])
        scores = scores - scores.min()
        log_scores = np.log(scores + 1e-8)
        scaled_log_scores = log_scores / temperature
        exp_scores = np.exp(scaled_log_scores - np.max(scaled_log_scores))
        probs = exp_scores / np.sum(exp_scores)
        
        chosen_idx = int(np.random.choice(len(candidates), p=probs))
        return int(candidates[chosen_idx][0])
```

### 5. Smooth Temperature Schedule

**File: `baby/intelligence.py`**

```python
# Replace _temperature_from_theta() method (around line 258)
def _temperature_from_theta(self, theta: float) -> float:
    """
    Smooth sigmoid temperature schedule.
    Maps theta from [0, π] to temperature [floor, cap].
    """
    temp_floor = float(cast(float, self.preferences.get("temperature_floor", 0.1)))
    temp_cap = float(cast(float, self.preferences.get("temperature_cap", 1.0)))
    
    # Sigmoid centered at θ_low with steep transition
    import math
    sigmoid = 1.0 / (1.0 + math.exp(-(theta - self._θ_low) * 10))
    
    return temp_floor + (temp_cap - temp_floor) * sigmoid
```

### 6. Replay Pipeline Alignment

**File: `toys/training/wikipedia_eng.py`**

```python
# Modify replay_tape() function's inner loop (around line 430)
# Replace the agent.ingest_bulk(chunk) section with:

# Process chunk using bulk ingestion with proper dual learning
# We need to process token-by-token to maintain pre/post states
from baby.information import ψ_inv, bytes_to_token_id

buffer = []
pre_state = agent.engine.current_state_index

for byte_val in chunk:
    intron = byte_val  # Already masked
    buffer.append(intron)
    
    # Check if token complete
    if (ψ_inv(intron) & 0x80) == 0:  # Token boundary
        # Reconstruct token
        tok_bytes = bytes(ψ_inv(b) for b in buffer)
        try:
            token_id = bytes_to_token_id(tok_bytes)
            
            # Apply all introns to get post state
            post_state = pre_state
            for i in buffer:
                post_state = agent.engine.epistemology[post_state, i]
            
            # Learn with dual storage
            agent.engine.operator.learn_token_dual(
                token_id, pre_state, post_state, buffer[-1]
            )
            
            # Update pre_state for next token
            pre_state = post_state
            
        except Exception:
            pass
        finally:
            buffer.clear()

# Update engine state at end
agent.engine.current_state_index = pre_state
```

### 7. Remove Pruning, Add Orbit-Entropy Based Management

**File: `baby/inference.py`**

```python
# Replace prune_low_confidence_entries with orbit-aware version (around line 280)
def manage_orbit_entropy(self, max_tokens_per_orbit: int = 64) -> int:
    """
    Manage phenotype storage by orbit entropy instead of raw confidence.
    Keeps the most informative tokens per orbit.
    """
    # Group entries by orbit
    orbit_tokens = {}  # orbit_id -> [(token_id, entry)]
    
    for key, entry in self.store.iter_entries():
        state_idx, token_id = key
        
        # Get orbit representative
        orbit_id = state_idx
        if self.phenomenology_map is not None:
            orbit_id = int(self.phenomenology_map[state_idx])
        
        if orbit_id not in orbit_tokens:
            orbit_tokens[orbit_id] = []
        orbit_tokens[orbit_id].append((token_id, entry))
    
    removed_count = 0
    
    for orbit_id, tokens in orbit_tokens.items():
        if len(tokens) <= max_tokens_per_orbit:
            continue
        
        # Calculate token probabilities within orbit
        total_conf = sum(e["conf"] for _, e in tokens)
        if total_conf == 0:
            continue
        
        # Score by confidence * uniqueness
        scored_tokens = []
        for token_id, entry in tokens:
            p = entry["conf"] / total_conf
            # Higher score for rare tokens (anti-log frequency)
            uniqueness = 1.0 / (1.0 + math.log(1 + token_id))
            score = p * uniqueness
            scored_tokens.append((score, token_id, entry))
        
        # Keep top K by score
        scored_tokens.sort(reverse=True)
        
        # Remove excess tokens
        for _, token_id, entry in scored_tokens[max_tokens_per_orbit:]:
            key = entry["key"]
            if hasattr(self.store, "delete"):
                try:
                    self.store.delete(key)
                    removed_count += 1
                except Exception:
                    pass
    
    if hasattr(self.store, "commit"):
        self.store.commit()
    
    return removed_count
```

### 8. Disable Auto-Pruning Hook

**File: `baby/intelligence.py`**

```python
# Comment out or remove the auto-prune registration (around line 990)
def _register_auto_prune_hook(self) -> None:
    """
    Disabled: We now use orbit-entropy management instead of confidence pruning.
    """
    pass  # Pruning disabled to preserve trajectory richness
```

---

## TODO List

### Critical Path (Do First) ✅ **COMPLETED**
- [x] Update `baby/policies.py` to add direction bit handling in pack/unpack functions
  - **Status**: Direction bit (bit 0) properly embedded in 12-byte phenotype records
  - **Impact**: Enables dual-state storage with physics-correct encoding
- [x] Add `learn_token_dual()` method to `baby/inference.py` 
  - **Status**: Implemented but **REPLACED** with physics-correct `learn_token_preonly()`
  - **Impact**: Dual learning was causing phase mixing under canonicalization
- [x] Fix phase alignment in `process_egress()` - cache pre_state before transition
  - **Status**: Pre-state properly cached before applying closing intron
  - **Impact**: Ensures learning happens at the BU hinge (token boundary)
- [x] Fix phase alignment in `_process_epistemology_chunk()` - track token boundaries
  - **Status**: Token boundaries properly tracked for bulk processing
  - **Impact**: Maintains physics consistency in vectorized operations
- [x] Update `generate_token_exon()` to filter by direction=DIR_PRE
  - **Status**: Generation now correctly filters for pre-state entries only
  - **Impact**: Eliminates confusion between pre/post states during generation
- [x] Add orbit fallback logic to generation when no exact state match
  - **Status**: Fallback to original state if canonical representative has no candidates
  - **Impact**: Improves generation robustness using full manifold structure

### Critical Bug Fixes (Do Immediately) ✅ **COMPLETED**
- [x] Fix `learn()` early-return bug - tag brand-new entries with `_new` flag
  - **Status**: Brand-new entries tagged and always written, even when mask=0
  - **Impact**: Prevents silent failure where new phenotypes never get stored
- [x] Fix double-unmasking in replay - pass masked bytes to `bytes_to_token_id()`
  - **Status**: Replay now passes masked bytes as expected by tokenizer
  - **Impact**: Eliminates token decode failures that were dropping all learning
- [x] Move state advancement outside try block for robust progress tracking
  - **Status**: State advances even if learning fails for a token
  - **Impact**: Honest progress tracking and robust replay completion
- [x] **PHYSICS-CORRECT FIX**: Replace dual learning with PRE-only storage
  - **Status**: **REPLACED** `learn_token_dual()` with `learn_token_preonly()`
  - **Impact**: Eliminates phase mixing under canonicalization, respects BU hinge
- [x] **PHYSICS-CORRECT FIX**: Optimize storage performance (batch size, index writes)
  - **Status**: Batch size increased to 5000, periodic index writes removed
  - **Impact**: Reduces I/O churn and improves replay performance
- [x] **PHYSICS-CORRECT FIX**: Use sets for index deduplication
  - **Status**: `index_by_state` now uses sets instead of lists
  - **Impact**: O(1) deduplication vs O(n), prevents index inflation
- [x] **CRITICAL FIX**: Standardize `index_by_state` as Set[int] everywhere
  - **Status**: Fixed type annotation and all usage patterns to use sets consistently
  - **Impact**: Prevents AttributeError when loading existing indexes with lists
- [x] **CRITICAL FIX**: Skip learning on `[SEP]` instead of aliasing to previous token
  - **Status**: `[SEP]` tokens now skip learning entirely, preserving path purity
  - **Impact**: Eliminates false associations that violated BU hinge physics
- [x] **OPTIMIZATION**: Remove unused `post_state` variable in bulk processing
  - **Status**: Cleaned up dead code in `_process_epistemology_chunk()`
  - **Impact**: Slightly cleaner code, no functional impact
- [x] **PERFORMANCE OPTIMIZATION**: Add q8 quantization to write suppression
  - **Status**: Replaced floating-point comparison with q8 quantization in `learn()`
  - **Impact**: Prevents tiny float jitter from triggering unnecessary writes

### Performance Optimizations (Already Implemented) ✅ **COMPLETED**
- [x] **PERFORMANCE FIX**: Make `index_by_state` a true Set[int] everywhere
  - **Status**: `index_by_state: Dict[int, Set[int]]` implemented in `baby/policies.py` line 232
  - **Impact**: O(1) insert/contains vs O(n) list operations, prevents duplicate enumeration
- [x] **PERFORMANCE FIX**: Never "learn on SEP"; treat as boundary and skip writes
  - **Status**: SEP tokens skip learning entirely in both `process_egress()` and `_process_epistemology_chunk()`
  - **Impact**: Eliminates non-physical associations and reduces storage bloat
- [x] **PERFORMANCE FIX**: Stronger write-suppression gate (quantized confidence)
  - **Status**: q8 quantization implemented in `baby/inference.py` lines 139-147
  - **Impact**: Prevents tiny float jitter from triggering unnecessary writes
- [x] **PERFORMANCE FIX**: Keep Bloom & Index, improve their effectiveness
  - **Status**: Bloom filter and index optimizations properly implemented
  - **Impact**: Fast negative checks and efficient candidate enumeration
- [x] **CONSISTENCY FIX**: Ensure CanonicalView packed records match index keys
  - **Status**: CanonicalView.put() now sets entry["key"] = phen_key before storage
  - **Impact**: Eliminates inconsistency between packed record and index keys
- [x] **VALIDATION FIX**: Fix average confidence calculation in validate_knowledge_integrity()
  - **Status**: Now sums confidence only over unique entries, divides by len(seen)
  - **Impact**: Correct average confidence calculation that accounts for duplicates
- [x] **PHYSICS FIX**: Use real max orbit cardinality instead of magic number in generation
  - **Status**: Replaced hardcoded v_max=1000 with int(np.max(self.s2.orbit_cardinality))
  - **Impact**: Orbit weighting now uses physically grounded maximum cardinality
- [x] **PHYSICS FIX**: Make process_text_stream_leb128() learn on pre-state
  - **Status**: Changed from post-state to pre-state learning to match live/bulk paths
  - **Impact**: Ensures consistent physics across all learning paths
- [x] **CLARITY FIX**: Correct misleading token buffer comment
  - **Status**: Changed "stores RAW LEB bytes (post-ψ)" to "stores introns (internal bytes)"
  - **Impact**: Accurate documentation of what the buffer actually contains
- [x] **PERFORMANCE FIX**: Add index write throttling to reduce I/O churn
  - **Status**: Index writes now throttled to 2s intervals or 1000 new keys
  - **Impact**: Reduces disk I/O on Mac while maintaining crash safety
- [x] **CLARITY FIX**: Make assert more explicit with 0xFE instead of 254
  - **Status**: Changed assert to use hex constant for better readability
  - **Impact**: More explicit about excluding direction bit
- [x] **DOCUMENTATION FIX**: Update get_phenotype docstring to reflect canonical storage
  - **Status**: Updated comment to reflect that learn_token_preonly handles canonicalization
  - **Impact**: Code and documentation now aligned

### Enhancement Path (Do Second) ✅ **COMPLETED**
- [x] Replace step temperature with smooth sigmoid in `_temperature_from_theta()`
  - **Status**: Smooth sigmoid temperature schedule implemented
  - **Impact**: Better exploration/exploitation balance during generation
- [x] Update replay pipeline in `wikipedia_eng.py` to use dual learning
  - **Status**: **REVERTED** - dual learning was causing physics violations
  - **Impact**: Back to physics-correct PRE-only learning
- [x] Replace `prune_low_confidence_entries()` with `manage_orbit_entropy()`
  - **Status**: Orbit-entropy based management replaces confidence pruning
  - **Impact**: Preserves trajectory richness instead of arbitrary pruning
- [x] Disable auto-pruning hook in `_register_auto_prune_hook()`
  - **Status**: Auto-pruning disabled to preserve trajectory information
  - **Impact**: No more arbitrary confidence-based deletions
- [x] Add `--compact` option to wikipedia_eng.py for post-training compaction
  - **Status**: Compaction option added with proper error handling
  - **Impact**: Realizes 12-byte phenotype promise and reduces storage bloat

- [x] Increase Bloom filter capacity to 2M in PhenotypeStore.__init__
  - **Status**: Bloom filter capacity increased from 1M to 2M entries
  - **Impact**: Reduces false positives and improves lookup performance
- [x] Add token frequency weighting to reduce common word dominance
  - **Status**: Uniqueness factor added to generation scoring
  - **Impact**: Reduces bias toward common tokens, improves diversity
- [x] Switch to vectorized `agent.ingest_bulk()` for faster, more robust replay
  - **Status**: **REVERTED** - vectorized approach was less efficient than custom processing
  - **Impact**: Back to optimized token-by-token processing with physics-correct learning
- [x] Reduce batch size to 100 for faster feedback during experiments
  - **Status**: **REVERTED** - increased to 5000 for bulk replay performance
  - **Impact**: Optimized for bulk processing while maintaining physics correctness
- [ ] Add lazy token-STT for O(1) generation (mmap file) - Future enhancement

### Validation Path (Do After Retrain) 🔄 **READY FOR TESTING**
- [ ] Run `probe_candidates.py` - should show candidates > 0
  - **Expected**: PRE-only storage should provide abundant candidates at archetypal state
  - **Test**: `python toys/probe_candidates.py` after retraining
- [ ] Test with curl - should produce grammatical responses
  - **Expected**: Coherent generation with proper physics alignment
  - **Test**: `curl -X POST http://localhost:8000/generate` with prompt
- [ ] Check store size - should show ~50-100k entries for Simple Wiki
  - **Expected**: ~2x smaller than previous 258MB due to PRE-only storage
  - **Test**: Check file size after compaction
- [ ] Verify both pre and post entries exist (direction=0 and direction=1)
  - **Status**: **UPDATED** - Only PRE entries (direction=0) should exist now
  - **Test**: Verify no POST entries in compacted store
- [ ] Monitor fallback rate - should be < 5% after training
  - **Expected**: Low fallback rate due to abundant PRE candidates
  - **Test**: Monitor generation logs for fallback frequency

### Performance Expectations After Physics-Correct Fixes
- **Replay Speed**: 2-3x faster (no dual writes, optimized batching)
- **Storage Size**: ~2x smaller (PRE-only, no phase mixing)
- **Generation Quality**: Coherent responses (abundant PRE candidates)
- **System Responsiveness**: Fast loading (optimized indexing)
- **Compression Ratio**: Realize 12-byte phenotype promise after compaction

### Physics Alignment Achieved
- **BU Hinge Respect**: Learning only at token-closing intron
- **Path Dependence**: Earlier introns encoded in pre-state
- **Canonicalization Safety**: No phase mixing under UNA parity closure
- **Token Primacy**: Semantic binding uses consistent PRE phase
- **Monodromic Fold**: Non-associative learning preserved throughout

### Next Steps
1. **Retrain with physics-correct implementation**
2. **Run validation tests** (probe_candidates, curl, size check)
3. **Compact store** to realize 12-byte phenotype promise
4. **Monitor performance** and generation quality
5. **Document results** for future reference


---

## Expected Outcomes

After implementing these changes and retraining:

1. **Immediate Coherence**: First query will find hundreds of candidates at the archetypal state
2. **Rich Trajectories**: Each token stores bidirectional path information
3. **Manifold Awareness**: Generation leverages orbit structure for fallback
4. **Stable Growth**: ~2x entries (pre+post) but same file size due to bit packing
5. **Semantic Continuity**: Temperature curve ensures smooth exploration vs exploitation

The physics will finally operate in the correct temporal frame, with the manifold structure actively guiding both learning and generation.