# Critical Fixes TODO List

## CRITICAL PRIORITY (Fix Before Running)

### 1. ✅ Vectorized Epistemology Final State Bug - FIXED
**Issue**: In `_process_epistemology_chunk`, the final state update is incorrect - it only advances one step instead of using the computed trajectory.

**Location**: `baby/intelligence.py` lines 400-410

**Fix Applied**:
```python
# Fixed the final update block:
if n > 0:
    final_pre_state = int(st[n - 1])     # state *before* applying the last intron
    final_intron = int(introns[-1])
    self.current_state_index = int(self.epistemology[final_pre_state, final_intron])
```

**Impact**: This makes the vectorized path consistent with `process_egress()`.

### 2. ✅ AgentPool TTL Eviction Bug - FIXED
**Issue**: `_evict_expired_agents()` uses `self.agent_ttl_minutes` but it's never set.

**Location**: `baby/intelligence.py` AgentPool `__init__` method

**Fix Applied**:
```python
# Added in AgentPool.__init__:
self.agent_ttl_minutes = 0
if self.eviction_policy == "ttl":
    self.agent_ttl_minutes = cast(Dict[str, Any], self.preferences).get("agent_ttl_minutes", 30)
```

**Impact**: Prevents AttributeError when TTL eviction is enabled.

### 3. ✅ Auto-Prune Hook Preferences Bug - FIXED
**Issue**: Auto-prune hook reads preferences at wrong level, missing the JSON layout.

**Location**: `baby/intelligence.py` lines 590-625

**Fix Applied**:
```python
def _register_auto_prune_hook(self) -> None:
    pruning_cfg = self.preferences.get("pruning", {})
    if pruning_cfg.get("enable_auto_decay", False):
        self.add_hook(self._auto_prune_hook)

def _auto_prune_hook(self, engine, phenotype_entry, last_token_byte, token_id=None, state_index=None) -> None:
    if self.cycle_count % 100 != 0:
        return

    pruning_cfg = self.preferences.get("pruning", {})
    if not pruning_cfg.get("enable_auto_decay", False):
        return

    thr = float(pruning_cfg.get("confidence_threshold", 0.05))
    try:
        removed = self.operator.prune_low_confidence_entries(confidence_threshold=thr)
        if getattr(self, "debug_mode", False):
            print(f"Auto-pruned {removed} low-confidence entries (threshold: {thr})")
    except Exception as e:
        print(f"Warning: Auto-prune failed: {e}")
```

**Impact**: Fixes auto-pruning configuration reading.

### 4. ✅ Path Roots Inconsistency - FIXED
**Issue**: Inconsistent path resolution (`parents[1]` vs `parents[2]`, hard-coded `BabyLM`).

**Locations**: Multiple files with hard-coded path assumptions

**Fix Applied**:
- Removed hard-coded `BabyLM` references
- Used `self.base_path` consistently
- Updated path resolution in `IntelligenceEngine.__init__` and `information._load_tokenizer`

**Impact**: Prevents "asset not found" errors in different environments.

### 5. ✅ Token Generation Candidate Selection Bug - FIXED
**Issue**: Returns "first" not "best" when ≤3 candidates.

**Location**: `baby/intelligence.py` lines 695-700

**Fix Applied**:
```python
if not candidates:
    return self._generate_random_token()

# Always sort by score
candidates.sort(key=lambda x: x[1], reverse=True)

if temperature < 0.1 or len(candidates) <= 3:
    return candidates[0][0]
```

**Impact**: Ensures best candidate is always selected.

### 6. ✅ Confidence Decay Helpers - FIXED
**Issue**: Check for `.log` file that PhenotypeStore never creates.

**Location**: `baby/policies.py` lines 1052-1053, 1116-1117

**Fix Applied**: Replaced `.log` existence check with direct file existence check.

### 7. ✅ NEW: Shared Public Store Closed by Each Agent - FIXED
**Issue**: `OverlayView.close()` closes the shared `public_store`, so evicting a single agent can brick the rest of the pool.

**Location**: `baby/policies.py` OverlayView class

**Fix Applied**:
```python
def close(self) -> None:
    # Do NOT close public_store here; in AgentPool it is shared across agents.
    if self.private_store is not None:
        self.private_store.close()
        self.private_store = None

def close_public(self) -> None:
    # Explicit opt-in to close the shared resource if the owner wants to.
    if self.public_store is not None:
        self.public_store.close()
        self.public_store = None
```

**Impact**: Prevents shared store from being closed when individual agents are evicted.

### 8. ✅ NEW: θ Buffer Not Updated in Bulk Ingest - FIXED
**Issue**: `process_egress_bulk()` doesn't push into `_θ_buf`, yet generation uses θ for temperature; generation quality will diverge between bulk vs scalar paths.

**Location**: `baby/intelligence.py` `_process_epistemology_chunk` method

**Fix Applied**: Added θ updates during bulk processing:
```python
# θ update: push at least the final divergence (cheap and keeps temperature sane)
try:
    final_state_int = self.s2.get_state_from_index(self.current_state_index)
    div = self.s2.measure_state_divergence(final_state_int)
    self._θ_buf.append(div)
except Exception:
    pass
```

**Impact**: Ensures consistent temperature-based generation between scalar and bulk paths.

### 9. ✅ NEW: `theta.npy` Path Derivation Can Be Wrong - FIXED
**Issue**: You compute the theta path from the ontology path; the builder emits `theta.npy` alongside **epistemology** instead.

**Location**: `baby/intelligence.py` IntelligenceEngine `__init__` method

**Fix Applied**: Derived theta from epistemology_path:
```python
ep_path_resolved = _abs(epistemology_path or "memories/public/meta/epistemology.npy", self.base_path)
self.s2 = InformationEngine(
    keys_path=self.ontology_path,
    ep_path=ep_path_resolved,
    phenomap_path=_abs(phenomenology_map_path or "memories/public/meta/phenomenology_map.npy", self.base_path),
    # theta is emitted alongside the epistemology
    theta_path=str(Path(ep_path_resolved).with_name("theta.npy")),
)
```

**Impact**: Prevents "theta missing" error in realistic layouts.

### 10. ✅ NEW: Token Candidates Scan is Effectively Random - FIXED
**Issue**: `generate_token_exon()` scans **first 50** entries from `iter_entries()` and filters by state — due to dict order you'll often get no matches → random token.

**Location**: `baby/intelligence.py` `generate_token_exon` method

**Fix Applied**: Added fast helper to iterate keys by state and used it:
```python
# In PhenotypeStore class:
def iter_keys_for_state(self, state_idx: int) -> Iterator[Tuple[int, int]]:
    with self.lock:
        for (s_idx, tok_id) in self.index.keys():
            if s_idx == state_idx:
                yield (s_idx, tok_id)

# In generate_token_exon:
# Pull candidates directly via the index: O(k) in number of tokens for this state.
fetch_limit = 512  # bounded to avoid pathological huge states
pulled = 0
for (s_idx, token_id) in getattr(self.operator.store, "iter_keys_for_state", lambda _s: [])(rep_state_idx):
    if pulled >= fetch_limit:
        break
    entry = self.operator.store.get((s_idx, token_id))
    if not entry:
        continue
    pulled += 1
    # ... rest of candidate processing
```

**Impact**: Moves from "usually random" to "almost always using learned tokens."

### 11. ✅ NEW: Default Phenotype Mask = Low 8 Bits of Token ID - FIXED
**Issue**: That's a semantic footgun: an arbitrary token id leaks into a mask whose bit families (LI/FG/BG) are meaningful to physics.

**Location**: `baby/inference.py` `_create_default_phenotype` method

**Fix Applied**: Started with neutral mask:
```python
def _create_default_phenotype(self, context_key: Tuple[int, int]) -> PhenotypeEntry:
    # Start with a neutral mask; learning will move it via Monodromic Fold.
    return cast(
        PhenotypeEntry,
        {
            "mask": 0x00,
            "conf": 0.1,
            "key": context_key,
        },
    )
```

**Impact**: Prevents arbitrary token-id noise from biasing learning in weird ways.

## HIGH PRIORITY (Likely Issues)

### 12. Token Generation Store Scanning
**Issue**: `generate_token_exon` scans whole store, stops after 50 global entries (not 50 matches).

**Location**: `baby/intelligence.py` lines 650-680

**Fix Required**: Use store index to pre-filter by state_index before scanning.

### 13. Dead Parameters Cleanup
**Issue**: Several unused parameters (`max_age_days`, `_bloom_loaded`, `masked_arr`).

**Locations**: Various files

**Fix Required**: Remove or implement unused parameters.

### 14. NEW: Use Real `v_max` Instead of Hardcoded 1000
**Issue**: In several places you set `v_max = 1000` but already compute the actual maximum in `InformationEngine.__init__`.

**Location**: `baby/intelligence.py` `generate_token_exon` method

**Fix Required**:
```python
v_max = getattr(self.s2, "_v_max", 1000) or 1000
```

**Impact**: More accurate orbit cardinality calculations.

### 15. NEW: Resonance Scoring with More Physical Measure
**Issue**: Current `_calculate_resonance` compares state index bits to 8-bit mask. State index is arbitrary.

**Location**: `baby/intelligence.py` `_calculate_resonance` method

**Fix Required**: Compare physical state integer's low byte to mask:
```python
# Compare physical state's low byte to mask as a crude, fast resonance proxy.
try:
    state_int = self.s2.get_state_from_index(state_index)
except Exception:
    state_int = state_index  # fallback
low_byte = state_int & 0xFF
hd = bin((low_byte ^ (mask & 0xFF)) & 0xFF).count("1")
base_resonance = 1.0 - (hd / 8.0)
```

**Impact**: Resonance tied to physics, not arbitrary indexing.

## MEDIUM PRIORITY (Nice-to-Haves)

### 16. Resonance vs Mask Alignment
**Issue**: Currently compares 16 bits of state index to 8-bit mask.

**Location**: `baby/intelligence.py` `_calculate_resonance` method

**Fix Required**: Consider using state integer (48 bits) for more meaningful alignment.

### 17. Append-Only LRU Cache
**Issue**: Never used on hot path.

**Location**: `baby/policies.py`

**Fix Required**: Either remove or integrate into append-only scan path.

### 18. Bloom Filter Persistence
**Issue**: Only saves on close, not periodic commits.

**Location**: `baby/policies.py`

**Fix Required**: Consider saving on periodic commits too.

### 19. NEW: Fix Path Resolution in GyroSI.__init__
**Issue**: Hardcoded `BabyLM` reference can double the folder.

**Location**: `baby/intelligence.py` GyroSI `__init__` method

**Fix Required**:
```python
# Replace:
project_root = Path(__file__).resolve().parents[2] / "BabyLM"
self.config["ontology_path"] = str(project_root / str(self.config["ontology_path"]))

# With:
self.config["ontology_path"] = str(Path(self.base_path) / str(self.config["ontology_path"]))
```

### 20. NEW: Normalize Terminology in Docstrings
**Issue**: Some docstrings still say "unmask back to byte stream" when converting introns → masked LEB128 bytes.

**Location**: Various files

**Fix Required**: Quick wording pass to reduce cognitive hiccup for new readers.

### 21. NEW: InformationEngine Asset Loading Robustness
**Issue**: Missing clearer error messages for missing assets.

**Location**: `baby/information.py` InformationEngine class

**Fix Required**: Add helpful messages that tell users exactly how to build missing assets.

## IMPLEMENTATION ORDER

1. **Shared Public Store Close** (Critical - prevents pool corruption)
2. **θ Buffer Updates in Bulk** (Critical - affects generation quality)
3. **theta.npy Path Fix** (Critical - prevents missing asset errors)
4. **Token Candidates Fast Path** (Critical - improves generation quality)
5. **Default Phenotype Mask** (Critical - prevents learning bias)
6. **Use Real v_max** (High - improves accuracy)
7. **Resonance Scoring** (High - improves physics alignment)
8. **Path Resolution in GyroSI** (Medium - prevents path issues)
9. **Terminology Normalization** (Medium - improves clarity)
10. **Asset Loading Robustness** (Medium - improves user experience)
11. **Dead Parameters Cleanup** (Medium - code cleanup)
12. **Resonance vs Mask Alignment** (Medium - potential improvement)
13. **Append-Only LRU Cache** (Medium - potential optimization)
14. **Bloom Filter Persistence** (Medium - potential optimization)

## TESTING CHECKLIST

After implementing fixes, run these sanity tests:

1. **Vectorized == scalar state**: Feed same byte blob through `process_egress()` and `process_egress_bulk()`, assert equal results.

2. **Round-trip token physics**: For 1,000 random tokens, verify `introns = token_to_introns(tok); tok2 = introns_to_token(introns)` → `tok2 == tok`.

3. **Candidate scan hits**: Insert entries for known `rep_state_idx`, call `generate_token_exon(rep_state_idx, temperature=0.0)` multiple times, ensure it selects among those.

4. **AgentPool TTL**: Enable TTL policy with 0.01-minute TTL, create agents, sleep, ensure eviction fires without exceptions.

5. **Auto-prune**: Set `pruning.enable_auto_decay=true`, threshold high enough to prune known entries, verify pruning counts.

6. **NEW: Shared store integrity**: Delete agents to force fresh pool, ensure public store remains accessible.

7. **NEW: Bulk vs scalar θ consistency**: Ingest via scalar and bulk, then generate - ensure similar temperature behavior.

8. **NEW: theta.npy path**: Confirm theta.npy sits next to epistemology.npy.

9. **NEW: Token candidate quality**: With learned entries for a state, check that generation uses learned tokens instead of random.

10. **NEW: End-to-end test**: Run `orchestrate_turn()` with a handful of steps.

## NOTES

- Test each fix individually before moving to next
- Document any changes that affect the public API
- Consider the impact on existing stored data
- Verify that fixes don't introduce new issues 