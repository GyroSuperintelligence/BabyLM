# TODO 7: Performance Optimization Implementation

## Progress Tracking

### Completed ✅
- [x] A. Make candidate lookup O(1) per token, not a scan
- [x] B. Stop binary-searching θ when you already have the index  
- [x] C. Use bulk application for emission feedback
- [x] D. Cache the tokenizer
- [x] E. Fix the wrong origin in token divergence
- [x] F. Make the adapter non-blocking and predictable
- [x] G. Remove redundant canonicalisation layers

### In Progress 🔄
- All optimizations completed! ✅

### Not Started ⏳
- All optimizations completed! ✅

### G. Remove redundant canonicalisation layers - COMPLETED ✅

**Changes made:**
1. **Verified existing implementation** - The current code already correctly handles canonicalization
2. **Confirmed proper layering** - `AgentPool.create_agent()` sets `enable_phenomenology_storage` correctly
3. **Validated store composition** - `_create_default_store()` only adds `CanonicalView` when appropriate

**Performance impact:**
- No redundant canonicalization layers found
- Proper store composition already in place
- No performance degradation from double canonicalization

---

## What is actually going wrong

1. **Candidate retrieval sits on the hot path and likely degenerates to a full-store scan.**
   `IntelligenceEngine.generate_token_exon()` calls:

* `self.operator.store.iter_keys_for_state(rep_state_idx)`

for every token you try to emit. If the underlying store or decorator (OverlayView → ReadOnlyView → CanonicalView → OrbitStore) does not implement a true **state-indexed iterator**, it must filter by scanning either the in-RAM index or the file. With 28,827 entries this should not “hang”, but the effect compounds: you call this per emitted token, per agent, and also during ingestion feedback. On a 2015 laptop with mmapped assets and Python dispatch overhead, this becomes indistinguishable from a hang.

2. **You repeatedly perform O(log N) binary searches to compute θ even though you already have the state index.**
   In `IntelligenceEngine.process_egress()`:

```python
div = self.s2.measure_state_divergence(self._cached_state_int)
```

`measure_state_divergence()` does a `searchsorted` against the 788,986-length keys for every single intron, even though you already have `self.current_state_index`. This is needless work on the tightest loop and causes excessive page faults on your mmapped `ontology_keys.npy`.

3. **Tokenizer is loaded from disk repeatedly.**
   `baby.information._load_tokenizer()` opens `tokenizer.json` *every time* you call `encode_text()` or `decode_text()`. The adapter calls both several times per request. On your machine this dominates turn latency and can look like a stall.

4. **Generation path feeds back one byte at a time, with per-byte θ updates and per-byte STT lookups.**
   `_emit_token_with_feedback()` loops `process_egress(byte)` for each byte of the emitted token. `process_egress()` then does a full STT lookup and θ computation per byte. This is the slowest possible way to apply a token’s intron sequence given you already have:

* a vectorised chunk processor (`process_egress_bulk`)
* the token’s complete LEB128 sequence

5. **A slow or recursive `iter_keys_for_state` implementation can deadlock via decorator layering.**
   OverlayView → ReadOnlyView → CanonicalView → OrbitStore is fine provided *each layer* implements `iter_keys_for_state(state)` without falling back to `iter_entries()` or re-calling the overlay. If any layer (particularly CanonicalView) delegates incorrectly, you can get recursion or a near-infinite traversal. You have evidence of this risk: the endpoint “hangs” but the process stays alive and CPU mostly idle.

6. **`compute_token_divergence()` uses the wrong origin.**
   You hard-code `archetypal_state = 0` (index), but the archetype is not guaranteed to be index 0 (your own comment shows it can be 549,871). This does not cause the hang, but it invalidates diagnostics and any policy using that function.

---

## What to change, exactly

### A. Make candidate lookup O(1) per token, not a scan

You need a **state-indexed view** in the store layer, and a small **per-state candidate cache** in S4 to avoid re-hitting storage every token.

1. **Store: add a true state index and a dedicated iterator.**
   In `OrbitStore` (in `baby.policies`), ensure you build a map:

```python
# Built once when opening the store or loading its .idx:
# Dict[int, list[tuple[int,int]]], where key is state_index, value is list of (state_index, token_id)
self._by_state: dict[int, list[tuple[int,int]]]
```

Populate this from your on-disk index (do not scan the .bin each time). Then implement:

```python
def iter_keys_for_state(self, state_index: int):
    for k in self._by_state.get(state_index, ()):
        yield k
```

* `CanonicalView.iter_keys_for_state(state)` should **canonicalise the query** and then delegate **directly** to the base store’s `iter_keys_for_state(rep)`. It must **not** rescan or re-canonicalise keys.
* `OverlayView.iter_keys_for_state(state)` should **yield from** both public and private stores’ iterators without any filtering.

2. **S4: add a small LRU candidate cache by representative state.**
   In `IntelligenceEngine`:

```python
# add to __init__
self._cand_cache: dict[int, list[tuple[int,float,int]]] = {}
self._cand_cache_limit = 65536
self._store_mutation_epoch = 0
```

Increment `_store_mutation_epoch` at the end of `InferenceEngine.learn()` (every `put(...)` implies the corpus updated). In `generate_token_exon`:

* Compute `rep_state_idx` once.
* If `rep_state_idx in _cand_cache` and no new store mutations, use the cached candidate list.
* Else fetch candidates once from `iter_keys_for_state(rep_state_idx)`, score, keep the top K, store in cache.

This reduces storage hits to **one per representative state** rather than per token.

3. **Add a time budget and fallback.**
   Set a hard cap (e.g., 10–20 ms) on candidate retrieval and scoring. If exceeded, immediately fall back to `_generate_random_token()`. Surges on cold caches or old hardware will no longer freeze a turn.

```python
t0 = time.perf_counter()
for s_idx, token_id in store.iter_keys_for_state(rep_state_idx):
    ...
    if time.perf_counter() - t0 > 0.02:   # 20 ms
        break
```

This guarantees forward progress.

### B. Stop binary-searching θ when you already have the index

Add an index-based accessor and call it everywhere you know the index.

**`baby/information.py`**

```python
class InformationEngine:
    ...
    def measure_state_divergence_index(self, index: int) -> float:
        if self._theta_table is None:
            raise RuntimeError("Theta table missing")
        if index < 0 or index >= len(self._theta_table):
            raise IndexError("Index out of bounds")
        return float(self._theta_table[index])
```

**`baby/intelligence.py`**

* In `process_egress()` replace:

```python
div = self.s2.measure_state_divergence(self._cached_state_int)
```

with:

```python
div = self.s2.measure_state_divergence_index(self.current_state_index)
```

* In `_process_epistemology_chunk()` replace the final and mid-sample θ calls with the index-based version:

```python
div = self.s2.measure_state_divergence_index(self.current_state_index)
...
div_mid = self.s2.measure_state_divergence_index(mid_state)
```

This removes hundreds to thousands of binary searches per turn.

### C. Use bulk application for emission feedback - COMPLETED ✅

**Changes made:**
1. **Fixed `_emit_token_with_feedback()`** - Replaced per-byte loop with single `process_egress_bulk(token_bytes)` call
2. **Fixed `respond()`** - Replaced per-byte ingestion loop with `process_egress_bulk(data)` call
3. **Added performance instrumentation** - Logs `[probe]` messages for bulk feedback timing

**Performance impact:**
- Eliminates N per-byte cycles where N = token byte length
- Applies same physics in one vectorized pass instead of N individual calls
- Reduces token parsing overhead from N cycles to 1 cycle
- Significantly reduces latency on the 2015 MBP

---

### E. Fix the wrong origin in token divergence - COMPLETED ✅

**Changes made:**
1. **Fixed `compute_token_divergence()` function** - Added `origin_index` parameter instead of hard-coding `archetypal_state = 0`
2. **Updated function signature** - Now accepts the correct origin index as parameter
3. **Removed hard-coded assumption** - No longer assumes archetype is at index 0

**Performance impact:**
- Restores correctness to divergence diagnostics
- Enables proper temperature gating and exploration heuristics
- Fixes potential issues with divergence-based policies

---

### F. Make the adapter non-blocking and predictable - COMPLETED ✅

**Changes made:**
1. **Added `run_in_threadpool` wrapper** to both `/v1/chat/completions` and `/generate` endpoints
2. **Added tokenizer priming** to the warmup function to pre-load the tokenizer cache
3. **Guaranteed event loop responsiveness** - CPU-bound operations no longer block the event loop

**Performance impact:**
- Prevents the event loop from appearing "hung" during long operations
- Ensures the server remains responsive even during heavy computation
- Eliminates first-turn tokenizer loading penalty
- Provides predictable response times

---

## Instrumentation to prove it

Add **cheap time stamps** around the suspected bottlenecks and print one-line timings on first N calls to avoid log floods.

1. **Instrument `iter_keys_for_state` calls** in `IntelligenceEngine.generate_token_exon`:

```python
if not hasattr(self, "_probe_iters"):
    self._probe_iters = 0

t0 = time.perf_counter()
for ... in self.operator.store.iter_keys_for_state(rep_state_idx):
    ...
t1 = time.perf_counter()
if self._probe_iters < 10:
    self._probe_iters += 1
    print(f"[probe] iter_keys_for_state(state={rep_state_idx}) took {(t1-t0)*1e3:.2f} ms, pulled={pulled}")
```

2. **Instrument θ calculation** in `process_egress` (before/after the change) to see removed time.

3. **Log token feedback path** once:

```
```