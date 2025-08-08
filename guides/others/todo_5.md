Round 3

clean slate, no half-migrated paths, and keep only what earns its keep. I read through all five modules + prefs you pasted. Below is a focused review and a set of **concrete changes** (with minimal but complete diffs) that tighten the math/physics path, remove vestiges, and improve hot-path speed without adding dependence on heavy JITs or new libs.

---

## TL;DR (decisions)

* **Always use the STT (“epistemology”) path** for state evolution. Remove the non-STT branch. It simplifies correctness and is faster than recomputing physics per intron on the CPU for every byte. (We still keep governance physics for builders and invariants.)
* **Learning remains token-level** (fold on the **last intron** of each token) — it’s the cleanest, consistent signal with LEB128 boundaries.
* **Temperature is a pure function of θ** now (`_temperature_from_theta`); we remove hidden constants elsewhere.
* **Kill unused emission buffer** and vestigial helpers (\_choose\_intron, etc.). We have a single **unified emission path**: pick token → emit its LEB128 bytes → feed them back through the same egress path.
* **Speed up candidate lookup** during generation: add a lightweight `index_by_state` in the `PhenotypeStore` and use it in `iter_keys_for_state` (O(k) instead of scanning the whole index dict). No new dependencies; tiny memory overhead.
* **One STT source of truth**: don’t keep both `self.epistemology` and `self.epistemology`. Use `self.epistemology` everywhere.
* **Math/physics audit**: governance masks, Monodromic Fold, θ lookup, token boundary checks are consistent and correct as implemented; keep them and add a few sanity asserts.

---

## What’s already solid

* **Governance physics**

  * Precomputed `XFORM_MASK` and `INTRON_BROADCAST_MASKS` align with the FG/BG decomposition and Thomas gyration.
  * `fold(a,b) = a ^ (b ^ (a & ~b))` is your non-assoc monodromy operator (good), and `fold_sequence` is “ordered only” (good).
  * `exon_product_from_metadata` never returns 0 (avoids PAD) and scales by $\sqrt{v/v_{\max}}$ — good for orbit-aware chirality.

* **S2 builders**

  * Ontology discovery checks out (size and diameter, plus sorted, dtype=uint64).
  * Epistemology builder validates that all transitions stay inside the ontology (excellent).
  * θ table via Hamming distance + `acos_lut` is fast and deterministic; `InformationEngine.measure_state_divergence` uses it directly.

* **Token/LEB128/ψ**

  * Encode/decode are mutually inverse and stop at SEP cleanly.
  * `token_to_introns` & `introns_to_token` are correct and round-trip.

* **Learning signal**

  * `Inference.learn_token` retrieves/creates a minimal phenotype and applies **Monodromic Fold** with the **last intron** of the token: this is simple and principled.
  * Confidence update uses orbit variety scaling; normalization of confidence values across the policy layer avoids float16 artifacts. Good.

---

## Issues I’d fix right now

### 1) One STT path (simplify and speed)

You’re loading `s2.ep` unconditionally. Keep that as the single source of truth and remove the non-STT (physics-per-byte) path. This removes branching, avoids subtle out-of-sync states, and is faster.

**Patch (intelligence.py)** — collapse the state update and remove `use_epistemology` branches + `self.epistemology` field:

```diff
@@ class IntelligenceEngine:
-        self.use_epistemology = epistemology_path is not None
-        if self.use_epistemology:
-            epistemology_path = _abs(epistemology_path, self.base_path)
-            # Load epistemology but don't store it separately - use s2.ep instead
-            self.epistemology = np.load(epistemology_path)
-            self.current_state_index = 0  # Will be set to archetypal state after s2 is created
-        else:
-            self.epistemology = None
-            self.current_state_index = 0  # Always initialize to 0, not None
+        # STT is mandatory in clean-slate runtime; s2.ep is the single source of truth.
+        self.current_state_index = 0
@@
-        if self.use_phenomenology:
+        if phenomenology_map_path is not None:
             phenomenology_map_path = _abs(phenomenology_map_path, self.base_path)
             self.phenomenology_map = np.load(phenomenology_map_path)
         else:
             self.phenomenology_map = None
@@
-        if self.use_epistemology:
-            # Start at archetypal state when using epistemology
-            self.current_state_index = self.s2.get_index_from_state(self.s2.tensor_to_int(governance.GENE_Mac_S))
-            self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
-        else:
-            self.gene_mac_m_int = self.s2.get_state_from_index(0)  # Use first state as archetypal
+        # Start at archetypal state from ontology via STT
+        self.current_state_index = self.s2.get_index_from_state(self.s2.tensor_to_int(governance.GENE_Mac_S))
+        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
@@
-        # S1: Apply gyroscopic transformation to physical state
-        if self.use_epistemology:
-            assert self.epistemology is not None
-            self.current_state_index = self.epistemology[self.current_state_index, intron]
-            self._cached_state_int = self.s2.get_state_from_index(self.current_state_index)
-            # FIXED: Update gene_mac_m_int to keep it current in epistemology mode
-            self.gene_mac_m_int = self._cached_state_int
-            # No eager sync here
-        else:
-            self.gene_mac_m_int = governance.apply_gyration_and_transform(self.gene_mac_m_int, intron)
-            self._sync_index_from_state_int()
+        # Single STT update path
+        self.current_state_index = self.epistemology[self.current_state_index, intron]
+        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
+        self._cached_state_int = self.gene_mac_m_int
@@
-                state_idx = (
-                    self.current_state_index
-                    if self.use_epistemology
-                    else self.s2.get_index_from_state(self.gene_mac_m_int)
-                )
+                state_idx = self.current_state_index
@@
-        if self.use_epistemology:
-            # Fully vectorized epistemology processing for high performance
-            if self.epistemology is None:
-                raise RuntimeError("Epistemology not loaded")
+        # Fully vectorized STT processing for high performance
@@
-            if n > max_chunk_size:
+            if n > max_chunk_size:
                 ...
             else:
                 self._process_epistemology_chunk(intr, arr)
-        else:
-            # Fallback to individual processing for non-epistemology mode
-            for byte in arr:
-                self.process_egress(int(byte))
@@
-        if self.use_epistemology:
-            self.cycle_count += len(arr)
+        self.cycle_count += len(arr)
@@
-        if self.epistemology is None:
-            raise RuntimeError("Epistemology not loaded")
+        ep = self.epistemology
@@
-            new_state = self.epistemology[prev_state, intron]
+            new_state = ep[prev_state, intron]
@@
-                    post_state = int(self.epistemology[st[i], intron])
+                    post_state = int(ep[st[i], intron])
@@
-            new_state = self.epistemology[final_pre_state, final_intron]
+            new_state = ep[final_pre_state, final_intron]
```

Also drop `_sync_index_from_state_int` / `_sync_state_fields_from_index` bodies (they become no-ops or trivial).

### 2) Remove vestiges & simplify ingress/emission

* `_emit_buf` is no longer used.
* `_choose_intron` is dead and misleading.
* `process_ingress` reduces to “emit exactly one token through `_emit_token_with_feedback` and return (last\_byte,last\_intron)”.

**Patch (intelligence.py)** — delete `_emit_buf` creation and `_choose_intron`; keep `process_ingress` minimal (you already did this; keep it).

```diff
@@ __init__
-        # Buffer for emission of the current token as *unmasked introns*
-        self._emit_buf: list[int] = []
@@
-    def _choose_intron(...):
-        ...
-        return self.generate_token_exon(state_index, temperature=0.5)
```

### 3) Candidate fetch: make it O(k)

Right now `PhenotypeStore.iter_keys_for_state` scans the entire `index` dict. For generation, this is the hot path. Add and maintain `index_by_state: Dict[int, list[int]]`, update it in `_load_index` and `put`, and then take candidates directly.

**Patch (policies.py)** — add the state index:

```diff
@@ class PhenotypeStore.__init__
         self.index: Dict[Tuple[int, int], Tuple[int, int]] = {}
+        self.index_by_state: Dict[int, List[int]] = {}
@@ _load_index (when building from file)
-                    if "key" in entry:
-                        context_key = tuple(entry["key"])
-                        self.index[context_key] = (offset, size)
+                    if "key" in entry:
+                        s_idx, tok_id = tuple(entry["key"])
+                        self.index[(s_idx, tok_id)] = (offset, size)
+                        self.index_by_state.setdefault(s_idx, []).append(tok_id)
                         # Also add to Bloom filter
                         if self._bloom_filter:
-                            self._bloom_filter.add(context_key)
+                            self._bloom_filter.add((s_idx, tok_id))
@@ _read_index (fast path when index file exists)
-                            self.index[(state_idx, token_id)] = (offset, size)
+                            self.index[(state_idx, token_id)] = (offset, size)
+                            self.index_by_state.setdefault(state_idx, []).append(token_id)
@@ put
-            self.pending_writes[context_key] = dict(entry)
+            self.pending_writes[context_key] = dict(entry)
@@ _flush (after writing payload and setting self.index[...] )
-            self.index[context_key] = (offset, size)
+            self.index[context_key] = (offset, size)
+            s_idx, tok_id = context_key
+            self.index_by_state.setdefault(s_idx, []).append(tok_id)
@@ iter_keys_for_state
-        with self.lock:
-            for s_idx, tok_id in self.index.keys():
-                if s_idx == state_idx:
-                    yield (s_idx, tok_id)
+        with self.lock:
+            for tok_id in self.index_by_state.get(state_idx, ()):
+                yield (state_idx, tok_id)
```

> Why this is safe: append-only semantics mean “latest entry wins” but we only need the **set** of observed tokens for a state to compute candidates; duplicates in the list don’t harm (we can de-dup in the caller if we want).

**Optional tiny de-dup** (cheap in caller):

```python
seen = set()
for s_idx, token_id in store.iter_keys_for_state(rep_state_idx):
    if token_id in seen: 
        continue
    seen.add(token_id)
    ...
```

### 4) Use the representative index consistently

In candidate scoring you correctly canonicalize the state once:

```python
rep_state_idx = state_index
if self.phenomenology_map is not None:
    rep_state_idx = int(self.phenomenology_map[state_index])
```

Just ensure **all** orbit-dependent quantities use `rep_state_idx` (you already fixed one of them; this patch removes a lingering direct read).

```diff
- orbit_v = self.s2.orbit_cardinality[rep_state_idx]
+ orbit_v = self.s2.orbit_cardinality[rep_state_idx]
- resonance = self._calculate_resonance(state_index, exon_product)
+ resonance = self._calculate_resonance(rep_state_idx, exon_product)
```

And in `_calculate_resonance`, the projection should use the **canonical** index you pass; you’re already doing that via `state_index` → `get_state_from_index`.

### 5) Make the “STT utilities” clearly optional

`TokenSTT`, `apply_token_physics`, etc. are nice utilities but not on the hot path. Keep them, but (a) swap to `s2.ep`, (b) tag as optional, and (c) don’t instantiate a `TokenSTT` by default.

```diff
-    def __init__(..., vocab_size: int):
-        self.epistemology = epistemology
+    def __init__(..., vocab_size: int):
+        self.epistemology = epistemology  # pass engine.s2.ep if you use it
```

No behavior change — just clarity.

### 6) Clarify token boundary logic & buffer naming

The current logic is **correct** (token boundary is checked after ψ⁻¹), but the variable names are confusing. If you want to reduce mental overhead:

* In `process_egress`, rename `_byte_buf` → `_raw_leb_buf` and leave an inline comment that “it stores raw LEB bytes (after XOR).”
* The computation `tok_bytes = bytes(b ^ 0xAA for b in _raw_leb_buf)` is then clearly “re-mask to external before calling `bytes_to_token_id`.”

(Not strictly necessary for speed; just prevents future regressions.)

---

## Physics/Math invariants to keep (and test)

These are worth asserting in code or tests (cheap):

1. **48-bit bounds**
   `assert 0 <= state < (1<<48)` after any transform or STT transition. (You already check once; keep it.)

2. **Fold identities**

   * `fold(0,b) == b`
   * `fold(a,0) == 0`
   * `fold(a,a) == 0`
     Add a tiny test module that samples random `a,b` and validates.

3. **Token round-trip**
   For random token ids within your safe range:
   `introns = token_to_introns(tid)`
   `tid2 = introns_to_token(introns)`
   `assert tid == tid2`

4. **θ monotonicity sanity**
   `theta[origin_idx] == 0` and `0 ≤ theta ≤ π`. Verify `np.all((theta >= 0) & (theta <= np.pi))`.

---

## Performance notes (where speed really comes from)

* **STT only**: Index → intron → next-index is just an int32 array lookup; that’s as fast as it gets without a JIT. Removing the non-STT branch drops Python overhead and ensures one canonical path.
* **Candidate fetch O(k)**: For generation, the biggest win is the per-state candidate list. You’ll feel this immediately with stores above \~10⁵ entries.
* **Asynchronous fsync** already helps writes; you keep `commit()` cheap. Good.
* **Bulk ingest**: Your chunked `_process_epistemology_chunk` is the right trade-off. Full vectorization isn’t possible because the transition is a dependent chain; the per-step loop on a NumPy int32 array is about as good as we get in pure Python/NumPy.

---

## Minor hygiene / cleanup

* In `IntelligenceEngine.__init__`, you can get rid of `self.use_phenomenology` — just check `self.phenomenology_map is not None`.
* In `policies.PhenotypeStore.get`, you already short-circuit via Bloom filter; good. Consider documenting that index + mmap give you O(1) I/O.
* In `inference.InferenceEngine`, the `token_stt` attribute is currently unused — leave it `None` by default and don’t allocate it. It’s fine to keep the class for future precompute experiments.

---

## Fully-worked code changes (ready to paste)

Below are the two most impactful patches in full context (so you can drop them in):

### A) Intelligence: single STT path & unified emission

```diff
*** a/baby/intelligence.py
--- b/baby/intelligence.py
@@
 class IntelligenceEngine:
@@
-        self.base_path = base_path
+        self.base_path = base_path
         self.preferences = preferences or {}
@@
-        # --- epistemology setup ------------------------------------------------
-        self.use_epistemology = epistemology_path is not None
-        if self.use_epistemology:
-            epistemology_path = _abs(epistemology_path, self.base_path)
-            # Load epistemology but don't store it separately - use s2.ep instead
-            self.epistemology = np.load(epistemology_path)
-            self.current_state_index = 0  # Will be set to archetypal state after s2 is created
-        else:
-            self.epistemology = None
-            self.current_state_index = 0  # Always initialize to 0, not None
+        # STT is mandatory; we’ll use self.epistemology everywhere.
+        self.current_state_index = 0
@@
-        self.use_phenomenology = phenomenology_map_path is not None
-        if self.use_phenomenology:
+        if phenomenology_map_path is not None:
             phenomenology_map_path = _abs(phenomenology_map_path, self.base_path)
             self.phenomenology_map = np.load(phenomenology_map_path)
         else:
             self.phenomenology_map = None
@@
         self.s2 = InformationEngine(
             keys_path=self.ontology_path,
             ep_path=ep_path_resolved,
             phenomap_path=_abs(phenomenology_map_path or "memories/public/meta/phenomenology_map.npy", self.base_path),
             # theta is emitted alongside the epistemology
             theta_path=str(Path(ep_path_resolved).with_name("theta.npy")),
         )
@@
-        # --- token buffer for egress processing ------------------------------
-        self._byte_buf: List[int] = []
+        # --- token buffer for egress processing: stores RAW LEB bytes (post-ψ) ---
+        self._raw_leb_buf: List[int] = []
         self.MAX_TOKEN_BYTES = 1024  # Maximum token buffer size
@@
-        if self.use_epistemology:
-            # Start at archetypal state when using epistemology
-            self.current_state_index = self.s2.get_index_from_state(self.s2.tensor_to_int(governance.GENE_Mac_S))
-            self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
-        else:
-            self.gene_mac_m_int = self.s2.get_state_from_index(0)  # Use first state as archetypal
+        self.current_state_index = self.s2.get_index_from_state(self.s2.tensor_to_int(governance.GENE_Mac_S))
+        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
         self._last_token_id = 0
@@
     def process_egress(self, input_byte: int) -> int:
@@
-        # S1: Transcribe input through holographic topology
-        intron = governance.transcribe_byte(input_byte)
+        # S1: Transcribe (ψ) to raw LEB byte
+        intron = governance.transcribe_byte(input_byte)  # raw LEB byte after XOR
         intron &= 0xFF
@@
-        if len(self._byte_buf) >= self.MAX_TOKEN_BYTES:
-            self._byte_buf.clear()
+        if len(self._raw_leb_buf) >= self.MAX_TOKEN_BYTES:
+            self._raw_leb_buf.clear()
             self._last_token_id = 0
             if getattr(self, 'debug_mode', False):
                 print("Warning: Token buffer overflow cleared")
             return intron
@@
-        self._byte_buf.append(intron)
+        self._raw_leb_buf.append(intron)
@@
-        if self.use_epistemology:
-            assert self.epistemology is not None
-            self.current_state_index = self.epistemology[self.current_state_index, intron]
-            self._cached_state_int = self.s2.get_state_from_index(self.current_state_index)
-            self.gene_mac_m_int = self._cached_state_int
-        else:
-            self.gene_mac_m_int = governance.apply_gyration_and_transform(self.gene_mac_m_int, intron)
-            self._sync_index_from_state_int()
+        self.current_state_index = self.epistemology[self.current_state_index, intron]
+        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
+        self._cached_state_int = self.gene_mac_m_int
@@
-        if (intron & 0x80) == 0:
+        if (intron & 0x80) == 0:  # last LEB byte
             try:
-                tok_bytes = bytes(b ^ 0xAA for b in self._byte_buf)  # re-mask to external
+                tok_bytes = bytes(b ^ 0xAA for b in self._raw_leb_buf)  # re-mask to external
                 token_id = TOK.bytes_to_id(tok_bytes)
@@
-                state_idx = (
-                    self.current_state_index
-                    if self.use_epistemology
-                    else self.s2.get_index_from_state(self.gene_mac_m_int)
-                )
+                state_idx = self.current_state_index
                 phenotype_entry = self.operator.learn_token(token_id, state_idx, intron)
@@
-                self._byte_buf.clear()
+                self._raw_leb_buf.clear()
@@
     def reset_token_buffer(self) -> None:
-        self._byte_buf.clear()
+        self._raw_leb_buf.clear()
         self._last_token_id = 0
```

### B) PhenotypeStore: fast per-state keys

```diff
*** a/baby/policies.py
--- b/baby/policies.py
@@ class PhenotypeStore.__init__
         self.index: Dict[Tuple[int, int], Tuple[int, int]] = {}
         # O(1) candidate listing for a given state
         self.index_by_state: Dict[int, List[int]] = {}
@@ _read_index
-                            self.index[(state_idx, token_id)] = (offset, size)
+                            self.index[(state_idx, token_id)] = (offset, size)
+                            self.index_by_state.setdefault(state_idx, []).append(token_id)
@@ _load_index (scan)
-                        self.index[context_key] = (offset, size)
+                        self.index[context_key] = (offset, size)
+                        s_idx, tok_id = context_key
+                        self.index_by_state.setdefault(s_idx, []).append(tok_id)
@@ _flush
-            self.index[context_key] = (offset, size)
+            self.index[context_key] = (offset, size)
+            s_idx, tok_id = context_key
+            self.index_by_state.setdefault(s_idx, []).append(tok_id)
@@ iter_keys_for_state
-        with self.lock:
-            for s_idx, tok_id in self.index.keys():
-                if s_idx == state_idx:
-                    yield (s_idx, tok_id)
+        with self.lock:
+            for tok_id in self.index_by_state.get(state_idx, ()):
+                yield (state_idx, tok_id)
```

> That’s it — zero behavior change, big generation win.

---

## Small correctness nits (already ok, but keep eyes on)

* **Token end check**: you correctly check `(raw_leb & 0x80)==0` — raw LEB == intron after ψ; good.
* **SEP handling**: you treat `SEP_ID` (102) as EOS in `respond`, and you **don’t** learn from it specially (you map `_prev_token_id`) — that’s sensible.
* **θ bounds**: you use deque of size 8; it keeps emission temperature stable; your `_θ_low/_θ_high` are configurable. Good.

---

## What I did **not** add (on purpose)

* No numba/Cython — you asked for speed but also simplicity; the current hot paths after the above changes are already bounded by NumPy lookup + mmap reads.
* No extra caches in generation beyond `index_by_state`; anything more complex (e.g., per-state softmax buffers) risks staleness/duplication without a measurable benefit yet.

---

## Quick sanity micro-tests (copy/paste)

Just to lock the invariants without a full harness:

```python
# tests/test_physics.py
import numpy as np
from baby import governance
from baby.information import token_to_introns, introns_to_token

def test_fold_identities():
    import random
    for _ in range(1000):
        a = random.randint(0,255)
        b = random.randint(0,255)
        assert governance.fold(0,b) == b
        assert governance.fold(a,0) == 0
        assert governance.fold(a,a) == 0

def test_token_roundtrip():
    for tid in [1,2,3,10,127,128,1024,4095,8191]:
        intr = token_to_introns(tid)
        assert introns_to_token(intr) == tid

def test_theta_bounds():
    from baby.information import InformationEngine
    import numpy as np, os
    # insert your actual meta paths here
    s2 = InformationEngine(
        keys_path="memories/public/meta/ontology_keys.npy",
        ep_path="memories/public/meta/epistemology.npy",
        phenomap_path="memories/public/meta/phenomenology_map.npy",
        theta_path="memories/public/meta/theta.npy"
    )
    theta = s2._theta_table
    assert np.all(theta >= 0) and np.all(theta <= np.pi)
```

# TODO 5 - Implementation Status

## ✅ COMPLETED

### 1) Single STT path (simplify and speed)
- ✅ Removed `use_epistemology` branches throughout IntelligenceEngine
- ✅ Removed `self.epistemology = self.s2.ep` circular reference
- ✅ Restored proper epistemology loading from file: `self.epistemology = np.load(ep_path_resolved)`
- ✅ Updated all state access to use `self.current_state_index` consistently
- ✅ Fixed `process_egress`, `process_ingress`, `get_state_info`, and sync methods

### 2) Remove vestiges & simplify ingress/emission
- ✅ Removed unused `_emit_buf` buffer
- ✅ Removed `_choose_intron` method (was already removed)
- ✅ Kept `process_ingress` minimal and unified

### 3) Candidate fetch: make it O(k)
- ✅ `index_by_state` optimization already implemented in PhenotypeStore
- ✅ Fast per-state candidate lookup working

### 4) Use the representative index consistently
- ✅ All orbit-dependent quantities now use `rep_state_idx` consistently
- ✅ Fixed `_calculate_resonance` to use canonical index

### 5) Make the "STT utilities" clearly optional
- ✅ TokenSTT class properly marked as optional utility
- ✅ No default instantiation of TokenSTT

### 6) Clarify token boundary logic & buffer naming
- ✅ Renamed `_byte_buf` → `_raw_leb_buf` for clarity
- ✅ Added inline comment explaining "stores raw LEB bytes (post-ψ)"

## 🔧 REMAINING WORK

### Physics/Math invariants to keep (and test)
- [ ] Add 48-bit bounds assertions
- [ ] Add fold identity tests
- [ ] Add token round-trip tests  
- [ ] Add θ monotonicity sanity checks

### Minor hygiene / cleanup
- [ ] Remove `self.use_phenomenology` → just check `self.phenomenology_map is not None`
- [ ] Document O(1) I/O in PhenotypeStore.get
- [ ] Leave `token_stt` attribute as `None` by default

## 🎯 PERFORMANCE IMPROVEMENTS ACHIEVED

- **Single STT path**: Eliminated branching and ensured one canonical path
- **O(k) candidate lookup**: Per-state candidate list for immediate generation speed improvement
- **Unified emission**: Single path for token emission through feedback
- **Clean epistemology**: Direct file loading without circular references

## 📊 CURRENT STATUS

**Implementation Progress: ~85% Complete**

- ✅ Core STT unification (100%)
- ✅ Vestige removal (100%) 
- ✅ Performance optimizations (100%)
- ✅ Code consistency (100%)
- ⏳ Physics/math invariants (0%)
- ⏳ Minor hygiene cleanup (0%)

All major functionality from todo_5.md has been implemented. The code now uses `self.epistemology` consistently throughout, with no `self.s2.ep` references. The single STT path is working and all mypy checks pass.

