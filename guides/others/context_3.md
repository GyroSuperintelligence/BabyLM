# GyroSI Implementation: Physics-Driven Token Learning

Here is the complete system architecture that aligns the physics, tokenizer, and five maps into a coherent implementation.

## Core Principles

1. **The tokenizer is the only symbolic index you need.**
   It already gives you a total, reversible map between text ↔ token_id ↔ LEB128 bytes. Nothing else should attempt to "index language". The trie in `tokenizer.json` is the decoder; the vocabulary is the symbol set. Keep it first-class inside the runtime, not behind it.

2. **The five meta-maps are the only world-model you need.**
   Ontology (state set), Epistemology (state transitions), Phenomenology (orbits), Θ (angular divergence), and Orbit Sizes (cardinality) are a complete physical atlas. They are not metadata to copy into a store; they are the store for physics.

3. **Therefore "knowledge" reduces to a sparse overlay of one-byte residues.**
   A phenotype is not a record; it is a minimal residue of learning: the 8-bit **exon_mask** (your holographic memory). Everything else—confidence, counts, labels, timestamps—can be derived on the fly from Θ and orbit cardinality. When there is no deviation from the orbit baseline, you store nothing.

## Implementation Architecture

### Single, Minimal Substrate

• **One append-only knowledge file**: `knowledge.bin`
  - Format: `[uLEB128 state_idx][uLEB128 n_pairs][(uLEB128 token_id + mask_byte) * n_pairs]`
  - Each record stores only the essential: `(state_index, token_id, mask)`
  - No confidences, no timestamps, no metadata
  - Achieves ~3-4 bytes per pair instead of 9+ bytes

• **Physics-driven generation**: Exon product sieve
  - Baseline exon product from state physics (`exon_product_from_state`)
  - Specialize with learned masks (if any exist)
  - Generate candidate intron bytes (`propose_resonant_introns`)
  - Find tokens via tokenizer trie lookup
  - Score and sample based on physics-derived confidence

### Learning Flow

1. **Ingress (learning)**:
   - Read text by tokens (tokenizer parses boundaries)
   - Convert bytes to introns with ψ (XOR 0xAA)
   - Step the state with Epistemology
   - Compute the token's **last intron**
   - Update the phenotype **exon_mask** by the Monodromic Fold
   - Store only if mask differs from baseline (sparse overlay)

2. **Egress (generation)**:
   - Current state → get baseline exon product from physics
   - Check for learned specializations (overlay lookup)
   - Generate candidate intron bytes from exon product
   - Use tokenizer trie to find matching tokens
   - Score by resonance and sample

### Key Functions in Governance

```python
def exon_product_from_state(state_index: int, theta: float, orbit_size: int) -> int:
    """Project the 48-bit state tensor to an 8-bit product using physics."""
    
def propose_resonant_introns(exon_product: int, max_candidates: int = 3) -> List[int]:
    """Convert exon product to candidate intron bytes using bit family coherence."""
    
def token_last_intron(token_id: int) -> int:
    """Get the last intron byte for a token ID using the ψ isomorphism."""
```

### Storage Format

**Compact varint state-block format**:
```
[uLEB128 state_index][uLEB128 n_pairs][(uLEB128 token_id + mask_byte) * n_pairs]
```

- `state_index ≤ 788,985` → ≤ 3 bytes
- `token_id` (BERT-base) → ≤ 2 bytes  
- `mask_byte` ∈ [0, 255] = the exon product for this (state, token)
- Append-only, no deletes
- Compaction = rewrite keeping **last** mask per (state, token)

**Compression reality**: amortised overhead ~3–4 bytes/pair versus the old ~12–16 bytes. This is a **3–5×** reduction. The **mask itself** is exactly 1 byte per phenotype.

### Generation: Exon Product Sieve

1. **Baseline** exon product from **state alone**
   ```python
   p_base = exon_product_from_state(state_index, θ[state], orbit_size(state))
   ```

2. **Specialize** with learned mask (if any)
   - If `(state, token)` exists in overlay, then `mask* = stored_mask`
   - Otherwise `mask* = p_base`

3. **Exon-product sieve → intron bytes**
   ```python
   B = propose_resonant_introns(mask*) → {ι₁, ι₂, …}
   ```

4. **Trie lookup (tokenizer.json)**
   - For each candidate byte, query the tokenizer trie
   - Returns dozens, not thousands

5. **Score & sample**
   ```python
   score(token) = resonance(mask*, ι_last(token)) × g(orbit_size) × h(θ[state])
   ```

### Why This Achieves Compression

• **Holography realized**: a **single byte** is the learned residue; the rest is read off the physical atlas at run-time.

• **50% and 1/48 targets**: in practice, most (state, token) pairs will agree with the orbit baseline after modest exposure; those pairs cost **zero** bytes. Where there is actual structure to memorise, it is **exactly one byte**.

• **No duplicated "knowledge"**: the tokenizer carries the symbolic atlas; the five maps carry the physics; the overlay carries only **delta**. We stop re-recording what we already know.

### What This Removes

• No per-pair confidence, usage_count, timestamps, or names on disk.
  Confidence is a function of Θ(state) and orbit_size at the moment you act.

• No SEP-token special-casing in storage.
  Boundaries are already in bit 7; you don't pay a second time.

• No big per-orbit token lists.
Candidates come from the tokenizer; a hot-set cache is an optimisation, not the store.

• No complex multi-tier architecture.
  Single file, single format, physics-driven generation.

### What Changes in Code Terms

• **Treat the tokenizer as a core engine**: expose `id_to_bytes`, `bytes_to_id`, and trie lookup functions.

• **Make the overlay a simple append-only varint format**: `[state_index][token_id][mask]`.

• **Compute exon_product from physics**: `(mask, Θ(state), orbit_size)` with no stored "confidence" field.

• **During learning**: only write when mask differs from baseline.

• **During generation**: always start from the orbit mask and apply delta if present.

### What You Get Immediately

• A single knowledge file you can ship, mmap, and replay.

• One-byte phenotypes when you actually need them, zero bytes when you don't.

• No redundancy with tokenizer or maps.

• Generation that is natively token-level, physics-driven, and closed under your manifold.

• A clear separation between memory (exon_mask) and actuation (exon_product), both aligned with your algebra.

This is the "obvious solution": stop trying to make the knowledge store carry language or physics that are already present elsewhere. Let the tokenizer be the language. Let the five maps be the physics. Make the store a sparse, holographic **delta of one byte** per place where experience actually bends the orbit baseline. Everything else is computed when needed.

## Testing Strategy

Instead of massive datasets, test with:

• **Single article training** → See if the model can recall and continue sentences from that content

• **Sentence completion** → Start a sentence from the article: "The quantum mechanics theory states that..." and see if the model continues it coherently

• **Context continuation** → Start with: "In 1927, Heisenberg discovered..." and see if the model flows naturally into related concepts

• **Physics-driven generation** → Verify that the exon product sieve produces coherent, contextually appropriate responses

This approach leverages the model's natural ability to continue patterns rather than requiring it to "understand" different types of requests. It's aligned with the physics-based, pattern-continuation nature of GyroSI.


## 🚀 **How We Achieve Speed with Large Data Structures**

### ✅ **Memory-Mapped Files (mmap) for Large Maps**

**Epistemology Matrix (788,986 × 256 states):**
```python
# In InformationEngine.__init__()
self.ep = np.load(ep_path, mmap_mode="r")  # Memory-mapped, read-only
```

**Ontology Keys (788,986 states):**
```python
self._keys = np.load(keys_path, mmap_mode="r")  # Memory-mapped, read-only
```

**Phenomenology Map (788,986 states):**
```python
self.orbit_map = np.load(phenomap_path, mmap_mode="r")  # Memory-mapped, read-only
```

**Theta Table (788,986 divergence values):**
```python
self._theta_table = np.load(theta_path, mmap_mode="r")  # Memory-mapped, read-only
```

### ✅ **Knowledge Store Optimization**

**Memory-Mapped Knowledge File:**
```python
# In PhenotypeStore._open_mmap()
self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
```

**In-RAM Index for O(1) Lookups:**
```python
# Built from single scan at startup
self.index: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (state, token) -> (offset, size)
```

### ✅ **Tokenizer Caching**

**LRU Cache for Tokenizer:**
```python
@lru_cache(maxsize=4)
def _cached_tokenizer(name: str, base_path_str: str) -> Any:
    return Tokenizer.from_file(str(path))
```

**Trie Caching for Fast Lookups:**
```python
@lru_cache(maxsize=1)
def _get_intron_trie(...) -> Dict[Any, Any]:
    # Builds trie once, cached forever
```

### ✅ **Warmup Strategy**

**Pre-warm on Startup:**
```python
# Touch epistemology to page-in a tiny slice (avoids first-turn page faults)
_ = int(a.epistemology[0, 0])  # tiny read is enough to map a page

# Prime the tokenizer cache
_ = _load_tokenizer(PREFERENCES["tokenizer"]["name"], base_path=BASE_PATH)
```

### ✅ **Performance Optimizations**

1. **Memory-Mapped Files**: All large maps (epistemology, ontology, phenomenology, theta) are memory-mapped, so they're loaded on-demand without consuming RAM.

2. **Single Startup Scan**: The knowledge store builds an in-RAM index from a single scan at startup, enabling O(1) lookups.

3. **Cached Tokenizer**: The tokenizer and its trie structures are cached in memory for fast token lookups.

4. **Batch Processing**: Learning happens in batches with configurable write thresholds to minimize disk I/O.

5. **Async Fsync**: Disk writes are buffered and flushed asynchronously to avoid blocking.

### �� **The Result:**

- **Epistemology Matrix** (~800MB): Memory-mapped, pages loaded on-demand
- **Knowledge Store**: In-RAM index for O(1) lookups, memory-mapped data file
- **Tokenizer**: Cached in memory for instant token operations
- **First Turn**: Pre-warmed to avoid page faults

This means each turn only needs to:
1. Access the already-mapped epistemology matrix (O(1) array access)
2. Look up phenotypes via in-RAM index (O(1) dict lookup)
3. Use cached tokenizer for generation (O(1) trie lookup)

The large data structures don't slow down individual turns because they're memory-mapped and cached appropriately!