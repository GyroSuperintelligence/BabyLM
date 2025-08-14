Thanks for sharing this. I’ll focus on concrete, high-leverage changes: correctness safeguards, performance fixes on hot paths, coherence between the “physics head” and the wrappers, and a couple of ergonomics issues that will bite later.

Here are the proposals.

Correctness and invariants you should enforce at load/init

Validate meta arrays. Assert the shapes and lengths match the ontology length at startup. This prevents subtle index drift later.

```python
def _load_physics_tables(self) -> None:
    meta_path = self.base_path / "public" / "meta"
    self.ontology = np.load(meta_path / "ontology_keys.npy", mmap_mode="r")
    self.epistemology = np.load(meta_path / "epistemology.npy", mmap_mode="r")
    self.theta = np.load(meta_path / "theta.npy", mmap_mode="r")
    self.phenomenology = np.load(meta_path / "phenomenology_map.npy", mmap_mode="r")
    self.orbit_sizes = np.load(meta_path / "orbit_sizes.npy", mmap_mode="r")

    n = len(self.ontology)
    assert self.epistemology.shape == (n, 256), f"epistemology shape {self.epistemology.shape} != {(n,256)}"
    assert self.theta.shape == (n,), f"theta length {len(self.theta)} != {n}"
    assert self.phenomenology.shape == (n,), f"phenomenology length {len(self.phenomenology)} != {n}"
    assert self.orbit_sizes.ndim == 1, "orbit_sizes must be 1-D"
```

Make stage thresholds consistent with stated stage angles. Right now `STAGE_ANGLES["ONA"]` equals `π/4`, but your `THETA_ONA` is `1.0` rad and BU ingress starts at `1.3`. Decide the intended bands and encode them in one place to avoid drift.

```python
# single source of truth for stage bands
STAGE_BANDS = [
    ("CS", (0.0, THETA_CS)),
    ("UNA", (THETA_CS, THETA_UNA)),
    ("ONA", (THETA_UNA, THETA_BU_IN)),
    ("BU_IN", (THETA_BU_IN, THETA_BU_EG)),
    ("BU_EG", (THETA_BU_EG, np.inf)),
]
def _get_stage(self, state_index: int) -> str:
    t = float(self.theta[state_index])
    for name, (lo, hi) in STAGE_BANDS:
        if lo <= t < hi:
            return name
    return "BU_EG"
```

Remove the hard-coded assistant marker `173781` in `GyroHeadTransformer`. It is model/tokeniser dependent and will misfire. Use Harmony’s encoding to detect the assistant boundary properly, or make it a constructor argument.

Ensure the stop token id is taken from Harmony, not a baked constant. In `generate_response`, pull it from the encoding (or pass it in) rather than `[200002]`.

Performance: fix the obvious hotspots first  
The two main offenders are: precomputing all `token_post_state_index` at init and the O(seq²·hidden) Python loops in the “pure physics transformer”.

Make `token_post_state_index` lazy with a small LRU. This removes an O(vocab) pass on start-up.

```python
from functools import lru_cache

@lru_cache(maxsize=8192)
def _post_state_from_cs(self, token_id: int) -> int:
    s = int(self.CS_STATE_INDEX)
    for intron in self.token_to_introns(token_id):
        s = int(self.epistemology[s, intron & 0xFF])
    return s

def _build_orbit_candidates(self) -> None:
    # Build only once, but lazily fill buckets on demand
    self._orbit_candidates.clear()

def _candidates_for_orbit(self, orbit_rep: int) -> List[int]:
    if orbit_rep not in self._orbit_candidates:
        # Fill on demand by scanning a bounded token range or a persisted map
        bucket: list[int] = []
        # Start with a reasonable prefix; allow expansion if needed
        for t in range(min(self.vocab_size, 50000)):
            ps = self._post_state_from_cs(t)
            if int(self.phenomenology[ps]) == orbit_rep:
                bucket.append(t)
        self._orbit_candidates[orbit_rep] = bucket
    return self._orbit_candidates[orbit_rep]
```

Vectorise the CS→UNA emission selection. Your current best-overlap search loops over 100 candidates. Precompute int representations for the UNA pool once and use bit-ops across a vector.

```python
def _precompute_una_pool(self) -> None:
    target, tol = np.pi/4, 0.1
    idx = np.argwhere(np.abs(self.theta - target) < tol).astype(np.int32).ravel()
    if idx.size == 0:
        idx = np.argwhere((self.theta > THETA_CS) & (self.theta < THETA_ONA)).astype(np.int32).ravel()
    self._UNA_pool = idx
    # cache 48-bit ints for fast overlap
    self._UNA_pool_ints = self.ontology[idx].astype(np.uint64)

def _best_una_match(self, emitted_int: int) -> int:
    # Hamming overlap = 48 - popcount(xor)
    xor = self._UNA_pool_ints ^ np.uint64(emitted_int)
    # vectorised popcount on uint64
    pop = xor.copy()
    pop = pop - ((pop >> 1) & np.uint64(0x5555555555555555))
    pop = (pop & np.uint64(0x3333333333333333)) + ((pop >> 2) & np.uint64(0x3333333333333333))
    pop = (((pop + (pop >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)) * np.uint64(0x0101010101010101)) >> np.uint64(56)
    best_idx = int(np.argmin(pop))  # min popcount = max overlap
    return int(self._UNA_pool[best_idx])
```

Then call `_best_una_match` inside `_apply_intron_and_gate`.

Speed up `tensor_to_int` and `int_to_tensor`. The current nested loops are pure-Python. Use vectorised conversion.

```python
def tensor_to_int(tensor: np.ndarray) -> int:
    t = tensor.reshape(48).astype(np.int8)
    bits = (t == -1).astype(np.uint8)
    out = 0
    for i in range(48):
        out |= int(bits[i]) << i
    return out

def int_to_tensor(state_int: int) -> np.ndarray:
    if not (0 <= state_int < (1 << 48)):
        raise ValueError("out of bounds for 48-bit")
    bits = np.unpackbits(np.array([state_int], dtype='>u8').view(np.uint8))[-48:]
    arr = np.where(bits == 1, -1, 1).astype(np.int8).reshape(4,2,3,2)
    return arr
```

Persist broadcast masks after first generation. You already generate them when absent; write them to disk to avoid re-doing the O(256·48) pass on each cold start.

```python
def _load_broadcast_masks(self) -> None:
    meta_path = self.base_path / "public" / "meta"
    path = meta_path / "intron_broadcast_masks.npy"
    if path.exists():
        self.INTRON_BROADCAST_MASKS = np.load(path, mmap_mode="r")
    else:
        os.makedirs(meta_path, exist_ok=True)
        masks = generate_intron_broadcast_masks()
        np.save(path, masks)
        self.INTRON_BROADCAST_MASKS = masks
```

The “Pure Physics Gyro Model” in `gpt_oss/torch/model.py` will be unbearably slow at realistic sizes because of Python loops in `_gyro_attention`, `_gyro_mlp`, and `_gyro_embedding`. If you need that surface for API compatibility, either:

replace its internals with thin calls into `GyroHead` (and delete the pseudo-attention/MLP entirely), or

rewrite those functions using pure tensor ops only (no Python loops), or

gate it behind a debug flag and keep `GyroHeadTransformer` as the production path.  
Right now you have two divergent “physics” implementations; that is a maintenance hazard.

Behavioural coherence and determinism

Align special tokens across `GyroHead`, `GyroHeadTransformer`, and `chat_oss.py`. At present:

`GyroHead` sets `IM_START/IM_END` to 1 and 2.

`GyroHeadTransformer` uses `CHANNEL=200005`, `MESSAGE=200008`, `RETURN=200002`, and two env-discoveries for “final” and a content token.

`chat_oss.py` also fixes `stop_tokens=[200002]`.  
Pick one source of truth (Harmony encoding) and pass the resolved ids into both the wrapper and the CLI, rather than encoding guesses via tokenizer heuristics.

Remove the forced “Harmony prelude” inside `GyroHeadTransformer.forward`. The Harmony encoder is already rendering the conversation. Injecting forced tokens based on “generated\_length since assistant marker” will occasionally corrupt outputs. If you need a guard, make the prelude an optional mode controlled by constructor arguments, and default it off.

Safer gating logic

`_apply_intron_and_gate` blocks transitions that regress stage. If a state is surrounded by regressions you can deadlock. Allow a limited number of “stagnation escapes” by permitting the least regressive transition after K blocked attempts or by relaxing the criterion within the same macrostage.

```python
def _apply_intron_and_gate(self, state_index: int, intron: int) -> int:
    next_index = int(self.epistemology[state_index, intron & 0xFF])
    cur, nxt = self._get_stage(state_index), self._get_stage(next_index)
    if STAGE_ORDER.index(nxt) < STAGE_ORDER.index(cur):
        # soft gate: allow if within same macro family, else block
        same_family = (cur.startswith("BU") and nxt.startswith("BU"))
        if not same_family:
            return state_index
    self.path_memory = fold(self.path_memory, intron if state_index != self.CS_STATE_INDEX else GENE_Mic_S)
    return next_index
```

Memory and accounting

`stats["memory_entries"] = sum(len(d) for d in self.memory.values())` does an O(N) scan on each update. Maintain it incrementally.

```python
before = len(self.memory.get(orbit_rep, {}))
self.memory.setdefault(orbit_rep, {})[token_id] = fold(existing, token_mask)
after = len(self.memory[orbit_rep])
self.stats["memory_entries"] += (after - before)
```

Consider persisting `memory` to disk (per-orbit shards) if you intend to “learn” across sessions. The current dict will grow without bounds.

Chat runner and packaging

In `chat_oss.py`, make the stop token and special ids come from Harmony:

```python
encoding, system_message = setup_harmony_format()
stop_tokens = [encoding.token_id_for_return()]  # e.g., via the library; if not present, plumb it through setup
```

Drop the hard-coded absolute path in `test_gyro_model.py`. Accept `--gyro_path` and default to project-relative.

```python
if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument("--gyro_path", default=str(pathlib.Path(__file__).parents[3] / "memories/models/gpt-oss-20b/gyro"))
    args = p.parse_args()
    success = test_model(args.gyro_path)
    exit(0 if success else 1)
```

…and update `test_model(gyro_path: str)` accordingly.

Clean separation of concerns  
Right now there are three layers with overlapping responsibilities:

`GyroHead` (physics engine + generation policy),

`GyroHeadTransformer` (thin wrapper), and

`GyroTransformer` (a second physics model).

I would consolidate to a single “engine” (`GyroHead`) and exactly one “compatibility wrapper” that only translates to the logits surface. Delete the alternate model or put it under `experiments/` with an explicit warning. This avoids divergence of physics rules and state handling.

Smaller quality fixes

Cache `token_to_introns` with `lru_cache`—LEB128 is cheap but this gets called frequently.

```python
from functools import lru_cache
@lru_cache(maxsize=1<<16)
def token_to_introns(self, token_id: int) -> List[int]:
    ...
```

In `_get_candidates_for_state`, do not raise on empty buckets; return an empty list and let the caller decide (your `generate_token` already does the right thing). At present you have two slightly different behaviours.

In `weights.py`, the legacy decode loop is fine, but guard `torch.ldexp` dtype on CPU BF16; if you see slowdowns, upcast to FP32 for the inner computation and downcast once at the end.

```python
sub32 = sub.to(torch.float32)
torch.ldexp(sub32, exp, out=sub32)
sub.copy_(sub32.to(sub.dtype))
```

Make `bootstrap.py` idempotent by checking presence before inserting into `sys.path` (very minor, but avoids path bloat in long-running sessions).

Tests that will catch the subtle bugs early  
Add a handful of fast unit tests:

Round-trip: `state_int -> tensor -> int` equals original for random 48-bit ints.

Broadcast mask: intron x repeated 6 bytes reconstructs the expected integer bits.

Stage monotonicity: applying a random sequence of introns should not regress stages under your gating rule (or should only regress under the permitted “soft gate”).

UNA emission: from CS with driving introns, emitted states are within UNA band and the overlap metric is actually maximised against a brute-force baseline (over a small pool).

If you’d like, I can sketch the vectorised rewrite of the “pure physics transformer” functions; but my recommendation is to retire that surface and keep the thin logits wrapper over `GyroHead`. It will simplify the codebase and avoid having two different definitions of “physics”.