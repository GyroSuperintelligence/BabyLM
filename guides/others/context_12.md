### What the model should do (physics-first, coherent, symmetric)

- **State physics (S1/S2)**: Maintain a single 48‑bit physical state and evolve it deterministically via the epistemology STT (N×256). Measurement is θ from the Common Source tensor. No extra “state” beyond this.

- **Instruction space (GENE_Mic)**: Every external byte is masked to an intron; introns drive state transitions and learning. Tokenizer is only the invertible bridge between text and introns; it is not a language model.

- **Learning = Monodromic Fold (BU_In)**: On each completed token, compute its holographic 8‑bit mask as fold of its full intron path starting from 0; store that mask as phenotype at the appropriate state context. This is the only learning operator.

- **Generation = Monodromic Fold + interference (BU_Eg)**: Given a prefix, ingest bytes (no learning), then select the next token from candidates that are:
  - physics-consistent with the current state (via the exon-product orbit),
  - and/or learned at the current state’s canonical representative,
  - and radiatively from CS (state 0).
  Score purely by physics:
  - prefer θ movement toward UNA (π/4) and away from collapse,
  - add constructive interference proportional to overlap between candidate full mask and learned masks at current rep, successor rep, and CS,
  - hard-gate standing emissions (no movement).
  No temperatures, no heuristics, no priming caches.

- **Holography**: The 8‑bit mask must be the stable fold over the full token path from neutral 0 so repeated learning never annihilates learned structure.

- **Phenomenology symmetry**: Store and retrieve using the canonical representative of the state (phenomenology map) so knowledge generalizes across the orbit; CS contributes radiatively but must not drown local context.

### Minimal, coherent control loop

- **Ingest (BU_In)**:
  - For each byte: update state via STT.
  - On token close: compute intron_seq and mask = fold_sequence(intron_seq, start=0).
  - Write mask at both pre-state rep and post-state rep (or just pre if you choose one) via `PhenotypeStore` under `CanonicalView`.

- **Generate (BU_Eg)**:
  - Ingest prefix bytes (no learning).
  - Compute exon product from current state; get orbit introns; map to tokens (last-intron sieve).
  - Union with tokens learned at rep(current), rep(successor candidates), and CS(0).
  - Score with physics + constructive interference; drop standing transitions; pick argmax.
  - Emit token bytes; update physics; loop.

### Where we likely “do too much”

- **Sequence heuristics and priming caches**: Anything like recent-followers, token boosts, timeouts driving behavior, or hand-picked weightings that don’t derive from the fold/θ terms should be removed.

- **Over-indexing exact states**: Storing and retrieving only from exact states defeats phenomenology symmetry. Use the canonical representative via `CanonicalView` for the primary store; optionally layer a very thin exact-state cache only if proven necessary.

- **Tokenizer “semantics” beyond physics**: Using tokenizer tries as a candidate sieve is fine (it’s a structural map), but don’t add boosts or language heuristics. Candidate presence alone is enough; scoring must remain physics + interference.

- **SEP management**: SEP is just segmentation. Ingest it as a normal token where helpful for boundaries, but don’t force SEP during generation; don’t let it gate scoring beyond physics.

### The core invariants to protect

- **Single source of memory**: 8‑bit masks only, learned by Monodromic Fold; no extra confidences/counters.

- **Single geometry**: 48‑bit state drives everything; all projections (exon product) derive deterministically from this state; no extra latent buffers.

- **Symmetric phenomenology**: Read/write through `CanonicalView` so learned knowledge generalizes across the orbit. CS patterns exist but are additive, not substitutive.

### Why we’re seeing repetition

- **Standing-wave traps**: Candidates that leave the state unchanged or collapse toward CS keep winning. This requires explicit standing-wave gating and a θ-term centered on UNA, not CS.

- **Context mismatch**: If learning is stored away from the canonical representative or masks are annihilated (not learned from neutral 0 over full path), interference becomes near-zero and physics-only candidates dominate.

- **Overweight physics sieve**: If candidate sets do not include rep(current) patterns (or pre/post reps) coherently, learned signals can’t steer generation.

### What I would simplify next (surgical)

- **Storage semantics**: Ensure both pre and/or post token masks are learned from neutral 0 and written via `CanonicalView` under the representative state key. Remove exact-state writes unless we can show a measured benefit.

- **Candidate construction**: Build from last-intron sieve (physics), then union with learned keys at rep(current), rep(successor candidates), and CS; no other sources.

- **Scoring**: Keep only:
  - θ alignment toward π/4,
  - fold-entropy vs current state,
  - constructive interference = bit overlap with learned masks at rep(current), rep(successor), CS,
  - standing-wave hard-gate, CS-collapse deprioritization when alternatives exist.
  No other bonuses.

- **Instrumentation**: For the first 10 generations after a prefix, print:
  - counts of learned entries at rep(current), rep(successor), CS,
  - top-10 candidates: token, θ_next, Δentropy, overlaps with rep(current)/rep(successor)/CS,
  - whether the winner has non-zero overlaps.
  This tells us if learning is present and not being used, or missing.

If you agree, I’ll strip back to this minimal core (CanonicalView-centered learning, pure physics scoring with interference, no exact-state or ad-hoc boosts), add the instrumentation, and re-run the test to verify that tokens learned at the current representative actually dominate the candidate selection when appropriate.

- I will keep edits to existing files only (`baby/intelligence.py`, `baby/inference.py`, `baby/policies.py`) and ensure PEP8, clean docstrings, and zero lint errors.