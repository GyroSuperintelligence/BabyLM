### Re-grounding on first principles

*Everything you need is already in the four tables you ship with the agent.*

| table                 | what it knows                                        | where it already lives |
| --------------------- | ---------------------------------------------------- | ---------------------- |
| **ontology keys**     | the 48-bit manifold itself (one row ↔ one state)     | `ontology_keys.npy`    |
| **epistemology**      | the deterministic successor map `N × 256`            | `epistemology.npy`     |
| **θ**                 | angular divergence of every state from the archetype | `theta.npy`            |
| **orbit cardinality** | size of each strongly-connected orbit                | `orbit_sizes.npy`      |

A **PhenotypeEntry** is *just* an 8-bit mask paired with `(state_idx, token_id)`.
Everything else (confidence, locality, frequency, stabiliser order …) is derivable on the fly from the four tables above.
So we absolutely **keep the phenotype schema untouched**.

---

## Where the mismatch really happens

1. **Learning** – you store to the *representative* `rep = phenomenology_map[state_idx]`.
   If the canonicalisation collapses many states into one rep, all those masks stack up on a single key.

2. **Generation** – you look-up *only the exact same rep* for candidates.
   When the trajectory wanders to a rep that never occurred during learning, you see an empty bucket and fall back to the naive physics prior.

The gap is *not* in the data that is stored, but in the **query radius** you use during retrieval.

---

## Fix: dynamic neighbourhood retrieval (no schema change)

We widen the candidate search **at query-time only**.
All data stay exactly where they are.

```python
# baby/intelligence.py  (inside IntelligenceEngine.generate_token_exon)

def _neighbourhood(self, state_idx: int, max_theta: float = 0.30) -> list[int]:
    """
    Return representative indices whose θ-distance to the current state
    is ≤ max_theta  *and* belong to the same stabiliser-order family.
    No extra tables: everything comes from self.s2.
    """
    θ0 = self.s2._theta_table[state_idx]
    stab0 = self.s2.stabiliser_order[state_idx]

    # vectorised filter over the whole manifold (few µs with numpy mem-map)
    θ = self.s2._theta_table
    stab = self.s2.stabiliser_order
    mask = (np.abs(θ - θ0) <= max_theta) & (stab == stab0)

    # apply phenomenology projection once to get representatives
    reps = self.phenomenology_map[mask] if self.phenomenology_map is not None else np.nonzero(mask)[0]
    return np.unique(reps).tolist()
```

Then, when you assemble token candidates:

```python
rep_list = self._neighbourhood(state_index)

for rep in rep_list:
    for s_idx, tok_id in self.operator.store.iter_keys_for_state(rep):
        # existing scoring path …
```

### Why this is enough

* **Topology aware** – θ and stabiliser order capture “how far” a state is in the curved space;
  you are no longer forced to hit the *exact* rep that happened during learning.
* **Zero new I/O** – the store is untouched, no new metadata, no migration.
* **Configurable** – `max_theta` gives you a single knob:
  small value → precise, larger value → more recall / more compute.
* **Physical interpretation** – you’re not giving “bonuses”; you’re sampling the local light-cone of the current state.

---

## Code footprint

* * **\~35 lines** in `intelligence.py`: one helper and one loop change.
* * **tests**: expect more non-empty candidate sets; generation should now reuse learned phenotypes.

No other module has to know.

---

## Optional refinements (still schema-free)

| idea                                                    | drop-in change                      | when useful                          |
| ------------------------------------------------------- | ----------------------------------- | ------------------------------------ |
| cache `neighbourhood(state_idx)`                        | `functools.lru_cache(maxsize=4k)`   | long sequential generations          |
| use **orbit cardinality** as a second radius            | `abs(orbits − orbits₀) ≤ Δ`         | if θ alone groups disparate dynamics |
| weight candidates by `1/(1+Δθ)` instead of hard cut-off | change the action-value aggregation | smoother behaviour                   |

All of these keep the phenotype record exactly as it is.

---

### Bottom line

You **do not** need a new map, new key type, or bigger phenotypes.
You only need to *query* the existing memory with a region mask that respects the geometry you already encode in **θ** and the stabiliser lattice.

**Short answer:**
Yes – the cure is exactly the same idea that spectroscopy uses to pick‐out resonant eigen-lines in a dense spectrum.
The 48-bit manifold already *is* your Hilbert space; the θ-table and the orbit sizes already tell you its local band-structure.
By querying phenotypes in a *θ-constrained neighbourhood* we are effectively projecting the current dynamical state onto a small set of near-degenerate eigen-states and letting the constructive-interference term (fold-mask overlap) pick the winner.
No new storage is required; what changes is *how we weight the existing masks when they interfere*.

Below is a more rigorous walk-through, showing that the “neighbourhood retrieval” I sketched **is not just a nearest-neighbour hack** – it is the discrete analogue of taking an inner-product with the local eigen-basis and selecting the component with the highest spectral density.

---

## 1  Interpret the four tables in spectral language

| existing table                   | spectral reading                                                                             |                                           |
| -------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **ontology keys** (48-bit basis) | computational basis (                                                                        | s⟩) of the Hilbert space                  |
| **epistemology** (N × 256)       | the *transition operator*  $T\_{s, i}$ : one column per intron → a row-stochastic linear map |                                           |
| **θ table**                      | dispersion relation: (\theta(s) = \arccos(\langle s ,                                        | ,s₀\rangle)) → energy above the archetype |
| **orbit sizes**                  | degeneracy (multiplicity) of each eigen-subspace                                             |                                           |

In other words you already shipped the *Hamiltonian* in factored form – you just never used it at query time.

---

## 2  Why canonicalisation alone breaks resonance

*Phenomenology* collapses each SCC of $T$ into one representative $r$.
Good for storage, but at retrieval time you ask:

```python
candidates = store.iter_keys_for_state(r_current)
```

If learning visited another rep $r_\text{old}\neq r_\text{current}$ (still inside the same SCC!) the amplitude you stored is invisible ⇒ destructive interference ⇒ fallback to prior ⇒ seemingly “random” words.

---

## 3  Neighbourhood retrieval = local spectral projection

The helper

```python
mask = (|θ(s)-θ₀| ≤ Δθ) & (stab(s)=stab₀)
reps = unique(phen_map[mask])
```

does two things:

* **Energy window** $|θ-θ₀|≤Δθ$ – selects states whose eigen-phase differs by at most Δθ.
  That is the *spectrographic slit*.

* **Good quantum number** `stab` – keeps you inside the same symmetry sector, so you never mix incompatible phases.

Thus the set `reps` is the local *degenerate sub-space*.
Fetching all masks attached to those reps and summing them (bitwise) is the discrete analogue of

$$
|\psi_\text{local}\rangle
   \;=\;
   \sum_{k\in\text{window}}
   \langle k\,|\,\psi\rangle\,|k\rangle
$$

Only now the coefficient $\langle k|\psi\rangle$ is replaced by the fold-mask overlap you already compute, so the whole projection costs just a few integer operations.

---

## 4  Action-value becomes spectral density, not “bonus”

Replace the ad-hoc `+ learned_bonus` with an *interference term*:

```python
# masks_from_neighbourhood is the OR of all phenotype masks of reps in the window
interference = popcount(exon_product & masks_from_neighbourhood) / 8.0
A = cooling_term + stabiliser_gain + interference - sink_penalty
```

* `interference` ∈ \[0, 1] measures **constructive overlap** between the instantaneous wave-front (exon\_product) and the stored local envelope.
* No reward shaping, no RL, no tunable weights – pure geometry.

Because $|ψ_{\text{local}}|^{2}$ is a *probability density*, summing masks is physically meaningful: each bit is an independent ±1 eigen-mode, and XOR-folding already enforces Pauli exclusion (bit twice ⇒ annihilation ⇒ 0).

---

## 5  Why this prevents “random words”

* Random output previously happened when `store.iter_keys_for_state(r_current)` was empty.
  With the spectral window that set is *never empty* as long as at least one state in the SCC was visited during learning.
* If nothing at all has ever been seen in that SCC **then** you fall back to the physics prior – which is exactly what a spectroscope shows you when no line exists at that frequency.

---

## 6  Optional but cheap refinements (still no schema change)

| refinement                                                                             | spectral meaning                                 | code delta                                      |
| -------------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- |
| **Gaussian window** $\exp[-(Δθ/σ)^2]$ instead of hard cut                              | smoother line-shape                              | 3 lines (pre-compute exp LUT)                   |
| **Iterative power-method on T** inside the window to get the true leading eigen-vector | removes residual gauge error of canonicalisation | offline pre-compute; runtime uses 256-entry LUT |
| **Cache window→mask OR**                                                               | memoises local spectral envelope                 | `lru_cache(maxsize=8192)`                       |

All independent of the store format.

---

## 7  No new “knowledge” object

Phenotypes stay *sleek*:

```python
PhenotypeEntry = { "mask": uint8, "key": (state_idx, token_id) }
```

The eigen-logic happens **at read time**, drawing only on the four immutable tables.
Storage, migration, and compatibility remain exactly as they are today.

---


