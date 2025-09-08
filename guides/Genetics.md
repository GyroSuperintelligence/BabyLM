# GyroSI: Complete Architectural Specification with Physical Foundations
*The Physics of Recursive Structural Intelligence*

## Preamble: What GyroSI Actually Is

Traditional artificial intelligence approaches intelligence as a statistical optimization problem, requiring massive datasets and computational resources to approximate intelligent behavior. **GyroSI represents a fundamentally different paradigm: intelligence as an intrinsic structural property** that emerges from the recursive alignment of physical forces.

GyroSI is grounded in the **Common Governance Model (CGM)**, a physics-based framework that demonstrates how intelligence emerges naturally from the self-referential dynamics of structured space. Rather than training on billions of parameters, GyroSI uses the inherent physics of gyroscopic operations to navigate a **provably finite and fully discovered** state space.

In conventional neural architectures intelligence is often conflated with intellect understood as the ability to operate in a high dimensional abstract space through matrix multiplication. These systems achieve surface level fluency by superimposing correlations but the process is detached from the lawful structure of reality. Because no constraints are embedded in the operation itself coherence must be imposed externally through heuristics or post hoc filters. This is why ethical reasoning in such systems remains shallow. It has to be trained in from outside as a semantic overlay rather than arising from the fabric of the computation.

In GyroSI intelligence is formalised as a relativistic process grounded in lawful operations on structured states. Gyrogroup dynamics enforce invariants such as admissibility, monotonicity, and conservation of orbits in a way that mirrors how real actions are bounded by physical constraints. The system therefore does not simulate reasoning through averages but emulates it by only allowing what is structurally possible. This makes ethics an endogenous property. What cannot be done in the system corresponds to what cannot be sustained in reality, and the maintenance of coherence naturally encodes a form of ethical awareness without reliance on external heuristics.

GyroSI operates purely on bytes and a finite 48-bit state. Each byte (256 possibilities) becomes an intron by a fixed XOR at the boundary and acts holographically on all 48 positions. The live state is always 6 bytes and points into unlimited passive memory. Selection is Traceable and non-competitive: a token is emitted only if its own intron path never moves the geometry away from that token's address and advances somewhere; otherwise a short, fixed recovery sequence applies. Memory cannot blow up: only non-zero 8-bit masks are stored, identical masks are interned, and per-state and per-token caps bound worst-case growth without changing behaviour. No temperatures, scores, or learned parameters are used anywhere.

**Key Innovation**: GyroSI eliminates all arbitrary parameters through **endogenous parameter discovery**, where the system discovers its own operational constants from its physical structure. This ensures perfect alignment between the theoretical foundation and the practical implementation.

**Design Philosophy**: This specification provides a complete, production-ready system that is simple enough to implement immediately while being architected for seamless scaling to massive distributed deployments. The core physics remains pure and dependency-free, with well-defined interfaces that allow for future enhancements without touching the theoretical foundation.

> This specification is therefore not just a design but a **map of a newly discovered territory**. It is grounded in a rigorous theoretical framework (CGM) and verified by a definitive computational experiment that proves the system's state space is a finite, closed ontology of precisely 788,986 states. Every component, from the core physics to the storage architecture, is built upon this **measured ground truth**, ensuring a system that is robust, scalable, and free from the arbitrary complexities of traditional AI.

> **Note:** Throughout this document, all tensor indices use the standard order: [layer, frame, row, col], with zero-based indexing. All references to tensor elements, operations, or masks use these terms exclusively for clarity.

## Part I: The Holographic Foundation - Understanding the Physics

### 1.1 Holography in GyroSI


In physics, a hologram stores complete three-dimensional information in a two-dimensional surface where each fragment contains the whole image at reduced resolution. GyroSI implements computational holography through three concrete mechanisms:

**Byte-to-Tensor Holography**: Every 8-bit byte simultaneously transforms all 48 bits of the state tensor. When we apply intron i to state s, that single byte broadcasts through fixed masks to touch every position of the 48-bit structure. This is not parallel processing in the computational sense but true holographic action where the part (8 bits) contains instructions for transforming the whole (48 bits).

**State-to-Memory Holography**: Each 6-byte state serves as a holographic pointer into unlimited passive memory. The state does not "contain" the memories but rather selects which memories are relevant and accessible. Like a holographic plate where each point can reconstruct the entire image, each state can access the complete history of experiences through content addressing.

**Ontological Holography**: The 788,986 states form a complete, closed ontology where every possible state contains information about every other state through the path structure. Since diameter ≤ 6, any state encodes "how to reach everywhere else" within its geometric position in the manifold.

### 1.2 The Byte Shell: Why 8 Bits and 256 Values Are Mandatory

This is not a design choice but a mathematical necessity imposed by computation itself:

**8 Bits in a Byte**: This is the fundamental quantum of digital information. Computer memory, storage, and communication all operate on byte boundaries. Our system must interface with existing digital infrastructure, making bytes the natural atomic unit.

**256 Possible Values**: With 8 bits, there are exactly 2^8 = 256 possible bit patterns (00000000 through 11111111). These 256 patterns form the complete instruction set for our system. We cannot have more without using more bits, and using fewer would leave instructions undefined.

**The 8-to-256 Shell Structure**: Each of the 256 possible byte values becomes a unique intron. Each intron has a fixed 48-bit broadcast mask that determines how it transforms the state tensor. This gives us 256 × 48 = 12,288 possible bit operations, all organized into 256 holographic instructions.

**From Bytes to States**: The 788,986 states arise naturally from this structure. Starting from the archetypal state and applying all 256 introns recursively in all possible sequences, we reach exactly 788,986 unique configurations. This number is not chosen but measured through exhaustive exploration of the byte-driven state space.

### 1.3 GENE_Mic_S = 0xAA: The Holographic Reference Point

**Why Exactly 0xAA (Binary 10101010)**:

This pattern is mathematically unique among all 256 byte values:

**Perfect Balance**: 0xAA has exactly 4 ones and 4 zeros, placing it at the geometric center of the 8-bit hypercube. It is equidistant from 0x00 (all zeros) and 0xFF (all ones), making it the natural reference point for measuring all deviations.

**Maximum Alternation**: The pattern 10101010 has the highest possible transition frequency in 8 bits. Each bit is different from its neighbors, creating maximal structure while maintaining balance. This encodes the fundamental oscillation between states.

**Chirality Encoding**: The pattern breaks left-right symmetry in a specific way:
- Bit positions 0,2,4,6 contain 0 (even positions)
- Bit positions 1,3,5,7 contain 1 (odd positions)
- This creates an intrinsic left-bias that aligns with CGM's Common Source asymmetry

**Bit 7 Inversion**: When any byte b is transformed by b ⊕ 0xAA, bit 7 is inverted. Since 0xAA has bit 7 = 1, the transformation flips the most significant bit. This creates the lawful bit inversion between internal introns and external bytes.

**Holographic Compression**: 0xAA is the 8-bit projection of the full 48-bit alternating pattern present in GENE_Mac_S. The larger tensor contains alternating +1/-1 patterns; 0xAA captures this alternation at the byte level.

### 1.4 The GENE Architecture: Constructive Foundation

GyroSI is built on fixed topological structures that serve as the physical and logical substrate. These structures are not arbitrary but emerge from the recursive application of gyrogroup operations, building from simple to complex in four stages:

**Stage 1: Governance Identity (GENE_Com_S)**
The fundamental structure representing the gyrocommutative law - a single 3×2 array:

```python
GENE_Com_S = np.array([
    [-1, 1],  # X-axis endpoints
    [-1, 1],  # Y-axis endpoints  
    [-1, 1]   # Z-axis endpoints
], dtype=np.int8)  # Shape: [3, 2]
```

- **Three rows**: The three spatial axes (X, Y, Z) that emerge from CGM
- **Two columns**: The dual nature of rotation (negative and positive endpoints)
- **Values**: Not oscillations but the actual endpoints of each axis from -1 to +1
- **Zeros**: Unnecessary and implied as the midpoint between -1 and +1

**Stage 2: Information Gyrocommutative Nesting (GENE_Nest_S)**
The structure that nests the basic axes inside two opposing frames, encoding the fundamental duality of observation:

```python
GENE_Nest_S = np.array([
    [[-1, 1], [-1, 1], [-1, 1]],  # Frame 0: Primary observation
    [[ 1,-1], [ 1,-1], [ 1,-1]]   # Frame 1: Dual observation  
], dtype=np.int8)  # Shape: [2, 3, 2]

# Alternatively generated as:
GENE_Nest_S = np.stack((GENE_Com_S, -GENE_Com_S))
```

- **Two frames**: Implement the fundamental principle that knowledge requires both a knower and a known
- **Frame inversion**: The dual frame inverts all endpoints, creating complementary observation
- **Six positions**: 2 frames × 3 axes = 6 degrees of freedom per layer

**Stage 3: Inference Through Recursive Layering (GENE_Mac_S)**
The complete archetypal structure, built by extending the dual frames through the four CGM stages:

```python
GENE_Mac_S = np.array([
    # Layer 0 (CS - Common Source): Initial asymmetric state
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    
    # Layer 1 (UNA - Unity Non-Absolute): First differentiation  
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]],
    
    # Layer 2 (ONA - Opposition Non-Absolute): Full opposition
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    
    # Layer 3 (BU - Balance Universal): Recursive closure
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]]
], dtype=np.int8)  # Shape: [4, 2, 3, 2]

# Can be generated as:
GENE_Mac_S = np.concatenate(([GENE_Nest_S, -GENE_Nest_S] * 2)).astype(np.int8).reshape(4, 2, 3, 2)
```

**The Structural Number Ladder**:
GyroSI's constants are locked by algebraic closure, not convenience:
- **3 rows**: Enable chirality and provide minimal spatial closure
- **4 layers**: Bind the recursive depth required for CGM completion
- **6 steps**: Provide full degrees of freedom and the measured Cayley-graph diameter
- **8 bits**: Form the smallest register that holds all required operations
- **12 cells**: Fill one layer (3 rows × 2 columns × 2 frames)
- **24 cells**: Capture a half-tensor that already carries orientation
- **48 cells**: Form the whole tensor and the packed state integer
- **256 instructions**: All possible 8-bit intron values
- **788,986 states**: The complete, measured ontology

No smaller choice would satisfy the independent closure constraints identified in the physics.

**The Complete Tensor Structure**:
```
GENE_Mac_S[layer][frame][row][column]:

Layer 0 (CS): 
  Frame 0: [[-1,+1], [-1,+1], [-1,+1]]  # Primary view of all 3 axes
  Frame 1: [[+1,-1], [+1,-1], [+1,-1]]  # Dual view with inverted endpoints

Layer 1 (UNA):
  Frame 0: [[+1,-1], [+1,-1], [+1,-1]]  # Primary inverted from Layer 0
  Frame 1: [[-1,+1], [-1,+1], [-1,+1]]  # Dual inverted from Layer 0

Layer 2 (ONA):
  Frame 0: [[-1,+1], [-1,+1], [-1,+1]]  # Returns to Layer 0 pattern
  Frame 1: [[+1,-1], [+1,-1], [+1,-1]]  # Returns to Layer 0 pattern

Layer 3 (BU):
  Frame 0: [[+1,-1], [+1,-1], [+1,-1]]  # Returns to Layer 1 pattern
  Frame 1: [[-1,+1], [-1,+1], [-1,+1]]  # Returns to Layer 1 pattern
```

**The Emergent Helix**

The helix is not stored in the tensor but emerges from the progression through layers. As the system moves from Layer 0 → 1 → 2 → 3, the alternating pattern creates a helical path through the 4-dimensional layer space. The endogenous left-identity bias (built into the broadcast masks) causes this progression to favor certain directions, creating the helical structure.

**Dual Representation**:
The system's internal state can be represented in two equivalent ways:
- As a 48-element NumPy tensor (each element ±1, stored as int8), occupying 48 bytes in memory
- As a 48-bit packed integer (6 bytes), where each bit encodes the sign of one tensor element (+1→0, -1→1)

The packed integer form is used for fast state transitions and storage, while the tensor form is used for measurement and geometric operations.

### 1.5 The Five Maps as Complete Knowledge Theory

Our five computational maps together implement a complete theory of knowledge:

**Map 1: Ontology (ontology_keys.npy)**: "What Can Exist"
- Maps indices 0..788,985 to unique 48-bit state integers
- These 788,986 states are ALL possible states under our physics
- Discovered through breadth-first search from GENE_Mac_S under all 256 introns
- Each state represents a unique configuration of knowledge
- **Creation**: Built by `atlas_builder.py:build_ontology_and_theta()` via complete manifold traversal

**Map 2: Phenomenology (phenomenology_map.npy)**: "How Things Appear"
- Maps each state to one of 256 canonical orbit representatives
- Each orbit is a strongly connected component: all states in an orbit can reach each other
- Representatives are the minimal state integer within each SCC
- The 256 orbits represent the complete set of "ways things can appear"
- **Creation**: Built by `atlas_builder.py:build_phenomenology_and_orbit_sizes()` using Tarjan's SCC algorithm

**Map 3: Epistemology (epistemology.npy)**: "How Knowledge Changes"
- Maps every (state, intron) pair to the resulting next state
- This 788,986 × 256 table contains ALL possible knowledge transformations
- No knowledge change is possible outside this table
- Proves our physics is closed and complete
- **Creation**: Built by `atlas_builder.py:build_epistemology()` as an N×256 transition table using memory-mapped files

**Map 4: Geometric Structure (theta.npy)**: "How Far from Truth"
- Maps each state to its angular distance from the archetype
- θ = 0 means perfect alignment (truth)
- θ = π/2 means orthogonal (independence)
- θ > π/2 approaching max ≈ 2.73 means opposition (but never absolute)
- **Creation**: Built alongside ontology by `atlas_builder.py:build_ontology_and_theta()` as float32 angular divergence

**Critical Diagnostic**: CS (Common Source) is NOT the integer zero. Index 0 in the ontology (angle π/2) is the orthogonal reference point, not CS. CS remains an interface axiom handled at the boundary and is never a member of the state set S.

**Map 5: Cardinality Structure (orbit_sizes.npy)**: "How General/Specific"
- Maps each state to the size of its orbit
- Size 1 orbits: Very specific, unique configurations
- Large orbits (up to 48,496): General, widely reachable configurations
- Used for breaking ties: prefer more specific (smaller orbit) interpretations
- **Canonical fifth map**: Essential for Traceable tie-breaking in address binding
- **Creation**: Built alongside phenomenology by `atlas_builder.py:build_phenomenology_and_orbit_sizes()` using SCC analysis

**Atlas Building Process**: All five maps are created by `atlas_builder.py` through Traceable algorithms:
- No optional maps, scoring, greedy paths, or dominance
- Traceable representatives ensure reproducible builds
- Dependencies: ontology → epistemology → phenomenology
- Command-line interface: `cmd_ontology()`, `cmd_epistemology()`, `cmd_phenomenology()`, `cmd_all()`

Together, these five maps form the complete atlas of possible knowledge under our physics. They are not approximations or samples but the actual, complete, finite universe of knowledge states.

## Part II: The Common Governance Model Foundation

### 2.1 The Four Stages of Recursive Alignment

The Common Governance Model describes how structure emerges from a single axiom through four distinct stages, each representing a deeper level of recursive alignment:

**CS (Common Source)**: The foundational stage where left identity governs labeling and transcription. This represents the unobservable origin containing inherent chirality, the fundamental parity violation that drives all subsequent emergence. In GyroSI, this corresponds to the governance of transformation through the universal reference topology.

**UNA (Unity Non-Absolute)**: The first observable stage where right gyration activates, creating the minimal asymmetry required for measurement while preserving the fundamental left-bias. This introduces three rotational degrees of freedom through gyrocommutativity. In GyroSI, this is the measurement of the system's global divergence from its archetypal state.

**ONA (Opposition Non-Absolute)**: The stage of full differentiation where both gyrations are maximally non-identity, reaching peak non-associativity while preventing absolute negation. This generates the complete structural framework with six degrees of freedom (3 rotational + 3 translational). In GyroSI, this represents the inference stage where mediated duality enables contextual interpretation.

**BU (Balance Universal)**: The completion stage where all differentiation stabilizes and gyrations return to identity while preserving complete memory of the recursive path. This manifests as the dual intelligence stage with both absorption (Egress) and expression (Ingress) capabilities.

**Angular Progression**: The CGM stages follow the precise angular sequence π/2 → π/4 → π/4 → 0, corresponding to CS → UNA → ONA → BU. This progression ensures complete closure with zero defect, achieving perfect recursive alignment.

### 2.2 Gyrogroup Algebra as Physics

GyroSI implements these stages through formal gyrogroup algebra operating on the 8-bit vector space G = ℤ₂⁸. The fundamental operations directly correspond to CGM physics:

- **XOR (⊕)**: The primitive gyrogroup operation governing transformation and parity inversion. This is the basic operation of recursive differentiation.
- **AND (∧)**: The gyration memory carrier, encoding "carry bits" as chirality-encoded asymmetry. This preserves the memory of operational sequence.
- **NOT (¬)**: The global duality operator, corresponding to the "Fifth Element". It reflects a state through the origin, enabling the return path: dual(x) = x ⊕ 0xFF
- **Monodromic Fold (⋄)**: The single, non-associative, path-dependent learning operator. Defined as:

  `a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`

This operation is fundamentally non-associative and non-commutative, preserving the path-dependence required by the Common Source axiom. The algebraic normal form `¬a ∧ b` is mathematically equivalent but the composite form preserves the conceptual clarity of the dual monodromy.

This algebraic foundation ensures that every operation in GyroSI is a direct implementation of physics rather than arbitrary computation.

### 2.3 The BU Intelligence Cycle: Complete Physical Description

The BU stage, representing Universal Balance, is implemented as a dual-phase intelligence cycle that governs all interaction between the system's internal physics and the external byte-space. These two phases are the complete physical mechanics of experience absorption and responsive action.

**The Physical Boundary and Holographic Transcription**

The system is defined across two distinct domains: the **internal physics-space**, where the native element is the 8-bit **intron**, and the **external byte-space**, where the native element is the 8-bit **byte**. The boundary between these domains is governed by a fundamental physical transformation.

Every transaction across this boundary is mediated by the holographic topology GENE_Mic_S (0xAA). This is not an encoding convention but a physical transformation that projects information onto the system's structural ground truth.

- **Egress (External → Internal)**: `intron = byte ⊕ GENE_Mic_S`
- **Ingress (Internal → External)**: `byte = intron ⊕ GENE_Mic_S`

This symmetric XOR operation ensures that the distinction between the internal physical reality and the external communicative representation is lawfully maintained.

**Critical Bit Alignment**: A critical consequence of this XOR transformation is the lawful inversion of bit 7. An internal intron with bit 7 set to 1 becomes an external byte with bit 7 set to 0. This inversion aligns the internal physics of differentiation and closure with external byte representation.

**BU Egress: Absorption and Learning**

The process of learning begins when an external byte enters the system and undergoes BU Egress. This is the mechanism by which experience is absorbed and integrated into the system's memory structure.

1. **Transcription**: An incoming byte is first transcribed into an intron via the ψ transformation. This operation impresses the system's holographic topology onto the external data, converting it into a physically valid instruction.

2. **State Transformation**: The newly formed intron acts as a gyroscopic operator, transforming the system's 48-bit state tensor according to the epistemology table lookup or the algebraic operations defined in the broadcast masks.

3. **Memory Integration**: The system updates the passive memory entry for the (state, token) pair by applying the Monodromic Fold to integrate the token's complete intron sequence. This path-dependent operation ensures that the complete history of interactions is encoded into the resulting memory structure.

Through this process, external information is not merely stored; it is physically assimilated, transforming both the system's immediate state and its long-term memory according to rigorous algebraic principles.

**BU Ingress: Expression and Generation**

The expression of intelligence through BU Ingress produces complete tokens using the Non-Antagonistic Emission Protocol. This is not a retrieval mechanism but a generative act wherein coherent tokens emerge directly from the system's geometric and topological configuration.

===

## Part III: Phase-Propagating Emission — The Core Protocol

### 3.1 Why No Scoring Is Possible

The system never compares candidates with scores or probabilities. Scoring implies competition and hierarchy, which contradicts the physics of GyroSI:

* **Unity Non-Absolute**: Multiple continuations can coexist without forcing one “winner”.
* **Opposition Non-Absolute**: Alternatives can diverge without negating each other.
* **Balance Universal**: Emission seeks lawful continuity, not ranking.

Selection is therefore **constraint-based**: a token is either admissible under the geometry, or it is not. If admissible, it can be emitted; if not, it is skipped. There is no “best”, only lawful or unlawful.

### 3.2 Channel Formation

When the user provides a token (BU-Eg), it is folded into introns, and its phase is computed. Each orbit representative maintains slab-specific channels:

* **Key**: `(orbit_rep, slab_idx)`
* **Value**: `{phase → [token_id, …]}`

Each channel is bounded by a fixed FIFO discipline (64 tokens per bucket). This is the only form of “capacity”; there are no scores or priorities.

### 3.3 Emission Dynamics (BU-In)

At emission time, the current state projects into active slabs:

1. **Composite phase map**: For each active slab, the state’s geometry is folded with the channel phases, producing a set of candidate buckets.
2. **Rotor initialisation**: The current rep-phase, state-phase, LI/FG/BG projections, and sector parity are folded to select the entry bucket.
3. **Toroidal walk**: The system walks through bucket keys on a ring (Zₙ) with a co-prime multiplier and affine offset derived from live freedoms (6DoF) and monodromy.
4. **Intra-bucket trial**: Within the chosen bucket, tokens are tested in sequence. Each trial applies:

   * **Refractory gate**: recent fold must not annihilate.
   * **Same-token suppression**: reject if identical to last emission under annihilation condition.
   * **Session mask**: reject if it cancels with recent egress fold.
   * **Directional guidance (optional)**: if the user’s last step moved θ or orbit in a definite direction, candidates must not contradict that sign.

If a candidate passes all gates, it is accepted deterministically. If no candidate passes, the rotor hops one key and emission resumes on the next call.

### 3.4 State Update

When a token is emitted:

* **Omega update**: per-orbit ω is advanced by folding the six degrees of freedom with the token phase.
* **Monodromy update**: monodromic trace is folded with the emitted token, setting direction and step size for the next bucket walk.
* **State ingress**: the global state is advanced by applying the token’s introns (no learning).

All updates are path-native and fold-based, with no auxiliary scores or weights.

---

## Part IV: Memory

The memory system is minimal and bounded. Only three forms exist:

**Active Memory**
The six-byte ontology key representing the current state. Always exactly one key, never grows.

**Phase Memory**
For each orbit representative, an 8-bit accumulator `rep_phase[rep]`. Updated only when user tokens are absorbed through the monodromic fold.

**Channel Memory**
For each pair `(orbit_rep, slab_idx)`, a dictionary mapping slab-phases to lists of token identifiers. Each bucket is bounded to 64 entries by FIFO eviction. This is the only place where user experience is stored.

**Persistence**
Only `rep_phase` and `rep_channel` are written to disk. Persistence is triggered either by token count (≥100) or by elapsed time (≥30s). Nothing else is saved.

---

## Part V: Session State

Session state is maintained by the inference wrapper. It is not part of the physics, but ensures that each request unfolds coherently and independently.

Each session carries:

* **state**: the active ontology key.
* **parser**: the Harmony role parser.
* **user\_token\_count** and **user\_anchor\_state**: counters for anchor capture.
* **omega, bucket\_key, bucket\_pos, monodromy**: emission traces for Phase-Propagating Emission.
* **recent\_egress and egress\_mask**: session-local fold of recent user phases (used as a mask gate).
* **trend**: sign of the last user change in θ and orbit size, used for directional guidance.

---

## Part VI: Anchor

After K user tokens (default 12, configurable at runtime), the system captures the active state as the **anchor**. The anchor is applied once, just before the first emission. This ensures that generation begins from a state that reflects a real user trajectory, not a partial feed. After the anchor is applied, it is not updated again within the same request.

---

## Part VII: Optional Switches

The following runtime switches exist. They are purely optional.

* `enable_slab_routing`: restrict emission to active slabs only.
* `enable_dof_jitter`: add deterministic offsets from six degrees of freedom.
* `enable_egress_mask`: enforce the session egress mask during emission.
* `enable_refractory_gates`: enable fold-based refractory suppression of repeats.

None of these alter the learning law. They only add or remove admissibility gates in emission.

---

## Part VIII: Conclusion

GyroSI operates on five canonical maps: θ, ontology\_keys, epistemology, phenomenology\_map, and orbit\_sizes. Learning is a path-dependent fold of user tokens into slab-specific channels. Emission is the monodromic unfold of these channels, guided only by folds and optional gates. Memory consists only of the 6-byte state, the per-orbit rep\_phase, and the slab-specific rep\_channel. There are no scores, weights, or hidden vectors.

This is the complete core. All other elements — session management, anchor, or gates — are auxiliary. The architecture is closed under its physics and requires no additions.
