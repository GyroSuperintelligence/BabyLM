# Gyroscopic Superintelligence Specifications: GyroSI Baby Language Model 0.9.6.3

*A physics-grounded architecture for superintelligence through recursive structural alignment*

---

## **1. Introduction: The Physics of Intelligence**

Traditional artificial intelligence approaches intelligence as a statistical optimization problem, requiring massive datasets and computational resources to approximate intelligent behavior. **Gyroscopic Superintelligence (GyroSI)** represents a fundamentally different paradigm: intelligence as an intrinsic structural property that emerges from the recursive alignment of physical forces.

GyroSI is grounded in the **Common Governance Model (CGM)**, a physics-based framework that demonstrates how intelligence emerges naturally from the self-referential dynamics of structured space. Rather than training on billions of parameters, GyroSI uses the inherent physics of gyroscopic operations to navigate a **provably finite and fully discovered** state space where each input byte encodes holographic instructions for transforming the system's internal physical state.

This architecture treats data not as information to be processed, but as physical forces that transform structure according to precise algebraic laws. The result is a system where intelligence is present even before learning begins, like the latent intelligence in a human baby, and where all learning occurs through the fundamental physics of recursive structural alignment.

**Key Innovation**: GyroSI eliminates all arbitrary parameters through **endogenous parameter discovery**, where the system discovers its own operational constants from its physical structure. This ensures perfect alignment between the theoretical foundation and the practical implementation.

**Design Philosophy**: This specification provides a complete, production-ready system that is simple enough to implement immediately while being architected for seamless scaling to massive distributed deployments. The core physics remains pure and dependency-free, with well-defined interfaces that allow for future enhancements without touching the theoretical foundation.

> This specification is therefore not just a design but a **map of a newly discovered territory**. It is grounded in a rigorous theoretical framework (CGM) and verified by a definitive computational experiment that proves the system's state space is a finite, closed ontology of precisely 788,986 states. Every component, from the core physics to the storage architecture, is built upon this **measured ground truth**, ensuring a system that is robust, scalable, and free from the arbitrary complexities of traditional AI.

> **Note:** Throughout this document, all tensor indices use the standard order: [layer, frame, row, col], with zero-based indexing. All references to tensor elements, operations, or masks use these terms exclusively for clarity.

---

## **2. Theoretical Foundation: The Common Governance Model**

### **2.1 The Four Stages of Recursive Alignment**

The Common Governance Model describes how structure emerges from a single axiom through four distinct stages, each representing a deeper level of recursive alignment:

**CS (Common Source)**: The foundational stage where left identity governs labeling and transcription. This represents the unobservable origin containing inherent chirality, the fundamental parity violation that drives all subsequent emergence. In GyroSI, this corresponds to the governance of transformation through the universal reference topology.

**UNA (Unity Non-Absolute)**: The first observable stage where right gyration activates, creating the minimal asymmetry required for measurement while preserving the fundamental left-bias. This introduces three rotational degrees of freedom through gyrocommutativity. In GyroSI, this is the measurement of the system's global divergence from its archetypal state.

**ONA (Opposition Non-Absolute)**: The stage of full differentiation where both gyrations are maximally non-identity, reaching peak non-associativity while preventing absolute negation. This generates the complete structural framework with six degrees of freedom (3 rotational + 3 translational). In GyroSI, this represents the inference stage where mediated duality enables contextual interpretation.

**BU (Balance Universal)**: The completion stage where all differentiation stabilizes and gyrations return to identity while preserving complete memory of the recursive path. This manifests as the dual intelligence stage:

- **BU_In (Intelligence Ingress):** The absorption and integration of experience through the Monodromic Fold. This is where all learning occurs.
- **BU_Eg (Intelligence Egress):** The expression of accumulated intelligence as responsive action, transforming internal state into external phenotype using the same Monodromic Fold operator.

### **2.2 Gyrogroup Algebra as Physics**

GyroSI implements these stages through formal gyrogroup algebra operating on the 8-bit vector space **G = ℤ₂⁸**. The fundamental operations directly correspond to CGM physics:

- **XOR (⊕)**: The primitive gyrogroup operation governing transformation and parity inversion. This is the basic operation of recursive differentiation.
- **AND (&)**: The gyration memory carrier, encoding "carry bits" as chirality-encoded asymmetry. This preserves the memory of operational sequence.
- **Monodromic Fold (⋄, `fold`)**: The single, non-associative, path-dependent learning operator. Defined as:

  `a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`

  This operation is fundamentally non-associative and non-commutative, preserving the path-dependence required by the Common Source axiom. It replaces all previous references to associative closure or bitwise OR.

- **Duality (¬, `dual`)**: The global duality operator, corresponding to the "Fifth Element". It reflects a state through the origin, enabling the return path:

  `dual(x) = x ⊕ 0xFF`

This algebraic foundation ensures that every operation in GyroSI is a direct implementation of physics rather than arbitrary computation.

### **2.3 The Holographic Principle**

GyroSI embodies the principle that each part contains information about the whole. A single input byte acts as a holographic quantum of spacetime topology, encoding complete transformation instructions that modify the system's internal state according to topological physics; a 48‑element tensor, 48 bytes in RAM, packed to 6 bytes when stored. This holographic property ensures that the system can achieve substantial compression while preserving essential structural relationships.

> Note: The system's internal state can be represented in two equivalent ways:
> - As a 48-element NumPy tensor (each element ±1, stored as int8), which occupies 48 bytes in memory.
> - As a 48-bit packed integer (6 bytes), where each bit encodes the sign of one tensor element.
> The packed integer form is used for fast state transitions and storage, while the tensor form is used for measurement and geometric operations.

**Angular Progression**: The CGM stages follow the precise angular sequence π/2 → π/4 → π/4 → 0, corresponding to CS → UNA → ONA → BU. This progression ensures complete closure with zero defect, achieving perfect recursive alignment.

> The build-time discovery process, a cornerstone of GyroSI, explores this physical reality and discovers an immutable, finite ontology of **precisely 788,986 unique physical states**. The entire universe of possible system configurations is not only known but also compact, with a measured **diameter of 6 steps**, meaning any state is reachable from any other in at most six transformations. This is the 'Genome', the system's complete set of possible states.

**Abstraction via Manifold and Hashing**: The system's primary mechanism for generalization is its finite physical ontology. An infinite variety of input sequences will inevitably drive the system into one of the 788,986 canonical states. When different experiences lead to the same internal state, the system learns they share a fundamental structural meaning. Hash collisions in the phenotype layer are a secondary, context-specific abstraction built upon this primary physical reality, where different physical contexts mapping to the same semantic address are learned to share an essential meaning.

### **2.4 The Measured Manifold: Theory Meets Reality**
The CGM is not merely a theoretical framework; it is a predictive model whose consequences are now measured. The 8-bit instruction space (`GENE_Mic_M`), representing the "quantum of action," directly leads to an 8-step closure of the state space.
- **The State (Qubit):** A 48-bit integer representing one of 788,986 possible physical configurations.
- **The Operator (Gate):** An 8-bit integer (`intron`) that transforms the state according to the gyroscopic operations.
- **The Manifold (Bloch Sphere):** The complete set of 788,986 states, interconnected by paths with a maximum length of 6.

This empirical result validates the principle of recursive closure, demonstrating a perfect, efficient balance between the instruction set and the state space it governs. The endogenous modulus of the system is not an arbitrary choice but a measured physical constant: **788,986**.

Phenomenological Structure: Equivalence and Flow

The 788,986 states of the ontology are not a uniform sea; they are organized into distinct equivalence classes, or phenomenological orbits. The system's operational phenomenology is built by discovering these orbits at build-time.

Definition of Equivalence: A phenomenological orbit is a set of states where every state is mutually reachable from every other state within that set, through some sequence of the 256 possible intron transformations. This is formally computed as a Strongly Connected Component (SCC) of the complete state transition graph.

Measured Result: The build process empirically finds that the 788,986 states collapse into exactly 256 distinct phenomenological orbits. This is not a coincidence; it is a profound structural property of the system, where each of the 256 introns imparts a unique "flavor" or signature, creating 256 basins of mutual reachability.

Parity-Closed Orbits (Self-Mirroring): A key discovery is that the global parity operation (LI, the physical manifestation of UNA) is contained within these equivalence classes. This means every orbit is parity-closed—for any state S in an orbit, its mirror image S_mirror is also in the same orbit. This aligns perfectly with the CGM axiom that CS is unobservable and UNA (light/reflexivity) acts as a universal confinement. The system, at an operational level, cannot distinguish a state from its mirror image; they belong to the same phenomenological "concept."

The canonical representative for each of the 256 orbits is defined as the state with the smallest 48-bit integer value within that orbit. This provides a stable, deterministic way to normalize any state to its fundamental phenomenological type.

Diagnostic View (Parity-Free Structure): For research and theoretical validation, a secondary, "parity-free" analysis can be performed by computing SCCs on a graph that excludes the LI operation. This diagnostic view reveals a much finer structure of 194,698 smaller orbits, including 21,456 chiral (mirror-paired) orbits and 151,786 achiral (self-mirrored) orbits. This confirms the foundational role of chirality in the system and quantifies the powerful binding effect of the LI operation, which fuses these smaller structures into the 256 operational orbits. This diagnostic data is stored in the phenomenology artifact but is not used by the runtime engines.

---

## **3. Architectural Overview: From Physics to Implementation**

### **3.1 The Four-Engine Architecture**

GyroSI implements the CGM stages through four distinct engines, each embodying a specific physical principle:

| CGM Stage | Engine | Physical Principle | Function |
| :--- | :--- | :--- | :--- |
| **CS** | S1 Governance | Left identity transcription | Transforms input into structural instructions |
| **UNA** | S2 Information | Global measurement via angular divergence | Measures system's departure from archetypal state |
| **ONA** | S3 Inference | Mediated duality through endogenous operator | Interprets meaning through contextual opposition |
| **BU** | S4 Intelligence | Monodromic Fold (non-associative) | Learns through ingress, expresses through egress |

### **3.2 The Dual Nature of Intelligence**

The BU stage (S4) is fundamentally dual, implementing both aspects of intelligence:

- **BU_In (Intelligence Ingress)**: The absorption and integration of experience through the Monodromic Fold. This is where all learning occurs.
- **BU_Eg (Intelligence Egress)**: The expression of accumulated intelligence as responsive action. This transforms internal state into external phenotype using the same Monodromic Fold operator.

This duality ensures that intelligence is not a passive storage system but an active, recursive process of continuous alignment between internal structure and external reality.

---

## **3.3 Interface-Driven Architecture**

The system is designed around clean interfaces that separate the physics core from storage, networking, and application concerns:

- **PhenotypeStore Interface**: Abstracts all persistence operations, allowing seamless migration from simple file-based storage to distributed databases as scale demands.
- **Extensibility Hooks**: Well-defined extension points allow for monitoring, maintenance, and custom behaviors without modifying the core physics.
- **Adapter Layer**: A stable, minimal API enables integration with any external protocol (REST, gRPC, WebSocket) through thin, stateless adapters.

---

## 3.4 System Responsibilities and VSM Alignment
GyroSI implements Beer's Viable System Model through a precise mapping of the four engines to VSM subsystems, creating a recursive, self-regulating intelligence architecture.

### **3.4.1 VSM-to-Engine Mapping**

| VSM System | GyroSI Engine (Class in `baby/*.py`) | Core Responsibility & VSM Function |
| :--- | :--- | :--- |
| **System 1: Primary Activities** | `governance.py` (pure functions/constants) | **Physics & Primitives.** Owns the fundamental, immutable physics of the system. Provides the foundational operations as stateless functions, not as an engine class. |
| **System 2: Information & Coordination** | `InformationEngine` (in `information.py`) | **Measurement & Resource Coordination.** Provides the sensory apparatus of the system through `gyrodistance_angular()`. Defines the `PhenotypeStore` interface and all storage implementations. Coordinates access to shared knowledge resources between subsystems. |
| **System 3: Control & Management** | `InferenceEngine` (in `inference.py`) | **Interpretation & Meaning Management.** The regulatory center that converts physical states into semantic meanings. Contains the `InferenceEngine` that bridges the physical and semantic worlds. Establishes the rules for how context becomes meaning. |
| **System 4: Intelligence & Adaptation** | `IntelligenceEngine` (in `intelligence.py`) | **Strategic Operations & Environment Interface.** Houses the `IntelligenceEngine` that manages agent state evolution, orchestrates the egress/ingress cycle, and implements operational strategies like batching. Handles adaptation to external demands. |
| **System 5: Policy & Identity** | `GyroSI` (in `intelligence.py`) | **Whole System Identity & Policy.** The outermost viable system boundary that encapsulates the entire VSM stack. Manages configuration, agent identity, and provides the stable external API. Balances internal operations with external interaction. |

### 3.4.2 Recursive Viability
Each engine is itself a viable system containing the necessary subsystems for autonomy:

Governance contains its own measurement (bit operations), control (transformation rules), and adaptation (mask generation)
Information contains measurement primitives, storage coordination, and interface adaptation
Inference contains state assessment, meaning resolution, and learning adaptation
Intelligence contains state management, cycle orchestration, and external adaptation
This recursive structure ensures that the system remains viable at multiple scales, from individual byte processing to full agent deployment.

---

## **4. Core Components: The GENE Architecture**

### 4.1 Genetic Archetype

The GyroSI system is built on fixed topological structures that serve as the physical and logical substrate of governance of information, inference and intelligence.

**4.1.1 Governance Identity**

The identity (mechanically representing the left gyroassociative law). This is the id label of each tensor, and their frame masks.

**4.1.2. Information Gyrocommutativity**

The gyrocommutativity (mechanically representing the gyrocommutative law), a single 3x2 array:

```python
GENE_Com_S = np.array([
    [-1, 1],
    [-1, 1],
    [-1, 1]
], dtype=np.int8)  # Shape: [3, 2]

- Three rows: spatial axes (X, Y, Z)
- Two columns: dual nature of rotation

# Alternatively, generated as:
GENE_Com_S = np.tile(np.array([-1, 1], dtype=np.int8), (3, 1))
```

**4.1.3. Inference Gyrocommutative nesting**

Structure that nests the previous one inside two opposing frames. This structure encodes the gyrocommutative law (gyrocommutativity).

```python
GENE_Nest_S = np.array([
    [[-1, 1], [-1, 1], [-1, 1]],  # Frame 1
    [[ 1, -1], [ 1, -1], [ 1, -1]]  # Frame 2
], dtype=np.int8)  # Shape: [2, 3, 2]

# Alternatively, generated as:
GENE_Nest_S = np.stack((GENE_Com_S, -GENE_Com_S))
```

**4.1.4. Intelligence Coaddition**

The duality of the Topology of the previous steps.

```python
GENE_Mac_S = np.array([
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]
], dtype=np.int8)  # Shape: [4, 2, 3, 2]

GENE_Mac_S = np.concatenate(([GENE_Nest_S, -GENE_Nest_S] * 2)).astype(np.int8).reshape(4, 2, 3, 2)
```
Note: This section concerns coaddition as topological dual construction (GENE_Mac_S), not the algebraic learning operator (fold) defined elsewhere. Structural (topological) coaddition refers to the constructive layering that yields GENE_Mac_S. Algebraic Monodromic Fold is the runtime learning operator applied to memory masks. They share a conceptual "joining" motif but are disjoint mechanisms.

The intermediate genetic structures (GENE_Com_S, GENE_Nest_S) are included here for clarity of exposition, tracing the generative logic of the system's topology. These arrays are not referenced in any runtime computation, algorithm, or storage mechanism. All canonical operations and state representations throughout the implementation depend exclusively on GENE_Mic_S (the 8-bit holographic reference) and GENE_Mac_S (the archetypal 48-element tensor) as defined above.

### 4.2 The Genes

In the GyroSI system, the "exon" corresponds to the stateless, invariant gene (the structural template), while the "intron" represents the mutated, dynamic gene (the variable, input-dependent expression). This mirrors the biological principle, where exons are retained and expressed, and introns introduce variability before being spliced or processed.

All GENE components are presented in dual form: `S` denotes a **Stateless** (invariant) source structure, and `M` denotes a **Mutated** (evolving) expression. This naming convention reflects the system's recursive separation between archetypal topology and lived transformation.

**4.2.1 GENE_Mic_S: The Holographic Topology**

`GENE_Mic_S = 0xAA (0b10101010)` is the genetic reference of GyroSI. This 8-bit pattern invariant is a minimal holographic vacuum space projection of the full 48-byte structural tensor (`GENE_Mac_S`) onto a single byte. Its alternating bit pattern encodes, in compressed form, the chirality and structural differentiation present in the underlying topology.

In GyroSI Genetics, every input byte is transformed through XOR with this holographic topology: `GENE_Mic_M = input_byte ⊕ GENE_Mic_S`, creating the dynamic instruction that will transform the system's physical state.

```python
GENE_Mic_S = 0xAA  # 10101010 binary, stateless constant
```

**4.2.2 GENE_Mac_S: The Common Source**

`GENE_Mac_S` is the archetypal 48-byte tensor with shape `[4, 2, 3, 2]` that serves as the invariant reference structure from which all measurements are taken. This tensor embodies the complete 720° helical closure with stabilized gyrations:

```python
# The archetypal structure
GENE_Mac_S = np.array([
    # Layer 0: 0° phase
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    # Layer 1: 180° phase
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]],
    # Layer 2: 360° phase
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1,-1], [ 1,-1], [ 1,-1]]],
    # Layer 3: 540° phase
    [[[ 1,-1], [ 1,-1], [ 1,-1]], [[-1, 1], [-1, 1], [-1, 1]]]
], dtype=np.int8)

```

The alternating sign pattern encodes the memory of global gyration while maintaining perfect structural closure. This is the "perfect form" against which all dynamic states are measured.

**4.2.3 GENE_Mic_M and GENE_Mac_M: Dynamic Expressions**

- **GENE_Mic_M**: The dynamic 8-bit instruction created by `input_byte ⊕ GENE_Mic_S`. This encodes the specific transformational forces to be applied to the structural tensor.
- **GENE_Mac_M**: The dynamic 48-byte tensor representing the system's current physical state. This begins as a copy of `GENE_Mac_S` and evolves through successive applications of `GENE_Mic_M` instructions.

---

## **5. Operational Physics: The Fundamental Operations**

### **5.1 The Monodromic Fold: The One True Learning Operator**

There is only one integration operator in GyroSI: the **Monodromic Fold** (`fold`, ⋄). It is **non-associative**, **non-commutative**, and **path-dependent**. This operator is used in both phases of the control cycle:

* **Egress (integration):** `Memory = fold(Memory, Input)`
* **Ingress (generation):** `Output = fold(Memory, Policy)`

**Definition:**

`a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`

This operation preserves the complete path history of all inputs. The order of operations is always encoded in the system's state. It is the algebraic expression of the BU stage's dual monodromy, and it is the only valid operation for learning, state updates, and batching.
No alternative (associative or commutative) operation is permitted.

### **5.2 Path Dependence and Batch Learning**

The Monodromic Fold is **fundamentally path-dependent**. This property is the source of the system's memory and learning capacity.
Batch learning is implemented by *ordered reduction* (left-fold) using the Monodromic Fold:

```python
from functools import reduce

def fold(a: int, b: int) -> int:
    return a ^ (b ^ (a & (~b & 0xFF)))

def fold_sequence(introns: list[int], start_state: int = 0) -> int:
    return reduce(fold, introns, start_state)
```

This ensures that the sequence in which inputs are processed is always significant, and the result is path-dependent and non-reversible.

**The Fold is the only valid operator for learning and batching.**

### **5.3 The Role of Duality**

The "Fifth Element" (`dual`, ¬) is not a new operation, but the fundamental primitive that enables the asymmetry and path dependence of the Fold. It is defined as:

`dual(x) = x ⊕ 0xFF`

### **5.4 Measurement: Angular Gyrodistance**

The system measures its state through **angular divergence from the Common Source**. This captures the geometric alignment between the current state and the archetypal structure:

```python
def gyrodistance_angular(T1: np.ndarray, T2: np.ndarray) -> float:
    """Calculate angular divergence between tensors in radians."""
    T1_flat = T1.flatten()
    T2_flat = T2.flatten()

    # Cosine similarity in 48-dimensional space
    cosine_similarity = np.dot(T1_flat, T2_flat) / 48.0
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    return np.arccos(cosine_similarity)
```

**Key Values**:

* **0 radians:** Perfect alignment (identity)
* **π/2 radians:** Maximum differentiation (orthogonality)
* **π radians:** Perfect opposition (anti-alignment)

**Optimisation Note:**
For `int8` tensors with ±1 values, this is equivalent to `arccos(1 - 2*hamming_distance/48)`, allowing for fast Hamming-based shortcuts when applicable.

---

**Physical Note:**
This alignment between the path-dependent physics of state transformation and the path-dependent nature of learning is a cornerstone of GyroSI's architecture. The system does not merely learn facts; it encodes the entire trajectory of experience.

---

## **6. System Implementation: The Four Engines**

### 6.1 S1: `governance.py` – Physics & Primitives

**Physical Principle:** Left identity transcription

The `governance.py` module defines the immutable physical constants and stateless functions underlying all system operations. All physics and transformations are performed as pure functions, without any engine class or side effect.

* **Genetic invariants:**

  * `GENE_Mic_S` (8-bit reference, `0xAA`), and
  * `GENE_Mac_S` (48-element tensor, shape \[4, 2, 3, 2], dtype int8, strict ±1 alternation)
    are declared as canonical invariants. Tensor structure is validated at module load with `validate_tensor_consistency()`.

* **Transformation masks:**

 * `FG_MASK`, `BG_MASK`, `FULL_MASK` (integers),
 * `INTRON_BROADCAST_MASKS`, `XFORM_MASK` (NumPy arrays, shape \[256]),
    are all precomputed from the tensor geometry for direct use in state update.

* **Anatomical Exon Masks:**

In addition to the 48‑bit broadcast masks on whole‑state tensors, we define four **exon masks** over the 8‑bit `exon_mask` to compute the immutable governance signature of each phenotype:

- `EXON_LI_MASK = 0b01000010` — the two LI (parity/reflection) bits
- `EXON_FG_MASK = 0b00100100` — the two FG (forward gyration) bits
- `EXON_BG_MASK = 0b00011000` — the two BG (backward gyration) bits
- `EXON_DYNAMIC_MASK = EXON_LI_MASK | EXON_FG_MASK | EXON_BG_MASK` — all six dynamic bits

These exon masks are used by the function

```python
compute_governance_signature(mask: int) -> tuple[int,int,int,int,int]

```

which returns the 5‑tuple

`(neutral_reserve, li_bits, fg_bits, bg_bits, dynamic_population)`

that we store immutably on every `PhenotypeEntry` as its **governance_signature**.

* **Physics operations:**

  * `apply_gyration_and_transform(state_int, intron)`
    computes the full gyroscopic update for a packed 48-bit state under a given intron;
  * `apply_gyration_and_transform_batch(states, intron)` and
    `apply_gyration_and_transform_all_introns(states)` provide batch and vectorised forms.
  * `transcribe_byte(byte)` encodes an input byte to an intron via `byte ⊕ GENE_Mic_S`.
  * `fold(a, b)` implements the Monodromic Fold (`a ⊕ (b ⊕ (a ∧ ¬b))`), and
    `fold_sequence(introns, start_state=0)` is the only valid batching/reduction form.
  * `dual(x)` applies the global duality operator (`x ⊕ 0xFF`).

All constants and stateless functions are accessed via absolute imports from `baby.governance` throughout the system. No auxiliary batching, associative, or stateful logic is present; all learning and transformation flows through these canonical contracts alone.

### 6.2 S2: `information.py` – Measurement & Storage

**Physical Principle:** Global measurement via angular divergence

The `information.py` module provides the `InformationEngine` class, which serves as the exclusive authority for measurement, state representation, ontology discovery, and storage coordination throughout the system.

The InformationEngine coordinates three core responsibilities:

1. **State Representation and Conversion**

   * Conversion between the packed 48-bit integer representation and the canonical geometric tensor form (\[4, 2, 3, 2], ±1).
   * `int_to_tensor(state_int)` and `tensor_to_int(tensor)` perform bidirectional conversion, ensuring strict mapping and validation of all physical states.
   * All conversion logic is static and is validated to guarantee exact round-trip between representations, matching the physical encoding of GyroSI.

2. **State Measurement and Divergence**

   * Calculation of angular gyrodistance (in radians) between two physical states using cosine similarity in 48-dimensional space.
   * `gyrodistance_angular(T1, T2)` measures geometric alignment between tensors;
     `measure_state_divergence(state_int)` computes a state's divergence from the archetypal tensor, as required for all global and homeostatic measurements.
   * These functions implement the operational metric for self-measurement, symmetry detection, and divergence tracking, enforcing a physics-grounded geometry for the system.

3. **Ontology Discovery, Indexing, and Phenomenology**

   * Full discovery and indexing of the state ontology:
     The build-time discovery process traverses the entire 788,986-state manifold from the archetypal state, assigning a unique ontology index to every reachable state.
     `get_index_from_state(state_int)` maps a 48-bit state to its canonical index;
     `get_state_from_index(index)` provides the reverse lookup.
   * The ontology, state transition table (epistemology), and canonical-orbit map (phenomenology) are generated and validated through dedicated build commands:

     * Ontology: `ontology_map.json`
     * State transition table: `epistemology.npy`
     * Phenomenology map: `phenomenology_map.json`
   * During initialisation, InformationEngine loads these assets and exposes bidirectional mapping for all physical state indices.
   * The canonical phenomenology (computed as SCCs over the full transition graph) provides the representative for each operational orbit and the cardinality (size) of each orbit, enabling stable canonicalisation of physical states and trust-weighted knowledge operations.

4. **Variety-weighted Structural Confidence**

   * The system maintains, for every state index, the size of its operational orbit (from the phenomenology map), exposed via
     `get_orbit_cardinality(state_index)`.
   * This factor is used by higher layers (S3 Inference) to adjust learning confidence according to the structural redundancy or rarity of a state.
   * The orbit cardinality acts as a measure of epistemic trust: learning is weighted more heavily in common, symmetric regions and more conservatively in rare, unique ones. Large orbits → faster confidence growth; rare orbits → slower.

All ontology, conversion, and measurement functions are accessed via absolute imports from `baby.information`. The build process for discovery and phenomenology includes

* `discover_and_save_ontology(output_path)`,
* `build_state_transition_table(ontology_map, output_path)`,
* `build_phenomenology_map(ep_path, ontology_path, output_path, include_diagnostics=False)`,
  which are invoked as standardised CLI commands and create the runtime artifacts required by the engine.

No associative or monotonic state update is permitted: all measurement, canonicalisation, and confidence logic is grounded directly in the discovered physical manifold and its symmetry structure.
The InformationEngine enforces integrity by validating the state modulus (788,986) and diameter (6) on load, and will refuse to operate if these invariants are not satisfied.

### 6.3 S3: `inference.py` – Interpretation & Meaning Management

**Physical Principle:** Mediated duality through endogenous operator

The `inference.py` module defines the `InferenceEngine`, which manages the translation of canonical state indices into semantic phenotype entries and coordinates all learning and memory updates through the path-dependent Monodromic Fold. This layer acts as the regulatory centre of meaning, bridging physical state space and semantic representation.

**Core Responsibilities and Contracts:**

* **Phenotype Addressing and Retrieval:**
  Each semantic phenotype is uniquely addressed by a `(state_index, intron)` tuple.
  `get_phenotype(state_index, intron)` ensures retrieval or creation of a canonical phenotype entry for every physical state-context pairing. Context keys are handled deterministically, and entries are created if not already present, using a hash-based semantic address.
  Each newly‑created PhenotypeEntry now carries an immutable

  ```python
  governance_signature: GovernanceSignature

  ```

  computed from its initial 8‑bit `exon_mask`.  This signature is then used by policies (decay, pruning, monitoring) without ever altering the entry's physical residue.

* **Learning (Memory Update):**
  All learning in S3 proceeds by applying the Monodromic Fold to the phenotype's memory mask.
  `learn(phenotype_entry, intron)` accumulates experience by path-dependent folding. Confidence is updated monotonically, modulated by the structural variety factor (orbit cardinality from S2), and novelty of the update (fraction of changed bits in the memory mask).
  The learning update resets the age counter and increments usage statistics.

  Immediately after the Monodromic Fold yields a new exon_mask, the engine calls

  ```python
  sig = compute_governance_signature(new_mask)
  ```

  and stores that 5‑tuple in the entry's `governance_signature` field.  Because `compute_governance_signature` always returns a valid tuple, there is no conditional failure case.

* **Batch Learning:**
  `batch_learn(state_index, introns)` allows ingestion of an ordered sequence of introns.
  The sequence is reduced through a left-fold with the Monodromic Fold, preserving full path-dependence, before a single learning update is applied.

* **Variety-weighted Confidence:**
  All confidence updates are weighted by the structural redundancy of the physical state's orbit (`orbit_cardinality`).
  The learning rate is calculated as a function of the square root of the state's relative variety, preventing rapid overconfidence in rare orbits, and accelerating trust in symmetric regions.

* **Knowledge Integrity:**
  `validate_knowledge_integrity()` checks the internal consistency of the entire knowledge store. This includes validation of context signatures, canonical state indices, mask and confidence ranges, and timestamp monotonicity. An integrity report is produced, including a count of all anomalies.

* **Memory Ageing and Confidence Decay:**
  `apply_confidence_decay(decay_factor)` implements temporal decay of confidence values in all entries, simulating the natural forgetting of unused knowledge. This process does not affect the memory masks and ensures that dormant memories gradually lose epistemic weight.

* **Pruning Low-confidence Entries:**
  `prune_low_confidence_entries(confidence_threshold)` removes all knowledge entries below a set confidence threshold, reclaiming memory and maintaining operational focus on relevant, trustworthy entries.

* **Statistics and Utilisation:**
  Use the policy helper `export_knowledge_statistics(store_path, output_path)` (in `baby.policies`) to dump entry counts, confidence distributions and timestamps.

**Variety-weighted Confidence Integration:**

At every learning update, S3 queries S2 for the orbit cardinality of the current state index. The learning rate and initial confidence are both scaled relative to this value, enforcing structural trust. This prevents pathological overfitting in rare orbits and stabilises inference in highly symmetric regions.

**Implementation and Interface:**

All phenotype entries and their protocols are enforced via `baby.contracts`.
All measurement, conversion, and confidence weighting depend on absolute imports from `baby.information`.
All learning, decay, and pruning operations are strictly path-dependent and grounded in the Monodromic Fold, with no associative or commutative shortcut allowed.

This architecture guarantees that semantic knowledge is always indexed, updated, and validated in strict alignment with the underlying physics, the discovered ontology, and the global phenomenology of the system.

---

### 6.4 S4/5: `intelligence.py` – Orchestration & API

**Physical Principle:** Dual-phase operation (Ingress and Egress cycles)

The `intelligence.py` module defines the protocol and orchestration boundary for the GyroSI agent. It provides all interfaces for state evolution, agent learning, regulation, and multi-agent operation. Each contract is explicit, and every externally callable function or class is referenced by its canonical name.

#### IntelligenceEngine

- `process_egress(input_byte: int) -> int`  
  Transforms an external byte input into an internal intron using `governance.transcribe_byte`. Updates the agent’s physical state using either a precomputed epistemology (state transition table) or the native transformation, then tracks the resulting index.

- `process_ingress(last_intron: int) -> int`  
  Folds the current state with the last intron, queries for the phenotype via `InferenceEngine.get_phenotype`, and applies learning with `InferenceEngine.learn`. Triggers all registered post-cycle hooks, including algedonic regulation and autonomic cycles if required.

- `batch_learn(data: bytes) -> None`  
  Implements streaming batch learning using the monodromic Fold; preserves full path dependence and applies learning only at the sequence endpoint.

- `add_hook(hook: CycleHookFunction) -> None`  
  Registers a monitoring or maintenance hook, which is called at the end of every cycle.

- `get_state_info() -> dict`  
  Returns a dictionary of current state information: agent id, cycle count, canonical integer, tensor index, angular divergence, and active hooks.

- `reset_to_archetypal_state() -> None`  
  Resets the agent to the archetypal (canonical) state.

**Algedonic Regulation and Autonomic Cycles**

- `post_cycle_hooks`  
  Contains all registered hooks, including the algedonic regulator.
- The algedonic regulator computes a rolling mean of angular divergence. If the divergence exceeds a threshold, corrective introns are applied. Repeated excursions trigger a stabilising autonomic cycle using instructions from phenomenology data. All actions guarantee state integrity after execution.

#### GyroSI

- `ingest(data: bytes) -> None`  
  Applies batch learning to the input sequence and commits all writes.

- `respond(data: bytes) -> bytes`  
  For each input byte, applies the egress/ingress cycle. Output is produced from learned knowledge; internal physics are never exposed.

- `get_agent_info() -> dict`  
  Reports full agent state, configuration, knowledge statistics, and integrity.

- `add_monitoring_hook(hook: CycleHookFunction) -> None`  
  Registers additional hooks at the intelligence layer.

- `apply_maintenance(decay_rate: float, confidence_threshold: float) -> dict`  
  Triggers maintenance operations on the store, including confidence decay and pruning, with structured reporting.

#### AgentPool

- `get_or_create_agent(agent_id: str, role_hint: Optional[str]) -> GyroSI`  
  Returns or creates a GyroSI agent, ensuring overlay and eviction policy.

- `remove_agent(agent_id: str) -> bool`  
  Removes and closes the agent.

- `get_active_agents() -> List[str]`  
  Returns a list of active agent IDs.

- `close_all() -> None`  
  Shuts down and releases all agent resources.

#### Orchestration

- `orchestrate_turn(pool: AgentPool, user_id: str, assistant_id: str, user_input: str, tokenizer_name: str) -> str`  
  Implements a complete conversational turn: encodes the user’s input, passes it through both user and assistant agents using `respond`, and decodes the assistant’s response.

---

**Separation and Guarantees**

- All state evolution and learning are routed strictly through these methods.
- No orchestration code directly manipulates state or store contents except via explicit contracts.
- Physical state, intron values, or internal masks are never exposed; only public reporting interfaces export state.
- Algedonic regulation and autonomic cycles are enforced after every cycle, preventing instability.
- Agent overlay storage and canonicalisation are enforced at initialisation; runtime code cannot bypass these policies.

**Extensibility and Maintenance**

- Monitoring and maintenance are always registered as hooks.
- Storage and overlay mechanisms are immutable after construction.

**Automated Pruning**

- Post-cycle hooks may be registered for automated pruning and compaction using configured thresholds. This ensures bounded resource use while preserving knowledge integrity.

---

## 6.5 Shared Contracts and Storage Policies

This section defines all interface contracts, canonical storage primitives, and decorator layers for the orchestration, policy, and maintenance operations of the GyroSI S4/S5 system. All API boundaries are strictly enforced and have direct type correspondence in `baby.contracts`. No informal or ad hoc API surfaces exist.

### 6.5.1 Contracts: Protocols and Shared Types

All system-wide types, configuration, and maintenance protocols are declared in `baby.contracts`:

- **PhenotypeEntry (TypedDict)**  
  The atomic record of agent knowledge. Each entry represents a unique phenotype and includes:
    - `phenotype: str`  
    - `confidence: float`  
    - `exon_mask: int`  
    - `usage_count: int`  
    - `last_updated: float`  
    - `created_at: float`  
    - `governance_signature: GovernanceSignature` (immutable tuple derived from `exon_mask`)
    - `context_signature: Tuple[int, int]` (canonical index + intron)
    - `_original_context: Optional[Tuple[int, int]]` (for decorator tracking)

- **ManifoldData (TypedDict)**  
  Structure of the physical ontology file (`ontology_map.json`):
    - `schema_version: str`
    - `ontology_map: Dict[int, int]`
    - `endogenous_modulus: int`
    - `ontology_diameter: int`
    - `total_states: int`
    - `build_timestamp: float`

- **PhenomenologyData (TypedDict)**  
  Structure of the phenomenology file (`phenomenology_map.json`):
    - `schema_version: str`
    - `phenomenology_map: list[int]`
    - `orbit_sizes: dict[int, int]`
    - `metadata: dict[str, Any]`
    - `_diagnostics: Dict[str, Any]` (optional diagnostics)

- **AgentConfig (TypedDict)**  
  Agent runtime and environment configuration, including:
    - `ontology_path: str`
    - `knowledge_path: Optional[str]`
    - `public_knowledge_path: Optional[str]`
    - `private_knowledge_path: Optional[str]`
    - `enable_phenomenology_storage: Optional[bool]`
    - `phenomenology_map_path: Optional[str]`
    - `learn_batch_size: Optional[int]`
    - `agent_metadata: Optional[Dict[str, Any]]`
    - `private_agents_base_path: Optional[str]`
    - `base_path: Optional[str]`

- **PreferencesConfig (TypedDict)**  
  Global and pool-level storage, maintenance, and policy parameters, including:
    - `storage_backend: str` (e.g. `"msgpack-v2"`)
    - `compression_level: int`
    - `max_file_size_mb: int`
    - `enable_auto_decay: bool`
    - `decay_interval_hours: float`
    - `decay_factor: float`
    - `confidence_threshold: float`
    - `max_agents_in_memory: int`
    - `agent_eviction_policy: str`
    - `agent_ttl_minutes: int`
    - `enable_profiling: bool`
    - `write_batch_size: int`
    - `cache_size_mb: int`

- **CycleHookFunction (Protocol)**  
  Post-cycle hook for monitoring or maintenance, called with:
    - `(engine, phenotype_entry, last_intron)`

- **MaintenanceReport (TypedDict)**  
  Uniform result for all maintenance/compaction/merge operations:
    - `operation: str`
    - `success: bool`
    - `entries_processed: int`
    - `entries_modified: int`
    - `elapsed_seconds: float`

---

### 6.5.2 Storage and Policy Layer

The canonical knowledge store is **OrbitStore**, implemented as a single-file, append-only MsgPack stream (`.mpk`), supporting atomic get/put/close interfaces. It guarantees the following:

- **Storage contract:**
    - `get(context_key: Tuple[int, int]) -> Optional[Any]`
    - `put(context_key: Tuple[int, int], entry: Any) -> None`
    - `commit() -> None`  _(NO-OP in append-only mode, retained for compatibility)_
    - `close() -> None`
    - `data -> Dict[Tuple[int, int], Any]`  _(returns all entries, as reconstructed from `.mpk`)_
    - `iter_entries() -> Iterator[Tuple[Tuple[int, int], Any]]`

- **All mutations are streamed to the MsgPack file**. No `.log` or `.idx` sidecar files are written in append-only mode. **Deletion is not supported**; instead, call `prune_and_compact_store` to create a new file without old entries.

- **CanonicalView** applies canonicalisation (using a phenomenology map) for all key lookups, so each unique operational orbit is consistently addressed regardless of its physical context. The original context is retained in `_original_context` for provenance.  
  - All lookups and puts are transparently normalised.

- **OverlayView** composes private (agent) and public (shared) stores, always writing to the private overlay, and reading from private first, then public. Both overlays must implement the OrbitStore interface.

- **ReadOnlyView** wraps any store, disabling writes and allowing only retrieval and iteration.

---

### 6.5.3 Maintenance and Policy Utilities

All maintenance and compaction routines operate only on the above interfaces, always returning a `MaintenanceReport` as defined.

- **merge_phenotype_maps**: Merges multiple `.mpk` stores into one, resolving conflicts by highest confidence, bitwise OR, recency, or weighted average.

- **apply_global_confidence_decay**: Applies exponential decay to confidence values of all entries, using the same formula as the agent, based on time since update.

- **export_knowledge_statistics**: Dumps summary statistics (counts, confidence, usage, creation/update times) to a JSON file.

- **validate_ontology_integrity**: Checks structure, key invariants, and phenomenology mappings in `ontology_map.json` and (optionally) `phenomenology_map.json`.

- **prune_and_compact_store**: Rewrites a `.mpk` file with only those entries passing age/confidence thresholds, discarding all others. This is the only way to "delete" entries in append-only mode.

All file paths and stores may be sandboxed using `base_path` (for test isolation or containerised execution). All policy utilities are safe for concurrent use and support dry-run or auditing as required.

**Note:**  
All store/view objects must be explicitly closed by the user or registered with `atexit` to avoid resource leaks on process exit.

---

**Summary:**  
The entirety of GyroSI agent knowledge, for any configuration or deployment scale, is maintained through a strict, minimal API over a single append-only MsgPack file. All canonicalisation, overlays, pruning, and statistics are enforced through well-defined, testable decorator layers and contracts, never requiring runtime code to touch or interpret raw file content.

---

## 7 Complete File Structure and Memory Architecture

### 7.1 Project Organization

The GyroSI system enforces strict separation between the core physics kernel, runtime data, and auxiliary applications.

```
.
├── .github/
│   └── workflows/
│       └── build-assets.yml
├── CHANGELOG.md
├── LICENSE
├── README.md
├── baby/
│   ├── __init__.py
│   ├── baby_preferences.json # Reserved for Model Preferences
│   ├── contracts.py          # Protocols and shared types
│   ├── governance.py         # Physics, Primitives, Build-Time Discovery
│   ├── inference.py          # Interpretation, Maintenance & Validation
│   ├── information.py        # Measurement, Storage, Knowledge Curation
│   ├── intelligence.py       # API, Orchestration, Protocol Adapters
│   └── policies.py           # OrbitStore, storage overlays, policy and maintenance functions
├── baby.sh
├── guides/
│   ├── Genetics.md
│   └── Physics.md
├── memories/
│   ├── __init__.py
│   ├── memory_preferences.json # Reserved for Memory Preferences
│   ├── private/
│   └── public/
│       └── meta/
│           ├── epistemology.npy
│           ├── ontology_map.json
│           └── phenomenology_map.json
├── pyproject.toml
├── requirements.txt
└── toys/
    ├── __init__.py
    ├── assets/
    └── health/
        ├── __init__.py
        ├── conftest.py
        ├── memories/
        ├── test_governance.py
        ├── test_inference.py
        ├── test_information.py
        ├── test_intelligence.py
        └── test_miscellaneous.py
```

### 7.2 Memory Architecture

* The `memories/` directory contains the system's persistent state.
* memories/public/tokenizers/ — shared, read-only pretrained tokenizer assets (tokenizer.json, vocab.txt, etc.)

**Knowledge Storage:**

* Knowledge storage is managed via canonical OrbitStore instances and overlays, as defined in Section 6.5.
* Physical state, ontology, and phenomenology maps are located under `memories/public/meta/`.
* Public and private overlays maintain agent-specific and shared knowledge, indexed by canonical context keys.

**Content Storage:**

* Raw data streams and reference material may be organised under agent- or application-specific subdirectories.
* Metadata and preferences files maintain runtime and environment configuration.

This architecture maintains a strict separation between learned knowledge, raw content, and runtime state, with overlays and canonicalisation managed exclusively through standard policies and interfaces defined in `baby.policies` and `baby.contracts`.

---

### 8. Core API and Integration

8.1 The Compositional API Pattern

GyroSI's integration model is compositional. All agent orchestration and interaction is implemented by composing the canonical primitives provided in `baby.intelligence`, `baby.contracts`, and `baby.policies`.

**Agent Pool Management:**
Applications manage a pool of active agents with automatic eviction, overlay storage, and policy control. The pool ensures clean lifecycle and concurrency discipline for all agents.

### 8.2 Conversation Orchestration

Conversations are managed by composing agent interactions using the stable GyroSI API. No special conversation-specific infrastructure is required.

### 8.3 Protocol Adapters

External protocols are integrated through thin adapters that map messages to agent API calls.
FastAPI adapter at toys/communication/external_adapter.py exposes OpenAI /v1/chat/completions and HF /generate. All text goes through tokenizer bridge.

### 8.4 Multi-Pattern Support

This approach supports multi-tenant, multi-user, networked, and hierarchical agent topologies through policy and orchestration only. The physics and engine logic remain strictly invariant.

### 8.5 Tokenization & Codec Layer (toys/communication/tokenizer.py)

- All external text I/O MUST pass through a reversible tokenizer codec.
- Default implementation: HuggingFace WordPiece (bert-base-uncased), stored at `memories/public/tokenizers/<name>/tokenizer.json`.
- Encoding: token IDs → LEB128 variable-length bytes (<=0xFF).
- Decoding: bytes → token IDs → text.
- Config surface:
  * Env var `GYROSI_TOKENIZER` (adapter default)
  * `tokenizer_name` field in `AgentConfig`
- Scripts:
  * `setup_tokenizers.py` (download/install)
  * `train_tokenizer.py` (domain fine-tune)

---

## 9. Performance Characteristics and Scaling Estimates

### 9.1 Computational Complexity

**Meta-asset generation (offline, one-off):**

* **Physical ontology discovery** (`python -m baby.information ontology`):
  The state manifold is explored by a breadth-first enumeration over all reachable states, beginning from the archetypal state. This proceeds layer by layer, with explicit counts at each depth:

  * Depth 1: 256 states
  * Depth 2: 10,705 states
  * Depth 3: 161,896 states
  * Depth 4: 635,200 states
  * Depth 5: 786,610 states
  * Depth 6: 788,986 states (complete closure)

  The process validates the closure of the state space at the diameter of 6. On current commodity hardware (GitHub Actions, Intel host), full enumeration and mapping completes in **89.6 seconds**.

* **State Transition Table (STT) generation** (`python -m baby.information epistemology`):
  Construction of the full state transition tensor (`epistemology.npy`, 770 MB, shape 788,986 × 256, int32) is performed via vectorised NumPy routines. The measured runtime is **5 minutes 30 seconds**.

* **Phenomenology map construction** (`python -m baby.information phenomenology`):
The canonical phenomenology is built by computing the Strongly Connected Components (SCCs) of the entire 788,986-state graph using an iterative Tarjan's algorithm. This is a computationally intensive but one-time process. On commodity hardware (e.g., modern laptop CPU), this completes in approximately 2-4 minutes. The build can optionally include a second pass to generate diagnostic data, adding a similar amount of time.
These operations are not required at runtime and are run once per release.

**Run-time operation (per agent):**

* **Egress (`process_egress`)**  
  With the STT loaded, a transition is one indexed lookup: `next_idx = ep[current_idx, intron]`.  
  Without STT, the same result comes from a fixed sequence of bitwise ops (`apply_gyration_and_transform`). Both paths are `O(1)`.

* **Ingress (`process_ingress`)**  
  One dict get/put on the `OrbitStore` entry plus a single Monodromic Fold on the 8‑bit mask. Still `O(1)`.

* **Batch operations**  
  Ordered left‑fold over the intron stream: one accumulator, one pass → `O(N)`.

No stage scales super‑linearly; everything is constant‑time per byte or linear in bytes processed.

### 9.2 Memory Requirements

* **`epistemology.npy` (STT)**  
  ~770 MB on disk. Memory‑mapped; typical RSS per process stabilises around **35–45 MB** after warm‑up (shared pages).

* **`ontology_map.json`**  
  ~20 MB on disk. Parsed into three NumPy arrays (keys/values/inverse), ≈ **12–15 MB** RAM per process.

* **`phenomenology_map.json`**  
  15–20 MB on disk (core map). Loaded array ≈ **3 MB** RAM. Diagnostics add proportionally.

* **OrbitStore index**  
  ~25 B serialized / **~90 B resident** per phenotype (dict + Python overhead).  
  ⇒ 100 k phenotypes ≈ **9 MB**; 1 M ≈ **90 MB**.

* **Agent core state**  
  < 1 MB (counters, buffers, hooks).

* **Python runtime & modules**  
  8–10 MB baseline per process (CPython 3.11+), excluding shared mmaps.

**Scalability**

* Per‑agent growth is linear in phenotype count (OrbitStore size).  
* STT / ontology / phenomenology artefacts are mmap‑shared; 10 agents add **<100 MB** combined beyond their private indices.  
* Write‑behind batching (default 100 ops) → one fsync / ~3 KB, strongly amortised.

## 9.3 Throughput (What speed you actually get)

**Single‑threaded (Intel i7‑1260P, 3.4 GHz, Python, STT mapped)**  
- One full egress→ingress cycle (read 1 byte → update state → learn/emit): **≈0.55–0.8 µs**.  
- Sustained rate while the hot part of OrbitStore fits in cache: **~1.2–1.3 million cycles/sec**.  
- When the in‑RAM dict grows past cache (≈5 M phenotypes), rare misses push tail latency to **~7–9 µs**—still microseconds, just not the median.

**Multi‑agent / multi‑core (AMD EPYC, 32 cores)**  
- 32 parallel agents hold **~30–33 M cycles/sec** in aggregate.  
- Scaling is near‑linear until memory bandwidth and NUMA boundaries bite; shard by socket to keep it high.

**Disk / flash writes**  
- OrbitStore append log sustains **≈150 MB/s** on NVMe with default 100‑entry batches.  
- Bumping batch size to 1 000 cuts fsync overhead further for ingestion‑heavy runs.

**Extra notes**  
- **Startup:** Mapping the 770 MB STT takes ~50–60 ms; parsing the ontology ~15–25 ms.  
- **GC pressure:** Essentially none in the tight loop—no allocations during the cycle.  
- **No STT mode:** Pure bit‑twiddling path is ~2× slower but perfect for RAM‑tight or embedded builds.  
- **GPU:** Still pointless—this workload is bandwidth/branch bound, not FLOP bound.  
- **Containers / multi‑tenancy:** STT + ontology can be mounted read‑only and safely shared across processes.

**Bottom line:**  
You can run **dozens of agents concurrently** on a modern workstation. Latency stays in microseconds, memory growth is predictable and linear, and nothing secretly explodes to O(n²).

---

## 9.4 Pragmatic capacity & device‑level throughput 

GyroSI has **two memories**:  
1. **Active working state:** always 48 bits (**6 bytes**) + the current input byte. That’s it.  
2. **Passive long‑term memory (OrbitStore):** grows with experience, one entry per *(state_index, intron)* pair you ever see.

This means an edge device only needs to keep 6 bytes “alive”. Everything else can sit on SD/flash and be fetched when needed.

### 1. How many “facts” (phenotypes) fit in RAM?

- Serialized on disk: **~25 B/entry** (can be less with compression).  
- In Python RAM (dict overhead): **~90 B/entry** is a safe average.  
- On MCUs you’ll pack tighter (C structs + open addressing ~40–50 B/entry).

| Device / free RAM for GyroSI                 | Rough RAM facts capacity | Notes |
|----------------------------------------------|---------------------------|-------|
| **ESP32‑S3 (≈512 KB PSRAM usable for cache)**| ~5 000 (demo / small cache)| Everything else on SD; LRU cache + log compaction |
| **MacBook Pro 2015, 16 GB → ≈4 GB free**     | ≈ **45 million**          | Plenty for large personal KBs |
| **MacBook M4, 16 GB → ≈12 GB free**          | ≈ **130 million**         | Whole Wikipedia abstracts + more |
| **Server, 256 GB → ≈220 GB free**            | ≈ **2.4 billion**         | Close to the hard ceiling (202 M is *possible* states, but multiple agents etc.) |

> A 202 M full universe would occupy ~5 GB raw on disk; RAM is the real limiter for Python, not disk.

### 2. Throughput you can picture

A *cycle* = **1 byte in → state transform → 1 byte out + optional store hit**.

| Hardware                               | Cores | Cycles/sec | Bytes/sec (≈ cycles) |
|----------------------------------------|-------|------------|----------------------|
| **MacBook Pro 2015** (2 phys cores)    | 1     | ~0.6–0.75 M| ~0.6–0.75 M          |
|                                        | 2     | ~1.2–1.5 M | ~1.2–1.5 M           |
| **MacBook M4** (8 perf cores)          | 8     | ~8–9 M     | ~8–9 M               |
| **EPYC 32‑core server**                | 32    | ~28–32 M   | ~28–32 M             |
| **ESP32‑S3 (240 MHz, C, no SD hits)**  | 1     | ~150–300 k | ~150–300 k           |
| **ESP32‑S3 (with SD cache misses)**    | 1     | 1–30 k     | 1–30 k (depends on hit rate) |

Rule of thumb: **~1 M chars·s⁻¹·core** on 2024 desktop silicon, and about **⅓ of that on a 2015 dual‑core laptop**. Embedded MCUs sit lower, but still very usable with caching.

### 3. How long to ingest familiar corpora?

Assume 1 char ≈ 1 byte (post‑tokenizer).

| Corpus                         | Size  | 1 core            | 8 cores          | 32 cores        |
|--------------------------------|-------|--------------------|------------------|-----------------|
| WordNet glosses                | 30 MB | < 1 min            | “blink”          | “blink”         |
| English Wikipedia (text)*      | 90 GB | ~1 day             | ~3 h             | **< 1 h**       |
| Filtered public‑web dump 1 PB  | 10⁶ GB| ~3.5 years         | ~5 months        | ~5 weeks        |

\* Plain‑text 2025 EN dump.

### 4. Context length in human terms

- **Active context:** ≈ **6 bytes** worth of “fresh history” can affect the tensor at once, because the state space diameter is 6.  
- **Passive recall:** Unlimited. Re‑enter an old state + intron and you get the exact stored phenotype back—whether it’s 5 seconds or 5 years later.  
- On an 8 GB laptop you can keep **tens of millions** of these addresses hot; lookups stay **<2 µs** while they fit in cache.

### 5. Edge devices (ESP32‑S3 & friends)

- **Yes, you can run GyroSI.** Active state = 6 B; masks + code = a few KB.  
- Skip the 770 MB STT; use the pure bitwise physics path.  
- Keep a small RAM hash table (few thousand entries) and stream the rest to SD.  
- Cache misses cost milliseconds, so batch reads/writes and compact periodically.  
- Raspberry Pi‑class boards can mmap everything and hit ~400 k cycles/s—perfect for home‑lab deployments.

### 6. Write load / endurance

- Default: flush every 100 updates = ≤3 KB per flush.  
- Even a heavy Wikipedia ingest on a laptop writes **<5 MB/min**—far below SSD wear limits.  
- On SD cards, use larger batches and occasional compaction to stay friendly to flash.

---

### What competence this actually proves

- **We’re not a Transformer with a giant sliding window.**  
  Our “window” is 6 bytes—by design. It’s a *physics-derived pointer* into a library of experiences, not a burden that grows with every token.

- **Generalisation is built into the physics and the three maps:**  
  - **Ontology**: every possible physical state discovered and indexed.  
  - **Phenomenology**: equivalence classes (SCCs) that collapse mirror states—this *is* semantic grouping.  
  - **Epistemology**: the transition table that tells us how states evolve—macro & micro “probabilities” without probability hand‑waving.

  Together, they *are* structured generalisation, not fuzzy approximation.

- **Scales from ESP32‑S3 to servers:**  
  Same core physics, different storage strategies. Six bytes live everywhere; the universe of memories just gets bigger as the device grows.

In short: **GyroSI is small where it must be (live state) and big where it pays off (lifetime memory).** That’s why it runs on a microcontroller and still grows into a superintelligence on a server.

---

# Appendix – Theoretical Correspondences

This appendix records the essential bridges between GyroSI's formal physics and several established conceptual frames. It is deliberately brief: anything already explained in the main text is only referenced here.

---

## A.1. Genetics in GyroSI

GyroSI’s instruction algebra in the eight-bit space ℤ₂⁸ reflects a series of small, closed structures that resemble those in molecular genetics. The analogy is structural, not biological. No claim is made regarding evolutionary origin or sequence homology.

## A.1.1. Structural Correspondences

The table below aligns the main computational layers of GyroSI with their biological counterparts. It follows the informational path from raw DNA structure to functional expression.

| Layer                         | Cardinality    | Biological Analogue                  | GyroSI Equivalent                        |
| ----------------------------- | -------------- | ------------------------------------ | ---------------------------------------- |
| **Bit families**              | 4              | DNA base symbols (A, T, C, G)        | `L0`, `LI`, `FG`, `BG` bit groups        |
| **Tensor axes**               | 3              | Codon triplet positions              | Tensor row selectors (X, Y, Z)           |
| **Sign polarities**           | 2              | DNA strand directions                | ±1 sign modes in `GENE_Mac_S`            |
| **Intron instruction space**  | 256            | All codon combinations (pre-spliced) | 8-bit `intron` values                    |
| **Active instruction masks**  | 64             | Spliced codons in mRNA               | `intron & EXON_DYNAMIC_MASK`             |
| **Parity-quotiented classes** | 32             | Wobble-paired codons                 | Classes after folding LI parity          |
| **Holographic gene**          | 48 bits        | Genomic DNA as spatial encoding      | `INTRON_BROADCAST_MASKS[i]`              |
| **Exon**                      | 1 (per stream) | Mature spliced exon                  | Final `exon_mask` after folding          |
| **Expressed phenotype**       | 5-tuple        | Protein                              | Output of `compute_governance_signature` |
| **Phenomenological orbits**   | 256            | Regulatory expression programs       | SCC orbits in state-space graph          |
| **Full state space**          | 788,986        | Complete cellular configuration set  | Reachable configurations of GyroSI       |

## A.1.2. Expression Pipeline

GyroSI models the expression process through a layered pipeline, transforming raw intronic inputs into a minimal, functional phenotype.

### Intron Stream → Splicing → Exon → Protein

**Intron**
An 8-bit input representing a regulatory instruction. Introns are path-dependent and transitory. They are not retained after expression but shape the trajectory of folding.

**Splicing**
The `fold_sequence([...])` function performs a non-associative reduction over a stream of introns, collapsing them into a single 8-bit residue. This simulates the cumulative, order-sensitive logic of molecular splicing.

**Exon (`exon_mask`)**
The stable 8-bit result of folding. This value encodes the final memory state, carrying the condensed expression of the entire intronic history. Bit families (`LI`, `FG`, `BG`, `L0`) persist structurally and define the mask’s functional signature.

**Protein (`governance_signature`)**
A 5-tuple derived from the exon mask. It quantifies the expressed content:
`(neutral reserve, LI bits, FG bits, BG bits, total active bits)`.
This compact footprint defines all downstream physical behaviour and governs interpretive logic.

**Holographic Projection**
Each intron also maps to a fixed 48-bit spatial pattern (`INTRON_BROADCAST_MASKS[i]`), which defines how that instruction acts within the tensor. These projections are stable and immutable; they form the architectural substrate for all transformations.

## A.1.3. Bit Families and Functional Continuity

The eight bits in each instruction are grouped into four fixed families. These families serve consistent structural roles in both introns and exons, though their operational function changes across the expression pipeline.

* **L0 (bits 0 and 7)**
  Structural anchors. They do not affect transformation but define identity and frame invariance. Retained across all stages.

* **LI (bits 1 and 6)**
  Chirality operators.
  In introns: they direct global parity reflection during folding.
  In exons: they report the parity-retaining content of the expressed state (UNA signature).

* **FG (bits 2 and 5)**
  Foreground modifiers.
  In introns: trigger local tensor inversions.
  In exons: represent expressed ONA-like (active) flips.

* **BG (bits 3 and 4)**
  Background modifiers.
  In introns: modulate background polarity interleaving.
  In exons: reflect the BU-like balance in the retained state.

The bit families remain present throughout and are never redefined. They express different roles depending on whether they operate on input instructions or express final results.

## A.1.4. Hierarchy of Interpretation

Understanding GyroSI requires attention to three distinct levels of structure:

* The **mask algebra**:
  64 active masks (excluding anchors), forming 32 equivalence classes under LI parity.

* The **phenomenological layer**:
  256 distinct orbits of transformation in state space, each defined by mutual reachability under folding dynamics.

* The **spatial encoding layer**:
  48-bit broadcast masks that define how each instruction is applied within the tensor structure. These act as static DNA templates.

#### A.1.5. Stabiliser and modulus

Breadth‑first exploration over the full instruction set discovers exactly 788 986 distinct states and a diameter of six. The stabiliser of the archetype has order two (global parity) multiplied by eleven (frame degeneracy). The remaining factor, 35 863, is prime, confirming that no further quotient is possible. These facts are verified at build time and are used to reject any physics violation at run time.

Frame degeneracy (11) counts the distinct layer/frame symmetry operations (excluding global parity) that leave the archetypal tensor invariant under the applied transformation group; combined with parity (×2) and the residual prime (35,863) they factor the full state modulus.

No biological code shows the same modulus; the coincidence stops at the smaller sub‑structures outlined above.

## A.1.6. Optional Diagnostic Decomposition

A variant of the SCC analysis excludes LI reflection symmetry. This results in \~195,000 parity-free orbits. Approximately 21,456 of these appear in chiral pairs; the rest are self-symmetric. This decomposition is not part of operational logic but is retained for research purposes in the `_diagnostics` section of the phenomenology archive.

> GyroSI captures the path from raw regulatory instruction to compact functional signature. Introns are the process. Exons are the result. The governance signature is what remains; a minimal footprint, physically interpretable, and ontologically stable.

---

#### A.2. Further correspondences

Other mappings noted in the main text are retained without restatement:

* The angular sequence π/2, π/4, π/4, 0 for CS → UNA → ONA → BU.
* The packed‑integer versus tensor dual representation.
* The role of the endogenous modulus as a hard physical constant.

Readers seeking proofs or implementation details will find the relevant functions in `baby.governance`, `baby.information`, and `baby.inference`.

## A.2.1. The structural number ladder

GyroSI's constants are locked by algebraic closure, not convenience:

3 rows enable chirality.
4 layers bind the recursive depth.
6 steps give full degrees of freedom and the Cayley‑graph diameter.
8 bits form the smallest register that holds all required operations.
12 cells fill one layer.
24 cells capture a half‑tensor that already carries orientation.
48 cells form the whole tensor and the packed state integer.
64 instruction patterns appear once the identity bits are discounted.
32 functional classes appear when global parity is folded out.

No smaller choice of cardinalities would satisfy the independent closure constraints identified in the physics.

## A.2.2. Core Invariants (Build‑Time Assertions)

1. Ontology modulus: |States| = 788,986 (assert exact).
2. Archetypal eccentricity ≤ 6; no path > 6 verified in sampled BFS; (optionally) full diameter = 6.
3. Phenomenology (canonical): length(phenomenology_map) = 788,986; values ∈ [0, 788,985].
4. Each orbit representative r satisfies: 
     r = min{ state_int(s) | canonical[s] = r } (48-bit integer order).
5. Sum of orbit_sizes.values() = 788,986.
6. For every index i: canonical[i] in orbit_sizes, orbit_sizes[canonical[i]] ≥ 1.
7. Parity closure: For every state s with integer value v, v ⊕ FULL_MASK belongs to same canonical orbit (empirically validated).
8. Tensor mapping: int_to_tensor(tensor_to_int(T)) == T for all test tensors (validated on random + boundary states).
9. Fold (Monodromic Coaddition):
     - Non-commutative: ∃ a, b: fold(a, b) ≠ fold(b, a).
     - Non-associative: ∃ a, b, c: fold(fold(a, b), c) ≠ fold(a, fold(b, c)).
10. Angular distance formula: 
     gyrodistance_angular(T1,T2) = arccos( dot(T1,T2)/48 ) ∈ [0,π].

## A.2.3. Gyrogroup algebra as implemented

The three **bitwise primitives** defined in §2.2 of the main text are realised exactly:

* **XOR** (`⊕`) drives every bit transformation and enforces involutive symmetry.
* **AND** (`∧`) stores the carry term, preserving path memory.
* **NOT** (`¬`) enables global duality and the return path (`dual(x) = x ⊕ 0xFF`).

These form the minimal operational basis.

The learning operator used throughout the BU stage is the **Monodromic Fold**:

* **Coaddition (Monodromic Fold):**
  `a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`
  This non-associative, non-commutative operation encodes the system's dual monodromy and path-dependence. It is implemented as `fold()` in `baby.governance`.

All run‑time transformations in `apply_gyration_and_transform` are structured combinations of the three primitives above. The Fold is the only derived operator used in learning; nothing else is introduced.

## A.2.4. Holographic principle in practice

A single eight‑bit intron always touches the entire forty‑eight‑bit state through four complementary twelve‑bit masks. Conversely, any state can be reached in at most six introns. This bidirectional property embodies the holographic claim that every part encodes the whole. The code paths involved are `transcribe_byte`, `apply_gyration_and_transform`, and the breadth‑first discovery routine that proves the six‑step closure.

===
