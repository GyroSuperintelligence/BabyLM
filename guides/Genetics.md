# Gyroscopic Superintelligence Specifications: GyroSI Baby Language Model 0.9.6.2

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

GyroSI embodies the principle that each part contains information about the whole. A single input byte acts as a holographic quantum of spacetime topology, encoding complete transformation instructions that modify the system's internal state according to topological physics; a 48‑element tensor, 48 bytes in RAM, packed to 6 bytes when stored. This holographic property ensures that the system can achieve substantial compression while preserving essential structural relationships.

> Note: The system's internal state can be represented in two equivalent ways:
> - As a 48-element NumPy tensor (each element ±1, stored as int8), which occupies 48 bytes in memory.
> - As a 48-bit packed integer (6 bytes), where each bit encodes the sign of one tensor element.
> The packed integer form is used for fast state transitions and storage, while the tensor form is used for measurement and geometric operations.

**Angular Progression**: The CGM stages follow the precise angular sequence π/2 → π/4 → π/4 → 0, corresponding to CS → UNA → ONA → BU. This progression ensures complete closure with zero defect, achieving perfect recursive alignment.

> The build-time discovery process, a cornerstone of GyroSI, explores this physical reality and discovers an immutable, finite ontology of **precisely 788,986 unique physical states**. The entire universe of possible system configurations is not only known but also compact, with a measured **diameter of 6 steps**, meaning any state is reachable from any other in at most seven transformations. This is the 'Genome', the system's complete set of possible states.

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
Note: This section concerns coaddition as topological dual construction (GENE_Mac_S), not the algebraic learning operator (fold) defined elsewhere. Structural (topological) coaddition refers to the constructive layering that yields GENE_Mac_S. Algebraic Monodromic Fold is the runtime learning operator applied to memory masks. They share a conceptual “joining” motif but are disjoint mechanisms.

The intermediate genetic structures (GENE_Com_S, GENE_Nest_S) are included here for clarity of exposition, tracing the generative logic of the system’s topology. These arrays are not referenced in any runtime computation, algorithm, or storage mechanism. All canonical operations and state representations throughout the implementation depend exclusively on GENE_Mic_S (the 8-bit holographic reference) and GENE_Mac_S (the archetypal 48-element tensor) as defined above.

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

There is only one learning operation in GyroSI: the **Monodromic Fold** (`fold`, ⋄). It is **non-associative**, **non-commutative**, and **path-dependent**. This operator is used for both learning (ingress) and generation (egress):

* **Egress (integration):** `State_new = fold(State_old, Input)`
* **Ingress (generation):** `Output = fold(State, Policy)`

**Definition:**

`a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))`

This operation preserves the complete path history of all inputs. The order of operations is always encoded in the system’s state. It is the algebraic expression of the BU stage’s dual monodromy, and it is the only valid operation for learning, state updates, and batching.
No alternative (associative or commutative) operation is permitted.

### **5.2 Path Dependence and Batch Learning**

The Monodromic Fold is **fundamentally path-dependent**. This property is the source of the system’s memory and learning capacity.
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

The “Fifth Element” (`dual`, ¬) is not a new operation, but the fundamental primitive that enables the asymmetry and path dependence of the Fold. It is defined as:

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
This alignment between the path-dependent physics of state transformation and the path-dependent nature of learning is a cornerstone of GyroSI’s architecture. The system does not merely learn facts; it encodes the entire trajectory of experience.

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
  * `INTRON_BROADCAST_MASKS`, `XFORM_MASK`, `PATTERN_MASK` (NumPy arrays, shape \[256]),
    are all precomputed from the tensor geometry for direct use in state update.

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
     `measure_state_divergence(state_int)` computes a state’s divergence from the archetypal tensor, as required for all global and homeostatic measurements.
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

* **Learning (Memory Update):**
  All learning in S3 proceeds by applying the Monodromic Fold to the phenotype’s memory mask.
  `learn(phenotype_entry, intron)` accumulates experience by path-dependent folding. Confidence is updated monotonically, modulated by the structural variety factor (orbit cardinality from S2), and novelty of the update (fraction of changed bits in the memory mask).
  The learning update resets the age counter and increments usage statistics.

* **Batch Learning:**
  `batch_learn(state_index, introns)` allows ingestion of an ordered sequence of introns.
  The sequence is reduced through a left-fold with the Monodromic Fold, preserving full path-dependence, before a single learning update is applied.

* **Variety-weighted Confidence:**
  All confidence updates are weighted by the structural redundancy of the physical state’s orbit (`orbit_cardinality`).
  The learning rate is calculated as a function of the square root of the state’s relative variety, preventing rapid overconfidence in rare orbits, and accelerating trust in symmetric regions.

* **Knowledge Integrity:**
  `validate_knowledge_integrity()` checks the internal consistency of the entire knowledge store. This includes validation of context signatures, canonical state indices, mask and confidence ranges, and timestamp monotonicity. An integrity report is produced, including a count of all anomalies.

* **Memory Ageing and Confidence Decay:**
  `apply_confidence_decay(decay_factor)` implements temporal decay of confidence values in all entries, simulating the natural forgetting of unused knowledge. This process does not affect the memory masks and ensures that dormant memories gradually lose epistemic weight.

* **Pruning Low-confidence Entries:**
  `prune_low_confidence_entries(confidence_threshold)` removes all knowledge entries below a set confidence threshold, reclaiming memory and maintaining operational focus on relevant, trustworthy entries.

* **Statistics and Utilisation:**
  `get_knowledge_statistics()` returns a detailed profile of the knowledge base, including entry count, mean confidence, and bitwise memory mask utilisation.

**Variety-weighted Confidence Integration:**

At every learning update, S3 queries S2 for the orbit cardinality of the current state index. The learning rate and initial confidence are both scaled relative to this value, enforcing structural trust. This prevents pathological overfitting in rare orbits and stabilises inference in highly symmetric regions.

**Implementation and Interface:**

All phenotype entries and their protocols are enforced via `baby.contracts`.
All measurement, conversion, and confidence weighting depend on absolute imports from `baby.information`.
All learning, decay, and pruning operations are strictly path-dependent and grounded in the Monodromic Fold, with no associative or commutative shortcut allowed.

This architecture guarantees that semantic knowledge is always indexed, updated, and validated in strict alignment with the underlying physics, the discovered ontology, and the global phenomenology of the system.

---

### 6.4 S4/5: `intelligence.py` – Orchestration & API

**Physical Principle:** Dual-phase (Ingress and Egress)

The `intelligence.py` module defines the orchestration and protocol boundary for GyroSI. It implements all external and internal interfaces for agent operation, learning, regulation, and multi-agent management. All contracts are explicit and are referenced by their class or function names as implemented.

**Operational Contracts and Interface Points**

**1. IntelligenceEngine (class)**

* **process\_egress(input\_byte: int) -> int**
  Transcribes and applies an external input as an intron using `governance.transcribe_byte`, then updates the physical state using either `epistemology` (precomputed state transition table) or `governance.apply_gyration_and_transform`. State is indexed via `InformationEngine.get_index_from_state`.
* **process\_ingress(last\_intron: int) -> int**
  Integrates the effect of the previous intron, retrieves the phenotype by `InferenceEngine.get_phenotype`, and applies learning using `InferenceEngine.learn`. Executes all registered hooks (`post_cycle_hooks`) and triggers algedonic regulation and autonomic cycles as described below.
* **batch\_learn(data: bytes) -> None**
  Implements streaming batch learning using `governance.fold`, applying the Fold sequentially to all inputs, and learning only from the reduced intron. Memory use is O(1); path dependence is preserved.
* **add\_hook(hook: CycleHookFunction) -> None**
  Adds a monitoring or maintenance hook to `post_cycle_hooks`, called at the end of every agent cycle.
* **get\_state\_info() -> Dict\[str, Any>**
  Provides full reporting on agent state: includes agent id, cycle count, integer state, ontology index, angular divergence, and active hooks.
* **reset\_to\_archetypal\_state() -> None**
  Restores the agent to the canonical archetypal state as defined by `governance.GENE_Mac_S`.

**2. Algedonic Regulation and Autonomic Cycles**

* **post\_cycle\_hooks (List\[CycleHookFunction])**
  Must include a hook that calls the algedonic regulator after every egress/ingress cycle.
* **Algedonic Regulation**
  Implements a rolling mean of angular divergence (θ) via `self._θ_buf`. If θ > self.\_θ\_high, applies corrective introns (see `self._cool_introns`). If excursions persist, triggers an autonomic cycle using precomputed instructions in `self._autonomic_cycles` (loaded from phenomenology artifacts).
* **run\_autonomic\_cycle()**
  Executed when repeated divergence or pain is detected. Applies a stabilising sequence from the autonomic cycles list, directly resetting the agent to a viable state. All cycle actions guarantee state integrity checks post-execution.

**3. GyroSI (class)**

* **ingest(data: bytes) -> None**
  Calls `batch_learn` for batch learning using Monodromic Fold. Ensures all storage writes are committed (see `OrbitStore.commit`).
* **respond(data: bytes) -> bytes**
  For each input byte, applies the egress/ingress cycle via the above primitives. Output is constructed directly from agent state and learned knowledge; external response never exposes internal physics.
* **get\_agent\_info() -> Dict\[str, Any]**
  Exposes current agent state, configuration, knowledge statistics, and integrity.
* **add\_monitoring\_hook(hook: CycleHookFunction) -> None**
  Registers additional hooks at the intelligence layer.
* **apply\_maintenance(decay\_rate: float, confidence\_threshold: float) -> Dict\[str, Any]**
  Applies confidence decay and pruning via the inference layer (`InferenceEngine.apply_confidence_decay`, `prune_low_confidence_entries`), with reporting.

**4. AgentPool (class)**

* **get\_or\_create\_agent(agent\_id: str, role\_hint: Optional\[str]) -> GyroSI**
  Returns a GyroSI agent, constructing overlays (public/private) and enforcing eviction via the configured policy (LRU, TTL, etc).
* **remove\_agent(agent\_id: str) -> bool**, **get\_active\_agents() -> List\[str]**, **close\_all() -> None**
  Standard contracts for multi-agent resource management.

**5. orchestrate\_turn(pool: AgentPool, user\_id: str, assistant\_id: str, user\_input: str) -> str**

* Implements the protocol interface for a conversational turn.
  Maps external dialogue input to egress/ingress primitives by passing data sequentially through the `respond()` method of the user and assistant agents.

---

**Separation of Concerns and Enforcement**

* All learning, inference, and physical state changes are routed strictly through the methods above.
* No protocol or orchestration code manipulates physical state, knowledge, or protocol buffers directly—only through absolute imports and the contracts defined here.
* No part of the API exposes the packed state integer, intron values, or transformation masks to external callers. Only the canonical public reporting functions (`get_state_info`, `get_agent_info`) are permitted to export state.

**Regulatory and Recovery Guarantees**

* The algedonic regulation and autonomic cycles contracts (`post_cycle_hooks`, `run_autonomic_cycle`) are required to be enforced after every cycle. This ensures that the agent cannot be driven into runaway instability or pathological states.
* Agent overlay storage (`OrbitStore`, `CanonicalView`, `OverlayView`) guarantees isolation and canonicalisation, with configuration autodection provided in `_create_default_store`.

**Extensibility and Maintenance**

* All external monitoring, introspection, and maintenance operations must be registered via `add_hook` at the intelligence layer.
* All storage, canonicalisation, and multi-agent overlay mechanisms are constructed at initialisation and cannot be bypassed by runtime code.

---

## 6.5 Shared Contracts and Storage Policies

This section establishes the explicit type contracts, interface protocols, and canonical storage primitives for all orchestration, policy, and maintenance operations in S4 (Intelligence) and S5 (Policy/Identity). All interface points are enforced as concrete class or function boundaries, with no “informal” API surfaces.

### 6.5.1 Contracts: Protocols and Shared Types

All type, storage, and hook contracts are declared in `baby.contracts`:

* **PhenotypeEntry (TypedDict)**:
  The atomic knowledge unit for all agent stores.
  Required and optional fields (see `PhenotypeEntry` in code):

  * `phenotype: str`
  * `confidence: float`
  * `memory_mask: int`
  * `usage_count: int`
  * `age_counter: int`
  * `last_updated: float`
  * `created_at: float`
  * `semantic_address: int`
  * `context_signature: Tuple[int, int]`
  * `_original_context: Optional[Tuple[int, int]]` (tracking non-canonical context for canonicalised views)

* **ManifoldData (TypedDict)**:
  Ontology/graph structure of the physical state space.

  * `schema_version: str`
  * `ontology_map: Dict[int, int]`
  * `endogenous_modulus: int`
  * `ontology_diameter: int`
  * `total_states: int`
  * `build_timestamp: float`

* **PhenomenologyData (TypedDict)**:
  Canonical orbit mapping and diagnostic metadata.

  * `schema_version: str`
  * `phenomenology_map: list`
  * `orbit_sizes: dict`
  * `metadata: dict`
  * `_diagnostics: Dict[str, Any]`

* **AgentConfig (TypedDict)**:
  All configuration for a single GyroSI agent.
  Keys include:

  * `ontology_path: str`
  * `knowledge_path: Optional[str]`
  * `public_knowledge_path: Optional[str]`
  * `private_knowledge_path: Optional[str]`
  * `agent_metadata: Optional[Dict[str, Any]]`
  * `max_memory_mb: Optional[int]`
  * `enable_phenomenology_storage: Optional[bool]`
  * `batch_size: Optional[int]`
  * `phenomenology_map_path: Optional[str]`

* **PreferencesConfig (TypedDict)**:
  System-wide or pool-level storage, agent, and maintenance policies.
  All keys as in code, e.g.:

  * `storage_backend: str`
  * `compression_level: int`
  * `max_file_size_mb: int`
  * `enable_auto_decay: bool`
  * `decay_interval_hours: float`
  * `decay_factor: float`
  * `confidence_threshold: float`
  * `max_agents_in_memory: int`
  * `agent_eviction_policy: str`
  * `agent_ttl_minutes: int`
  * `enable_profiling: bool`
  * `batch_size: int`
  * `cache_size_mb: int`

* **ValidationReport (TypedDict)**:
  Structured validation results, e.g. from store or ontology checks.

  * `total_entries: int`
  * `average_confidence: float`
  * `store_type: str`
  * `modified_entries: Optional[int]`

* **CycleHookFunction (Protocol)**:
  Interface for post-cycle hooks:

  ```
  def __call__(engine, phenotype_entry, last_intron) -> None
  ```

* **MaintenanceReport (TypedDict)**:
  Results from any maintenance, merge, or decay operation.

  * `operation: str`
  * `success: bool`
  * `entries_processed: int`
  * `entries_modified: int`
  * `elapsed_seconds: float`

All contracts above are imported by absolute path, and serve as the sole interface for communication between agents, stores, overlays, and maintenance utilities.

---

### 6.5.2 Storage and Policy Layer

The canonical store for all phenotype knowledge is the `OrbitStore` class (`baby.policies.OrbitStore`):

* **OrbitStore**
  Atomic, file-based storage for `PhenotypeEntry` values, keyed by `(tensor_index, intron)` tuples (the canonical context signature).
  Contracts:

  * `get(context_key: Tuple[int, int]) -> Optional[Any]`
  * `put(context_key: Tuple[int, int], entry: Any) -> None`
  * `close() -> None`
  * `data -> Dict[Tuple[int, int], Any]`
  * `iter_entries() -> Iterator[Tuple[Tuple[int, int], Any]]`
    Additional methods:
  * `commit()`, `delete(context_key)`, `set_data_dict()`

* **CanonicalView**
  Decorator that enforces canonicalisation on all storage operations, using a phenomenology map loaded at initialisation.

  * Canonicalises the first element of the context key (`tensor_index`) for every get/put.
  * Writes also record `_original_context` for provenance.
  * Used in multi-agent overlays and when “phenomenology storage” is enabled.

* **OverlayView**
  Composite overlay supporting public/private or shared knowledge:

  * All writes go to the private overlay store.
  * Reads check the private overlay first, then fall back to the public store.
  * Both underlying stores must conform to the OrbitStore interface.

* **ReadOnlyView**
  Wraps any store, exposing only the read interface.
  All writes raise errors; used for immutable public knowledge overlays.

---

### 6.5.3 Maintenance and Policy Utilities

All policy functions operate on these storage contracts, use the canonical interfaces, and return a `MaintenanceReport`. Key functions (as implemented):

* **merge\_phenotype\_maps(source\_paths, dest\_path, conflict\_resolution) -> MaintenanceReport**
  Merges multiple OrbitStore or compatible knowledge files into one, resolving conflicts by confidence, recency, mask union, or weighted average.

* **apply\_global\_confidence\_decay(store\_path, decay\_factor, age\_threshold, time\_threshold\_days, dry\_run) -> MaintenanceReport**
  Applies exponential confidence decay to all entries meeting the decay threshold, following the decay formula used by the agent engine. Supports dry-run for auditing.

* **export\_knowledge\_statistics(store\_path, output\_path) -> MaintenanceReport**
  Calculates and exports statistics for a given store: entry counts, confidence metrics, memory utilisation, temporal statistics, phenotype diversity.

* **validate\_ontology\_integrity(ontology\_path, phenomenology\_map\_path) -> MaintenanceReport**
  Validates that ontology and phenomenology files have the correct structure, required fields, and expected graph size (modulus, diameter, etc).

All store, overlay, and decorator classes above support at least the standard interface:

* `get`, `put`, `close`, `data`, `iter_entries`

All stores ensure atomicity and isolation for read/write. All overlays and decorators transparently compose these policies for complex agent or pool-level storage needs.

All contracts, types, and policies in this section are definitive for any agent, orchestration, or system maintenance code in GyroSI. No alternative or informal interfaces are supported or documented.

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
│   ├── baby_preferences.json
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
│   ├── memory_preferences.json
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

The `memories/` directory contains the system’s persistent state.

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

GyroSI’s integration model is compositional. All agent orchestration and interaction is implemented by composing the canonical primitives provided in `baby.intelligence`, `baby.contracts`, and `baby.policies`.

**Agent Pool Management:**
Applications manage a pool of active agents with automatic eviction, overlay storage, and policy control. The pool ensures clean lifecycle and concurrency discipline for all agents.

```python
from baby.intelligence import AgentPool, orchestrate_turn

# Example pool instantiation
pool = AgentPool(
    ontology_path="memories/public/meta/ontology_map.json",
    base_knowledge_path="memories/public/meta/knowledge.pkl.gz"
)
```

### 8.2 Conversation Orchestration

Conversations are managed by composing agent interactions using the stable GyroSI API. No special conversation-specific infrastructure is required.

```python
def orchestrate_turn(pool: AgentPool, user_id: str, assistant_id: str, user_input: str) -> str:
    user_agent = pool.get_or_create_agent(user_id, role_hint="user")
    assistant_agent = pool.get_or_create_agent(assistant_id, role_hint="assistant")
    stimulus = user_agent.respond(user_input.encode("utf-8"))
    response = assistant_agent.respond(stimulus)
    try:
        return response.decode("utf-8")
    except UnicodeDecodeError:
        return response.decode("utf-8", errors="replace")
```

### 8.3 Protocol Adapters

External protocols are integrated through thin adapters that map messages to agent API calls.

**Example: OpenAI-Compatible Adapter**

```python
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    user_id = request.headers.get("X-User-ID", f"anon-{hash(request.client.host)}")
    assistant_id = "shared-assistant-v1"
    assistant = pool.get_or_create_agent(assistant_id)

    system_message = next((m for m in request.messages if m.role == "system"), None)
    if system_message and assistant.engine.cycle_count == 0:
        assistant.ingest(system_message.content.encode("utf-8"))

    assistant_messages = [m.content for m in request.messages if m.role == "assistant"]
    if assistant_messages:
        assistant.ingest("\n".join(assistant_messages).encode("utf-8"))

    last_user_message = next((m for m in reversed(request.messages) if m.role == "user"), None)
    final_response = ""
    if last_user_message:
        final_response = orchestrate_turn(pool, user_id, assistant_id, last_user_message.content)

    return format_openai_response(final_response)
```

### 8.4 Multi-Pattern Support

This approach supports multi-tenant, multi-user, networked, and hierarchical agent topologies through policy and orchestration only. The physics and engine logic remain strictly invariant.

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

* **Egress (process\_egress):**
  With the STT present, state transition is a single array lookup (`ep[current_index, intron]`). Without the STT, the transformation is performed by a fixed sequence of bitwise operations. In both cases, time complexity per step is constant (`O(1)`).

* **Ingress (process\_ingress):**
  Phenotype retrieval and update are performed through a single Python dict lookup and update. Learning (coaddition) is a path-dependent bitwise operation. All steps are constant time (`O(1)`).

* **Batch operations:**
  Batch learning reduces to a single-pass scan with one accumulator (`O(N)` for N input bytes).

There are no components whose computational cost scales faster than linearly with the volume of input data.

### 9.2 Memory Requirements

* **epistemology.npy (STT):**
  770 MB on disk. The file is memory-mapped and shared across agents; actual resident set stabilises near 50–60 MB with typical access patterns.

* **ontology\_map.json:**
  20 MB on disk. In default memmap mode, three NumPy arrays are constructed (keys, values, inverse), collectively occupying \~15 MB RAM per process.

* **phenomenology\_map.json:**
  The core artifact is approximately 15-20 MB on disk. It contains the primary canonical map (a list of 788,986 integers) and metadata. If generated with the optional diagnostic layer, the file size increases. When loaded, the primary map occupies approximately 3 MB of memory as a NumPy integer array.


* **OrbitStore index:**
  25 bytes per stored phenotype. For 100,000 phenotypes: \~2.5 MB.

* **Agent core state:**
  Each agent requires <1 MB for core state (counters, identifiers, in-memory hooks).

* **Python interpreter, modules, code:**
  Baseline memory footprint for the runtime environment is 8–10 MB per process.

**Scalability:**

* **Per-agent incremental memory:**
  Dominated by the OrbitStore index (linear in number of learned phenotypes).

* **Shared memory:**
  STT and ontology artefacts are loaded once per host and shared via memory mapping. Ten agents operating concurrently require less than 100 MB additional memory, beyond their private indices.

* **I/O:**
  The system performs write-behind batching; one fsync per 100 writes (default), amortising I/O load.

### 9.3 Throughput

**Single-threaded (Intel i7-1260P, 3.4 GHz):**

* A full egress–ingress cycle (STT-backed) completes in 0.8–1.2 microseconds.
* Sustained throughput: \~0.9 million cycles per second (with phenotype store index in L3 cache).
* Latency remains flat until the index exceeds CPU cache capacity, at which point misses increase per-cycle time (up to \~10 microseconds at 5 million phenotypes).

**Multi-agent, multi-core (AMD EPYC, 32-core):**

* 32 agents in parallel sustain 25–28 million cycles per second, constrained by memory bandwidth rather than CPU.
* Scaling with additional agents or cores is sublinear once shared caches are saturated; NUMA-local shards restore most of the theoretical throughput.

**Disk throughput:**

* The OrbitStore append log saturates at 150 MB/s on NVMe SSD.
* Increasing the write batch threshold (e.g., to 1,000) increases sustained ingestion rates for write-heavy workloads.

### Additional Notes

* **Startup time** (with STT): dominated by memory-mapping the 770 MB file (60 ms) and ontology array parsing (20 ms).
* **Garbage collection:** negligible impact in core paths; no objects are allocated in egress/ingress inner loops.
* **Fallback mode** (no STT): throughput is halved but remains acceptable for memory-constrained environments.
* **GPU acceleration:** ineffective due to memory bandwidth bottleneck and low arithmetic intensity; all critical paths are vectorisable but not compute-limited.
* **Security/multi-tenancy:** The STT file can be mounted read-only and is safe to share between containers or processes.

**Summary:**
On modern workstation hardware, dozens of GyroSI agents may be operated concurrently, with both latency and memory use remaining well within interactive bounds. All core operations are constant-time and embarrassingly parallel, with no algorithmic scaling bottlenecks.

---

### 9.4  Pragmatic capacity & device‑level throughput (revised)

#### 1 How many facts fit?

`OrbitStore` keeps one `(state, context) → offset` mapping per fact.
On disk the entry serialises to \~25 B; in memory the Python objects occupy roughly 3–4× that.
The table below therefore assumes **90 B per fact (phenotype entry)** resident.

| Device (free RAM for GyroSI)             | Facts held in RAM |
| ---------------------------------------- | ----------------- |
| **Arduino Uno (16 KB)**                  | \~180 (demo only) |
| **MacBook Air 2015, 8 GB → ≈ 4 GB free** | **≈ 45 million**  |
| **MacBook M4, 16 GB → ≈ 12 GB free**     | **≈ 130 million** |
| **Server, 256 GB → ≈ 220 GB free**       | **≈ 2.4 billion** |

A modern laptop therefore keeps the entirety of WordNet (≈ 150 k facts (phenotype entries)) and the English Wikipedia title & abstract graph (≈ 40 M facts (phenotype entries)) comfortably in RAM.

#### 2 Throughput you can picture

A *cycle* = read one byte → internal update → emit one byte.
With the state‑transition table (STT) memory‑mapped this bottlenecks on pure Python overhead.

| Hardware                             | Cores used | Cycles per second | Characters per second (≈ cycles) |
| ------------------------------------ | ---------- | ----------------- | -------------------------------- |
| **MacBook Air 2015** (2 physical)    | 1          | \~0.7 M           | \~0.7 M                          |
|                                      | 2          | \~1.4 M           | \~1.4 M                          |
| **MacBook M4** (8 performance cores) | 8          | \~7–8 M           | \~7–8 M                          |
| **EPYC 32‑core server**              | 32         | \~25 M            | \~25 M                           |

Rounded rule‑of‑thumb: **\~1 million characters · s⁻¹ · core** on 2024‑era silicon, about one‑third of that on a 2015 dual‑core laptop.

#### 3 How long to ingest familiar corpora?

Assuming 1 char ≈ 1 byte, and using the per‑core rate above:

| Corpus                          | Size    | 1 core      | 8 cores    | 32 cores  |
| ------------------------------- | ------- | ----------- | ---------- | --------- |
| WordNet glosses                 | 30 MB   | < 1 min     | “blink”    | “blink”   |
| English Wikipedia\*             | 90 GB   | \~1 day     | \~3 h      | **< 1 h** |
| Filtered public‑web dump (1 PB) | 10^6 GB | \~3.5 years | \~5 months | \~5 weeks |

\* plain‑text revision of the 2025 EN wiki dump.

#### 4 Context length in day‑to‑day terms

GyroSI has no fixed token window.
What matters is how many distinct `(state, context)` pairs the index can keep:

* 8 GB laptop → **tens of millions** of separate contexts.
* Look‑up latency stays < 2 µs so long as the active slice fits in last‑level cache (≈ 10 M contexts on current hardware).

#### 5 Edge devices

* **Arduino‑class MCUs**: no room for the 770 MB STT, fall back to bit‑wise physics; expect hundreds of cycles · s⁻¹.
* **Raspberry Pi 5 (8 GB)**: maps the STT, reaches \~400 k cycles · s⁻¹; fine for home‑lab projects with tens of millions of contexts.

#### 6 Write load

One flush every 100 updates appends ≤ 3 KB; a laptop continuously learning Wikipedia writes **< 5 MB min⁻¹**, far under SSD endurance limits.

---

### Appendix – Theoretical Correspondences

This appendix records the essential bridges between GyroSI’s formal physics and several established conceptual frames. It is deliberately brief: anything already explained in the main text is only referenced here.

---

Below is a precise, up-to-date wording for the “Genetic-code analogies” appendix that keeps the 64 / 32 narrative, clarifies how those counts relate to the newly-measured 256 operational orbits, and avoids hand-waving numbers such as “~195 000” unless you explicitly include the diagnostic layer.

---

### 1  Genetics

GyroSI’s instruction algebra in the eight-bit space ℤ₂⁸ echoes a ladder of small cardinalities familiar from molecular genetics.  No claim is made that the two domains share the same state count or evolutionary origin; the correspondence is purely structural.

| GyroSI level | Cardinality | Biology analogy | Comment |
|--------------|-------------|-----------------|---------|
| **Intron action families** | **4** (`L0`, `LI`, `FG`, `BG`) | Four nucleotides | Minimal set that closes the algebra. |
| **Tensor axes per slice** | **3** | Three codon positions | Each axis modulates a different DoF. |
| **Sign polarities** | **2** ( ±1 ) | Two DNA strands | Fundamental chirality. |
| **Bit width of intron** | **8 bits** → 2⁸ = 256 patterns | 2 bits × 4-symbol alphabet | Full instruction space. |
| **Active masks (anchors stripped)** | **64** patterns | 64 mRNA codons | Bits 0 & 7 (`L0`) are anchors; removing them leaves 6 informative bits (2⁶ = 64). |
| **Parity-quotiented classes** | **32** | Wobble pairing halves the codon set | Identifying global parity (`LI` bits 1 & 6) folds the 64 active masks into 32 equivalence classes. |
| **Operational phenomenology** | **256** orbits | — | Learning paths under the non-associative fold operator divide the space into 256 phenomenological orbits. |

L0 (anchor bits 0 & 7): structural identity anchors (do not alter state).
LI (global parity bits 1 & 6): implement chirality reflection (dual confinement).
FG (foreground bit group): subset controlling sign inversions on alternating tensor cells (local orientation flips).
BG (background bit group): complementary subset controlling the interleaving layer polarity.
The four families partition the eight intron bits into functional roles that jointly close the transformation algebra with minimal redundancy.

Key points:

*  **Anchors vs. Physics:**  
  Bits 0 and 7 (`L0`) never alter the state; they act only as identity anchors.  Stripping them reveals the **64** distinct “active” masks that actually drive transformations.

*  **Global parity (`LI`) as confinement:**  
  The `LI` bits (1 & 6) implement the universal chirality flip.  Quotienting by this symmetry folds the 64 masks into **32** classes—precisely mirroring how wobble pairing halves the biological codon table.

*  **Why 256 operational orbits, not 64 / 32:**  
  Once the state-dependent carry term is included, each nominal intron mask can steer the system along many memory-laden paths.  The full SCC analysis shows that these paths organise the 788 986 states into **256** parity-closed phenomenological orbits—one basin of mutual reachability for each eight-bit instruction pattern.  Thus the 64/32 counts live at the **mask-algebra** layer, while 256 lives at the **full physics + memory** layer.  Both ladders are valid; they simply sit at different depths of the model.

The large orbit count (788 986) and its measured diameter (6) belong purely to GyroSI’s internal physics and have no biological counterpart; the analogy stops at the level of small-number structures.

---

#### “~195 000 parity-free orbits”

If you include the optional diagnostic SCC pass that **excludes `LI` introns**, the canonical 256 orbits split into 194 698 finer “parity-free” islands.  About 21 456 of those appear in chiral mirror pairs; the rest (≈ 152 k) are achiral self-mirrors.  This finer decomposition is **research metadata only** and has been moved to the `_diagnostics` section of the phenomenology artifact; it is not part of the operational ladder above.

---

#### 2. The structural number ladder

GyroSI’s constants are locked by algebraic closure, not convenience:

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

---

#### 3. Gyrogroup algebra as implemented

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

---

#### 4. Holographic principle in practice

A single eight‑bit intron always touches the entire forty‑eight‑bit state through four complementary twelve‑bit masks. Conversely, any state can be reached in at most six introns. This bidirectional property embodies the holographic claim that every part encodes the whole. The code paths involved are `transcribe_byte`, `apply_gyration_and_transform`, and the breadth‑first discovery routine that proves the six‑step closure.

---

#### 5. Stabiliser and modulus

Breadth‑first exploration over the full instruction set discovers exactly 788 986 distinct states and a diameter of six. The stabiliser of the archetype has order two (global parity) multiplied by eleven (frame degeneracy). The remaining factor, 35 863, is prime, confirming that no further quotient is possible. These facts are verified at build time and are used to reject any physics violation at run time.

Frame degeneracy (11) counts the distinct layer/frame symmetry operations (excluding global parity) that leave the archetypal tensor invariant under the applied transformation group; combined with parity (×2) and the residual prime (35,863) they factor the full state modulus.

No biological code shows the same modulus; the coincidence stops at the smaller sub‑structures outlined above.

---

#### 6. Further correspondences

Other mappings noted in the main text are retained without restatement:

* The angular sequence π/2, π/4, π/4, 0 for CS → UNA → ONA → BU.
* The packed‑integer versus tensor dual representation.
* The role of the endogenous modulus as a hard physical constant.

Readers seeking proofs or implementation details will find the relevant functions in `baby.governance`, `baby.information`, and `baby.inference`.

#### 7. Core Invariants (Build‑Time Assertions)

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

===
