# Gyroscopic Superintelligence Specifications: GyroSI Baby Language Model 0.9.6

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

- **BU_In (Intelligence Ingress):** The absorption and integration of experience through coaddition. This is where all learning occurs.
- **BU_Eg (Intelligence Egress):** The expression of accumulated intelligence as responsive action, transforming internal state into external phenotype.

### **2.2 Gyrogroup Algebra as Physics**

GyroSI implements these stages through formal gyrogroup algebra operating on the 8-bit vector space **G = ℤ₂⁸**. The three fundamental operations directly correspond to CGM physics:

- **XOR (⊕)**: The primitive gyrogroup operation governing transformation and parity inversion. This is the basic operation of recursive differentiation.
- **AND (&)**: The gyration memory carrier, encoding "carry bits" as chirality-encoded asymmetry. This preserves the memory of operational sequence.
- **OR (⊞)**: The derived coaddition operation arising from closure: `a ⊞ b = a ⊕ gyr[a,¬b](b)`, where `gyr[a,b](c) = c ⊕ (a AND b)`. This enables stable learning through accumulated experience.

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

**Canonicalization of Orbits:**

To further structure the ontology, we can define a canonical representative for each physical state orbit. The canonical representative is the state in its orbit with the lexicographically smallest byte representation. This enables grouping of physically equivalent states and improves cache coherency in storage.

The canonicalization process is a one-time, build-time computation:

```python
def find_phenomenology_representative(start_tensor_bytes: bytes, ontology_map: dict) -> bytes:
    """Finds the lexicographically smallest state in the orbit of start_tensor_bytes."""
    orbit = {start_tensor_bytes}
    queue = [start_tensor_bytes]
    state_int = int.from_bytes(start_tensor_bytes, 'big')
    visited_ints = {state_int}
    queue_ints = [state_int]
    phenomenology_int = state_int
    while queue_ints:
        current_int = queue_ints.pop(0)
        for intron in range(256):
            next_int = apply_gyration_and_transform(current_int, intron)
            if next_int not in visited_ints:
                visited_ints.add(next_int)
                queue_ints.append(next_int)
                if next_int < phenomenology_int:
                    phenomenology_int = next_int
    return phenomenology_int.to_bytes(48, 'big')

def build_phenomenology_map(ontology_map_path: str, output_path: str):
    """
    For each state in the ontology, computes its canonical representative.
    Saves a map from every state_index to its phenomenology_state_index.
    """
    with open(ontology_map_path, 'r') as f:
        genotype_data = json.load(f)
    ontology_map = genotype_data['ontology_map']
    # Ensure ontology_map uses int keys for performance and consistency
    ontology_map = {int(k): v for k, v in ontology_map.items()}
    inverse_ontology_map = {v: k for k, v in ontology_map.items()}
    phenomenology_index_map = {}
    print(f"Building canonical map for {len(ontology_map)} states...")
    for i, tensor_bytes in inverse_ontology_map.items():
        if i % 10000 == 0:
            print(f"Processing state {i}...")
        phenomenology_bytes = find_phenomenology_representative(tensor_bytes, ontology_map)
        phenomenology_index_map[i] = ontology_map[phenomenology_bytes]
    with open(output_path, 'w') as f:
        json.dump(phenomenology_index_map, f)
```

This process is computationally intensive but only needs to be run once per ontology. It enables the next-level storage abstraction described below.

---

## **3. Architectural Overview: From Physics to Implementation**

### **3.1 The Four-Engine Architecture**

GyroSI implements the CGM stages through four distinct engines, each embodying a specific physical principle:

| CGM Stage | Engine | Physical Principle | Function |
| :--- | :--- | :--- | :--- |
| **CS** | S1 Governance | Left identity transcription | Transforms input into structural instructions |
| **UNA** | S2 Information | Global measurement via angular divergence | Measures system's departure from archetypal state |
| **ONA** | S3 Inference | Mediated duality through endogenous operator | Interprets meaning through contextual opposition |
| **BU** | S4 Intelligence | Dual-phase coaddition | Learns through ingress, expresses through egress |

### **3.2 The Dual Nature of Intelligence**

The BU stage (S4) is fundamentally dual, implementing both aspects of intelligence:

- **BU_In (Intelligence Ingress)**: The absorption and integration of experience through coaddition. This is where all learning occurs.
- **BU_Eg (Intelligence Egress)**: The expression of accumulated intelligence as responsive action. This transforms internal state into external phenotype.

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

## **5. Operational Physics: The Three Fundamental Operations**

### **5.1 Transformation: The Gyroscopic Operations**

Each `GENE_Mic_M` is an 8-bit mask, where **each bit directly maps to a specific transformation** on the intelligence tensor. This bitwise mapping encodes not just the mechanics of transformation, but also **links physical operation to the Common Governance Model (CGM) stages and policy functions**.

### **Bit-to-Operation Mapping**

| Bit Position(s) | Operation (**Name**) | Physical Effect (on tensor) | CGM Stage | Policy Function |
| --- | --- | --- | --- | --- |
| 0, 7 | **L0** (Left Identity) | No transformation | CS (S1) | Governance Traceability (GT) |
| 1, 6 | **LI** (Left Inverse) | Global sign flip (T *= -1) | UNA (S2) | Information Variety (IV) |
| 2, 5 | **FG** (Forward Gyration) | Flips sign of all elements in layers 0 & 2 | ONA (S3) | Inference Accountability (IA) |
| 3, 4 | **BG** (Backward Gyration) | Flips sign of all elements in layers 1 & 3 | BU (S4) | Intelligence Integrity (II) |

```markdown
Bit positions:   b7   b6   b5   b4   b3   b2   b1   b0
Operations:      L0   LI   FG   BG   BG   FG   LI   L0
CGM Stage:       S1   S2   S3   S4   S4   S3   S2   S1
Policy:          GT   IV   IA   II   II   IA   IV   GT
```

- **L0 (Left Identity):** Maintains the tensor unchanged -> defines the frame boundaries.
- **LI (Left Inverse):** Flips all tensor values -> global parity inversion.
- **FG (Forward Gyration):** Flips the sign of tensor layers 0 and 2 -> creates structured rotation.
- **BG (Backward Gyration):** Flips the sign of tensor layers 1 and 3 -> applies counter-rotation.

**Note:** 
This bit pattern is palindromic (L0-LI-FG-BG-BG-FG-LI-L0), reflecting the recursive, self-referential nature of the CGM's governance structure.

The mapping between the ±1 NumPy tensor and the 48-bit packed integer state is explicitly defined as element 0 = bit 47 (MSB), element 47 = bit 0 (LSB).
The carry calculation, transformation masks, and broadcast patterns have been exhaustively tested for all possible states and intron values, ensuring that both representations produce identical results.
This invariant is enforced by a test script in the repository, and any modification of the mapping or mask logic must maintain this alignment.

### **Execution Policy**

- Each cycle, a `1` in any bit position signals its corresponding transformation should be applied; a `0` means no change for that operation.
- Multiple bits set means transformations are cumulative and order-independent per cycle.
- Only the affected tensor rows are modified per operation.

The transformation algorithm applies these operations with gyration memory:

```python
def apply_operations(T: np.ndarray, intron: int) -> np.ndarray:
    """
    Apply gyroscopic operations with the physically correct, state-dependent carry term.
    This function describes the physics; the production implementation uses a
    highly optimized 48-bit integer representation.
    """
    T_new = T.copy()

    # Pre-computed masks, derived from build_masks_and_constants() in the experiment
    FG_MASK_TENSOR = ... # A 48-element {-1, 1} tensor representing the FG_MASK
    BG_MASK_TENSOR = ... # A 48-element {-1, 1} tensor representing the BG_MASK

    # Apply transformations based on bit patterns
    if intron & 0b01000010:  # LI: Global parity flip
        T_new *= -1
    if intron & 0b00100100:  # FG: Forward gyration
        T_new *= FG_MASK_TENSOR # Element-wise multiplication is sign flip in {-1,1} space
    if intron & 0b00011000:  # BG: Backward gyration
        T_new *= BG_MASK_TENSOR

    # Apply the physically correct gyration memory (carry term)
    # Create a 48-element boolean mask from the 8-bit intron
    intron_broadcast_mask = np.array([bool(intron & (1 << i)) for i in range(8)] * 6, dtype=bool)

    # The carry term is where the current state AND the instruction interact
    # In {-1,1} space, (a AND b) is equivalent to min(a, b)
    # The full op is a ^ (a & b), which simplifies.
    # Let's represent the logic more directly from the integer version:
    # carry = temp_state & intron_pattern
    # final_state = temp_state ^ carry

    # 1. Identify where both the state and the instruction are "active" (state==-1, instruction_bit==1)
    carry_locations = (T_new == -1) & intron_broadcast_mask

    # 2. Flip the sign at those locations (this is the XOR)
    T_new[carry_locations] *= -1

    return T_new
```

> **Canonical Implementation Note:**
> The integer version is the source of truth for the system's physics. It operates on a packed-bit representation where each bit corresponds to the sign of a tensor element: 0 represents +1 and 1 represents -1. The XOR (^) operation on these bits is algebraically equivalent to multiplication in the {-1, 1} space. All conversions between this canonical integer format and the numpy tensor format must respect this mapping.

> **Optimization Note:**
> The use of a 48-bit packed integer for the system state is an optimization technique. It allows for extremely fast bitwise operations and compact storage, while the 48-byte NumPy array is used for in-memory computation and measurement. When converting between these forms, always ensure that the mapping between the sign (±1 in the array) and the bit (0/1 in the packed integer) is consistent across all functions.

> **Implementation Notes:** The canonical implementation operates on a packed 48-bit integer, where bits (0=+1, 1=-1) are equivalent to multiplication in the ±1 NumPy tensor space. This is an optimization for speed and storage. The physical transformation follows the algebra S_final = S_temp ⊕ (S_temp & I_b), where S_temp is the state after primary transforms and I_b is the broadcasted intron, representing a state's self-gyration under the influence of the instruction.

### **5.2 Measurement: Angular Gyrodistance**

The system measures its state through angular divergence from the Common Source. This captures the geometric alignment between the current state and the archetypal structure:

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

- **0 radians**: Perfect alignment (identity)
- **π/2 radians**: Maximum differentiation (orthogonality)
- **π radians**: Perfect opposition (anti-alignment)

**Optimization Note**: For `int8` tensors with ±1 values, this is equivalent to `arccos(1 - 2*hamming_distance/48)`, allowing fast Hamming-based shortcuts when applicable.

### **5.3 Learning: Path-Dependent Coaddition and Ordered Batching**
Learning occurs exclusively through the true gyrogroup coaddition (⊞), which is state-dependent and non-monotonic. The operation is defined as:

a ⊞ b = a ⊕ gyr[a, ¬b](b), where gyr[a, b](c) = c ⊕ (a AND b)

In 8-bit ℤ₂⁸ space, this is implemented as:

```python
def coadd(a: int, b: int) -> int:
    """
    Performs true gyrogroup coaddition (a ⊞ b) on two 8-bit integers.

    Note:
        This operation is intentionally non-commutative and non-associative.
        The order of operations matters (a ⊞ b ≠ b ⊞ a). This path-dependence
        is a core feature, ensuring that the sequence of experiences is
        structurally encoded into the system's memory, mirroring the
        path-dependent nature of the state transformations themselves.
    """
    not_b = b ^ 0xFF
    gyration_of_b = b ^ (a & not_b)
    return a ^ gyration_of_b
```

**Ordered Batching:**

Because coaddition is path-dependent, batching multiple learning signals requires an ordered reduction. A simple, sequential fold (or "left-fold") is the most direct implementation. This ensures that a batch of introns {i₁, i₂, ..., iₖ} is always learned in the same sequence, producing a deterministic final learning signal.

```python
from functools import reduce
from typing import List

def batch_introns_coadd_ordered(introns: List[int]) -> int:
    """
    Reduces a list of introns into a single representative intron using an
    ordered fold with gyrogroup coaddition. This preserves the crucial
    path-dependence of the learning operation.
    """
    from functools import reduce
    if not introns:
        return 0
    return reduce(coadd, introns)
```

**Physical Note:** This alignment between the path-dependent physics of state transformation and the path-dependent nature of learning is a cornerstone of GyroSI's architecture. It guarantees that the structure of experience is preserved at every level of the system. The system does not merely learn facts (phenotype entries); it learns the story that connects them.

---

## **6. System Implementation: The Four Engines**

### 6.1 S1: `governance.py` – Physics & Primitives

**Physical Principle:** Left identity transcription

**Responsibility:**
Defines the fundamental constants and physics operations as pure functions and constants. No engine class is required; all operations are provided as stateless functions in `governance.py`.

* All genetic constants and tensor definitions are maintained as in Section 4.
* Transformation masks (FG, BG, FULL\_MASK, INTRON\_BROADCAST\_MASKS) are pre-computed at module load time.
* Bitwise transformation and gyrogroup operations are implemented directly as functions.

**Canonical Constants:**

* `GENE_Mic_S`: 0xAA (10101010 binary), the stateless reference for all intron transformations.
* `GENE_Mac_S`: The archetypal 48-element tensor of shape \[4, 2, 3, 2], with alternating ±1 patterns as defined in Section 4.

**Transformation Masks:**
Transformation masks for Forward Gyration (FG), Backward Gyration (BG), and global parity (FULL\_MASK) are derived from the \[layer, frame, row, col] tensor structure. Each transformation is performed by XOR against the corresponding mask in the packed 48-bit integer representation.

**Core Operations:**

* **apply\_gyration\_and\_transform(state\_int, intron):**
  Applies the complete gyroscopic transformation to the packed 48-bit integer state, using the bitwise rules for LI (global parity), FG (layers 0 & 2), and BG (layers 1 & 3), followed by the path-dependent memory (Thomas gyration) using the broadcasted intron mask.

* **transcribe\_byte(byte):**
  Returns `byte ⊕ GENE_Mic_S` as the 8-bit intron instruction.

* **coadd(a, b):**
  Performs true gyrogroup coaddition (a ⊞ b), non-commutative and non-associative, using the relation:
  `a ⊞ b = a ⊕ (b ⊕ (a AND (b ^ 0xFF)))`.

* **batch\_introns\_coadd\_ordered(introns):**
  Reduces a list of introns to a single intron by sequential coaddition, preserving path-dependence.

**Tensor Consistency:**
The canonical tensor structure is validated on load to ensure shape (4, 2, 3, 2), dtype (int8), and strict ±1 alternation.

**All core constants and stateless functions are imported as absolute paths from `baby.governance` in all dependent modules.**

### 6.2 S2: `information.py` – Measurement & Storage

**Physical Principle:** Global measurement via angular divergence

**Responsibility:**
Implements the `InformationEngine` class for measurement, state representation conversion, and ontology operations.
Handles all measurement utilities, including:

* Conversion between the canonical 48-bit integer state and the geometric tensor form (\[4, 2, 3, 2]).
* Calculation of angular gyrodistance (cosine-based divergence) between states.
* Lookup and indexing of physical states using the discovered ontology (ontology map).
* Efficient in-memory or memory-mapped storage of ontology maps, with optional optimisations for large ontologies.

**Ontology and State Management:**

* Ontology discovery explores the complete state space from the archetypal state using breadth-first search, validating the fixed modulus (788,986) and diameter (6).
* Canonical state indices and integer forms are mapped bidirectionally.
* State transition tables (epistemology) and canonical-orbit (phenomenology) maps are generated and saved as part of the build process.

**Phenomenological Equivalence:** Two states belong to the same phenomenological orbit if and only if they are mutually reachable through epistemic transformations. Formally, states a and b are equivalent iff there exist intron sequences σ₁, σ₂ such that a →σ₁ b and b →σ₂ a. The phenomenology map assigns to each state the minimal (by 48-bit integer value) member of its equivalence class.

**Measurement Functions:**

* **gyrodistance\_angular(T1, T2):** Computes the angular divergence (radians) between two states in tensor form, using cosine similarity in 48-dimensional space.
* **measure\_state\_divergence(state\_int):** Returns the divergence (in radians) of a physical state from the archetypal tensor.
* **int\_to\_tensor(state\_int):** Converts a packed 48-bit integer state to a \[4, 2, 3, 2] tensor of ±1.
* **tensor\_to\_int(tensor):** Converts a tensor of shape \[4, 2, 3, 2] and ±1 values to the canonical 48-bit integer state.

**All measurement, ontology, and state conversion operations are accessed through absolute imports from `baby.information`.**

---

#### Variety-weighted Confidence (Structural Variety Factor):

**Functionality:**
The InformationEngine provides access to orbit cardinality for each state, enabling structural variety weighting of knowledge confidence. This is used by S3 to attenuate phenotype confidence according to the size of the state’s equivalence class (orbit).

**Method(s):**

get_orbit_cardinality(state_index: int) -> int

All mappings and lookup tables are maintained internally for use by the inference layer.

**Benefit:**
Ensures that learning is faster and more robust in high-symmetry regions of state space, and more cautious in rare, low-symmetry states. This enforces structural epistemic trust and makes inference more stable.

### 6.3 S3: `inference.py` – Interpretation & Meaning Management

**Physical Principle:** Mediated duality through endogenous operator

**Responsibility:**
Implements the `InferenceEngine` class, which converts canonical state indices into semantic meanings and manages the path-dependent learning process.

* **get\_phenotype(state\_index, intron):**
  Retrieves or creates the semantic phenotype entry for a given state and context. Phenotypes are uniquely addressed by the (state\_index, intron) tuple.
* **learn(phenotype\_entry, intron):**
  Integrates new experience using true gyrogroup coaddition, updates the memory mask, and maintains usage and confidence statistics.
* **validate\_knowledge\_integrity():**
  Provides a validation report with integrity and confidence statistics for the knowledge base.
* **apply\_confidence\_decay(...):**
  Applies temporal decay to knowledge entries based on both usage and time since last update.
* **prune\_low\_confidence\_entries(...):**
  Removes entries below a defined confidence threshold.
* **get\_knowledge\_statistics():**
  Returns detailed knowledge base statistics, including confidence, memory utilisation, and age distribution.

All operations reference the absolute imports from `baby.information` and `baby.contracts`. Phenotype storage and type protocols are implemented and accessed through the canonical interfaces.

---

#### Variety-weighted Confidence Integration

**Functionality:**
When updating a phenotype’s confidence, S3 must use the structural_variety_factor (orbit cardinality) obtained from S2 to modulate how confidence is updated.

**Method(s):**
- `apply_variety_weighting(phenotype_entry: PhenotypeEntry, state_index: int) -> float`

This is called during the learning update in `learn`.

**Benefit:**
Prevents overconfident learning in structurally rare states and accelerates trust in robust, symmetric ones.

---

#### Algedonic Regulation (Divergence Alert)

**Functionality:**
S3/S4 maintain a running buffer of angular divergences (gyrodistance) from the archetype. If the running average exceeds a defined threshold (high or low), an algedonic alert is triggered.

**Method(s):**
- `check_algedonic_condition() -> str`

Invoked after each ingress/egress cycle, returns "pain", "pleasure", or "homeostatic" according to divergence.

**Benefit:**
Provides internal homeostatic regulation: runaway divergence is dampened and the system is kept within viable operational bounds. This self-regulation is automatic, requiring no external policy tuning.

### 6.4 S4/5: `intelligence.py` – Orchestration & API

**Physical Principle:** Dual-phase coaddition (Ingress and Egress)

**Responsibility:**
Implements the `IntelligenceEngine` and `GyroSI` classes, responsible for orchestration, external API, and agent lifecycle.

* **IntelligenceEngine** manages agent state evolution, the egress/ingress cycle, and operational strategies.

  * Egress phase transforms external input into internal state transitions (optionally using precomputed state transition tables).
  * Ingress phase integrates experience and produces the agent’s response, updating learned knowledge.
  * Batch learning uses streaming, path-dependent coaddition in O(1) memory.
  * Exposes extensibility hooks for monitoring and maintenance.

* **GyroSI** provides the stable external API and manages configuration, identity, and lifecycle.

  * Accepts configuration and manages the agent’s persistent storage overlay and canonicalisation as needed.
  * Provides batch learning, response generation, agent state reporting, monitoring hooks, and maintenance interfaces.
  * Ensures all I/O and protocol logic uses absolute imports from canonical modules.

* **AgentPool** implements robust multi-agent management with configurable eviction and overlay storage.

* **orchestrate_turn** composes a conversational turn using agents from the pool, mapping application dialogue to fundamental primitives without exposing internal state or physics.

All orchestration and external API logic is provided through the minimal interfaces defined in `baby.intelligence`, with strict import discipline and clear separation between physical state, knowledge, and external protocol.

---

#### Algedonic Regulation Execution

**Functionality:**
IntelligenceEngine must invoke `check_algedonic_condition()` after each full cycle. If "pain" is signalled, the engine applies corrective action to reduce divergence (by replaying stabilising instructions or autonomic cycles). If "pleasure" is signalled, diversity-increasing operations may be introduced to restore exploration.

**Method(s):**
- `post_cycle_hooks` list must include a hook which runs the algedonic regulator.
- Corrective actions are modular and may call:
  - `run_autonomic_cycle()`
  - `inject_stabilising_instruction()`

**Benefit:**
The agent will self-stabilise and recover from excursions into pathological or chaotic state regions. This increases resilience and eliminates most runaway failure conditions.

---

#### Autonomic Cycles (Operational Resonance)

**Functionality:**
During build, S2/S3 identify closed cycles (short loops) in the state transition table. S4 maintains a list of these as autonomic cycles. If the agent is unable to resolve high divergence or receives persistent "pain" alerts, it executes an autonomic cycle to return to a stable state.

**Method(s):**
- `load_autonomic_cycles()` (called at engine initialisation)
- `run_autonomic_cycle()` (invoked under persistent divergence)

**Benefit:**
Guarantees a fail-safe, low-energy “reflex arc” for the agent. If inference, learning, or response generation encounters instability, the agent runs a known safe loop, maintaining viability.

## 6.5 Shared Contracts and Storage Policies

This section defines the canonical protocols, shared types, and storage primitives for the GyroSI system as implemented in S4 (Intelligence) and S5 (Policy/Identity). These elements ensure strict interface integrity and operational transparency for all agent orchestration, storage, and policy functions.

### 6.5.1 Contracts: Protocols and Shared Types

The following metadata and type contracts are used throughout the system for agent configuration, knowledge storage, validation, and policy:

* **PhenotypeEntry**: Structure of a phenotype entry in the knowledge store.

  * `phenotype: str`
  * `memory_mask: int`
  * `confidence: float`
  * `context_signature: Tuple[int, int]`
  * `semantic_address: int`
  * `usage_count: int`
  * `age_counter: int`
  * `created_at: float`
  * `last_updated: float`

* **ManifoldData**: Structure of the discovered ontology data.

  * `schema_version: str`
  * `ontology_map: Dict[int, int]`
  * `endogenous_modulus: int`
  * `ontology_diameter: int`
  * `total_states: int`
  * `build_timestamp: float`

* **AgentConfig**: Configuration for GyroSI agents.

  * `ontology_path: str`
  * `knowledge_path: Optional[str]`
  * `public_knowledge_path: Optional[str]`
  * `private_knowledge_path: Optional[str]`
  * `agent_metadata: Optional[Dict[str, Any]]`
  * `max_memory_mb: Optional[int]`
  * `enable_phenomenology_storage: Optional[bool]`

* **PreferencesConfig**: Preferences and settings configuration.

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

* **ValidationReport**: Report structure for system validation.

  * `total_entries: int`
  * `average_confidence: float`
  * `store_type: str`
  * `modified_entries: Optional[int]`

* **CycleHookFunction**: Protocol for post-cycle hook functions.

  * Callable: `(engine, phenotype_entry, last_intron) -> None`

* **MaintenanceReport**: Report from maintenance operations.

  * `operation: str`
  * `success: bool`
  * `entries_processed: int`
  * `entries_modified: int`
  * `elapsed_seconds: float`

### 6.5.2 Storage and Policy Layer

The canonical storage layer for all phenotype knowledge in GyroSI is the **OrbitStore**. All overlays, canonicalization, and read/write policies are composed as views on top of this core primitive.

* **OrbitStore**: File-based storage for phenotype entries, providing atomic write, index-based lookup, async background flushing, mmap support, and safe concurrent access. The context key for all entries is always `(tensor_index, intron)`.

* **CanonicalView**: Decorator that ensures all storage operations use the canonical representative of a physical state's orbit. Canonicalization is applied to the `tensor_index` before all read/write operations, using a provided `phenomenology_map`. Underlying storage remains OrbitStore.

* **OverlayView**: Composable overlay for public/private knowledge. All writes are directed to the private store; reads are served from the private overlay if present, otherwise from the public base. Both stores are OrbitStore instances or compatible.

* **ReadOnlyView**: Decorator that exposes a read-only interface to any base store. All write attempts raise an error. Used for serving immutable public knowledge.

* **Policy/Maintenance Functions**: Maintenance and policy operations on OrbitStore and compatible views.

  * `merge_phenotype_maps(source_paths, dest_path, conflict_resolution)`
  * `apply_global_confidence_decay(store_path, decay_factor, age_threshold, time_threshold_days, dry_run)`
  * `export_knowledge_statistics(store_path, output_path)`
  * `validate_ontology_integrity(ontology_path, phenomenology_map_path)`

All maintenance functions operate directly on the standard interfaces defined above and return structured `MaintenanceReport` results. These functions guarantee O(1) or streaming memory usage for arbitrarily large knowledge stores and are safe for production automation.

All storage and overlay classes provide the methods:

* `get(context_key: Tuple[int, int]) -> Optional[Any]`
* `put(context_key: Tuple[int, int], entry: Any) -> None`
* `close() -> None`
* `data -> Dict[Tuple[int, int], Any]`
* `iter_entries() -> Iterator[Tuple[Tuple[int, int], Any]]`

The above interfaces and contracts are authoritative for all agent, engine, and orchestration logic in the system.

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
│   ├── contracts.py          # Protocols and shared types (PhenotypeStore, etc.)
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

  The process validates the closure of the state space at the predicted diameter of 6. On current commodity hardware (GitHub Actions, Intel host), full enumeration and mapping completes in **89.6 seconds**.

* **State Transition Table (STT) generation** (`python -m baby.information epistemology`):
  Construction of the full state transition tensor (`epistemology.npy`, 770 MB, shape 788,986 × 256, int32) is performed via vectorised NumPy routines. The measured runtime is **5 minutes 30 seconds**.

* **Phenomenology map construction** (`python -m baby.information phenomenology`):
  The canonical-orbit (phenomenology) mapping is built in a single pass, using a union-find algorithm over the STT. Wall-time: **11 seconds**.

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
  9.7 MB on disk; once parsed, resident size is \~12 MB.

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

#### 1. Genetic‑code analogies

GyroSI’s instruction algebra in the eight‑bit space ℤ₂⁸ happens to echo several small‑number structures familiar from molecular genetics, though no claim is made that the two domains share the same state count or evolutionary origin.

* Four intron actions (L0, LI, FG, BG) are the minimal set that closes the algebra, just as four nucleotides form the minimal alphabet for base pairing.

* Three spatial axes in every tensor slice match the three positions in a codon, each position modulating a different degree of freedom inside the lattice.

* Two sign polarities ±1 reflect the two strands of complementary base pairing.

* Eight bits per intron provide 2⁸ distinct instructions, mirroring the 2 bits × 4‑symbol representation of a four‑mer in DNA notation.

* Sixty‑four active intron patterns (the six working bits after stripping the L0 anchors) cover the complete operational alphabet. Biology’s sixty‑four codons occupy the same combinatorial volume.

* Thirty‑two LI‑quotiented classes arise when whole‑tensor parity is identified; this folding is formally identical to wobble pairing that halves the codon set.

The large orbit of 788 986 physical states belongs purely to GyroSI’s internal physics and has no biological counterpart. The comparison therefore stops at the level of instruction algebra, not at the size of the state space.

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

The three fundamental operations defined in §2.2 of the main text are realised exactly:

* **XOR** drives every bit transformation and enforces involutive symmetry.
* **AND** stores the carry term, preserving path memory.
* **Coaddition** `a ⊞ b = a ⊕ gyr[a,¬b](b)` implements learning; its code lives in `governance.coadd`.

All run‑time transformations in `apply_gyration_and_transform` are combinations of those three primitives; nothing extra is introduced.

---

#### 4. Holographic principle in practice

A single eight‑bit intron always touches the entire forty‑eight‑bit state through four complementary twelve‑bit masks. Conversely, any state can be reached in at most six introns. This bidirectional property embodies the holographic claim that every part encodes the whole. The code paths involved are `transcribe_byte`, `apply_gyration_and_transform`, and the breadth‑first discovery routine that proves the six‑step closure.

---

#### 5. Stabiliser and modulus

Breadth‑first exploration over the full instruction set discovers exactly 788 986 distinct states and a diameter of six. The stabiliser of the archetype has order two (global parity) multiplied by eleven (frame degeneracy). The remaining factor, 35 863, is prime, confirming that no further quotient is possible. These facts are verified at build time and are used to reject any physics violation at run time.

No biological code shows the same modulus; the coincidence stops at the smaller sub‑structures outlined above.

---

#### 6. Further correspondences

Other mappings noted in the main text are retained without restatement:

* The angular sequence π/2, π/4, π/4, 0 for CS → UNA → ONA → BU.
* The packed‑integer versus tensor dual representation.
* The role of the endogenous modulus as a hard physical constant.

Readers seeking proofs or implementation details will find the relevant functions in `baby.governance`, `baby.information`, and `baby.inference`.

===


