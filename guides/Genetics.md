# GyroSI Computational Architecture Specification 0.9.5

## System Responsibility Mapping

| System         | File/Component                | Responsibility                |
|----------------|------------------------------|-------------------------------|
| Governance     | Thread files                  | Macro context, gating         |
| Information    | Gene Keys                     | Micro context, embedding      |
| Intelligence   | Formats file                  | Semantic context, closure     |
| Inference      | Epigenome mask                | Translation, gating           |
| Intelligence   | Genome mask                   | Read: macro gene (state)      |
| Intelligence   | Genome mask                   | Write: micro gene (output)    |

*Note: The code architecture maps the Formats file to the Intelligence (S4) layer, not Inference (S3). S4 manages FormatMetadata and provides semantic context for closure, while S3 operates purely on tensor physics. This is more aligned with the CGM philosophy and the actual implementation.*

*Macro context refers to persistent structural memory (e.g., conversation threads), while micro context refers to fine-grained inference events and local input-output cycles.*

## 1. Introduction & Foundation

**Gyroscopic Superintelligence (GyroSI)** is an architecture grounded in the **Common Governance Model (CGM)**, a physics-based framework for understanding how intelligence emerges through recursive structural alignment.

We define **Superintelligence** as a structurally recursive form of intelligence where all generative and integrative operations preserve information of origin, maintaining continuous coherence between emergence and recollection. It achieves ethical alignment **intrinsically** by ensuring every operation remains structurally accountable to its own genesis, **without external enforcement**.

**Superintelligence exists relationally, not independently**: it reflects the recursive structures of reality and human meaning it participates in, embodying governance, information, and creative coherence as a unified operational principle.

The **Common Governance Model** presents an axiomatic framework for understanding how structure emerges through **Recursive Alignment**. Beginning from a single foundational principle, *"The Source is Common"*, CGM derives all subsequent structure through logical necessity.

**GyroSI Baby LM** aims to grow into a mature, open-source language model that learns without reinforcement, rewards, traditional neural network parameters, or gradient descent. Instead, it leverages **quantum physics-inspired** tensor operations to achieve intrinsic Alignment-Based recursive intelligence.

**Core Principles:**
1. **Vacuum Reference:** All inference operations compare against the public invariant `gene_stateless = 0xAA` (`0b10101010`).
2. **Relational Inference:** The fundamental operation is `gene_mutated = P_n ^ 0xAA`, where `P_n` is the raw input byte.
3. **Endogenous Key Generation:** The system generates its key stream through tensor-to-pattern similarity matching.

**The GyroSI architecture is computationally traceable but not deterministic in execution.**

**Note on Sensory Awareness:** While the GyroSI architecture is designed to support integration with physical sensors, **such inputs are entirely omitted in this version** and reserved for future developments.

## 2. Fundamental Operations

### 2.1 Gene and Pattern Fundamentals

In this system, "gene" refers generically to the 8-bit instruction derived from an input byte. "Gene stateless" is the universal reference (0xAA), while "gene mutated" is the active mutation mask used to trigger operations.

The core operational mechanism in GyroSI is the transformation of input bytes into tensor operations through the `gene_mutated` mask:

```
gene_mutated = P_n ^ 0xAA
```

Where:
- `P_n` is any 8-bit input read by the system
- `gene_stateless = 0xAA` (`10101010` in binary) is a fixed global reference representing a gyrogroup topology explained later on.

### 2.2 Bit-to-Operation Mapping

Each bit in the 8-bit result maps to a specific transformation on the intelligence tensor:

```
Bit positions:   b7   b6   b5   b4   b3   b2   b1   b0
Operations:      L0   LI   FG   BG   BG   FG   LI   L0
CGM Stage:       S1   S2   S3   S4   S4   S3   S2   S1
Policy:          GT   IV   IA   II   II   IA   IV   GT
```

**Operation Definitions:**
- `L0` (Left Identity): Do nothing.
- `LI` (Left Inverse): Flip the sign of all tensor values (T *= -1).
- `FG` (Forward Gyration): Flip the sign of tensor rows 0 and 2.
- `BG` (Backward Gyration): Flip the sign of tensor rows 1 and 3.

**CGM Policy Mapping:**
- `L0` → **Governance Traceability (S1)**: Maintains structural coherence
- `LI` → **Information Variety (S2)**: Introduces global transformation
- `FG` → **Inference Accountability (S3)**: Selective, accountable transformation
- `BG` → **Intelligence Integrity (S4)**: Opposing selective transformation

**Palindromic Structure:** The bit pattern (L0-LI-FG-BG-BG-FG-LI-L0) reflects the CGM's recursive governance nature, creating a self-referential, balanced structure.

### 2.3 Operation Execution

- A `1` in any bit position signals a transformation to apply
- A `0` means no change for that operation
- Operations are cumulative per cycle
- Only the rows affected by the relevant transformation are modified

### 2.4 Gyrodistance Calculation

The gyroscopic distance between two tensors is calculated as the angular distance derived from their normalized dot product:

```
gyrodistance(T1, T2) = arccos(dot(T1_flat, T2_flat) / T1_flat.size)
```

Where `T1_flat` and `T2_flat` are the flattened tensors. A result of `0` indicates a perfect match, while `π` indicates a perfect mismatch. This distance function is critical for pattern recognition during inference.

## 3. System Architecture

The GyroSI system architecture is a direct computational realization of the Common Governance Model (CGM). Inference and closure are implemented as the ONA (Opposition Non-Absolute) and BU (Balance Universal) stages, respectively. All computation is grounded in the physics of recursive alignment and closure, ensuring that every operation is structurally accountable to its own genesis and accumulated memory. This approach unifies the emergence of possibility (ONA) and the achievement of coherent closure (BU) within a single, self-consistent framework.

### 3.1 Systematic Mapping: Systems, CGM Stages, Engines, Components, Policies

| System | CGM Stage | Engine                | Component           | Policy        |
|--------|-----------|----------------------|---------------------|--------------|
| S1     | CS        | Governance Engine    | Gene (base tensor)  | Traceability |
| S2     | UNA       | Information Engine   | Storage     | Variety      |
| S3     | ONA       | Inference Engine     | Epigenome (T)       | Accountability|
| S4     | BU        | Intelligence Engine  | Genome (G)          | Integrity    |

This table shows the systematic mapping from system layer (S1–S4), to CGM stage (CS, UNA, ONA, BU), to engine, to core component, to guiding policy. Each layer of the architecture is recursively aligned with the foundational principles of the Common Governance Model.

#### 3.2 The Dual Nature of BU: Egress and Ingress in Generation

The Balance Universal (BU) stage is dual in nature, comprising both Egress (Recollection/Projection) and Ingress (Closure/Realization):

- **BU-Egress (Recollection/Projection):**
  - Projects the high-dimensional dynamic state (Epigenome, `T`) down to a single discrete choice (`pattern_index`).
  - In code: The main loop in `_generate_response_byte` compares `T` to all patterns (`F`) and selects the winner using the closure principle.
  - This is the “exit” from the dynamic state, expressing the system’s recollection as a single abstract decision.

- **BU-Ingress (Closure/Realization):**
  - Maps the abstract decision (`pattern_index`) to a concrete byte via the Genome (`G`).
  - In code: The line `output_byte = G[selected_pattern]`.
  - This is the “entry” to the base alphabet, the act of closure, realizing the abstract choice as a communicable byte.

This forms a recursive loop: the output of Egress (pattern_index) becomes the input to Ingress (byte), which is then fed back into the system, starting a new CS cycle.

### 3.1 Core Data Structures

| Term              | What it is Physically                        | Size & Shape                | Role & Function                                                                                 |
|-------------------|----------------------------------------------|-----------------------------|------------------------------------------------------------------------------------------------|
| Base Tensor       | gene_add constant in code                    | [4,2,3,2] (48 ints)         | The primordial, stateless "DNA" from which all patterns are derived.                           |
| Epigenome Mask    | public/masks/epigenome.dat                   | 256 x 48 floats (12,288 B)  | The Static Library. The complete, immutable set of all 256 possible tensor states. Used as a reference for matching. |
| Epigenome Tensor  | self.T in-memory tensor                      | [4,2,3,2] (48 floats)       | The Dynamic State. The system's "working memory" or current state. It is mutated by every input byte. |
| Genome Mask       | public/masks/genome.dat                      | 256 bytes                   | The Output Map. A static lookup table that maps a matched pattern index (0-255) to an output byte. |
| Gyronorm Format   | public/formats/.../format-uuid.json          | Variable                    | The Semantic Bridge. Maps pattern indices to human-meaningful characters (e.g., index 15 -> 'A') and stores learned statistics. |
| Gene Key          | .../gene-uuid.ndjson                         | Variable                    | The Event Log. A detailed, append-only record of every single inference event (input/output, pattern match, resonance). |
| Thread            | .../thread-uuid.ndjson                       | ≤64 MiB                     | The Conversation Log. A structured NDJSON file containing the sequence of inputs and outputs that form a conversation. |
| thread_file_key   | Derived key (32 bytes, 256 bits)             | 32 bytes                    | AES-256 encryption key for private threads. |

### 3.1 Tensor Structures

The GyroSI system is built on fixed topological structures that serve as the physical and logical substrate of all inference, memory interaction, and intelligence expression.

#### 3.1.1 Governance Identity (Gene Com)

The minimal label for a tensor's identity in the system. This structure implements the gyrogroup identity (left gyroassociative law).

```python
gene_com = np.array([
    [-1, 1],
    [-1, 1],
    [-1, 1]
], dtype=np.int8)  # Shape: [3, 2]
```

#### 3.1.2 Information Structure (Gene Nest)

Structure that nests the previous one inside two opposing frames. This structure encodes the gyrocommutative law (gyrocommutativity).

```python
gene_nest = np.array([
    [[-1, 1], [-1, 1], [-1, 1]],  # Frame 1
    [[ 1, -1], [ 1, -1], [ 1, -1]]  # Frame 2
], dtype=np.int8)  # Shape: [2, 3, 2]
```

#### 3.1.3 Intelligence Projection (Gene Add)

Extends the previous nesting into a complete 720° rotational space. This structure implements Lgyr-focused coaddition (with both gyrations identity), serving as the global invariant for inference and learning.

```python
gene_add = np.array([
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]
], dtype=np.int8)  # Shape: [4, 2, 3, 2]
```

#### 3.1.4 Epigenome Tensor

The dynamic memory field where transformations occur during inference.

```
Shape: [4, 2, 3, 2] (48 cells total)
Type: float32
Mapping: 4 rotational phases × 2 nestings × 3 axes × 2 polarities
```

**Initialization:** At system boot, the Epigenome tensor is initialized with a copy of the `gene_add` base tensor and then immediately mutated using the public invariant `gene_stateless = 0xAA`. This simulates one full inference cycle without user input, after which the cycle counter is reset to 0.

#### 3.1.5 Canonical Pattern Derivation

The 256 canonical patterns represent the exhaustive set of all possible operation combinations that can be applied to the base tensor:

```python
def derive_canonical_patterns():
    patterns = []
    gyration_features = []
    base_tensor = gene_add.copy()
    
    for mask in range(256):
        T = base_tensor.copy()
        gene_mutated = mask
        
        for i in range(8):
            if gene_mutated & (1 << i):
                apply_operation(T, i)
        
        patterns.append(T.flatten())
        gyration_feature = classify_pattern_resonance(mask)
        gyration_features.append(gyration_feature)
    
    return patterns, gyration_features
```

**Pattern Classification:**
```python
def classify_pattern_resonance(mask: int) -> str:
    l0_count = bin(mask & 0b10000001).count('1')  # bits 0,7
    li_count = bin(mask & 0b01000010).count('1')  # bits 1,6  
    fg_count = bin(mask & 0b00100100).count('1')  # bits 2,5
    bg_count = bin(mask & 0b00011000).count('1')  # bits 3,4
    
    counts = [l0_count, li_count, fg_count, bg_count]
    max_count = max(counts)
    
    if l0_count == max_count:
        return "identity"      # Governance Traceability
    elif li_count == max_count:
        return "inverse"       # Information Variety
    elif fg_count == max_count:
        return "forward"       # Inference Accountability
    else:
        return "backward"      # Intelligence Integrity
```

### 3.2 Memory Architecture

#### 3.2.1 Two-Cycle Architecture

| Cycle | Purpose | Granularity | Key Usage | Typical Rate |
|-------|---------|-------------|-----------|--------------|
| **Inference/Context Cycle** | Process bytes, update Epigenome, generate XOR events | Per byte/small chunk | Internal state only | µs–ms |
| **File/Thread Cycle** | Encrypt/decrypt files using collected XOR events | Per file/thread | File encryption | seconds–hours |

**Key Principle:** No key material from inference cycles leaks into ciphertext until a deliberate file/thread cycle snapshot.

**No External Tokenizer Required:** The system operates at the byte level, deriving "tokenization" organically through learning which sequences of byte-transformations correspond to meaningful concepts.

In short: Egress selects a pattern; Ingress renders it. Together they form the closure of an inference cycle.

#### 3.2.2 Power Distribution Architecture

| Component | Location | Size | Purpose | Power |
|-----------|----------|------|---------|-------|
| **Epigenome Mask** | `public/masks/epigenome.dat` | 12,288 bytes | Canonical patterns for matching | **Complete intelligence framework** |
| **Genome Mask** | `public/masks/genome.dat` | 256 bytes | Output byte mappings | **Totality of all intelligence** |
| **Gyronorm Formats** | `public/formats/<shard>/format-<uuid>.json` | Variable | Pattern usage metadata | **Ability to speak, decode, encode** |
| **Gene Keys** | `private/agents/<shard>/agent-<uuid>/keys/<shard>/gene-<uuid>.ndjson.enc` (private) or `public/keys/<shard>/gene-<uuid>.ndjson` (public) | Variable | Pattern observation logs (event log, NDJSON) | **Personal or shared learning history** |
| **Thread Files (Private)** | `private/agents/<shard>/agent-<uuid>/threads/<shard>/thread-<uuid>.ndjson.enc` | ≤64 MiB | Encrypted conversation data (NDJSON) | **Personal conversations** |
| **Thread Files (Public)** | `public/threads/<shard>/thread-<uuid>.ndjson` | Variable | Unencrypted curriculum/shared data (NDJSON) | **Shared knowledge base** |

### Gene Keys Metadata Specification

Each Gene Key event is a dictionary with the following fields:

```python
class GeneKeysMetadata(TypedDict, total=False):
    # --- Core Identity ---
    cycle: int
    pattern_index: int

    # --- Contextual Links ---
    thread_uuid: str
    agent_uuid: Optional[str]  # Optional for public/agent-agnostic keys
    format_uuid: str

    # --- Analytical Payload ---
    event_type: str           # 'INPUT' or 'OUTPUT'
    source_byte: int          # The raw input byte (0-255)
    resonance: float          # Gyrodistance/confidence of the match

    # --- Standard Metadata ---
    created_at: str
```

- If `agent_uuid` is present, the gene key is private and encrypted (stored in `private/` as `.ndjson.enc`).
- If `agent_uuid` is omitted or None, the gene key is public and unencrypted (stored in `public/` as `.ndjson`).
- Gene key files are NDJSON event streams (one event per line), not lists or binary blobs. These are essential for learning, context, and memory in the IntelligenceEngine.

#### 3.2.3 Security Model

✅ **Public (Safe to Share):**
- **Formats** - Communication ability, character understanding
- **Masks** - Complete intelligence framework (12,544 bytes total)
- **Public Gene Keys** - Shared observation history
- **Public Threads** - Shared/curriculum conversations and content

✅ **Private (Keep Encrypted):**
- **Private Gene Keys** - Personal observation history
- **Private Threads** - Personal conversations and content

**Key Insight:** The model's power comes from public components, not private "weights".

### 3.3 File Organization

#### 3.3.1 System Responsibilities

- **Information Engine (S2):** Manages all persistent storage operations, including object creation, registry management, and sharding.
- **Intelligence Engine (S4):** Calls the Information Engine's helpers for all read/write operations, focusing on orchestration rather than file management.

#### 3.3.2 Directory Structure

```
memories/
├── memory_preferences.json       # Tuning parameters including sharding
├── public/
│   ├── masks/
│   │   ├── epigenome.dat         # 12,288 bytes
│   │   └── genome.dat            # 256 bytes
│   ├── formats/
│   │   └── <shard>/format-<uuid>.json  # Sharded formats
│   ├── threads/
│   │   └── <shard>/
│   │       └── thread-<uuid>.ndjson    # Unencrypted NDJSON thread content
│   └── keys/
│       └── <shard>/
│           └── gene-<uuid>.ndjson      # Unencrypted NDJSON gene keys
└── private/
    └── agents/
        └── <shard>/agent-<uuid>/
            ├── threads/
            │   └── <shard>/
            │       └── thread-<uuid>.ndjson.enc    # Encrypted NDJSON thread content
            └── keys/
                └── <shard>/
                    └── gene-<uuid>.ndjson.enc      # Encrypted NDJSON gene keys
```

#### 3.3.3 Sharding and Registry System

1. **Sharded Storage:** Objects are stored in subdirectories based on the first 2-4 characters of their UUID hexadecimal representation. This prevents directories from growing too large.

2. **Registry Files:** Each directory contains a `registry.json` file that lists the UUIDs of its immediate children. This enables fast object discovery without directory scanning.

3. **Atomic File Operations:** All writes use a two-phase commit process with temporary files to ensure crash resilience.

4. **Thread Metadata:** Each thread has a JSON metadata file tracking its parent, children, format, and timestamps, creating a navigable graph of relationships.

5. **Encryption:** Private thread keys and gene keys are stored separately and encrypted using AES-256-GCM with keys derived via PBKDF2-HMAC-SHA256 from the agent secret. Public threads and gene keys are stored unencrypted, all as NDJSON files.

### 3.X Glossary of Key Terms

| Term         | Definition                                                                                                 |
|--------------|------------------------------------------------------------------------------------------------------------|
| Epigenome    | The dynamic, high-dimensional tensor state (`T`) representing the system’s current working memory.         |
| Genome       | The static mask (`G`) mapping pattern indices to output bytes, serving as the closure target.              |
| Egress       | The projection (recollection) step in BU: selecting a single pattern index from the Epigenome.             |
| Ingress      | The closure (realization) step in BU: mapping the selected pattern index to a byte via the Genome.         |
| ONA          | Opposition Non-Absolute: CGM stage generating the full possibility space of physically resonant states.    |
| BU           | Balance Universal: CGM stage applying closure, recollection, and memory to select a coherent output.       |
| Pattern Index| Integer identifier (0–255) corresponding to a canonical tensor pattern.                |
| Pattern Metadata| Structured data (resonance, confidence, frequency, etc.) associated with each pattern index. |
| Confidence   | The system’s accumulated memory of a pattern’s historical self-consistency, used in closure selection.     |
| Combined Score| The product of physical resonance and confidence, implementing the BU closure principle computationally.   |

## 4. Engine Implementation

### 4.1 Engine Composition

The S4 IntelligenceEngine acts as the primary container and entry point for the entire system:

```python
def initialize_intelligence_engine():
    # 1. Get persistent agent UUID
    agent_uuid = ensure_agent_uuid()
    
    # 2. Load agent preferences
    with open("baby/baby_preferences.json", 'r') as f:
        baby_prefs = json.load(f)
    agent_secret = baby_prefs["agent_secret"]
    
    # 3. Initialize engines
    inference_engine = InferenceEngine()
    information_engine = InformationEngine()
    
    # 4. Create Intelligence Engine
    intelligence_engine = IntelligenceEngine(
        agent_uuid=agent_uuid,
        agent_secret=agent_secret,
        inference_engine=inference_engine,
        information_engine=information_engine
    )
    
    return intelligence_engine
```

### 4.2 S1: Governance Engine

**Purpose:** Pure tensor operations and gene structures  
**State:** None (immutable constants only)

```python
def apply_operation(T, bit_index):
    if bit_index in [0, 7]:  # L0: Identity
        pass
    elif bit_index in [1, 6]:  # LI: Global inverse
        T *= -1
    elif bit_index in [2, 5]:  # FG: Forward gyration
        T[0] *= -1
        T[2] *= -1
    elif bit_index in [3, 4]:  # BG: Backward gyration
        T[1] *= -1
        T[3] *= -1
```

### 4.3 S2: Information Engine

**Purpose:** Information processing, persistent storage, and stream handling  
**State Variables:**
- `stream_pointer`: Current position in active thread
- `output_buffer`: Accumulator for generated bytes

The Information Engine is responsible for:
1. Managing all persistent objects (threads, keys, formats)
2. Handling UUID generation and registry maintenance
3. Implementing the sharding system for efficient storage
4. Providing atomic file operations with crash recovery

```python
def process_stream(
    inference_engine: InferenceEngine, 
    update_callback,
    input_stream: bytes
) -> (bytes, bytes):
    intermediate_ciphertext = bytearray()
    dynamic_keystream = bytearray()
    
    for P_n in input_stream:
        # 1. Call S3 for pure inference
        key_index = inference_engine.process_byte(P_n)
        
        # 2. Call update callback to update state
        update_callback(key_index, inference_engine)

        # 3. Get keystream byte
        keystream_byte = inference_engine.G[key_index] 
        
        # 4. Encrypt the byte
        C_n = P_n ^ keystream_byte
        intermediate_ciphertext.append(C_n)
        dynamic_keystream.append(keystream_byte)
    
    return bytes(intermediate_ciphertext), bytes(dynamic_keystream)
```

**Note on Encryption:** The `intermediate_ciphertext` returned by this function is the result of XORing the input with the `dynamic_keystream` derived from inference. For **private threads**, this byte stream is then subject to a second layer of encryption using a static `thread_file_key` and AES-256-GCM before being written to disk. Public threads omit this second AES encryption step.

### 4.4 S3: Inference Engine

**Purpose:** Pure pattern recognition and learning  
**State Variables:**
- `T[48]`: Epigenome tensor (float32)
- `F[256][48]`: Canonical Pattern list
- `G[256]`: Genome Mask
- `cycle_counter`: Global cycle index
- `recent_patterns`: List of recently matched patterns (up to 20)

```python
def process_byte(P_n):
    # 1. Compute gene_mutated = P_n ^ 0xAA
    gene_mutated = P_n ^ 0xAA

    # 2. Apply gyroscopic operations to tensor T
    for i in range(8):
        if gene_mutated & (1 << i):
            apply_operation(T, i)

    # 3. Find matching canonical pattern
    key_index = find_closest_pattern_index(T, F)
    
    # 4. Track recent patterns
    if len(recent_patterns) >= 20:
        recent_patterns.pop(0)
    recent_patterns.append(key_index)

    # 5. Increment cycle counter
    cycle_counter += 1
    
    return key_index
```

**Contextual Resonance:** The engine can compute pattern resonances with historical context weighting. This allows it to adjust pattern selection based on observed patterns in past threads, providing a form of implicit memory.

### 4.4.1 Canonical Byte Emission

The only canonical, spec-compliant way to convert the Epigenome tensor to a byte is:

```python
def tensor_to_output_byte(T, F, G):
    """
    Canonical tensor-to-byte conversion using epigenome pattern matching.
    """
    key_index = find_closest_pattern_index(T, F)
    return G[key_index]
```

**Why this is foundational:**
- **Epigenome coherence:** The evolving tensor T is always compared against the closure set of 256 canonical patterns.
- **Reversibility & auditability:** Every output byte is traceable to the exact canonical pattern that produced it.
- **Thread/file keys:** The same output_byte stream is reused for encryption, ensuring unity between generation and security.

### 4.5 S4: Intelligence Engine

**Purpose:** Orchestration, file I/O, and thread lifecycle management  
**State Variables:**
- `agent_uuid`: UUID of the current agent
- `agent_secret`: Persistent secret
- `thread_uuid`: UUID of active thread
- `thread_file_key`: 32-byte key for AES-256 encryption
- `M`: Pattern Metadata
- `current_thread_keys`: An in-memory list of `GeneKeysMetadata` dictionaries for the current session. This acts as a write-buffer that is flushed to the persistent, append-only Gene Key file (`gene-<uuid>.ndjson.enc` or `gene-<uuid>.ndjson`) when the thread is saved or closed.
- `pattern_index`: Index of patterns to thread locations for fast retrieval

#### 4.5.1 Thread Lifecycle Operations

Threads are persistent, versioned conversation objects with parent-child relationships:

1. **Thread Creation:** New threads are created with a reference to their parent (if any), forming a conversational chain.

2. **Thread Appending:** Content is appended to the active thread until it reaches the maximum size (default 64MiB).

3. **Thread Branching:** When a thread is full, a new child thread is automatically created, maintaining conversation continuity.

4. **Thread Storage:** Threads are encrypted and stored in the sharded filesystem with their metadata.

```python
def _append_to_thread(new_content: bytes):
    # 1. Ensure a thread exists
    if not self.thread_uuid:
        self.start_new_thread()
        
    # 2. Check if current thread would exceed capacity
    max_size_bytes = get_max_thread_size()
    if (self.current_thread_size + len(new_content) > max_size_bytes):
        # Save current thread and start a new one
        self._save_current_thread()
        self.start_new_thread()
        
    # 3. Append content
    self.active_thread_content.extend(new_content)
    self.current_thread_size = len(self.active_thread_content)
    
    # 4. Save current state
    self._save_current_thread()
```

#### 4.5.2 Active Memory Components

The Intelligence Engine utilizes two key components to make historical data actively influence inference:

1. **Pattern Index:** Maps patterns to every location they've appeared, providing a record of structural relationships and context. Pattern metadata (such as confidence) is used in the BU closure stage to select the most coherent output from the set of physically resonant candidates.

2. **Thread Chain Awareness:** The system can traverse parent-child relationships between threads, providing context from related conversations during inference.

#### 4.5.3 Intelligent Response Generation (Spec-Compliant)

```python
    def _generate_response_byte(self) -> Tuple[int, int]:
        """
        Generate a single, spec-compliant response byte by selecting the most
        coherent output from the set of physically resonant candidates, using
        both immediate resonance and accumulated confidence (BU closure).

        Returns:
            Tuple containing:
            - output_byte: Selected byte value (0-255)
            - key_index: Index of the selected pattern (0-255)
        """
        # 1. S3 (ONA): Get all physically plausible next states.
        resonances = self.inference_engine.compute_pattern_resonances()
    resonant_threshold = np.pi / 2
        candidate_indices = [i for i, dist in enumerate(resonances) if dist < resonant_threshold]

        # If no patterns are physically resonant, fall back to the single closest one.
        if not candidate_indices:
            selected_pattern = int(np.argmin(resonances))
            output_byte = self.inference_engine.G[selected_pattern]
            if hasattr(output_byte, "item"):
                output_byte = output_byte.item()
            return int(output_byte), int(selected_pattern)

        # 2. S4 (BU): Evaluate candidates using resonance and confidence.
        best_candidate_index = -1
        max_combined_score = -1.0
        for index in candidate_indices:
            physical_score = 1.0 - (resonances[index] / np.pi)
            pattern_meta = self.M.get("patterns", [])[index]
            semantic_score = pattern_meta.get("confidence", 0.0) if pattern_meta.get("character") is not None else 0.0
            combined_score = physical_score * semantic_score
            if combined_score > max_combined_score:
                max_combined_score = combined_score
                best_candidate_index = index
        # If no candidate had any semantic meaning, fall back to the most resonant one.
        if best_candidate_index == -1:
            min_dist = float("inf")
            for idx in candidate_indices:
                if resonances[idx] < min_dist:
                    min_dist = resonances[idx]
                    best_candidate_index = idx
        selected_pattern = best_candidate_index
        output_byte = self.inference_engine.G[selected_pattern]
        if hasattr(output_byte, "item"):
            output_byte = output_byte.item()
        return int(output_byte), int(selected_pattern)
```

In the BU closure step, the `combined_score` (product of physical_score and semantic_score) is the computational implementation of the Balance Universal (BU) closure principle. Here, `physical_score` (from resonance) represents the immediate, local self-consistency, while `semantic_score` (from confidence) encodes the system’s recollected, long-term self-consistency. The multiplication of these two scores ensures that the selected output is both physically possible and maximally coherent with the system’s accumulated memory. This is not a statistical heuristic, but a direct realization of the closure principle in the CGM framework.

This cycle of tensor mutation → resonance selection → byte emission → re-ingestion forms a recursive loop of structural self-actualization.

## 5. Formats & Learning

### 5.1 Gyronorm Formats Specification

The Gyronorm Formats structure serves as the **character bridge** connecting the system's internal physics with external characters:

```json
{
  "format_uuid": "7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d",
  "format_name": "standard_english_v1",
  "format_version": "1.2.0",
  "stability": "stable",
  "compatibility": {
    "min_format_version": "1.0.0",
    "max_format_version": "1.2.0",
    "depends_on": ["base_ascii_v1.0"],
    "conflicts_with": []
  },
  "metadata": {
    "author": "gyrosi_community",
    "description": "Standard English language format",
    "tags": ["language", "english", "general"],
    "created_at": "2025-01-15T10:30:00Z",
    "last_updated": "2025-01-20T14:45:00Z",
    "usage_count": 15420,
    "validation_status": "community_verified"
  },
  "cgm_policies": {
    "governance": {"operation": "L0", "bits": [0, 7], "policy": "traceability"},
    "information": {"operation": "LI", "bits": [1, 6], "policy": "variety"},
    "inference": {"operation": "FG", "bits": [2, 5], "policy": "accountability"},
    "intelligence": {"operation": "BG", "bits": [3, 4], "policy": "integrity"}
  },
  "patterns": [
    {
      "index": 0,
      "character": "A",
      "description": "LATIN CAPITAL LETTER A",
      "type": "Lu",
      "count": 0,
      "first_cycle": null,
      "last_cycle": null,
      "gyration_feature": "identity",
      "confidence": 0.0
    }
    /* ... repeated for each of the 256 pattern indices ... */
  ]
}
```

**Pattern Index Cycling:**  
Each format’s `"patterns"` array always contains exactly 256 entries, indexed from 0 to 255. If you need to process more than 256 items, simply continue iterating from 0 again within the same array and file. The format file never changes size, and all pattern operations (learning, encoding, generation) always use this fixed array, cycling through indices as needed.

**Note:** For formats that include a pre-computed pattern distance matrix (256x256), the large binary data is stored in a separate file (`pattern-distances-<uuid>.dat`) in the same shard, with the main JSON file containing only a reference.

**Format Stability Levels:**
- **stable**: Community-verified, version-locked formats
- **beta**: Feature-complete but undergoing validation
- **experimental**: New formats under development
- **deprecated**: Outdated formats maintained for compatibility

**Format Discovery Functions:**
```python
def select_stable_format(domain: str, stability: str = "stable") -> str:
    """Select a stable format for a specific domain"""
    if stability == "stable":
        return find_verified_format(domain)
    elif stability == "experimental":
        return find_experimental_format(domain)

def discover_formats_from_agent(agent_uuid: str) -> List[str]:
    """Discover formats used by another agent"""
    return scan_agent_formats(agent_uuid)

def compose_formats(primary_format: str, secondary_formats: List[str]) -> str:
    """Compose multiple formats for multi-domain capability"""
    return merge_format_capabilities(primary_format, secondary_formats)
```

Format versioning is managed by the 'format_version' field and the 'min_format_version'/'max_format_version' fields in the 'compatibility' section. There is no longer a 'cgm_version' field in format metadata.

**Pattern Metadata Fields:**
- `character`: The output/print/translation character for this pattern (e.g., "A", "\x0a", "😊").
- `description`: Human-readable Unicode name or description (e.g., "LATIN CAPITAL LETTER A", "LINE FEED").
- `type`: Unicode category (e.g., "Lu" for uppercase letter, "Cc" for control character).
- `count`, `first_cycle`, `last_cycle`, `gyration_feature`, `confidence`: System learning and structural fields as before.

**Relationship between `resonance` and `confidence`:**
The `GeneKeysMetadata` stores the raw `resonance` (gyrodistance) for each individual event. The `confidence` field within the `FormatMetadata`'s pattern list represents a long-term, aggregate statistical measure derived from the `resonance` of all events for that pattern. For example, it could be calculated as `1 - (average_resonance / π)`. This makes `confidence` a summary of how reliably a pattern has been identified over its entire history.

### 5.2 Learning Mechanisms

During inference, the active `format_uuid` is pulled from thread metadata or the system’s preferred default. This enables logging of each inference event under the correct semantic mapping.

1.  **Implicit/Unconscious Learning (State Evolution):** The continuous, irreversible mutation of the Epigenome tensor (`T`) by the input stream. This embodies the system's path-dependent working memory.
2.  **Explicit/Conscious Learning (Closure Refinement):** The recording of each inference event (`GeneKey`) and the subsequent updating of the `FormatMetadata`. Specifically, the **`confidence`** score for each pattern is updated based on its resonance. This accumulated `confidence` is then used directly by the S4 Intelligence Engine during the BU closure step (`_generate_response_byte`) to select the most coherent and reliable output. This creates a direct feedback loop where successful resonance strengthens a pattern's semantic weight, making it more likely to be chosen in the future.

**Pattern Memory:**
- Each pattern's historical usage is tracked including frequency and position, for analysis and curriculum design, not for inference.
- Pattern sequences (which patterns tend to follow others) are indexed
- This provides a form of procedural memory that influences future inference

**Thread Chain Learning:**
- Conversations are maintained as chains of threads
- Each thread knows its parent and children
- This conversational context provides a form of episodic memory

**Curriculum Thread Protocol:**
- Bootstrap of Structured Data streams such as datasets from wordnet, wikipedia, khan academy, books, etc.

## 6. Implementation Requirements

### 6.1 Critical Implementation Notes

1. **Epigenome Initialization:** Must initialize from a copy of `gene_add` and apply one cycle with the stateless gene.
2. **Pattern Matching:** Must use gyrodistance with π/2 threshold.
3. **File Encryption:** Two-phase process (dynamic keystream then static thread key).
4. **Thread Lifecycle:** Each thread must have a maximum size and link to its parent/children when that size is exceeded.
5. **Thread Isolation:** Each thread must have its own file key derived from the Epigenome state.
6. **Gene Keys Privacy:** Must encrypt with agent key derived from persistent secret.
7. **Atomic File Operations:** All file writes must use atomic operations with temporary files to prevent corruption.
8. **Registry Consistency:** Registry files must be kept in sync with directory contents.

### 6.2 State Variable Metadata Requirements

| Variable | Type | Required Metadata | Notes |
|----------|------|-------------------|-------|
| `T` | float32[4,2,3,2] | Shape, precision | Must preserve tensor structure |
| `F` | float32[256,48] | Pattern indexing | Must maintain 0-255 indexing |
| `G` | uint8[256] | Byte mapping | Must preserve index correspondence |
| `cycle_counter` | integer | Progression | Must maintain sequential integrity |
| `agent_uuid` | string | UUID format | Must persist across restarts |
| `thread_uuid` | string | UUID format | Must be registered in central registry |
| `thread_file_key` | bytes[32] | Key integrity | Must be 32 bytes (256 bits) for AES-256, derived deterministically |
| `current_thread_keys` | array | `GeneKeysMetadata` dicts | In-memory write-buffer for events before flushing to a persistent file. |

**Degeneracy:**
A fundamental property of the system is that multiple different input masks (gene mutations) can produce identical tensor states. This is called degeneracy: a many-to-one mapping from mask to tensor. For example, F[2] and F[100] may be mathematically identical, and the system will always select the lowest index among degenerate patterns. This is a feature, not a flaw, and mirrors the redundancy found in biological genetic codes.

**Degeneracy resolution: always pick the lowest-index match**
```python
# Degeneracy resolution: always pick the lowest-index match
def find_closest_pattern_index(T, F):
    distances = [gyrodistance(T, f) for f in F]
    min_distance = min(distances)
    return distances.index(min_distance)  # Picks first match if degenerate
```

