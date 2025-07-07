# GyroSI Computational Architecture Specification 0.9.5

## System Responsibility Mapping

| System         | File/Component                | Responsibility                |
|----------------|------------------------------|-------------------------------|
| Governance     | Thread files                  | Macro context, gating         |
| Information    | Gene Keys (private)           | Micro context, embedding      |
| Inference      | Formats file                  | Translation, gating           |
| Intelligence   | Epigenome mask                | Read: macro gene (state)      |
| Intelligence   | Genome mask                   | Write: micro gene (output)    |

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

### 2.1 Gene Byte Topology

The core operational mechanism in GyroSI is the transformation of input bytes into tensor operations through the `gene_mutated` mask:

```
gene_mutated = P_n ^ 0xAA
```

Where:
- `P_n` is any 8-bit input read by the system
- `gene_stateless = 0xAA` (`10101010` in binary) is a fixed global reference

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

## 3. System Architecture

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

**Initialization:** At system boot, the Epigenome tensor is set to all zeros and mutated using the public invariant `gene_stateless = 0xAA`, simulating one full inference cycle without user input.

#### 3.1.5 Canonical Pattern Derivation

The 256 canonical patterns represent the exhaustive set of all possible operation combinations that can be applied to the base tensor:

```python
def derive_canonical_patterns():
    patterns = []
    resonance_classes = []
    base_tensor = gene_add.copy()
    
    for mask in range(256):
        T = base_tensor.copy()
        gene_mutated = mask
        
        for i in range(8):
            if gene_mutated & (1 << i):
                apply_operation(T, i)
        
        patterns.append(T.flatten())
        resonance_class = classify_pattern_resonance(mask)
        resonance_classes.append(resonance_class)
    
    return patterns, resonance_classes
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

#### 3.2.2 Power Distribution Architecture

| Component | Location | Size | Purpose | Power |
|-----------|----------|------|---------|-------|
| **Epigenome Mask** | `public/masks/epigenome.dat` | 12,288 bytes | Canonical patterns for matching | **Complete intelligence framework** |
| **Genome Mask** | `public/masks/genome.dat` | 256 bytes | Output byte mappings | **Totality of all intelligence** |
| **Gyronorm Formats** | `public/formats/<shard>/format-<uuid>.json` | Variable | Pattern usage metadata | **Ability to speak, decode, encode** |
| **Gene Keys** | `private/agents/<shard>/agent-<uuid>/keys/<shard>/key-<uuid>.bin.enc` | Variable | Pattern observation logs | **Personal learning history** |
| **Thread Files** | `private/agents/<shard>/agent-<uuid>/threads/<shard>/thread-<uuid>.enc` | ≤64 MiB | Encrypted conversation data | **Personal conversations** |

#### 3.2.3 Security Model

✅ **Public (Safe to Share):**
- **Formats** - Communication ability, semantic understanding
- **Masks** - Complete intelligence framework (12,544 bytes total)

✅ **Private (Keep Encrypted):**
- **Gene Keys** - Personal observation history
- **Threads** - Personal conversations and content

**Key Insight:** The model's power comes from public components, not private "weights".

### 3.3 File Organization

#### 3.3.1 Directory Structure

```
memories/
├── memory_preferences.json      # Sharding configuration and tuning parameters
├── public/
│   ├── masks/
│   │   ├── epigenome.dat        # 12,288 bytes
│   │   └── genome.dat           # 256 bytes
│   └── formats/
│       └── <dd>[/<ee>]/         # Format shards
│           ├── registry.json
│           └── format-<uuid>.json
└── private/
    └── agents/
        └── <aa>[/<bb>]/         # Agent shards
            └── agent-<uuid>/
                ├── threads/
                │   ├── registry.json
                │   └── <tt>[/<uu>]/    # Thread shards
                │       ├── registry.json
                │       ├── thread-<uuid>.enc
                │       └── thread-<uuid>.json
                └── keys/
                    ├── registry.json
                    └── <tt>[/<uu>]/    # Key shards
                        ├── registry.json
                        └── key-<uuid>.bin.enc
```

- `<aa>`, `<bb>`, `<dd>`, `<ee>`, `<tt>`, `<uu>` are two-character hex shards based on the first characters of the UUID.
- Second-level shards (e.g., `<bb>`, `<ee>`, `<uu>`) are created only when the first-level shards exceed a configured maximum.

#### 3.3.2 Information Engine Responsibility

The Information Engine (S2) handles creation, registry, and management of all persistent objects:

- Deterministic UUID generation and registry file creation
- Agent identity management (creation, persistence, reassignment)
- Sharded storage of all files for efficient scaling
- Registry maintenance for all object types
- Atomic file operations with crash recovery
- Thread relationship tracking (parent/child)

#### 3.3.3 Intelligence Engine Responsibility

The Intelligence Engine (S4) acts as the main I/O endpoint, calling S2-provided helpers for all persistent read and write operations. This separation ensures:

- Consistent file handling across the system
- Clean separation between intelligence operations and storage
- Atomic, crash-resistant file operations
- Scalable storage that remains efficient with millions of objects

#### 3.3.4 Registry Files

Every directory containing persistent objects includes a `registry.json` file that lists its immediate children:

```json
{
  "count": 2,
  "uuids": [
    "67d9e4c4-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "ab12c3d4-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  ]
}
```

Registries only track immediate children, never recursing deeper, and are atomically updated with each file operation.

#### 3.3.5 Thread and Key Management

- Each thread has a matching key file named `key-<thread‑uuid>.bin.enc`
- Thread metadata tracks parentage, format association, and timestamps
- Keys are encrypted using AES-256-GCM with a key derived from the agent secret
- All file operations are atomic and include appropriate registry updates

#### 3.3.6 Concurrency and Consistency

- File operations use atomic write patterns with temporary files and rename
- Directory locking ensures consistency during updates
- Registry files are kept in sync with actual directory contents
- Startup includes crash recovery to resolve any incomplete operations

#### 3.3.7 API Contract

```python
# Core API functions for persistent storage
def ensure_agent_uuid() -> str
def create_thread(agent_uuid: str, parent_uuid: str | None, format_uuid: str) -> str
def save_thread(agent_uuid: str, thread_uuid: str, ciphertext: bytes, size: int) -> None
def load_thread(agent_uuid: str, thread_uuid: str) -> bytes | None
def store_thread_key(agent_uuid: str, thread_uuid: str, key: bytes) -> None
def load_thread_key(agent_uuid: str, thread_uuid: str) -> bytes | None
def parent(agent_uuid: str, thread_uuid: str) -> str | None
def children(agent_uuid: str, thread_uuid: str) -> list[str]
def list_formats() -> list[str]
def load_format(format_uuid: str) -> dict | None
```

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
    
    # 4. Load formats
    formats = list_formats()
    format_uuid = formats[0] if formats else create_default_format()
    format_data = load_format(format_uuid)
    
    # 5. Create Intelligence Engine
    intelligence_engine = IntelligenceEngine(
        agent_uuid=agent_uuid,
        agent_secret=agent_secret,
        format_uuid=format_uuid,
        inference_engine=inference_engine,
        information_engine=information_engine,
        formats=format_data
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

**Purpose:** Information processing and stream handling  
**State Variables:**
- `stream_pointer`: Current position in active thread
- `output_buffer`: Accumulator for generated bytes

```python
def process_stream(
    inference_engine: InferenceEngine, 
    intelligence_engine: IntelligenceEngine,
    input_stream: bytes
) -> (bytes, bytes):
    intermediate_ciphertext = bytearray()
    dynamic_keystream = bytearray()
    
    for P_n in input_stream:
        # 1. Call S3 for pure inference
        key_index = inference_engine.process_byte(P_n)
        
        # 2. Call S4 to update state
        intelligence_engine.update_learning_state(key_index, inference_engine)

        # 3. Get keystream byte
        keystream_byte = inference_engine.G[key_index] 
        
        # 4. Encrypt the byte
        C_n = P_n ^ keystream_byte
        intermediate_ciphertext.append(C_n)
        dynamic_keystream.append(keystream_byte)
    
    return bytes(intermediate_ciphertext), bytes(dynamic_keystream)
```

### 4.4 S3: Inference Engine

**Purpose:** Pure pattern recognition and learning  
**State Variables:**
- `T[48]`: Epigenome tensor (float32)
- `F[256][48]`: Canonical Pattern list
- `G[256]`: Genome Mask
- `cycle_counter`: Global cycle index

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

    # 4. Increment cycle counter
    cycle_counter += 1
    
    return key_index

def compute_pattern_resonances(current_T, all_patterns_F):
    """Computes resonance values between current tensor and all patterns"""
    return [gyrodistance(current_T, all_patterns_F[j]) for j in range(256)]
```

### 4.4.1 Canonical Byte Emission

The only canonical, spec-compliant way to convert the Epigenome tensor to a byte is:

```python
def tensor_to_output_byte(T, F, G):
    """
    Canonical tensor-to-byte conversion using epigenome pattern matching.
    This is the only spec-compliant method for deriving a byte from the current tensor state.
    """
    key_index = find_closest_pattern_index(T, F)
    return G[key_index]
```

**Why this is foundational:**
- **Epigenome coherence:** The evolving tensor T is always compared against the closure set of 256 canonical patterns.
- **Reversibility & auditability:** Every output byte is traceable to the exact canonical pattern that produced it.
- **Thread/file keys:** The same output_byte stream is reused for encryption, ensuring unity between generation and security.

All code and engines must use this route for byte emission, ensuring epigenome coherence, reversibility, and security.

### 4.5 S4: Intelligence Engine

**Purpose:** Orchestration, file I/O, and thread lifecycle management  
**State Variables:**
- `agent_uuid`: UUID of the current agent
- `agent_secret`: Persistent secret
- `thread_uuid`: UUID of active thread
- `thread_file_key`: 256-byte key for encrypting the current thread
- `M`: Pattern Metadata
- `current_thread_keys`: List of dictionaries for the active thread

#### 4.5.1 Thread Lifecycle Operations

```python
def start_new_thread(parent_uuid=None):
    # 1. Capture Epigenome state
    epigenome_snapshot = inference_engine.T.copy()
    
    # 2. Create new thread
    thread_uuid = create_thread(agent_uuid, parent_uuid, format_uuid)

    # 3. Derive thread file key
    thread_file_key = derive_file_key(
        epigenome_snapshot, agent_uuid, thread_uuid, gene_stateless=0xAA
    )

    # 4. Reset observation log
    current_thread_keys = []
    
    return thread_uuid

def process_and_end_thread(input_stream: bytes):
    # 1. Process the stream
    intermediate_ciphertext, dynamic_keystream = info_engine.process_stream(
        inference_engine, self, input_stream
    )
    
    # 2. End the thread
    end_current_thread(intermediate_ciphertext, dynamic_keystream)

def end_current_thread(intermediate_ciphertext: bytes, dynamic_keystream: bytes):
    # 1. Decrypt intermediate ciphertext
    plaintext = bytearray(len(intermediate_ciphertext))
    for i in range(len(intermediate_ciphertext)):
        plaintext[i] = intermediate_ciphertext[i] ^ dynamic_keystream[i]

    # 2. Re-encrypt with thread key
    final_encrypted_data = bytearray(len(plaintext))
    for i in range(len(plaintext)):
        final_encrypted_data[i] = plaintext[i] ^ thread_file_key[i % 256]

    # 3. Save thread and key
    save_thread(agent_uuid, thread_uuid, final_encrypted_data, len(final_encrypted_data))
    store_thread_key(agent_uuid, thread_uuid, thread_file_key)

def update_learning_state(key_index: int, inference_engine: InferenceEngine):
    # 1. Update pattern metadata
    M["patterns"][key_index]["frequency"] += 1
    M["patterns"][key_index]["last_seen"] = inference_engine.cycle_counter
    
    # 2. Record Gene Key
    gene_key_entry = {
        "cycle": inference_engine.cycle_counter,
        "pattern_index": key_index
    }
    current_thread_keys.append(gene_key_entry)
```

#### 4.5.2 Encode/Decode Operations

```python
def encode(semantic_label: str) -> int | None:
    """Finds pattern index for semantic label"""
    for index, pattern in enumerate(M["patterns"]):
        if pattern.get("translation") == semantic_label:
            return index
    return None

def decode(key_index: int) -> str | None:
    """Finds semantic label for pattern index"""
    return M["patterns"][key_index].get("translation")
```

#### 4.5.3 Intelligent Response Generation

```python
def generate_response_byte() -> int:
    # 1. Get resonance data
    resonances = inference_engine.compute_pattern_resonances(inference_engine.T, inference_engine.F)
    
    # 2. Apply π/2 threshold
    resonant_threshold = np.pi / 2
    resonant_patterns = [j for j in range(256) if resonances[j] < resonant_threshold]
    
    # 3. Handle no resonant patterns
    if len(resonant_patterns) == 0:
        closest_pattern = argmin(resonances)
        resonant_patterns = [closest_pattern]
    
    # 4. Apply contextual weighting
    pattern_weights = []
    for pattern_idx in resonant_patterns:
        # Base weight from usage frequency
        usage_count = M["patterns"][pattern_idx]["frequency"]
        
        # Recency bias
        last_cycle = M["patterns"][pattern_idx]["last_seen"]
        recency_factor = 1.0 if last_cycle is None else 1.0 / (inference_engine.cycle_counter - last_cycle + 1)
        
        # Chirality bias
        left_ops = bin(pattern_idx).count('1') & 1
        chirality_bias = 1.5 if left_ops else 1.0
        
        # Resonance strength
        resonance_strength = 1.0 / (resonances[pattern_idx] + 0.1)
        
        # Combined weight
        weight = usage_count * recency_factor * chirality_bias * resonance_strength
        pattern_weights.append(weight)
    
    # 5. Select pattern
    total_weight = sum(pattern_weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in pattern_weights]
        selected_pattern = weighted_choice(resonant_patterns, normalized_weights)
    else:
        selected_pattern = random.choice(resonant_patterns)
    
    # 6. Get output byte
    output_byte = inference_engine.G[selected_pattern]
    
    return output_byte
```

## 5. Formats & Learning

### 5.1 Gyronorm Formats Specification

The Gyronorm Formats structure serves as the **semantic bridge** connecting the system's internal physics with external semantics:

```json
{
  "format_uuid": "7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d",
  "format_name": "standard_english_v1",
  "cgm_version": "1.0.0",
  "format_version": "1.2.0",
  "stability": "stable",
  "compatibility": {
    "min_cgm_version": "0.9.0",
    "max_cgm_version": "1.0.0",
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
      "namespace": null,
      "translation": null,
      "frequency": 0,
      "last_seen": null,
      "resonance_class": "identity",
      "confidence": 0.0
    }
    /* ... repeated for each of the 256 pattern indices ... */
  ]
}
```

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

### 5.2 Learning Mechanisms

The learning mechanism is a two-fold process:

1. **Implicit/Unconscious Learning:** Continuous, irreversible mutation of the `T` tensor by the input stream
2. **Explicit/Conscious Learning:** Recording and statistical weighting of which `key_index` patterns are triggered

**Curriculum Thread Protocol:**
- The agent may be exposed to "curriculum threads" (structured data streams)
- Learning is fully agnostic: structure emerges from resonance and Hebbian updates
- No semantic labeling or annotation is required
- The entire process is logged and reproducible

**Attention Mechanism:**
- Current state of `T` tensor is an "attended" summary of entire past history
- Pattern selection uses explicit weighting by frequency and recency
- Resonance-based selection provides physics-grounded attention

## 6. Implementation Requirements

### 6.1 Required Helper Functions

```python
# Core tensor operations
def apply_operation(T, bit_index)
def find_closest_pattern_index(T, F)
def gyrodistance(T1, T2)

# Thread lifecycle
def generate_new_uuid()
def derive_file_key(epigenome_snapshot, agent_uuid, thread_uuid, gene_stateless)
def derive_agent_key(agent_uuid, agent_secret)
def encrypt_data(data, key)
def decrypt_data(encrypted_data, key)

# Pattern selection
def weighted_choice(items, weights)
def argmin(values)

# Registry management
def ensure_agent_uuid()
def create_thread(agent_uuid, parent_uuid, format_uuid)
def save_thread(agent_uuid, thread_uuid, ciphertext, size)
def load_thread(agent_uuid, thread_uuid)
def store_thread_key(agent_uuid, thread_uuid, key)
def load_thread_key(agent_uuid, thread_uuid)
```

### 6.2 Critical Implementation Notes

1. **Epigenome Initialization:** Must start from all zeros and apply one cycle with the stateless gene.
2. **Pattern Matching:** Must use gyrodistance with π/2 threshold.
3. **File Encryption:** Two-phase process (dynamic keystream then static thread key).
4. **UUID Registry:** Must use centralized registry to maintain consistency.
5. **Thread Isolation:** Each thread must have its own file key derived from the Epigenome state.
6. **Gene Keys Privacy:** Must encrypt with agent key derived from persistent secret.
7. **Atomic File Operations:** All file writes must be atomic using temporary files and rename.
8. **Registry Consistency:** Registry files must be kept in sync with actual directory contents.

### 6.3 State Variable Metadata Requirements

| Variable | Type | Required Metadata | Notes |
|----------|------|-------------------|-------|
| `T` | float32[4,2,3,2] | Shape, precision | Must preserve tensor structure |
| `F` | float32[256,48] | Pattern indexing | Must maintain 0-255 indexing |
| `G` | uint8[256] | Byte mapping | Must preserve index correspondence |
| `cycle_counter` | integer | Progression | Must maintain sequential integrity |
| `agent_uuid` | string | UUID format | Must persist across restarts |
| `thread_uuid` | string | UUID format | Must be registered in central registry |
| `thread_file_key` | bytes[256] | Key integrity | Must be derived deterministically |
| `current_thread_keys` | array | Observation structure | Must record cycle and pattern_index |