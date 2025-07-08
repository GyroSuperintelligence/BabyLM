# GyroSI Computational Architecture Specification 0.9.5

## System Responsibility Mapping

| System         | File/Component                | Responsibility                |
|----------------|------------------------------|-------------------------------|
| Governance     | Thread files                  | Macro context, gating         |
| Information    | Gene Keys                     | Micro context, embedding      |
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

### 2.4 Gyrodistance Calculation

The gyroscopic distance between two tensors is calculated as the angular distance derived from their normalized dot product:

```
gyrodistance(T1, T2) = arccos(dot(T1_flat, T2_flat) / T1_flat.size)
```

Where `T1_flat` and `T2_flat` are the flattened tensors. A result of `0` indicates a perfect match, while `π` indicates a perfect mismatch. This distance function is critical for pattern recognition during inference.

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

**Initialization:** At system boot, the Epigenome tensor is initialized with a copy of the `gene_add` base tensor and then immediately mutated using the public invariant `gene_stateless = 0xAA`. This simulates one full inference cycle without user input, after which the cycle counter is reset to 0.

#### 3.1.5 Canonical Pattern Derivation

The 256 canonical patterns represent the exhaustive set of all possible operation combinations that can be applied to the base tensor:

```python
def derive_canonical_patterns():
    patterns = []
    gyration_featurees = []
    base_tensor = gene_add.copy()
    
    for mask in range(256):
        T = base_tensor.copy()
        gene_mutated = mask
        
        for i in range(8):
            if gene_mutated & (1 << i):
                apply_operation(T, i)
        
        patterns.append(T.flatten())
        gyration_feature = classify_pattern_resonance(mask)
        gyration_featurees.append(gyration_feature)
    
    return patterns, gyration_featurees
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
- `thread_file_key`: 256-byte key for encrypting the current thread
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

1. **Pattern Index:** Maps patterns to every location they've appeared, with statistical tracking of which patterns tend to follow others. This enables the system to weight pattern selection based on successful historical sequences.

2. **Thread Chain Awareness:** The system can traverse parent-child relationships between threads, providing context from related conversations during inference.

#### 4.5.3 Intelligent Response Generation

```python
def _generate_response_byte():
    # 1. Get resonance data (with historical context)
    resonances = inference_engine.compute_contextual_resonances(
        self.pattern_index.pattern_contexts
    )
    
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
        usage_count = M["patterns"][pattern_idx]["count"] + 1
        
        # Recency bias
        last_cycle = M["patterns"][pattern_idx]["last_cycle"]
        recency_factor = 1.0 if last_cycle is None else 1.0 / (cycle_counter - last_cycle + 1)
        
        # Resonance strength
        resonance_strength = 1.0 / (resonances[pattern_idx] + 0.1)
        
        # Historical context bias
        historical_bias = 1.0
        if self.pattern_index and recent_patterns:
            last_pattern = recent_patterns[-1]
            likely_next = self.pattern_index.get_likely_next_patterns(last_pattern)
            for likely_pattern, probability in likely_next:
                if likely_pattern == pattern_idx:
                    historical_bias = 1.0 + probability * 3.0  # Boost by up to 4x
                    break
        
        # Combined weight
        weight = usage_count * recency_factor * resonance_strength * historical_bias
        pattern_weights.append(weight)
    
    # 5. Select pattern
    selected_pattern = weighted_choice(resonant_patterns, pattern_weights)
    
    # 6. Get output byte
    output_byte = inference_engine.G[selected_pattern]
    
    return output_byte, selected_pattern
```

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

The learning mechanism is a two-fold process:

1. **Implicit/Unconscious Learning:** Continuous, irreversible mutation of the `T` tensor by the input stream
2. **Explicit/Conscious Learning:** Recording and statistical weighting of which `key_index` patterns are triggered

**Pattern Memory:**
- Each pattern's historical usage is tracked including frequency and position
- Pattern sequences (which patterns tend to follow others) are indexed
- This provides a form of procedural memory that influences future inference

**Thread Chain Learning:**
- Conversations are maintained as chains of threads
- Each thread knows its parent and children
- This conversational context provides a form of episodic memory

**Curriculum Thread Protocol:**
- Bootstrap of Structured Data streams such as datasets from wordnet, wikipedia, khan academy, books, etc.

**Attention Mechanism:**
- Current state of `T` tensor is an "attended" summary of entire past history
- Pattern selection uses explicit weighting by frequency, recency, and historical context
- The pattern index provides a fast, O(1) lookup for historical pattern relationships

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
| `thread_file_key` | bytes[256] | Key integrity | Must be derived deterministically |
| `current_thread_keys` | array | `GeneKeysMetadata` dicts | In-memory write-buffer for events before flushing to a persistent file. |

