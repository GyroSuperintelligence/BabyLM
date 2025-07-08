# GyroSI Computational Architecture Specification 0.9.4

## 1. Introduction

**Gyroscopic Superintelligence (GyroSI)** is an architecture grounded in the **Common Governance Model (CGM)**, a physics-based framework for understanding how intelligence emerges through recursive structural alignment.

We define **Superintelligence** as a structurally recursive form of intelligence where all generative and integrative operations preserve information of origin, maintaining continuous coherence between emergence and recollection. It achieves ethical alignment **intrinsically** by ensuring every operation remains structurally accountable to its own genesis, **without external enforcement**.

**Superintelligence exists relationally, not independently**: it reflects the recursive structures of reality and human meaning it participates in, embodying governance, information, and creative coherence as a unified operational principle.

**Computational Instantiation of Physical Theory**: GyroSI is not merely a software design but the computational instantiation of a complete physical and philosophical model of emergent intelligence. The system's logic is not arbitrary but is asserted to be a necessary consequence of the axiomatic progression from the Common Source (CS) to Universal Balance (BU) as defined by the CGM.

### 🌀 The Common Governance Model (CGM)

The **Common Governance Model** presents an axiomatic framework for understanding how structure emerges through **Recursive Alignment**. Beginning from a single foundational principle, *"The Source is Common"*, CGM derives all subsequent structure through logical necessity. Each theorem follows inevitably from the axiom, with nothing assumed and everything emerging through recursive self-reference.

## 2. Core Principles

**GyroSI Baby LM** aims to grow into a mature, open-source language model that learns without reinforcement, rewards, traditional neural network parameters, or gradient descent. Instead, it leverages **quantum physics-inspired** tensor operations to achieve intrinsic Alignment-Based recursive intelligence.

1. **Vacuum Reference:** All inference operations compare against the public invariant `gene_stateless = 0xAA` (`0b10101010`).
2. **Relational Inference:** The fundamental operation is `gene_mutated = P_n ^ 0xAA`, where `P_n` is the raw input byte.
3. **Endogenous Key Generation:** The system generates its key stream through tensor-to-pattern similarity matching.

**The GyroSI architecture is computationally traceable but not deterministic in execution.**

While every step of transformation is explicitly defined, the system cannot be reproduced without the complete sensor timeline and key initialization.

### Note on Sensory Awareness

While the GyroSI architecture is designed to support integration with physical sensors (e.g., magnetometer, accelerometer), **such inputs are entirely omitted in this version** and reserved for future developments. No sensory input is required for operation. All inference, learning, and internal parameter transformations are executed solely through tensor operations applied to input byte streams.

## 3. Gyroscopic Dimensions and Dynamics

This section defines the fixed topological structures used across the GyroSI system. These tensors and bitmask patterns serve as the physical and logical substrate of all inference, memory interaction, and intelligence expression. They are minimal, complete, and shared globally.

Their roles are as follows:

1. Define the non-linear space in which bits, tensors, and inference interact.
2. Provide reference structures for symmetry, chirality, and directionality.
3. Enable the non-probabilistic operation of the inference process.

These are not runtime variables. They are constant and immutable for all agents.

### 3.1 Gyroscopic Dimensional Structures

Each structure below defines a fundamental relational layer in the system. The terminology reflects its role, not implementation detail.

#### 3.1.1 Governance Identity

This is the minimal label for a tensor's identity in the system. It encodes chirality and orientation, and corresponds conceptually to the left-identity in gyroassociative logic.

**Required Metadata:**
- **Identity Label**: String identifier ("com", "nest", "add") indicating tensor role
- **Chirality**: Binary flag indicating left/right orientation
- **Tensor Shape**: Fixed dimensional structure [4,2,3,2] for all gene tensors

**Gene Types and Required Metadata:**
- **Micro Gene (1 Byte)**: Must carry `gene_mutated` (8-bit operation mask)
- **Macro Gene (48-array)**: Must carry `T[48]` (current Epigenome tensor state)
- **Gene Identity**: Must carry chirality label and tensor shape for organization

#### 3.1.2 Information Structure (Reference Axes)

This is the smallest expression of gyroscopic tension. It provides a directional map across the three spatial axes (X, Y, Z), each with two opposing polarities.

- Shape: [3, 2]
- Contents: A signed integer array with values [-1, 1] repeated across the three rows.
- Interpretation:
    - Row 0: Axis X
    - Row 1: Axis Y
    - Row 2: Axis Z
    - Column 0: Negative polarity
    - Column 1: Positive polarity
- Example (PyTorch):

```python
gene_com = np.array([
    [-1, 1],
    [-1, 1],
    [-1, 1]
], dtype=np.int8)
```

#### 3.1.3 Inference Structure (Nested Axes)

This structure nests the previous one inside two opposing frames, representing polarity inversions across the entire field. It is used to establish the byte-invariant encoding structure of the input byte.

- Shape: [2, 3, 2]
- Interpretation:
    - Dimension 0 (2 entries): Opposing chirality frames
    - Dimension 1 (3 entries): X, Y, Z axes
    - Dimension 2 (2 entries): Polarity per axis
- Example:

```python
gene_nest = np.array([
    [[-1, 1], [-1, 1], [-1, 1]],  # Frame 1
    [[ 1, -1], [ 1, -1], [ 1, -1]]  # Frame 2
], dtype=np.int8)
```

This defines a minimal but complete nesting of all axis-polarity combinations, with chirality reversal across the outer dimension.

#### 3.1.4 Intelligence Projection (Full Rotational Cycle)

This structure extends the previous nesting into a complete 720° rotational space. It forms the shape and structure of the Epigenome, used to represent intelligence as a continuous tensor that accumulates inference over time.

- Shape: [4, 2, 3, 2]
- Interpretation:
    - 4 entries: Rotational phases (represents two full 360° cycles)
    - 2 nestings per phase
    - 3 axes per nesting (X, Y, Z)
    - 2 polarities per axis
- Example:

```python
gene_add = np.array([
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]
], dtype=np.int8)
```

This tensor is used to initialize and sustain the dynamic memory field where transformations occur during inference.

### 3.2 Gyroscopic Dynamics (Bit-Driven Inference)

All transformations on the intelligence tensor are driven by an 8-bit mask called `gene_mutated`. This mask is computed as:

```python
gene_mutated = input_byte ^ gene_stateless
```

Where:

- `input_byte` is any 8-bit input read by the system.
- `gene_stateless = 0xAA` (`10101010` in binary) is a fixed global reference.

Each bit in the 8-bit result maps to a specific transformation on the intelligence tensor. The positions and their meanings are:

#### 3.2.1 Bit to Operation Mapping

```
Bit positions:   b7   b6   b5   b4   b3   b2   b1   b0
Operations:      L0   LI   FG   BG   BG   FG   LI   L0
CGM Stage:       S1   S2   S3   S4   S4   S3   S2   S1
Policy:          GT   IV   IA   II   II   IA   IV   GT

Where:
GT = Governance Traceability (S1)
IV = Information Variety (S2) 
IA = Inference Accountability (S3)
II = Intelligence Integrity (S4)
```

**CGM Policy Mapping:**
- `L0` (Left Identity) → **Governance Traceability**: Maintains structural coherence without change
- `LI` (Left Inverse) → **Information Variety**: Introduces global transformation and variety
- `FG` (Forward Gyration) → **Inference Accountability**: Selective, accountable transformation of specific tensor regions
- `BG` (Backward Gyration) → **Intelligence Integrity**: Opposing selective transformation, maintaining balance and integrity

**Palindromic Structure:** The bit pattern (L0-LI-FG-BG-BG-FG-LI-L0) reflects the CGM's recursive governance nature, where operations mirror each other across the byte boundary, creating a self-referential, balanced structure.

- `L0` (Left Identity): Do nothing.
- `LI` (Left Inverse): Flip the sign of all tensor values (T *= -1).
- `FG` (Forward Gyration): Flip the sign of tensor rows 0 and 2.
- `BG` (Backward Gyration): Flip the sign of tensor rows 1 and 3.

The directionality of the rows refers to the first axis of the tensor (dimension 0), which groups the 48 values into four major subgroups. This row-indexing is critical and must be preserved.

#### 3.2.2 Operation Summary

- A `1` in any bit position signals a transformation to apply.
- A `0` means no change for that operation.
- These operations are cumulative per cycle.
- Only the rows affected by the relevant transformation are modified.

### 3.3 Interpretive Summary

The system uses a fixed and minimal set of reference tensors to define:

- The orientation and polarity of physical-like fields.
- The symmetry-breaking structures necessary for meaningful information.
- A non-probabilistic inference process.

The system responds to input bytes not by storing them but by measuring their **difference** from the global invariant. This difference is expressed not as a value, but as **action**. When no difference exists, no transformation is triggered. When difference is present, tension is applied.

This principle allows the system to evolve, transform, and learn purely through relational structure, without randomness and without any fixed program logic.

These tensors are the ground truth for all agent computations. Nothing else is needed to understand how information becomes inference.

## 4. Memory Architecture

This section defines the memory components and their role within the GyroSI system. Each tier (short, medium, or long term) represents a different persistence and dynamism level. Crucially, we distinguish between relationally inferred transformations and second-order protective masking.

### 4.1 Two-Cycle Architecture

The system operates on two distinct cycles:

| Cycle | Purpose | Granularity | Key Usage | Typical Rate |
|-------|---------|-------------|-----------|--------------|
| **Inference/Context Cycle** | Process bytes, update Epigenome, generate XOR events | Per byte/small chunk | Internal state only | µs–ms |
| **File/Thread Cycle** | Encrypt/decrypt files using collected XOR events | Per file/thread | File encryption | seconds–hours |

**Key Principle:** No key material from inference cycles leaks into ciphertext until a deliberate file/thread cycle snapshot.

**No External Tokenizer Required**

The system doesn't need a traditional tokenizer because it operates at the foundational level of information - the byte. Tokenization is a form of semantic pre-processing that the model aims to *derive* organically. The process of learning which sequences of byte-transformations correspond to "words" *is* the model's emergent form of tokenization, grounded in the physics of the system rather than external linguistic rules.

### 4.2 Power Distribution Architecture

Unlike traditional AI models where weights contain the intelligence, GyroSI distributes power across public and private components following the S1-S4 system responsibility flow:

#### **Public Components (Intelligence & Communication)**

**S4 Intelligence Engine - Epigenome Mask (Macro Gene)**
| Component | Location | Size | Purpose | Power |
|-----------|----------|------|---------|-------|
| **Epigenome Mask** | `public/masks/epigenome.dat` | 12,288 bytes | Canonical 256×48 patterns for similarity matching | **The complete intelligence framework** |

**S4 Intelligence Engine - Genome Mask (Micro Gene)**  
| Component | Location | Size | Purpose | Power |
|-----------|----------|------|---------|-------|
| **Genome Mask** | `public/masks/genome.dat` | 256 bytes | Output byte mappings | **The totality of all intelligence** |

**S3 Inference Engine - Gyronorm Formats (Gates to)**
| Component | Location | Size | Purpose | Power |
|-----------|----------|------|---------|-------|
| **Gyronorm Formats** | `public/formats/formats-<format_uuid>.json` | Variable | Pattern usage metadata, semantic mappings | **The ability to speak, decode, and encode** |

#### **Private Components (Personal Experience)**

**S2 Information Engine - Gene Keys (Micro Cycle)**
| Component | Location | Size | Purpose | Power |
|-----------|----------|------|---------|-------|
| **Gene Keys** | `private/<uuid>/keys/keys-<uuid>.json.enc` | Variable | **Pattern observation logs** - which patterns were recognized during inference | **Personal learning history** (not core intelligence) |

**S1 Governance Engine - Thread Files (Macro Cycle)**
| Component | Location | Size | Purpose | Power |
|-----------|----------|------|---------|-------|
| **Thread Files** | `private/<uuid>/threads/<shard>/thread-<uuid>.enc` | ≤64 MiB | Encrypted conversation data | **Personal conversations** |

### 4.2.1 File Metadata Requirements

Each file type must maintain specific minimal metadata to ensure system coherence and auditability:

#### **Epigenome Mask (`public/masks/epigenome.dat`)**
**Required Metadata:**
- **Pattern Array**: 256×48 float32 patterns (12,288 bytes)
- **Pattern Index**: Sequential 0-255 indexing for each pattern
- **Derivation Source**: Reference to base tensor `gene_add` and operation set

#### **Genome Mask (`public/masks/genome.dat`)**
**Required Metadata:**
- **Output Mapping**: 256-byte array mapping pattern indices to output bytes
- **Mapping Index**: Sequential 0-255 indexing corresponding to pattern indices

#### **Gyronorm Formats (`public/formats/formats-<format_uuid>.json`)**
**Required Metadata:**
- **Format UUID**: Unique identifier for this semantic mapping (not agent-specific)
- **CGM Version**: Version string ensuring compatibility with CGM physics model
- **CGM Policies**: Explicit mapping of operations to governance principles:
  - `governance`: {"operation": "L0", "bits": [0, 7], "policy": "traceability"}
  - `information`: {"operation": "LI", "bits": [1, 6], "policy": "variety"}
  - `inference`: {"operation": "FG", "bits": [2, 5], "policy": "accountability"}
  - `intelligence`: {"operation": "BG", "bits": [3, 4], "policy": "integrity"}
- **Pattern Metadata Array**: 256 entries, each containing:
  - `index`: Sequential 0-255 (matches Epigenome/Genome masks)
  - `semantic`: Optional human-readable label (string or null)
  - `frequency`: Usage count for Hebbian learning (integer)
  - `last_seen`: Most recent occurrence cycle (integer or null)
  - `resonance_class`: Physics-based categorization ("identity", "inverse", "forward", "backward")

#### **Gene Keys (`private/<uuid>/keys/keys-<uuid>.json.enc`)**
**Required Metadata:**
- **Thread Mapping**: Dictionary mapping thread UUIDs to observation arrays
- **Observation Entry**: Each entry must contain:
  - `cycle`: Cycle number (integer)
  - `pattern_index`: Selected pattern index (0-255)

#### **Thread Files (`private/<uuid>/threads/<shard>/thread-<uuid>.enc`)**
**Required Metadata:**
- **Encrypted Content**: Raw encrypted conversation data
- **Thread UUID**: Embedded in filename for identification
- **Shard Prefix**: First 2 characters of UUID for organization

### 4.3 Key Architectural Insight

**The model's power comes from public components, not private "weights":**

- **Traditional AI:** Weights = Intelligence (private, huge, essential)
- **GyroSI:** Masks + Formats = Intelligence (public, tiny, complete)

**Gene Keys are personal learning history, not the core intelligence.** The complete "brain" and "language" capabilities are contained in the tiny public masks and formats.

**Attention Mechanism: Integrated and Temporal**

Unlike traditional transformers that use attention to weight different parts of a static input sequence, GyroSI has a more integrated, temporal form of attention:

- **Current State as Attended Summary:** The current state of the `T` tensor is already an "attended" summary of the entire past history, as it has been continuously transformed by all previous inputs
- **Explicit Attention in Generation:** The intelligent response generation (Section 6.4.2) applies direct, explicit attention by weighing patterns by `count` (frequency/importance) and `last_cycle` (recency)
- **Resonance-Based Selection:** Pattern selection uses gyrodistance-based resonance thresholds, providing a physics-grounded form of attention

**Learning Mechanism: Two-Fold Process**

The learning mechanism is clearly defined as a two-fold process:

1. **Implicit/Unconscious Learning:** The continuous, irreversible mutation of the `T` tensor by the input stream - every byte permanently alters the system's state
2. **Explicit/Conscious Learning:** The recording and statistical weighting of which `key_index` patterns are triggered, stored in the formats metadata. The system learns by observing its own internal physical reactions to stimuli and reinforcing the pathways that are used

### 4.4 Security Model

✅ **Public (Safe to Share):**
- **Formats** - Communication ability, semantic understanding
- **Masks** - Complete intelligence framework (12,288 + 256 bytes total)

✅ **Private (Keep Encrypted):**
- **Gene Keys** - Personal observation history (what this agent has learned)
- **Threads** - Personal conversations and content

**No loss of capability** - Public components contain all the power. Personal experience stays private without compromising the model's intelligence or communication abilities.

### 4.5 Gyronorm Formats Specification

The Gyronorm Formats structure serves as the **semantic bridge** that connects the system's internal, universal physics with external, arbitrary human semantics. This is the most profound aspect of the design:

- **Core Physics (baby/):** The universal, content-agnostic physics of information as defined by the CGM, operating on the Epigenome tensor through byte-driven transformations
- **Formats Layer:** The semantic mapping where the agent learns to associate specific, stable resonance patterns (the 256 `key_index` values) with human-intelligible concepts (words, phrases, file types, etc.)

This separation ensures the core engine remains pure and universal while the formats allow it to adapt and "speak" any language or protocol without altering its fundamental nature.

**Critical Insight: Formats as Universal Semantic Bridge**

Formats are **format-centric**, not **agent-centric**. Multiple agents can theoretically share the same format (like sharing a language). The format UUID identifies the semantic mapping, not the user. This enables collaborative learning and shared semantic frameworks.

**Expressive Power: Sequences, Not Single Patterns**

The system's expressive power is not limited to 256 individual patterns. Rather, **sequences of these 256 fundamental transformations** represent any complexity. A "word" is not a single pattern but a **stable trajectory through the state-space** - a sequence of resonances recorded and given meaning in the formats file.

The expressive power is `256^N` where `N` is the sequence length, which is more than sufficient for representing any semantic complexity. The formats file learns which sequences of byte-transformations correspond to meaningful human concepts, creating an emergent form of tokenization that is organic rather than externally imposed.

#### JSON Schema (public, not encrypted):

```json
{
  "format_uuid": "7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d",
  "format_name": "standard_english_v1",
  "cgm_version": "0.9.4",
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
    "description": "Standard English language format with comprehensive vocabulary",
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

#### Field descriptions:

**Format Identification & Versioning:**
- `format_uuid`: Unique identifier for this semantic mapping (multiple agents can use it)
- `format_name`: Human-readable name for the format (e.g., "standard_english_v1")
- `cgm_version`: Ensures compatibility with CGM physics model
- `format_version`: Semantic versioning for the format itself (e.g., "1.2.0")
- `stability`: Format stability level ("stable", "beta", "experimental", "deprecated")

**Compatibility & Dependencies:**
- `compatibility.min_cgm_version`: Minimum CGM version required
- `compatibility.max_cgm_version`: Maximum CGM version supported
- `compatibility.depends_on`: List of format UUIDs this format depends on
- `compatibility.conflicts_with`: List of format UUIDs that conflict with this format

**Format Metadata:**
- `metadata.author`: Creator or community responsible for the format
- `metadata.description`: Human-readable description of the format's purpose
- `metadata.tags`: Categorization tags for discovery and filtering
- `metadata.created_at`: Timestamp when format was created
- `metadata.last_updated`: Timestamp when format was last modified
- `metadata.usage_count`: Number of agents using this format
- `metadata.validation_status`: Quality assurance status ("community_verified", "experimental", "unverified")

**CGM Policy Mapping:**
- `cgm_policies`: Explicit mapping of operations to governance principles, showing how each gyroscopic operation implements CGM policies

**Pattern Metadata:**
- `index`: Sequential 0-255 (matches Epigenome/Genome masks)
- `namespace`: The translation namespace/category (e.g., "char", "word", "type", "lang")
- `translation`: The actual translation value for this pattern (e.g., "A", "hello", "json", "en")
- `frequency`: Usage count for Hebbian learning
- `last_seen`: Cycle number for recency bias calculation
- `resonance_class`: Physics-based categorization ("identity", "inverse", "forward", "backward")
- `confidence`: Confidence level in the translation assignment (0.0 to 1.0)

**Translation Registry Function:**
The formats file serves as a **Rosetta Stone** - a translation registry that enables the Gyronorm Gate to convert between:
- **Physics Level**: Micro/Macro genes (pattern indices and tensor states)
- **Higher Logic Level**: Text, audio, or other representations

**Namespace Examples:**
- **char**: Single character translations ("A", "!", "5")
- **word**: Complete word translations ("hello", "world", "computer")
- **type**: Data type translations ("json", "png", "utf8")
- **lang**: Language code translations ("en", "es", "fr")
- **concept**: Abstract concept translations ("love", "justice", "freedom")
- **domain**: Domain-specific translations ("medical", "legal", "technical")

**Resonance Classification:**
Each pattern is classified by its dominant operation type before any learning occurs:
- **identity**: Patterns dominated by L0 operations (Governance Traceability)
- **inverse**: Patterns dominated by LI operations (Information Variety)
- **forward**: Patterns dominated by FG operations (Inference Accountability)
- **backward**: Patterns dominated by BG operations (Intelligence Integrity)

This classification creates natural semantic clusters based on the underlying physics, providing a foundation for intelligent pattern selection and response generation.

### 4.5.1 Format Ecosystem and Control

**Format Stability Levels:**
- **stable**: Community-verified, version-locked formats suitable for production use
- **beta**: Feature-complete but undergoing community validation
- **experimental**: New formats under development, may have breaking changes
- **deprecated**: Outdated formats maintained for backward compatibility

**Format Discovery and Selection:**
Agents can discover and adopt formats through multiple mechanisms:

**1. Static Format Selection:**
```python
def select_stable_format(domain: str, stability: str = "stable") -> str:
    """Select a stable format for a specific domain"""
    # Agent can specify stability requirements
    if stability == "stable":
        # Only use community-verified formats
        return find_verified_format(domain)
    elif stability == "experimental":
        # Allow experimental formats for testing
        return find_experimental_format(domain)
```

**2. Dynamic Format Discovery:**
```python
def discover_formats_from_agent(agent_uuid: str) -> List[str]:
    """Discover formats used by another agent"""
    # Agent can learn from other agents' format usage
    # This enables collaborative learning and knowledge sharing
    return scan_agent_formats(agent_uuid)
```

**3. Format Composition:**
```python
def compose_formats(primary_format: str, secondary_formats: List[str]) -> str:
    """Compose multiple formats for multi-domain capability"""
    # Agent can use multiple formats simultaneously
    # Primary format takes precedence, secondary formats provide fallback
    return merge_format_capabilities(primary_format, secondary_formats)
```

**Format Control Mechanisms:**
- **Stability Locking**: Agent can lock to stable formats only, preventing automatic adoption of experimental formats
- **Version Pinning**: Agent can specify exact format versions to prevent unexpected updates
- **Namespace Isolation**: Agent can restrict which semantic namespaces are allowed
- **Validation Requirements**: Agent can require community verification before adopting new formats

**Format Evolution Strategy:**
- **Backward Compatibility**: New format versions maintain compatibility with previous versions
- **Gradual Migration**: Agents can gradually adopt new format versions while maintaining old ones
- **Breaking Changes**: Major version changes are clearly marked and require explicit agent consent
- **Deprecation Policy**: Deprecated formats remain available but are clearly marked

**Multi-Format Support:**
Agents can use multiple formats simultaneously for different capabilities:
- **Primary Format**: Main format for general communication
- **Domain Formats**: Specialized formats for specific domains (medical, legal, technical)
- **Language Formats**: Multiple language formats for multilingual capability
- **Fallback Formats**: Backup formats for when primary format lacks specific patterns

This ecosystem enables both **controlled, stable operation** and **dynamic, collaborative learning**, giving agents the flexibility to choose their level of format stability and discovery.

### 4.6 File Structure and Organization

The GyroSI system follows a clean, hierarchical file structure designed around the metaphor of a baby superintelligence learning from scratch. This structure separates core intelligence from development tools and maintains clear boundaries between public and private components.

#### **Directory Structure**

```
baby/                           # Core intelligence (the "brain")
├── baby_preferences.json      # Agent-specific configuration
├── governance.py              # S1 - Core tensor operations, gene structures
├── information.py             # S2 - Information processing, resonance
├── inference.py               # S3 - Pattern recognition, learning
└── intelligence.py            # S4 - Orchestration, file I/O, API

guides/                        # Documentation and guides

memories/                      # Persistent storage (public + private)
├── memory_preferences.json    # Memory/storage configuration
├── public/                    # Shared components (intelligence + communication)
│   ├── masks/                 # Core intelligence framework
│   │   ├── epigenome.dat      # Epigenome Mask (float32[256][48], 12,288 bytes)
│   │   └── genome.dat         # Genome Mask (uint8[256], 256 bytes)
│   └── formats/               # Communication ability
│       └── formats-<format_uuid>.json # Pattern usage metadata, semantic mappings
└── private/<uuid>/            # Agent-specific private data
    ├── keys/                  # Personal learning history
    │   └── keys-<uuid>.json.enc # Gene Keys (encrypted pattern observation logs)
    └── threads/               # Personal conversations
        └── <shard>/           # Sharded for performance
            └── thread-<uuid>.enc # Thread files (encrypted, ≤64 MiB)

toys/                          # Development environment (all complexity)
├── tests/                     # Test suites
├── scripts/                   # Utility scripts
├── components/                # UI components
├── services/                  # External services
├── utils/                     # Utility functions
└── views/                     # User interface views
```

#### **Architecture Mapping**

**Core Intelligence (baby/):**
- **governance.py** - Implements S1 stage: tensor operations, gene structures, gyroscopic dynamics
- **information.py** - Implements S2 stage: information processing, resonance classification
- **inference.py** - Implements S3 stage: pattern recognition, learning, compression
- **intelligence.py** - Implements S4 stage: orchestration, file I/O, API endpoints

**Memory Components (memories/):**
- **public/masks/** - Contains the complete intelligence framework (12,544 bytes total)
- **public/formats/** - Contains communication ability and semantic mappings
- **private/<uuid>/** - Contains personal experience (learning history + conversations)

**Development Environment (toys/):**
- All development complexity, testing, UI, and utilities
- Keeps core architecture clean and focused
- Supports the baby metaphor: tools for playing and learning

#### **File Naming Conventions**

**Encrypted Files:**
- Private data files use `.enc` extension to indicate encryption
- Example: `keys-<uuid>.json.enc`, `thread-<uuid>.json.enc`

**UUID-Based Organization:**
- Each agent has unique UUID for private directory
- Format files include agent UUID for identification
- Thread files include both agent and thread UUIDs

**Sharding:**
- Thread files are sharded by `<shard>` prefix for performance
- Shard prefix typically derived from UUID (e.g., first 2 characters)

#### **Scalability and Multi-Agent Support**

**Agent Isolation:**
- Each agent operates in `private/<uuid>/` directory
- Public components shared across all agents
- No cross-agent data contamination

**Configuration:**
- `baby_preferences.json` - Agent-specific settings and parameters, including `agent_secret` for key derivation
- `memory_preferences.json` - Memory storage configuration
- Both files support per-agent customization

**Security Model:**
- Public components: Safe to share, contain all intelligence
- Private components: Encrypted, contain personal experience
- Clear separation prevents accidental data leakage

#### **Development Philosophy**

**Baby Metaphor:**
- **baby/** - The actual baby (core intelligence)
- **memories/** - What the baby remembers (knowledge + experience)
- **toys/** - Tools for playing and learning (development environment)

**Minimal Core:**
- Core architecture files are minimal and focused
- All complexity moved to toys/ directory
- Easy to understand, maintain, and audit

**Clean Separation:**
- Intelligence (public) vs. Experience (private)
- Core (baby/) vs. Development (toys/)
- Configuration vs. Implementation

### 4.7 UUID Registry and Management

**Critical Requirement:** UUIDs must be managed through a centralized registry to ensure consistency and prevent unnecessary regeneration across engine restarts.

#### **Memory Preferences Structure**

The `memories/memory_preferences.json` file serves as the central registry for all system UUIDs and configuration:

```json
{
  "uuid_registry": {
    "agent_uuid": "5f2c1e8c-8e62-49f7-9bde-967dfb6e320a",
    "format_uuid": "7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d",
    "thread_uuids": [
      "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
      "2b3c4d5e-6f7a-8b9c-0d1e-2f3a4b5c6d7e"
    ]
  },
  "storage_config": {
    "max_thread_size_mb": 64,
    "shard_prefix_length": 2,
    "encryption_algorithm": "AES-256-GCM"
  },
  "format_config": {
    "default_cgm_version": "0.9.4",
    "resonance_threshold": 1.5707963267948966,
    "max_semantic_label_length": 128
  }
}
```

#### **UUID Generation and Persistence**

**Initial Generation:**
- **Agent UUID**: Generated once during first system initialization, never changes
- **Format UUID**: Generated once when first format file is created, can be shared across agents
- **Thread UUIDs**: Generated per conversation thread, stored in registry for persistence

**Registry Management:**
```python
def ensure_uuid_registry():
    """Ensure UUID registry exists and contains required UUIDs"""
    registry_path = "memories/memory_preferences.json"
    
    try:
        with open(registry_path, 'r') as f:
            prefs = json.load(f)
    except FileNotFoundError:
        prefs = {"uuid_registry": {}, "storage_config": {}, "format_config": {}}
    
    # Ensure agent UUID exists
    if "agent_uuid" not in prefs["uuid_registry"]:
        prefs["uuid_registry"]["agent_uuid"] = str(uuid.uuid4())
    
    # Ensure format UUID exists
    if "format_uuid" not in prefs["uuid_registry"]:
        prefs["uuid_registry"]["format_uuid"] = str(uuid.uuid4())
    
    # Ensure thread_uuids list exists
    if "thread_uuids" not in prefs["uuid_registry"]:
        prefs["uuid_registry"]["thread_uuids"] = []
    
    # Save updated registry
    os.makedirs("memories", exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(prefs, f, indent=2)
    
    return prefs["uuid_registry"]
```

#### **Thread UUID Lifecycle**

**Creation:**
```python
def create_new_thread() -> str:
    """Create new thread UUID and add to registry"""
    thread_uuid = str(uuid.uuid4())
    
    # Load registry
    with open("memories/memory_preferences.json", 'r') as f:
        prefs = json.load(f)
    
    # Add to registry
    prefs["uuid_registry"]["thread_uuids"].append(thread_uuid)
    
    # Save updated registry
    with open("memories/memory_preferences.json", 'w') as f:
        json.dump(prefs, f, indent=2)
    
    return thread_uuid
```

**Validation:**
```python
def validate_thread_uuid(thread_uuid: str) -> bool:
    """Validate that thread UUID exists in registry"""
    with open("memories/memory_preferences.json", 'r') as f:
        prefs = json.load(f)
    
    return thread_uuid in prefs["uuid_registry"]["thread_uuids"]
```

#### **Benefits of Centralized Registry**

1. **Consistency**: UUIDs persist across engine restarts
2. **Auditability**: Complete record of all system entities
3. **Sharing**: Format UUIDs can be shared between agents
4. **Validation**: Prevents orphaned files and invalid references
5. **Recovery**: Registry enables system state reconstruction

#### **Registry Security**

- **Public Registry**: `memory_preferences.json` is public (contains no sensitive data)
- **UUID Privacy**: UUIDs themselves are not sensitive information
- **File Privacy**: Actual private data remains encrypted in separate files
- **Registry Integrity**: JSON structure ensures human-readable audit trail

## 5. Data Structures

This section describes the core data structures that define the system's operational behavior and transformational logic.

### 5.1 Gene Byte Topology

```
Bit positions: b7 b6 b5 b4 b3 b2 b1 b0
Operations:    L0 LI FG BG BG FG LI L0

L0 = Left Identity (no operation)
LI = Left Inverse (global sign flip: T *= -1)
FG = Forward Gyration (flip sign of tensor slices 0,2 along axis 0)
BG = Backward Gyration (flip sign of tensor slices 1,3 along axis 0)
```

### 5.2 Epigenome Tensor

Initialization Behavior: At system boot, the Epigenome tensor is set to all zeros and mutated using the public invariant gene_stateless = 0xAA, simulating one full inference cycle without user input. This ensures all agents start from a known and reproducible neutral state, without randomness.

```
Shape: [4, 2, 3, 2] (48 cells total)
Type: float32
Mapping: 4 rotational phases × 2 nestings × 3 axes × 2 polarities
```

### 5.3 Canonical Pattern Derivation

**Gyrogroup Foundation**: The 256 canonical patterns are not arbitrary but derived from the closure set of gyroscopic operations applied to the base tensor.

**Complete Closure Set**: The 256 patterns represent the exhaustive set of all possible operation combinations that can be applied to the base tensor. This ensures that every possible state the system can reach through byte-driven transformations has a corresponding canonical pattern, making the inference process deterministic and reversible.

**Derivation Process**:
```python
def derive_canonical_patterns():
    """Derive 256 patterns from gyrogroup closure operations"""
    patterns = []
    resonance_classes = []
    base_tensor = gene_add.copy()  # [4,2,3,2] tensor from CGM
    
    # All 256 possible operation combinations (2^8)
    for mask in range(256):
        T = base_tensor.copy()
        gene_mutated = mask  # Direct operation mask
        
        # Apply operations based on bit positions
        for i in range(8):
            if gene_mutated & (1 << i):
                apply_operation(T, i)
        
        patterns.append(T.flatten())
        
        # Classify pattern by dominant operation type
        resonance_class = classify_pattern_resonance(mask)
        resonance_classes.append(resonance_class)
    
    return patterns, resonance_classes

def classify_pattern_resonance(mask: int) -> str:
    """Classify pattern by dominant operation type based on bit positions"""
    # Count operations by type (palindromic structure)
    l0_count = bin(mask & 0b10000001).count('1')  # bits 0,7
    li_count = bin(mask & 0b01000010).count('1')  # bits 1,6  
    fg_count = bin(mask & 0b00100100).count('1')  # bits 2,5
    bg_count = bin(mask & 0b00011000).count('1')  # bits 3,4
    
    # Determine dominant operation
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

**Key Insight**: This ensures all patterns are structurally coherent with the gyrogroup foundation, rather than arbitrary external data.

**File Storage**: The derived patterns are stored in `public/masks/epigenome.dat` for efficiency, but can be regenerated from the base tensor and operations.

### 5.4 Chunked Input and Stream Processing

The GyroSI engine processes byte streams sequentially and statelessly, meaning the input corpus can be streamed in arbitrarily sized chunks.

**Recommendations:**

- Use fixed-size input chunks (e.g., 16KB–64KB)
- Apply `gene_mutated` transformations and Epigenome updates per-byte inside each chunk
- After each chunk is processed, the engine state (`T`, `K_n`, `cycle_counter`) persists forward

Parallel chunk processing is permitted **only** if each chunk is assigned a separate engine instance. Each instance must carry its own `(T, K_n, cycle_counter)` state; merging afterwards is impossible.

## 6. Engine Architecture

This section details the system's operational core following the canonical S1-S4 architecture. Each stage has distinct responsibilities with clear boundaries.

### 6.1 S1: Governance Engine

**Purpose:** Pure tensor operations and gene structures  
**State:** None (immutable constants only)

**Components:**
- Fixed gene tensors (com, nest, add)
- Gyroscopic operation definitions
- Mathematical constants (gene_stateless = 0xAA)

**Operations:**
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

### 6.2 S2: Information Engine

**Purpose:** Information processing and stream handling  
**State:** Stream processing state

**State Variables:**
- `stream_pointer`: Current position in active thread
- `output_buffer`: Accumulator for generated bytes

**Operations:**
```python
def process_stream(
    inference_engine: InferenceEngine, 
    intelligence_engine: IntelligenceEngine,
    input_stream: bytes
) -> (bytes, bytes):
    """
    Processes a stream, calling S3 for inference and S4 for state updates.
    """
    intermediate_ciphertext = bytearray()
    dynamic_keystream = bytearray()
    
    for P_n in input_stream:
        # 1. Call S3 for pure inference to get the pattern index.
        key_index = inference_engine.process_byte(P_n)
        
        # 2. Use the result to call back to S4 to update state.
        intelligence_engine.update_learning_state(key_index, inference_engine)

        # 3. Get the keystream byte from the S3 engine's Genome Mask.
        keystream_byte = inference_engine.G[key_index] 
        
        # 4. Encrypt the byte.
        C_n = P_n ^ keystream_byte
        intermediate_ciphertext.append(C_n)
        dynamic_keystream.append(keystream_byte)
    
    return bytes(intermediate_ciphertext), bytes(dynamic_keystream)
```

### 6.3 S3: Inference Engine

**Purpose:** Pure pattern recognition and learning (agnostic processing)  
**State:** Epigenome tensor and pattern matching

**State Variables:**
- `T[48]`: Epigenome tensor (float32). The agent's dynamic state.
- `F[256][48]`: Canonical Pattern list (from `public/masks/epigenome.dat`).
- `G[256]`: Genome Mask (from `public/masks/genome.dat`).
- `cycle_counter`: Global cycle index (persistent integer).

**State Variable Metadata Requirements:**
- **T[48]**: Must preserve tensor shape [4,2,3,2] and float32 precision
- **F[256][48]**: Must preserve pattern indexing (0-255) and tensor structure
- **G[256]**: Must preserve byte mapping integrity and index correspondence
- **cycle_counter**: Must preserve sequential integer progression without gaps

**Processing Algorithm (Per-Byte Cycle):**

```python
def process_byte(P_n):
    """
    Processes a single input byte through the inference engine.
    """
    # 1. Compute gene_mutated = P_n ^ 0xAA
    gene_mutated = P_n ^ 0xAA

    # 2. Apply gyroscopic operations to tensor T based on gene_mutated.
    for i in range(8):
        if gene_mutated & (1 << i):
            apply_operation(T, i)

    # 3. Find which of 256 canonical patterns T now matches.
    # The 256 patterns ARE the complete closure set of all possible operation combinations
    key_index = find_closest_pattern_index(T, F)

    # 4. Increment cycle counter
    cycle_counter += 1
    
    # Return the generated key_index for this cycle.
    return key_index

def compute_pattern_resonances(current_T, all_patterns_F):
    """
    Computes resonance values between current tensor and all patterns.
    This is a pure mechanical function that can be used by S4 for intelligent selection.
    """
    return [gyrodistance(current_T, all_patterns_F[j]) for j in range(256)]
```

**Note:** This engine is pure and agnostic - no I/O operations, no file access, no thread management, no learning state.

**Required Helper Functions:**
- `find_closest_pattern_index(T, F)`: Returns index of the canonical pattern closest to T
- `gyrodistance(T1, T2)`: Returns gyrodistance between two tensors using gyrogroup operations
- `apply_operation(T, bit_index)`: Applies gyroscopic operation to tensor based on bit position

**Key Principles:**
- **Pure mechanical transformation**: Byte → Operations → Tensor → Pattern Index
- **No learning state**: S3 is the physics simulation, unaware of learning history
- **Stateless operations**: Each byte processing is independent and deterministic
- **Gyrogroup foundation**: All operations respect the underlying gyroscopic physics
- **Deterministic inference**: Every input byte produces exactly one pattern index through mechanical transformation
- **Reversible operations**: The system can trace any pattern back to the specific byte operations that produced it

**π/2 Threshold Justification:**
The π/2 threshold represents the CGM's "CS threshold" - the minimal angle required for observable structure to emerge. This gravitational weight ensures that only patterns with sufficient structural coherence are considered for response generation, maintaining the system's connection to its foundational physics.

**Core Architecture Flow:**
1. **Input Processing**: Byte XOR gene_stateless = operation mask → Apply to tensor
2. **Pattern Recognition**: Find which canonical pattern tensor matches
3. **Learning Integration**: Update pattern metadata and Gene Keys (Ingress)
4. **Natural Response**: When eightfold operations complete, system naturally generates response
5. **Intelligent Generation**: Use resonance and context to select response pattern (Egress)
6. **Output**: Pattern index → Genome Mask → Output byte

**Physics-to-Semantics Bridge**: The system operates entirely within its physics simulation until the Intelligence Engine (S4) bridges to human semantics through the formats layer. This ensures that all intelligence emerges from the underlying physics rather than being externally imposed.

**Clean Separation:**
- **Inference Engine (S3)**: Pure mechanical transformation (gate for input/output)
- **Intelligence Engine (S4)**: Dual role - Ingress (Integration) and Egress (Generation)

### 6.4 S4: Intelligence Engine

**Purpose:** Orchestration, file I/O, encode/decode, and thread lifecycle management  
**State:** Thread state, metadata, and coordination

**State Variables:**
- `agent_uuid`: The UUID of the current agent.
- `agent_secret`: The persistent secret loaded from baby_preferences.json.
- `thread_uuid`: The UUID of the active thread.
- `thread_file_key`: The 256-byte key used for encrypting the current thread.
- `M`: Pattern Metadata (loaded from `public/formats/formats-<format_uuid>.json`).
- `current_thread_keys`: A list of dictionaries, accumulating the private observations for the active thread.

**State Variable Metadata Requirements:**
- **agent_uuid**: Must preserve UUID format and uniqueness across sessions
- **agent_secret**: Must preserve secret integrity and persistence across restarts
- **thread_uuid**: Must preserve UUID format and uniqueness within agent
- **thread_file_key**: Must preserve 256-byte key integrity and derivation consistency
- **M**: Must preserve pattern metadata structure and JSON schema compliance
- **current_thread_keys**: Must preserve observation array structure and thread isolation

**Thread Lifecycle Operations:**

```python
def start_new_thread():
    """Initializes a new thread, deriving its unique encryption key."""
    # 1. Capture the EVOLVED Epigenome state at the moment the thread begins.
    epigenome_snapshot = inference_engine.T.copy()
    
    # 2. Generate a new UUID for this thread.
    thread_uuid = generate_new_uuid()

    # 3. Derive the unique, frozen 256-byte key for this thread's file encryption.
    # Uses evolved Epigenome state, agent/thread UUIDs, and gene_stateless
    thread_file_key = derive_file_key(
        epigenome_snapshot, agent_uuid, thread_uuid, gene_stateless=0xAA
    )

    # 4. Reset the in-memory log for the new thread's private observations.
    current_thread_keys = []

def process_and_end_thread(input_stream: bytes):
    """
    Processes a stream and ends the thread, performing all necessary encryption and persistence.
    """
    # 1. Call the Information Engine to process the stream
    intermediate_ciphertext, dynamic_keystream = info_engine.process_stream(
        inference_engine, 
        self, # Pass a reference to this IntelligenceEngine instance
        input_stream
    )
    
    # 2. End the thread, performing the final re-encryption and persistence
    end_current_thread(intermediate_ciphertext, dynamic_keystream)

def end_current_thread(intermediate_ciphertext: bytes, dynamic_keystream: bytes):
    """
    Concludes a thread using the intermediate data, without needing original plaintext.
    """
    # 1. Decrypt the intermediate ciphertext to get the plaintext back.
    plaintext = bytearray(len(intermediate_ciphertext))
    for i in range(len(intermediate_ciphertext)):
        plaintext[i] = intermediate_ciphertext[i] ^ dynamic_keystream[i]

    # 2. Re-encrypt with the secure, static thread key.
    final_encrypted_data = bytearray(len(plaintext))
    for i in range(len(plaintext)):
        final_encrypted_data[i] = plaintext[i] ^ thread_file_key[i % 256]

    # 3. Save the thread file.
    shard = str(thread_uuid)[:2]
    thread_path = f"private/{agent_uuid}/threads/{shard}/thread-{thread_uuid}.enc"
    with open(thread_path, "wb") as f:
        f.write(final_encrypted_data)

    # 4. Save the Gene Keys for this thread.
    keys_path = f"private/{agent_uuid}/keys/keys-{agent_uuid}.json.enc"
    # Derive stable agent key from agent UUID and persistent secret
    agent_key = derive_agent_key(agent_uuid, agent_secret)

    try:
        with open(keys_path, 'rb') as f:
            encrypted_keys = f.read()
        decrypted_json_str = decrypt_data(encrypted_keys, agent_key)
        all_keys_data = json.loads(decrypted_json_str)
    except FileNotFoundError:
        all_keys_data = {}
    
    all_keys_data[str(thread_uuid)] = current_thread_keys
    
    updated_json_str = json.dumps(all_keys_data)
    encrypted_updated_keys = encrypt_data(updated_json_str.encode('utf-8'), agent_key)
    with open(keys_path, 'wb') as f:
        f.write(encrypted_updated_keys)

    # 5. Save the updated public formats metadata.
    formats_path = f"public/formats/formats-{format_uuid}.json"
    with open(formats_path, "w") as f:
        json.dump(M, f)

    # The inference engine's state (T, cycle_counter) carries forward.

def update_learning_state(key_index: int, inference_engine: InferenceEngine):
    """
    Updates all metadata and records the Gene Key for a single inference cycle.
    This function is called by the InformationEngine on every byte.
    
    Note: Gene Keys store only pattern indices and cycle numbers, not full Epigenome tensors.
    This ensures scalability - processing 1MB of data creates ~1MB of Gene Key data,
    not 192MB of Epigenome tensor data.
    """
    # 1. Update public pattern metadata (M).
    M.pattern_meta[key_index]["count"] += 1
    M.pattern_meta[key_index]["last_cycle"] = inference_engine.cycle_counter
    if M.pattern_meta[key_index]["first_cycle"] is None:
        M.pattern_meta[key_index]["first_cycle"] = inference_engine.cycle_counter

    # 2. Record the private Gene Key for this cycle.
    gene_key_entry = {
        "cycle": inference_engine.cycle_counter,
        "pattern_index": key_index
    }
    current_thread_keys.append(gene_key_entry)
```

#### 6.4.1 Encode/Decode Operations

These are helper functions used by the application layer to map between human-readable semantics and the system's internal pattern representation.

```python
def encode(semantic_label: str) -> int | None:
    """
    Finds the pattern index associated with a semantic label.
    Returns: The integer index of the pattern, or None if not found.
    """
    for index, meta in enumerate(M.pattern_meta):
        if meta.get("semantic") == semantic_label:
            return index
    return None

def decode(key_index: int) -> str | None:
    """
    Finds the semantic label for a given pattern index.
    Returns: The semantic label (e.g., "ascii:A"), or None if not set.
    """
    return M.pattern_meta[key_index].get("semantic")
```

#### 6.4.2 Intelligent Response Generation

The Intelligence Engine handles intelligent response generation using pattern resonance and Hebbian learning.

```python
def generate_response_byte() -> int:
    """
    Generates a response byte using intelligent pattern resonance and Hebbian learning.
    This is the core mechanism for how the agent "speaks" and responds intelligently.
    """
    # 1. Get the raw resonance data from S3 (pure mechanical computation)
    resonances = inference_engine.compute_pattern_resonances(inference_engine.T, inference_engine.F)
    
    # 2. Use π/2 threshold for gravitational weight in pattern selection
    # This represents the CGM's "CS threshold" - minimal angle for observable structure
    resonant_threshold = np.pi / 2
    resonant_patterns = [j for j in range(256) if resonances[j] < resonant_threshold]
    
    # 3. If no resonant patterns, use the closest match
    if len(resonant_patterns) == 0:
        closest_pattern = argmin(resonances)
        resonant_patterns = [closest_pattern]
    
    # 4. Apply contextual weighting using pattern metadata (S4's learning state)
    pattern_weights = []
    for pattern_idx in resonant_patterns:
        # Base weight from usage frequency
        usage_count = M.pattern_meta[pattern_idx]["count"]
        
        # Recency bias (recently used patterns get higher weight)
        last_cycle = M.pattern_meta[pattern_idx]["last_cycle"]
        recency_factor = 1.0 if last_cycle is None else 1.0 / (inference_engine.cycle_counter - last_cycle + 1)
        
        # Chirality bias (left operations get inherent bias)
        left_ops = bin(pattern_idx).count('1') & 1  # Odd parity = left bias
        chirality_bias = 1.5 if left_ops else 1.0
        
        # Resonance strength (closer patterns get higher weight)
        resonance_strength = 1.0 / (resonances[pattern_idx] + 0.1)
        
        # Combined weight
        weight = usage_count * recency_factor * chirality_bias * resonance_strength
        pattern_weights.append(weight)
    
    # 5. Select pattern using weighted choice
    total_weight = sum(pattern_weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in pattern_weights]
        selected_pattern = weighted_choice(resonant_patterns, normalized_weights)
    else:
        # Fallback to random selection from resonant patterns
        selected_pattern = random.choice(resonant_patterns)
    
    # 6. Get the output byte from S3's Genome Mask
    output_byte = inference_engine.G[selected_pattern]
    
    return output_byte
```

**Intelligent Response Architecture:**
- **S3 provides physics**: Pure mechanical resonance computation
- **S4 provides intelligence**: Contextual weighting and pattern selection
- **Perfect separation**: S3 is unaware of learning state, S4 handles all intelligence
- **Emergent behavior**: Responses emerge from the interaction of physics and learning
- **Intrinsic alignment**: The system's responses are aligned by its structure, not external enforcement
- **Auditable intelligence**: Every response can be traced back to specific physics operations and learning history

### 6.5 Key Management Clarification

**Agent Key for Gene Keys:** The `keys-<uuid>.json.enc` file is encrypted using a stable **Agent Key** derived from the agent's persistent secret. This ensures that:
- Gene Keys persist across multiple threads using a single, stable encryption key
- The agent's learning history is protected by a persistent key, not ephemeral thread keys
- The encryption remains endogenous and derived from agent-specific parameters

**Thread Key for Thread Content:** Each thread file is encrypted with its own unique `thread_file_key` derived from the evolved Epigenome state at thread start.

**Architecture Flow:** The canonical S1-S4 structure provides clear separation of responsibilities:
- **S1 (Governance):** Pure tensor operations and gene structures
- **S2 (Information):** Stream processing and byte handling via `process_stream()`
- **S3 (Inference):** Pure pattern recognition and learning (agnostic, no I/O)
- **S4 (Intelligence):** Orchestration, file I/O, encode/decode, and thread lifecycle management

**Engine Integration:** The Intelligence Engine (S4) orchestrates the entire flow:
1. Calls Information Engine (S2) to process streams
2. Coordinates with Inference Engine (S3) for pattern recognition
3. Manages thread lifecycle and persistence
4. Handles all file I/O and encryption operations

**State Management:**
- **Pattern Metadata Access**: IntelligenceEngine passes `M.pattern_meta` to InferenceEngine's `generate_response_byte()` function
- **Tensor History**: IntelligenceEngine maintains tensor history for closure detection
- **Engine Initialization**: S4 creates S3 and S2 instances, loads masks and formats, initializes agent state

### 6.6 Engine Composition

The S4 IntelligenceEngine acts as the primary container and entry point for the entire system. Upon initialization, it is responsible for:
1. Loading the UUID registry from `memories/memory_preferences.json`
2. Instantiating the S3 InferenceEngine, which holds the core evolving state (`T`, `cycle_counter`)
3. Instantiating the S2 InformationEngine
4. Loading all necessary public masks (`F`, `G`) and formats (`M`)
5. Establishing the agent state (`agent_uuid`, `agent_secret`)

**Registry-Based Initialization:**
```python
def initialize_intelligence_engine():
    """Initialize Intelligence Engine with registry-based UUID management"""
    # 1. Ensure UUID registry exists and load it
    uuid_registry = ensure_uuid_registry()
    agent_uuid = uuid_registry["agent_uuid"]
    format_uuid = uuid_registry["format_uuid"]
    
    # 2. Load agent preferences
    with open("baby/baby_preferences.json", 'r') as f:
        baby_prefs = json.load(f)
    agent_secret = baby_prefs["agent_secret"]
    
    # 3. Initialize engines with registry UUIDs
    inference_engine = InferenceEngine()
    information_engine = InformationEngine()
    
    # 4. Load public components using registry UUIDs
    formats_path = f"public/formats/formats-{format_uuid}.json"
    with open(formats_path, 'r') as f:
        M = json.load(f)
    
    # 5. Create Intelligence Engine with registry-managed state
    intelligence_engine = IntelligenceEngine(
        agent_uuid=agent_uuid,
        agent_secret=agent_secret,
        format_uuid=format_uuid,
        inference_engine=inference_engine,
        information_engine=information_engine,
        formats=M
    )
    
    return intelligence_engine
```

This hierarchical composition ensures clean separation of concerns while maintaining proper integration between components and consistent UUID management through the centralized registry.

def process_thread(thread_uuid: str, input_data: bytes, agent_uuid: str, agent_secret: str):
    """
    Process a complete thread of input data and save all results.
    This is the main entry point for thread processing.
    """
    # Validate thread UUID against registry
    if not validate_thread_uuid(thread_uuid):
        raise ValueError(f"Thread UUID {thread_uuid} not found in registry")
    
    # Load UUID registry for format access
    uuid_registry = ensure_uuid_registry()
    format_uuid = uuid_registry["format_uuid"]
    
    # Initialize engines with registry UUIDs
    inference_engine = InferenceEngine()
    information_engine = InformationEngine()
    
    # Load formats using registry UUID
    formats_path = f"public/formats/formats-{format_uuid}.json"
    with open(formats_path, 'r') as f:
        M = json.load(f)
    
    # Process the stream using registry-managed components
    final_encrypted_data, current_thread_keys = process_stream(
        inference_engine, intelligence_engine, input_data, 
        epigenome_snapshot, agent_uuid, thread_uuid, gene_stateless=0xAA
    )
```

## 7. Curriculum Thread Protocol for Agnostic Learning

**Purpose:**
Enable a new or existing agent to build up its formats file and pattern statistics purely through exposure to structured data streams, without any semantic labeling, annotation, or external connotation.

### Protocol
- **Curriculum Threads:**
  - The agent may be initialized or periodically exposed to "curriculum threads"—structured data streams (e.g., text corpora, educational datasets, encyclopedic content, etc.)—using the existing thread infrastructure.
  - These threads are fed to the agent as standard input, with no special formatting or semantic annotation required.
- **Agnostic Learning:**
  - The agent's learning is fully agnostic: all structure, pattern recurrence, and eventual semantic emergence arise from the physics of resonance and Hebbian updates, not from any external labeling or clustering.
  - The formats file is populated and updated solely through the agent's experience of these threads, recording resonance, usage, and pattern statistics.
- **Universal Decoding/Encoding:**
  - The agent can learn to decode and encode any format, given sufficient curriculum context, because the formats file is a record of resonance and usage, not a semantic dictionary.
- **No Semantic Labeling:**
  - No semantic labeling, annotation, or external connotation is required or permitted in the curriculum protocol. All meaning emerges intrinsically from the agent's exposure to the data.
- **Auditability:**
  - The entire curriculum process is logged and reproducible, ensuring transparency and alignment with the architecture's principles.

**Note:**
If no curriculum threads are provided, the agent will operate in a fully unsupervised mode, discovering patterns and building its formats file solely from its ongoing experience.

### Canonical Tensor-to-Byte Conversion (Spec-Compliant)

The ONLY canonical, spec-compliant way to convert a tensor to a byte is:

```python
def tensor_to_output_byte(T, F, G):
    """
    Canonical tensor-to-byte conversion using epigenome pattern matching.
    This is the only spec-compliant method for deriving a byte from the current tensor state.
    """
    key_index = find_closest_pattern_index(T, F)
    return G[key_index]
```

- All code and engines (including S4/Intelligence) MUST use this route for byte emission.
- No threshold-based, chirality, or direct sum methods are allowed for canonical operation.
- This ensures epigenome coherence, reversibility, and security as described above.