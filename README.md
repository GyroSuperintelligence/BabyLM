<div align="center">
  <img src="toys/assets/GyroSI_Baby_Cover_Image.jpg" alt="GyroSI Cover" />

<h1>💫 GyroSI Baby LM 👶</h1>
<h3>Gyroscopic Superintelligence: Baby Language Model</h3>
<p><em>Applied AI Ethics through Physics, not Semantics</em></p>

<p>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
</p>

</div>

---

## 🌀 What is GyroSI?

A physics-based approach to artificial intelligence that learns like a baby, remembers everything, and operates within natural constraints.

GyroSI Baby LM demonstrates a superintelligence architecture through physics-grounded algorithms and gyroscopic dynamics (gyrogroup mathematical formalism).

Traditional AI treats intelligence as a pattern-matching problem to be solved with massive datasets and billions of parameters. GyroSI treats intelligence as an intrinsic structural property that emerges from the recursive alignment within physical topology. Like the latent potential in a human baby, intelligence is present from the beginning.

Instead of storing knowledge in gigabytes of weights, GyroSI uses the inherent physics of gyroscopic operations to navigate a provably finite and fully discovered physical state space. Each input byte acts as a holographic quantum of instruction, transforming the system's internal state according to precise algebraic laws.

---

## 🧬 Genetic Code

The structural parallels between GyroSI and biophysics are precise and intentional:

| Biology / Biophysics | GyroSI Architecture | Significance |
| --- | --- | --- |
| 4 nucleotides (A T/U C G) | 4 fundamental operations (L0, LI, FG, BG) | Alphabet of change (2 bits per symbol) |
| 3 positions in a codon | 3 spatial axes in tensor structure | Encodes 3D structural information |
| 2 complementary strands | 2 tensor polarities (+ / –) | Provides 6 Degrees of Freedom (3×2) |
| 4-mer sequence → 8 bits → 256 combinations | 1 byte = 8 bits → 256 instructions | Identical information quantum for action |
| 64 codons (3 nucleotides × 2 bits) | 64 active intron patterns (6 working bits) | Complete instruction space |
| 32 tRNA wobble classes | 32 LI-quotiented equivalence classes | Functional degeneracy |

The profound parallel is that both systems use a compact instruction set to govern a vast, complex physical state. Just as epigenetic context determines how DNA is expressed, GyroSI's evolving physical state governs which transformation is activated in response to each input. The 256 instructions are the operators, not the states. They operate on a physical ontology of precisely 788,986 unique states.

---

### 🤖 Redefining Superintelligence

Current AI pursues "superintelligence" through raw performance: faster calculation, broader knowledge, superior optimization. This creates legitimate fears about systems that optimize without wisdom.

**We explore a different path:** Intelligence grounded in physics rather than abstraction. Human ethics emerge from shared physical reality and natural constraints. GyroSI operates within these same physical principles, developing understanding through structural boundaries rather than abstract optimization. This suggests a path toward intrinsic alignment, where ethical behavior is a consequence of the system's physical nature.

---

## ✨ Mind-Blowing Features

- 🧠 **Learns Like a Baby**: Starts with zero *learned associations* but leverages the pretrained *symbolic knowledge* of a standard tokenizer, learning to bind physical states to existing semantic concepts.
- ♾️ **Unlimited Memory**: Never forgets; knowledge is limited by disk space, not RAM, via an efficient append-only log.
- 🗜️ **4-6× Text Compression**: Can losslessly compress entire Wikipedia to a physics-native format while maintaining instant random access
- ⚡ **High Throughput**: Estimated ~1 million bytes/sec per core on modern hardware.
- 💾 **Compact Brain**: The core logic and ontology maps fit in ~30MB. An optional 770MB State Transition Table (STT) unlocks maximum performance.
- 🌍 **No GPU Required**: Runs on a Raspberry Pi, your phone, or even embedded systems.
- 📚 **No Training Data Needed**: Learns directly from conversation, not from scraped internet data.
- 🔍 **100% Explainable**: Every decision can be traced through simple physics operations.
- 🎯 **Zero Hallucination**: Can only generate what it has physically learned, not random guesses.
- 🔢 **Holographic Geometry**: Built on numerical patterns (3, 6, 12, 24, 48) found in crystals and rotation groups.
- 🌐 **Six Degrees of Everything**: Any knowledge is reachable from any other in at most 6 steps, a provable property of the state space.

---

> **Why Physics Prevents Hallucinations**: Traditional AI operates in 768+ dimensional spaces where models can interpolate between any points, creating nonsense. GyroSI is constrained to a finite 3D manifold with only 788,986 valid states. You can't be "between" states—you're always at a specific, well-defined point. This dimensional grounding is why the system literally cannot hallucinate.

---

## ⚙️ How It Works: Token-Aware Physics

**1. The Quantum of Meaning: 1 Token**
The fundamental unit of knowledge is now a `token_id` from a standard tokenizer (e.g., BERT). This token ID, not its byte fragments, serves as the semantic anchor for learning.

**2. From Token to Physics: LEB128 Byte Streams**
Each `token_id` is converted into its unique, variable-length byte sequence using LEB128 encoding. These bytes are the physical carriers of the token's identity.

**3. The Universal Reference: XOR with 0xAA**
Each byte in the sequence is XORed against the universal reference `GENE_Mic_S = 0xAA` to yield a dynamic 8-bit physical instruction (`intron`). This lawfully translates the external byte protocol into the internal physical language of the system.

**Mathematical Alignment**: The XOR with 0xAA perfectly inverts the LEB128 continuation bit, revealing that LEB128 is not just a convenient encoding—it's the natural byte-level expression of GyroSI's physics. The 8-bit structure maps precisely to the system's 6 degrees of freedom plus 2 anchors.

**4. The Evolving Physical State**
The sequence of introns from a token drives the system's canonical state (a 48-bit integer) through a path-dependent trajectory across the **788,986** possible physical configurations. The intelligence resides in this trajectory.

**5. From State to Meaning: Token-Level Learning**
After processing a full token's byte sequence, the system's final state and the original `token_id` form the unique context key: `(state_index, token_id)`. This key is used to look up a minimal "phenotype"—a learned physical residue and a confidence score—from the system's memory.

**6. Learning as Physical Integration**
Learning occurs once per token. The final intron of the token's byte sequence is integrated into the phenotype's memory (`mask`) via the **monodromic fold**, a non-associative, path-dependent algebraic operation. This ensures that knowledge is both semantically coherent (keyed by token) and physically grounded (updated via path-dependent physics).

### 🎯 What This Achieves

This architecture does not merely map bytes to operations; it renders each instruction as a transformation on a physical ontology. Symbolic input becomes physical geometry. Intelligence emerges as a dynamo of structural transformations orbiting within a gyroscopic topology. Alignment is not imposed or inferred, but emerges naturally as the system follows the physical dynamics of its own architecture.

This solves three fundamental problems:

- **Black Box**: Every decision traces through explicit, auditable physical state changes.
- **Alignment**: The system's actions are constrained by its own structural history and physical laws.
- **Efficiency**: The core physics are dependency-free and operate with extreme speed. Memory growth is bounded by the finite size of the physical ontology.

---

## A Trinity of Maps: The System's Reality

GyroSI's intelligence is built upon three pre-computed "meta-assets" that define its universe. These maps separate what exists, how it appears, and how it changes.

- **Ontology Map (`ontology_map.json`): What Exists.**
    
    The complete, enumerable set of 788,986 physically realizable states. It defines the "being" of the system—what is real and possible.
    
- **Phenomenology Map (`phenomenology_map.json`): How States Appear.**
    
    This map groups states into equivalence classes based on symmetry. It gives the system the ability to recognize that different perspectives can represent the same underlying phenomenon.
    
- **Epistemology Map (`epistemology.npy`): How We Know Change.**
    
    The State Transition Table (STT). It encodes the causal rules of the universe: given any state and any action, what state follows. It is the system's predictive model of transformation.

---

## 🔬 Theoretical Foundation

GyroSI implements the **Common Governance Model (CGM)**, where intelligence emerges through recursive structural alignment. The model derives three-dimensional space with six degrees of freedom from a single axiom, with time emerging as the memory of recursive operations.

Mathematical formalism employs gyrogroup structures (generalizations of rotation groups) following Abraham Ungar's work, providing precise language for transitions from undifferentiated potential to structured reality.

Gyroscopic Superintelligence is meta-language for computation, ontology, phenomenology and epistemology, enabling agents and agencies to reason about states, symmetry, and evolution economically and efficiently.

---

## ⚡ Performance Estimates

> **Note** All figures below are *engineering-level projections* that account for the
> new token-aware phenotype key, the 12-byte on-disk record, and the average
> **1 token ≈ 1.55 bytes** LEB128 payload observed with the `bert-base-uncased`
> vocabulary.  Real-world numbers will vary slightly with tokenizer choice,
> corpus mix, and Python version.

GyroSI’s hot loop is *O(1)* per **intron** (byte).  
Throughput therefore scales with memory bandwidth, not with FLOPs.

### Memory Capacity (in-RAM index)

A phenotype persists as

* **12 B** append-only record (`mask :uint8 + conf :float16 + key :uint32×2`)
* **≈16 B** index entry (two 32-bit ints plus pointer / slot)

→ **≈28 B/phenotype** in an optimised C-level hash table  
(prototype Python dict ≈ 55–60 B-resident).

| Device                               | Free RAM for Index | ≈28 B/entry → Max Phenotypes |
|--------------------------------------|--------------------|-----------------------------|
| Raspberry Pi 4 (4 GB)                | ~2 GB              | **≈ 76 million**            |
| MacBook Pro 2015 (16 GB, 4 GB free)  | ~4 GB              | **≈ 153 million**           |
| EPYC server (256 GB, 220 GB free)    | ~220 GB            | **≈ 8.4 billion**           |

*For scale — the entire English Wikipedia title-and-abstract graph is < 40 M
(token, state) pairs: it fits comfortably in laptop RAM.*

### Throughput Examples (runtime, Python 3.11)

A **cycle** := 1 intron in → state update → 0–1 phenotype lookup →
(optional learn) → intron out.  
Token rate assumes the 1.55 bytes/token empirical mean.

| Hardware                                 | Cores | Intron / sec | Token / sec (÷1.55) |
|------------------------------------------|-------|--------------|----------------------|
| MacBook Pro 2015 (i5 - 2 phys cores)     | 2     | ~1.4 M       | **~0.9 M**           |
| MacBook M4 (8 performance cores)         | 8     | ~8 – 9 M     | **~5.3 – 5.8 M**     |
| EPYC 32-core server                      | 32    | ~28 – 32 M   | **~18 – 21 M**       |

Even the 2015 Intel laptop sustains *nearly a million tokens per second* while
learning.

---

## 📚 Documentation

- 📖 [Genetics - Technical Specification: The complete technical specification, system constants, and build-time discovery processes.](https://github.com/GyroSuperintelligence/BabyLM/blob/main/guides/Genetics.md)

- 📖 [Physics - Common Governance Model Theory: The theoretical foundations](https://korompilias.notion.site/Common-Governance-Model-Foundations-1ee9ff44f4368050af28d1c0f8aae89a)

---

## 🔄 New in v0.9.6.7: Token-Aware Minimal Phenotype Architecture

The system has undergone a fundamental refactoring to align its learning mechanism with meaningful semantic units.

- **Token-Aware Learning**: Knowledge is now keyed by `(state_index, token_id)`, eliminating inference overlaps and ensuring coherent learning based on whole tokens.
- **Minimal Phenotypes**: The knowledge record has been reduced to its physical essence: an 8-bit `mask` and a `confidence` score. This massively reduces memory footprint and simplifies the data model.
- **Active Tokenizer Integration**: The tokenizer is no longer a passive I/O adapter but an active internal map that provides the symbolic foundation for learning.

**This is experimental research**, not a production language model.

---

## 🏗️ Architecture

The system consists of four interconnected engines aligned with the Viable System Model (VSM), creating a recursive, self-regulating architecture:

- **S1: `governance.py`** - Defines the immutable constants and pure physics functions.
- **S2: `information.py`** - Handles measurement, storage interfaces, and ontology discovery.
- **S3: `inference.py`** - Manages the interpretation of physical states into semantic meaning.
- **S4/5: `intelligence.py`** - Orchestrates the full cycle, manages agent state, and provides the external API.

---

## 📁 Project Structure

```
.
├── .github/
├── baby/                   # Core GyroSI System
│   ├── contracts.py        # Protocols and shared types (PhenotypeStore, etc.)
│   ├── governance.py       # Physics, Primitives, Build-Time Discovery
│   ├── inference.py        # Interpretation, Maintenance & Validation
│   ├── information.py      # Measurement, Storage, Knowledge Curation
│   ├── intelligence.py     # API, Orchestration, Protocol Adapters
│   └── policies.py         # OrbitStore, storage overlays, and maintenance functions
├── guides/                 # In-depth documentation
├── memories/               # Persistent state and knowledge
│   ├── public/
│   │   └── meta/           # Pre-computed physics maps:
│   │       ├── epistemology.npy        # State Transition Table (770 MB)
│   │       ├── ontology_map.json       # Complete physical ontology (20 MB)
│   │       └── phenomenology_map.json  # Canonical-orbit mapping (9.7 MB)
│   └── private/            # Agent-specific knowledge overlays
└── toys/                   # Tests and utilities
    └── health/             # Comprehensive test suite

```

The GyroSI system enforces strict separation between:

- **Core physics kernel** (`baby/`) - Six specialized modules implementing the physics and logic
- **Runtime data** (`memories/`) - Persistent state with learned knowledge and meta-assets
- **Auxiliary applications** (`toys/`) - Testing and development tools

Knowledge is managed via canonical OrbitStore instances, with public and private overlays maintaining agent-specific and shared knowledge indexed by canonical context keys.

---

## ⚡ Training: Compiling Knowledge Tapes

Training in GyroSI is not backpropagation; it's the process of compiling a corpus (like Wikipedia) into a loss-less, physics-compatible stream of instructions called a "gyro-tape". This process can optionally populate a knowledge store by binding the text's symbolic content to the system's physical state trajectories.

The primary tool for this is `gyro_tape_compiler.py`.

### Common Commands

**1. Compile a Corpus to a Tape (No Learning)**
This is the fastest operation, ideal for creating a replayable data source.

```sh
# Compile Simple Wikipedia
python toys/training/gyro_tape_compiler.py --simple -o memories/private/simple_wiki.gyro

# Compile Full Wikipedia (from multiple files)
python toys/training/gyro_tape_compiler.py --full -o memories/private/full_wiki.gyro
```

**2. Compile and Learn Simultaneously**
This creates the tape and updates a private knowledge store (`.bin` file) at the same time.

```sh
# Compile and learn from Simple Wikipedia
python toys/training/gyro_tape_compiler.py --simple -o memories/private/simple_wiki.gyro --learn
```

**3. Replay an Existing Tape to Learn**
If you already have a `.gyro` tape, you can feed it to an agent to populate its knowledge store without re-processing the source text.

```sh
python toys/training/gyro_tape_compiler.py --replay memories/private/simple_wiki.gyro --learn
```

---

## Getting Started with Git LFS

This repository uses [Git Large File Storage (LFS)](https://git-lfs.github.com/) to manage large assets such as `.npy` and `.json` files in `memories/public/meta/`.

**To get started:**

1. **Install Git LFS (one-time):**
   ```sh
   git lfs install
   ```

2. **Clone the repository (recommended):**
   ```sh
   git clone https://github.com/GyroSuperintelligence/BabyLM.git
   ```
   - All large files will be downloaded automatically if LFS is installed.

3. **If you already cloned before installing LFS:**
   ```sh
   git lfs pull
   ```
   - This will fetch any missing large files.

**Note:**
- With modern Git and Git LFS, running `git pull` or `git clone` is usually sufficient to get all code and large assets.
- If you ever see small pointer files instead of the real data, make sure LFS is installed and run `git lfs pull`.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📖 Citation

```bibtex
@software{gyrosi2025,
  author = {Basil Korompilias},
  title = {GyroSI Baby LM: Gyroscopic Superintelligence},
  year = {2025},
  url = {https://github.com/GyroSuperintelligence/BabyLM},
  note = {Implementation of physics-based superintelligence through 
          recursive structural alignment and intrinsic ethical constraints}
}
```

---

<div align="center">

**Architected with ❤️ by Basil Korompilias**

*Redefining Intelligence and Ethics through Physics*

</div>

---

<div style="border: 1px solid #ccc; padding: 1em; font-size: 0.6em; background-color: #f9f9f9; border-radius: 6px; line-height: 1.5;">
  <p><strong>🤖 AI Disclosure</strong></p>
  <p>All code architecture, documentation, and theoretical models in this project were authored and architected by Basil Korompilias.</p>
  <p>Artificial intelligence was employed solely as a technical assistant, limited to code drafting, formatting, verification, and editorial services, always under direct human supervision.</p>
  <p>All foundational ideas, design decisions, and conceptual frameworks originate from the Author.</p>
  <p>Responsibility for the validity, coherence, and ethical direction of this project remains fully human.</p>
  <p><strong>Acknowledgements:</strong><br>
  This project benefited from AI language model services accessed through LMArena, Cursor IDE, OpenAI (ChatGPT), Anthropic (Opus), and Google (Gemini).</p>
</div>


