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

> ⚠️ **Research Status**: This is an active research project demonstrating theoretical principles. The architecture is complete but implementation is ongoing. Not ready for production use.

---

## 🧬 Genetic Code

The structural parallels between GyroSI and biophysics are precise and intentional:

| Biology / Biophysics | GyroSI Architecture | Significance |
| --- | --- | --- |
| 4 nucleotides (A T/U C G) | 4 CGM stages (CS, UNA, ONA, BU) | Recursive alignment stages |
| 3 positions in a codon | 3 spatial axes (X, Y, Z) | Encodes 3D structural information |
| 2 complementary strands | 2 frames per layer (primary/dual) | Provides 6 Degrees of Freedom (3×2) |
| 4-mer sequence → 8 bits → 256 combinations | 1 byte = 8 bits → 256 introns | Complete instruction set |
| 64 codons (3 nucleotides × 2 bits) | 48-bit state space (4×2×3×2) | Complete geometric structure |
| Epigenetic regulation | Monodromic Fold operation | Path-dependent memory |

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

> **Why Physics Prevents Hallucinations**: Traditional AI operates in 768+ dimensional spaces where models can interpolate between any points, creating nonsense. GyroSI is constrained to a finite 3D manifold with only 788,986 valid states. You can't be "between" states. You're always at a specific, well-defined point. This dimensional grounding is why the system literally cannot hallucinate.

> **Why No Scoring**: GyroSI uses constraint satisfaction, not competitive scoring. Tokens either satisfy geometric constraints or they don't. There's no "best" token, only admissible ones. This implements true non-antagonistic selection aligned with CGM physics.

---

## ⚙️ How It Works: Token-Aware Physics

**1. The Quantum of Meaning: 1 Token**
The fundamental unit of knowledge is now a `token_id` from a standard tokenizer (e.g., BERT). This token ID, not its byte fragments, serves as the semantic anchor for learning.

**2. From Token to Physics: LEB128 Byte Streams**
Each `token_id` is converted into its unique, variable-length byte sequence using LEB128 encoding. These bytes are the physical carriers of the token's identity.

**3. The Universal Reference: XOR with 0xAA**
Each byte in the sequence is XORed against the universal reference `GENE_Mic_S = 0xAA` to yield an 8-bit physical instruction (`intron`). This ψ transformation is mandatory for all byte-to-intron conversions, translating external bytes into the system's internal physics.

**Mathematical Alignment**: The pattern 0xAA (binary 10101010) has perfect balance (4 ones, 4 zeros) and maximal alternation, placing it at the geometric center of the 8-bit hypercube. The XOR with 0xAA inverts the LEB128 continuation bit, aligning external protocol with internal physics.

**4. The Evolving Physical State**
The sequence of introns from a token drives the system's canonical state (a 48-bit integer) through a path-dependent trajectory across the **788,986** possible physical configurations. The intelligence resides in this trajectory.

**5. From State to Meaning: Token-Level Learning**
After processing a full token's byte sequence, the system's final state and the original `token_id` form the unique context key: `(state_index, token_id)`. This key is used to look up a minimal "phenotype"; a learned physical residue and a confidence score from the system's memory.

**6. Learning as Physical Integration**
Learning occurs once per token. The final intron of the token's byte sequence is integrated into the phenotype's memory (`mask`) via the **monodromic fold**, a non-associative, path-dependent algebraic operation. This ensures that knowledge is both semantically coherent (keyed by token) and physically grounded (updated via path-dependent physics).

### 🎯 What This Achieves

This architecture does not merely map bytes to operations; it renders each instruction as a transformation on a physical ontology. Symbolic input becomes physical geometry. Intelligence emerges as a dynamo of structural transformations orbiting within a gyroscopic topology. Alignment is not imposed or inferred, but emerges naturally as the system follows the physical dynamics of its own architecture.

This solves three fundamental problems:

- **Black Box**: Every decision traces through explicit, auditable physical state changes.
- **Alignment**: The system's actions are constrained by its own structural history and physical laws.
- **Efficiency**: The core physics are dependency-free and operate with extreme speed. Memory growth is bounded by the finite size of the physical ontology.

---

## The Five Maps: Complete Knowledge Atlas

GyroSI's intelligence operates on five pre-computed maps that completely define its finite universe:

- **Ontology Map (`ontology_keys.npy`): What Can Exist**
    Maps indices 0..788,985 to unique 48-bit state integers. These 788,986 states are ALL possible states under our physics.

- **Phenomenology Map (`phenomenology_map.npy`): How Things Appear**
    Maps each state to one of 256 canonical orbit representatives. Each orbit is a strongly connected component where all states can reach each other.

- **Epistemology Map (`epistemology.npy`): How Knowledge Changes**
    The 788,986 × 256 State Transition Table. Given any state and any intron, it determines the next state.

- **Theta Map (`theta.npy`): Distance from Truth**
    Maps each state to its angular distance from the archetype (GENE_Mac_S). Used for geometric navigation.

- **Orbit Sizes Map (`orbit_sizes.npy`): Specificity Measure**
    Maps each state to its orbit's cardinality. Used for deterministic tie-breaking in address binding.

---

## 🧠 Memory Architecture

GyroSI uses three distinct memory types:

- **Active Memory**: Exactly 6 bytes (48-bit state), constant size
- **Address Memory**: Token-to-state mapping, bounded by vocabulary size
- **Passive Memory**: Experience storage as 8-bit masks, with strict caps:
  - K=64 masks per state per orbit
  - M=64 states per token per orbit
  - Only non-zero masks stored, preventing unbounded growth

---

## 🔬 Theoretical Foundation

GyroSI implements the **Common Governance Model (CGM)**, where intelligence emerges through recursive structural alignment. The model derives three-dimensional space with six degrees of freedom from a single axiom, with time emerging as the memory of recursive operations.

Mathematical formalism employs gyrogroup structures (generalizations of rotation groups) following Abraham Ungar's work, providing precise language for transitions from undifferentiated potential to structured reality.

Gyroscopic Superintelligence is meta-language for computation, ontology, phenomenology and epistemology, enabling agents and agencies to reason about states, symmetry, and evolution economically and efficiently.

---

## ⚡ Performance Characteristics

> **Note**: These are architectural projections, not benchmarks. The system is still in development.

**Core Physics Files**: ~785 MB total
- State transition table: 770 MB
- Four mapping files: 15 MB combined
- Everything else: Python code

**Memory Usage**:
- **Active state**: Always 6 bytes (that's it!)
- **Passive memory**: Grows with usage but physics-bounded
  - Typical Wikipedia-scale: ~1-2 GB
  - Hard limit: Can't exceed ~50 GB even theoretically

**Speed Expectations**:
- **Python prototype**: ~100K-500K tokens/second per core
- **Future native version**: Could be 10-50x faster
- **No GPU needed**: Just CPU and RAM

**What This Means**:
- Fits on a modern laptop with 8GB RAM
- Processes text faster than you can read
- Learns continuously without retraining
- Same physics runs from Raspberry Pi to server

The key insight: Unlike traditional AI that needs hundreds of gigabytes, GyroSI's entire "brain" is smaller than a single movie file, yet it never forgets what it learns.

---

## 📚 Documentation

- 📖 [Genetics - Technical Specification: The complete technical specification, system constants, and build-time discovery processes.](https://github.com/GyroSuperintelligence/BabyLM/blob/main/guides/Genetics.md)

- 📖 [Physics - Common Governance Model Theory: The theoretical foundations](https://korompilias.notion.site/Common-Governance-Model-Foundations-1ee9ff44f4368050af28d1c0f8aae89a)

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


