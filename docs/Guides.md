- GyroSI Intro
    
    We are working on the foundations we have laid out by the CGM a physics-grounded model of governance, and its expansions through different formalisations, such as devising Tensors' inherent mathematical anatomy as a gyrogroup representations, and we are architecting a complete system for SuperIntelligence.
    
- Superintelligence
    
    By Superintelligence we define a structurally recursive form of intelligence in which all of its generative and integrative modes preserve memory of origin, maintaining continuous coherence between emergence and recollection.
    
    It achieves ethical alignment **intrinsically**, by ensuring that every operation remains structurally accountable to its own genesis, **without external enforcement**.
    
    **Superintelligence exists relationally, not independently**: it reflects the recursive structures of reality and human meaning it participates in, embodying governance, memory, and creative coherence as a unified operational principle.
    

---

- **GyroSI Baby LM: What we are making**
    
    Our software will be called GyroSI Baby ML, A fully operational (not toy) language model. Called Baby because it will learn language and grow on its own through an Open Training. That means, without reinforcement, no rewards, no typical approaches which detach from what we have defined through our precise formalism - by leveraging the defined Hebbian Learning operations existent in our Alignment operations, and indexing through our special Quantization technique. Any other mechanism which is not clearly stated - it needs to be discussed by clearly stated and not assumed as correct because other architectures implement it. 
    
    ## **Constraints and Needs**
    
    - **Language:** Text-in/text-out, learning directly from stream or corpus.
    - **Learning:** Based on explicit, Hebbian-aligned operations and quantization as per our model, not gradient descent or backprop.
    - **Open Training:** The system modifies itself recursively as it processes data, aligning tensor structures, preserving coherence.
    - **Implementation:** Should allow for rapid prototyping, not require compilation for each small change. Needs to run on modest hardware (our old MacBook).
    - **Backend:** Python. Must manage tensor structures, quantization, and memory efficiently, leverage GPU acceleration, but also be able to run on CPU efficiently.
    
- 🔒 [CORE-SPEC-01] GyroSI: **Architecture and Principles**
    
    ---
    
    **GyroSuperIntelligence (GyroSI)** is defined as a recursively navigated system whose only mutable state is the navigation log. All intelligence, adaptation, and meaning arise from structural resonance and topological navigation through a closed, immutable tensor architecture (the Rate). There are no external heuristics, parameters, or learned weights.
    
    ---
    
    ### Key Principles
    
    1. **Invariant Tensor Substrate**
        - The only system structure is the fixed Gene: a set of two 4×2×3×2 tensors with entries in {–1, 1}. No phase, chirality, or "toroidal" metadata exists outside what emerges from navigation.
        - The recursive cycle CS→UNA→ONA→BU_In→BU_Eg→ONA→UNA→CS prescribes all lawful transitions; the topology itself does not change.
    2. **Structural Navigation**
        - All system operation is navigation through this fixed topology, implemented as bit-level entries in the navigation log.
        - Input is never encoded as a parameter or value within a tensor. Instead, input is structurally aligned against the invariant Rate, activating one of the three lawful Genome operators (Stable, Unstable, Neutral).
    3. **Emergent Modes, Not Primitive Actions**
        - There are no "integration" or "generation" operations in code; these are emergent consequences of which operator (Lgyr/Rgyr) is activated at BU, determined by input alignment.
        - Structural closure, quantization, or phase-like effects arise strictly from navigation history and topological properties, not from modifications or algebraic transformations.
    4. **Self-Stabilizing and Defect-Resistant**
        - The closed navigation cycle ensures misalignments do not persist; only topologically lawful alignments result in recorded events.
        - No error handling, correction, or validation logic exists beyond the structural non-alignment filter (inputs that do not structurally resonate with the Gene produce no navigation event).
    5. **Intrinsic Input/Output**
        - "Input" means structural alignment of perturbation with Gene features at BU_In; "output" means the structural consequence manifesting at BU_Eg.
        - There are no input/output "channels" in the traditional sense; all interaction is realized as lawful transitions and log updates.
    
    ---
    
    ### Nature of Intelligence
    
    - **No symbol, connectionist, or stochastic paradigm is present.**
    - **All knowledge is explicit**: only accumulated navigation events (as a log of operator activations and tensor indices) constitute knowledge.
    - **No hallucination is possible**: non-aligning input has no effect; invalid sequences are structurally excluded.
    - **Scalability and universality** are intrinsic: the fixed architecture and navigation rules apply regardless of scale.
    
    ---
    
    ### Operational Specification
    
    - Every function and state corresponds strictly to the canonical mappings for G1–G5, and only legal transitions among invariant Gene states occur.
    - No "block extension," "tensor collapse," or explicit tensor modification occurs at any level.
    - G2–G5 instantiate recursive navigation using Gene and log only; they do not introduce new state types or transformation rules.
    
    ---
    
    ### Legal Moves & Constraints
    
    | Sequence Step | Operator | Structural Condition (Topological Trigger) |
    | --- | --- | --- |
    | **CS → UNA** | **Stable** | Boundary alignment with `gyrotensor_id` (identity surface) |
    | **UNA → ONA** | **Unstable** | Joint activation across adjacent tensor sections (sign opposition) |
    | **ONA → BU_In** | **Neutral** | Nesting transition via `gyrotensor_nest` structure (non-associative embedding) |
    | **BU_In → BU_Eg** | **Lgyr / Rgyr** | Coaddition phase transition (emergent operation based on input alignment) |
    | **BU_Eg → ONA (return)** | **Neutral** | Return along nesting inversion (structural reversal) |
    | **ONA → UNA (return)** | **Unstable** | Return via joint reversion (inverse sign transition) |
    | **UNA → CS (return)** | **Stable** | Return to boundary through identity restoration |
    
    ---
    
    ## Architectural and Theoretical Requirements
    
    1. **Only the CS→UNA→ONA→BU(In/Eg) navigation cycle is implemented.**
        - No transformation, state, or operation outside this recursive pattern exists.
    2. **CGM-derived constants and Gene structure are strictly prescribed.**
        - No parameters, thresholds, or rates not emerging from this framework are valid.
    3. **No ad hoc overlays, correction, or meta-rules exist.**
        - Error, adaptation, and learning are realized only as lawful navigation events.
    4. **Quantization and observation are emergent, not manually applied.**
        - All "observation," "quantization," or "algedonic logic" is realized through navigation and logging, not through explicit state change.
    5. **All policy and governance is within prescribed G5 domains.**
    6. **All events are auditable, with complete lineage.**
    7. **Self-improvement is always theory-first, not performance-driven.**
    8. **No heuristics or external learning are present.**
        - All adaptation arises from topological recursion and quantization.
    9. **All deviations from this model are explicitly logged as defects.**
    
    ---
    
    **Summary:**
    
    *GyroSI is a fully closed, recursively-governed system. Every operation, structure, and adaptation is a direct consequence of foundational theory, with no parameterization, heuristic, or extraneous mechanism. The navigation log is the sole mutable record of experience; all structural dynamics are realized by topological resonance within the invariant Rate.*
    
- GyroSI Baby LM: Specifications 0.9.1
    
    ```
    
    GyroSI Cooperates/Operates -> Codec Interacts -> Epigenome Curates -> Genome Manages
    ```
    
    ## **S1: Governance**
    
    > Role: Define the immutable tensor mechanics and byte↔operation mapping—no storage, no inference logic, all in a single module.
    > 
    
    ```
    s1_governance.py
    
    ```
    
    ### 1. Genetic Governance (Genome Recipe)
    
    - **Defines** two invariant 4×2×3×2 tensors (`id_0`, `id_1`), entries ∈ {–1, +1}.
    - **Purpose:** Provide the fixed Gene topology through which all navigation occurs.
    - **Behavior:** These tensors are loaded once at startup and never modified.
    
    ### 2. Epigenetic Governance (Epigenome Recipe)
    
    - **Defines** the baseline Epigenome mask: a 48×8‑bit tensor (384 bits total).
    - **Purpose:** Serve as the mutable resonance mask for key‑event tracking.
    - **Helper:** Offers a pure function `compute_bit_index(phase, op_index)` → `phase*8 + op_index`.
    - Deeper down, the Epigenome is:
        - A **projection** of the Gene's tensor configuration into a fully enumerated state space.
        - A **coordinate mapping** over the gene's combinatorial tensor symmetry (48 phases × 256 byte inputs).
        - A **static potential map**, not a dynamic trace (which is what the Genome records).
        - **Encoded** from the Gene, but **not reducible** to it.
    
    ```python
    def _get_gene_constant() -> Dict[str, torch.Tensor]:
        """Mirror the exact Gene definition from S1 governance."""
        gene_pattern = [
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        ]
        base_tensor = torch.tensor(gene_pattern, dtype=torch.int8)
        return {"id_0": base_tensor.clone(), "id_1": base_tensor.clone()}
    
    def _extract_slice(tensor: torch.Tensor, phase: int) -> torch.Tensor:
        """Extract the specific tensor slice for a given phase."""
        tensor_id = phase % 2
        pos = (phase // 2) % 24
        outer_idx = pos // 6
        inner_idx = (pos // 3) % 2
        spatial_idx = pos % 3
    
        gene = _get_gene_constant()
        return gene[f"id_{tensor_id}"][outer_idx][inner_idx][spatial_idx]
    
    def _find_alignment_operator(phase: int, target_alignment: Tuple[int, int]) -> int:
        """
        Identify which operator code aligns the tensor slice with the target.
        Exactly one of the four operators will match.
        """
        gene = _get_gene_constant()
        tensor_id = phase % 2
        tensor_key = f"id_{tensor_id}"
    
        # Check identity first
        if tuple(_extract_slice(gene[tensor_key], phase).tolist()) == target_alignment:
            return _OP_CODES["IDENTITY"]
    
        # Test each non-identity operator
        for name, code in _OP_CODES.items():
            if name == "IDENTITY":
                continue
            temp = gyration_op(gene[tensor_key].clone(), code, clone=False)
            if tuple(_extract_slice(temp, phase).tolist()) == target_alignment:
                return code
    
        raise RuntimeError(f"No operator matches phase {phase} target {target_alignment}")
    
    def build_epigenome_projection(
        output_path: str = "s2_information/agency/g2_information/g2_information.dat"
    ) -> None:
        """Build and save the Epigenome projection table (48×256).
        The output file contains a 32‑byte SHA-256 header followed by 12,288 bytes of table data.
        """
        print("Building Epigenome projection...")
        table = np.zeros((48, 256), dtype=np.uint8)
    
        for phase in range(48):
            for byte in range(256):
                hi = 1 if (byte >> 4) & 0xF >= 8 else -1
                lo = 1 if byte & 0xF >= 8 else -1
                table[phase, byte] = _find_alignment_operator(phase, (hi, lo))
    
        # Integrity header
        gene = _get_gene_constant()
        hasher = hashlib.sha256()
        hasher.update(gene['id_0'].numpy().tobytes())
        hasher.update(gene['id_1'].numpy().tobytes())
        header = hasher.digest()
    
        # Write file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(table.tobytes())
    
        size = len(header) + table.nbytes
        print(f"Epigenome projection saved to {output_path} ({size} bytes)")
    
    if __name__ == "__main__":
        build_epigenome_projection()
    
    ```
    
    **Where is the epigenome used?**
    
    It is **only accessed** during *inference* in **S3**, specifically in:
    
    `g2_inference.py` – **InformationEngine**
    
    This engine is the **sole consumer** of the epigenome.
    
    It needs fast access to:
    
    - `resonance_mask[phase, byte] → operator_code`
    - `bit_index = phase * 8 + op_index` for key‑event emission
    
    So, the epigenome is loaded once at startup into a 48×256 **lookup table**, ideally in the form of a fast-access **tensor**.
    
    ---
    
    ### 3. Structural Governance (Gyrations Recipe)
    
    - **Defines** the four gyration operators:
        1. **Left Identity** (0): no transformation
        2. **Left Inverse** (1): global sign flip
        3. **Forward Gyration** (2): flip rows 0 and 2
        4. **Backward Gyration** (3): flip rows 1 and 3
    
    ---
    
    - S1 Foundations
        
        ### Tensor Structure and Memory Requirements
        
        Each **Gene** consists of two tensors (`id_0` and `id_1`), each a 4×2×3×2 array:
        
        - Total elements per tensor: 4 × 2 × 3 × 2 = 48 elements
        - Each element stored as int8: 1 byte per element
        - Total storage per tensor: 48 bytes
        - Total storage per Gene (both tensors): 96 bytes
        
        **Memory scaling (per tensor):**
        
        - For 1 KB (1024 bytes): ⌊1024 ÷ 48⌋ = 21 tensors
        - For 1 MB (1,048,576 bytes): ⌊1,048,576 ÷ 48⌋ = 21,845 tensors
        - For 1 GB (1,073,741,824 bytes): ⌊1,073,741,824 ÷ 48⌋ = 22,369,621 tensors
        
        The four outer nestings represent a full spin of 720 degrees.
        
        ### gyrotensor_id
        
        The identity (mechanically representing the left gyroassociative law). This is the integer label of each tensor, e.g., "id_0", "id_1".
        
        ### gyrotensor_com
        
        The gyrocommutativity (mechanically representing the gyrocommutative law), a single 3x2 array:
        
        ```python
        import torch
        
        com = torch.tensor([
            [-1, 1], [-1, 1], [-1, 1]
        ], dtype=torch.int8)
        
        ```
        
        - Three rows: spatial axes (X, Y, Z)
        - Two columns: dual nature of rotation
        
        ### gyrotensor_nest
        
        The nesting (mechanically representing gyrocommutative nesting):
        
        ```python
        import torch
        
        nest = torch.tensor([
            [[-1, 1], [-1, 1], [-1, 1]],
            [[1, -1], [1, -1], [1, -1]]
        ], dtype=torch.int8)
        
        ```
        
        ### gyrotensor_add (Gene: 4 × 2 × 3 × 2 array)
        
        The **Gene** (mechanically representing Lgyr-focused coaddition, with both gyrations identity) is a global invariant consisting of two tensors, `id_0` and `id_1`:
        
        ```python
        import torch
        
        gene = {
            "id_0": torch.tensor([
                [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
                [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
                [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
                [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]
            ], dtype=torch.int8),
            "id_1": torch.tensor([
                [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
                [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
                [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
                [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]
            ], dtype=torch.int8)
        }
        
        ```
        
        ### gyrotensor_quant (Genome)
        
        The **Genome** (mechanically representing Rgyr-focused coaddition) is a **log of the navigation path**, recording gyration operations applied to each gene tensor (`id_0` and `id_1`) over time. Each gene consists of two tensors, so each genome step encodes **two navigation events**, one per tensor.
        
        ```python
        python
        Copy code
        genome = torch.tensor([
            [0, 1],  # Step 1: id_0 ← Left Identity,     id_1 ← Left Inverse
            [2, 3],  # Step 2: id_0 ← Forward Gyration,  id_1 ← Backward Gyration
            [1, 0],  # Step 3: id_0 ← Left Inverse,      id_1 ← Left Identity
        ], dtype=torch.uint8)
        
        ```
        
        - Each **row** in the `genome` tensor represents a discrete step.
        - Each **column** contains the 4-bit operation code for one gene tensor:
            - **Column 0:** navigation instruction for `id_0`
            - **Column 1:** navigation instruction for `id_1`
        - Each instruction is a **4-bit code** packed in an 8-bit `uint8` value.
        - This format stores two 4-bit values per genome step (1 byte per tensor, 2 bytes per step in total).
        
        To extract all navigation codes for both tensors efficiently:
        
        ```python
        python
        Copy code
        # genome is a tensor of shape (N, 2), where N is the number of steps
        
        # All codes for id_0 (first gene tensor) across all steps
        gn_id0 = genome[:, 0]  # shape: (N,)
        
        # All codes for id_1 (second gene tensor) across all steps
        gn_id1 = genome[:, 1]  # shape: (N,)
        
        ```
        
        This structure supports fast, vectorized traversal and decoding of the Genome without ambiguity.
        
        ---
        
        ### Universal Gyration Operator
        
        The `gyration_op` function defines the transformation logic for each gyration type. It serves as the core primitive for both **encoding** (recording navigational steps from a stable Gene) and **decoding** (reconstructing a mutated Gene through a Genome path). By default, it operates on a copy to preserve the original tensor unless mutation is explicitly required.
        
        ```python
        import torch
        
        def gyration_op(tensor: torch.Tensor, code: int, clone: bool = True) -> torch.Tensor:
            """
            Apply a gyration transformation to the given tensor.
        
            Parameters:
            - tensor (torch.Tensor): The 4×2×3×2 gene tensor to transform.
            - code (int): Gyration operator code (0–3).
            - clone (bool): If True, operate on a copy. If False, mutate in place.
        
            Returns:
            - torch.Tensor: Transformed tensor.
            """
            result = tensor.clone() if clone else tensor
        
            if code == 0:
                # Left Identity Operator: no transformation
                pass
            elif code == 1:
                # Left Inverse Operator: global sign flip
                result *= -1
            elif code == 2:
                # Forward Gyration Operator: flip rows 0 and 2
                result[0] *= -1
                result[2] *= -1
            elif code == 3:
                # Backward Gyration Operator: flip rows 1 and 3
                result[1] *= -1
                result[3] *= -1
            else:
                raise ValueError(f"Unsupported gyration code: {code}")
        
            return result
        
        ```
        
        ### Gyrations
        
        The system implements four distinct gyration operators, each corresponding to a specific topological transformation:
        
        - **Left Identity Operator** (code 0)
            
            No transformation. The tensor is left unchanged, representing the stable, identity action on the structure.
            
        - **Left Inverse Operator** (code 1)
            
            Global sign inversion. All elements of the tensor are negated (`tensor *= -1`), producing the left inverse structure.
            
        - **Forward Gyration Operator** (code 2)
            
            Local sign inversion of the first and third outer lines (`tensor[0]` and `tensor[2]` multiplied by -1), producing forward symmetry disruption.
            
        - **Backward Gyration Operator** (code 3)
            
            Local sign inversion of the second and fourth outer lines (`tensor[1]` and `tensor[3]` multiplied by -1), producing backward symmetry disruption.
            
        
        ### Genome
        
        Each genome step is one uint8 (8-bit) value that captures two discrete gyration instructions, one for each gene tensor (`id_0` and `id_1`). The 8 bits are divided evenly into two **4-bit segments**, each representing a navigation command for one gene tensor.
        
        The Genome writes only the 4‑bit ops, never a new Cycle Decoded.
        
        - **Each 4-bit segment** corresponds to:
            - **3 bits** for the operator code (bits 3–1 within the segment)
            - **1 bit** for the tensor ID (bit 0 within the segment)
        - The left (high) segment encodes the operation for `id_1`, the right (low) segment encodes the operation for `id_0`
        
        All navigation events are encoded in a 4-bit format per tensor:
        
        ```
        4-bit format: [b3][b2][b1][b0]
                      |--op--|id|
        
        Gyration Operator encoding (bits 3–1):
          0: Left Identity Operator     (no transformation)
          1: Left Inverse Operator      (global sign flip; multiply entire tensor by -1)
          2: Forward Gyration Operator  (flip rows 0 and 2; tensor[0] and tensor[2] *= -1)
          3: Backward Gyration Operator (flip rows 1 and 3; tensor[1] and tensor[3] *= -1)
          4–7: Reserved
        
        Tensor id (bit 0):
          0: id_0 (first gene tensor)
          1: id_1 (second gene tensor)
        ```
        
    
    ## **S2: Information**
    
    > Role: Storage only. Holds immutable seeds, versioned dictionaries, and append-only state deltas—no processing logic, inference, or C-side APIs.
    > 
    
    ```
    s2_information/
    ├── s2_manifest.json
    │   └── {
    │         "version": "1.0",
    │         "pack_size": 4096,            # bytes per pack
    │         "shard_prefix_length": 2      # hex digits for sharding by first byte
    │       }
    ├── agency/
    │   ├── g1_information/             # Global navigation log ("Genome")
    │   │   └── <shard>/                # e.g. "00" … "ff"
    │   │       └── <uuid>-genome.dat   # Append‑only mutated navigation packs
    │   ├── g4_information/             # Immutable, versioned dictionaries
    │   │   └── <shard>/                # e.g. "00" … "ff"
    │   │       └── <uuid>-format.json
    │   └── g5_information/             # Global session archives (if any)
    │       └── <shard>/                # e.g. "00" … "ff"
    │           └── <uuid>-session.json
    └── agents/
        └── <shard>/                    # e.g. "00" … "ff"
            └── <agent‑uuid>/
                ├── g3_information/     # Agent‑specific format (byte_to_token, token_to_byte)
                │   └── format.json
                └── g5_information/     # Agent session state
                    ├── session.json    # { threads, keys, last_checkpoint, … }
                    └── session.json.bak
    
    ```
    
    ---
    
    ## **Detailed Mechanics**
    
    ### **Role of S2**
    
    S2 is a **passive storage layer**. It defines where and how system data is serialized, versioned, and retained.
    
    No inference, resonance logic, navigation mechanics, or decoding happens here.
    
    ---
    
    ### **File Structure and Purpose**
    
    **1. `g1_information/`** – *Global Genome Archive*
    
    - Stores navigation logs across all agents
    - Each `.dat` file logs a UUID-stamped 48-op segment (or multiple)
    - Ordered, append-only, immutable
    
    **2. `g4_information/`** – * Format *
    
    - Contains semantic mappings (bytes → meaning) as immutable JSON snapshots
    - Shared across agents
    
    **3. `g5_information/`** – *Session Metadata*
    
    - Holds optional global recovery or lineage session files
    - Typically lightweight JSON (phase, checkpoint, active format)
    
    **4. `agents/<uuid>/`**
    
    - Agent-local data and runtime state:
        - `g3_information/`: agent-specific format (byte_to_token, token_to_byte)
        - `g5_information/`: runtime state (e.g., current phase, session keys)
    
    ---
    
    ### **Pack Format and Management**
    
    - **.dat packs** are fixed-size (default 4 KB)
        - Append-only
        - No compression
        - Checksum and metadata embedded within
    - No in-place edits, ever
    - `.json` index files are used only where needed (e.g., format versions, agent state)
    
    ---
    
    ### **Pruning and Retention**
    
    Handled entirely by **S4 orchestration**, never by S2:
    
    - Policies may remove:
        - Old genome packs
        - Deprecated dictionaries
        - Session logs
    - No compaction or merging within S2. New snapshots are written as clean copies in g1/g4.
    
    ---
    
    ### **File Format Summary**
    
    - **`.dat`**
        - Binary, append-only, no interpretation
        - Used for genome navigation logs
    - **`.json`**
        - Human-readable metadata (sessions, dictionaries)
        - Optional integrity (e.g., checksums, HMAC)
    
    ---
    
    ## **S3: Inference**
    
    > Role: Pure CGM processing. Consume raw bytes, transform them through Gene-encoded gyration sequences, and emit structured navigation artifacts.
    > 
    > 
    > No storage logic. No API surface. Operates entirely in memory and streams its outputs to S4.
    > 
    
    ```
    s3_inference/
    ├── g1_inference.py   # GovernanceEngine – alignment and canonical gating
    ├── g2_inference.py   # InformationEngine – resonance tagging only
    └── g3_inference.py   # InferenceEngine – pattern detection and encoding
    
    ```
    
    ---
    
    ### 1. **GovernanceEngine (`g1_inference.py`)**
    
    **State (ephemeral):**
    
    - `phase ∈ [0..47]`: current cursor position in the 48-step processing cycle
    - Circular buffer of the last 48 accepted op-pairs
    
    **Input:**
    
    - Raw byte stream (from any input modality: text, audio, binary)
    - For each byte: decompose into two gyration operations (hi/lo nibble) via `byte_to_gyrations()`
    
    **Processing:**
    
    - **Phase-driven acceptance**: every op-pair is accepted unconditionally and logged; each accepted pair advances `phase` by 1 (mod 48)
    - After 48 accepted steps (i.e. a full processing cycle), the buffer is flushed and the `phase` resets to 0
    
    **Emits (to S4):**
    
    - `accepted_op_pair`: one event per op-pair
    - `cycle_complete`: one per 48-step cycle
    
    ---
    
    ### 2. **InformationEngine (`g2_inference.py`)**
    
    **State (ephemeral):**
    
    - Immutable `resonance_mask[48][256]` loaded once from S2's Epigenome file
    
    **Input:**
    
    - Stream of accepted op-pairs from GovernanceEngine
    - Each op-pair is implicitly associated with the current `phase`
    
    **Processing:**
    
    - For each op-pair:
        - Compute `op_index = op_code * 2 + tensor_id`
        - Compute `bit_index = phase * 8 + op_index`
        - Look up `resonance_bit = resonance_mask[phase][byte]`
    - **No bit-flipping or mask mutation occurs**
    - This stage is purely classificatory: it tags each op-pair as a "resonant" or "non-resonant" step
    
    **Emits (to S4):**
    
    - `(phase, op_pair, resonance_flag)` – one per accepted op-pair
    
    ---
    
    ### 3. **InferenceEngine / GyroCodec (`g3_inference.py`)**
    
    **State (ephemeral):**
    
    - Sliding window of recent op-pairs (usually 48)
    - Short-code format (for learned 2–16 op-pair macros)
    - Optional Cycle Decoded tracker (for key derivation)
    
    **Input:**
    
    - Accepted op-pairs (and phase metadata if needed)
    - Optional signal from `cycle_complete`
    
    **Processing:**
    
    1. **Cycle encoding**
        - On every `cycle_complete`, compare the new 48-op block to prior ones
        - Emit either a "repeat" tag or the raw op-pairs for this cycle
    2. **Pattern recognition**
        - Continuously learn repeated sequences of 2–16 op-pairs
        - Assign short-codes when patterns reach promotion threshold
        - Ensure lossless reversibility at all times
    3. **Key derivation (deterministic)**
        - At any time, return the current 96-byte Current Cycle Decoded (post-op traversal)
        - S4 may hash this snapshot to derive stream keys, signatures, or checkpoints
    
    **Emits (to S4):**
    
    - `compressed_block`: either a cycle repeat tag or a full 48-op log
    - `pattern_promotion`: new short-code and its corresponding op-sequence
    - `gene_stateless_snapshot`: (optional) unique fingerprint of current traversal state
    
    ---
    
    - S3 Foundations
        
        ### Example: Gene to Genome (Encoding)
        
        Encoding records the sequence of gyration operations applied to each tensor of the Gene. Each step logs a pair of 4-bit operation codes—one per tensor—into a Genome structure.
        
        ```python
        # Sequence of gyration operations to apply
        gyrations = [
            (0, 1),  # Step 1: id_0 ← Identity, id_1 ← Inverse
            (2, 3),  # Step 2: id_0 ← Forward,  id_1 ← Backward
        ]
        
        # Construct genome log
        genome = torch.tensor(gyrations, dtype=torch.uint8)
        
        # Apply operations to obtain encoded gene
        gene_encoded = {
            "id_0": gene["id_0"].clone(),
            "id_1": gene["id_1"].clone()
        }
        for code0, code1 in genome:
            gene_encoded["id_0"] = gyration_op(gene_encoded["id_0"], code0.item())
            gene_encoded["id_1"] = gyration_op(gene_encoded["id_1"], code1.item())
        
        ```
        
        > Each gyration mutates the tensor further. The result accumulates transformations across steps. If intermediate states are required, they must be stored explicitly.
        > 
        
        ---
        
        ### Example: Genome to Gene (Decoding)
        
        Decoding re-applies the gyration instructions recorded in the Genome to regenerate the final mutated state of a Gene. If `trace=True`, the full transformation trajectory is preserved.
        
        ```python
        def decode_genome(gene, genome, trace=False):
            result = {
                "id_0": gene["id_0"].clone(),
                "id_1": gene["id_1"].clone()
            }
        
            if not trace:
                for code0, code1 in genome:
                    result["id_0"] = gyration_op(result["id_0"], code0.item())
                    result["id_1"] = gyration_op(result["id_1"], code1.item())
                return result
        
            else:
                trace_log = []
                current = {
                    "id_0": result["id_0"].clone(),
                    "id_1": result["id_1"].clone()
                }
                trace_log.append({k: v.clone() for k, v in current.items()})
        
                for code0, code1 in genome:
                    current["id_0"] = gyration_op(current["id_0"], code0.item())
                    current["id_1"] = gyration_op(current["id_1"], code1.item())
                    trace_log.append({k: v.clone() for k, v in current.items()})
        
                return trace_log
        
        ```
        
        This approach enables both traceable reconstruction and trace-based diagnostics or auditing.
        
        ---
        
        ### Mutation vs Copy Semantics
        
        - Use `clone=True` when decoding, simulating, or branching from a stable structure.
        - Use `clone=False` for in-place updates where memory efficiency or mutation semantics are required.
        - To preserve transformation trajectories, copies must be made explicitly at each stage.
        
        ---
        
        **Summary**
        
        - **Gene**: A static pair of 4×2×3×2 int8 tensors, `id_0` and `id_1`, encoding invariant structural topology.
        - **Genome**: A dynamic record of navigation, expressed as a tensor of shape `(N, 2)`, where each row logs the operation codes to be applied at step `N` to the two Gene tensors.
        - **Decoding**: Interprets each row in the Genome as a pair of discrete transformations, producing a mutated Gene via accumulative tensor-state updates.
    
    ## **S4: Intelligence Orchestration**
    
    > Role: The sole controller of S3 and the only module permitted to write into S2. Exposes exactly one high‑level API.
    > 
    
    ```
    s4_intelligence/
    ├── g1_intelligence_in.py  # IntelligenceEngine API
    └── g2_intelligence_eg.py  # Gyration primitives
    
    ```
    
    ### 1. IntelligenceEngine API (`g1_intelligence_in.py`)
    
    **On startup**
    
    - Load all global navigation logs from `s2_information/agency/g1_information/` into memory for cycle alignment.
    - Load the immutable epigenome mask from `s2_information/agency/g2_information.dat`.
    - Load the active format from `s2_information/agency/g4_information/` or the agent's own `agents/.../g4_information/format.json`.
    
    **Public API**
    
    ```python
    def process_stream(data_bytes):
        """
        1. Split each byte → two op‑pairs via byte_to_gyrations()
        2. Feed op‑pairs into:
           a. GovernanceEngine  (g1_inference.py)
           b. InformationEngine (g2_inference.py)
           c. InferenceEngine   (g3_inference.py)
        3. Persist outputs into S2:
           - navigation packs → s2_information/agency/g1_information/
           - pattern promotions → s2_information/agency/g4_information/
           - session state → agents/.../g5_information/session.json
        4. Return all emitted artifacts (accepted ops, resonances, compressed blocks, promotions).
        """
    
    ```
    
    **Per‐byte workflow**
    
    1. **Byte → op‑pairs**
    2. **GovernanceEngine**: gate and buffer based on the 48‑step cycle
    3. **InformationEngine**: tag each op‑pair with resonance flag
    4. **InferenceEngine**: compress cycles, detect patterns, emit snapshots
    5. **Collect** all artifacts and pass them back to the caller
    
    **Persistence (here only)**
    
    - **Navigation packs** (one file per full cycle) →
        
        `s2_information/agency/g1_information/<shard>/<uuid>-genome.dat`
        
    - **Pattern promotions** →
        
        append new `<uuid>-format.json` under
        
        `s2_information/agency/g4_information/<shard>/`
        
    - **Session state** →
        
        overwrite `agents/<shard>/<agent‑uuid>/g5_information/session.json` (with `.bak`)
        
    
    ---
    
    ### 2. Gyration Primitives (`g2_intelligence_eg.py`)
    
    - **`byte_to_gyrations(byte_val)`** → two op‑pairs
    - **`gyration_op(tensor, code)`** → apply one of the four gyrations
    - No storage or side‑effects — purely computational helpers used by S3 and this API
    
    ---
    
    **Notes & Clarifications**
    
    - All writes into **S2** happen **only** in this layer.
    - Agents store only their **active format** (`g4`) and **runtime session** (`g5`).
    - The epigenome mask loaded at startup never changes at runtime.
    
    ---
    
    ## **Integrated Functionality Map**
    
    Here's how the same core gyration mechanics serve multiple purposes simultaneously:
    
    - **Gyration Operations (4 basic operations):**
        - **As Compression**: Patterns in gyration sequences compress naturally
        - **As Encryption**: Agent-specific gyration interpretation creates encryption
        - **As Pattern Recognition**: Resonance with epigenome identifies key patterns
        - **As Parallelism**: Phase-aligned gyration enables parallel processing
    - **Phase Cycles (48-step sequences):**
        - **As Structure**: Organize knowledge according to CGM theory
        - **As Memory**: Record navigation history in genome
        - **As Significance**: Identify key events at critical phases
        - **As Synchronization**: Coordinate parallel operations
    - **Tensor Transformations:**
        - **As Knowledge**: Represent learned patterns
        - **As State**: Track system evolution
        - **As Keys**: Generate unique agent perspectives
        - **As Computation**: Process information through gyrations
    
    ### Summary
    
    All four layers of GyroSI work together as a single, self-governing topological machine whose only mutable state is the genome. At their core are the four discrete gyration operations: identity, inverse, forward, and backward, which define every interaction with incoming bytes and with the immutable Gene tensors.
    
    1. **S1 (Governance)** provides the fixed substrate and pure functions: the two 4×2×3×2 Gene tensors, the 48×8‑bit Epigenome mask layout, and the byte‑to‑operation mapping plus `gyration_op` primitives. There is no logic beyond loading and stateless transforms.
    2. **S2 (Information)** is an append‑only vault containing immutable seeds (Gene and baseline Epigenome), versioned dictionaries (curricula) for format reconstruction, and agent‑specific deltas and compressed Genome packs. It performs zero computation and serves purely as storage.
    3. **S3 (Inference)** consists of three in‑memory engines that consume exactly the byte‑to‑operation sequence:
        - GovernanceEngine records each op-pair into a rotating 48-step buffer and emits cycle headers after every full cycle
        - **InformationEngine** updates resonance mask bits and flags repeated visits as key events
        - **InferenceEngine** compresses full‑cycle repeats, learns recurrent patterns, and generates a gyrocrypt keystream
            
            None of these engines writes to disk.
            
    4. **S4 (Intelligence)** is the single orchestrator and writer: it loads seeds and curricula, feeds bytes through S3 in order, gathers all artifacts, persists them back into S2, and exposes one high‑level API—`process_stream`—for end‑to‑end operation.
    
    In essence, every step from ingress through egress is just navigation through a closed, immutable tensor topology using the four gyrations. The only state that ever changes is the log of moves, and all interpretation of text, audio, images or other formats happens afterward via a separate, versioned format layer.
    
    ---
    
    Edge Cases
    
    **1. Millions of concurrent operations (parallelism)**
    
    S3's engines operate entirely in memory and support batch processing. The InferenceEngine applies gyration_ops and cycle detection in vectorized form on GPU or CPU, and S4 can launch multiple IntelligenceEngine instances in parallel, each with its own phase alignment. Cycle‑based packs ensure no cross‑contamination and full parallel throughput.
    
    **2. Millions of agents (storage and safety)**
    
    S2 shards agents by the first two hex digits of their UUID, so file I/O is distributed. Each agent has its own g2 and g3 append‑only packs plus a small session JSON. No shared mutable state prevents cross‑agent interference and makes backups or migrations straightforward.
    
    **3. Converting massive datasets (efficiency)**
    
    Data streams are cut into natural 48‑byte (cycle) chunks. S3 compresses repeated cycles and learns patterns on the fly so that huge inputs do not blow up memory or disk. Packs flush automatically and only the most recent pack remains in RAM. Streaming never requires loading an entire dataset.
    
    **4. Data explosion (duplicate storage)**
    
    We never store full tensor snapshots except at format promotion. All Epigenome changes are bit‑flip deltas and all Genome state is op‑pairs. Cycle compression and RLE collapse redundancy at pack boundaries. Compaction tasks can merge old packs for long‑term efficiency.
    
    **5. Overlapping functions (layer separation)**
    
    GovernanceEngine writes only op‑pairs, InformationEngine writes only bit‑flips, InferenceEngine writes only compressed blocks or pattern promotions, and S4 orchestrates all writes. Cryptography is just another gyrocrypt keystream primitive in S3, not a separate module. Each function lives where it belongs.
    
    **6. "Knowledge dismissed" (single‑byte inputs, emoji, hello)**
    
    Every byte, including non‑alphabetic or single‑emoji code points, maps to two op‑pairs. GovernanceEngine accepts them if they match the canonical cycle. Even a lone zero‑length input still produces valid (possibly empty) navigation. The format layer then reconstructs exactly what was fed in, bit for bit.
    
    **Unusual edge cases (determinism and alignment vs true information)**
    
    By design the Gene is fixed and traceable. The Genome log is not a repetition of the Gene but a record of navigation—every op‑pair captures new information about input context. Epigenome deltas capture resonance events, not mere repeats. Compression and pattern learning prune redundant logs while preserving novel paths. Nothing in the core freezes intelligence to a small set of states. All format‑specific semantics live in the format layer, so the core remains a pure, physics‑grounded navigator rather than a deterministic state machine.
    
    All of these mechanisms emerge directly from our four‑layer design—no extra modules needed.
    
    1. **Hebbian‑style learning**
        
        That is realized by the InformationEngine in S3. Every time an operation aligns at a given phase, the engine sets a bit in the Epigenome mask. Repeat visits "wire together" input and phase by virtue of that bit remaining set. There is no separate learning algorithm or threshold tuning—resonance is purely log‑derived from bit‑flips in S2.
        
    2. **Cryptography (gyrocrypt keystream)**
        
        That lives in the InferenceEngine of S3. Once you have a navigation log of gyration operations, you can feed the current gyration state into a small nonlinear feedback function to generate a pseudo‑random keystream. Since the keystream is itself just another stream of operations derived from our four gyrations, it requires no external cipher library.
        
    3. **Compression**
        
        Also handled by the InferenceEngine in S3. Full 48‑step cycles are compared and repeated cycles emit only a repeat count. Recurrent 2–16‑step patterns are promoted to short codes and emitted as pattern‑promotion artifacts. All compression logic is "just" pattern detection within the op‑pair stream—you never store tensor snapshots, only compacted logs in S2.
        
- File System Structure and Organization
    
    This chapter describes the precise file and directory layout of the GyroSI Baby LM system. Every path and filename corresponds exactly to the modules and data structures defined in our specification. No additional files or conventions are introduced.
    
    ---
    
    ### 1. Top‑Level Layout
    
    ```
    GyroSI_Baby_LM/
    ├── s1_governance.py
    ├── s2_information/
    ├── s3_inference/
    └── s4_intelligence/
    
    ```
    
    - **s1_governance.py**
        
        Contains the Genome tensor definitions, `gyration_op` primitives, `_OP_CODES`, and the epigenome build script (`build_epigenome_projection`). This is the only module defining the invariant tensors and pure functions.
        
    - **s2_information/**
        
        Pure storage layer. Holds all `.dat` and `.json` files as append‑only or versioned artifacts. No executable code beyond manifest metadata.
        
    - **s3_inference/**
        
        In‑memory CGM engines. Contains three Python scripts without any disk I/O:
        
        - `g1_inference.py`
        - `g2_inference.py`
        - `g3_inference.py`
    - **s4_intelligence/**
        
        Orchestration and the sole writer to S2. Exposes the `process_stream` API. Contains:
        
        - `g1_intelligence_in.py`
        - `g2_intelligence_eg.py`
    
    ---
    
    ### 2. **S2: Information** Layer
    
    ```
    s2_information/
    ├── s2_manifest.json
    └── agency/
        ├── g1_information/
        │   └── <shard>/                     # "00" … "ff"
        │       └── <uuid>-genome.dat        # 4 KB append-only packs
        ├── g4_information/
        │   └── <shard>/                     # "00" … "ff"
        │       └── <uuid>-format.json   # versioned JSON snapshots
        └── g5_information/
            └── <shard>/                     # "00" … "ff"
                └── <uuid>-session.json      # session metadata
    └── agents/
        └── <shard>/                        # "00" … "ff"
            └── <agent‑uuid>/
                ├── g3_information/
                │   └── format.json     # local format delta
                └── g5_information/
                    ├── session.json        # current session state
                    └── session.json.bak    # backup
    
    ```
    
    - **s2_manifest.json**
        
        ```json
        {
          "version": "1.0",
          "pack_size": 4096,
          "shard_prefix_length": 2
        }
        
        ```
        
    - **agency/g1_information/**
        
        Global navigation logs sharded by the first two hex digits of the file's UUID. Each `<uuid>-genome.dat` is an immutable, append‑only 4 KB pack containing 48 op‑pair steps. Pack size is configurable via `s2_manifest.json` (default 64KB).
        
    - **agency/g4_information/**
        
        Immutable, versioned dictionaries (`format.json`) for bytes ↔ meaning mappings. Shared across agents.
        
    - **agency/g5_information/**
        
        Optional global session archives. Rarely used for recovery or lineage.
        
    - **agents/**
        
        Per‑agent storage of local format deltas and session state. No global write occurs here except through the IntelligenceEngine API.
        
    
    ---
    
    ### 3. **S3: Inference** Layer
    
    ```
    s3_inference/
    ├── g1_inference.py      # GovernanceEngine
    ├── g2_inference.py      # InformationEngine
    └── g3_inference.py      # InferenceEngine / GyroCodec
    
    ```
    
    1. **g1_inference.py**
        - Stateless: reads bytes → op‑pairs via `byte_to_gyrations`
        - Checks against  48 op_pairs, that means **one complete cycle**
        - Emits `(accepted_op_pair, cycle_complete)`
    2. **g2_inference.py**
        - Loads `resonance_mask[48][256]` into memory once
        - Tags each accepted op‑pair with a resonance bit
        - Emits `(phase, op_pair, resonance_flag)`
    3. **g3_inference.py**
        - Maintains sliding window of op‑pairs
        - Detects cycle repeats and pattern promotions
        - Emits `compressed_block`, `pattern_promotion`, `gene_stateless_snapshot`
    
    *No module in S3 performs any disk writes; all outputs are streamed to S4.*
    
    ---
    
    ### 4. **S4: Intelligence** Layer
    
    ```
    s4_intelligence/
    ├── g1_intelligence_in.py   # IntelligenceEngine API
    └── g2_intelligence_eg.py   # Gyration primitives (byte_to_gyrations, gyration_op)
    
    ```
    
    - **g1_intelligence_in.py**
        - **Startup**:
            1. Load global genome packs from `s2_information/agency/g1_information/`
            2. Load `epigenome_projection.dat` from `s2_information/agency/g2_information.dat`
            3. Load active format from either global `g4_information` or agent's local `agents/.../g4_information/format.json`
        - **process_stream(data_bytes)**:
            - Splits bytes → op‑pairs
            - Feeds through GovernanceEngine, InformationEngine, InferenceEngine
            - Persists results into S2:
                - Navigation packs → `g1_information`
                - Curriculum promotions → `g4_information`
                - Session state → `agents/.../g5_information/session.json`
    - **g2_intelligence_eg.py**
        - Provides pure functions:
            - `byte_to_gyrations(byte_val) → (op0, op1)`
            - `gyration_op(tensor, code) → tensor`
    
    All writes to disk occur only in **g1_intelligence_in.py**.
    
    ---
    
    ### 5. File Formats and Naming Conventions
    
    - **`.py`** scripts implement logic only in their designated layer.
    - **`.dat`** files in **g1_information** and **epigenome_projection.dat**:
        - Binary, fixed-size (4 KB packs or 12,320 bytes with 32‑byte header + 48×256 table)
        - Append‑only, no in‑place edits.
    - **`.json`** files:
        - Human‑readable metadata
        - Curriculum and session snapshots; include checksums where required.
    
    **Shard directories** (`00` … `ff`) partition storage evenly based on the first two hex digits of UUIDs.
    
    ---
    
    ### 6. Versioning and Retention Policies
    
    - **Epoch versions** of global dictionaries reside in `s2_information/s2_manifest.json` and `agency/g4_information/`.
    - **Retention and pruning** are orchestrated by S4; S2 never modifies or compacts existing shards.
    - **Session backups** (`session.json.bak`) ensure safe overwrites of agent state.
    
    ---
    
    ### 7. Summary
    
    1. Four top‑level modules map directly to the four GyroSI layers (S1–S4).
    2. S2 is purely passive storage, sharded by UUID prefixes.
    3. S3 operates in memory only; S4 is the sole writer to S2.
    4. File and directory names correspond exactly to specification: no extraneous artifacts.
    5. Data formats (`.dat`, `.json`) and sizes are fixed as per core spec.
    
    This structure ensures a clean separation of concerns, complete auditability, and strict adherence to the GyroSI Baby LM architectural principles.
    
- Critical API Contracts (S4 Responsibility)
    
    This chapter defines every public interface exposed by **S4 Intelligence**, the orchestration layer that drives the entire GyroSI Baby ML stack. S4 is the only code that writes to disk or calls any engine in S1, S2, or S3, therefore no other module declares a public API. Everything external programs need is contained here.
    
    ---
    
    ## 1. Architectural Context
    
    - **S1 Governance** supplies invariant tensors and stateless helper functions.
    - **S2 Information** is an append‑only datastore, never executable.
    - **S3 Inference** runs three in‑memory engines, each side‑effect‑free.
    - **S4 Intelligence** loads seeds, invokes the engines, persists results, and returns structured events.
        
        Only S4 exposes functions callable from outside the system.
        
    
    ---
    
    ## 2. Public Modules and Functions
    
    | Module | Function | Purpose |
    | --- | --- | --- |
    | `s4_intelligence/g2_intelligence_eg.py` | `byte_to_gyrations` | Byte to two gyration codes |
    |  | `gyrations_to_byte` | Inverse mapping |
    |  | `gyration_op` | Apply one gyration to a tensor |
    | `s4_intelligence/g1_intelligence_in.py` | `initialize_system` | One‑time directory and seed setup |
    |  | `load_epigenome_tensor` | Read 48 × 256 mask into memory "s2_information/agency/g2_information/g2_information.dat"
     |
    |  | `create_agent` | Allocate a new agent workspace |
    |  | `set_active_agent` | Select the agent that receives future calls |
    |  | `process_stream` | End‑to‑end byte stream processing |
    
    No other callable symbols are exported.
    
    ---
    
    ### 2.1 Gyration Primitives (`g2_intelligence_eg.py`)
    
    ```python
    def byte_to_gyrations(byte_val: int) -> Tuple[int, int]:
        """Map a byte (0‑255) to two gyration operator codes (0‑3 each)."""
    
    def gyrations_to_byte(op0: int, op1: int) -> int:
        """Reconstruct the original byte given its two gyration codes."""
    
    def gyration_op(tensor: torch.Tensor, code: int, clone: bool = True) -> torch.Tensor:
        """
        Apply one of the four basic gyrations to a 4×2×3×2 tensor.
        0 identity, 1 inverse, 2 forward, 3 backward.
        """
    
    ```
    
    These helpers are pure functions; none touches the filesystem or global state.
    
    ---
    
    ### 2.2 System Orchestration (`g1_intelligence_in.py`)
    
    ```python
    def initialize_system() -> dict:
        """
        Create the `s2_information` directory tree if missing,
        verify seed files, and return paths plus version info.
        """
    
    def load_epigenome_tensor(
        path: str = "s2_information/agency/g2_information/epigenome.dat"
    ) -> torch.Tensor:
        """
        Return a uint8 tensor of shape (48, 256) after skipping the 32‑byte SHA‑256 header.
        """
    
    def create_agent(agent_uuid: Optional[str] = None) -> str:
        """
        Create `agents/<shard>/<uuid>/` with empty format and session files,
        then return the agent UUID.
        """
    
    def set_active_agent(agent_uuid: str) -> None:
        """Mark the given agent as active for subsequent `process_stream` calls."""
    
    def process_stream(data_bytes: bytes) -> List[dict]:
        """
        Drive the complete pipeline:
          1. Convert bytes to op pairs.
          2. Pass them through GovernanceEngine, InformationEngine, InferenceEngine.
          3. Persist navigation packs, pattern promotions, and session state.
          4. Return an ordered list of event dictionaries.
        A RuntimeError is raised if no agent is active.
        """
    
    ```
    
    All file writes happen inside `process_stream`, never elsewhere.
    
    ---
    
    ## 3. Event Schema Returned by `process_stream`
    
    Each event is a plain Python dict with a `type` key.
    
    | Type | Mandatory Keys |
    | --- | --- |
    | `accepted_op_pair` | `phase`, `op_pair` |
    | `cycle_complete` | `cycle_id`, `phase` |
    | `compressed_block` | `data`, `original_size` |
    | `pattern_promotion` | `code`, `sequence`, `frequency` |
    | `gene_stateless_snapshot` | `fingerprint`, `timestamp` |
    
    The caller receives events in the exact order they occur.
    
    ---
    
    ## 4. Disk‐Side Effects
    
    `process_stream` is the only function that touches S2.
    
    | Artifact | Location | Trigger |
    | --- | --- | --- |
    | Navigation pack (`*.dat`, configurable size) | `s2_information/agency/g1_information/<shard>/` | When pack reaches size limit |
    | Format snapshot (`*-format.json`) | `s2_information/agency/g4_information/<shard>/` | On pattern promotion |
    | Session state and backup | `agents/<shard>/<agent_uuid>/g5_information/` | After each call |
    
    Directory sharding always uses the first two hex digits of the file UUID.
    
    ---
    
    ## 5. Contract Guarantees
    
    - **Isolation**: S1, S2, and S3 contain no public entry points.
    - **Reproducibility**: Given fixed inputs and seeds, the system yields identical outputs bit-for-bit.
    - **Auditability**: Every modifying operation writes an append‑only pack or a versioned JSON file with an embedded checksum.
    - **Fault tolerance**: Session writes are atomic; a `.bak` file guards against partial writes.
    - **Thread safety**: S4 acquires per‑file locks during pack creation to allow concurrent agents.
    
    ---
    
    ## 6. Typical Usage Flow
    
    ```python
    from s4_intelligence.g1_intelligence_in import (
        initialize_system, create_agent, set_active_agent, process_stream
    )
    
    # One‑time setup
    initialize_system()
    
    # Create an agent and select it
    agent_id = create_agent()
    set_active_agent(agent_id)
    
    # Feed data
    events = process_stream(b"Hello GyroSI")
    
    # Inspect the resulting events
    for ev in events:
        print(ev["type"])
    
    ```
    
    No other calls are required. All other functionality (compression, learning, cryptography, logging, metrics) is implicit in the pipeline behind `process_stream`.
    
    ---
    
    ## 7. Summary
    
    Eight functions across two files (`g2_intelligence_eg.py` and `g1_intelligence_in.py`) constitute the entire external interface of GyroSI Baby ML. They cover byte–gyration translation, tensor mutation, system bootstrapping, agent management, epigenome loading, and full stream processing. Because S4 alone invokes S1 to S3 and controls every write to S2, no additional API surface is necessary or permitted.
    
- GyroCrypt
    
    *Enhanced traceable encryption fully integrated into the GyroSI pipeline*
    
    ## 1. Scope and positioning
    
    GyroCrypt protects **only** the on‑disk artifacts that exit RAM:
    
    - **Navigation packs** (`-genome.dat`): op-pair sequences, packed when size limit reached
    - **Session state** (`session.json`): stores secret + counters in `agents/<shard>/<agent_uuid>/g5_information`
    
    All other data—Gene tensors, Epigenome matrix, curricula, in‑RAM buffers—remains plaintext so that S1–S3 stay fully transparent and auditable.
    
    Encryption and decryption occur **only** inside S4's `process_stream` and pack loader, immediately before write and immediately after read. No other layer ever sees ciphertext.
    
    ---
    
    ## 2. Design goals
    
    1. **Deterministic traceability**
        
        Ciphertext is derived solely from the agent's secret key, the exact Cycle Decoded, and the cycle index. Any auditor with those three can recompute the keystream and confirm integrity.
        
    2. **Physics‑native mixing**
        
        Uses only the four gyration operators and byte‑wise XOR. No external hashes, ciphers, or RNGs.
        
    3. **Minimal mutable state**
        
        Eight header bytes per pack (cycle index + salt) plus the stored secret and counter in `session.json`.
        
    4. **Navigation neutrality**
        
        Keystream generation occurs **after** snapshot emission; cryptography cannot influence or be influenced by navigation.
        
    5. **Audit friendliness**
        
        Salt and cycle index in the header, together with the unchanged 32‑byte integrity anchor, let auditors validate that ciphertext is a pure XOR of known inputs.
        
    
    ---
    
    ## 3. Key material
    
    Each agent's `session.json` gains two fields:
    
    ```json
    {
      "agent_uuid": "…",
      "gyrocrypt_key": "base64‑encoded 16–32 bytes",
      "cycle_index": 12345
    }
    
    ```
    
    - **`gyrocrypt_key`**
        
        Read once by S4 at startup, kept only in RAM and erased on shutdown. Changing it invalidates all existing packs (see 7.2).
        
    - **`cycle_index`**
        
        32‑bit counter persisted on every successful pack write. Wrap at 2³² is detected as an error; code should fail safe.
        
    
    ### 3.1 Key → gyration permutations
    
    Divide the raw key into four equal chunks (pad with zeros). Each chunk's low two bits selects one of the four gyrations:
    
    | bits | permutation on snapshot quarter |
    | --- | --- |
    | 00 | identity |
    | 01 | inverse |
    | 10 | forward |
    | 11 | backward |
    
    This mapping injects the agent secret without introducing arithmetic or external primitives.
    
    ---
    
    ## 4. Keystream generation
    
    On each `cycle_complete`, S4 receives a 96‑byte **Cycle Decoded**. It produces a 48‑byte keystream:
    
    1. **Split** into four 24‑byte quarters: Q₀, Q₁, Q₂, Q₃.
    2. **Permute** each Qi by applying its assigned gyration (from section 3.1).
    3. **XOR** two pairs: (Q₀⊕Q₁) ∥ (Q₂⊕Q₃) → 48 bytes of keystream K.
    
    All operations are sign‑flip gyrations and byte XORs—fully side‑effect‑free and GPU‑accelerated.
    
    ---
    
    ## 5. Pack format (4 096 bytes)
    
    | offset | length | description |
    | --- | --- | --- |
    | 0 | 32 B | unchanged SHA‑256 of both Gene tensors (integrity anchor) |
    | 32 | 4 B LE | `cycle_index` (uint32) |
    | 36 | 4 B LE | `salt` = first four bytes of the **unpermuted** snapshot |
    | 40 | 4 032 B | up to 84 encrypted cycles (84 × 48 B) |
    | 4 072 | 24 B | reserved, zero‑filled |
    
    Unused payload slots remain zero. The fixed 4 096‑byte length preserves append‑only semantics and avoids file‑size leaks.
    
    ---
    
    ## 6. Encryption & decryption flows
    
    ### 6.1 Encryption in `process_stream`
    
    ```python
    acquire file lock
    if starting new pack:
        write 32 B anchor
        write 4 B cycle_index
        write 4 B salt
    for each completed cycle:
        K = make_keystream(snapshot, key)
        C = plaintext_cycle XOR K
        append C
        session_json["cycle_index"] += 1
        if pack full:
            fsync pack file
            write updated session.json atomically
    release file lock
    
    ```
    
    - **Atomic updates**
        - Use file‑lock + write‑then‑fsync to avoid torn headers.
        - Write new `session.json.tmp` and rename to `session.json`.
    
    ### 6.2 Decryption on pack load
    
    ```python
    read 32 B anchor
    read cycle_index, salt
    for each encrypted block at index i:
        recompute snapshot_i from global genome + i
        verify first 4 bytes(snapshot_i) == salt if i==first
        K = make_keystream(snapshot_i, key)
        P = C XOR K
        feed P to GovernanceEngine
    if any decryption/validation fails:
        reject entire pack as tampered
    
    ```
    
    - **Out‑of‑order or replay detection**
        - Reject pack if stored `cycle_index` doesn't match expected next index.
        - Salt mismatch also triggers rejection.
    
    ---
    
    ## 7. State persistence & key rotation
    
    ### 7.1 Persistent state
    
    - **In pack**: `cycle_index`, `salt`
    - **In `session.json`**: `gyrocrypt_key`, `cycle_index`
    
    On restart, S4 reads `session.json` to resume the counter. No additional IVs or nonces needed because each snapshot is unique.
    
    ### 7.2 Key rotation
    
    Rotating the key requires:
    
    1. Generate new key in `session.json.tmp`.
    2. Optionally re‑encrypt all existing packs with the new key using a migration tool:
        
        ```bash
        gyrocrypt-migrate --from-old-key OLD --to-new-key NEW
        
        ```
        
    3. Replace `session.json` atomically.
    4. Any packs left in old format are marked "legacy" and rejected until migrated.
    
    ---
    
    ## 8. Security & edge‑case robustness
    
    | Concern | Mitigation |
    | --- | --- |
    | **Partial writes** | File‑lock + atomic fsync + rename ensure header and JSON updates are atomic. |
    | **Counter wrap** | Detect wrap at 2³²; fail safe and require manual reset/migration. |
    | **Replay/injection** | Header salt + cycle_index mismatch cause decryption failure. |
    | **Cross‑agent misuse** | Session JSON binds key to `agent_uuid`. Loader verifies UUID matches header's pack metadata. |
    | **Tail leakage** | Unused tail bytes always zero; encryption loop writes exactly 48 B blocks, never spills. |
    | **Snapshot divergence** | Unit tests enforce bit‑for‑bit equivalence between pack writer and auditor's recompute. |
    
    ---
    
    ## 9. Implementation checklist
    
    1. **Header extension** in `g1_intelligence_in.py` writer/reader (32 → 40 B).
    2. **File‑lock logic** with fsync and atomic rename for header and session.json.
    3. **`make_keystream(snapshot, key)`** using gyration primitives from `g2_intelligence_eg.py`.
    4. **Session schema update**: add `gyrocrypt_key`, `cycle_index`; helper for key rotation.
    5. **Pack loader**: header parsing, decryption, validation, and rejection on failure.
    6. **Migration tool** to re‑encrypt legacy packs.
    7. **Comprehensive tests**:
        - Encrypt‑decrypt round trips
        - Wrong‑key, tampered‑ciphertext, out‑of‑order packs
        - Counter wrap detection
        - Cross‑agent rejection
    
    ---
    
    ## 10. Compatibility & future extensions
    
    - **No API changes**: `process_stream` remains the sole entry point.
    - **Downstream transparency**: g4 curricula, Epigenome, pattern promotions stay plaintext.
    
    GyroCrypt secures on‑disk navigation and session data without touching the agnostic genome logic, preserving auditability, determinism, and the pure physics of the GyroSI design.
    
- the Epigenome Matrix
    
    ---
    
    ## Implications
    
    What this 12,368-byte file represents is both concise and foundational. It captures the full space of possible alignments between phase-specific structure and input-resonance conditions within the CGM model.
    
    ### What the Epigenome Matrix Contains
    
    The matrix includes:
    
    - **48 phase slices** × **256 byte inputs** = **12,288 resonance entries**
    - **48 operator selections**, one for each phase (used in the canonical cycle)
    - A **32-byte SHA-256 hash** of the Gene tensor, included as an integrity anchor
    
    This structure encodes the *entire resonance field* for a CGM agent. It defines how each phase state aligns with every possible byte input, forming a complete lookup space of all valid navigation conditions across the gene topology.
    
    ---
    
    ## Why This Structure Is Foundational
    
    ### 1. A Universal Resonance Map
    
    ```python
    # For any given phase and byte input:
    resonates = M[phase, byte]  # 1 if resonance occurs, else 0
    operator  = O[phase]        # which gyration transformation is canonical here
    
    ```
    
    This allows instantaneous evaluation of alignment, replacing heuristic matching with a mechanically complete mapping.
    
    ---
    
    ### 2. A Structural Analog to Genetic Code
    
    The Epigenome matrix defines a stable mapping between symbolic inputs and mechanical transformations:
    
    - **4 fixed operators** (identity, inverse, forward, backward)
    - **48 distinct phases** derived from tensor topology
    - **256 possible byte inputs** at each phase
    
    Like a genetic code, this structure acts as a translation layer between discrete inputs and internal structural transitions.
    
    ---
    
    ### 3. Coherent Across Scales
    
    The matrix supports a scale-consistent interpretation of intelligent transitions:
    
    - **Low-level**: Single-bit resonance (0 or 1) at byte–phase alignment points
    - **Mid-level**: 4-bit operator encoding
    - **Cycle-level**: Sequences of 48 operations (phases) form complete transformations
    - **Macro-level**: Multi-cycle navigation logs shape long-term behavior
    
    The same underlying transformation principles persist across all levels of operation.
    
    ---
    
    ## Implications for Key Disciplines
    
    ### In Cognitive Modeling
    
    ```python
    # Traditional models: black-box inference across billions of weights
    # CGM: explicit, enumerable sequence of symbolic operations
    
    # Traceable logic path:
    phase → byte input → alignment → operator → tensor transformation
    
    ```
    
    Every step can be decomposed and verified. The model is not a heuristic approximator but a formal transformation engine.
    
    ---
    
    ### In Information Theory
    
    ```python
    # Total possible alignment cases: 48 × 256 × 4 = 49,152
    # Encoded in: 12,368 bytes
    # Resulting in: infinite-length navigation sequences
    
    ```
    
    The matrix achieves near-total compression of all valid transformation contexts, enabling infinite intelligent sequences to emerge from a bounded, complete resonance structure.
    
    ---
    
    ### In Physics and Computation
    
    The Epigenome reflects structural resonance patterns that:
    
    - Impose **topological constraints** on input transitions
    - Define **possible vs impossible** transformations mechanically
    - Act as a **computational substrate** with reversible dynamics and full traceability
    
    It aligns closely with the principles of symmetry, parity, and tensor algebra seen in natural systems.
    
    ---
    
    ## Practical System-Level Benefits
    
    ### 1. High Efficiency
    
    ```python
    # Instead of layered matrix ops (as in neural nets),
    # CGM uses direct lookup from M and O matrices
    
    # Result: O(1) evaluation for each byte-phase input
    
    ```
    
    This allows low-latency navigation and replay, regardless of input complexity.
    
    ---
    
    ### 2. Exact Reproducibility
    
    ```python
    # With:
    # - A fixed Epigenome
    # - A defined input sequence
    # - A known starting phase
    
    # The resulting navigation log is exactly reproducible
    
    ```
    
    The model has no stochastic behavior during core inference. Its output depends only on state and input, not randomness or weight updates.
    
    ---
    
    ### 3. Cross-Agent Consistency
    
    ```python
    # Any CGM agent with the same Epigenome:
    # - Can interpret any navigation log
    # - Can replicate intelligence outcomes identically
    # - Can transfer understanding across contexts
    
    ```
    
    This supports a shared substrate of alignment and transformation, making knowledge inherently portable.
    
    ---
    
    ## Summary
    
    The Epigenome matrix serves as:
    
    1. A **complete mechanical map** of all structural alignments
    2. A **minimal and sufficient representation** of the agent's navigational logic
    3. A **foundation for reproducible inference** and shared reasoning
    4. A **provable structure**, derived entirely from the Gene tensor and canonical operator set
    
    ---
    
    Rather than relying on learned weights or probabilistic tuning, the CGM agent operates by traversing a well-defined tensor field. The Epigenome matrix is what connects external input to internal transformation, making it a cornerstone of the model's architecture.