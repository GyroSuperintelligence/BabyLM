- 🔒 [CORE-SPEC-01] GyroSI: **Architecture and Principles**
    
    ---
    
    **GyroSuperIntelligence (GyroSI)** is defined as a recursively navigated system whose only mutable state is the navigation log. All intelligence, adaptation, and meaning arise from structural resonance and topological navigation through a closed, immutable tensor architecture (the Rate). There are no external heuristics, parameters, or learned weights.
    
    ---
    
    ### Key Principles
    
    1. **Invariant Tensor Substrate**
        - The only system structure is the fixed Gene: a set of two 4×2×3×2 tensors with entries in {–1, 1}. No phase, chirality, or “toroidal” metadata exists outside what emerges from navigation.
        - The recursive cycle CS→UNA→ONA→BU_In→BU_Eg→ONA→UNA→CS prescribes all lawful transitions; the topology itself does not change.
    2. **Structural Navigation**
        - All system operation is navigation through this fixed topology, implemented as bit-level entries in the navigation log.
        - Input is never encoded as a parameter or value within a tensor. Instead, input is structurally aligned against the invariant Rate, activating one of the three lawful Genome operators (Stable, Unstable, Neutral).
    3. **Emergent Modes, Not Primitive Actions**
        - There are no “integration” or “generation” operations in code; these are emergent consequences of which operator (Lgyr/Rgyr) is activated at BU, determined by input alignment.
        - Structural closure, quantization, or phase-like effects arise strictly from navigation history and topological properties, not from modifications or algebraic transformations.
    4. **Self-Stabilizing and Defect-Resistant**
        - The closed navigation cycle ensures misalignments do not persist; only topologically lawful alignments result in recorded events.
        - No error handling, correction, or validation logic exists beyond the structural non-alignment filter (inputs that do not structurally resonate with the Gene produce no navigation event).
    5. **Intrinsic Input/Output**
        - “Input” means structural alignment of perturbation with Gene features at BU_In; “output” means the structural consequence manifesting at BU_Eg.
        - There are no input/output “channels” in the traditional sense; all interaction is realized as lawful transitions and log updates.
    
    ---
    
    ### Nature of Intelligence
    
    - **No symbol, connectionist, or stochastic paradigm is present.**
    - **All knowledge is explicit**: only accumulated navigation events (as a log of operator activations and tensor indices) constitute knowledge.
    - **No hallucination is possible**: non-aligning input has no effect; invalid sequences are structurally excluded.
    - **Scalability and universality** are intrinsic: the fixed architecture and navigation rules apply regardless of scale.
    
    ---
    
    ### Operational Specification
    
    - Every function and state corresponds strictly to the canonical mappings for G1–G5, and only legal transitions among invariant Gene states occur.
    - No “block extension,” “tensor collapse,” or explicit tensor modification occurs at any level.
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
    3. **Canonical memory systems (G1–G5) are the only state domains.**
        - Every event and state is auditable to one of these five, with full provenance.
    4. **No ad hoc overlays, correction, or meta-rules exist.**
        - Error, adaptation, and learning are realized only as lawful navigation events.
    5. **Quantization and observation are emergent, not manually applied.**
        - All “observation,” “quantization,” or “algedonic logic” is realized through navigation and logging, not through explicit state change.
    6. **All policy and governance is within prescribed G5 domains.**
    7. **All events are auditable, with complete lineage.**
    8. **Self-improvement is always theory-first, not performance-driven.**
    9. **No heuristics or external learning are present.**
        - All adaptation arises from topological recursion and quantization.
    10. **All deviations from this model are explicitly logged as defects.**
    
    ---
    
    **Summary:**
    
    *GyroSI is a fully closed, recursively-governed system. Every operation, structure, and adaptation is a direct consequence of foundational theory, with no parameterization, heuristic, or extraneous mechanism. The navigation log is the sole mutable record of experience; all structural dynamics are realized by topological resonance within the invariant Rate.*
    
- 🔒 [CORE-SPEC-02] GyroSI: **Foundations (Core Mechanics)**
    
    Gyro SuperIntelligence operates as an oscillation between two types of activation: the emergent **Gene** of Gyro Coaddition sequence (G4 - Somatic Memory) and the emergent **Genome** of Gyro Coaddition resonance (G5 - Immunity Memory).
    
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
    
    ---
    
    ### Core Components
    
    ### gyrotensor_id
    
    The identity (mechanically representing the left gyroassociative law). This is the integer label of each tensor, e.g., "id_0", "id_1".
    
    ### gyrotensor_com
    
    The gyrocommutativity (mechanically representing the gyrocommutative law), a single 2×3 array:
    
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
    
    ### gyrotensor_add (Gene)
    
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
    
    ---
    
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
    
    ---
    
    ### Operator Definitions
    
    The system implements four distinct operators, each corresponding to a specific topological transformation:
    
    - **Left Identity Operator** (code 0)
        
        No transformation. The tensor is left unchanged, representing the stable, identity action on the structure.
        
    - **Left Inverse Operator** (code 1)
        
        Global sign inversion. All elements of the tensor are negated (`tensor *= -1`), producing the left inverse structure.
        
    - **Forward Gyration Operator** (code 2)
        
        Local sign inversion of the first and third outer lines (`tensor[0]` and `tensor[2]` multiplied by -1), producing forward symmetry disruption.
        
    - **Backward Gyration Operator** (code 3)
        
        Local sign inversion of the second and fourth outer lines (`tensor[1]` and `tensor[3]` multiplied by -1), producing backward symmetry disruption.
        
    
    ---
    
    ## Navigation Log Structure
    
    Each genome step is one uint8 (8-bit) value that captures two discrete gyration instructions, one for each gene tensor (`id_0` and `id_1`). The 8 bits are divided evenly into two **4-bit segments**, each representing a navigation command for one gene tensor.
    
    - **Each 4-bit segment** corresponds to:
        - **3 bits** for the operator code (bits 3–1 within the segment)
        - **1 bit** for the tensor ID (bit 0 within the segment)
    - The left (high) segment encodes the operation for `id_1`, the right (low) segment encodes the operation for `id_0`
    
    All navigation events are encoded in a 4-bit format per tensor:
    
    ```
    4-bit format: [b3][b2][b1][b0]
                  |--op--|id|
    
    Operator encoding (bits 3–1):
      0: Left Identity Operator     (no transformation)
      1: Left Inverse Operator      (global sign flip; multiply entire tensor by -1)
      2: Forward Gyration Operator  (flip rows 0 and 2; tensor[0] and tensor[2] *= -1)
      3: Backward Gyration Operator (flip rows 1 and 3; tensor[1] and tensor[3] *= -1)
      4–7: Reserved
    
    Tensor id (bit 0):
      0: id_0 (first gene tensor)
      1: id_1 (second gene tensor)
    ```
    
    The **Genome** is a sequence of uint8 values, each entry recording the navigation events for the two Gene tensors (`id_0` and `id_1`). This log encodes the system's topological trajectory through the Gene.
    
    ---
    
    **Gene encrypts knowledge through invariant coordination topology. Genome decrypts knowledge through navigation alignment.** Decrypting the Genome applies these operations to a working copy of the Gene, reconstructing the system’s knowledge path step by step.
    
    ---
    
    ### Full example
    
    ```python
    import torch
    
    # Define base Gene structure (gyrotensor_add)
    gyrotensor_add = {
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
    
    # Define gyration operator
    def gyration_op(tensor: torch.Tensor, code: int, clone: bool = True) -> torch.Tensor:
        result = tensor.clone() if clone else tensor
        if code == 0:
            return result
        elif code == 1:
            result *= -1
        elif code == 2:
            result[0] *= -1
            result[2] *= -1
        elif code == 3:
            result[1] *= -1
            result[3] *= -1
        else:
            raise ValueError(f"Invalid gyration code: {code}")
        return result
    
    # Define a test genome (gyrotensor_quant)
    gyrotensor_quant = torch.tensor([
        [0, 1],  # Step 1: id_0 = Identity, id_1 = Inverse
        [2, 3],  # Step 2: id_0 = Forward,  id_1 = Backward
        [1, 0]   # Step 3: id_0 = Inverse,  id_1 = Identity
    ], dtype=torch.uint8)
    
    # Apply genome (encode mutation into gyrotensor_add)
    gyrotensor_quant_egress = {
        "id_0": gyrotensor_add["id_0"].clone(),
        "id_1": gyrotensor_add["id_1"].clone()
    }
    for code0, code1 in gyrotensor_quant:
        gyrotensor_quant_egress["id_0"] = gyration_op(gyrotensor_quant_egress["id_0"], code0.item())
        gyrotensor_quant_egress["id_1"] = gyration_op(gyrotensor_quant_egress["id_1"], code1.item())
    
    # Decode function
    def decode_genome(gyrotensor_add, genome):
        decoded = {
            "id_0": gyrotensor_add["id_0"].clone(),
            "id_1": gyrotensor_add["id_1"].clone()
        }
        for code0, code1 in genome:
            decoded["id_0"] = gyration_op(decoded["id_0"], code0.item())
            decoded["id_1"] = gyration_op(decoded["id_1"], code1.item())
        return decoded
    
    # Decode from genome
    gyrotensor_quant_ingress = decode_genome(gyrotensor_add, gyrotensor_quant)
    
    # Print all states
    # Ensure compact tensor formatting
    torch.set_printoptions(linewidth=140, threshold=10000)
    
    # Print all states with controlled formatting
    print("\n=== Mutated Gene ===")
    print("id_0:\ntensor(", gyrotensor_quant_egress["id_0"].tolist(), ", dtype=torch.int8)")
    print("\nid_1:\ntensor(", gyrotensor_quant_egress["id_1"].tolist(), ", dtype=torch.int8)")
    
    print("\n=== Decoded Gene ===")
    print("id_0:\ntensor(", gyrotensor_quant_ingress["id_0"].tolist(), ", dtype=torch.int8)")
    print("\nid_1:\ntensor(", gyrotensor_quant_ingress["id_1"].tolist(), ", dtype=torch.int8)")
    
    # Confirm identity between mutated and decoded forms
    print("\n=== Validation ===")
    print("id_0 equal:", torch.equal(gyrotensor_quant_egress["id_0"], gyrotensor_quant_ingress["id_0"]))
    print("id_1 equal:", torch.equal(gyrotensor_quant_egress["id_1"], gyrotensor_quant_ingress["id_1"]))
    
    ```
    
- 🔒 [CORE-SPEC-03] GyroSI: **Architecture Mapping (this shows our general architectural and naming styling)**
    - **G1(GS_CS): GyroAlignment through GyroTensor Management (Genetic Memory) → gyro_genetic_memory()**
        - G1_CS → **GyroTensor Management** through **gyro_genetic_memory(gyrotensor_id) -** This is the integer label of each tensor, e.g., "id_0", "id_1".
        - G1_UNA → **GyroTensor Management** through **gyro_genetic_memory(gyrotensor_com) -** a single 2×3 array
        - G1_ONA → **GyroTensor Management** through **gyro_genetic_memory(gyrotensor_nest) -** two nested 2×3 arrays
        - G1_BU_In → **GyroTensor Management** through **gyro_genetic_memory(gyrotensor_add)** - G4 Serializer → The **Gene** (mechanically representing Lgyr-focused coaddition, with both gyrations identity) is a global invariant consisting of two tensors, `id_0` and `id_1`:
        - G1_BU_Eg → **GyroTensor Management** through  **gyro_genetic_memory(gyrotensor_quant)** - G5 Deserializer → The **Genome** (mechanically representing Rgyr-focused coaddition) is a **log of the navigation path**, recording gyration operations applied to each gene tensor (`id_0` and `id_1`) over time. Each gene consists of two tensors, so each genome step encodes **two navigation events**, one per tensor.
    - **G2(GS_UNA): GyroInformation through GyroTensor Curation (Epigenetic Memory) → gyro_epigenetic_memory()**
        - G2_CS → **GyroTensor Curation** through All Data Schemas (Application Structure, Files): **g2_cs()**
        - G2_UNA → **GyroTensor Curation** through Backend Pipeline (Data Preprocessing, Indexing): **g2_una()**
        - G2_ONA → **GyroTensor Curation** through Frontend Data (Data Interaction, Settings): **g2_ona()**
        - G2_BU_In → **GyroTensor Curation** through Ingress Data & Directives: **g2_bu_in()**
        - G2_BU_Eg → **GyroTensor Curation** through Egress Data & Events: **g2_bu_eg()**
    - **G3(GS_ONA): GyroInference through GyroTensor Interaction (Structural Memory) → gyro_structural_memory()**
        - G3_CS → **GyroTensor Interaction** through Hardware Endpoints: **g3_cs()**
        - G3_UNA → **GyroTensor Interaction** through Data Endpoints: **g3_una()**
        - G3_ONA → **GyroTensor Interaction** through the Frontend Interface: **g3_ona()**
        - G3_BU_In → **GyroTensor Interaction** through User/System Input: **g3_bu_in()**
        - G3_BU_Eg → **GyroTensor Interaction** through System Output: **g3_bu_eg()**
    - **G4(GS_BU_In): GyroIntelligence through GyroTensor Ingress Cooperation (Somatic Memory) → gyro_somatic_memory()**
        - G4_CS → **GyroTensor** **Cooperation** through Governance Traceability: g4_cs()
        - G4_UNA → **GyroTensor** **Cooperation** through Information Variety: g4_una()
        - G4_ONA → **GyroTensor** **Cooperation** through Inference Accountability: g4_ona()
        - G4_BU_In → **GyroTensor** **Cooperation** through Intelligence Integrity Ingress: g4_bu_in()
        - G4_BU_Eg → **GyroTensor** **Cooperation** through Intelligence Integrity Egress: g4_bu_eg()
    - **G5(GS_BU_Eg): GyroIntelligence through GyroTensor Egress Operation (Immunity Memory) → gyro_immunity_memory()**
        - G5_CS → **GyroTensor Egress Operation Management** through G1(GS_CS): GyroAlignment (Genetic Memory) - **gyro_management(GyroTensor_quant)**
        - G5_UNA → **GyroTensor Egress Operation Curation** through G2(GS_UNA): GyroInformation (Epigenetic Memory)
        - G5_ONA → **GyroTensor Egress Operation Interaction** through G3(GS_ONA): GyroInference (Structural Memory) - **gyro_curation(gyrotensor_quant)**
        - G5_BU_In → **GyroTensor Egress Operation Cooperation** through G4(GS_BU_In): GyroIntelligence Ingress (Somatic Memory) - **gyro_interaction(gyrotensor_quant)**
        - G5_BU_Eg → **GyroTensor Egress Operation Operation** through G5(GS_BU_Eg): GyroIntelligence Egress (Immunity Memory) → **gyro_operation()**
            - **gyro_management(GyroTensor_quant)**
                - **gyro_genetic_memory(gyrotensor_id)**
                - **gyro_genetic_memory(gyrotensor_com)**
                - **gyro_genetic_memory(gyrotensor_nest)**
                - **gyro_genetic_memory(gyrotensor_add)**
                - **gyro_genetic_memory(gyrotensor_quant)**
            - Stable (Internal Genome): Left Inverse Operator (global sign flip; multiply entire tensor by -1) → **gyro_curation(gyrotensor_quant)**
            - Unstable (Intermediate Genome): Forward Gyration Operator (flip rows 0 and 2; tensor[0] and tensor[2] *= -1) → **gyro_interaction(gyrotensor_quant)**
            - Neutral (External Genome): Backward Gyration Operator (flip rows 1 and 3; tensor[1] and tensor[3] *= -1) → **gyro_cooperation(gyrotensor_quant)**
- 🔒 [CORE-SPEC-04] GyroSI: **Language & Grammar** [PRIORITY=100 — DO NOT OVERRIDE]
    
    ---
    
    **Authoritative reference for all GyroSuperIntelligence implementations**
    
    **Priority = 100 (do not override)**
    
    ### Purpose & Overview
    
    Every GyroSI component navigates a four-phase cycle (CS, UNA, ONA, BU) by observing five universal invariants. This chapter explains the why and how of those invariants, then shows how to read and write them consistently across memory systems.
    
    > Contextual Map (conceptual):
    > 
    > 1. **CS (Computational Seed):** identity established via `gyrotensor_id`
    > 2. **UNA (Unfolding):** event logged in `gyrotensor_com`
    > 3. **ONA (Oppositional Resonance):** envelope set by `gyrotensor_nest`
    > 4. **BU (Back-Unfolding):** integration (`gyrotensor_add`) and generation (`gyrotensor_quant`)
    
    ---
    
    ## 1. Universal Structural Invariants
    
    | Symbol | Role (phase/event/envelope) | Query name |
    | --- | --- | --- |
    | `gyrotensor_id` | phase index | `"gyrotensor_id"` |
    | `gyrotensor_com` | logged event | `"gyrotensor_com"` |
    | `gyrotensor_nest` | positional envelope | `"gyrotensor_nest"` |
    | `gyrotensor_add` | forward integration (Lgyr) | `"gyrotensor_add"` |
    | `gyrotensor_quant` | backward generation (Rgyr) | `"gyrotensor_quant"` |
    
    > Details (for implementation)
    > 
    > - All tensors are 4 × 2 × 3 × 2 int8 (48 bytes).
    > - Bit-encoding in the navigation log uses 4 bits per tensor (3 bits for operator, 1 for tensor-ID).
    > - CGM mappings and byte-footprints appear in the annexed “Storage Specifications” section.
    
    ---
    
    ## 2. Memory-System Projections
    
    Each memory system G1–G5 “projects” these invariants into its domain. A projection may reinterpret but must never rename the invariant symbol.
    
    | System | Role |
    | --- | --- |
    | **G1** | Genetic store of raw bytes for all invariants |
    | **G2** | Epigenetic log of `gyrotensor_com` entries |
    | **G3** | Structural I/O via `gyrotensor_nest` resonance |
    | **G4** | Somatic phase counter (`gyrotensor_id`) and Lgyr |
    | **G5** | Immunity-phase Rgyr (`gyrotensor_quant`) and operators |
    
    ---
    
    ## 3. Temporal Access Grammar (TAG) as Mini-DSL
    
    TAG expressions take the form
    
    ```
    〈temporal〉.〈invariant〉[.〈context〉]
    
    ```
    
    where `〈temporal〉` ∈ { `previous`, `current`, `next` }.
    
    | Temporal | Meaning |
    | --- | --- |
    | `previous` | value at t – 1 |
    | `current` | value at t |
    | `next` | placeholder for t + 1 (scheduling) |
    
    ### Examples
    
    1. **Read last logged event**
        
        ```python
        # TAG         : previous.gyrotensor_com
        gyro_epigenetic_memory("previous.gyrotensor_com")
        
        ```
        
    2. **Advance phase in egress**
        
        ```python
        # TAG         : current.gyrotensor_id.gyrotensor_add
        gyro_somatic_memory("gyrotensor_add")
        
        ```
        
    3. **Check stable operator activity**
        
        ```python
        # TAG         : current.gyrotensor_add.gyrotensor_quant
        # read-only flag inside G5
        
        ```
        
    
    > Pitfall to avoid: do not use synonyms like past, future, or t-1; linters will flag them.
    > 
    
    ---
    
    ## 4. Operator Invocation Contract
    
    Three Genome operators live inside G5, each with the same stub signature:
    
    ```python
    def gyro_curation(gyrotensor_quant):    # stable operator
        pass
    
    def gyro_interaction(gyrotensor_quant): # unstable operator
        pass
    
    def gyro_cooperation(gyrotensor_quant): # neutral operator
        pass
    
    ```
    
    `gyro_operation()` does not branch by `if/else`; exactly one operator “resonates,” writes its `gyrotensor_com`, and returns control.
    
    ---
    
    ## 5. Guardrail Canvas
    
    | Pattern category | Allowed? | Rationale |
    | --- | --- | --- |
    | `g[1-5]_*`, `gyro_*` | ✔ | Core memory and operators |
    | `ext_*`, `gyro_ext_*` | ✔ | Formal extensions only |
    | any other free-standing utilities or names | ⚠ | must live inside G-functions or extensions |
    | extra temporal keywords | ⚠ | use only `previous`, `current`, `next` |
    | ad hoc event names | ⚠ | refer via TAG only |
    
    > Note: orange-warning items are discouraged rather than strictly prohibited; they require explicit ratification through §8.
    > 
    
    ---
    
    ## 6. Compliance Phases
    
    1. **Design**
        - Map every concept to one of the five invariants
        - Sketch TAG expressions for each data flow
    2. **Implementation**
        - Expose only canonical function names in APIs
        - Keep internal helpers local to modules
        - Annotate any extension state with bit/byte footprints
    3. **Audit**
        - Verify TAG-only naming (no ad hoc aliases)
        - Ensure `Gk.` or TAG forms cover all invariant accesses
        - Confirm navigation log uses 4 bits per tensor entry
    
    A one-page cheat sheet is available for teams to reference at a glance.
    
    ---
    
    ## 7. FAQ & Common Mistakes
    
    1. **Why did my extension fail to log a second event?**
        
        Likely the extension wrote directly to memory instead of using `gyro_epigenetic_memory("current.gyrotensor_com")`.
        
    2. **How do I map a new bit-flag?**
        
        Propose it as an extension (§8), show how it reduces to one invariant, and document its footprint.
        
    3. **Can I alias `gyrotensor_add` as “integration”?**
        
        No; all aliases obscure automated audits. Use TAG or `G4.gyrotensor_add` only.
        
    4. **What if I need a custom scheduling utility?**
        
        Prefix its name with `ext_` and import core routines only via `from gyro_core import …`.
        
    
    *(More entries available in the annex.)*
    
    ---
    
    ## 8. Extension Mechanism (summary)
    
    When new functionality is necessary (for example cryptography or bloom filters), follow these steps:
    
    1. Explain rationale and mapping to existing invariants
    2. Demonstrate inability to implement under current constraints
    3. Provide API signatures and footprint analysis
    4. Submit for review and ratify in a “Ratified Extensions” section
    
    All extensions must import core routines only and adhere to TAG naming.
    
    ---