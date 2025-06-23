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