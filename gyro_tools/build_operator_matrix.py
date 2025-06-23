#!/usr/bin/env python3
"""
One-time build script to generate the deterministic operator matrix.
This replaces all heuristic logic with provably mechanical operation.
"""
import os, sys, hashlib, numpy as np, torch
from typing import Dict, Tuple

# Import from your existing core
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from core.gyro_core import _OP_CODES, gyration_op


def _get_gene_constant() -> Dict[str, torch.Tensor]:
    """Mirror your exact Gene definition from gyro_core.py"""
    gene_pattern = [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ]
    base_tensor = torch.tensor(gene_pattern, dtype=torch.int8)
    return {"id_0": base_tensor.clone(), "id_1": base_tensor.clone()}


def _extract_slice(tensor: torch.Tensor, phase: int) -> torch.Tensor:
    """Extract the specific tensor slice for a given phase"""
    tensor_id = phase % 2
    position_in_tensor = (phase // 2) % 24
    outer_idx = position_in_tensor // 6
    inner_idx = (position_in_tensor // 3) % 2
    spatial_idx = position_in_tensor % 3

    tensor_key = f"id_{tensor_id}"
    gene = _get_gene_constant()
    return gene[tensor_key][outer_idx][inner_idx][spatial_idx]


def _find_alignment_operator(phase: int, target_alignment: Tuple[int, int]) -> int:
    """
    Find which operator achieves the target alignment at this phase.
    This is the core mechanical proof - exactly one operator works.
    """
    gene = _get_gene_constant()
    tensor_id = phase % 2
    tensor_key = f"id_{tensor_id}"

    # Check if already aligned (Identity)
    current_slice = _extract_slice(gene[tensor_key], phase)
    if tuple(current_slice.tolist()) == target_alignment:
        return _OP_CODES["IDENTITY"]

    # Test each operator to find the one that works
    for op_name, op_code in _OP_CODES.items():
        if op_name == "IDENTITY":
            continue

        # Apply operator and check result
        temp_tensor = gyration_op(gene[tensor_key].clone(), op_code, clone=False)
        result_slice = _extract_slice(temp_tensor, phase)

        if tuple(result_slice.tolist()) == target_alignment:
            return op_code

    # This should never happen with correct implementation
    raise Exception(f"No operator found for phase {phase}, target {target_alignment}")


def build_operator_matrix(output_path: str = "src/core/operator_matrix.dat") -> None:
    """Build the complete 48x256 operator matrix"""
    print("Building mechanical operator matrix...")

    # Create the lookup table
    matrix = np.zeros((48, 256), dtype=np.uint8)

    for phase in range(48):
        print(f"  Processing phase {phase}/47...")
        for byte_val in range(256):
            # Map byte nibbles to target alignment
            high_nibble = (byte_val >> 4) & 0x0F
            low_nibble = byte_val & 0x0F

            # Convert to alignment values
            high_alignment = 1 if high_nibble >= 8 else -1
            low_alignment = 1 if low_nibble >= 8 else -1
            target = (high_alignment, low_alignment)

            # Find the operator that achieves this alignment
            matrix[phase, byte_val] = _find_alignment_operator(phase, target)

    # Create integrity header (SHA-256 of Gene)
    gene = _get_gene_constant()
    hasher = hashlib.sha256()
    hasher.update(gene["id_0"].numpy().tobytes())
    hasher.update(gene["id_1"].numpy().tobytes())
    header = hasher.digest()

    # Save matrix with header
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)  # 32-byte SHA-256 header
        f.write(matrix.tobytes())  # 48x256 = 12,288 bytes

    print(f"✓ Operator matrix saved to {output_path}")
    print(f"  Size: {32 + matrix.nbytes} bytes (32-byte header + 12,288-byte matrix)")


if __name__ == "__main__":
    build_operator_matrix()
