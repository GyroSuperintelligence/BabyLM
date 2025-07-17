"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.
"""

import numpy as np
import json
import time
import os
from collections import deque
from typing import Dict, Any, Optional, Set
from baby import governance


class InformationEngine:
    """
    S2: Measurement & Resource Coordination.

    Sole authority for measurement and conversion between state representations.
    Provides the sensory apparatus through angular gyrodistance measurement.

    If use_memmap is True, genotype_map and inverse_genotype_map are stored as numpy arrays for better memory/cache performance.
    """

    def __init__(
        self,
        manifold_data: Dict[str, Any],
        use_memmap: Optional[bool] = None,
    ):
        # Auto-enable memmap if not set and large manifold
        if use_memmap is None:
            use_memmap = manifold_data["endogenous_modulus"] > 100_000
        self.use_memmap = use_memmap
        self.genotype_map = manifold_data["genotype_map"]
        keys = list(self.genotype_map.keys())
        if keys and isinstance(keys[0], str):
            self.genotype_map = {int(k): v for k, v in self.genotype_map.items()}
        self.endogenous_modulus = manifold_data["endogenous_modulus"]
        self.manifold_diameter = manifold_data["manifold_diameter"]
        if use_memmap:
            keys_arr = np.array(sorted(self.genotype_map.keys()), dtype=np.uint64)
            self._keys = keys_arr
            self._values = np.arange(len(self._keys), dtype=np.int32)
            self._inverse = self._keys  # index -> state_int
            self.genotype_map = {k: i for i, k in enumerate(self._keys)}  # tiny dict for fallback
            self.inverse_genotype_map = None  # free ~30 MB
        else:
            self._keys = None
            self._values = None
            self._inverse = None
        self.inverse_genotype_map = {v: k for k, v in self.genotype_map.items()}

        # Validate expected constants
        if self.endogenous_modulus != 788_986:
            raise ValueError(f"Expected endogenous modulus 788,986, got {self.endogenous_modulus}")
        if self.manifold_diameter != 6:
            raise ValueError(f"Expected manifold diameter 6, got {self.manifold_diameter}")

    def get_index_from_state(self, state_int: int) -> int:
        """
        Looks up the canonical index for a physical state integer.

        Args:
            state_int: 48-bit integer representing physical state

        Returns:
            Index in the discovered manifold (0 to 788,985)

        Raises:
            ValueError: If state not found in manifold
        """
        if self.use_memmap:
            if self._keys is None or self._values is None:
                raise RuntimeError("Memmap arrays not initialized.")
            idx = np.searchsorted(self._keys, state_int)
            if idx == len(self._keys) or self._keys[idx] != state_int:
                raise ValueError(
                    f"State integer {state_int} not found in discovered manifold. "
                    f"This indicates a fundamental physics violation."
                )
            return int(self._values[idx])
        else:
            index = self.genotype_map.get(state_int, -1)
            if index == -1:
                raise ValueError(
                    f"CRITICAL: State integer {state_int} not found in discovered manifold. "
                    f"This indicates a fundamental physics violation."
                )
            return index

    def get_state_from_index(self, index: int) -> int:
        """
        Get state integer from canonical index.

        Args:
            index: Canonical index (0 to 788,985)

        Returns:
            48-bit state integer
        """
        if self.use_memmap:
            if self._inverse is None:
                raise RuntimeError("Memmap arrays not initialized.")
            if index < 0 or index >= len(self._inverse):
                raise ValueError(f"Index {index} out of bounds for memmap.")
            state_int = int(self._inverse[index])
        else:
            if self.inverse_genotype_map is None:
                raise RuntimeError("inverse_genotype_map not initialized.")
            state_int = self.inverse_genotype_map.get(index)
            if state_int is None or state_int == -1:
                raise ValueError(f"Invalid index {index}, must be 0 to {self.endogenous_modulus - 1}")
            return state_int
        return state_int

    @staticmethod
    def int_to_tensor(state_int: int) -> np.ndarray:
        """
        Converts a canonical 48-bit integer state to geometric tensor.

        The mapping is: bit 0 (LSB) = element 47, bit 47 (MSB) = element 0
        Bit values: 0 = +1, 1 = -1

        Args:
            state_int: 48-bit integer state

        Returns:
            Tensor with shape [4, 2, 3, 2] and values ±1
        """
        # Convert to 6 bytes (48 bits), big-endian
        state_packed_bytes = state_int.to_bytes(6, "big")

        # Unpack to individual bits
        bits = np.unpackbits(np.frombuffer(state_packed_bytes, dtype=np.uint8))

        # Convert: 0 -> +1, 1 -> -1
        tensor_flat = (1 - 2 * bits).astype(np.int8)

        # Reshape to proper geometry
        return tensor_flat.reshape(4, 2, 3, 2)

    @staticmethod
    def tensor_to_int(tensor: np.ndarray) -> int:
        """
        Converts a geometric tensor to its canonical 48-bit integer state.

        Args:
            tensor: NumPy array with shape [4, 2, 3, 2] and values ±1

        Returns:
            48-bit integer representation
        """
        # Flatten in C-order and convert: +1 -> 0, -1 -> 1
        bits = (tensor.flatten(order="C") == -1).astype(np.uint8)

        # Pack bits into bytes
        packed = np.packbits(bits)

        # Convert to integer, big-endian
        return int.from_bytes(packed.tobytes(), "big")

    def gyrodistance_angular(self, T1: np.ndarray, T2: np.ndarray) -> float:
        """
        Calculate angular divergence between tensors in radians.

        This measures the geometric alignment between two states in
        48-dimensional space using cosine similarity.

        Args:
            T1: First tensor [4, 2, 3, 2]
            T2: Second tensor [4, 2, 3, 2]

        Returns:
            Angular distance in radians (0 to π)
        """
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()

        # Cosine similarity in 48-dimensional space
        cosine_similarity = np.dot(T1_flat, T2_flat) / 48.0
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        return np.arccos(cosine_similarity)

    def measure_state_divergence(self, state_int: int) -> float:
        """
        Measure angular divergence from the archetypal state (GENE_Mac_S).

        Args:
            state_int: Current physical state

        Returns:
            Angular divergence in radians
        """
        current_tensor = self.int_to_tensor(state_int)
        return self.gyrodistance_angular(current_tensor, governance.GENE_Mac_S)


def discover_and_save_manifold(output_path: str) -> None:
    """
    S2 responsibility: Discovers the complete physical manifold.

    Explores the full state space starting from GENE_Mac_S and discovers
    all reachable states. Validates the expected 788,986 states at diameter 6.

    Args:
        output_path: Path to save manifold data

    Raises:
        RuntimeError: If discovered manifold doesn't match expected constants
    """
    print("Discovering physical manifold...")
    start_time = time.time()

    # Start from the archetypal state
    origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    discovered_states = {origin_int}
    queue = [origin_int]
    depth = 0

    # Breadth-first exploration
    while queue:
        next_queue = []

        for current_int in queue:
            # Try all possible intron transformations
            for intron in range(256):
                next_int = governance.apply_gyration_and_transform(current_int, intron)

                if next_int not in discovered_states:
                    discovered_states.add(next_int)
                    next_queue.append(next_int)

        if not next_queue:
            break

        queue = next_queue
        depth += 1

        if depth % 1 == 0:
            print(f"Depth {depth}: {len(discovered_states):,} states discovered")

    # Validate against expected constants
    if len(discovered_states) != 788_986:
        raise RuntimeError(
            f"CRITICAL: Expected 788,986 states, found {len(discovered_states):,}. "
            f"This indicates a fundamental error in the physics implementation."
        )

    if depth != 6:
        raise RuntimeError(
            f"CRITICAL: Expected manifold diameter 6, found {depth}. " f"This violates the theoretical predictions."
        )

    # Create canonical mapping
    sorted_state_ints = sorted(discovered_states)
    genotype_map = {state_int: i for i, state_int in enumerate(sorted_state_ints)}

    # Package manifold data
    manifold_data: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "genotype_map": genotype_map,
        "endogenous_modulus": len(genotype_map),
        "manifold_diameter": depth,
        "total_states": len(discovered_states),
        "build_timestamp": time.time(),
    }

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifold_data, f)

    elapsed = time.time() - start_time
    print(f"Manifold discovery complete in {elapsed:.2f}s")
    print(f"Discovered {len(discovered_states):,} states at diameter {depth}")
    print(f"Saved to: {output_path}")


def build_canonical_map(genotype_map_path: str, output_path: str) -> None:
    """
    Build canonical mapping for orbit representatives.

    For each state in the manifold, computes its canonical representative
    (lexicographically smallest state in its orbit). This enables grouping
    of physically equivalent states and improves cache coherency.

    Args:
        genotype_map_path: Path to genotype map JSON
        output_path: Path to save canonical map
    """
    print("Building canonical map...")
    start_time = time.time()

    # Load genotype map
    with open(genotype_map_path, "r") as f:
        genotype_data = json.load(f)

    genotype_map = genotype_data["genotype_map"]
    # Ensure integer keys
    genotype_map = {int(k): v for k, v in genotype_map.items()}

    # Build inverse map
    inverse_genotype_map = {v: k for k, v in genotype_map.items()}

    # Find canonical representative for each state
    canonical_index_map = {}
    processed_states: Set[int] = set()

    print(f"Processing {len(genotype_map)} states...")

    for i, state_int in enumerate(inverse_genotype_map.values()):
        if i % 10000 == 0:
            print(f"Progress: {i}/{len(genotype_map)} states processed")

        if state_int in processed_states:
            continue

        # Find all states in the orbit
        orbit = {state_int}
        queue = deque([state_int])
        canonical_int = state_int

        while queue:
            current_int = queue.popleft()

            for intron in range(256):
                next_int = governance.apply_gyration_and_transform(current_int, intron)

                if next_int not in orbit:
                    orbit.add(next_int)
                    queue.append(next_int)

                    # Update canonical if we found a smaller one
                    if next_int < canonical_int:
                        canonical_int = next_int

        # Map all states in orbit to canonical representative
        canonical_index = genotype_map[canonical_int]
        for orbit_state in orbit:
            if orbit_state in genotype_map:
                orbit_index = genotype_map[orbit_state]
                canonical_index_map[orbit_index] = canonical_index
                processed_states.add(orbit_state)

    # Save canonical map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(canonical_index_map, f)

    elapsed = time.time() - start_time
    unique_canonicals = len(set(canonical_index_map.values()))

    print(f"Canonical map built in {elapsed:.2f}s")
    print(f"Found {unique_canonicals:,} unique canonical representatives")
    print(f"Compression ratio: {len(genotype_map) / unique_canonicals:.2f}x")
    print(f"Saved to: {output_path}")


def build_state_transition_table(manifold_path: str, output_path: str) -> None:
    """
    Build and save the State Transition Table (STT) as a NumPy array.
    Each entry [i, intron] gives the next state index for state i and intron.
    """
    import numpy as np

    # Load the manifold data
    with open(manifold_path, "r") as f:
        manifold_data = json.load(f)
    genotype_map = {int(k): v for k, v in manifold_data["genotype_map"].items()}
    inverse_genotype_map = {v: k for k, v in genotype_map.items()}
    num_states = len(genotype_map)
    stt = np.zeros((num_states, 256), dtype=np.int32)
    for idx in range(num_states):
        state_int = inverse_genotype_map[idx]
        for intron in range(256):
            next_state_int = governance.apply_gyration_and_transform(state_int, intron)
            next_idx = genotype_map[next_state_int]
            stt[idx, intron] = next_idx
    np.save(output_path, stt)
    print(f"STT saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build manifold assets (manifold, canonical map, STT)")
    subparsers = parser.add_subparsers(dest="command")

    parser_manifold = subparsers.add_parser("manifold", help="Build and save the manifold")
    parser_manifold.add_argument("--output", required=True, help="Path to output genotype_map.json")

    parser_canonical = subparsers.add_parser("canonical", help="Build and save the canonical map")
    parser_canonical.add_argument("--genotype_map", required=True, help="Path to genotype_map.json")
    parser_canonical.add_argument("--output", required=True, help="Path to output canonical_map.json")

    parser_stt = subparsers.add_parser("stt", help="Build and save the state transition table (STT)")
    parser_stt.add_argument("--manifold", required=True, help="Path to genotype_map.json")
    parser_stt.add_argument("--output", required=True, help="Path to output stt.npy")

    args = parser.parse_args()
    if args.command == "manifold":
        discover_and_save_manifold(args.output)
    elif args.command == "canonical":
        build_canonical_map(args.genotype_map, args.output)
    elif args.command == "stt":
        build_state_transition_table(args.manifold, args.output)
    else:
        print("\nHint: You can run this as 'python -m baby.information manifold --output ...' to avoid path headaches.")
        parser.print_help()
