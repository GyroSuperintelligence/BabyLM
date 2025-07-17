# 1. Generate ontology_map.json (the ontology)
# python -m baby.information ontology --output memories/public/ontology/ontology_map.json
#
# 2. Generate phenomenology_map.json (the canonical mapping)
# python -m baby.information canonical --ontology_map memories/public/ontology/ontology_map.json --output memories/public/ontology/phenomenology_map.json
#
# 3. Generate epistemology.npy (the state transition table)
# python -m baby.information epistemology --ontology memories/public/ontology/ontology_map.json --output memories/public/ontology/epistemology.npy
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
from concurrent.futures import ProcessPoolExecutor, as_completed

CHUNK = 50_000  # 788,986 / 50,000 ≈ 16 chunks – fits in 2 GB RAM

def _orbit_chunk(start_idx, stop_idx, indices, states):
    """Worker: return {orbit_index: canonical_index} for local slice."""
    from collections import deque
    out = {}
    for canon_idx in range(start_idx, stop_idx):
        state_int = states[canon_idx]
        if state_int in out:  # already mapped by earlier orbit
            continue
        orbit = [state_int]
        q = deque([state_int])
        canonical_int = state_int
        while q:
            s = q.popleft()
            for intron in range(256):
                nxt = governance.apply_gyration_and_transform(s, intron)
                if nxt not in orbit:
                    orbit.append(nxt)
                    q.append(nxt)
                    canonical_int = min(canonical_int, nxt)
        canonical_index = indices[canonical_int]
        for s in orbit:
            out[indices[s]] = canonical_index
    return out

class InformationEngine:
    """
    S2: Measurement & Resource Coordination.

    Sole authority for measurement and conversion between state representations.
    Provides the sensory apparatus through angular gyrodistance measurement.

    If use_memmap is True, ontology_map and inverse_ontology_map are stored as numpy arrays for better memory/cache performance.
    """

    def __init__(
        self,
        ontology_data: Dict[str, Any],
        use_memmap: Optional[bool] = None,
    ):
        # Auto-enable memmap if not set and large ontology
        if use_memmap is None:
            use_memmap = ontology_data["endogenous_modulus"] > 100_000
        self.use_memmap = use_memmap
        self.ontology_map = ontology_data["ontology_map"]
        keys = list(self.ontology_map.keys())
        if keys and isinstance(keys[0], str):
            self.ontology_map = {int(k): v for k, v in self.ontology_map.items()}
        self.endogenous_modulus = ontology_data["endogenous_modulus"]
        self.ontology_diameter = ontology_data["ontology_diameter"]
        if use_memmap:
            keys_arr = np.array(sorted(self.ontology_map.keys()), dtype=np.uint64)
            self._keys = keys_arr
            self._values = np.arange(len(self._keys), dtype=np.int32)
            self._inverse = self._keys  # index -> state_int
            self.ontology_map = {k: i for i, k in enumerate(self._keys)}  # tiny dict for fallback
            self.inverse_ontology_map = None  # free ~30 MB
        else:
            self._keys = None
            self._values = None
            self._inverse = None
            self.inverse_ontology_map = {v: k for k, v in self.ontology_map.items()}

        # Validate expected constants
        if self.endogenous_modulus != 788_986:
            raise ValueError(f"Expected endogenous modulus 788,986, got {self.endogenous_modulus}")
        if self.ontology_diameter != 6:
            raise ValueError(f"Expected ontology diameter 6, got {self.ontology_diameter}")

    def get_index_from_state(self, state_int: int) -> int:
        """
        Looks up the canonical index for a physical state integer.

        Args:
            state_int: 48-bit integer representing physical state

        Returns:
            Index in the discovered ontology (0 to 788,985)

        Raises:
            ValueError: If state not found in ontology
        """
        if self.use_memmap:
            if self._keys is None or self._values is None:
                raise RuntimeError("Memmap arrays not initialized.")
            idx = np.searchsorted(self._keys, state_int)
            if idx == len(self._keys) or self._keys[idx] != state_int:
                raise ValueError(
                    f"State integer {state_int} not found in discovered ontology. "
                    f"This indicates a fundamental physics violation."
                )
            return int(self._values[idx])
        else:
            index = self.ontology_map.get(state_int, -1)
            if index == -1:
                raise ValueError(
                    f"CRITICAL: State integer {state_int} not found in discovered ontology. "
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
            if self.inverse_ontology_map is None:
                raise RuntimeError("inverse_ontology_map not initialized.")
            state_int = self.inverse_ontology_map.get(index)
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
        bits = np.unpackbits(np.frombuffer(state_packed_bytes, dtype=np.uint8), bitorder='big')

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
        packed = np.packbits(bits, bitorder='big')

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


def discover_and_save_ontology(output_path: str) -> None:
    """
    S2 responsibility: Discovers the complete physical ontology.

    Explores the full state space starting from GENE_Mac_S and discovers
    all reachable states. Validates the expected 788,986 states at diameter 6.

    Args:
        output_path: Path to save ontology data

    Raises:
        RuntimeError: If discovered ontology doesn't match expected constants
    """
    print("Discovering physical ontology...")
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
            f"CRITICAL: Expected ontology diameter 6, found {depth}. " f"This violates the theoretical predictions."
        )

    # Create canonical mapping
    sorted_state_ints = sorted(discovered_states)
    ontology_map = {state_int: i for i, state_int in enumerate(sorted_state_ints)}

    # Package ontology data
    ontology_data: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "ontology_map": ontology_map,
        "endogenous_modulus": len(ontology_map),
        "ontology_diameter": depth,
        "total_states": len(discovered_states),
        "build_timestamp": time.time(),
    }

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ontology_data, f)

    elapsed = time.time() - start_time
    print(f"Manifold discovery complete in {elapsed:.2f}s")
    print(f"Discovered {len(discovered_states):,} states at diameter {depth}")
    print(f"Saved to: {output_path}")


def build_phenomenology_map(ontology_map_path: str, output_path: str) -> None:
    """
    Parallel build of canonical mapping for orbit representatives (phenomenology).
    """
    print("Building phenomenology map (parallel)...")
    start_time = time.time()
    with open(ontology_map_path, "r") as f:
        data = json.load(f)
    idx_of = {int(k): v for k, v in data["ontology_map"].items()}
    state_by_idx = {v: k for k, v in idx_of.items()}
    N = len(idx_of)
    slice_bounds = list(range(0, N, CHUNK)) + [N]
    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(_orbit_chunk, slice_bounds[i], slice_bounds[i + 1], idx_of, state_by_idx)
            for i in range(len(slice_bounds) - 1)
        ]
        canonical = {}
        for f in as_completed(futures):
            canonical.update(f.result())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(canonical, f)
    elapsed = time.time() - start_time
    print(f"Phenomenology map built in {elapsed:.2f}s → {output_path}")


def build_state_transition_table(ontology_path: str, output_path: str) -> None:
    """
    Vectorized build of the state transition table (epistemology) using NumPy and memmap.
    """
    import numpy as np
    data = json.load(open(ontology_path))
    idx_of = {int(k): v for k, v in data["ontology_map"].items()}
    states = np.fromiter((int(k) for k in sorted(idx_of.keys())), dtype=np.uint64)
    N = len(states)
    ep = np.memmap(output_path, dtype=np.int32, mode="w+", shape=(N, 256))
    sorted_states = states.copy()
    for intron in range(256):
        next_states = np.vectorize(governance.apply_gyration_and_transform, otypes=[np.uint64])(states, intron)
        ep[:, intron] = np.searchsorted(sorted_states, next_states)
    ep.flush()
    print(f"Epistemology table built → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ontology assets (ontology, canonical map, STT)")
    subparsers = parser.add_subparsers(dest="command")

    parser_ontology = subparsers.add_parser("ontology", help="Build and save the ontology")
    parser_ontology.add_argument("--output", required=True, help="Path to output ontology_map.json")

    parser_phenomenology = subparsers.add_parser("phenomenology", help="Build and save the phenomenology map")
    parser_phenomenology.add_argument("--ontology_map", required=True, help="Path to ontology_map.json")
    parser_phenomenology.add_argument("--output", required=True, help="Path to output phenomenology_map.json")

    parser_epistemology = subparsers.add_parser("epistemology", help="Build and save the state transition table (STT)")
    parser_epistemology.add_argument("--ontology", required=True, help="Path to ontology_map.json")
    parser_epistemology.add_argument("--output", required=True, help="Path to output epistemology.npy")

    args = parser.parse_args()
    if args.command == "ontology":
        discover_and_save_ontology(args.output)
    elif args.command == "phenomenology":
        build_phenomenology_map(args.ontology_map, args.output)
    elif args.command == "epistemology":
        build_state_transition_table(args.ontology, args.output)
    else:
        print("\nHint: You can run this as 'python -m baby.information ontology --output ...' to avoid path headaches.")
        parser.print_help()
