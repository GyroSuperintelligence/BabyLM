import warnings
warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*")
# 1. Generate ontology_map.json (the ontology)
# python -m baby.information ontology --output memories/public/meta/ontology_map.json
#
# 2. Generate phenomenology_map.json (the phenomenology mapping)
# python -m baby.information phenomenology --ontology_map memories/public/meta/ontology_map.json --output memories/public/meta/phenomenology_map.json
#
# 3. Generate epistemology.npy (the state transition table)
# python -m baby.information epistemology --ontology memories/public/meta/ontology_map.json --output memories/public/meta/epistemology.npy
"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.
"""

import numpy as np

# Try to use ujson for speed, fall back to standard json if unavailable
# NOTE: The following import may trigger a false positive from Pyright (reportMissingModuleSource)
# if 'ujson' is not installed in your environment. This is expected and safe to ignore:
# the code falls back to standard 'json' if 'ujson' is unavailable.
try:
    import ujson as json  # type: ignore[import]
except ImportError:
    import json  # type: ignore
import time
import os
from typing import Dict, Any, Optional, Set
from baby import governance


# ---------- Phenomenology builder ----------
def build_phenomenology_map(ep_path: str, output_path: str, ontology_path: str) -> None:
    """
    Robustly derives the canonical-orbit map using explicit orbit finding (BFS) as described in Genetics.md.
    For each state, finds all states in its orbit, determines the lex smallest state, and maps each state index to its canonical representative.
    Args:
        ep_path: Path to epistemology.npy (int32 array, shape (N,256)).
        output_path: Where to write phenomenology_map.json
        ontology_path: Path to ontology_map.json (needed for state_int <-> index mapping)
    """
    import numpy as np, json, os, time, itertools, sys
    from baby import governance

    done = 0
    ep = np.load(ep_path, mmap_mode="r")  # N × 256 int32
    N = ep.shape[0]

    # canonical representative for every state, –1 = unknown
    rep = np.full(N, -1, np.int32)

    # index → physical‑state integer (needed to choose lexicographic minima)
    inverse = np.empty(N, np.uint64)
    with open(ontology_path) as f:
        data = json.load(f)["ontology_map"]
    for k, v in data.items():
        inverse[v] = int(k)

    orbit_sizes = {}
    start_time = time.time()
    sample_indices = set()
    if N >= 3:
        sample_indices = {0, N // 2, N - 1}
    elif N == 2:
        sample_indices = {0, 1}
    elif N == 1:
        sample_indices = {0}
    sample_printed = 0
    max_samples = 3

    def print_progress_bar(iteration, total, length=40, elapsed=None):
        percent = 100 * (iteration / float(total))
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        msg = f'\rPhenomenology: |{bar}| {percent:5.1f}% ({iteration}/{total})'
        if elapsed is not None:
            msg += f' | Elapsed: {elapsed:.1f}s'
        print(msg, end='')
        sys.stdout.flush()
        if iteration == total:
            print()  # Newline on completion

    BATCH = 100_000   # 100 000 × 256 × 4 B = 98 MB
    done = 0
    for seed in range(N):
        if rep[seed] != -1:
            continue
        orbit_start = time.time()
        # Vectorized BFS with batching and deduplication
        frontier = np.array([seed], dtype=np.int32)
        members = []
        while frontier.size:
            members.append(frontier)
            new_chunks = []
            for start in range(0, frontier.size, BATCH):
                f_batch = frontier[start:start+BATCH]
                neigh   = ep[f_batch].ravel()
                mask    = rep[neigh] == -1
                if mask.any():
                    unseen = neigh[mask]
                    rep[unseen] = seed
                    new_chunks.append(unseen)
            if new_chunks:
                frontier = np.unique(np.concatenate(new_chunks))  # deduplicate
            else:
                frontier = np.empty(0, np.int32)
        members = np.concatenate(members)
        # pick lexicographically smallest physical state as final representative
        lex_rep_state = inverse[members].min()
        lex_rep_idx = members[inverse[members].argmin()]
        rep[members] = lex_rep_idx
        orbit_sizes[int(lex_rep_idx)] = members.size
        done += members.size
        orbit_elapsed = time.time() - orbit_start
        if orbit_elapsed > 300:
            print(f"\nWarning: Orbit {seed} took {orbit_elapsed:.1f}s")
        if sample_printed < max_samples and seed in sample_indices:
            sample_members = sorted(list(members))[:5]
            print(f"\nSample orbit {sample_printed+1}/{max_samples} (seed={seed}): size={members.size}, rep={lex_rep_idx}, members={sample_members} ...")
            sample_printed += 1
        # Progress bar
        update_interval = max(1, min(N // 1000, 100))
        if seed % update_interval == 0 or done == N:
            elapsed = time.time() - start_time
            print_progress_bar((rep != -1).sum(), N, elapsed=elapsed)
    # Final progress bar
    print_progress_bar(N, N, elapsed=time.time() - start_time)

    # Compute autonomic cycles of length 2-6 from archetype
    def _enumerate_autonomic_cycles(inverse, governance):
        origin = int.from_bytes(governance.GENE_Mac_S.tobytes(), 'big')
        autonomic_cycles = []
        for length in range(2, 7):
            for seq in itertools.product(range(256), repeat=length):
                s = origin
                for intr in seq:
                    s = governance.apply_gyration_and_transform(s, intr)
                if s == origin:
                    autonomic_cycles.append(seq)
        return autonomic_cycles

    autonomic_cycles = _enumerate_autonomic_cycles(inverse, governance)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "phenomenology": rep.tolist(),
                "orbit_sizes": {str(k): int(v) for k, v in orbit_sizes.items()},
                "autonomic_cycles": autonomic_cycles,
            },
            f,
        )
    print(f"\nphenomenology_map written → {output_path}")


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

        # Load orbit sizes if available
        self.orbit_cardinality = np.ones(self.endogenous_modulus, dtype=np.uint32)
        phenomap_path = ontology_data.get('phenomap_path')
        if not phenomap_path and hasattr(self, 'ontology_map'):
            # Try to infer path from known conventions
            phenomap_path = None
            if hasattr(self, 'ontology_map'):
                # If ontology_map was loaded from a file, try to guess the path
                # (This is a fallback; ideally, the caller should provide the path)
                pass
        if not phenomap_path and 'ontology_map_path' in ontology_data:
            phenomap_path = ontology_data['ontology_map_path'].replace("ontology_map.json","phenomenology_map.json")
        if not phenomap_path:
            phenomap_path = "memories/public/meta/phenomenology_map.json"
        if os.path.exists(phenomap_path):
            with open(phenomap_path) as f:
                payload = json.load(f)
            sz = payload.get("orbit_sizes", {})
            self.orbit_cardinality = np.fromiter(
                (sz.get(str(i),1) for i in range(self.endogenous_modulus)),
                dtype=np.uint32)

    def get_index_from_state(self, state_int: int) -> int:
        """
        Looks up the phenomenology index for a physical state integer.

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
        bits = np.unpackbits(np.frombuffer(state_packed_bytes, dtype=np.uint8), bitorder="big")

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
        packed = np.packbits(bits, bitorder="big")

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
        "schema_version": "0.9.6",
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


def build_state_transition_table(ontology_path: str, output_path: str) -> None:
    """
    Vectorized build of the state transition table (epistemology) using NumPy and memmap.
    """
    import numpy as np
    from numpy.lib.format import open_memmap
    import time

    start_time = time.time()
    data = json.load(open(ontology_path))
    idx_of = {int(k): v for k, v in data["ontology_map"].items()}
    states = np.fromiter((int(k) for k in sorted(idx_of.keys())), dtype=np.uint64)
    N = len(states)
    ep = open_memmap(output_path, dtype=np.int32, mode="w+", shape=(N, 256))  # N rows, 256 introns
    sorted_states = states.copy()
    for intron in range(256):
        next_states = np.vectorize(governance.apply_gyration_and_transform, otypes=[np.uint64])(states, intron)
        ep[:, intron] = np.searchsorted(sorted_states, next_states)
    ep.flush()
    elapsed = time.time() - start_time
    print(f"Epistemology table built → {output_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ontology assets (ontology, canonical map, STT)")
    subparsers = parser.add_subparsers(dest="command")

    parser_ontology = subparsers.add_parser("ontology", help="Build and save the ontology")
    parser_ontology.add_argument("--output", required=True, help="Path to output ontology_map.json")

    parser_phenomenology = subparsers.add_parser("phenomenology", help="Build and save the phenomenology map (fast)")
    parser_phenomenology.add_argument("--ep", required=True, help="Path to epistemology.npy")
    parser_phenomenology.add_argument("--output", required=True, help="Path to output phenomenology_map.json")
    parser_phenomenology.add_argument("--ontology", required=True, help="Path to ontology_map.json")

    parser_epistemology = subparsers.add_parser("epistemology", help="Build and save the state transition table (STT)")
    parser_epistemology.add_argument("--ontology", required=True, help="Path to ontology_map.json")
    parser_epistemology.add_argument("--output", required=True, help="Path to output epistemology.npy")

    args = parser.parse_args()
    if args.command == "ontology":
        discover_and_save_ontology(args.output)
    elif args.command == "phenomenology":
        build_phenomenology_map(args.ep, args.output, args.ontology)
    elif args.command == "epistemology":
        build_state_transition_table(args.ontology, args.output)
    else:
        print("\nHint: You can run this as 'python -m baby.information ontology --output ...' to avoid path headaches.")
        parser.print_help()
