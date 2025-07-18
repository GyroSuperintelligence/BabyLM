import warnings

warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*")
# 1. Generate ontology_map.json (the ontology)
# python -m baby.information ontology --output memories/public/meta/ontology_map.json
#
# 2. Generate epistemology.npy (the state transition table)
# python -m baby.information epistemology --ontology memories/public/meta/ontology_map.json --output memories/public/meta/epistemology.npy
#
# 3. Generate phenomenology_map.json (the phenomenology mapping)
# python -m baby.information phenomenology --ep memories/public/meta/epistemology.npy --output memories/public/meta/phenomenology_map.json --ontology memories/public/meta/ontology_map.json
"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.
"""

import numpy as np
import argparse
import time
import os
import sys
from typing import Dict, Any, List, Tuple, Set, Optional

# Try to use ujson for speed, fall back to standard json if unavailable
try:
    import ujson as json  # type: ignore[import]
except ImportError:
    import json  # type: ignore

from baby import governance
from baby.contracts import PhenomenologyData


class InformationEngine:
    """
    S2: Measurement & Resource Coordination.

    Sole authority for measurement and conversion between state representations.
    Provides the sensory apparatus through angular gyrodistance measurement.

    If use_array_indexing is True, ontology_map and inverse_ontology_map are stored as numpy arrays for better memory/cache performance.
    """

    def __init__(
        self,
        ontology_data: Dict[str, Any],
        use_array_indexing: Optional[bool] = None,
        strict_validation: bool = True,
    ):
        # Auto-enable array indexing if not set and large ontology
        if use_array_indexing is None:
            use_array_indexing = ontology_data["endogenous_modulus"] > 100_000
        self.use_array_indexing = use_array_indexing
        self.ontology_map = ontology_data["ontology_map"]
        keys = list(self.ontology_map.keys())
        if keys and isinstance(keys[0], str):
            self.ontology_map = {int(k): v for k, v in self.ontology_map.items()}
        self.endogenous_modulus = ontology_data["endogenous_modulus"]
        self.ontology_diameter = ontology_data["ontology_diameter"]

        if use_array_indexing:
            # Note: This assumes ontology indices were assigned in sorted order of state integers
            # If ontology was generated differently, this implicitly redefines indices via sorted position
            keys_arr = np.array(sorted(self.ontology_map.keys()), dtype=np.uint64)
            self._keys = keys_arr
            self._inverse = keys_arr  # index -> state_int
            # Free memory: drop the original mapping in array mode
            self.ontology_map = None
            self.inverse_ontology_map = None  # free memory
        else:
            self._keys = None
            self._inverse = None
            self.inverse_ontology_map = {v: k for k, v in self.ontology_map.items()}

        # Validate expected constants (allow override for testing)
        if strict_validation:
            if self.endogenous_modulus != 788_986:
                raise ValueError(f"Expected endogenous modulus 788,986, got {self.endogenous_modulus}")
            if self.ontology_diameter != 6:
                raise ValueError(f"Expected ontology diameter 6, got {self.ontology_diameter}")

        # Load orbit sizes if available
        self.orbit_cardinality = np.ones(self.endogenous_modulus, dtype=np.uint32)
        phenomap_path = ontology_data.get("phenomap_path")
        if not phenomap_path and "ontology_map_path" in ontology_data:
            phenomap_path = ontology_data["ontology_map_path"].replace("ontology_map.json", "phenomenology_map.json")
        if not phenomap_path:
            phenomap_path = "memories/public/meta/phenomenology_map.json"

        if os.path.exists(phenomap_path):
            with open(phenomap_path) as f:
                payload = json.load(f)

            pheno_map = payload.get("phenomenology_map")
            orbit_sizes = payload.get("orbit_sizes", {})

            if pheno_map and isinstance(pheno_map, list):
                pheno_arr = np.array(pheno_map, dtype=np.int32)
                # Fast vectorized fill: look up size via representative
                size_lookup = np.ones(self.endogenous_modulus, dtype=np.uint32)
                for rep_str, sz in orbit_sizes.items():
                    size_lookup[int(rep_str)] = int(sz)
                self.orbit_cardinality = size_lookup[pheno_arr]
            else:
                # Fallback: keep all ones
                pass

    def get_index_from_state(self, state_int: int) -> int:
        """
        Return ontology (state) index (0..N-1) for a 48-bit state integer.
        Uses fast array indexing if use_array_indexing is True, otherwise dict lookup.

        Args:
            state_int: 48-bit integer representing physical state

        Returns:
            Ontology index (0 to N-1)

        Raises:
            ValueError: If state not found in ontology
        """
        if self.use_array_indexing:
            if self._keys is None:
                raise RuntimeError("Array indexing arrays not initialized.")
            idx = np.searchsorted(self._keys, state_int)
            if idx == len(self._keys) or self._keys[idx] != state_int:
                raise ValueError(
                    f"State integer {state_int} not found in discovered ontology. "
                    f"This indicates a fundamental physics violation."
                )
            return int(idx)
        else:
            if self.ontology_map is None:
                raise RuntimeError("ontology_map is not available in array indexing mode.")
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
            index: Canonical index (0 to N-1)

        Returns:
            48-bit state integer
        """
        if self.use_array_indexing:
            if self._inverse is None:
                raise RuntimeError("Array indexing arrays not initialized.")
            if index < 0 or index >= len(self._inverse):
                raise ValueError(f"Index {index} out of bounds for array indexing.")
            return int(self._inverse[index])
        else:
            if self.inverse_ontology_map is None:
                raise RuntimeError("inverse_ontology_map not initialized.")
            state_int = self.inverse_ontology_map.get(index)
            if state_int is None:
                raise ValueError(f"Invalid index {index}, must be 0 to {self.endogenous_modulus - 1}")
            return state_int

    @staticmethod
    def int_to_tensor(state_int: int) -> np.ndarray:
        """
        Converts a canonical 48-bit integer state to geometric tensor.

        Encoding: bit 0 (LSB) maps to element 47, bit 47 (MSB) maps to element 0.
        Bit values: 0 = +1, 1 = -1

        Args:
            state_int: 48-bit integer state

        Returns:
            Tensor with shape [4, 2, 3, 2] and values ±1
        """
        if state_int >= (1 << 48) or state_int < 0:
            raise ValueError(f"state_int {state_int} out of bounds for 48-bit representation")

        # Convert to 6 bytes (48 bits), big-endian
        state_packed_bytes = state_int.to_bytes(6, "big")

        # Unpack to individual bits
        bits = np.unpackbits(np.frombuffer(state_packed_bytes, dtype=np.uint8), bitorder="big")

        if bits.size != 48:
            raise ValueError(f"Expected 48 bits, got {bits.size}")

        # Convert: 0 -> +1, 1 -> -1 (encoding: bit = (value == -1))
        tensor_flat = (1 - 2 * bits).astype(np.int8)

        # Reshape to proper geometry with lexicographic ordering (C order)
        return tensor_flat.reshape(4, 2, 3, 2)

    @staticmethod
    def tensor_to_int(tensor: np.ndarray) -> int:
        """
        Converts a geometric tensor to its canonical 48-bit integer state.

        Encoding: bit = (value == -1), element 0 -> bit 47 (MSB), element 47 -> bit 0 (LSB)

        Args:
            tensor: NumPy array with shape [4, 2, 3, 2] and values ±1

        Returns:
            48-bit integer representation
        """
        if tensor.shape != (4, 2, 3, 2):
            raise ValueError(f"Expected tensor shape (4, 2, 3, 2), got {tensor.shape}")

        # Flatten in C-order and convert: +1 -> 0, -1 -> 1
        bits = (tensor.flatten(order="C") == -1).astype(np.uint8)

        # Pack bits into bytes
        packed = np.packbits(bits, bitorder="big")

        # Convert to integer, big-endian
        result = int.from_bytes(packed.tobytes(), "big")

        # Round-trip validation (debug mode)
        if __debug__:
            round_trip = InformationEngine.int_to_tensor(result)
            assert round_trip.shape == tensor.shape
            assert np.array_equal(round_trip, tensor)

        return result

    def gyrodistance_angular(self, T1: np.ndarray, T2: np.ndarray) -> float:
        """
        Calculate angular divergence between tensors in radians.

        This measures the geometric alignment between two states in
        N-dimensional space using cosine similarity.

        Args:
            T1: First tensor [4, 2, 3, 2]
            T2: Second tensor [4, 2, 3, 2]

        Returns:
            Angular distance in radians (0 to π)
        """
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()

        # Cosine similarity in N-dimensional space
        cosine_similarity = np.dot(T1_flat, T2_flat) / T1_flat.size
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        return np.arccos(cosine_similarity)

    def measure_state_divergence(self, state_int: int) -> float:
        """
        Measure angular divergence from the archetypal tensor state (GENE_Mac_S).

        Args:
            state_int: Current physical state

        Returns:
            Angular divergence in radians
        """
        current_tensor = self.int_to_tensor(state_int)
        return self.gyrodistance_angular(current_tensor, governance.GENE_Mac_S)

    def get_orbit_cardinality(self, state_index: int) -> int:
        return int(self.orbit_cardinality[state_index])


# ==============================================================================
# Utility: Clean single-line progress reporter
# ==============================================================================
class ProgressReporter:
    def __init__(self, desc: str):
        self.desc = desc
        self.start_time = time.time()
        self.last_update = 0
        self.first_update = True

    def update(self, current: int, total: Optional[int] = None, extra: str = ""):
        now = time.time()
        # Always show first update immediately
        if not self.first_update and now - self.last_update < 0.1 and (total is None or current != total):
            return

        self.first_update = False
        elapsed = now - self.start_time
        rate = current / elapsed if elapsed > 0 else 0

        msg = f"\r{self.desc}: {current:,}"
        if total is not None:
            pct = 100.0 * current / total
            msg += f"/{total:,} ({pct:.1f}%)"
        msg += f" | {rate:.0f}/s | {elapsed:.1f}s"
        if extra:
            msg += f" | {extra}"

        print(msg + " " * 20, end="", flush=True)
        self.last_update = now

    def done(self):
        elapsed = time.time() - self.start_time
        print(f"\r{self.desc}: Done in {elapsed:.1f}s" + " " * 50)


# ==============================================================================
# STEP 1: Ontology Discovery
# ==============================================================================
def discover_and_save_ontology(output_path: str) -> Dict[int, int]:
    """Discovers the complete 788,986 state manifold via BFS."""
    progress = ProgressReporter("Discovering ontology")

    origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    discovered = {origin_int}
    current_level = [origin_int]
    depth = 0
    layer_sizes = []  # Track number of new states at each BFS depth

    while current_level:
        next_level_set = set()
        for state in current_level:
            for intron in range(256):
                next_state = governance.apply_gyration_and_transform(state, intron)
                if next_state not in discovered:
                    discovered.add(next_state)
                    next_level_set.add(next_state)

        current_level = list(next_level_set)
        if not current_level:
            break
        depth += 1
        layer_sizes.append(len(current_level))
        progress.update(len(discovered), total=788_986, extra=f"depth={depth}")

    progress.done()

    # Print expansion pattern for verification
    print(f"Layer sizes (expansion pattern): {layer_sizes}")
    assert sum(layer_sizes) + 1 == len(discovered), "Layer sizes do not sum to total states (including origin)"

    # Validate
    if len(discovered) != 788_986:
        raise RuntimeError(f"Expected 788,986 states, found {len(discovered):,}")

    if depth != 6:
        raise RuntimeError(f"Expected diameter 6, found {depth}")

    # Create mapping
    ontology_map = {state: idx for idx, state in enumerate(sorted(discovered))}

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json_compatible_map = {str(k): v for k, v in ontology_map.items()}
        json.dump(
            {
                "schema_version": "0.9.6",
                "ontology_map": json_compatible_map,
                "endogenous_modulus": 788_986,
                "ontology_diameter": 6,
                "total_states": 788_986,
                "build_timestamp": int(time.time()),
            },
            f,
        )

    return ontology_map


# ==============================================================================
# STEP 2: Epistemology Table
# ==============================================================================
def build_state_transition_table(ontology_map: Dict[int, int], output_path: str):
    """Builds the N×256 state transition table with validation."""
    progress = ProgressReporter("Building epistemology")

    N = len(ontology_map)
    states = np.array(sorted(ontology_map.keys()), dtype=np.uint64)

    # Memory-mapped output
    ep = np.lib.format.open_memmap(output_path, dtype=np.int32, mode="w+", shape=(N, 256))

    # Process in chunks for memory efficiency
    CHUNK_SIZE = 10_000
    for chunk_start in range(0, N, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        chunk_states = states[chunk_start:chunk_end]
        for intron in range(256):
            next_states = governance.apply_gyration_and_transform_batch(chunk_states, intron)
            idxs = np.searchsorted(states, next_states)
            # Debug check: ensure all next_states are in the ontology
            if __debug__:
                if idxs.max() >= states.size or not np.array_equal(states[idxs], next_states):
                    raise RuntimeError("Transition produced unknown state.")
            ep[chunk_start:chunk_end, intron] = idxs
        progress.update(chunk_end, N)

    ep.flush()
    progress.done()


# ==============================================================================
# STEP 3: Phenomenology Map (OPTIMIZED)
# ==============================================================================
def build_phenomenology_map(ep_path: str, ontology_path: str, output_path: str) -> None:
    """
    Builds the definitive, theoretically-sound phenomenology artifact.

    This version is based on the verified physical properties of the system:
    1.  Equivalence is defined as mutual reachability via *any* sequence of
        transformations. This is computed as an SCC over the full graph of
        all 256 introns.
    2.  The mirror map is computed via the global parity (LI) operation.
    3.  The artifact is lean, versioned, and contains only this verified data.
    """
    print("=== [Phenomenology Final Builder] ===")

    # --- 1. Setup ---
    print("Step 1: Loading data...")
    ep = np.load(ep_path, mmap_mode="r")
    N = ep.shape[0]

    with open(ontology_path) as f:
        ontology_data = json.load(f)

    s2 = InformationEngine(ontology_data, use_array_indexing=True, strict_validation=False)

    # --- 2. Tarjan's SCC over the FULL graph ---
    print("Step 2: Computing orbits (SCCs over all 256 introns)...")

    # Build index→state array for vectorized lookup
    print("  Building index→state lookup array...")
    idx_to_state = np.empty(N, dtype=np.uint64)
    # Fill it by iterating over the loaded dictionary
    for state_int, idx in ontology_data["ontology_map"].items():
        idx_to_state[idx] = int(state_int)

    indices = np.full(N, -1, dtype=np.int32)
    lowlink = np.zeros(N, dtype=np.int32)
    on_stack = np.zeros(N, dtype=bool)
    stack: List[int] = []
    canonical = np.full(N, -1, dtype=np.int32)
    orbit_sizes: Dict[int, int] = {}
    reps: List[int] = []
    counter = 0

    def neighbors(v: int) -> np.ndarray:
        return ep[v]  # accept duplicates; negligible logical impact

    processed = 0
    REPORT_INTERVAL = max(1, N // 20)

    for root in range(N):
        if indices[root] != -1:
            continue
        dfs_stack = [(root, iter(neighbors(root)))]
        indices[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while dfs_stack:
            v, children = dfs_stack[-1]
            try:
                w = next(children)
                if indices[w] == -1:
                    indices[w] = lowlink[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    dfs_stack.append((w, iter(neighbors(w))))
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])
            except StopIteration:
                dfs_stack.pop()
                if dfs_stack:
                    parent_v, _ = dfs_stack[-1]
                    lowlink[parent_v] = min(lowlink[parent_v], lowlink[v])
                if lowlink[v] == indices[v]:
                    component = []
                    while True:
                        node = stack.pop()
                        on_stack[node] = False
                        component.append(node)
                        if node == v:
                            break

                    comp_arr = np.array(component, dtype=np.int32)
                    comp_states = idx_to_state[comp_arr]
                    rep = int(comp_arr[np.argmin(comp_states)])

                    canonical[comp_arr] = rep
                    orbit_sizes[rep] = len(component)
                    reps.append(rep)

                    processed += len(component)
                    if processed % REPORT_INTERVAL == 0:
                        pct = processed * 100.0 / N
                        print(f"\r  Progress: {processed}/{N} ({pct:.1f}%) | Orbits found: {len(reps)}", end="")

    print(f"\nSCC computation complete. Found {len(reps)} orbits.")
    assert np.all(canonical >= 0), "FATAL: Not all states were assigned to an orbit."

    # --- 3. Mirror Map ---
    print("Step 3: Identifying mirror pairs...")
    mirror_map: Dict[int, int] = {}
    for rep in reps:
        if rep in mirror_map:
            continue
        state_int = s2.get_state_from_index(rep)
        mirror_state = state_int ^ governance.FULL_MASK
        try:
            mirror_idx = s2.get_index_from_state(mirror_state)
            mirror_rep = int(canonical[mirror_idx])
            mirror_map[rep] = mirror_rep
            mirror_map[mirror_rep] = rep
        except ValueError:
            mirror_map[rep] = -1

    # --- 4. Create and Save Artifact ---
    print("Step 4: Writing final artifact...")
    largest_orbit = int(max(orbit_sizes.values())) if orbit_sizes else 0
    num_orbits = len(reps)
    num_mirror_pairs = sum(1 for k, v in mirror_map.items() if k < v and v != -1)

    artifact: PhenomenologyData = {
        "schema_version": "0.9.6",
        "phenomenology_map": canonical.tolist(),
        "orbit_sizes": {str(k): int(v) for k, v in orbit_sizes.items()},
        "mirror_map": {str(k): int(v) for k, v in mirror_map.items()},
        "metadata": {
            "total_states": N,
            "total_orbits": num_orbits,
            "total_mirror_pairs": num_mirror_pairs,
            "largest_orbit": largest_orbit,
            "build_timestamp": int(time.time()),
            "construction": {
                "method": "scc_full_graph",
                "notes": [
                    "Equivalence is defined as mutual reachability under the full set of 256 introns.",
                    "This is the definitive, assumption-free definition based on verified physics.",
                    f"Finding {num_orbits} orbits is a measured property of the system.",
                ],
            },
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"✓ Saved final phenomenology to: {output_path}")
    print(f"  - Total Orbits: {num_orbits}")
    print(f"  - Total Mirror Pairs: {num_mirror_pairs}")
    print(f"  - Largest Orbit: {largest_orbit}")


def parse_args():
    parser = argparse.ArgumentParser(description="GyroSI asset builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ontology
    p_ont = subparsers.add_parser("ontology", help="Step 1: Discover the full state manifold")
    p_ont.add_argument("--output", required=True, help="Path to save ontology_map.json")

    # Epistemology
    p_epi = subparsers.add_parser("epistemology", help="Step 2: Build state transition table")
    p_epi.add_argument("--ontology", required=True, help="Path to ontology_map.json")
    p_epi.add_argument("--output", required=True, help="Path to save epistemology.npy")

    # Phenomenology
    p_pheno = subparsers.add_parser("phenomenology", help="Step 3: Build canonical orbit map")
    p_pheno.add_argument("--ep", required=True, help="Path to epistemology.npy")
    p_pheno.add_argument("--output", required=True, help="Path to save phenomenology_map.json")
    p_pheno.add_argument("--ontology", required=True, help="Path to ontology_map.json")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        if args.command == "ontology":
            print("=== [Step 1] Ontology Generation ===")
            ontology_map = discover_and_save_ontology(args.output)
            print(f"\n✓ Saved: {args.output}")
            print(f"✓ Total states discovered: {len(ontology_map):,}\n")

        elif args.command == "epistemology":
            print("=== [Step 2] Epistemology Table ===")
            with open(args.ontology) as f:
                ontology_data = json.load(f)
            ontology_map = {int(k): v for k, v in ontology_data["ontology_map"].items()}
            build_state_transition_table(ontology_map, args.output)
            file_size = os.path.getsize(args.output) / 1024**2
            print(f"\n✓ Saved: {args.output}")
            print(f"✓ File size: {file_size:.1f} MB\n")

        elif args.command == "phenomenology":
            print("=== [Step 3] Phenomenology Mapping ===")
            build_phenomenology_map(args.ep, args.ontology, args.output)
            with open(args.output) as f:
                pheno_data = json.load(f)
            with open(args.ontology) as f:
                ontology_data = json.load(f)
            print(f"\n✓ Saved: {args.output}")
            metadata = pheno_data.get("metadata", {})
            print(f"  - Total states: {ontology_data['endogenous_modulus']:,}")
            print(f"  - Unique orbits: {metadata.get('total_orbits', len(pheno_data.get('orbit_sizes', {}))):,}")
            print(f"  - Largest orbit: {metadata.get('largest_orbit', 0):,} states\n")

        else:
            print("Unknown command", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Re-raise to preserve stack trace for debugging
        raise
