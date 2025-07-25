import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from baby import governance

warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*")
"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.

Build steps:
    python -m baby.information ontology     --output memories/public/meta/ontology_keys.npy
    python -m baby.information epistemology --keys memories/public/meta/ontology_keys.npy \
           --output  memories/public/meta/epistemology.npy
    python -m baby.information phenomenology --ep memories/public/meta/epistemology.npy \
           --keys memories/public/meta/ontology_keys.npy \
           --output memories/public/meta/phenomenology_map.npy
"""


class InformationEngine:
    """
    S2: Measurement & Resource Coordination.

    Sole authority for measurement and conversion between state representations.
    Provides the sensory apparatus through angular gyrodistance measurement.

    If use_array_indexing is True, ontology_map and inverse_ontology_map are stored
    as numpy arrays for better memory/cache performance.
    """

    _keys: NDArray[np.uint64] | None
    _inverse: NDArray[np.uint64] | None
    ontology_map: None
    inverse_ontology_map: None
    use_array_indexing: bool
    orbit_cardinality: NDArray[np.uint32]
    _theta_table: NDArray[np.float32] | None
    _v_max: int

    def __init__(self, keys_path: str, ep_path: str, phenomap_path: str, theta_path: str):
        import numpy as np
        from pathlib import Path
        self._keys = np.load(keys_path, mmap_mode="r")
        self._inverse = self._keys
        self.ontology_map = None
        self.inverse_ontology_map = None
        self.use_array_indexing = True
        self.ep = np.load(ep_path, mmap_mode="r")
        self.orbit_cardinality = np.ones(len(self._keys) if self._keys is not None else 0, dtype=np.uint32)
        if phenomap_path:
            self.orbit_map = np.load(phenomap_path, mmap_mode="r")
            sizes_path = Path(phenomap_path).with_name("orbit_sizes.npy")
            if sizes_path.exists():
                self.orbit_cardinality = np.load(sizes_path, mmap_mode="r")
            else:
                self.orbit_cardinality = np.ones(len(self._keys) if self._keys is not None else 0, dtype=np.uint32)
        if theta_path:
            try:
                self._theta_table = np.load(theta_path, mmap_mode="r")
            except Exception:
                self._theta_table = None
        else:
            self._theta_table = None
        self._v_max = 1 if self.orbit_cardinality is None else int(np.max(self.orbit_cardinality))
        # Early fail if theta.npy is missing or corrupt
        if self._theta_table is None:
            raise RuntimeError("theta.npy is missing or corrupt; required for divergence calculations.")

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
            assert self._inverse is not None
            if index < 0 or index >= len(self._inverse):
                raise ValueError(f"Index {index} out of bounds for array indexing.")
            return int(self._inverse[index])
        else:
            if self.inverse_ontology_map is None:
                raise RuntimeError("inverse_ontology_map not initialized.")
            state_int = self.inverse_ontology_map.get(index)
            if state_int is None:
                assert self._keys is not None
                raise ValueError(f"Invalid index {index}, must be 0 to {len(self._keys) - 1}")
            return state_int

    @staticmethod
    def int_to_tensor(state_int: int) -> NDArray[np.int8]:
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
    def tensor_to_int(tensor: NDArray[np.int8]) -> int:
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

        return result

    def gyrodistance_angular(self, T1: NDArray[np.int8], T2: NDArray[np.int8]) -> float:
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

        return float(np.arccos(cosine_similarity))

    def measure_state_divergence(self, state_int: int) -> float:
        if self._theta_table is None:
            raise RuntimeError("Theta table is not loaded. Cannot compute state divergence.")
        idx = self.get_index_from_state(state_int)
        return float(self._theta_table[idx])

    def get_orbit_cardinality(self, state_index: int) -> int:
        return int(self.orbit_cardinality[state_index])


# ==============================================================================
# Utility: Clean single-line progress reporter
# ==============================================================================
class ProgressReporter:
    def __init__(self, desc: str):
        self.desc = desc
        self.start_time = time.time()
        self.last_update = 0.0
        self.first_update = True

    def update(self, current: int, total: Optional[int] = None, extra: str = "") -> None:
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

    def done(self) -> None:
        elapsed = time.time() - self.start_time
        print(f"\r{self.desc}: Done in {elapsed:.1f}s" + " " * 50)


def open_memmap_int32(
    filename: str,
    mode: str,
    shape: tuple[int, ...],
) -> NDArray[np.int32]:
    from numpy.lib.format import open_memmap as _open_memmap

    arr = _open_memmap(filename, dtype=np.int32, mode=mode, shape=shape)  # type: ignore[no-untyped-call]
    from typing import cast

    return cast(NDArray[np.int32], arr)


# ==============================================================================
# STEP 1: Ontology Discovery
# ==============================================================================
def discover_and_save_ontology(output_path: str) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """Discovers the complete 788,986 state manifold via BFS."""
    progress = ProgressReporter("Discovering ontology")

    origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    discovered = {origin_int}
    current_level = [origin_int]
    depth = 0
    layer_sizes: List[int] = []  # Track number of new states at each BFS depth

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

    # Save sorted keys as npy
    keys = np.array(sorted(discovered), dtype=np.uint64)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, keys)
    print(f"✓ Saved ontology keys to: {output_path}")

    return keys


# ==============================================================================
# STEP 2: Epistemology Table
# ==============================================================================
def build_state_transition_table(keys_path: str, output_path: str) -> None:
    """Builds the N×256 state transition table with validation."""
    progress = ProgressReporter("Building epistemology")

    states = np.load(keys_path, mmap_mode="r")
    N = len(states)

    # ----- θ table (angular divergence from origin) -----
    theta_path = output_path.replace("epistemology.npy", "theta.npy")
    origin = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    acos_lut = np.arccos(1 - 2 * np.arange(49) / 48.0).astype(np.float32)
    theta = np.empty(N, dtype=np.float32)
    for i, s in enumerate(states):
        h = int(s ^ origin).bit_count()
        theta[i] = acos_lut[h]

    # Memory-mapped output
    ep = open_memmap_int32(output_path, "w+", (N, 256))

    # Process in chunks for memory efficiency
    CHUNK_SIZE = 10_000
    for chunk_start in range(0, N, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        chunk_states = states[chunk_start:chunk_end]
        # Vectorized: apply all introns at once
        next_states_all = governance.apply_gyration_and_transform_all_introns(chunk_states)
        # next_states_all shape: (chunk_len, 256)
        idxs = np.searchsorted(states, next_states_all, side="left")
        # Debug check: ensure all next_states are in the ontology
        if __debug__:
            if idxs.max() >= states.size or not np.all(states[idxs] == next_states_all):
                raise RuntimeError("Transition produced unknown state.")
        ep[chunk_start:chunk_end, :] = idxs
        progress.update(chunk_end, N)

    # Save theta table
    np.save(theta_path, theta)
    ep.flush()  # type: ignore[attr-defined]
    progress.done()


# ==============================================================================
# STEP 3: Phenomenology Map (Core + Optional Diagnostics)
# ==============================================================================


def _compute_sccs(
    ep: NDArray[Any], idx_to_state: NDArray[Any], introns_to_use: List[int]
) -> Tuple[NDArray[Any], Dict[int, int], List[int]]:
    """
    Core Tarjan's SCC algorithm restricted to a subset of introns.
    Optimized: neighbors() does not np.unique, just returns ep[v, introns_arr].
    """
    N = ep.shape[0]
    indices = np.full(N, -1, dtype=np.int32)
    lowlink = np.zeros(N, dtype=np.int32)
    on_stack = np.zeros(N, dtype=bool)
    stack: List[int] = []
    canonical = np.full(N, -1, dtype=np.int32)
    orbit_sizes: Dict[int, int] = {}
    reps: List[int] = []
    counter = 0
    introns_arr = np.array(introns_to_use, dtype=np.int32)

    def neighbors(v: int) -> NDArray[np.int32]:
        # Return all neighbors; duplicates are fine
        return np.asarray(ep[v, introns_arr], dtype=np.int32)

    for root in range(N):
        if indices[root] != -1:
            continue
        dfs_stack = [(root, iter(neighbors(root)))]
        indices[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while dfs_stack:
            v, child_iter = dfs_stack[-1]
            try:
                while True:
                    w = int(next(child_iter))
                    if indices[w] == -1:
                        # Tree edge: recurse
                        indices[w] = lowlink[w] = counter
                        counter += 1
                        stack.append(w)
                        on_stack[w] = True
                        dfs_stack.append((w, iter(neighbors(w))))
                        break
                    elif on_stack[w]:
                        # Back edge to a node in current SCC: update lowlink
                        if indices[w] < lowlink[v]:
                            lowlink[v] = indices[w]
                        continue  # keep looping children
                    else:
                        # Edge to an already closed SCC – ignore
                        continue
            except StopIteration:
                dfs_stack.pop()
                if dfs_stack:
                    parent_v, _ = dfs_stack[-1]
                    if lowlink[v] < lowlink[parent_v]:
                        lowlink[parent_v] = lowlink[v]
                if lowlink[v] == indices[v]:
                    comp = []
                    while True:
                        node = stack.pop()
                        on_stack[node] = False
                        comp.append(node)
                        if node == v:
                            break
                    comp_arr = np.array(comp, dtype=np.int32)
                    comp_states = idx_to_state[comp_arr]
                    rep = int(comp_arr[np.argmin(comp_states)])
                    canonical[comp_arr] = rep
                    orbit_sizes[rep] = comp_arr.size
                    reps.append(rep)

    assert np.all(canonical >= 0), "Unassigned nodes after SCC computation"
    return canonical, orbit_sizes, reps


def build_phenomenology_map(
    ep_path: str, keys_path: str, output_path: str
) -> None:
    """
    Builds the canonical phenomenology map for GyroSI runtime operations.
    Args:
        ep_path: Path to epistemology.npy
        keys_path: Path to ontology_keys.npy
        output_path: Path to save phenomenology_map.npy
    """
    print("=== [Phenomenology Core Builder] ===")

    # Load data
    ep = np.load(ep_path, mmap_mode="r")
    keys = np.load(keys_path, mmap_mode="r")
    N = ep.shape[0]

    # Build index→state lookup array
    idx_to_state = keys

    # Core: Compute canonical phenomenology (all 256 introns)
    print("Computing canonical phenomenology (all 256 introns)...")
    all_introns = list(range(256))
    canonical, orbit_sizes, _ = _compute_sccs(ep, idx_to_state, all_introns)
    print(f"  Found {len(np.unique(canonical))} canonical orbits (expected 256)")

    # Save canonical as .npy
    np.save(output_path, canonical.astype(np.int32))
    # Save orbit_sizes as orbit_sizes.npy
    sizes = np.zeros(N, dtype=np.uint32)
    for rep, sz in orbit_sizes.items():
        sizes[rep] = sz
    np.save(str(Path(output_path).with_name("orbit_sizes.npy")), sizes)
    print(f"\n✓ Saved canonical phenomenology to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GyroSI asset builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ontology
    p_ont = subparsers.add_parser("ontology", help="Step 1: Discover the full state manifold")
    p_ont.add_argument(
        "--output",
        required=True,
        help="Path to save ontology_keys.npy (recommended: memories/public/meta/ontology_keys.npy)",
    )

    # Epistemology
    p_epi = subparsers.add_parser("epistemology", help="Step 2: Build state transition table")
    p_epi.add_argument(
        "--keys", required=True, help="Path to ontology_keys.npy (recommended: memories/public/meta/ontology_keys.npy)"
    )
    p_epi.add_argument(
        "--output",
        required=True,
        help="Path to save epistemology.npy (recommended: memories/public/meta/epistemology.npy)",
    )

    # Phenomenology
    p_pheno = subparsers.add_parser("phenomenology", help="Step 3: Build canonical orbit map")
    p_pheno.add_argument(
        "--ep", required=True, help="Path to epistemology.npy (recommended: memories/public/meta/epistemology.npy)"
    )
    p_pheno.add_argument(
        "--keys", required=True, help="Path to ontology_keys.npy (recommended: memories/public/meta/ontology_keys.npy)"
    )
    p_pheno.add_argument(
        "--output",
        required=True,
        help="Path to save phenomenology_map.npy (recommended: memories/public/meta/phenomenology_map.npy)",
    )

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
            build_state_transition_table(args.keys, args.output)
            file_size = os.path.getsize(args.output) / 1024**2
            print(f"\n✓ Saved: {args.output}")
            print(f"✓ File size: {file_size:.1f} MB\n")

        elif args.command == "phenomenology":
            print("=== [Step 3] Phenomenology Mapping ===")
            build_phenomenology_map(args.ep, args.keys, args.output)
            # No final summary, as output is now a .npy file

        else:
            print("Unknown command", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Re-raise to preserve stack trace for debugging
        raise
