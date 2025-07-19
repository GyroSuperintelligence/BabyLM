import warnings
warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*")
"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.

Build steps:
1. Generate ontology_map.json (the ontology)
   python -m baby.information ontology --output memories/public/meta/ontology_map.json

2. Generate epistemology.npy (the state transition table)
   python -m baby.information epistemology --ontology memories/public/meta/ontology_map.json --output memories/public/meta/epistemology.npy

3. Generate phenomenology_map.json (the phenomenology mapping)
   python -m baby.information phenomenology --ep memories/public/meta/epistemology.npy --output memories/public/meta/phenomenology_map.json --ontology memories/public/meta/ontology_map.json
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
# STEP 3: Phenomenology Map (Dual-Structure)
# ==============================================================================
# Global parity (LI) bit mask pattern
LI_MASK = 0b01000010  

def _compute_sccs(
    ep: np.ndarray,
    idx_to_state: np.ndarray,
    introns_to_use: List[int]
) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    """
    Core Tarjan's SCC algorithm restricted to a subset of introns.
    
    Args:
        ep: Epistemology table of shape (N, 256)
        idx_to_state: Mapping from index to state integer
        introns_to_use: Subset of introns to include in the graph
        
    Returns:
        Tuple of (canonical_map, orbit_sizes, representatives)
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

    def neighbors(v: int) -> np.ndarray:
        return np.unique(ep[v, introns_arr])

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
                w = int(next(child_iter))
                if indices[w] == -1:
                    indices[w] = lowlink[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    dfs_stack.append((w, iter(neighbors(w))))
                elif on_stack[w]:
                    if indices[w] < lowlink[v]:
                        lowlink[v] = indices[w]
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


def build_phenomenology_map(ep_path: str, ontology_path: str, output_path: str) -> None:
    """
    Builds dual-structure phenomenology artifact.
    
    This includes two distinct views of the state space:
    1. full_graph: SCCs including LI (global parity) transitions
       In this view, every orbit is closed under parity and self-mirrored.
    2. parity_free: SCCs excluding LI transitions
       This view exposes the 128 mirror-pair structure predicted by theory.
       
    The artifact also includes cross-layer mappings to show how parity-free
    orbits combine into full-graph orbits.
    """
    print("=== [Phenomenology Dual-Structure Builder] ===")
    
    # Load data
    ep = np.load(ep_path, mmap_mode="r")
    with open(ontology_path) as f:
        ontology_data = json.load(f)
    N = ep.shape[0]

    # Build index→state lookup array
    # This is robust (no ordering assumptions) and used consistently throughout
    idx_to_state = np.empty(N, dtype=np.uint64)
    for k_str, idx in ontology_data["ontology_map"].items():
        idx_to_state[idx] = int(k_str)

    # Step 1: Compute SCCs on the full graph
    print("Step 1: Computing orbits on full graph (all 256 introns)...")
    all_introns = list(range(256))
    canonical_full, orbit_sizes_full, reps_full = _compute_sccs(ep, idx_to_state, all_introns)
    print(f"  Found {len(reps_full)} full-graph orbits")

    # Step 2: Compute SCCs on the parity-free graph
    print("Step 2: Computing orbits on parity-free graph (excluding LI)...")
    parity_free_introns = [i for i in all_introns if not (i & LI_MASK)]
    canonical_pf, orbit_sizes_pf, reps_pf = _compute_sccs(ep, idx_to_state, parity_free_introns)
    print(f"  Found {len(reps_pf)} parity-free orbits")

    # Step 3: Identify mirror pairs on the parity-free structure
    print("Step 3: Identifying mirror pairs using parity-free orbits...")
    mirror_map: Dict[int, int] = {}
    for rep in reps_pf:
        if rep in mirror_map:
            continue
        state_int = int(idx_to_state[rep])
        mirror_state = state_int ^ governance.FULL_MASK
        
        # Binary search in idx_to_state for the mirror state
        m_pos = np.searchsorted(idx_to_state, mirror_state)
        if m_pos >= N or idx_to_state[m_pos] != mirror_state:
            mirror_map[rep] = -1
            continue
            
        mirror_rep = int(canonical_pf[m_pos])
        mirror_map[rep] = mirror_rep
        mirror_map[mirror_rep] = rep
        
    mirror_pairs = [(a, b) for a, b in mirror_map.items() if a < b and b != -1]
    num_mirror_pairs = len(mirror_pairs)
    print(f"  Found {num_mirror_pairs} non-trivial mirror pairs")

    # Step 4: Build cross-layer mapping 
    print("Step 4: Building cross-layer mapping...")
    pf_rep_to_full_rep: Dict[int, int] = {}
    for pf_rep in reps_pf:
        pf_rep_to_full_rep[pf_rep] = int(canonical_full[pf_rep])
        
    # Analyze how parity-free orbits combine in the full graph
    split_counter: Dict[int, int] = {}
    for full_rep in pf_rep_to_full_rep.values():
        split_counter[full_rep] = split_counter.get(full_rep, 0) + 1
        
    # Build histogram of split counts
    split_hist: Dict[str, int] = {}
    for cnt in split_counter.values():
        split_hist[str(cnt)] = split_hist.get(str(cnt), 0) + 1
    print(f"  Full-orbit split histogram: {split_hist}")

    # Step 5: Create and save the artifact
    artifact = {
        "schema_version": "phenomenology/dual/1.0.0",
        "metadata": {
            "build_timestamp": int(time.time()),
            "notes": [
                "This artifact contains two phenomenological views:",
                "full_graph: SCCs over all introns (parity collapsed, self-mirrored)",
                "parity_free: SCCs excluding LI introns (reveals mirror pairing)"
            ]
        },
        "full_graph": {
            "map": canonical_full.tolist(),
            "orbit_sizes": {str(k): int(v) for k, v in orbit_sizes_full.items()},
            "stats": {
                "total_orbits": len(reps_full),
                "largest_orbit": int(max(orbit_sizes_full.values())) if orbit_sizes_full else 0
            }
        },
        "parity_free": {
            "map": canonical_pf.tolist(),
            "orbit_sizes": {str(k): int(v) for k, v in orbit_sizes_pf.items()},
            "mirror_map": {str(k): int(v) for k, v in mirror_map.items()},
            "stats": {
                "total_orbits": len(reps_pf),
                "mirror_pairs": num_mirror_pairs,
                "largest_orbit": int(max(orbit_sizes_pf.values())) if orbit_sizes_pf else 0
            }
        },
        "cross_layer": {
            "pf_rep_to_full_rep": {str(k): int(v) for k, v in pf_rep_to_full_rep.items()},
            "full_orbit_split_histogram": split_hist
        }
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    # Print summary
    print("\nSummary:")
    print(f"  Full graph orbits: {artifact['full_graph']['stats']['total_orbits']}")
    print(f"  Parity-free orbits: {artifact['parity_free']['stats']['total_orbits']}")
    print(f"  Parity-free mirror pairs: {artifact['parity_free']['stats']['mirror_pairs']}")
    print(f"  Full-orbit split histogram: {artifact['cross_layer']['full_orbit_split_histogram']}")
    print(f"✓ Saved dual-structure phenomenology to: {output_path}")


def parse_args():
    """Parse command line arguments."""
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
                pheno = json.load(f)
            with open(args.ontology) as f:
                ont = json.load(f)
            print("\n--- Summary ---")
            print(f"Total states: {ont['endogenous_modulus']:,}")
            fg = pheno["full_graph"]["stats"]
            pf = pheno["parity_free"]["stats"]
            print(f"Full graph: {fg['total_orbits']} orbits | largest {fg['largest_orbit']}")
            print(f"Parity-free: {pf['total_orbits']} orbits | largest {pf['largest_orbit']} | mirror pairs {pf['mirror_pairs']}")
            print(f"Split histogram: {pheno['cross_layer']['full_orbit_split_histogram']}")

        else:
            print("Unknown command", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Re-raise to preserve stack trace for debugging
        raise