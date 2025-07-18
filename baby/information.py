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
        phenomap_path = ontology_data.get('phenomap_path')
        if not phenomap_path and 'ontology_map_path' in ontology_data:
            phenomap_path = ontology_data['ontology_map_path'].replace("ontology_map.json","phenomenology_map.json")
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
        
        print(msg + " " * 20, end='', flush=True)
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
        json.dump({
            "schema_version": "0.9.6",
            "ontology_map": json_compatible_map,
            "endogenous_modulus": 788_986,
            "ontology_diameter": 6,
            "total_states": 788_986,
            "build_timestamp": int(time.time())
        }, f)
    
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

## Curated autonomic cycle discovery
def _is_elementary(cycle: Tuple[int, ...]) -> bool:
    n = len(cycle)
    for k in range(1, n // 2 + 1):
        if n % k == 0 and cycle == cycle[:k] * (n // k):
            return False
    return True

def _canonical_rotation(cycle: Tuple[int, ...]) -> Tuple[int, ...]:
    if not cycle:
        return cycle
    best = cycle
    for i in range(1, len(cycle)):
        rotated = cycle[i:] + cycle[:i]
        if rotated < best:
            best = rotated
    return best

def _mean_divergence_for_cycle(ep: np.ndarray, idx_to_state: np.ndarray, origin_idx: int, cycle: Tuple[int, ...]) -> float:
    archetype_tensor = governance.GENE_Mac_S
    visited_indices = []
    current = origin_idx
    for intron in cycle[:-1]:
        current = ep[current, intron]
        visited_indices.append(current)
    if not visited_indices:
        return 0.0
    state_ints = idx_to_state[np.array(visited_indices, dtype=np.int32)]  # shape (k,)
    # We exclude the final origin revisit to avoid double-weighting the archetype.
    # Extract bits: bit 47 = MSB → position 0
    bit_positions = np.arange(47, -1, -1, dtype=np.uint64)  # 47..0
    bits = ((state_ints[:, None] >> bit_positions) & 1).astype(np.uint8)  # shape (k,48)
    tensors = (1 - 2 * bits).astype(np.int8).reshape(-1, 4, 2, 3, 2)
    flat_archetype = archetype_tensor.flatten()
    flats = tensors.reshape(tensors.shape[0], -1)
    cos = (flats @ flat_archetype) / 48.0
    np.clip(cos, -1.0, 1.0, out=cos)
    angles = np.arccos(cos)
    return float(np.mean(angles))

def curate_autonomic_cycles(ep: np.ndarray, origin_idx: int, idx_to_state: np.ndarray, max_length: int = 6, per_length: int = 2, operation_cap: int = 5_000_000) -> dict:
    raw_count = 0
    elementary_count = 0
    canonical_count = 0
    operation_count = 0
    op_cap_hit = False
    store = {L: {"homeo": [], "explore": []} for L in range(2, max_length + 1)}
    canonical_seen: set = set()
    for target_length in range(2, max_length + 1):
        stack = [(origin_idx, tuple(), {origin_idx})]
        while stack:
            operation_count += 1
            if operation_count > operation_cap:
                op_cap_hit = True
                stack.clear()
                break
            current_idx, path, visited = stack.pop()
            path_len = len(path)
            if path_len == target_length - 1:
                returning_introns = np.where(ep[current_idx] == origin_idx)[0]
                for intron in returning_introns:
                    raw_count += 1
                    cycle = path + (int(intron),)
                    if not _is_elementary(cycle):
                        continue
                    elementary_count += 1
                    canon = _canonical_rotation(cycle)
                    if canon in canonical_seen:
                        continue
                    canonical_seen.add(canon)
                    canonical_count += 1
                    divergence = _mean_divergence_for_cycle(ep, idx_to_state, origin_idx, canon)
                    L = target_length
                    homeo = store[L]["homeo"]
                    homeo.append((divergence, canon))
                    homeo.sort(key=lambda x: x[0])
                    if len(homeo) > per_length:
                        homeo.pop()
                    explore = store[L]["explore"]
                    explore.append((divergence, canon))
                    explore.sort(key=lambda x: x[0], reverse=True)
                    if len(explore) > per_length:
                        explore.pop()
                continue
            if path_len < target_length - 1:
                for intron in range(256):
                    nxt = ep[current_idx, intron]
                    if nxt not in visited:
                        stack.append((nxt, path + (intron,), visited | {nxt}))
    structured = {}
    stored_total = 0
    for L in range(2, max_length + 1):
        homeo_cycles = [list(c) for _, c in store[L]["homeo"]]
        explore_cycles = [list(c) for _, c in store[L]["explore"]]
        stored_total += len(homeo_cycles) + len(explore_cycles)
        structured[f"length_{L}"] = {
            "homeostatic": homeo_cycles,
            "exploratory": explore_cycles
        }
    return {
        "schema_version": "1",
        "cycles": structured,
        "stats": {
            "max_length": max_length,
            "per_length_each_class": per_length,
            "total_raw": raw_count,
            "total_elementary": elementary_count,
            "total_canonical": canonical_count,
            "stored": stored_total,
            "operation_cap_reached": op_cap_hit,
            "operation_cap": operation_cap
        }
    }

# ==============================================================================
# STEP 3: Phenomenology Map (OPTIMIZED)
# ==============================================================================
def build_phenomenology_map(ep_path: str, ontology_map: Dict[int, int], output_path: str, curate_cycles: bool = True, cycles_per_length: int = 2):
    """
    Build the phenomenology (canonical orbit) map using *strongly connected components* (SCCs).

    Definition (authoritative):
        Two states belong to the same phenomenological orbit iff they are mutually reachable
        via sequences of intron-induced transitions (i.e. they lie in the same SCC of the
        directed transition graph defined by epistemology table rows).

    Canonical representative:
        For each SCC C, let rep = argmin_{i in C} state_integer(i).
        The phenomenology map assigns every i in C → rep (using the *index* of that rep).
        This ensures:
            - Idempotence of canonicalization
            - Representative is minimal in its class (tests can verify)
            - Stable under rebuild if ontology_map ordering is stable

    Output JSON structure:
        {
            "phenomenology_map": [... representative_index per node ...],
            "orbit_sizes": { "<rep_index>": size, ... },
            "autonomic_cycles": [...],  # legacy: flattened, bounded list
            "autonomic_cycles_curated": {...},  # structured, stratified cycles and stats
            "build_timestamp": <unix>,
            "metadata": {
                "total_orbits": <count>,
                "largest_orbit": <max_size>,
                "total_cycles": <cycles_found>,
                "cycles_truncated": <bool>
            }
        }

    Performance notes:
        - Uses an *iterative* Tarjan SCC algorithm (no Python recursion).
        - Each node's outgoing edges are deduplicated with np.unique to reduce work
          when transitions repeat (common in highly symmetric regions).
        - Memory overhead: O(N) for arrays (indices, lowlink, stack flags, canonical map).

    Args:
        ep_path: Path to epistemology.npy (shape N x 256, int32).
        ontology_map: Dict[state_int -> index] (size N).
        output_path: Destination JSON file.
    """
    ep = np.load(ep_path, mmap_mode="r")
    N = ep.shape[0]

    # Map: index -> 48-bit state integer (uint64)
    idx_to_state = np.empty(N, dtype=np.uint64)
    for state_int, idx in ontology_map.items():
        idx_to_state[idx] = state_int

    # Tarjan (iterative) data structures
    indices = np.full(N, -1, dtype=np.int32)       # discovery index
    lowlink = np.zeros(N, dtype=np.int32)
    on_stack = np.zeros(N, dtype=bool)
    stack = []                                     # SCC stack (node indices)
    canonical = np.full(N, -1, dtype=np.int32)     # final representative index per node

    orbit_sizes: Dict[int, int] = {}
    index_counter = 0
    orbits_found = 0
    processed_nodes = 0

    # Progress
    start_time = time.time()
    REPORT_EVERY = 50_000

    def print_progress(force: bool = False):
        if processed_nodes == 0 and not force:
            return
        if force or processed_nodes % REPORT_EVERY == 0:
            elapsed = time.time() - start_time
            rate = processed_nodes / elapsed if elapsed > 0 else 0
            pct = 100.0 * processed_nodes / N
            msg = (f"\rPhenomenology (SCC): {processed_nodes:,}/{N:,} "
                   f"({pct:5.1f}%) | {rate:,.0f}/s | orbits={orbits_found}")
            print(msg, end='', flush=True)

    # Frame structure for iterative DFS:
    # (node, next_child_idx, neighbors_array)
    class Frame:
        __slots__ = ("v", "i", "nbrs")
        def __init__(self, v: int, nbrs: np.ndarray):
            self.v = v
            self.i = 0
            self.nbrs = nbrs

    # Main loop: visit all nodes
    for root in range(N):
        if indices[root] != -1:
            continue

        # Build neighbor list for root (deduplicate as per docstring)
        root_neighbors = np.unique(ep[root])
        # Initialize DFS stack with root frame AFTER assigning index/root state
        dfs_stack: List[Frame] = []

        # "Visit" root
        indices[root] = index_counter
        lowlink[root] = index_counter
        index_counter += 1
        stack.append(root)
        on_stack[root] = True
        dfs_stack.append(Frame(root, root_neighbors))

        # Iterative Tarjan DFS
        while dfs_stack:
            frame = dfs_stack[-1]
            v = frame.v

            if frame.i < frame.nbrs.size:
                w = int(frame.nbrs[frame.i])
                frame.i += 1

                if indices[w] == -1:
                    # First time we see w
                    nbrs_w = np.unique(ep[w])
                    indices[w] = index_counter
                    lowlink[w] = index_counter
                    index_counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    dfs_stack.append(Frame(w, nbrs_w))
                elif on_stack[w]:
                    # Update lowlink on back-edge
                    if indices[w] < lowlink[v]:
                        lowlink[v] = indices[w]
            else:
                # Finished exploring v's neighbors
                dfs_stack.pop()
                # Propagate lowlink to parent (if any)
                if dfs_stack:
                    parent = dfs_stack[-1].v
                    if lowlink[v] < lowlink[parent]:
                        lowlink[parent] = lowlink[v]

                # If root of SCC
                if lowlink[v] == indices[v]:
                    # Pop stack until v included
                    component = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        component.append(w)
                        if w == v:
                            break

                    comp_arr = np.array(component, dtype=np.int32)
                    # Choose representative by minimal *state integer*
                    comp_states = idx_to_state[comp_arr]
                    rep_local_idx = int(comp_arr[np.argmin(comp_states)])
                    canonical[comp_arr] = rep_local_idx
                    size = comp_arr.size
                    orbit_sizes[rep_local_idx] = size
                    orbits_found += 1
                    processed_nodes += size
                    print_progress()

    print_progress(force=True)
    print()  # newline after progress line

    # Validation
    assert processed_nodes == N, f"SCC coverage mismatch: processed {processed_nodes} of {N}"
    assert np.all(canonical >= 0), "Canonical map contains unassigned entries"
    assert len(orbit_sizes) == len(set(orbit_sizes.keys())), "Duplicate representatives detected"

    # --- Autonomic cycles (curated) ---
    print("Curating autonomic cycles...", end='', flush=True)
    origin_state = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    origin_idx = ontology_map[origin_state]
    if curate_cycles:
        cycles_payload = curate_autonomic_cycles(
            ep,
            origin_idx,
            idx_to_state,
            max_length=6,
            per_length=cycles_per_length
        )
        cycles_truncated = cycles_payload["stats"]["operation_cap_reached"]
        legacy_cycle_list = []
        for L_key, classes in cycles_payload["cycles"].items():
            legacy_cycle_list.extend(classes["homeostatic"])
            legacy_cycle_list.extend(classes["exploratory"])
        print(f"\rCurated {cycles_payload['stats']['stored']} autonomic cycles (bounded)." + " " * 30)
        if curate_cycles and cycles_payload["stats"]["stored"] == 0:
            print("\nWarning: No autonomic cycles curated.")
    else:
        cycles_payload = {
            "schema_version": "1",
            "cycles": {},
            "stats": {
                "stored": 0,
                "operation_cap_reached": False
            }
        }
        legacy_cycle_list = []
        cycles_truncated = False

    # Persist
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "phenomenology_map": canonical.tolist(),
            "orbit_sizes": {str(k): int(v) for k, v in orbit_sizes.items()},
            "autonomic_cycles": legacy_cycle_list,
            "autonomic_cycles_curated": cycles_payload,
            "build_timestamp": int(time.time()),
            "metadata": {
                "total_orbits": len(orbit_sizes),
                "largest_orbit": int(max(orbit_sizes.values())) if orbit_sizes else 0,
                "total_cycles": cycles_payload["stats"]["stored"],
                "cycles_truncated": cycles_truncated
            }
        }, f)

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
            with open(args.ontology) as f:
                ontology_data = json.load(f)
            ontology_map = {int(k): v for k, v in ontology_data["ontology_map"].items()}
            build_phenomenology_map(args.ep, ontology_map, args.output)
            with open(args.output) as f:
                pheno_data = json.load(f)
            print(f"\n✓ Saved: {args.output}")
            metadata = pheno_data.get("metadata", {})
            print(f"  - Total states: {ontology_data['endogenous_modulus']:,}")
            print(f"  - Unique orbits: {metadata.get('total_orbits', len(pheno_data.get('orbit_sizes', {}))):,}")
            print(f"  - Autonomic cycles: {metadata.get('total_cycles', len(pheno_data.get('autonomic_cycles', []))):,}")
            if metadata.get('cycles_truncated', False):
                print("  - Warning: Cycle search was truncated due to operation limit")
            print(f"  - Largest orbit: {metadata.get('largest_orbit', 0):,} states\n")

        else:
            print("Unknown command", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Re-raise to preserve stack trace for debugging
        raise