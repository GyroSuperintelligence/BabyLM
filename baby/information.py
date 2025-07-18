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
import numpy as np
import sys
from typing import Dict, Any, List, Tuple, Set, Optional

from baby import governance

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
    
    
# ==============================================================================
# Utility: Clean single-line progress reporter
# ==============================================================================
class ProgressReporter:
    def __init__(self, desc: str):
        self.desc = desc
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, current: int, total: Optional[int] = None, extra: str = ""):
        now = time.time()
        if now - self.last_update < 0.1:  # Throttle updates to 10Hz
            return
        
        elapsed = now - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        
        msg = f"\r{self.desc}: {current:,}"
        if total is not None:
            pct = 100.0 * current / total
            msg += f"/{total:,} ({pct:.1f}%)"
        msg += f" | {rate:.0f}/s | {elapsed:.1f}s"
        if extra:
            msg += f" | {extra}"
        
        print(msg + " " * 10, end='', flush=True)
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
    
    while current_level:
        next_level = []
        for state in current_level:
            for intron in range(256):
                next_state = governance.apply_gyration_and_transform(state, intron)
                if next_state not in discovered:
                    discovered.add(next_state)
                    next_level.append(next_state)
        
        current_level = next_level
        depth += 1
        progress.update(len(discovered), extra=f"depth={depth}")
    
    progress.done()
    
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
        json.dump({
            "schema_version": "0.9.6",
            "ontology_map": ontology_map,
            "endogenous_modulus": 788_986,
            "ontology_diameter": 6,
            "total_states": 788_986,
            "build_timestamp": time.time()
        }, f)
    
    return ontology_map

# ==============================================================================
# STEP 2: Epistemology Table
# ==============================================================================
def build_state_transition_table(ontology_map: Dict[int, int], output_path: str):
    """Builds the N×256 state transition table."""
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
            # Vectorized transformation
            next_states = np.array([
                governance.apply_gyration_and_transform(int(s), intron) 
                for s in chunk_states
            ], dtype=np.uint64)
            
            # Map to indices via binary search
            ep[chunk_start:chunk_end, intron] = np.searchsorted(states, next_states)
        
        progress.update(chunk_end, N)
    
    ep.flush()
    progress.done()
    return ep

# ==============================================================================
# STEP 3: Phenomenology Map with Fixed Autonomic Cycles
# ==============================================================================
def find_simple_cycles(ep: np.ndarray, start_idx: int, max_length: int = 6) -> List[Tuple[int, ...]]:
    """Finds simple cycles using iterative deepening DFS with path tracking."""
    cycles = []
    for target_length in range(2, max_length + 1):
        # Use iterative deepening to find cycles of exactly target_length
        stack: List[Tuple[int, Tuple[int, ...], Set[int]]] = [(start_idx, tuple(), set([start_idx]))]
        while stack:
            curr_idx, path, visited = stack.pop()
            if len(path) == target_length:
                # Check if we can return to start
                if start_idx in ep[curr_idx]:
                    intron = np.where(ep[curr_idx] == start_idx)[0][0]
                    cycles.append(path + (int(intron),))
                continue
            # Explore neighbors
            for intron in range(256):
                next_idx = ep[curr_idx, intron]
                if next_idx not in visited or (next_idx == start_idx and len(path) == target_length - 1):
                    new_visited = visited | {next_idx} if next_idx != start_idx else visited
                    stack.append((next_idx, path + (intron,), new_visited))
    return cycles

def build_phenomenology_map(ep_path: str, ontology_map: Dict[int, int], output_path: str):
    """Builds canonical orbit map and finds autonomic cycles."""
    progress = ProgressReporter("Building phenomenology")
    
    ep = np.load(ep_path, mmap_mode="r")
    N = ep.shape[0]
    
    # Canonical representatives
    canonical = np.full(N, -1, dtype=np.int32)
    
    # Inverse mapping for lexicographic comparison
    idx_to_state = {idx: state for state, idx in ontology_map.items()}
    
    orbit_sizes = {}
    orbits_found = 0
    
    # Find orbits
    for seed in range(N):
        if canonical[seed] != -1:
            continue
        
        # BFS to find orbit
        orbit = {seed}
        frontier = [seed]
        min_state = idx_to_state[seed]
        min_idx = seed
        
        while frontier:
            current = frontier.pop(0)
            for next_idx in ep[current]:
                if next_idx not in orbit:
                    orbit.add(next_idx)
                    frontier.append(next_idx)
                    
                    # Track lexicographic minimum
                    state = idx_to_state[next_idx]
                    if state < min_state:
                        min_state = state
                        min_idx = next_idx
        
        # Assign canonical representative
        for idx in orbit:
            canonical[idx] = min_idx
        
        orbit_sizes[min_idx] = len(orbit)
        orbits_found += 1
        progress.update(sum(orbit_sizes.values()), N, f"orbits={orbits_found}")
    
    progress.done()
    
    # Find autonomic cycles from archetype
    print("\rFinding autonomic cycles...", end='', flush=True)
    origin_state = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    origin_idx = ontology_map[origin_state]
    cycles = find_simple_cycles(ep, origin_idx, max_length=6)
    print(f"\rFound {len(cycles)} autonomic cycles" + " " * 30)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump({
            "phenomenology_map": canonical.tolist(),
            "orbit_sizes": {str(k): int(v) for k, v in orbit_sizes.items()},
            "autonomic_cycles": [list(c) for c in cycles]
        }, f)
            
def parse_args():
    parser = argparse.ArgumentParser(description="GyroSI asset builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ontology
    p_ont = subparsers.add_parser("ontology")
    p_ont.add_argument("--output", required=True)

    # Epistemology
    p_epi = subparsers.add_parser("epistemology")
    p_epi.add_argument("--ontology", required=True)
    p_epi.add_argument("--output", required=True)

    # Phenomenology
    p_pheno = subparsers.add_parser("phenomenology")
    p_pheno.add_argument("--ep", required=True)
    p_pheno.add_argument("--output", required=True)
    p_pheno.add_argument("--ontology", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "ontology":
        print("=== [Step 1] Ontology Generation ===")
        ontology_map = discover_and_save_ontology(args.output)
        print(f"✓ Saved: {args.output}")
        print(f"✓ Total states discovered: {len(ontology_map):,}")
        print()

    elif args.command == "epistemology":
        print("=== [Step 2] Epistemology Table ===")
        with open(args.ontology) as f:
            ontology_data = json.load(f)
        ontology_map = {int(k): v for k, v in ontology_data["ontology_map"].items()}
        build_state_transition_table(ontology_map, args.output)
        file_size = os.path.getsize(args.output) / 1024**2
        print(f"✓ Saved: {args.output}")
        print(f"✓ File size: {file_size:.1f} MB")
        print()

    elif args.command == "phenomenology":
        print("=== [Step 3] Phenomenology Mapping ===")
        with open(args.ontology) as f:
            ontology_data = json.load(f)
        ontology_map = {int(k): v for k, v in ontology_data["ontology_map"].items()}
        build_phenomenology_map(args.ep, ontology_map, args.output)
        print(f"✓ Saved: {args.output}")
        with open(args.output) as f:
            pheno_data = json.load(f)
        print(f"✓ Total states: {ontology_data['endogenous_modulus']:,}")
        print(f"✓ Unique orbits: {len(pheno_data['orbit_sizes']):,}")
        print(f"✓ Autonomic cycles: {len(pheno_data['autonomic_cycles']):,}")
        print(f"✓ Largest orbit: {max(pheno_data['orbit_sizes'].values()):,} states")
        print()

    else:
        print("Unknown command", file=sys.stderr)
        sys.exit(1)
