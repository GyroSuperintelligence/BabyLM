import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from baby import governance

warnings.filterwarnings("ignore", message=".*found in sys.modules after import of package.*")
"""
S2: Information - Measurement & Storage

This module provides the InformationEngine class responsible for measurement,
storage coordination, and conversion between state representations.

Build steps:
    python -m baby.information ontology     --output memories/public/meta/ontology_map.json
    python -m baby.information epistemology --ontology memories/public/meta/ontology_map.json \
           --output  memories/public/meta/epistemology.npy
    python -m baby.information phenomenology --ep memories/public/meta/epistemology.npy \
           --ontology memories/public/meta/ontology_map.json \
           --output memories/public/meta/phenomenology_map.json
"""


# Try to use ujson for speed, fall back to standard json if unavailable
try:
    import ujson as json
except ImportError:
    import json


class InformationEngine:
    """
    S2: Measurement & Resource Coordination.

    Sole authority for measurement and conversion between state representations.
    Provides the sensory apparatus through angular gyrodistance measurement.

    If use_array_indexing is True, ontology_map and inverse_ontology_map are stored
    as numpy arrays for better memory/cache performance.
    """

    ontology_map: Optional[Dict[int, int]]
    inverse_ontology_map: Optional[Dict[int, int]]
    _keys: Optional[np.ndarray]
    _inverse: Optional[np.ndarray]

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
        if self.ontology_map is not None:
            keys = list(self.ontology_map.keys())
            if keys and isinstance(keys[0], str):
                self.ontology_map = {int(k): v for k, v in self.ontology_map.items()}
        self.endogenous_modulus = ontology_data["endogenous_modulus"]
        self.ontology_diameter = ontology_data["ontology_diameter"]

        if use_array_indexing:
            # Note: This assumes ontology indices were assigned in sorted order of state integers
            # If ontology was generated differently, this implicitly redefines indices via sorted position
            if self.ontology_map is None:
                raise RuntimeError("Ontology map must be provided.")
            keys_arr = np.array(sorted(self.ontology_map.keys()), dtype=np.uint64)
            # Assert that the mapping is index == sorted position (array mode assumption)
            sorted_map = self.ontology_map
            assert all(v == i for i, v in enumerate(sorted_map.values())), (
                "Ontology map indices must match sorted order for array mode. "
                "If this fails, store a real inverse array instead."
            )
            self._keys = keys_arr
            self._inverse = keys_arr  # index -> state_int
            # Free memory: drop the original mapping in array mode
            self.ontology_map = None
            self.inverse_ontology_map = None  # free memory
        else:
            self._keys = None
            self._inverse = None
            if self.ontology_map is not None:
                self.inverse_ontology_map = {v: k for k, v in self.ontology_map.items()}
            else:
                self.inverse_ontology_map = None

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

        # Load θ table (theta.npy) if available
        self._theta_table = None
        default_theta = "memories/public/meta/theta.npy"
        if "ontology_map_path" in ontology_data:
            default_theta = ontology_data["ontology_map_path"].replace("ontology_map.json", "theta.npy")
        elif "phenomap_path" in ontology_data:
            default_theta = ontology_data["phenomap_path"].replace("phenomenology_map.json", "theta.npy")
        if os.path.exists(default_theta):
            self._theta_table = np.load(default_theta, mmap_mode="r")

        # Cache the archetypal state as an int
        self._origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        # Precompute acos LUT for 0..48 bits
        self._acos_lut = np.arccos(1 - 2 * np.arange(49) / 48.0).astype(np.float32)
        # Debug/env switch for fallback
        self._use_fast_divergence = os.environ.get("BABYLM_USE_FAST_DIVERGENCE", "1") != "0"

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
    def int_to_tensor(state_int: int) -> "np.ndarray[np.int8, Any]":
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

        return float(np.arccos(cosine_similarity))

    def _angular_divergence_fast(self, state_int: int) -> float:
        """
        Fast angular divergence using XOR+bit_count and LUT.
        """
        h = (state_int ^ self._origin_int).bit_count()
        return float(self._acos_lut[h])

    def measure_state_divergence(self, state_int: int) -> float:
        if self._theta_table is not None:
            idx = self.get_index_from_state(state_int)
            return float(self._theta_table[idx])
        # Fallbacks (shouldn’t trigger in production)
        if self._use_fast_divergence:
            return self._angular_divergence_fast(state_int)
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
def build_state_transition_table(ontology_map: Dict[int, int], output_path: str) -> None:
    """Builds the N×256 state transition table with validation."""
    progress = ProgressReporter("Building epistemology")

    N = len(ontology_map)
    states = np.array(sorted(ontology_map.keys()), dtype=np.uint64)

    # ----- θ table (angular divergence from origin) -----
    theta_path = output_path.replace("epistemology.npy", "theta.npy")
    origin = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
    acos_lut = np.arccos(1 - 2 * np.arange(49) / 48.0).astype(np.float32)
    theta = np.empty(N, dtype=np.float32)
    for i, s in enumerate(states):
        h = int(s ^ origin).bit_count()
        theta[i] = acos_lut[h]

    # Memory-mapped output
    ep = np.lib.format.open_memmap(output_path, dtype=np.int32, mode="w+", shape=(N, 256))

    # Process in chunks for memory efficiency
    CHUNK_SIZE = 10_000
    for chunk_start in range(0, N, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        chunk_states = states[chunk_start:chunk_end]
        # Vectorized: apply all introns at once
        next_states_all = governance.apply_gyration_and_transform_all_introns(chunk_states)
        # next_states_all shape: (chunk_len, 256)
        idxs = np.searchsorted(states, next_states_all, side='left')
        # Debug check: ensure all next_states are in the ontology
        if __debug__:
            if idxs.max() >= states.size or not np.all(states[idxs] == next_states_all):
                raise RuntimeError("Transition produced unknown state.")
        ep[chunk_start:chunk_end, :] = idxs
        progress.update(chunk_end, N)

    # Save theta table
    np.save(theta_path, theta)
    ep.flush()
    progress.done()


# ==============================================================================
# STEP 3: Phenomenology Map (Core + Optional Diagnostics)
# ==============================================================================


def _compute_sccs(
    ep: np.ndarray, idx_to_state: np.ndarray, introns_to_use: List[int]
) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
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

    def neighbors(v: int) -> "np.ndarray[np.int32, Any]":
        # Return all neighbors; duplicates are fine
        return ep[v, introns_arr]

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
    ep_path: str, ontology_path: str, output_path: str, include_diagnostics: bool = False
) -> None:
    """
    Builds the canonical phenomenology map for GyroSI runtime operations.

    The core phenomenology uses SCCs over all 256 introns, creating 256 parity-closed
    orbits. This honors the theoretical principle that CS (Common Source) is unobservable
    and that UNA (global parity/LI) represents the reflexive confinement of light itself.

    Each orbit is self-mirrored, meaning states and their parity complements belong to
    the same equivalence class. This preserves the fundamental symmetry while providing
    the deterministic canonicalization needed for the knowledge store.

    Args:
        ep_path: Path to epistemology.npy
        ontology_path: Path to ontology_map.json
        output_path: Path to save phenomenology_map.json
        include_diagnostics: If True, also compute parity-free analysis for research
    """
    print("=== [Phenomenology Core Builder] ===")

    # Load data
    ep = np.load(ep_path, mmap_mode="r")
    with open(ontology_path) as f:
        ontology_data = json.load(f)
    N = ep.shape[0]

    # Build index→state lookup array
    idx_to_state = np.empty(N, dtype=np.uint64)
    for k_str, idx in ontology_data["ontology_map"].items():
        idx_to_state[idx] = int(k_str)

    # Core: Compute canonical phenomenology (all 256 introns)
    print("Computing canonical phenomenology (all 256 introns)...")
    all_introns = list(range(256))
    canonical, orbit_sizes, representatives = _compute_sccs(ep, idx_to_state, all_introns)
    print(f"  Found {len(representatives)} canonical orbits (expected 256)")

    # Create core artifact
    artifact: Dict[str, Any] = {
        "schema_version": "phenomenology/core/1.0.0",
        "phenomenology_map": canonical.tolist(),
        "orbit_sizes": {str(k): int(v) for k, v in orbit_sizes.items()},
        "metadata": {
            "total_states": N,
            "total_orbits": len(representatives),
            "largest_orbit": int(max(orbit_sizes.values())) if orbit_sizes else 0,
            "build_timestamp": int(time.time()),
            "construction": {
                "method": "scc_all_introns",
                "notes": [
                    "Canonical phenomenology with LI (global parity) included.",
                    "Each orbit is parity-closed and self-mirrored.",
                    "This honors the principle that CS (Common Source) is unobservable.",
                    "UNA (LI) represents reflexive confinement - the 'light' that cannot be stepped outside of.",
                ],
            },
        },
    }

    # Optional: Add diagnostic analysis
    if include_diagnostics:
        print("Computing diagnostic analysis (parity-free structure)...")

        # Global parity (LI) bit mask pattern
        LI_MASK = 0b01000010
        parity_free_introns = [i for i in all_introns if not (i & LI_MASK)]
        canonical_pf, orbit_sizes_pf, reps_pf = _compute_sccs(ep, idx_to_state, parity_free_introns)

        # Mirror pair analysis on parity-free structure
        mirror_map: Dict[int, int] = {}
        for rep in reps_pf:
            if rep in mirror_map:
                continue
            state_int = int(idx_to_state[rep])
            mirror_state = state_int ^ governance.FULL_MASK

            m_pos = np.searchsorted(idx_to_state, mirror_state)
            if m_pos >= N or idx_to_state[m_pos] != mirror_state:
                mirror_map[rep] = -1
                continue

            mirror_rep = int(canonical_pf[m_pos])
            mirror_map[rep] = mirror_rep
            mirror_map[mirror_rep] = rep

        mirror_pairs = [(a, b) for a, b in mirror_map.items() if a < b and b != -1]

        # Cross-layer mapping analysis
        pf_to_canonical = {pf_rep: int(canonical[pf_rep]) for pf_rep in reps_pf}
        split_counter: Dict[int, int] = {}
        for canonical_rep in pf_to_canonical.values():
            split_counter[canonical_rep] = split_counter.get(canonical_rep, 0) + 1
        split_histogram: Dict[str, int] = {}
        for count in split_counter.values():
            split_histogram[str(count)] = split_histogram.get(str(count), 0) + 1

        # Add diagnostics to artifact
        artifact["_diagnostics"] = {
            "note": "Research data - not used by runtime engines",
            "parity_free_analysis": {
                "total_orbits": len(reps_pf),
                "mirror_pairs": len(mirror_pairs),
                "largest_orbit": int(max(orbit_sizes_pf.values())) if orbit_sizes_pf else 0,
                "canonical_orbit_split_histogram": split_histogram,
            },
            "theoretical_insights": [
                f"Removing LI reveals {len(reps_pf)} fine-grained orbits vs {len(representatives)} canonical orbits",
                f"Found {len(mirror_pairs)} chiral pairs and {len(reps_pf) - 2*len(mirror_pairs)} achiral orbits",
                "This demonstrates the binding power of global parity (UNA/LI) in the system",
            ],
        }

        print(f"  Diagnostic: {len(reps_pf)} parity-free orbits, {len(mirror_pairs)} mirror pairs")

    # Save artifact
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    # Print summary
    print(f"\n✓ Saved canonical phenomenology to: {output_path}")
    print(f"  - Canonical orbits: {artifact['metadata']['total_orbits']}")
    print(f"  - Largest orbit: {artifact['metadata']['largest_orbit']} states")
    if include_diagnostics:
        diag = artifact["_diagnostics"]["parity_free_analysis"]
        print(f"  - Diagnostic: {diag['total_orbits']} parity-free orbits, {diag['mirror_pairs']} mirror pairs")


def parse_args() -> argparse.Namespace:
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
    p_pheno.add_argument(
        "--diagnostics", action="store_true", help="Include parity-free analysis for research (optional)"
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
            with open(args.ontology) as f:
                ontology_data = json.load(f)
            ontology_map = {int(k): v for k, v in ontology_data["ontology_map"].items()}
            build_state_transition_table(ontology_map, args.output)
            file_size = os.path.getsize(args.output) / 1024**2
            print(f"\n✓ Saved: {args.output}")
            print(f"✓ File size: {file_size:.1f} MB\n")

        elif args.command == "phenomenology":
            print("=== [Step 3] Phenomenology Mapping ===")
            include_diag = getattr(args, "diagnostics", False)
            build_phenomenology_map(args.ep, args.ontology, args.output, include_diag)

            # Print final summary
            with open(args.output) as f:
                pheno = json.load(f)
            with open(args.ontology) as f:
                ont = json.load(f)

            print("\n--- Final Summary ---")
            print(f"Total states: {ont['endogenous_modulus']:,}")
            print(f"Canonical orbits: {pheno['metadata']['total_orbits']} (parity-closed)")
            print(f"Largest orbit: {pheno['metadata']['largest_orbit']:,} states")

            if "_diagnostics" in pheno:
                diag = pheno["_diagnostics"]["parity_free_analysis"]
                print(
                    f"Research diagnostic: {diag['total_orbits']} parity-free orbits, "
                    f"{diag['mirror_pairs']} chiral pairs"
                )

        else:
            print("Unknown command", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # Re-raise to preserve stack trace for debugging
        raise
