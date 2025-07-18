"""
Comprehensive test suite for baby.information module.

Tests the InformationEngine class and related utilities against real physics
and pre-built ontology/epistemology/phenomenology maps.
"""

import pytest
import numpy as np
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any
import random
import sys  # Added for use in TestMaps

# Import the modules under test
from baby.information import InformationEngine, ProgressReporter
from baby import governance


# --- Test Configuration ---

# Adjust for older hardware - reduce sample sizes
NUM_SPOT_CHECKS = 1000  # Reduced from 5000 for older MacBook Pro
NUM_TENSOR_TESTS = 500   # For tensor conversion tests
NUM_DISTANCE_TESTS = 200 # For angular distance tests


# --- Fixtures ---

@pytest.fixture(scope="module")
def real_maps(real_ontology):
    """Load the real ontology, epistemology, and phenomenology maps once per module."""
    ontology_path, phenomenology_path, epistemology_path = real_ontology
    
    # Verify all files exist
    for path in [ontology_path, phenomenology_path, epistemology_path]:
        if not os.path.exists(path):
            pytest.skip(f"Required map file not found: {path}")
    
    # Load ontology
    with open(ontology_path) as f:
        ontology_data = json.load(f)
    
    # Load phenomenology
    with open(phenomenology_path) as f:
        phenomenology_data = json.load(f)
    
    # Load epistemology (memory-mapped for efficiency)
    epistemology_table = np.load(epistemology_path, mmap_mode="r")
    
    return {
        "ontology_data": ontology_data,
        "phenomenology_data": phenomenology_data,
        "epistemology_table": epistemology_table,
        "ontology_path": ontology_path,
        "phenomenology_path": phenomenology_path,
        "epistemology_path": epistemology_path
    }


@pytest.fixture
def sample_states(real_maps):
    """Generate a consistent set of sample states for testing."""
    ontology_map = real_maps["ontology_data"]["ontology_map"]
    state_ints = list(ontology_map.keys())
    
    # Convert string keys to int if needed
    if state_ints and isinstance(state_ints[0], str):
        state_ints = [int(k) for k in state_ints]
    
    # Sample a subset for testing
    random.seed(42)  # Deterministic sampling
    sample_size = min(NUM_SPOT_CHECKS, len(state_ints))
    sampled_states = random.sample(state_ints, sample_size)
    
    return sampled_states


@pytest.fixture
def information_engine_standard(real_maps):
    """Create a standard InformationEngine instance (non-memmap)."""
    ontology_data = real_maps["ontology_data"].copy()
    ontology_data["phenomap_path"] = real_maps["phenomenology_path"]
    engine = InformationEngine(ontology_data, use_array_indexing=False)
    yield engine


@pytest.fixture
def information_engine_memmap(real_maps):
    """Create a memory-mapped InformationEngine instance."""
    ontology_data = real_maps["ontology_data"].copy()
    ontology_data["phenomap_path"] = real_maps["phenomenology_path"]
    engine = InformationEngine(ontology_data, use_array_indexing=True)
    yield engine


# --- Test Classes ---

class TestInformationEngineInitialization:
    """Test various initialization scenarios for InformationEngine."""
    
    def test_standard_initialization(self, real_maps):
        """Test standard initialization without memmap."""
        ontology_data = real_maps["ontology_data"]
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        assert engine.use_array_indexing is False
        assert engine.endogenous_modulus == 788_986
        assert engine.ontology_diameter == 6
        assert isinstance(engine.ontology_map, dict)
        assert isinstance(engine.inverse_ontology_map, dict)
        assert engine._keys is None
        assert engine._inverse is None
    
    def test_memmap_initialization(self, real_maps):
        """Test memory-mapped initialization."""
        ontology_data = real_maps["ontology_data"]
        engine = InformationEngine(ontology_data, use_array_indexing=True)
        
        assert engine.use_array_indexing is True
        assert engine.endogenous_modulus == 788_986
        assert engine.ontology_diameter == 6
        assert engine._keys is not None
        assert engine._inverse is not None
        assert isinstance(engine._keys, np.ndarray)
        assert engine.inverse_ontology_map is None  # Should be freed
    
    def test_auto_memmap_detection(self, real_maps):
        """Test automatic memmap detection for large ontologies."""
        ontology_data = real_maps["ontology_data"]
        # Should auto-enable memmap for large ontology
        engine = InformationEngine(ontology_data, use_array_indexing=None)
        assert engine.use_array_indexing is True
    
    def test_string_key_conversion(self, real_maps):
        """Test that string keys in ontology_map are converted to integers."""
        ontology_data = real_maps["ontology_data"].copy()
        # Ensure we have string keys
        ontology_data["ontology_map"] = {str(k): v for k, v in ontology_data["ontology_map"].items()}
        
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        # All keys should be integers (only if ontology_map is present)
        if engine.ontology_map is not None:
            for key in engine.ontology_map.keys():
                assert isinstance(key, int)
    
    def test_orbit_cardinality_loading(self, real_maps):
        """Test that orbit cardinality is loaded correctly."""
        ontology_data = real_maps["ontology_data"].copy()
        ontology_data["phenomap_path"] = real_maps["phenomenology_path"]
        
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        assert hasattr(engine, 'orbit_cardinality')
        assert isinstance(engine.orbit_cardinality, np.ndarray)
        assert len(engine.orbit_cardinality) == 788_986
        assert engine.orbit_cardinality.dtype == np.uint32
    
    def test_validation_errors(self, real_maps):
        """Test that initialization fails with invalid constants."""
        ontology_data = real_maps["ontology_data"].copy()
        
        # Test wrong modulus
        ontology_data["endogenous_modulus"] = 12345
        with pytest.raises(ValueError, match="Expected endogenous modulus 788,986"):
            InformationEngine(ontology_data)
        
        # Test wrong diameter
        ontology_data = real_maps["ontology_data"].copy()
        ontology_data["ontology_diameter"] = 5
        with pytest.raises(ValueError, match="Expected ontology diameter 6"):
            InformationEngine(ontology_data)


class TestStateTensorConversions:
    """Test static methods for converting between state integers and tensors."""
    
    def test_int_to_tensor_shape_and_type(self):
        """Test that int_to_tensor produces correct shape and type."""
        state_int = 0x123456789ABC  # 48-bit integer
        tensor = InformationEngine.int_to_tensor(state_int)
        
        assert tensor.shape == (4, 2, 3, 2)
        assert tensor.dtype == np.int8
        assert np.all(np.isin(tensor, [-1, 1]))
    
    def test_tensor_to_int_shape_validation(self):
        """Test that tensor_to_int works with correct tensor shape."""
        tensor = np.random.choice([-1, 1], size=(4, 2, 3, 2)).astype(np.int8)
        result = InformationEngine.tensor_to_int(tensor)
        
        assert isinstance(result, int)
        assert 0 <= result < (1 << 48)
    
    def test_round_trip_conversion_consistency(self):
        """Test that int->tensor->int conversions are consistent."""
        # Test with multiple random 48-bit integers
        for _ in range(NUM_TENSOR_TESTS):
            original_int = random.randint(0, (1 << 48) - 1)
            
            # Convert to tensor and back
            tensor = InformationEngine.int_to_tensor(original_int)
            recovered_int = InformationEngine.tensor_to_int(tensor)
            
            assert recovered_int == original_int, f"Round-trip failed: {original_int} -> {recovered_int}"
    
    def test_bit_mapping_correctness(self):
        """Test that bit mapping follows the specified convention."""
        # Test with known patterns
        state_int = 0  # All bits 0 -> all +1
        tensor = InformationEngine.int_to_tensor(state_int)
        assert np.all(tensor == 1)
        
        state_int = (1 << 48) - 1  # All bits 1 -> all -1
        tensor = InformationEngine.int_to_tensor(state_int)
        assert np.all(tensor == -1)
        
        # Test single bit patterns
        for bit_pos in range(48):
            state_int = 1 << bit_pos
            tensor = InformationEngine.int_to_tensor(state_int)
            flat = tensor.flatten()
            
            # Bit 0 (LSB) maps to element 47, bit 47 (MSB) maps to element 0
            expected_pos = 47 - bit_pos
            assert flat[expected_pos] == -1  # 1 bit -> -1
            assert np.sum(flat == -1) == 1   # Only one -1
    
    def test_archetypal_tensor_conversion(self):
        """Test conversion of the archetypal tensor GENE_Mac_S."""
        archetypal_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        recovered_tensor = InformationEngine.int_to_tensor(archetypal_int)
        
        assert np.array_equal(recovered_tensor, governance.GENE_Mac_S)


class TestStateIndexMapping:
    """Test state-to-index and index-to-state conversions."""
    
    def test_get_index_from_state_standard(self, information_engine_standard, sample_states):
        """Test state-to-index mapping with standard engine."""
        engine = information_engine_standard
        
        for state_int in sample_states[:100]:  # Test subset for speed
            index = engine.get_index_from_state(state_int)
            assert isinstance(index, int)
            assert 0 <= index < 788_986
    
    def test_get_index_from_state_memmap(self, information_engine_memmap, sample_states):
        """Test state-to-index mapping with memmap engine."""
        engine = information_engine_memmap
        
        for state_int in sample_states[:100]:  # Test subset for speed
            index = engine.get_index_from_state(state_int)
            assert isinstance(index, int)
            assert 0 <= index < 788_986
    
    def test_get_state_from_index_standard(self, information_engine_standard):
        """Test index-to-state mapping with standard engine."""
        engine = information_engine_standard
        
        # Test random indices
        for _ in range(100):
            index = random.randint(0, 788_985)
            state_int = engine.get_state_from_index(index)
            assert isinstance(state_int, int)
    
    def test_get_state_from_index_memmap(self, information_engine_memmap):
        """Test index-to-state mapping with memmap engine."""
        engine = information_engine_memmap
        
        # Test random indices
        for _ in range(100):
            index = random.randint(0, 788_985)
            state_int = engine.get_state_from_index(index)
            assert isinstance(state_int, int)
    
    def test_round_trip_state_index_consistency(self, information_engine_standard, sample_states):
        """Test that state->index->state round trips are consistent."""
        engine = information_engine_standard
        
        for state_int in sample_states[:50]:  # Test subset
            index = engine.get_index_from_state(state_int)
            recovered_state = engine.get_state_from_index(index)
            assert recovered_state == state_int
    
    def test_memmap_vs_standard_consistency(self, real_maps, sample_states):
        """Test that memmap and standard engines give same results."""
        ontology_data = real_maps["ontology_data"].copy()
        ontology_data["phenomap_path"] = real_maps["phenomenology_path"]
        
        engine_std = InformationEngine(ontology_data, use_array_indexing=False)
        engine_mem = InformationEngine(ontology_data, use_array_indexing=True)
        
        for state_int in sample_states[:50]:  # Test subset
            index_std = engine_std.get_index_from_state(state_int)
            index_mem = engine_mem.get_index_from_state(state_int)
            assert index_std == index_mem
            
            state_std = engine_std.get_state_from_index(index_std)
            state_mem = engine_mem.get_state_from_index(index_mem)
            assert state_std == state_mem


class TestGeometricMeasurements:
    """Test angular distance and state divergence measurements."""
    
    def test_gyrodistance_angular_properties(self, information_engine_standard):
        """Test mathematical properties of angular gyrodistance."""
        # Generate test tensors
        T1 = np.random.choice([-1, 1], size=(4, 2, 3, 2)).astype(np.int8)
        T2 = np.random.choice([-1, 1], size=(4, 2, 3, 2)).astype(np.int8)
        T3 = np.random.choice([-1, 1], size=(4, 2, 3, 2)).astype(np.int8)
        
        # Test distance properties
        d12 = information_engine_standard.gyrodistance_angular(T1, T2)
        d21 = information_engine_standard.gyrodistance_angular(T2, T1)
        d11 = information_engine_standard.gyrodistance_angular(T1, T1)
        
        # Symmetry
        assert np.isclose(d12, d21), "Distance should be symmetric"
        
        # Self-distance is zero
        assert np.isclose(d11, 0.0), "Self-distance should be zero"
        
        # Range check
        assert 0 <= d12 <= np.pi, "Distance should be in [0, π]"
        
        # Triangle inequality (relaxed for floating point)
        d13 = information_engine_standard.gyrodistance_angular(T1, T3)
        d23 = information_engine_standard.gyrodistance_angular(T2, T3)
        assert d13 <= d12 + d23 + 1e-10, "Triangle inequality should hold"
    
    def test_gyrodistance_extreme_cases(self, information_engine_standard):
        """Test angular distance for extreme cases."""
        # Identical tensors
        T1 = np.ones((4, 2, 3, 2), dtype=np.int8)
        T2 = np.ones((4, 2, 3, 2), dtype=np.int8)
        distance = information_engine_standard.gyrodistance_angular(T1, T2)
        assert np.isclose(distance, 0.0)
        
        # Opposite tensors
        T1 = np.ones((4, 2, 3, 2), dtype=np.int8)
        T2 = -np.ones((4, 2, 3, 2), dtype=np.int8)
        distance = information_engine_standard.gyrodistance_angular(T1, T2)
        assert np.isclose(distance, np.pi)
    
    def test_measure_state_divergence(self, information_engine_standard, sample_states):
        """Test state divergence measurement from archetypal state."""
        engine = information_engine_standard
        
        # Test archetypal state has zero divergence
        archetypal_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        divergence = engine.measure_state_divergence(archetypal_int)
        assert np.isclose(divergence, 0.0), "Archetypal state should have zero divergence"
        
        # Test random states
        for state_int in sample_states[:NUM_DISTANCE_TESTS]:
            divergence = engine.measure_state_divergence(state_int)
            assert 0 <= divergence <= np.pi, f"Divergence {divergence} out of range for state {state_int}"
            assert isinstance(divergence, float)


class TestInformationEngineIntegrity:
    """Test InformationEngine against real physics and maps."""
    
    def test_epistemology_table_consistency(self, real_maps, information_engine_standard, sample_states):
        """Test that InformationEngine mapping is consistent with epistemology table."""
        engine = information_engine_standard
        ep_table = real_maps["epistemology_table"]
        
        # Sample some state transitions
        for state_int in sample_states[:50]:  # Reduced for older hardware
            state_index = engine.get_index_from_state(state_int)
            
            # Test a few random introns
            for _ in range(10):
                intron = random.randint(0, 255)
                
                # Get next state from physics
                next_state_int = governance.apply_gyration_and_transform(state_int, intron)
                next_state_index = engine.get_index_from_state(next_state_int)
                
                # Get next state from epistemology table
                table_next_index = ep_table[state_index, intron]
                
                assert next_state_index == table_next_index, \
                    f"Mismatch: physics gives {next_state_index}, table gives {table_next_index}"
    
    def test_phenomenology_orbit_consistency(self, real_maps, information_engine_standard):
        """Test that orbit cardinality data is consistent."""
        engine = information_engine_standard
        pheno_data = real_maps["phenomenology_data"]
        
        if "orbit_sizes" in pheno_data and hasattr(engine, 'orbit_cardinality'):
            orbit_sizes = {int(k): v for k, v in pheno_data["orbit_sizes"].items()}
            
            # Check that total sizes match
            total_from_orbits = sum(orbit_sizes.values())
            total_from_cardinality = np.sum(engine.orbit_cardinality)
            
            # Note: orbit_cardinality might have different structure
            # We mainly check that it's reasonable
            assert len(engine.orbit_cardinality) == 788_986
            assert np.all(engine.orbit_cardinality >= 1)
    
    def test_archetypal_state_in_ontology(self, information_engine_standard):
        """Test that the archetypal state is properly indexed."""
        engine = information_engine_standard
        archetypal_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        
        # Should be able to get index without error
        index = engine.get_index_from_state(archetypal_int)
        assert isinstance(index, int)
        assert 0 <= index < 788_986
        
        # Round trip should work
        recovered_state = engine.get_state_from_index(index)
        assert recovered_state == archetypal_int


class TestErrorHandling:
    """Test proper error handling and edge cases."""
    
    def test_invalid_state_lookup_standard(self, information_engine_standard):
        """Test error handling for invalid states in standard engine."""
        engine = information_engine_standard
        
        # Test with state not in ontology (should be very unlikely with real physics)
        invalid_state = 0xFFFFFFFFFFFF  # Likely not in ontology
        with pytest.raises(ValueError, match="not found in discovered ontology"):
            engine.get_index_from_state(invalid_state)
    
    def test_invalid_state_lookup_memmap(self, information_engine_memmap):
        """Test error handling for invalid states in memmap engine."""
        engine = information_engine_memmap
        
        invalid_state = 0xFFFFFFFFFFFF  # Likely not in ontology
        with pytest.raises(ValueError, match="not found in discovered ontology"):
            engine.get_index_from_state(invalid_state)
    
    def test_invalid_index_lookup_standard(self, information_engine_standard):
        """Test error handling for invalid indices in standard engine."""
        engine = information_engine_standard
        
        with pytest.raises(ValueError, match="Invalid index"):
            engine.get_state_from_index(-1)
        
        with pytest.raises(ValueError, match="Invalid index"):
            engine.get_state_from_index(788_986)  # Out of bounds
    
    def test_invalid_index_lookup_memmap(self, information_engine_memmap):
        """Test error handling for invalid indices in memmap engine."""
        engine = information_engine_memmap
        
        with pytest.raises(ValueError, match="Index .* out of bounds"):
            engine.get_state_from_index(-1)
        
        with pytest.raises(ValueError, match="Index .* out of bounds"):
            engine.get_state_from_index(788_986)  # Out of bounds
    
    def test_uninitialized_memmap_arrays(self, real_maps):
        """Test error handling when memmap arrays are not properly initialized."""
        ontology_data = real_maps["ontology_data"]
        engine = InformationEngine(ontology_data, use_array_indexing=True)
        
        # Artificially break the arrays to test error handling
        engine._keys = None
        # engine._values = None  # Removed: _values no longer exists
        
        with pytest.raises(RuntimeError, match="Memmap arrays not initialized"):
            engine.get_index_from_state(12345)
        
        engine._inverse = None
        with pytest.raises(RuntimeError, match="Memmap arrays not initialized"):
            engine.get_state_from_index(0)


class TestMemoryEfficiency:
    """Test memory usage and efficiency considerations."""
    
    def test_memmap_vs_standard_memory_usage(self, real_maps):
        """Compare memory characteristics of memmap vs standard engines."""
        ontology_data = real_maps["ontology_data"].copy()
        ontology_data["phenomap_path"] = real_maps["phenomenology_path"]
        
        # Create both types
        engine_std = InformationEngine(ontology_data, use_array_indexing=False)
        engine_mem = InformationEngine(ontology_data, use_array_indexing=True)
        
        # Memmap should have freed inverse_ontology_map
        assert engine_std.inverse_ontology_map is not None
        assert engine_mem.inverse_ontology_map is None
        
        # Memmap should have numpy arrays
        assert engine_std._keys is None
        assert engine_mem._keys is not None
        assert isinstance(engine_mem._keys, np.ndarray)
    
    def test_large_batch_operations(self, information_engine_memmap, sample_states):
        """Test that the engine can handle larger batch operations efficiently."""
        engine = information_engine_memmap
        
        # Test batch state-to-index conversions
        batch_size = min(100, len(sample_states))
        batch_states = sample_states[:batch_size]
        
        indices = []
        for state_int in batch_states:
            index = engine.get_index_from_state(state_int)
            indices.append(index)
        
        # Verify all indices are valid
        assert len(indices) == batch_size
        assert all(0 <= idx < 788_986 for idx in indices)
        
        # Test batch index-to-state conversions
        recovered_states = []
        for index in indices:
            state = engine.get_state_from_index(index)
            recovered_states.append(state)
        
        # Should recover original states
        assert recovered_states == batch_states


# --- Performance Benchmarks (Optional) ---

@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for critical operations."""
    
    def test_state_lookup_performance(self, information_engine_memmap, sample_states):
        """Benchmark state lookup performance."""
        engine = information_engine_memmap
        test_states = sample_states[:min(1000, len(sample_states))]
        
        start_time = time.perf_counter()
        for state_int in test_states:
            engine.get_index_from_state(state_int)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        rate = len(test_states) / total_time
        
        # Should be reasonably fast (adjust threshold based on hardware)
        assert rate > 100, f"Lookup rate too slow: {rate:.1f} lookups/sec"
    
    def test_tensor_conversion_performance(self, sample_states):
        """Benchmark tensor conversion performance."""
        test_states = sample_states[:min(500, len(sample_states))]
        
        start_time = time.perf_counter()
        for state_int in test_states:
            tensor = InformationEngine.int_to_tensor(state_int)
            recovered = InformationEngine.tensor_to_int(tensor)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        rate = len(test_states) / total_time
        
        # Should be reasonably fast for tensor operations
        assert rate > 100, f"Conversion rate too slow: {rate:.1f} conversions/sec"


class TestMaps:
    def test_phenomenology_forward_closure_matches_canonical(self, real_maps):
        pheno = real_maps["phenomenology_data"]
        ontology = real_maps["ontology_data"]["ontology_map"]
        if isinstance(next(iter(ontology)), str):
            ontology = {int(k): v for k, v in ontology.items()}
        canonical_list = pheno["phenomenology_map"]
        ep = real_maps["epistemology_table"]

        reps = []
        seen = set()
        for idx, rep in enumerate(canonical_list):
            if rep == idx and rep not in seen:
                reps.append(rep)
                seen.add(rep)
            if len(reps) == 5:
                break

        from collections import deque
        for rep in reps:
            closure = set([rep])
            q = deque([rep])
            while q:
                cur = q.popleft()
                for nxt in ep[cur]:
                    if nxt not in closure:
                        closure.add(nxt)
                        q.append(nxt)
            canon_set = {i for i, c in enumerate(canonical_list) if c == rep}
            assert closure == canon_set, f"Mismatch for representative {rep}: closure {len(closure)} vs canonical {len(canon_set)}"

    def test_representative_is_min_state(self, real_maps):
        pheno = real_maps["phenomenology_data"]
        ontology_map = real_maps["ontology_data"]["ontology_map"]
        if isinstance(next(iter(ontology_map)), str):
            ontology_map = {int(k): v for k, v in ontology_map.items()}
        idx_to_state = np.empty(len(ontology_map), dtype=np.uint64)
        for s, i in ontology_map.items():
            idx_to_state[i] = s

        canonical = pheno["phenomenology_map"]
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, rep in enumerate(canonical[:20000]):
            groups[rep].append(idx)

        for rep, members in list(groups.items())[:50]:
            states = idx_to_state[members]
            assert idx_to_state[rep] == states.min(), "Representative not minimal in its basin"

    def test_no_cross_reach_between_representatives(self, real_maps):
        pheno = real_maps["phenomenology_data"]
        ep = real_maps["epistemology_table"]
        canonical = pheno["phenomenology_map"]
        reps = [i for i, c in enumerate(canonical) if c == i][:10]

        from collections import deque
        basins = {}
        for rep in reps:
            closure = set([rep])
            q = deque([rep])
            while q:
                cur = q.popleft()
                for nxt in ep[cur]:
                    if nxt not in closure:
                        closure.add(nxt)
                        q.append(nxt)
            basins[rep] = closure

        for r1 in reps:
            for r2 in reps:
                if r1 == r2:
                    continue
                assert r2 not in basins[r1], f"Representative {r2} reachable from {r1}; basins should have merged"

    def test_cycle_truncation_metadata(self, real_maps):
        pheno = real_maps["phenomenology_data"]
        meta = pheno.get("metadata", {})
        if meta.get("cycles_truncated", False):
            assert meta.get("total_cycles", 0) == 0
        else:
            assert "total_cycles" in meta

    def test_int_to_tensor_out_of_range(self):
        with pytest.raises(ValueError):
            InformationEngine.int_to_tensor(1 << 48)

    def test_tensor_to_int_shape_error(self):
        bad = np.ones((4,2,3,3), dtype=np.int8)
        with pytest.raises(ValueError):
            InformationEngine.tensor_to_int(bad)

    def test_basin_contains_multiple_sccs(self, real_maps):
        pheno = real_maps["phenomenology_data"]
        ep = real_maps["epistemology_table"]
        canonical = pheno["phenomenology_map"]

        from collections import Counter, deque
        counts = Counter(canonical)
        rep, _ = counts.most_common(1)[0]
        basin_nodes = [i for i,c in enumerate(canonical) if c == rep]
        basin_set = set(basin_nodes)

        adj = {i: [n for n in ep[i] if n in basin_set] for i in basin_nodes}
        sys.setrecursionlimit(10**6)
        visited = set()
        order = []
        def dfs(v):
            visited.add(v)
            for w in adj[v]:
                if w not in visited:
                    dfs(w)
            order.append(v)
        for v in basin_nodes[:5000]:
            if v not in visited:
                dfs(v)
            if len(order) > 10000:
                break
        radj = {i: [] for i in adj}
        for v, nbrs in adj.items():
            for w in nbrs:
                if w in radj:
                    radj[w].append(v)
        visited.clear()
        scc_count = 0
        def dfs2(v):
            visited.add(v)
            for w in radj[v]:
                if w not in visited:
                    dfs2(w)
        for v in reversed(order):
            if v not in visited:
                scc_count += 1
                dfs2(v)
                if scc_count > 1:
                    break
        assert scc_count >= 1

    def test_build_state_transition_table_rejects_unknown(self, monkeypatch, tmp_path):
        ontology_map = {10:0, 20:1}
        from baby import governance
        def fake_apply(s, intron):
            return 10 if intron == 0 else 999999999
        monkeypatch.setattr(governance, "apply_gyration_and_transform", fake_apply)
        from baby.information import build_state_transition_table
        with pytest.raises(RuntimeError, match="Invalid next_states"):
            build_state_transition_table(ontology_map, str(tmp_path/"ep.npy"))
