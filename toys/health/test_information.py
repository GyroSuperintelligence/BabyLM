"""
Comprehensive tests for information.py - the measurement and storage coordination layer.
Tests state representations, distance calculations, and discovery operations.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest

from baby import governance
from baby.information import (
    InformationEngine,
    ProgressReporter,
    discover_and_save_ontology,
    build_state_transition_table,
    build_phenomenology_map,
    open_memmap_int32,
)


class TestInformationEngine:
    """Test the core InformationEngine functionality."""

    def test_initialization_with_real_ontology(self, meta_paths: Dict[str, str]) -> None:
        """Test engine initializes correctly with real ontology data."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data)
        
        assert engine.endogenous_modulus == 788_986
        assert engine.ontology_diameter == 6
        assert engine.orbit_cardinality is not None
        assert len(engine.orbit_cardinality) == 788_986

    def test_initialization_dict_mode(self, meta_paths: Dict[str, str]) -> None:
        """Test engine in dictionary indexing mode."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        assert not engine.use_array_indexing
        assert engine.ontology_map is not None
        assert engine.inverse_ontology_map is not None
        assert engine._keys is None
        assert engine._inverse is None

    def test_initialization_array_mode(self, meta_paths: Dict[str, str]) -> None:
        """Test engine in array indexing mode."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=True)
        
        assert engine.use_array_indexing
        assert engine.ontology_map is None  # Freed in array mode
        assert engine.inverse_ontology_map is None
        assert engine._keys is not None
        assert engine._inverse is not None

    def test_strict_validation_enabled(self, meta_paths: Dict[str, str]) -> None:
        """Test strict validation catches incorrect constants."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        # Modify constants to invalid values
        bad_data = ontology_data.copy()
        bad_data["endogenous_modulus"] = 123456
        
        with pytest.raises(ValueError, match="Expected endogenous modulus"):
            InformationEngine(bad_data, strict_validation=True)

class TestTensorConversion:
    """Test static methods for tensor ↔ integer conversion."""

    def test_tensor_to_int_archetypal_state(self) -> None:
        """Test conversion of archetypal tensor to integer."""
        tensor = governance.GENE_Mac_S
        state_int = InformationEngine.tensor_to_int(tensor)
        
        assert isinstance(state_int, int)
        assert 0 <= state_int < (1 << 48)

    def test_int_to_tensor_archetypal_state(self) -> None:
        """Test conversion of archetypal integer to tensor."""
        state_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        reconstructed = InformationEngine.int_to_tensor(state_int)
        
        assert reconstructed.shape == (4, 2, 3, 2)
        assert reconstructed.dtype == np.int8
        assert np.array_equal(reconstructed, governance.GENE_Mac_S)

    def test_conversion_roundtrip(self) -> None:
        """Test tensor → int → tensor roundtrip preserves data."""
        original = governance.GENE_Mac_S
        state_int = InformationEngine.tensor_to_int(original)
        reconstructed = InformationEngine.int_to_tensor(state_int)
        
        assert np.array_equal(original, reconstructed)

    def test_int_to_tensor_boundary_values(self) -> None:
        """Test conversion of boundary integer values."""
        # Test zero state
        zero_tensor = InformationEngine.int_to_tensor(0)
        assert np.all(zero_tensor == 1)  # All +1 values
        
        # Test maximum 48-bit value
        max_48bit = (1 << 48) - 1
        max_tensor = InformationEngine.int_to_tensor(max_48bit)
        assert np.all(max_tensor == -1)  # All -1 values

    def test_tensor_to_int_boundary_values(self) -> None:
        """Test conversion of boundary tensor values."""
        # All +1 tensor
        all_plus = np.ones((4, 2, 3, 2), dtype=np.int8)
        assert InformationEngine.tensor_to_int(all_plus) == 0
        
        # All -1 tensor  
        all_minus = np.full((4, 2, 3, 2), -1, dtype=np.int8)
        assert InformationEngine.tensor_to_int(all_minus) == (1 << 48) - 1

    def test_tensor_conversion_errors(self) -> None:
        """Test conversion error handling."""
        # Wrong shape
        wrong_shape = np.ones((3, 2, 3, 2), dtype=np.int8)
        with pytest.raises(ValueError, match="Expected tensor shape"):
            InformationEngine.tensor_to_int(wrong_shape)
        
        # Out of bounds integer
        with pytest.raises(ValueError, match="out of bounds"):
            InformationEngine.int_to_tensor(1 << 48)
        
        with pytest.raises(ValueError, match="out of bounds"):
            InformationEngine.int_to_tensor(-1)

    def test_bit_encoding_consistency(self) -> None:
        """Test that bit encoding follows specification."""
        # Create simple test pattern
        test_tensor = np.ones((4, 2, 3, 2), dtype=np.int8)
        test_tensor[0, 0, 0, 0] = -1  # Set MSB
        
        state_int = InformationEngine.tensor_to_int(test_tensor)
        
        # MSB should be set in the integer
        assert state_int & (1 << 47) != 0


class TestStateIndexing:
    """Test state index lookup operations."""

    def test_get_index_dict_mode(self, meta_paths: Dict[str, str]) -> None:
        """Test state index lookup in dictionary mode."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        # Test with known state (archetypal)
        origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        index = engine.get_index_from_state(origin_int)
        
        assert isinstance(index, int)
        assert 0 <= index < engine.endogenous_modulus

    def test_get_index_array_mode(self, meta_paths: Dict[str, str]) -> None:
        """Test state index lookup in array mode."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=True)
        
        origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        index = engine.get_index_from_state(origin_int)
        
        assert isinstance(index, int)
        assert 0 <= index < engine.endogenous_modulus

    def test_get_state_from_index_dict_mode(self, meta_paths: Dict[str, str]) -> None:
        """Test state retrieval from index in dictionary mode."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        # Test roundtrip: state → index → state
        origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        index = engine.get_index_from_state(origin_int)
        recovered_state = engine.get_state_from_index(index)
        
        assert recovered_state == origin_int

    def test_get_state_from_index_array_mode(self, meta_paths: Dict[str, str]) -> None:
        """Test state retrieval from index in array mode."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=True)
        
        origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        index = engine.get_index_from_state(origin_int)
        recovered_state = engine.get_state_from_index(index)
        
        assert recovered_state == origin_int

    def test_invalid_state_lookup(self, meta_paths: Dict[str, str]) -> None:
        """Test error handling for invalid state lookup."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        # Use a state that's not in the ontology
        invalid_state = 999999999
        
        with pytest.raises(ValueError, match="not found in discovered ontology"):
            engine.get_index_from_state(invalid_state)

    def test_invalid_index_lookup(self, meta_paths: Dict[str, str]) -> None:
        """Test error handling for invalid index lookup."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data, use_array_indexing=False)
        
        # Use invalid indices
        with pytest.raises(ValueError, match="Invalid index"):
            engine.get_state_from_index(-1)
        
        with pytest.raises(ValueError, match="Invalid index"):
            engine.get_state_from_index(engine.endogenous_modulus)


class TestDistanceMeasurement:
    """Test gyrodistance and divergence calculations."""

    def test_gyrodistance_angular_identical_tensors(self, meta_paths: Dict[str, str]) -> None:
        """Test angular distance between identical tensors is zero."""
        tensor = governance.GENE_Mac_S
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        engine = InformationEngine(ontology_data)
        distance = engine.gyrodistance_angular(tensor, tensor)
        assert abs(distance) < 1e-10  # Should be essentially zero

    def test_gyrodistance_angular_opposite_tensors(self, meta_paths: Dict[str, str]) -> None:
        """Test angular distance between opposite tensors is π."""
        tensor1 = governance.GENE_Mac_S
        tensor2 = -tensor1  # All signs flipped
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        engine = InformationEngine(ontology_data)
        distance = engine.gyrodistance_angular(tensor1, tensor2)
        assert abs(distance - np.pi) < 1e-10

    def test_gyrodistance_angular_range(self, meta_paths: Dict[str, str]) -> None:
        """Test angular distance is always in [0, π]."""
        tensor1 = governance.GENE_Mac_S
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        engine = InformationEngine(ontology_data)
        # Test with some transformed states
        for intron in [0, 42, 128, 255]:
            state_int = InformationEngine.tensor_to_int(tensor1)
            state_int = governance.apply_gyration_and_transform(state_int, intron)
            tensor2 = InformationEngine.int_to_tensor(state_int)
            distance = engine.gyrodistance_angular(tensor1, tensor2)
            assert 0 <= distance <= np.pi

    def test_measure_state_divergence_with_theta_table(self, meta_paths: Dict[str, str]) -> None:
        """Test state divergence measurement using theta table."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        # Ensure theta table exists
        theta_path = Path(meta_paths["ontology"]).parent / "theta.npy"
        if theta_path.exists():
            engine = InformationEngine(ontology_data)
            
            origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
            divergence = engine.measure_state_divergence(origin_int)
            
            assert isinstance(divergence, float)
            assert 0 <= divergence <= np.pi

class TestOrbitHandling:
    """Test orbit cardinality and phenomenology integration."""

    def test_orbit_cardinality_initialization(self, meta_paths: Dict[str, str]) -> None:
        """Test orbit cardinality loads correctly."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data)
        
        assert len(engine.orbit_cardinality) == engine.endogenous_modulus
        assert engine.orbit_cardinality.dtype == np.uint32
        assert np.all(engine.orbit_cardinality >= 1)  # All orbits have at least 1 member

    def test_get_orbit_cardinality(self, meta_paths: Dict[str, str]) -> None:
        """Test individual orbit cardinality lookup."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data)
        
        # Test valid indices
        cardinality = engine.get_orbit_cardinality(0)
        assert isinstance(cardinality, int)
        assert cardinality >= 1

    def test_phenomenology_integration(self, meta_paths: Dict[str, str]) -> None:
        """Test phenomenology map integration if available."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        # Check if phenomenology map exists
        pheno_path = Path(meta_paths["ontology"]).parent / "phenomenology_map.json"
        if pheno_path.exists():
            engine = InformationEngine(ontology_data)
            
            # Orbit cardinalities should be loaded from phenomenology data
            unique_cardinalities = np.unique(engine.orbit_cardinality)
            assert len(unique_cardinalities) > 1  # Should have varied orbit sizes


class TestProgressReporter:
    """Test the progress reporting utility."""

    def test_progress_reporter_basic(self) -> None:
        """Test basic progress reporter functionality."""
        reporter = ProgressReporter("Test Operation")
        
        assert reporter.desc == "Test Operation"
        assert hasattr(reporter, 'start_time')
        assert hasattr(reporter, 'last_update')

    def test_progress_reporter_update(self, capsys) -> None:
        """Test progress updates are printed."""
        reporter = ProgressReporter("Test")
        
        reporter.update(50, 100)
        captured = capsys.readouterr()
        
        assert "Test:" in captured.out
        assert "50/100" in captured.out
        assert "%" in captured.out

    def test_progress_reporter_done(self, capsys) -> None:
        """Test completion message."""
        reporter = ProgressReporter("Test")
        
        reporter.done()
        captured = capsys.readouterr()
        
        assert "Test: Done" in captured.out

    def test_progress_reporter_with_extra(self, capsys) -> None:
        """Test progress with extra information."""
        reporter = ProgressReporter("Test")
        
        reporter.update(25, 100, extra="processing data")
        captured = capsys.readouterr()
        
        assert "processing data" in captured.out

class TestMemoryMapping:
    """Test memory mapping utilities."""

    def test_open_memmap_int32(self, temp_dir: Path) -> None:
        """Test memory-mapped array creation."""
        filename = str(temp_dir / "test_mmap.npy")
        shape = (100, 256)
        
        # Create memory-mapped array
        arr = open_memmap_int32(filename, 'w+', shape)
        
        assert arr.shape == shape
        assert arr.dtype == np.int32
        
        # Test write/read
        arr[0, 0] = 42
        assert arr[0, 0] == 42
        
        # Clean up
        del arr


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_string_keys_in_ontology_map(self, meta_paths: Dict[str, str]) -> None:
        """Test handling of string keys in ontology map using real ontology."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        engine = InformationEngine(ontology_data)
        # Check that all keys are integers
        if not engine.use_array_indexing:
            assert all(isinstance(k, int) for k in engine.ontology_map.keys())

    def test_array_indexing_mode_assumptions(self, meta_paths: Dict[str, str]) -> None:
        """Test array indexing mode assumptions about ordering."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        # Test that array mode assumes sorted order mapping
        engine = InformationEngine(ontology_data, use_array_indexing=True)
        
        if engine._keys is not None:
            # Keys should be sorted
            assert np.all(engine._keys[:-1] <= engine._keys[1:])

    def test_large_state_conversion_boundary(self) -> None:
        """Test tensor conversion at 48-bit boundary."""
        # Test largest valid 48-bit value
        max_valid = (1 << 48) - 1
        tensor = InformationEngine.int_to_tensor(max_valid)
        recovered = InformationEngine.tensor_to_int(tensor)
        assert recovered == max_valid
        
        # Test just over the boundary
        over_boundary = 1 << 48
        with pytest.raises(ValueError):
            InformationEngine.int_to_tensor(over_boundary)

    def test_gyrodistance_edge_cases(self, meta_paths: Dict[str, str]) -> None:
        """Test gyrodistance calculation edge cases using real ontology."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        engine = InformationEngine(ontology_data)
        # Test with identical tensors
        t1 = governance.GENE_Mac_S
        distance = engine.gyrodistance_angular(t1, t1)
        assert abs(distance) < 1e-10
        # Test with maximally different tensors
        t2 = -t1
        distance = engine.gyrodistance_angular(t1, t2)
        assert abs(distance - np.pi) < 1e-10
        # Test numerical stability with extreme values
        extreme_tensor = np.full((4, 2, 3, 2), 1, dtype=np.int8)
        distance = engine.gyrodistance_angular(extreme_tensor, extreme_tensor)
        assert abs(distance) < 1e-10


class TestDataConsistency:
    """Test data consistency and integrity."""

    def test_ontology_map_consistency(self, meta_paths: Dict[str, str]) -> None:
        """Test ontology map has expected properties."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        ont_map = ontology_data["ontology_map"]
        
        # Should have expected number of states
        assert len(ont_map) == 788_986
        
        # Values should be 0 to N-1
        values = list(ont_map.values())
        assert min(values) == 0
        assert max(values) == 788_985
        assert len(set(values)) == len(values)  # All unique

    def test_state_index_bidirectional_consistency(self, meta_paths: Dict[str, str]) -> None:
        """Test state ↔ index conversion is bijective."""
        with open(meta_paths["ontology"]) as f:
            ontology_data = json.load(f)
        
        engine = InformationEngine(ontology_data)
        
        # Test a sample of states for bidirectional consistency
        origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        
        # Forward: state → index → state
        index = engine.get_index_from_state(origin_int)
        recovered_state = engine.get_state_from_index(index)
        assert recovered_state == origin_int
        
        # Backward: index → state → index  
        test_state = engine.get_state_from_index(100)  # Arbitrary valid index
        recovered_index = engine.get_index_from_state(test_state)
        assert recovered_index == 100