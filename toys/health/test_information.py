"""
Tests for S2: Information - Measurement & Storage

This module tests the InformationEngine class and related functions
responsible for measurement, storage coordination, and conversion
between state representations.
"""

import pytest
import numpy as np
import os
import tempfile
import json
from pathlib import Path

# Try to use ujson for speed, fall back to standard json if unavailable
try:
    import ujson as json  # type: ignore[import]
except ImportError:
    import json  # type: ignore

from baby import governance
from baby.information import (
    InformationEngine,
    build_phenomenology_map_fast,
    discover_and_save_ontology,
    build_state_transition_table,
)
from baby.contracts import ManifoldData

import random

class TestInformationEngine:
    """Test the InformationEngine class using the real ontology from the meta folder."""

    @pytest.fixture
    def ontology_data(self, real_ontology):
        ontology_path, _, _ = real_ontology
        with open(ontology_path, "r") as f:
            return json.load(f)

    def test_init_with_dict_ontology(self, ontology_data):
        """Test initialization with dictionary ontology."""
        engine = InformationEngine(ontology_data, use_memmap=False)
        assert engine.endogenous_modulus == 788_986
        assert engine.ontology_diameter == 6
        assert isinstance(engine.ontology_map, dict)
        assert isinstance(engine.inverse_ontology_map, dict)
        assert engine.use_memmap is False
        assert engine._keys is None
        assert engine._values is None
        assert engine._inverse is None

    def test_init_with_memmap_ontology(self, ontology_data):
        """Test initialization with memmap ontology."""
        engine = InformationEngine(ontology_data, use_memmap=True)
        assert engine.endogenous_modulus == 788_986
        assert engine.ontology_diameter == 6
        assert isinstance(engine.ontology_map, dict)
        assert engine.inverse_ontology_map is None  # Should be None with memmap
        assert engine.use_memmap is True
        assert engine._keys is not None
        assert engine._values is not None
        assert engine._inverse is not None
        assert isinstance(engine._keys, np.ndarray)
        assert isinstance(engine._values, np.ndarray)
        assert isinstance(engine._inverse, np.ndarray)

    def test_int_to_tensor_conversion(self):
        """Test conversion from integer to tensor."""
        origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        tensor = InformationEngine.int_to_tensor(origin_int)
        assert tensor.shape == (4, 2, 3, 2)
        assert tensor.dtype == np.int8
        assert np.array_equal(tensor, governance.GENE_Mac_S)
        test_int = 0x123456789ABC
        tensor = InformationEngine.int_to_tensor(test_int)
        assert tensor.shape == (4, 2, 3, 2)
        assert tensor.dtype == np.int8
        assert InformationEngine.tensor_to_int(tensor) == test_int

    def test_tensor_to_int_conversion(self):
        """Test conversion from tensor to integer."""
        int_value = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        test_tensor = np.ones((4, 2, 3, 2), dtype=np.int8)
        test_tensor[0, 0, 0, 0] = -1
        test_int = InformationEngine.tensor_to_int(test_tensor)
        assert np.array_equal(InformationEngine.int_to_tensor(test_int), test_tensor)


class TestOntologyDiscovery:
    """Test the ontology discovery and related functions."""
    
    def test_load_existing_ontology(self, real_ontology):
        """Test loading the existing ontology map."""
        ontology_path, _, _ = real_ontology
        
        # Load the ontology map
        with open(ontology_path, "r") as f:
            ontology_data = json.load(f)
        
        # Verify the structure
        assert "ontology_map" in ontology_data
        assert "endogenous_modulus" in ontology_data
        assert "ontology_diameter" in ontology_data
        assert ontology_data["endogenous_modulus"] == 788_986
        assert ontology_data["ontology_diameter"] == 6
        
        # Create an InformationEngine with the loaded ontology
        engine = InformationEngine(ontology_data)
        
        # Verify the engine works correctly
        origin_int = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        assert engine.get_index_from_state(origin_int) is not None
    
    def test_load_existing_phenomenology(self, real_ontology):
        """Test loading the existing phenomenology map."""
        _, phenomenology_path, _ = real_ontology
        
        # Load the phenomenology map
        with open(phenomenology_path, "r") as f:
            phenomenology = json.load(f)
        
        # Verify it's a non-empty dictionary
        assert isinstance(phenomenology, dict)
        assert len(phenomenology) > 0
        
        # Convert keys to integers (json keys are strings)
        phenomenology = {int(k): int(v) for k, v in phenomenology.items()}
        
        # Verify some basic properties
        # Each state should map to a representative in the same orbit
        for state, representative in phenomenology.items():
            assert representative in phenomenology.values()
    
    def test_load_existing_epistemology(self, real_ontology):
        """Test loading the existing epistemology array."""
        _, _, epistemology_path = real_ontology
        
        # Load the epistemology array
        ep = np.load(epistemology_path)
        
        # Verify the shape and type
        assert ep.ndim == 2
        assert ep.shape[1] == 256  # Should have 256 introns
        assert ep.dtype == np.int32
        
        # Verify basic properties
        # For each state, applying an intron should result in another valid state
        for i in range(min(10, ep.shape[0])):  # Check first 10 states
            for j in range(10):  # Check first 10 introns
                next_state = ep[i, j]
                assert 0 <= next_state < ep.shape[0]


class TestMapConsistency:
    """Cross-map consistency tests for ontology, phenomenology, and epistemology."""

    @pytest.fixture
    def maps(self, real_ontology):
        ontology_path, phenomenology_path, epistemology_path = real_ontology
        with open(ontology_path, "r") as f:
            ontology_data = json.load(f)
        with open(phenomenology_path, "r") as f:
            phenomenology_map = json.load(f)
        ep = np.load(epistemology_path, mmap_mode="r")
        # Convert keys to int for performance
        ontology_map = {int(k): v for k, v in ontology_data["ontology_map"].items()}
        inverse_ontology_map = {v: int(k) for k, v in ontology_data["ontology_map"].items()}
        phenomenology_map = {int(k): int(v) for k, v in phenomenology_map.items()}
        return ontology_map, inverse_ontology_map, phenomenology_map, ep

    def test_cross_map_consistency(self, maps):
        ontology_map, inverse_ontology_map, phenomenology_map, ep = maps
        N = 10
        all_indices = list(inverse_ontology_map.keys())
        random.seed(42)
        sample_indices = random.sample(all_indices, N)
        for idx in sample_indices:
            state_int = inverse_ontology_map[idx]
            # 1. The state's index in the ontology maps to a valid canonical representative in the phenomenology map
            assert idx in phenomenology_map, f"Index {idx} missing in phenomenology map"
            rep_idx = phenomenology_map[idx]
            assert rep_idx in all_indices, f"Phenomenology representative {rep_idx} not a valid ontology index"
            # 2. For a random intron, the epistemology transition from the state lands in a state whose canonical representative is consistent
            intron = random.randint(0, 255)
            next_idx = ep[idx, intron]
            assert 0 <= next_idx < len(all_indices), f"Epistemology transition out of bounds: {next_idx}"
            # The next state's canonical representative should also be valid
            assert next_idx in phenomenology_map, f"Next index {next_idx} missing in phenomenology map"
            next_rep_idx = phenomenology_map[next_idx]
            assert next_rep_idx in all_indices, f"Next state's representative {next_rep_idx} not a valid ontology index"

    def test_phenomenology_canonicalization(self, maps):
        """Test that the phenomenology map's representative is the lex smallest in its orbit."""
        ontology_map, inverse_ontology_map, phenomenology_map, ep = maps
        N = 5
        all_indices = list(inverse_ontology_map.keys())
        random.seed(123)
        sample_indices = random.sample(all_indices, N)
        for idx in sample_indices:
            state_int = inverse_ontology_map[idx]
            # Reconstruct the orbit by applying all 256 introns
            orbit = set()
            for intron in range(256):
                next_int = governance.apply_gyration_and_transform(state_int, intron)
                orbit.add(next_int)
            # The lex smallest state in the orbit
            lex_smallest = min(orbit)
            # The representative index from the phenomenology map
            rep_idx = phenomenology_map[idx]
            rep_state_int = inverse_ontology_map[rep_idx]
            assert rep_state_int == lex_smallest, (
                f"Phenomenology map for idx {idx} (state {state_int}) gives rep idx {rep_idx} (state {rep_state_int}), "
                f"but lex smallest in orbit is {lex_smallest}")

    def test_full_transition_cycles(self, maps):
        """Test that after a sequence of introns, the resulting state is reachable and canonicalizable."""
        ontology_map, inverse_ontology_map, phenomenology_map, ep = maps
        N = 5
        all_indices = list(inverse_ontology_map.keys())
        random.seed(456)
        sample_indices = random.sample(all_indices, N)
        for idx in sample_indices:
            current_idx = idx
            # Apply a random sequence of 10 introns
            introns = [random.randint(0, 255) for _ in range(10)]
            for intron in introns:
                current_idx = ep[current_idx, intron]
                assert 0 <= current_idx < len(all_indices), f"Transitioned to invalid index {current_idx}"
            # Check that the resulting state is in the ontology and has a valid canonical representative
            assert current_idx in phenomenology_map, f"Resulting index {current_idx} missing in phenomenology map"
            rep_idx = phenomenology_map[current_idx]
            rep_state_int = inverse_ontology_map[rep_idx]
            # Reconstruct the orbit for the resulting state
            state_int = inverse_ontology_map[current_idx]
            orbit = set()
            for intron in range(256):
                next_int = governance.apply_gyration_and_transform(state_int, intron)
                orbit.add(next_int)
            lex_smallest = min(orbit)
            assert rep_state_int == lex_smallest, (
                f"After transitions, rep idx {rep_idx} (state {rep_state_int}) does not match lex smallest {lex_smallest}")

    def test_invariant_checks(self, maps):
        """Test invariants: unique orbits in phenomenology, and closure of epistemology map."""
        ontology_map, inverse_ontology_map, phenomenology_map, ep = maps
        all_indices = set(inverse_ontology_map.keys())
        # 1. Number of unique orbits in the phenomenology map
        unique_orbits = set(phenomenology_map.values())
        # The number of unique orbits should be less than or equal to the number of states, and nontrivial
        assert 1 < len(unique_orbits) < len(all_indices), (
            f"Unexpected number of unique orbits: {len(unique_orbits)} (states: {len(all_indices)})")
        # 2. Epistemology closure: all transitions land in valid ontology indices
        n_states, n_introns = ep.shape
        for idx in range(n_states):
            for intron in range(n_introns):
                next_idx = ep[idx, intron]
                assert next_idx in all_indices, (
                    f"Epistemology transition from {idx} with intron {intron} lands at invalid index {next_idx}")