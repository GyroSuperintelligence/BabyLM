"""
Tests for S2: Information - Measurement & Storage
"""

import pytest
import numpy as np
# Try to use ujson for speed, fall back to standard json if unavailable
try:
    import ujson as json  # type: ignore[import]
except ImportError:
    import json  # type: ignore
import os
import threading
import time
from pathlib import Path

from baby import governance
from baby.information import (
    InformationEngine,
    OrbitStore,
)


class TestInformationEngine:
    """Test the measurement and conversion engine."""

    def test_initialization(self, ontology_data):
        """Test engine initialization with ontology data."""
        ontology_path, mock_ontology = ontology_data
        engine = InformationEngine(mock_ontology)

        assert engine.endogenous_modulus == 788_986
        assert engine.ontology_diameter == 6
        assert len(engine.ontology_map) == 1000  # Our mock has 1000 states

    def test_state_index_conversion(self, ontology_data):
        """Test conversion between state integers and indices."""
        ontology_path, mock_ontology = ontology_data
        engine = InformationEngine(mock_ontology)

        # Test known mapping from mock
        for i in range(10):
            assert engine.get_index_from_state(i) == i
            assert engine.get_state_from_index(i) == i

    def test_state_not_in_ontology(self, ontology_data):
        """Test error handling for invalid states."""
        ontology_path, mock_ontology = ontology_data
        engine = InformationEngine(mock_ontology)

        with pytest.raises(ValueError, match="CRITICAL"):
            engine.get_index_from_state(999999)

    def test_tensor_int_conversion(self):
        """Test conversion between tensor and integer representations."""
        # Test with GENE_Mac_S
        tensor = governance.GENE_Mac_S
        state_int = InformationEngine.tensor_to_int(tensor)
        tensor_back = InformationEngine.int_to_tensor(state_int)

        assert np.array_equal(tensor, tensor_back)

    def test_tensor_int_bijection(self):
        """Test that tensor<->int conversion is bijective."""
        # Create various test tensors
        test_tensors = [
            np.ones((4, 2, 3, 2), dtype=np.int8),
            -np.ones((4, 2, 3, 2), dtype=np.int8),
            governance.GENE_Mac_S,
        ]

        # Add a random tensor
        random_tensor = np.random.choice([-1, 1], size=(4, 2, 3, 2)).astype(np.int8)
        test_tensors.append(random_tensor)

        for tensor in test_tensors:
            state_int = InformationEngine.tensor_to_int(tensor)
            tensor_back = InformationEngine.int_to_tensor(state_int)
            assert np.array_equal(tensor, tensor_back)

    def test_gyrodistance_angular(self):
        """Test angular distance measurement."""
        engine = InformationEngine(
            {
                "ontology_map": {},
                "endogenous_modulus": 788_986,
                "ontology_diameter": 6,
                "schema_version": "1.0.0",
                "total_states": 0,
                "build_timestamp": 0.0,
            }
        )

        # Same tensors should have distance 0
        T1 = governance.GENE_Mac_S
        distance = engine.gyrodistance_angular(T1, T1)
        assert abs(distance) < 1e-10

        # Opposite tensors should have distance π
        T2 = -governance.GENE_Mac_S
        distance = engine.gyrodistance_angular(T1, T2)
        assert abs(distance - np.pi) < 1e-10

        # Check triangle inequality
        T3 = np.ones((4, 2, 3, 2), dtype=np.int8)
        d12 = engine.gyrodistance_angular(T1, T2)
        d23 = engine.gyrodistance_angular(T2, T3)
        d13 = engine.gyrodistance_angular(T1, T3)

        # Triangle inequality: d13 <= d12 + d23
        assert d13 <= d12 + d23 + 1e-10


class TestOrbitStore:
    """Test the OrbitStore implementation."""

    def test_basic_operations(self, orbit_store):
        """Test basic get/put operations."""
        # Initially empty
        assert orbit_store.get((0, 0)) is None

        # Put and retrieve
        entry = {"phenotype": "A", "confidence": 0.8}
        orbit_store.put((0, 0), entry)

        retrieved = orbit_store.get((0, 0))
        assert retrieved == entry

        # Verify it returns a copy
        if retrieved is not None:
            retrieved["confidence"] = 0.9
            assert orbit_store.get((0, 0))["confidence"] == 0.8

    def test_persistence(self, temp_dir):
        """Test that data persists across store instances."""
        store_path = os.path.join(temp_dir, "test.pkl.gz")

        # Create and populate store
        store1 = OrbitStore(store_path)
        store1.put((42, 7), {"phenotype": "X", "confidence": 1.0})
        store1.close()

        # Load in new store
        store2 = OrbitStore(store_path)
        entry = store2.get((42, 7))
        assert entry is not None and entry.get("phenotype") == "X"
        store2.close()

    def test_atomic_saves(self, temp_dir):
        """Test atomic save operations."""
        store_path = os.path.join(temp_dir, "atomic.pkl.gz")
        store = OrbitStore(store_path)

        # Add some data
        for i in range(10):
            store.put((i, 0), {"phenotype": f"P{i}", "confidence": 1.0})

        # Verify temp files are cleaned up
        temp_files = [f for f in os.listdir(temp_dir) if f.startswith(".gyro_temp_")]
        assert len(temp_files) == 0

        store.close()

    def test_corruption_handling(self, temp_dir):
        """Test handling of corrupted store files."""
        store_path = os.path.join(temp_dir, "corrupt.pkl.gz")

        # Create corrupted file
        with open(store_path, "wb") as f:
            f.write(b"This is not a valid pickle file!")

        # Should handle gracefully
        store = OrbitStore(store_path)
        assert len(store.data) == 0

        # Should create backup
        backup_files = [f for f in os.listdir(temp_dir) if f.startswith("corrupt.pkl.gz.corrupt.")]
        assert len(backup_files) == 1

        store.close()

    def test_thread_safety(self, temp_dir):
        """Test thread-safe operations."""
        store_path = os.path.join(temp_dir, "threaded.pkl.gz")
        store = OrbitStore(store_path)
        errors = []

        def writer_thread(thread_id):
            try:
                for i in range(100):
                    store.put((thread_id, i), {"phenotype": f"T{thread_id}-{i}", "confidence": 1.0})
            except Exception as e:
                errors.append(e)

        def reader_thread(thread_id):
            try:
                for i in range(100):
                    # Read various keys
                    store.get((thread_id % 5, i))
            except Exception as e:
                errors.append(e)

        # Launch threads
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=writer_thread, args=(i,))
            t2 = threading.Thread(target=reader_thread, args=(i,))
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        store.close()


class TestOrbitStoreOverlay:
    """Test the OrbitStore overlay functionality."""

    def test_read_through_cache(self, multi_agent_store):
        """Test that private knowledge overrides public."""
        # Check public knowledge is accessible
        entry = multi_agent_store.get((0, 0))
        assert entry is not None and entry.get("phenotype") == "public"

        # Add private knowledge
        multi_agent_store.put((0, 0), {"phenotype": "private", "confidence": 0.95})

        # Private should override
        entry = multi_agent_store.get((0, 0))
        assert entry is not None and entry.get("phenotype") == "private"

    def test_public_read_only(self, multi_agent_store):
        """Test that public store remains read-only."""
        # Try to write to public through multi-agent store
        multi_agent_store.put((1, 1), {"phenotype": "new"})

        # Should only be in private
        assert (1, 1) in multi_agent_store.private_deltas

        # Public store should be unchanged
        assert multi_agent_store.public_store.get((1, 1)) is None

    def test_reload_public_knowledge(self, temp_dir):
        """Test reloading public knowledge."""
        public_path = os.path.join(temp_dir, "public.pkl.gz")
        private_path = os.path.join(temp_dir, "private.pkl.gz")

        # Create initial public knowledge
        public_store = OrbitStore(public_path, read_only=True)
        private_store = OrbitStore(private_path)
        ma_store = OrbitStore(private_path, public_store=public_store, private_store=private_store)
        entry = ma_store.get((0, 0))
        assert entry is not None and entry.get("phenotype") == "v1"

        # Update public knowledge externally
        public_store = OrbitStore(public_path, read_only=True)
        private_store = OrbitStore(private_path)
        ma_store.reload_public_knowledge()
        entry = ma_store.get((0, 0))
        assert entry is not None and entry.get("phenotype") == "v2"

        ma_store.close()

    def test_private_persistence(self, temp_dir):
        """Test that private deltas persist."""
        public_path = os.path.join(temp_dir, "public.pkl.gz")
        private_path = os.path.join(temp_dir, "private.pkl.gz")

        # Create empty public store
        OrbitStore(public_path).close()

        # Create and populate multi-agent store
        public_store = OrbitStore(public_path, read_only=True)
        private_store1 = OrbitStore(private_path)
        private_store2 = OrbitStore(private_path)
        ma_store1 = OrbitStore(private_path, public_store=public_store, private_store=private_store1)
        ma_store2 = OrbitStore(private_path, public_store=public_store, private_store=private_store2)
        entry = ma_store2.get((7, 42))
        assert entry is not None and entry.get("phenotype") == "private_data"
        ma_store2.close()


class TestCanonicalOrbitStore:
    """Test canonical storage functionality in OrbitStore."""

    @pytest.fixture
    def phenomenology_map(self, temp_dir):
        """Create a simple canonical map for testing."""
        # Map: 0->0, 1->0, 2->2, 3->2 (two orbits)
        phenomenology_map = {0: 0, 1: 0, 2: 2, 3: 2}

        map_path = os.path.join(temp_dir, "canonical.json")
        with open(map_path, "w") as f:
            json.dump(phenomenology_map, f)

        return map_path, phenomenology_map

    def test_phenomenology_resolution(self, orbit_store, phenomenology_map):
        """Test that equivalent states map to same storage."""
        map_path, _ = phenomenology_map
        canon_store = OrbitStore("temp_canonical.pkl.gz", phenomenology_map=map_path)

        # Store under state 0
        canon_store.put((0, 10), {"phenotype": "A", "confidence": 1.0})

        # Retrieve under equivalent state 1
        entry = canon_store.get((1, 10))
        assert entry is not None and entry.get("phenotype") == "A"

        # Both should resolve to same entry
        assert canon_store.get((0, 10)) == canon_store.get((1, 10))

    def test_original_context_preserved(self, orbit_store, phenomenology_map):
        """Test that original context is preserved in entries."""
        map_path, _ = phenomenology_map
        canon_store = OrbitStore("temp_canonical.pkl.gz", phenomenology_map=map_path)

        # Store under non-canonical state
        canon_store.put((1, 10), {"phenotype": "B", "confidence": 1.0})

        # Check that context signature preserves original
        entry = canon_store.get((1, 10))
        assert entry is not None and entry.get("context_signature") == (1, 10)


class TestStorageIntegration:
    """Integration tests for storage components."""

    def test_pickle_to_phenomenology_upgrade(self, temp_dir):
        """Test upgrading from simple pickle to canonical storage."""
        # Start with simple pickle store
        store_path = os.path.join(temp_dir, "store.pkl.gz")
        orbit_store = OrbitStore(store_path)

        # Add some entries
        for i in range(10):
            orbit_store.put((i, 0), {"phenotype": f"Entry{i}", "confidence": 1.0})

        # Create canonical map (identity map for simplicity)
        phenomenology_map = {i: i for i in range(10)}
        map_path = os.path.join(temp_dir, "canonical.json")
        with open(map_path, "w") as f:
            json.dump(phenomenology_map, f)

        # Upgrade to canonical store
        canon_store = OrbitStore("temp_canonical.pkl.gz", phenomenology_map=map_path)

        # Verify all entries still accessible
        for i in range(10):
            entry = canon_store.get((i, 0))
            assert entry is not None and entry.get("phenotype") == f"Entry{i}"

        canon_store.close()
