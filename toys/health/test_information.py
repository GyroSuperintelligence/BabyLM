"""
Tests for S2: Information - Measurement & Storage
"""

import pytest
import numpy as np
import json
import os
import threading
import time
from pathlib import Path

from baby import governance
from baby.information import (
    InformationEngine,
    PickleStore,
    MultiAgentPhenotypeStore,
    CanonicalizingStore,
    discover_and_save_manifold,
    build_canonical_map,
)


class TestInformationEngine:
    """Test the measurement and conversion engine."""
    
    def test_initialization(self, manifold_data):
        """Test engine initialization with manifold data."""
        manifold_path, mock_manifold = manifold_data
        engine = InformationEngine(mock_manifold)
        
        assert engine.endogenous_modulus == 788_986
        assert engine.manifold_diameter == 6
        assert len(engine.genotype_map) == 1000  # Our mock has 1000 states
    
    def test_state_index_conversion(self, manifold_data):
        """Test conversion between state integers and indices."""
        manifold_path, mock_manifold = manifold_data
        engine = InformationEngine(mock_manifold)
        
        # Test known mapping from mock
        for i in range(10):
            assert engine.get_index_from_state(i) == i
            assert engine.get_state_from_index(i) == i
    
    def test_state_not_in_manifold(self, manifold_data):
        """Test error handling for invalid states."""
        manifold_path, mock_manifold = manifold_data
        engine = InformationEngine(mock_manifold)
        
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
        engine = InformationEngine({"genotype_map": {}, "endogenous_modulus": 788_986, "manifold_diameter": 6})
        
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


class TestPickleStore:
    """Test the file-based storage implementation."""
    
    def test_basic_operations(self, pickle_store):
        """Test basic get/put operations."""
        # Initially empty
        assert pickle_store.get((0, 0)) is None
        
        # Put and retrieve
        entry = {"phenotype": "A", "confidence": 0.8}
        pickle_store.put((0, 0), entry)
        
        retrieved = pickle_store.get((0, 0))
        assert retrieved == entry
        
        # Verify it returns a copy
        retrieved["confidence"] = 0.9
        assert pickle_store.get((0, 0))["confidence"] == 0.8
    
    def test_persistence(self, temp_dir):
        """Test that data persists across store instances."""
        store_path = os.path.join(temp_dir, "test.pkl.gz")
        
        # Create and populate store
        store1 = PickleStore(store_path)
        store1.put((42, 7), {"phenotype": "X", "value": 123})
        store1.close()
        
        # Load in new store
        store2 = PickleStore(store_path)
        assert store2.get((42, 7))["value"] == 123
        store2.close()
    
    def test_atomic_saves(self, temp_dir):
        """Test atomic save operations."""
        store_path = os.path.join(temp_dir, "atomic.pkl.gz")
        store = PickleStore(store_path)
        
        # Add some data
        for i in range(10):
            store.put((i, 0), {"value": i})
        
        # Verify temp files are cleaned up
        temp_files = [f for f in os.listdir(temp_dir) if f.startswith(".gyro_temp_")]
        assert len(temp_files) == 0
        
        store.close()
    
    def test_corruption_handling(self, temp_dir):
        """Test handling of corrupted store files."""
        store_path = os.path.join(temp_dir, "corrupt.pkl.gz")
        
        # Create corrupted file
        with open(store_path, 'wb') as f:
            f.write(b"This is not a valid pickle file!")
        
        # Should handle gracefully
        store = PickleStore(store_path)
        assert len(store.data) == 0
        
        # Should create backup
        backup_files = [f for f in os.listdir(temp_dir) if f.startswith("corrupt.pkl.gz.corrupt.")]
        assert len(backup_files) == 1
        
        store.close()
    
    def test_thread_safety(self, temp_dir):
        """Test thread-safe operations."""
        store_path = os.path.join(temp_dir, "threaded.pkl.gz")
        store = PickleStore(store_path)
        errors = []
        
        def writer_thread(thread_id):
            try:
                for i in range(100):
                    store.put((thread_id, i), {"value": i})
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


class TestMultiAgentStore:
    """Test the multi-agent storage with public/private separation."""
    
    def test_read_through_cache(self, multi_agent_store):
        """Test that private knowledge overrides public."""
        # Check public knowledge is accessible
        assert multi_agent_store.get((0, 0))["phenotype"] == "public"
        
        # Add private knowledge
        multi_agent_store.put((0, 0), {"phenotype": "private", "confidence": 0.95})
        
        # Private should override
        assert multi_agent_store.get((0, 0))["phenotype"] == "private"
    
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
        public_store = PickleStore(public_path)
        public_store.put((0, 0), {"phenotype": "v1"})
        public_store.close()
        
        # Create multi-agent store
        ma_store = MultiAgentPhenotypeStore(public_path, private_path)
        assert ma_store.get((0, 0))["phenotype"] == "v1"
        
        # Update public knowledge externally
        public_store = PickleStore(public_path)
        public_store.put((0, 0), {"phenotype": "v2"})
        public_store.close()
        
        # Reload and verify update
        ma_store.reload_public_knowledge()
        assert ma_store.get((0, 0))["phenotype"] == "v2"
        
        ma_store.close()
    
    def test_private_persistence(self, temp_dir):
        """Test that private deltas persist."""
        public_path = os.path.join(temp_dir, "public.pkl.gz")
        private_path = os.path.join(temp_dir, "private.pkl.gz")
        
        # Create empty public store
        PickleStore(public_path).close()
        
        # Create and populate multi-agent store
        ma_store1 = MultiAgentPhenotypeStore(public_path, private_path)
        ma_store1.put((7, 42), {"phenotype": "private_data"})
        ma_store1.close()
        
        # Load in new instance
        ma_store2 = MultiAgentPhenotypeStore(public_path, private_path)
        assert ma_store2.get((7, 42))["phenotype"] == "private_data"
        ma_store2.close()


class TestCanonicalizingStore:
    """Test the canonicalizing decorator."""
    
    @pytest.fixture
    def canonical_map(self, temp_dir):
        """Create a simple canonical map for testing."""
        # Map: 0->0, 1->0, 2->2, 3->2 (two orbits)
        canonical_map = {0: 0, 1: 0, 2: 2, 3: 2}
        
        map_path = os.path.join(temp_dir, "canonical.json")
        with open(map_path, 'w') as f:
            json.dump(canonical_map, f)
        
        return map_path, canonical_map
    
    def test_canonical_resolution(self, pickle_store, canonical_map):
        """Test that equivalent states map to same storage."""
        map_path, _ = canonical_map
        canon_store = CanonicalizingStore(pickle_store, map_path)
        
        # Store under state 0
        canon_store.put((0, 10), {"phenotype": "A"})
        
        # Retrieve under equivalent state 1
        assert canon_store.get((1, 10))["phenotype"] == "A"
        
        # Both should resolve to same entry
        assert canon_store.get((0, 10)) == canon_store.get((1, 10))
    
    def test_original_context_preserved(self, pickle_store, canonical_map):
        """Test that original context is preserved in entries."""
        map_path, _ = canonical_map
        canon_store = CanonicalizingStore(pickle_store, map_path)
        
        # Store under non-canonical state
        canon_store.put((1, 10), {"phenotype": "B"})
        
        # Check that context signature preserves original
        entry = canon_store.get((1, 10))
        assert entry["context_signature"] == (1, 10)


class TestManifoldDiscovery:
    """Test manifold discovery and canonical map building."""
    
    @pytest.mark.slow
    def test_discover_manifold(self, temp_dir):
        """Test full manifold discovery (expensive - marked slow)."""
        manifold_path = os.path.join(temp_dir, "manifold.json")
        
        # This is the real discovery - will take some time
        discover_and_save_manifold(manifold_path)
        
        # Verify results
        with open(manifold_path, 'r') as f:
            manifold = json.load(f)
        
        assert manifold["endogenous_modulus"] == 788_986
        assert manifold["manifold_diameter"] == 6
        assert len(manifold["genotype_map"]) == 788_986
    
    @pytest.mark.slow
    def test_build_canonical_map(self, real_manifold, temp_dir):
        """Test canonical map building (expensive - marked slow)."""
        manifold_path, canonical_path, _ = real_manifold
        
        # Verify canonical map was built
        assert os.path.exists(canonical_path)
        
        with open(canonical_path, 'r') as f:
            canonical_map = json.load(f)
        
        # Every state should have a canonical mapping
        assert len(canonical_map) == 788_986
        
        # All canonical indices should be valid
        for idx, canonical_idx in canonical_map.items():
            assert 0 <= int(canonical_idx) < 788_986


class TestStorageIntegration:
    """Integration tests for storage components."""
    
    def test_pickle_to_canonical_upgrade(self, temp_dir):
        """Test upgrading from simple pickle to canonical storage."""
        # Start with simple pickle store
        store_path = os.path.join(temp_dir, "store.pkl.gz")
        pickle_store = PickleStore(store_path)
        
        # Add some entries
        for i in range(10):
            pickle_store.put((i, 0), {"phenotype": f"Entry{i}"})
        
        # Create canonical map (identity map for simplicity)
        canonical_map = {i: i for i in range(10)}
        map_path = os.path.join(temp_dir, "canonical.json")
        with open(map_path, 'w') as f:
            json.dump(canonical_map, f)
        
        # Upgrade to canonical store
        canon_store = CanonicalizingStore(pickle_store, map_path)
        
        # Verify all entries still accessible
        for i in range(10):
            assert canon_store.get((i, 0))["phenotype"] == f"Entry{i}"
        
        canon_store.close()