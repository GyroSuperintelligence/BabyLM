"""
Comprehensive tests for contracts.py and policies.py - type definitions and storage policies.
Tests data structures, storage backends, maintenance operations, and policy enforcement.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import pytest

from baby.contracts import (
    AgentConfig,
    CycleHookFunction,
    GovernanceSignature,
    MaintenanceReport,
    ManifoldData,
    PhenotypeEntry,
    PhenomenologyData,
    PreferencesConfig,
    ValidationReport,
)
from baby.policies import (
    CanonicalView,
    OrbitStore,
    OverlayView,
    ReadOnlyView,
    apply_global_confidence_decay,
    export_knowledge_statistics,
    load_phenomenology_map,
    merge_phenotype_maps,
    prune_and_compact_store,
    to_native,
    validate_ontology_integrity,
)


class TestContracts:
    """Test type definitions and contracts from contracts.py."""

    def test_governance_signature_structure(self) -> None:
        """Test GovernanceSignature TypedDict structure."""
        sig: GovernanceSignature = {
            "neutral": 6,
            "li": 0,
            "fg": 0,
            "bg": 0,
            "dyn": 0,
        }
        
        # Should have all required keys
        assert "neutral" in sig
        assert "li" in sig
        assert "fg" in sig
        assert "bg" in sig
        assert "dyn" in sig
        
        # Values should be integers
        assert isinstance(sig["neutral"], int)
        assert isinstance(sig["li"], int)
        assert isinstance(sig["fg"], int)
        assert isinstance(sig["bg"], int)
        assert isinstance(sig["dyn"], int)

    def test_phenotype_entry_structure(self, sample_phenotype: Dict[str, Any]) -> None:
        """Test PhenotypeEntry TypedDict structure."""
        entry: PhenotypeEntry = sample_phenotype
        
        # Required fields
        required_fields = [
            "phenotype", "confidence", "exon_mask", "usage_count",
            "last_updated", "created_at", "governance_signature",
            "context_signature", "_original_context"
        ]
        
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"
        
        # Type validation
        assert isinstance(entry["phenotype"], str)
        assert isinstance(entry["confidence"], float)
        assert isinstance(entry["exon_mask"], int)
        assert isinstance(entry["usage_count"], int)
        assert isinstance(entry["last_updated"], float)
        assert isinstance(entry["created_at"], float)
        assert isinstance(entry["governance_signature"], dict)
        assert isinstance(entry["context_signature"], tuple)
        assert len(entry["context_signature"]) == 2

    def test_manifold_data_structure(self) -> None:
        """Test ManifoldData TypedDict structure."""
        data: ManifoldData = {
            "schema_version": "0.9.6",
            "ontology_map": {0: 0, 1: 1, 2: 2},
            "endogenous_modulus": 788_986,
            "ontology_diameter": 6,
            "total_states": 788_986,
            "build_timestamp": time.time(),
        }
        
        # Required fields
        assert "schema_version" in data
        assert "ontology_map" in data
        assert "endogenous_modulus" in data
        assert "ontology_diameter" in data
        assert "total_states" in data
        assert "build_timestamp" in data
        
        # Type validation
        assert isinstance(data["schema_version"], str)
        assert isinstance(data["ontology_map"], dict)
        assert isinstance(data["endogenous_modulus"], int)
        assert isinstance(data["ontology_diameter"], int)
        assert isinstance(data["total_states"], int)
        assert isinstance(data["build_timestamp"], float)

    def test_phenomenology_data_structure(self) -> None:
        """Test PhenomenologyData TypedDict structure."""
        data: PhenomenologyData = {
            "schema_version": "phenomenology/core/1.0.0",
            "phenomenology_map": [0, 1, 2, 0, 1, 2],
            "orbit_sizes": {0: 3, 1: 2, 2: 1},
            "metadata": {"total_orbits": 3},
            "_diagnostics": {"note": "test"},
        }
        
        # Check field types
        assert isinstance(data["schema_version"], str)
        assert isinstance(data["phenomenology_map"], list)
        assert isinstance(data["orbit_sizes"], dict)
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["_diagnostics"], dict)

    def test_agent_config_structure(self) -> None:
        """Test AgentConfig TypedDict structure."""
        config: AgentConfig = {
            "ontology_path": "/path/to/ontology.json",
            "knowledge_path": "/path/to/knowledge.pkl.gz",
            "public_knowledge_path": "/path/to/public.pkl.gz",
            "private_knowledge_path": "/path/to/private.pkl.gz",
            "agent_metadata": {"role": "assistant"},
            "max_memory_mb": 1024,
            "enable_phenomenology_storage": True,
            "learn_batch_size": 100,
            "phenomenology_map_path": "/path/to/pheno.json",
            "tokenizer_name": "bert-base-uncased",
            "tokenizer_mode": "input",
        }
        
        # Should accept all fields
        assert config["ontology_path"] == "/path/to/ontology.json"
        assert config["enable_phenomenology_storage"] is True
        assert config["learn_batch_size"] == 100

    def test_preferences_config_structure(self) -> None:
        """Test PreferencesConfig TypedDict structure."""
        prefs: PreferencesConfig = {
            "storage_backend": "pickle",
            "compression_level": 6,
            "max_file_size_mb": 100,
            "enable_auto_decay": True,
            "decay_interval_hours": 24.0,
            "decay_factor": 0.999,
            "confidence_threshold": 0.05,
            "max_agents_in_memory": 1000,
            "agent_eviction_policy": "lru",
            "agent_ttl_minutes": 60,
            "enable_profiling": False,
            "write_batch_size": 100,
            "cache_size_mb": 64,
        }
        
        # Type validation
        assert isinstance(prefs["storage_backend"], str)
        assert isinstance(prefs["compression_level"], int)
        assert isinstance(prefs["enable_auto_decay"], bool)
        assert isinstance(prefs["decay_factor"], float)

    def test_validation_report_structure(self) -> None:
        """Test ValidationReport TypedDict structure."""
        report: ValidationReport = {
            "total_entries": 100,
            "average_confidence": 0.75,
            "store_type": "OrbitStore",
            "modified_entries": 5,
        }
        
        assert isinstance(report["total_entries"], int)
        assert isinstance(report["average_confidence"], float)
        assert isinstance(report["store_type"], str)
        assert report["modified_entries"] == 5

    def test_maintenance_report_structure(self) -> None:
        """Test MaintenanceReport TypedDict structure."""
        report: MaintenanceReport = {
            "operation": "test_operation",
            "success": True,
            "entries_processed": 50,
            "entries_modified": 10,
            "elapsed_seconds": 1.5,
        }
        
        assert isinstance(report["operation"], str)
        assert isinstance(report["success"], bool)
        assert isinstance(report["entries_processed"], int)
        assert isinstance(report["entries_modified"], int)
        assert isinstance(report["elapsed_seconds"], float)

    def test_cycle_hook_function_protocol(self) -> None:
        """Test CycleHookFunction protocol compliance."""
        def valid_hook(engine: Any, phenotype_entry: PhenotypeEntry, last_intron: int) -> None:
            """Valid hook implementation."""
            pass
        
        # Should be callable with correct signature
        mock_engine = Mock()
        mock_entry: PhenotypeEntry = {
            "phenotype": "test",
            "confidence": 0.5,
            "exon_mask": 42,
            "usage_count": 1,
            "last_updated": time.time(),
            "created_at": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "context_signature": (0, 42),
            "_original_context": None,
        }
        
        # Should not raise
        valid_hook(mock_engine, mock_entry, 42)

    def test_governance_signature_value_ranges(self) -> None:
        """Test GovernanceSignature value constraints."""
        # Test valid ranges
        valid_sig: GovernanceSignature = {
            "neutral": 6,  # 0-6
            "li": 2,       # 0-2
            "fg": 2,       # 0-2
            "bg": 2,       # 0-2
            "dyn": 6,      # 0-6
        }
        
        # Should accept valid values
        assert 0 <= valid_sig["neutral"] <= 6
        assert 0 <= valid_sig["li"] <= 2
        assert 0 <= valid_sig["fg"] <= 2
        assert 0 <= valid_sig["bg"] <= 2
        assert 0 <= valid_sig["dyn"] <= 6

    def test_phenotype_entry_constraints(self, sample_phenotype: Dict[str, Any]) -> None:
        """Test PhenotypeEntry value constraints."""
        entry: PhenotypeEntry = sample_phenotype
        
        # Validate constraints
        assert 0 <= entry["exon_mask"] <= 255
        assert 0 <= entry["confidence"] <= 1.0
        assert entry["usage_count"] >= 0
        assert entry["created_at"] > 0
        assert entry["last_updated"] >= entry["created_at"]


class TestOrbitStore:
    """Test OrbitStore storage backend functionality."""

    def test_orbit_store_initialization(self, temp_dir: Path) -> None:
        """Test OrbitStore initialization."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        
        store = OrbitStore(store_path)
        
        assert store.store_path == store_path
        assert store.index_path == store_path + ".idx"
        assert store.log_path == store_path + ".log"
        assert store.write_threshold == 100  # Default
        assert isinstance(store.index, dict)
        assert len(store.index) == 0
        
        store.close()

    def test_orbit_store_put_and_get(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test basic put and get operations."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path)
        
        context_key = (100, 42)
        entry = sample_phenotype.copy()
        
        # Put entry
        store.put(context_key, entry)
        
        # Get entry
        retrieved = store.get(context_key)
        
        assert retrieved is not None
        assert retrieved["phenotype"] == entry["phenotype"]
        assert retrieved["context_signature"] == context_key
        
        store.close()

    def test_orbit_store_pending_writes(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test pending writes functionality."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path, write_threshold=2)
        
        entry1 = sample_phenotype.copy()
        entry2 = sample_phenotype.copy()
        
        # Add entries below threshold
        store.put((100, 42), entry1)
        assert (100, 42) in store.pending_writes
        
        # Should be retrievable from pending
        retrieved = store.get((100, 42))
        assert retrieved is not None
        
        # Add another to trigger flush
        store.put((101, 42), entry2)
        
        # Should have flushed pending writes
        assert len(store.pending_writes) == 0
        
        store.close()

    def test_orbit_store_commit(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test commit operation."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path)
        
        entry = sample_phenotype.copy()
        context_key = (100, 42)
        
        store.put(context_key, entry)
        store.commit()
        
        # Index should be saved
        index_path = Path(store.index_path)
        assert index_path.exists()
        
        store.close()

    def test_orbit_store_persistence(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test data persistence across store instances."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        
        # Create and populate store
        store1 = OrbitStore(store_path)
        context_key = (100, 42)
        entry = sample_phenotype.copy()
        
        store1.put(context_key, entry)
        store1.commit()
        store1.close()
        
        # Reopen store
        store2 = OrbitStore(store_path)
        
        # Should load existing data
        retrieved = store2.get(context_key)
        assert retrieved is not None
        assert retrieved["phenotype"] == entry["phenotype"]
        
        store2.close()

    def test_orbit_store_iter_entries(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test iteration over entries."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path)
        
        # Add multiple entries
        entries = {}
        for i in range(5):
            key = (i, 42)
            entry = sample_phenotype.copy()
            entry["phenotype"] = f"P[{i}:42]"
            store.put(key, entry)
            entries[key] = entry
        
        store.commit()
        
        # Iterate and verify
        found_entries = {}
        for key, entry in store.iter_entries():
            found_entries[key] = entry
        
        assert len(found_entries) == 5
        for key in entries:
            assert key in found_entries
            assert found_entries[key]["phenotype"] == entries[key]["phenotype"]
        
        store.close()

    def test_orbit_store_delete(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test entry deletion."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path)
        
        context_key = (100, 42)
        store.put(context_key, sample_phenotype)
        store.commit()
        
        # Verify exists
        assert store.get(context_key) is not None
        
        # Delete
        store.delete(context_key)
        
        # Should be removed from index
        assert context_key not in store.index
        
        store.close()

    def test_orbit_store_mark_dirty(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test mark_dirty functionality."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path)
        
        context_key = (100, 42)
        entry = sample_phenotype.copy()
        
        store.mark_dirty(context_key, entry)
        
        # Should be in pending writes
        assert context_key in store.pending_writes
        assert store.pending_writes[context_key] == entry
        
        store.close()

    def test_orbit_store_mmap_mode(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test memory mapping mode."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        
        # Create store with mmap enabled
        store = OrbitStore(store_path, use_mmap=True)
        
        context_key = (100, 42)
        store.put(context_key, sample_phenotype)
        store.commit()
        
        # Should handle mmap operations
        retrieved = store.get(context_key)
        assert retrieved is not None
        
        store.close()

    def test_orbit_store_data_property(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test data property returns all entries."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path)
        
        # Add entries
        for i in range(3):
            key = (i, 42)
            entry = sample_phenotype.copy()
            entry["phenotype"] = f"P[{i}:42]"
            store.put(key, entry)
        
        store.commit()
        
        # Get all data
        all_data = store.data
        
        assert isinstance(all_data, dict)
        assert len(all_data) == 3
        
        store.close()

    def test_orbit_store_set_data_dict(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test setting entire data dictionary."""
        store_path = str(temp_dir / "test_store.pkl.gz")
        store = OrbitStore(store_path)
        
        # Create test data
        test_data = {}
        for i in range(3):
            key = (i, 42)
            entry = sample_phenotype.copy()
            entry["phenotype"] = f"P[{i}:42]"
            test_data[key] = entry
        
        # Set data
        store.set_data_dict(test_data)
        
        # Verify all entries exist
        for key, expected_entry in test_data.items():
            retrieved = store.get(key)
            assert retrieved is not None
            assert retrieved["phenotype"] == expected_entry["phenotype"]
        
        store.close()


class TestCanonicalView:
    """Test CanonicalView phenomenology canonicalization."""

    def test_canonical_view_initialization(self, temp_dir: Path, meta_paths: Dict[str, str]) -> None:
        """Test CanonicalView initialization."""
        store_path = str(temp_dir / "base_store.pkl.gz")
        base_store = OrbitStore(store_path)
        
        # Check if phenomenology map exists
        pheno_path = meta_paths.get("phenomenology")
        if pheno_path and os.path.exists(pheno_path):
            view = CanonicalView(base_store, pheno_path)
            
            assert view.base_store is base_store
            assert isinstance(view.phenomenology_map, dict)
            
            view.close()
        else:
            # Create minimal phenomenology map for testing
            test_pheno_path = str(temp_dir / "test_pheno.json")
            test_pheno = {
                "phenomenology_map": [0, 1, 2, 0, 1, 2],
                "orbit_sizes": {"0": 3, "1": 2, "2": 1},
            }
            
            with open(test_pheno_path, 'w') as f:
                json.dump(test_pheno, f)
            
            view = CanonicalView(base_store, test_pheno_path)
            
            assert view.base_store is base_store
            assert isinstance(view.phenomenology_map, dict)
            
            view.close()

    def test_canonical_view_phenomenology_mapping(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test phenomenology index mapping."""
        # Create test phenomenology map
        test_pheno_path = str(temp_dir / "test_pheno.json")
        test_pheno = {
            "phenomenology_map": [0, 0, 1, 1, 2, 2],  # Maps indices to representatives
            "orbit_sizes": {"0": 2, "1": 2, "2": 2},
        }
        
        with open(test_pheno_path, 'w') as f:
            json.dump(test_pheno, f)
        
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        view = CanonicalView(base_store, test_pheno_path)
        
        # Put entry with non-canonical index
        original_key = (3, 42)  # Index 3 maps to representative 1
        entry = sample_phenotype.copy()
        
        view.put(original_key, entry)
        
        # Should be stored with canonical key
        canonical_key = (1, 42)  # Representative for index 3
        retrieved = view.get(original_key)
        
        assert retrieved is not None
        assert retrieved["context_signature"] == canonical_key
        
        view.close()

    def test_canonical_view_original_context_preservation(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test preservation of original context."""
        test_pheno_path = str(temp_dir / "test_pheno.json")
        test_pheno = {
            "phenomenology_map": [0, 0, 1, 1],
            "orbit_sizes": {"0": 2, "1": 2},
        }
        
        with open(test_pheno_path, 'w') as f:
            json.dump(test_pheno, f)
        
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        view = CanonicalView(base_store, test_pheno_path)
        
        original_key = (3, 42)
        entry = sample_phenotype.copy()
        
        view.put(original_key, entry)
        
        # Check that original context is preserved in storage
        canonical_key = (1, 42)
        stored_entry = base_store.get(canonical_key)
        
        assert stored_entry is not None
        assert stored_entry["_original_context"] == original_key
        
        view.close()

    def test_canonical_view_get_cleans_metadata(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test get method cleans canonical metadata."""
        test_pheno_path = str(temp_dir / "test_pheno.json")
        test_pheno = {
            "phenomenology_map": [0, 1, 0, 1],
            "orbit_sizes": {"0": 2, "1": 2},
        }
        
        with open(test_pheno_path, 'w') as f:
            json.dump(test_pheno, f)
        
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        view = CanonicalView(base_store, test_pheno_path)
        
        key = (2, 42)  # Maps to representative 0
        entry = sample_phenotype.copy()
        
        view.put(key, entry)
        retrieved = view.get(key)
        
        # Should not expose canonical metadata
        assert "_original_context" not in retrieved
        
        view.close()


class TestOverlayView:
    """Test OverlayView public/private layering."""

    def test_overlay_view_initialization(self, temp_dir: Path) -> None:
        """Test OverlayView initialization."""
        public_store = OrbitStore(str(temp_dir / "public.pkl.gz"))
        private_store = OrbitStore(str(temp_dir / "private.pkl.gz"))
        
        view = OverlayView(public_store, private_store)
        
        assert view.public_store is public_store
        assert view.private_store is private_store
        
        view.close()

    def test_overlay_view_private_priority(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test private store takes priority over public."""
        public_store = OrbitStore(str(temp_dir / "public.pkl.gz"))
        private_store = OrbitStore(str(temp_dir / "private.pkl.gz"))
        view = OverlayView(public_store, private_store)
        
        context_key = (100, 42)
        
        # Add to public store
        public_entry = sample_phenotype.copy()
        public_entry["phenotype"] = "PUBLIC"
        public_store.put(context_key, public_entry)
        
        # Add to private store (should override)
        private_entry = sample_phenotype.copy()
        private_entry["phenotype"] = "PRIVATE"
        view.put(context_key, private_entry)
        
        # Should get private version
        retrieved = view.get(context_key)
        assert retrieved["phenotype"] == "PRIVATE"
        
        view.close()

    def test_overlay_view_public_fallback(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test fallback to public store when not in private."""
        public_store = OrbitStore(str(temp_dir / "public.pkl.gz"))
        private_store = OrbitStore(str(temp_dir / "private.pkl.gz"))
        view = OverlayView(public_store, private_store)
        
        context_key = (100, 42)
        
        # Add only to public store
        public_entry = sample_phenotype.copy()
        public_entry["phenotype"] = "PUBLIC_ONLY"
        public_store.put(context_key, public_entry)
        
        # Should fallback to public
        retrieved = view.get(context_key)
        assert retrieved["phenotype"] == "PUBLIC_ONLY"
        
        view.close()

    def test_overlay_view_put_goes_to_private(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test put operations go to private store."""
        public_store = OrbitStore(str(temp_dir / "public.pkl.gz"))
        private_store = OrbitStore(str(temp_dir / "private.pkl.gz"))
        view = OverlayView(public_store, private_store)
        
        context_key = (100, 42)
        entry = sample_phenotype.copy()
        
        view.put(context_key, entry)
        
        # Should be in private store
        private_retrieved = private_store.get(context_key)
        assert private_retrieved is not None
        
        # Should not be in public store
        public_retrieved = public_store.get(context_key)
        assert public_retrieved is None
        
        view.close()

    def test_overlay_view_iter_entries_merging(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test iter_entries merges both stores with private priority."""
        public_store = OrbitStore(str(temp_dir / "public.pkl.gz"))
        private_store = OrbitStore(str(temp_dir / "private.pkl.gz"))
        view = OverlayView(public_store, private_store)
        
        # Add to public
        public_key = (100, 42)
        public_entry = sample_phenotype.copy()
        public_entry["phenotype"] = "PUBLIC"
        public_store.put(public_key, public_entry)
        
        # Add to private (different key)
        private_key = (101, 42)
        private_entry = sample_phenotype.copy()
        private_entry["phenotype"] = "PRIVATE"
        private_store.put(private_key, private_entry)
        
        # Add overlapping key to private (should shadow public)
        overlap_key = (102, 42)
        public_overlap = sample_phenotype.copy()
        public_overlap["phenotype"] = "PUBLIC_OVERLAP"
        public_store.put(overlap_key, public_overlap)
        
        private_overlap = sample_phenotype.copy()
        private_overlap["phenotype"] = "PRIVATE_OVERLAP"
        private_store.put(overlap_key, private_overlap)
        
        public_store.commit()
        private_store.commit()
        
        # Iterate and collect
        found_entries = {}
        for key, entry in view.iter_entries():
            found_entries[key] = entry
        
        # Should have all unique keys
        assert public_key in found_entries
        assert private_key in found_entries
        assert overlap_key in found_entries
        
        # Private should take priority for overlap
        assert found_entries[overlap_key]["phenotype"] == "PRIVATE_OVERLAP"
        
        view.close()

    def test_overlay_view_data_property(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test data property combines both stores."""
        public_store = OrbitStore(str(temp_dir / "public.pkl.gz"))
        private_store = OrbitStore(str(temp_dir / "private.pkl.gz"))
        view = OverlayView(public_store, private_store)
        
        # Add entries to both stores
        public_store.put((100, 42), sample_phenotype)
        private_store.put((101, 42), sample_phenotype)
        
        public_store.commit()
        private_store.commit()
        
        combined_data = view.data
        
        assert isinstance(combined_data, dict)
        assert (100, 42) in combined_data
        assert (101, 42) in combined_data
        
        view.close()


class TestReadOnlyView:
    """Test ReadOnlyView restrictions."""

    def test_read_only_view_initialization(self, temp_dir: Path) -> None:
        """Test ReadOnlyView initialization."""
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        view = ReadOnlyView(base_store)
        
        assert view.base_store is base_store
        
        view.close()

    def test_read_only_view_get_works(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test get operations work in read-only view."""
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        
        # Add data to base store
        context_key = (100, 42)
        base_store.put(context_key, sample_phenotype)
        
        view = ReadOnlyView(base_store)
        
        # Should be able to read
        retrieved = view.get(context_key)
        assert retrieved is not None
        assert retrieved["phenotype"] == sample_phenotype["phenotype"]
        
        view.close()

    def test_read_only_view_put_raises(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test put operations raise error in read-only view."""
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        view = ReadOnlyView(base_store)
        
        with pytest.raises(RuntimeError, match="read-only"):
            view.put((100, 42), sample_phenotype)
        
        view.close()

    def test_read_only_view_data_property(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test data property works in read-only view."""
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        base_store.put((100, 42), sample_phenotype)
        
        view = ReadOnlyView(base_store)
        
        data = view.data
        assert isinstance(data, dict)
        assert (100, 42) in data
        
        view.close()


class TestPhenomenologyMapLoading:
    """Test phenomenology map loading and caching."""

    def test_load_phenomenology_map_list_format(self, temp_dir: Path) -> None:
        """Test loading phenomenology map in list format."""
        pheno_path = str(temp_dir / "pheno_list.json")
        test_map = [0, 1, 2, 0, 1, 2]
        
        with open(pheno_path, 'w') as f:
            json.dump(test_map, f)
        
        result = load_phenomenology_map(pheno_path)
        
        expected = {i: rep for i, rep in enumerate(test_map)}
        assert result == expected

    def test_load_phenomenology_map_dict_format(self, temp_dir: Path) -> None:
        """Test loading phenomenology map in dict format."""
        pheno_path = str(temp_dir / "pheno_dict.json")
        test_data = {
            "phenomenology_map": [0, 1, 2, 0, 1, 2],
            "orbit_sizes": {"0": 3, "1": 2, "2": 1},
        }
        
        with open(pheno_path, 'w') as f:
            json.dump(test_data, f)
        
        result = load_phenomenology_map(pheno_path)
        
        expected = {i: rep for i, rep in enumerate(test_data["phenomenology_map"])}
        assert result == expected

    def test_load_phenomenology_map_caching(self, temp_dir: Path) -> None:
        """Test phenomenology map caching."""
        pheno_path = str(temp_dir / "pheno_cache.json")
        test_map = [0, 1, 2]
        
        with open(pheno_path, 'w') as f:
            json.dump(test_map, f)
        
        # Load twice
        result1 = load_phenomenology_map(pheno_path)
        result2 = load_phenomenology_map(pheno_path)
        
        # Should be same object (cached)
        assert result1 is result2

    def test_load_phenomenology_map_invalid_format(self, temp_dir: Path) -> None:
        """Test error handling for invalid format."""
        pheno_path = str(temp_dir / "pheno_invalid.json")
        
        with open(pheno_path, 'w') as f:
            json.dump({"invalid": "format"}, f)
        
        with pytest.raises(ValueError, match="Unrecognized phenomenology map format"):
            load_phenomenology_map(pheno_path)


class TestMaintenanceFunctions:
    """Test maintenance and utility functions."""

    def test_merge_phenotype_maps_basic(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test basic phenotype map merging."""
        # Create source stores
        source1_path = str(temp_dir / "source1.pkl.gz")
        source2_path = str(temp_dir / "source2.pkl.gz")
        dest_path = str(temp_dir / "merged.pkl.gz")
        
        store1 = OrbitStore(source1_path)
        store2 = OrbitStore(source2_path)
        
        # Add different entries
        entry1 = sample_phenotype.copy()
        entry1["phenotype"] = "SOURCE1"
        store1.put((100, 42), entry1)
        
        entry2 = sample_phenotype.copy()
        entry2["phenotype"] = "SOURCE2"
        store2.put((101, 42), entry2)
        
        store1.commit()
        store2.commit()
        store1.close()
        store2.close()
        
        # Merge
        report = merge_phenotype_maps([source1_path, source2_path], dest_path)
        
        assert report["success"] is True
        assert report["entries_processed"] == 2
        assert report["entries_modified"] == 2
        
        # Verify merged result
        merged_store = OrbitStore(dest_path)
        result1 = merged_store.get((100, 42))
        result2 = merged_store.get((101, 42))
        
        assert result1["phenotype"] == "SOURCE1"
        assert result2["phenotype"] == "SOURCE2"
        
        merged_store.close()

    def test_merge_phenotype_maps_conflict_resolution(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test conflict resolution in merging."""
        source1_path = str(temp_dir / "source1.pkl.gz")
        source2_path = str(temp_dir / "source2.pkl.gz")
        dest_path = str(temp_dir / "merged.pkl.gz")
        
        store1 = OrbitStore(source1_path)
        store2 = OrbitStore(source2_path)
        
        # Add conflicting entries (same key)
        key = (100, 42)
        
        entry1 = sample_phenotype.copy()
        entry1["confidence"] = 0.3
        entry1["phenotype"] = "LOW_CONF"
        store1.put(key, entry1)
        
        entry2 = sample_phenotype.copy()
        entry2["confidence"] = 0.8
        entry2["phenotype"] = "HIGH_CONF"
        store2.put(key, entry2)
        
        store1.commit()
        store2.commit()
        store1.close()
        store2.close()
        
        # Merge with highest confidence strategy
        report = merge_phenotype_maps(
            [source1_path, source2_path], 
            dest_path, 
            conflict_resolution="highest_confidence"
        )
        
        assert report["success"] is True
        
        # Should keep higher confidence entry
        merged_store = OrbitStore(dest_path)
        result = merged_store.get(key)
        assert result["phenotype"] == "HIGH_CONF"
        
        merged_store.close()

    def test_apply_global_confidence_decay(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test global confidence decay application."""
        store_path = str(temp_dir / "decay_test.pkl.gz")
        store = OrbitStore(store_path)
        
        # Add entry with older timestamp
        entry = sample_phenotype.copy()
        entry["confidence"] = 0.8
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        entry["last_updated"] = old_time
        
        store.put((100, 42), entry)
        store.commit()
        store.close()
        
        # Apply decay
        report = apply_global_confidence_decay(
            store_path,
            decay_factor=0.1,
            time_threshold_days=30.0
        )
        
        assert report["success"] is True
        assert report["entries_processed"] == 1
        assert report["entries_modified"] == 1
        
        # Verify decay was applied
        store2 = OrbitStore(store_path)
        updated_entry = store2.get((100, 42))
        assert updated_entry["confidence"] < 0.8
        
        store2.close()

    def test_export_knowledge_statistics(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test knowledge statistics export."""
        store_path = str(temp_dir / "stats_test.pkl.gz")
        output_path = str(temp_dir / "stats.json")
        
        store = OrbitStore(store_path)
        
        # Add test entries
        for i in range(5):
            entry = sample_phenotype.copy()
            entry["confidence"] = 0.1 * i + 0.5  # Varying confidence
            entry["exon_mask"] = i * 10  # Varying masks
            entry["usage_count"] = i + 1
            store.put((i, 42), entry)
        
        store.commit()
        store.close()
        
        # Export statistics
        report = export_knowledge_statistics(store_path, output_path)
        
        assert report["success"] is True
        assert report["entries_processed"] == 5
        
        # Verify statistics file
        assert os.path.exists(output_path)
        
        with open(output_path) as f:
            stats = json.load(f)
        
        assert stats["total_entries"] == 5
        assert "confidence" in stats
        assert "memory" in stats
        assert "usage" in stats

    def test_validate_ontology_integrity(self, meta_paths: Dict[str, str]) -> None:
        """Test ontology integrity validation."""
        if "ontology" in meta_paths and os.path.exists(meta_paths["ontology"]):
            report = validate_ontology_integrity(meta_paths["ontology"])
            
            assert report["success"] is True
            assert report["entries_processed"] > 0
        else:
            # Create test ontology
            test_ont_path = str(Path(meta_paths.get("ontology", "")).parent / "test_ont.json")
            test_ont = {
                "schema_version": "0.9.6",
                "ontology_map": {str(i): i for i in range(788_986)},
                "endogenous_modulus": 788_986,
                "ontology_diameter": 6,
                "total_states": 788_986,
            }
            
            with open(test_ont_path, 'w') as f:
                json.dump(test_ont, f)
            
            report = validate_ontology_integrity(test_ont_path)
            assert report["success"] is True

    def test_prune_and_compact_store(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test store pruning and compaction."""
        store_path = str(temp_dir / "prune_test.pkl.gz")
        store = OrbitStore(store_path)
        
        current_time = time.time()
        
        # Add entries with different ages and confidences
        for i in range(5):
            entry = sample_phenotype.copy()
            entry["confidence"] = 0.01 if i < 2 else 0.5  # Low confidence for first 2
            entry["last_updated"] = current_time - (i * 10 * 24 * 3600)  # Varying ages
            store.put((i, 42), entry)
        
        store.commit()
        store.close()
        
        # Prune low confidence entries
        report = prune_and_compact_store(
            store_path,
            min_confidence=0.02,
            max_age_days=20.0
        )
        
        assert report["success"] is True
        assert report["entries_processed"] == 5
        assert report["entries_modified"] > 0  # Should have pruned some
        
        # Verify pruning
        store2 = OrbitStore(store_path)
        remaining_entries = list(store2.iter_entries())
        assert len(remaining_entries) < 5  # Some should be pruned
        
        store2.close()

    def test_prune_and_compact_store_dry_run(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test dry run mode of pruning."""
        store_path = str(temp_dir / "dry_run_test.pkl.gz")
        store = OrbitStore(store_path)
        
        entry = sample_phenotype.copy()
        entry["confidence"] = 0.01  # Low confidence
        store.put((100, 42), entry)
        store.commit()
        store.close()
        
        # Dry run
        report = prune_and_compact_store(
            store_path,
            min_confidence=0.02,
            dry_run=True
        )
        
        assert report["success"] is True
        assert report["entries_modified"] == 1  # Would be pruned
        
        # Entry should still exist
        store2 = OrbitStore(store_path)
        still_exists = store2.get((100, 42))
        assert still_exists is not None
        
        store2.close()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_to_native_basic_types(self) -> None:
        """Test to_native with basic Python types."""
        assert to_native(42) == 42
        assert to_native("hello") == "hello"
        assert to_native([1, 2, 3]) == [1, 2, 3]
        assert to_native({"a": 1}) == {"a": 1}

    def test_to_native_numpy_types(self) -> None:
        """Test to_native with numpy types."""
        import numpy as np
        
        # Numpy scalars
        assert to_native(np.int32(42)) == 42
        assert to_native(np.float64(3.14)) == 3.14
        assert to_native(np.bool_(True)) is True

    def test_to_native_nested_structures(self) -> None:
        """Test to_native with nested data structures."""
        import numpy as np
        
        nested = {
            "list": [1, np.int32(2), 3],
            "tuple": (np.float64(1.0), 2, 3),
            "dict": {"nested": np.bool_(False)},
        }
        
        result = to_native(nested)
        
        assert result["list"] == [1, 2, 3]
        assert result["tuple"] == (1.0, 2, 3)
        assert result["dict"]["nested"] is False

    def test_to_native_preserves_structure(self) -> None:
        """Test to_native preserves data structure types."""
        original_dict = {"a": 1, "b": 2}
        original_list = [1, 2, 3]
        original_tuple = (1, 2, 3)
        
        assert isinstance(to_native(original_dict), dict)
        assert isinstance(to_native(original_list), list)
        assert isinstance(to_native(original_tuple), tuple)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_orbit_store_invalid_path(self) -> None:
        """Test OrbitStore with invalid path."""
        # Should handle gracefully during initialization
        store = OrbitStore("/invalid/path/store.pkl.gz")
        
        # Operations may fail, but initialization should work
        assert store.store_path == "/invalid/path/store.pkl.gz"
        
        # Close should not raise
        store.close()

    def test_canonical_view_missing_phenomenology(self, temp_dir: Path) -> None:
        """Test CanonicalView with missing phenomenology file."""
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        missing_pheno_path = str(temp_dir / "missing.json")
        
        with pytest.raises(FileNotFoundError):
            CanonicalView(base_store, missing_pheno_path)
        
        base_store.close()

    def test_merge_phenotype_maps_missing_sources(self, temp_dir: Path) -> None:
        """Test merging with missing source files."""
        dest_path = str(temp_dir / "merged.pkl.gz")
        
        # Should handle missing sources gracefully
        report = merge_phenotype_maps(["/missing/source1.pkl.gz"], dest_path)
        
        assert report["success"] is True
        assert report["entries_processed"] == 0

    def test_maintenance_functions_missing_store(self, temp_dir: Path) -> None:
        """Test maintenance functions with missing store."""
        missing_path = str(temp_dir / "missing.pkl.gz")
        
        # Should return failure reports
        decay_report = apply_global_confidence_decay(missing_path)
        assert decay_report["success"] is False
        
        stats_report = export_knowledge_statistics(missing_path, str(temp_dir / "stats.json"))
        assert stats_report["success"] is False
        
        prune_report = prune_and_compact_store(missing_path)
        assert prune_report["success"] is False

    def test_orbit_store_concurrent_access(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test OrbitStore handles concurrent-like access patterns."""
        store_path = str(temp_dir / "concurrent.pkl.gz")
        
        # Create and use store
        store1 = OrbitStore(store_path)
        store1.put((100, 42), sample_phenotype)
        store1.commit()
        store1.close()
        
        # Open another instance (simulates concurrent access)
        store2 = OrbitStore(store_path)
        retrieved = store2.get((100, 42))
        
        assert retrieved is not None
        assert retrieved["phenotype"] == sample_phenotype["phenotype"]
        
        store2.close()

    def test_orbit_store_large_batch_threshold(self, temp_dir: Path, sample_phenotype: Dict[str, Any]) -> None:
        """Test OrbitStore with large batch operations."""
        store_path = str(temp_dir / "large_batch.pkl.gz")
        store = OrbitStore(store_path, write_threshold=10)
        
        # Add many entries
        for i in range(15):
            entry = sample_phenotype.copy()
            entry["phenotype"] = f"Entry_{i}"
            store.put((i, 42), entry)
        
        # Should have flushed automatically
        assert len(store.pending_writes) < 15
        
        store.close()

    def test_view_delegation_methods(self, temp_dir: Path) -> None:
        """Test that views properly delegate to base stores."""
        base_store = OrbitStore(str(temp_dir / "base.pkl.gz"))
        
        # Test ReadOnlyView delegation
        readonly = ReadOnlyView(base_store)
        
        # Should delegate data property
        assert readonly.data is not None
        
        # Should delegate _load_index if available
        if hasattr(base_store, '_load_index'):
            readonly._load_index()  # Should not raise
        
        readonly.close()

    def test_phenomenology_map_string_keys(self, temp_dir: Path) -> None:
        """Test phenomenology map handles string keys."""
        pheno_path = str(temp_dir / "string_keys.json")
        test_data = {
            "phenomenology_map": {"0": 0, "1": 1, "2": 0}  # String keys
        }
        
        with open(pheno_path, 'w') as f:
            json.dump(test_data, f)
        
        result = load_phenomenology_map(pheno_path)
        
        # Should convert to integer keys
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert isinstance(list(result.keys())[0], int)

    def test_maintenance_report_validation(self) -> None:
        """Test maintenance report structure validation."""
        # Valid report
        valid_report: MaintenanceReport = {
            "operation": "test",
            "success": True,
            "entries_processed": 10,
            "entries_modified": 5,
            "elapsed_seconds": 1.5,
        }
        
        # Should have all required fields
        assert "operation" in valid_report
        assert "success" in valid_report
        assert "entries_processed" in valid_report
        assert "entries_modified" in valid_report
        assert "elapsed_seconds" in valid_report
        
        # Validate types
        assert isinstance(valid_report["operation"], str)
        assert isinstance(valid_report["success"], bool)
        assert isinstance(valid_report["entries_processed"], int)
        assert isinstance(valid_report["entries_modified"], int)
        assert isinstance(valid_report["elapsed_seconds"], (int, float))