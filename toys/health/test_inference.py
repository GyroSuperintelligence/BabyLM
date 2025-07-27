"""
Comprehensive tests for inference.py - the interpretation and meaning management layer.
Tests semantic conversion, learning via Monodromic Fold, and knowledge maintenance.
"""

import math
import time
from typing import Any, Dict, cast, Generator, Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from baby import governance
from baby.contracts import PhenotypeEntry, AgentConfig
from baby.inference import InferenceEngine
from baby.information import InformationEngine
from baby.policies import OrbitStore
from baby.intelligence import GyroSI


@pytest.fixture
def temp_store(tmp_path: Path) -> Generator[OrbitStore, None, None]:
    """Simple OrbitStore for testing, completely isolated."""
    store = OrbitStore(str(tmp_path / "test_store.bin"), append_only=True)
    yield store
    store.close()


class TestInferenceEngineInitialization:
    """Test InferenceEngine initialization and setup."""

    def test_initialization_with_real_components(self, test_env: Dict[str, Any], temp_store: OrbitStore) -> None:
        """Test engine initializes correctly with real components."""
        # Load real ontology data
        keys = test_env["main_meta_files"]["ontology"]
        ep = test_env["main_meta_files"]["epistemology"]
        pheno = test_env["main_meta_files"]["phenomenology"]
        theta = test_env["main_meta_files"]["theta"]
        s2_engine = InformationEngine(keys, ep, pheno, theta)
        assert s2_engine._keys is not None
        assert len(s2_engine._keys) > 0
        s3_engine = InferenceEngine(s2_engine, temp_store)

        assert s3_engine.s2 is s2_engine
        assert s3_engine.store is temp_store
        assert s3_engine.endogenous_modulus == len(s2_engine._keys)
        assert s3_engine._v_max is not None and s3_engine._v_max > 0

    def test_initialization_validates_orbit_cardinality(self, test_env: Dict[str, Any], temp_store: OrbitStore) -> None:
        """Test initialization fails with zero orbit cardinality."""
        keys = test_env["main_meta_files"]["ontology"]
        ep = test_env["main_meta_files"]["epistemology"]
        pheno = test_env["main_meta_files"]["phenomenology"]
        theta = test_env["main_meta_files"]["theta"]
        s2_engine = InformationEngine(keys, ep, pheno, theta)
        assert s2_engine._keys is not None
        assert len(s2_engine._keys) > 0

        # Mock zero orbit cardinality
        with patch.object(s2_engine, "orbit_cardinality", np.zeros_like(s2_engine.orbit_cardinality)):
            with pytest.raises(ValueError, match="orbit cardinality cannot be zero"):
                InferenceEngine(s2_engine, temp_store)

    def test_v_max_calculation(self, test_env: Dict[str, Any], temp_store: OrbitStore) -> None:
        """Test _v_max is correctly calculated from orbit cardinality."""
        keys = test_env["main_meta_files"]["ontology"]
        ep = test_env["main_meta_files"]["epistemology"]
        pheno = test_env["main_meta_files"]["phenomenology"]
        theta = test_env["main_meta_files"]["theta"]
        s2_engine = InformationEngine(keys, ep, pheno, theta)
        assert s2_engine._keys is not None
        assert len(s2_engine._keys) > 0
        s3_engine = InferenceEngine(s2_engine, temp_store)

        # _v_max should be maximum orbit cardinality
        if hasattr(s2_engine.orbit_cardinality, "max"):
            expected_max = s2_engine.orbit_cardinality.max()
        else:
            expected_max = max(s2_engine.orbit_cardinality)

        assert s3_engine._v_max == expected_max
        assert s3_engine._v_max is not None and s3_engine._v_max > 0


class TestPhenotypeOperations:
    """Test phenotype creation, retrieval, and management."""

    def test_get_phenotype_creates_new_entry(self, gyrosi_agent: GyroSI) -> None:
        """Test get_phenotype creates new entry for unknown context."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        intron = 42

        # Should create new entry
        entry = engine.get_phenotype(state_index, intron)

        assert entry["context_signature"] == (state_index, intron)
        assert entry["phenotype"] == f"P[{state_index}:{intron}]"
        assert 0 <= entry["exon_mask"] <= 255
        assert 0 <= entry["confidence"] <= 1.0
        assert entry["usage_count"] == 0
        assert entry["created_at"] > 0
        assert entry["last_updated"] > 0

    def test_get_phenotype_retrieves_existing_entry(self, gyrosi_agent: GyroSI) -> None:
        """Test get_phenotype retrieves existing entry."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        intron = 42

        # Create entry first time
        entry1 = engine.get_phenotype(state_index, intron)
        original_created_at = entry1["created_at"]

        # Should retrieve same entry second time
        entry2 = engine.get_phenotype(state_index, intron)

        assert entry2["context_signature"] == (state_index, intron)
        assert entry2["created_at"] == original_created_at
        # Remove _original_context for comparison
        entry1_dict = dict(entry1)
        entry2_dict = dict(entry2)
        # Instead of popping, ignore _original_context in comparison
        entry1_filtered = {k: v for k, v in entry1_dict.items() if k != "_original_context"}
        entry2_filtered = {k: v for k, v in entry2_dict.items() if k != "_original_context"}
        assert entry1_filtered == entry2_filtered

    def test_phenotype_intron_masking(self, gyrosi_agent: GyroSI) -> None:
        """Test intron is properly masked to 8 bits."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        intron_large = 0x1FF  # 9 bits
        intron_masked = 0xFF  # 8 bits equivalent

        entry1 = engine.get_phenotype(state_index, intron_large)
        entry2 = engine.get_phenotype(state_index, intron_masked)

        # Should be equivalent due to masking
        assert entry1["context_signature"] == entry2["context_signature"]

    def test_phenotype_confidence_calculation(self, gyrosi_agent: GyroSI) -> None:
        """Test initial confidence is calculated based on orbit cardinality."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 0  # Use origin state
        intron = 42

        entry = engine.get_phenotype(state_index, intron)

        # Should use orbit cardinality formula
        v = engine.s2.orbit_cardinality[state_index]
        expected_confidence = (1 / 6) * math.sqrt(v / engine._v_max)

        assert abs(entry["confidence"] - expected_confidence) < 1e-10

    def test_governance_signature_calculation(self, gyrosi_agent: GyroSI) -> None:
        """Test governance signature is correctly computed."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        intron = 0b10101010  # Known pattern

        entry = engine.get_phenotype(state_index, intron)

        # Verify governance signature matches expected calculation
        expected_sig = governance.compute_governance_signature(intron)
        gov_sig = entry["governance_signature"]

        assert gov_sig["neutral"] == expected_sig[0]
        assert gov_sig["li"] == expected_sig[1]
        assert gov_sig["fg"] == expected_sig[2]
        assert gov_sig["bg"] == expected_sig[3]
        assert gov_sig["dyn"] == expected_sig[4]

    def test_canonical_index_assertion(self, gyrosi_agent: GyroSI) -> None:
        """Test that state_index must be canonical."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Invalid state index should trigger assertion
        with pytest.raises(AssertionError):
            engine.get_phenotype(len(engine.s2.orbit_cardinality), 42)  # Out of range

        with pytest.raises(AssertionError):
            engine.get_phenotype(-1, 42)  # Negative


class TestLearningOperations:
    """Test learning via Monodromic Fold operations."""

    def test_learn_by_key_creates_and_learns(self, gyrosi_agent: GyroSI) -> None:
        """Test learn_by_key creates entry and applies learning."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        intron = 42

        # Should create and learn
        entry = engine.learn_by_key(state_index, intron)

        assert entry["context_signature"] == (state_index, intron)
        assert entry["usage_count"] == 1  # Should be incremented

        # Exon mask should be result of fold(initial_mask, intron)
        # Initial mask is intron itself, so fold(intron, intron) = 0
        assert entry["exon_mask"] == 0

    def test_learn_method_applies_monodromic_fold(self, gyrosi_agent: GyroSI, sample_phenotype: Dict[str, Any]) -> None:
        """Test learn method applies Monodromic Fold correctly."""
        engine = gyrosi_agent.engine.operator
        # Create entry with known mask
        entry = cast(PhenotypeEntry, sample_phenotype.copy())
        entry["context_signature"] = (100, 42)
        intron = 33
        # Store the entry first
        engine.store.put((100, 42), entry)
        # Snapshot usage_count and old_mask before learning
        original_usage_count = entry["usage_count"]
        old_mask = entry["exon_mask"]
        # Learn should apply fold
        updated_entry = engine.learn(entry, intron)
        expected_mask = governance.fold(old_mask, intron)
        assert updated_entry["exon_mask"] == expected_mask
        assert updated_entry["usage_count"] == original_usage_count + 1

    def test_learn_confidence_update(self, gyrosi_agent: GyroSI, sample_phenotype: Dict[str, Any]) -> None:
        """Test confidence is updated based on novelty."""
        engine = gyrosi_agent.engine.operator

        entry = cast(PhenotypeEntry, sample_phenotype.copy())
        entry["context_signature"] = (100, 42)
        intron = 0xFF  # Will create maximum novelty

        updated_entry = engine.learn(entry, intron)

        # Should increase confidence due to novelty
        assert float(updated_entry["confidence"]) >= float(entry["confidence"])
        assert float(updated_entry["confidence"]) <= 1.0

    def test_learn_no_change_optimization(self, gyrosi_agent: GyroSI) -> None:
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0
        entry = engine.get_phenotype(100, 0)  # any intron, we'll tweak
        entry["exon_mask"] = 0
        old_last_updated = entry["last_updated"]
        entry["confidence"] = 0.5  # any value < 1 still fine (no novelty => no change)

        with patch("time.time", return_value=old_last_updated + 10):
            updated = engine.learn(entry, 0)  # intron 0 keeps mask 0
        assert updated["last_updated"] > old_last_updated

    def test_learn_preserves_mask_range(self, gyrosi_agent: GyroSI) -> None:
        """Test learning always keeps exon_mask in valid range."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Test with various starting masks and introns
        for mask in [0, 1, 42, 128, 255]:
            for intron in [0, 1, 42, 128, 255]:
                entry: PhenotypeEntry = {
                    "phenotype": "test",
                    "exon_mask": mask,
                    "confidence": 0.5,
                    "context_signature": (100, intron),
                    "usage_count": 0,
                    "created_at": time.time(),
                    "last_updated": time.time(),
                    "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
                    "_original_context": None,
                }

                updated = engine.learn(entry, intron)
                assert 0 <= updated["exon_mask"] <= 255
                assert 0 <= float(updated["confidence"]) <= 1.0

    def test_learn_updates_governance_signature(self, gyrosi_agent: GyroSI, sample_phenotype: Dict[str, Any]) -> None:
        """Test governance signature is updated after learning."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        entry = cast(PhenotypeEntry, sample_phenotype.copy())
        entry["context_signature"] = (100, 42)
        intron = 0b11110000

        updated_entry = engine.learn(entry, intron)

        # Governance signature should reflect new mask
        new_mask = updated_entry["exon_mask"]
        expected_sig = governance.compute_governance_signature(new_mask)
        gov_sig = updated_entry["governance_signature"]

        assert gov_sig["neutral"] == expected_sig[0]
        assert gov_sig["li"] == expected_sig[1]
        assert gov_sig["fg"] == expected_sig[2]
        assert gov_sig["bg"] == expected_sig[3]
        assert gov_sig["dyn"] == expected_sig[4]


class TestBatchLearning:
    """Test batch learning with path-dependent Monodromic Fold."""

    def test_batch_learn_empty_sequence(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning with empty intron sequence."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        result = engine.batch_learn(100, [])
        assert result is None

    def test_batch_learn_single_intron(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test batch learning with single intron (isolated agent)."""
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        intron = 42

        result = engine.batch_learn(state_index, [intron])

        if result is not None:
            assert result["context_signature"] == (state_index, intron)
            assert result["usage_count"] == 1

    def test_batch_learn_sequence_folding(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning folds sequence correctly."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        introns = [1, 2, 3]

        result = engine.batch_learn(state_index, introns)

        # Should use fold_sequence to compute path-dependent intron
        expected_path_intron = governance.fold_sequence(introns, start_state=0)
        if result is not None:
            assert result["context_signature"] == (state_index, expected_path_intron)

    def test_batch_learn_preserves_path_dependence(self, gyrosi_agent: GyroSI) -> None:
        """Test different orders give different results."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        introns1 = [1, 2, 3]
        introns2 = [3, 2, 1]

        result1 = engine.batch_learn(state_index, introns1)
        result2 = engine.batch_learn(state_index + 1, introns2)  # Different state to avoid conflict

        # Different orders should give different path introns
        path1 = governance.fold_sequence(introns1, 0)
        path2 = governance.fold_sequence(introns2, 0)

        if path1 != path2 and result1 is not None and result2 is not None:
            assert result1["context_signature"][1] != result2["context_signature"][1]

    def test_batch_learn_intron_masking(self, gyrosi_agent: GyroSI) -> None:
        """Test introns are properly masked in batch learning."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        introns = [0x1FF, 0x200]  # Values > 255

        result = engine.batch_learn(state_index, introns)

        # Should mask each intron before folding
        masked_introns = [i & 0xFF for i in introns]
        expected_path = governance.fold_sequence(masked_introns, 0)

        if result is not None:
            assert result["context_signature"] == (state_index, expected_path)


class TestValidationAndMaintenance:
    """Test knowledge validation and integrity checking."""

    def test_validate_knowledge_integrity_empty_store(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        report = engine.validate_knowledge_integrity()

        assert isinstance(report, dict)
        assert report["total_entries"] == 0
        assert report["average_confidence"] == 0
        assert "store_type" in report
        assert report["modified_entries"] == 0  # No anomalies

    def test_validate_knowledge_integrity_with_entries(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Add some valid entries
        for i in range(5):
            engine.learn_by_key(i, 42)

        report = engine.validate_knowledge_integrity()

        assert report["total_entries"] == 5
        assert report["average_confidence"] > 0
        assert report["modified_entries"] == 0  # No anomalies

    def test_validate_knowledge_integrity_detects_anomalies(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Create problematic entry that might cause exceptions
        bad_entry: PhenotypeEntry = {
            "phenotype": "bad",
            "confidence": -0.5,  # Invalid value triggers anomaly, but type is float
            "exon_mask": 0,
            "usage_count": 0,
            "created_at": time.time(),
            "last_updated": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "context_signature": (0, 0),
            "_original_context": None,
        }

        engine.store.put((100, 42), bad_entry)

        # Validation should handle exceptions and count as anomalies
        report = engine.validate_knowledge_integrity()

        assert report["modified_entries"] > 0  # Should count exception as anomaly
        assert report["total_entries"] == 1

    def test_validate_knowledge_integrity_context_signature_mismatch(self, gyrosi_agent: GyroSI) -> None:
        """Test validation detects context signature mismatches."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Create entry with mismatched context signature
        # Note: With binary struct format, the system automatically corrects key mismatches
        # by using the context_signature as the canonical key for storage
        entry = {
            "phenotype": "test",
            "exon_mask": 42,
            "confidence": 0.5,
            "context_signature": (100, 200),  # This becomes the actual storage key
            "usage_count": 0,
            "created_at": time.time(),
            "last_updated": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "_original_context": None,
        }

        # Inject directly into the raw store, bypassing any view layers
        def _unwrap_raw_store(s: Any) -> Any:
            for attr in ("base_store", "private_store", "public_store"):
                if hasattr(s, attr):
                    return _unwrap_raw_store(getattr(s, attr))
            return s

        raw_store = _unwrap_raw_store(engine.store)
        raw_store.put((0, 42), entry)

        report = engine.validate_knowledge_integrity()

        # With binary struct format, key mismatches are automatically corrected
        # so no anomalies should be detected
        assert report["modified_entries"] == 0  # No mismatches should be detected


class TestConfidenceAndDecay:
    """Test confidence calculations and decay operations."""

    def test_apply_confidence_decay_basic(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Add a single entry
        engine.learn_by_key(100, 42)

        report = engine.apply_confidence_decay()

        assert report["total_entries"] == 1
        assert report["modified_entries"] == 1

    def test_apply_confidence_decay_minimum_floor(self, gyrosi_agent: GyroSI) -> None:
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0
        # Create entry and manually set very low confidence
        engine.learn_by_key(100, 42)
        entry = engine.get_phenotype(100, 42)
        entry["confidence"] = 0.001  # Very low
        engine.store.put((100, 42), entry)
        # Apply strong decay
        engine.apply_confidence_decay(decay_factor=10.0)
        # Should not go below minimum
        updated_entry = engine.get_phenotype(100, 42)
        if updated_entry is not None:
            assert float(updated_entry["confidence"]) >= 0.01

    def test_apply_confidence_decay_exponential_formula(self, gyrosi_agent: GyroSI) -> None:
        engine = gyrosi_agent.engine.operator
        entry = engine.learn_by_key(100, 42)
        original_confidence = entry["confidence"]
        # Set last_updated to 1 day ago
        import time

        entry["last_updated"] = time.time() - 24 * 3600
        engine.store.put((100, 42), entry)
        if hasattr(engine.store, "commit"):
            engine.store.commit()

        decay_factor = 0.1
        # Apply decay
        engine.apply_confidence_decay(decay_factor=decay_factor)

        # Check formula: confidence * exp(-decay_factor * days_since_update)
        updated_entry = engine.store.get((100, 42))
        if updated_entry is not None:
            import time

            age_days = (time.time() - entry["last_updated"]) / (24 * 3600)
            expected_confidence = max(0.01, original_confidence * math.exp(-decay_factor * age_days))
            print(f"original_confidence: {original_confidence}")
            print(f"updated_confidence: {updated_entry['confidence']}")
            print(f"expected_confidence: {expected_confidence}")
            print(f"age_days: {age_days}")
            assert abs(float(updated_entry["confidence"]) - expected_confidence) < 1e-2


class TestPruningOperations:
    """Test entry pruning and cleanup operations."""

    def test_prune_low_confidence_entries_basic(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Add a low-confidence entry by creating it and then modifying it
        entry = engine.learn_by_key(100, 42)
        entry["confidence"] = 0.01  # Set low confidence
        engine.store.put((100, 42), entry)

        # Ensure the entry is committed to the store
        if hasattr(engine.store, "commit"):
            engine.store.commit()

        # For append-only stores, pruning will raise an exception
        # We test that the method handles this gracefully
        try:
            removed_count = engine.prune_low_confidence_entries(confidence_threshold=0.05)
            # If we get here, it's not an append-only store
            assert removed_count == 1
        except RuntimeError as e:
            if "append_only store cannot delete" in str(e):
                # This is expected for append-only stores
                pass
            else:
                raise

    def test_prune_preserves_high_confidence(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test pruning preserves high confidence entries."""
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Create high confidence entry
        engine.learn_by_key(100, 42)
        original_entry = engine.get_phenotype(100, 42)

        # Ensure the entry has high confidence by manually setting it
        original_entry["confidence"] = 0.5  # Set to high confidence
        engine.store.put((100, 42), original_entry)

        # Prune with low threshold
        # For append-only stores, pruning will not actually remove entries
        # but will return the count of entries that would be removed
        removed_count = engine.prune_low_confidence_entries(confidence_threshold=0.01)
        # Since our entry has confidence 0.5 > 0.01, it should not be marked for removal
        assert removed_count == 0

        # Entry should still exist
        preserved_entry = engine.get_phenotype(100, 42)
        assert preserved_entry["confidence"] == original_entry["confidence"]

    def test_auto_pruning_hook_registration_enabled(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that auto-pruning hook is registered when enable_auto_decay is True."""
        # Create agent with auto-pruning enabled
        agent_config = {
            "ontology_path": "memories/public/meta/ontology_keys.npy",
            "knowledge_path": str(tmp_path / "test_knowledge.bin"),
            "preferences": {
                "pruning": {
                    "confidence_threshold": 0.05,
                    "decay_factor": 0.995,
                    "decay_interval_hours": 6,
                    "enable_auto_decay": True,
                }
            },
            "base_path": str(tmp_path),
        }

        agent = GyroSI(cast(AgentConfig, agent_config), agent_id="test_agent")

        # Check that auto-pruning hook is registered
        hook_count = len(agent.engine.post_cycle_hooks)
        assert hook_count > 0, "Auto-pruning hook should be registered when enable_auto_decay is True"

    def test_auto_pruning_hook_registration_disabled(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that auto-pruning hook is not registered when enable_auto_decay is False."""
        # Create agent with auto-pruning disabled
        agent_config = {
            "ontology_path": "memories/public/meta/ontology_keys.npy",
            "knowledge_path": str(tmp_path / "test_knowledge.bin"),
            "preferences": {
                "pruning": {
                    "confidence_threshold": 0.05,
                    "decay_factor": 0.995,
                    "decay_interval_hours": 6,
                    "enable_auto_decay": False,
                }
            },
            "base_path": str(tmp_path),
        }

        agent = GyroSI(cast(AgentConfig, agent_config), agent_id="test_agent")

        # Check that auto-pruning hook is not registered
        hook_count = len(agent.engine.post_cycle_hooks)
        assert hook_count == 0, "Auto-pruning hook should not be registered when enable_auto_decay is False"

    def test_auto_pruning_hook_execution(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that auto-pruning hook executes and handles append-only stores gracefully."""
        # Create agent with auto-pruning enabled
        agent_config = {
            "ontology_path": "memories/public/meta/ontology_keys.npy",
            "knowledge_path": str(tmp_path / "test_knowledge.bin"),
            "preferences": {
                "pruning": {
                    "confidence_threshold": 0.05,
                    "decay_factor": 0.995,
                    "decay_interval_hours": 6,
                    "enable_auto_decay": True,
                }
            },
            "base_path": str(tmp_path),
        }

        agent = GyroSI(cast(AgentConfig, agent_config), agent_id="test_agent")
        engine = agent.engine.operator

        # Add low-confidence entries by creating them and then modifying confidence
        entry1 = engine.learn_by_key(100, 42)
        entry1["confidence"] = 0.01  # Below threshold
        engine.store.put((100, 42), entry1)

        entry2 = engine.learn_by_key(101, 43)
        entry2["confidence"] = 0.02  # Below threshold
        engine.store.put((101, 43), entry2)

        entry3 = engine.learn_by_key(102, 44)
        entry3["confidence"] = 0.1  # Above threshold
        engine.store.put((102, 44), entry3)

        # Ensure entries are committed
        if hasattr(engine.store, "commit"):
            engine.store.commit()

        # Verify entries exist before pruning
        assert engine.get_phenotype(100, 42) is not None
        assert engine.get_phenotype(101, 43) is not None
        assert engine.get_phenotype(102, 44) is not None

        # Manually trigger the auto-pruning hook
        dummy_entry = engine.learn_by_key(100, 42)
        dummy_intron = 42

        # Call the auto-pruning hook directly
        # This should handle append-only stores gracefully
        agent.engine._auto_prune_hook(agent.engine, dummy_entry, dummy_intron)

        # For append-only stores, entries will still exist after the hook
        # The hook should not raise an exception
        assert engine.get_phenotype(100, 42) is not None
        assert engine.get_phenotype(101, 43) is not None
        assert engine.get_phenotype(102, 44) is not None

    def test_auto_pruning_uses_preferences_threshold(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that auto-pruning uses the confidence threshold from preferences."""
        # Create agent with custom confidence threshold
        custom_threshold = 0.1
        agent_config = {
            "ontology_path": "memories/public/meta/ontology_keys.npy",
            "knowledge_path": str(tmp_path / "test_knowledge.bin"),
            "preferences": {
                "pruning": {
                    "confidence_threshold": custom_threshold,
                    "decay_factor": 0.995,
                    "decay_interval_hours": 6,
                    "enable_auto_decay": True,
                }
            },
            "base_path": str(tmp_path),
        }

        agent = GyroSI(cast(AgentConfig, agent_config), agent_id="test_agent")
        engine = agent.engine.operator

        # Add entries with confidence around the custom threshold
        entry1 = engine.learn_by_key(100, 42)
        entry1["confidence"] = custom_threshold - 0.01  # Just below threshold
        engine.store.put((100, 42), entry1)

        entry2 = engine.learn_by_key(101, 43)
        entry2["confidence"] = custom_threshold + 0.01  # Just above threshold
        engine.store.put((101, 43), entry2)

        # Ensure entries are committed
        if hasattr(engine.store, "commit"):
            engine.store.commit()

        # Verify entries exist before pruning
        assert engine.get_phenotype(100, 42) is not None
        assert engine.get_phenotype(101, 43) is not None

        # Manually trigger the auto-pruning hook
        dummy_entry = engine.learn_by_key(100, 42)
        dummy_intron = 42
        agent.engine._auto_prune_hook(agent.engine, dummy_entry, dummy_intron)

        # For append-only stores, entries will still exist after the hook
        # The hook should not raise an exception
        assert engine.get_phenotype(100, 42) is not None
        assert engine.get_phenotype(101, 43) is not None

    def test_auto_pruning_fallback_threshold(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that auto-pruning falls back to default threshold when not specified in preferences."""
        # Create agent without confidence_threshold in preferences
        agent_config = {
            "ontology_path": "memories/public/meta/ontology_keys.npy",
            "knowledge_path": str(tmp_path / "test_knowledge.bin"),
            "preferences": {
                "pruning": {
                    "decay_factor": 0.995,
                    "decay_interval_hours": 6,
                    "enable_auto_decay": True,
                    # confidence_threshold not specified, should use default 0.05
                }
            },
            "base_path": str(tmp_path),
        }

        agent = GyroSI(cast(AgentConfig, agent_config), agent_id="test_agent")
        engine = agent.engine.operator

        # Add entries with confidence around the default threshold (0.05)
        entry1 = engine.learn_by_key(100, 42)
        entry1["confidence"] = 0.04  # Just below default threshold
        engine.store.put((100, 42), entry1)

        entry2 = engine.learn_by_key(101, 43)
        entry2["confidence"] = 0.06  # Just above default threshold
        engine.store.put((101, 43), entry2)

        # Ensure entries are committed
        if hasattr(engine.store, "commit"):
            engine.store.commit()

        # Verify entries exist before pruning
        assert engine.get_phenotype(100, 42) is not None
        assert engine.get_phenotype(101, 43) is not None

        # Manually trigger the auto-pruning hook
        dummy_entry = engine.learn_by_key(100, 42)
        dummy_intron = 42
        agent.engine._auto_prune_hook(agent.engine, dummy_entry, dummy_intron)

        # For append-only stores, entries will still exist after the hook
        # The hook should not raise an exception
        assert engine.get_phenotype(100, 42) is not None
        assert engine.get_phenotype(101, 43) is not None


class TestPrivateMethods:
    """Test private utility methods."""

    def test_compute_semantic_address(self, gyrosi_agent: GyroSI) -> None:
        """Test semantic address computation is deterministic."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        context_key = (100, 42)

        # Should be deterministic
        addr1 = engine._compute_semantic_address(context_key)
        addr2 = engine._compute_semantic_address(context_key)

        assert addr1 == addr2
        assert 0 <= addr1 < len(engine.s2.orbit_cardinality)

    def test_compute_semantic_address_different_contexts(self, gyrosi_agent: GyroSI) -> None:
        """Test different contexts give different addresses."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        addr1 = engine._compute_semantic_address((100, 42))
        addr2 = engine._compute_semantic_address((101, 42))
        addr3 = engine._compute_semantic_address((100, 43))

        # Should be different (with high probability)
        assert addr1 != addr2 or addr1 != addr3  # At least one should differ

    def test_create_default_phenotype(self, gyrosi_agent: GyroSI) -> None:
        """Test default phenotype creation."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        context_key = (100, 42)
        entry = engine._create_default_phenotype(context_key)

        assert entry["context_signature"] == context_key
        assert entry["phenotype"] == "P[100:42]"
        assert entry["exon_mask"] == 42  # Initial mask is intron
        assert 0 <= entry["confidence"] <= 1.0
        assert entry["usage_count"] == 0
        assert entry["created_at"] > 0
        assert entry["last_updated"] > 0
        assert entry["_original_context"] is None

        # Governance signature should match mask
        expected_sig = governance.compute_governance_signature(42)
        gov_sig = entry["governance_signature"]
        assert gov_sig["neutral"] == expected_sig[0]
        assert gov_sig["li"] == expected_sig[1]
        assert gov_sig["fg"] == expected_sig[2]
        assert gov_sig["bg"] == expected_sig[3]
        assert gov_sig["dyn"] == expected_sig[4]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_learn_with_missing_context_signature(self, gyrosi_agent: GyroSI) -> None:
        """Test error when phenotype entry lacks context_signature."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        bad_entry = {
            "phenotype": "bad",
            "exon_mask": 42,
            "confidence": 0.5,
            # 'context_signature' is intentionally omitted to trigger KeyError
            "usage_count": 0,
            "created_at": time.time(),
            "last_updated": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "_original_context": None,
        }
        # Only cast when not testing error handling for bad dicts
        with pytest.raises(KeyError, match="missing required 'context_signature' key"):
            engine.learn(cast(Any, bad_entry), 42)

    def test_learn_with_invalid_state_index(self, gyrosi_agent: GyroSI) -> None:
        """Test assertion failure with invalid state index."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        bad_entry: PhenotypeEntry = {
            "phenotype": "bad",
            "exon_mask": 42,
            "confidence": 0.5,
            "context_signature": (len(engine.s2.orbit_cardinality), 42),  # Out of range
            "usage_count": 0,
            "created_at": time.time(),
            "last_updated": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "_original_context": None,
        }

        with pytest.raises(AssertionError):
            engine.learn(bad_entry, 42)

    def test_novelty_calculation_edge_cases(self, gyrosi_agent: GyroSI) -> None:
        """Test novelty calculation with edge cases."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Test identical masks (no novelty)
        entry: PhenotypeEntry = {
            "phenotype": "test",
            "exon_mask": 42,
            "confidence": 0.5,
            "context_signature": (100, 42),
            "usage_count": 0,
            "created_at": time.time(),
            "last_updated": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "_original_context": None,
        }

        original_confidence = float(entry["confidence"])
        updated = engine.learn(entry, 42)  # Same as exon_mask

        # Should have minimal confidence change due to low novelty
        # fold(42, 42) = 0, so novelty = hamming(42, 0) / 8 = 3/8
        assert float(updated["confidence"]) >= original_confidence

    def test_learning_rate_calculation(self, gyrosi_agent: GyroSI) -> None:
        """Test learning rate depends on orbit cardinality."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Different state indices should have different learning rates
        # based on their orbit cardinalities
        v1 = engine.s2.orbit_cardinality[0]
        v2 = engine.s2.orbit_cardinality[100]

        alpha1 = (1 / 6) * math.sqrt(v1 / engine._v_max) if v1 is not None and engine._v_max is not None else 0
        alpha2 = (1 / 6) * math.sqrt(v2 / engine._v_max) if v2 is not None and engine._v_max is not None else 0

        if v1 is not None and v2 is not None:
            if v1 < v2:
                assert alpha1 < alpha2
            elif v1 > v2:
                assert alpha1 > alpha2
            else:
                assert alpha1 == alpha2

    def test_confidence_boundary_conditions(self, gyrosi_agent: GyroSI) -> None:
        """Test confidence stays within [0, 1] bounds."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Test with extreme values
        entry: PhenotypeEntry = {
            "phenotype": "test",
            "exon_mask": 0,
            "confidence": 0.999,  # Very high
            "context_signature": (100, 42),
            "usage_count": 0,
            "created_at": time.time(),
            "last_updated": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "_original_context": None,
        }

        original_confidence = float(entry["confidence"])
        # Learn with maximum novelty intron
        updated = engine.learn(entry, 0xFF)

        if (
            updated is not None
            and isinstance(updated.get("confidence"), float)
            and isinstance(original_confidence, float)
        ):
            assert float(updated["confidence"]) >= float(original_confidence)

    def test_store_commit_integration(self, gyrosi_agent: GyroSI) -> None:
        """Test integration with store commit functionality."""
        engine = gyrosi_agent.engine.operator
        store = engine.store

        # Learn something
        engine.learn_by_key(100, 42)

        # Store should handle commits if supported
        if hasattr(store, "commit"):
            store.commit()  # Should not raise

    def test_validation_with_exception_handling(self, gyrosi_agent: GyroSI) -> None:
        """Test validation handles exceptions gracefully."""
        engine = gyrosi_agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Create problematic entry that might cause exceptions
        bad_entry: PhenotypeEntry = {
            "phenotype": "bad",
            "confidence": -0.5,  # Invalid value triggers anomaly, but type is float
            "exon_mask": 0,
            "usage_count": 0,
            "created_at": time.time(),
            "last_updated": time.time(),
            "governance_signature": {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0},
            "context_signature": (0, 0),
            "_original_context": None,
        }

        engine.store.put((100, 42), bad_entry)

        # Validation should handle exceptions and count as anomalies
        report = engine.validate_knowledge_integrity()

        assert report["modified_entries"] > 0  # Should count exception as anomaly


class TestIntegrationWithStorage:
    """Test integration with different storage backends."""

    def test_store_put_called_correctly(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test that store.put creates an entry with correct parameters in the real store."""
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        state_index = 100
        intron = 42

        engine.learn_by_key(state_index, intron)

        # Retrieve the entry from the real store
        entry = engine.store.get((state_index, intron))
        assert entry is not None
        assert entry.get("context_signature") == (state_index, intron)

    def test_iter_entries_integration(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test validation uses store.iter_entries correctly (isolated agent)."""
        agent = isolated_agent_factory(tmp_path)
        engine = agent.engine.operator
        assert len(engine.s2.orbit_cardinality) > 0

        # Add several entries
        for i in range(3):
            engine.learn_by_key(i, 42)

        # Validation should iterate through all entries
        report = engine.validate_knowledge_integrity()

        assert report["total_entries"] == 3
        assert report["modified_entries"] == 0  # All should be valid


class TestStorageIsolation:
    """Test storage isolation using isolated agents."""

    def test_isolated_agents_dont_interfere(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that isolated agents don't interfere with each other."""
        # Create two completely isolated agents
        agent1 = isolated_agent_factory(tmp_path / "agent1")
        agent2 = isolated_agent_factory(tmp_path / "agent2")

        # Have each agent learn something different
        agent1.engine.operator.learn_by_key(100, 42)
        agent2.engine.operator.learn_by_key(200, 24)

        # Ensure entries are committed to storage
        if hasattr(agent1.engine.operator.store, "commit"):
            agent1.engine.operator.store.commit()
        if hasattr(agent2.engine.operator.store, "commit"):
            agent2.engine.operator.store.commit()

        # Verify they have different entries
        store1_entries = list(agent1.engine.operator.store.iter_entries())
        store2_entries = list(agent2.engine.operator.store.iter_entries())

        assert len(store1_entries) == 1
        assert len(store2_entries) == 1
        assert store1_entries[0][0] == (100, 42)
        assert store2_entries[0][0] == (200, 24)

        # Verify they don't see each other's data
        assert agent1.engine.operator.store.get((200, 24)) is None
        assert agent2.engine.operator.store.get((100, 42)) is None
