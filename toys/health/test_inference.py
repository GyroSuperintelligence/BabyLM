"""
Tests for the inference module functionality.
"""

import pytest
import math
from unittest.mock import MagicMock
import numpy as np

from baby.inference import InferenceEngine
from baby.governance import fold
from toys.health.conftest import assert_phenotype_entry_valid


@pytest.fixture
def mock_s2_engine():
    """Create a mock S2 engine for testing."""
    mock_engine = MagicMock()
    mock_engine.endogenous_modulus = 100
    mock_engine.orbit_cardinality = np.arange(1, 101, dtype=np.uint32)  # 1 to 100 inclusive
    mock_engine.is_canonical = lambda idx: 0 <= idx < 100
    return mock_engine


@pytest.fixture
def inference_engine(mock_s2_engine, orbit_store):
    """Create an inference engine for testing."""
    return InferenceEngine(mock_s2_engine, orbit_store)


class TestInferenceEngineInitialization:
    """Tests for InferenceEngine initialization."""

    def test_init_with_valid_params(self, mock_s2_engine, orbit_store):
        """Test initialization with valid parameters."""
        engine = InferenceEngine(mock_s2_engine, orbit_store)
        assert engine.s2 == mock_s2_engine
        assert engine.store == orbit_store
        assert engine.endogenous_modulus == mock_s2_engine.endogenous_modulus
        assert engine._v_max == 100  # Max of orbit_cardinality values

    def test_init_with_zero_cardinality(self, mock_s2_engine, orbit_store):
        """Test initialization fails with zero cardinality."""
        mock_s2_engine.orbit_cardinality = {i: 0 for i in range(100)}
        with pytest.raises(ValueError):
            InferenceEngine(mock_s2_engine, orbit_store)


class TestPhenotypeRetrieval:
    """Tests for phenotype retrieval and creation."""

    def test_get_phenotype_new(self, inference_engine):
        """Test getting a phenotype that doesn't exist yet."""
        state_index = 10
        intron = 42

        # Should not exist initially
        assert inference_engine.store.get((state_index, intron)) is None

        # Get the phenotype (should create a new one)
        entry = inference_engine.get_phenotype(state_index, intron)
        inference_engine.store.commit()  # Ensure persistence

        # Verify the entry was created with correct values
        assert entry["phenotype"] == f"P[{state_index}:{intron}]"
        assert entry["memory_mask"] == intron
        assert entry["context_signature"] == (state_index, intron)
        assert 0 < entry["confidence"] < 1.0  # Should be variety-weighted
        assert entry["usage_count"] == 0
        assert entry["age_counter"] == 0

        # Validate the entry
        assert_phenotype_entry_valid(entry)

        # Should now exist in the store
        assert inference_engine.store.get((state_index, intron)) is not None

    def test_get_phenotype_existing(self, inference_engine):
        """Test getting a phenotype that already exists."""
        state_index = 20
        intron = 123

        # Create an entry
        existing = inference_engine.get_phenotype(state_index, intron)
        inference_engine.store.commit()

        # Get it again and verify it's the same (compare relevant fields)
        retrieved = inference_engine.get_phenotype(state_index, intron)
        for key in [
            "phenotype",
            "memory_mask",
            "context_signature",
            "semantic_address",
            "usage_count",
            "age_counter",
            "confidence",
        ]:
            assert retrieved[key] == existing[key]

    def test_get_phenotype_clamps_intron(self, inference_engine):
        """Test that intron values are clamped to 8 bits."""
        state_index = 30
        intron = 300  # Greater than 255

        entry = inference_engine.get_phenotype(state_index, intron)
        assert entry["context_signature"] == (state_index, intron & 0xFF)
        assert entry["memory_mask"] == intron & 0xFF

    def test_get_phenotype_validates_state_index(self, inference_engine):
        """Test that state_index is validated."""
        # Invalid state index (negative)
        with pytest.raises(AssertionError):
            inference_engine.get_phenotype(-1, 42)

        # Invalid state index (too large)
        with pytest.raises(AssertionError):
            inference_engine.get_phenotype(100, 42)


class TestLearningFunctionality:
    """Tests for learning functionality."""

    def test_learn_changes_mask(self, inference_engine):
        """Test basic learning functionality."""
        state_index = 40
        intron1 = 100
        intron2 = 200

        # Get initial entry
        entry = inference_engine.get_phenotype(state_index, intron1)
        old_confidence = entry["confidence"]

        # Learn with a different intron
        inference_engine.learn(entry, intron2)

        # Verify mask and confidence changed
        assert entry["memory_mask"] == fold(intron1, intron2) & 0xFF
        assert entry["confidence"] > old_confidence
        assert entry["usage_count"] == 1
        assert entry["age_counter"] == 0

    def test_learn_no_change(self, inference_engine):
        """Test learning with the same intron (no change)."""
        state_index = 50
        intron = 150
        # Get initial entry
        entry = inference_engine.get_phenotype(state_index, intron)
        inference_engine.store.commit()  # Ensure the initial value is persisted
        old_confidence = entry["confidence"]
        old_usage_count = entry["usage_count"]
        # Learn with the same intron
        result = inference_engine.learn(entry, intron)
        inference_engine.store.commit()
        # Debug: print all keys and the entry at the expected key
        print("Store keys after learn_no_change:", list(inference_engine.store.data.keys()))
        print("Entry at (state_index, intron):", inference_engine.store.get((state_index, intron)))
        refreshed = inference_engine.store.get((state_index, intron))
        # The mask should be fold(intron, intron) & 0xFF
        expected_mask = fold(intron, intron) & 0xFF
        assert refreshed["memory_mask"] == expected_mask
        assert refreshed["confidence"] != old_confidence  # Should update
        assert refreshed["usage_count"] == old_usage_count + 1

    def test_learn_by_key(self, inference_engine):
        """Test learning by key (state_index, intron)."""
        state_index = 60
        intron = 180

        # Learn by key (should create and update)
        entry = inference_engine.learn_by_key(state_index, intron)
        inference_engine.store.commit()
        refreshed = inference_engine.store.get((state_index, intron))

        # The mask should be fold(intron, intron) & 0xFF
        expected_mask = fold(intron, intron) & 0xFF
        assert refreshed["context_signature"] == (state_index, intron)
        assert refreshed["memory_mask"] == expected_mask
        assert refreshed["usage_count"] == 1  # Should increment usage_count

    def test_batch_learn_empty(self, inference_engine):
        """Test batch learning with empty intron list."""
        result = inference_engine.batch_learn(70, [])
        assert result is None

    def test_batch_learn_single(self, inference_engine):
        """Test batch learning with a single intron."""
        state_index = 70
        intron = 210

        result = inference_engine.batch_learn(state_index, [intron])
        inference_engine.store.commit()
        refreshed = inference_engine.store.get((state_index, intron))

        # The mask should be fold(intron, intron) & 0xFF
        expected_mask = fold(intron, intron) & 0xFF
        assert refreshed["context_signature"] == (state_index, intron)
        assert refreshed["memory_mask"] == expected_mask

    def test_batch_learn_multiple(self, inference_engine, generate_test_introns):
        """Test batch learning with multiple introns."""
        state_index = 80
        introns = generate_test_introns(4)
        # Calculate the expected result (path-dependent reduction)
        expected_mask = introns[0] & 0xFF
        for i in introns[1:]:
            expected_mask = fold(expected_mask, i & 0xFF) & 0xFF
        result = inference_engine.batch_learn(state_index, introns)
        inference_engine.store.commit()
        # Debug: print all keys and the entry at the expected key
        print("Store keys after batch_learn_multiple:", list(inference_engine.store.data.keys()))
        print("Entry at (state_index, expected_mask):", inference_engine.store.get((state_index, expected_mask)))
        refreshed = inference_engine.store.get((state_index, expected_mask))
        # The final memory mask inside the entry will be fold(expected_mask, expected_mask) & 0xFF
        final_mask = fold(expected_mask, expected_mask) & 0xFF
        assert refreshed["context_signature"] == (state_index, expected_mask)
        assert refreshed["memory_mask"] == final_mask
        assert refreshed["usage_count"] == 1


class TestDefaultPhenotypeCreation:
    """Tests for default phenotype creation."""

    def test_create_default_phenotype(self, inference_engine, mock_time):
        """Test creation of default phenotype entries."""
        state_index = 90
        intron = 240
        context_key = (state_index, intron)
        semantic_address = 12345

        # Create default phenotype
        entry = inference_engine._create_default_phenotype(context_key, semantic_address)

        # Verify all fields
        assert entry["phenotype"] == f"P[{state_index}:{intron}]"
        assert entry["memory_mask"] == intron
        assert entry["context_signature"] == context_key
        assert entry["semantic_address"] == semantic_address
        assert entry["usage_count"] == 0
        assert entry["age_counter"] == 0
        assert entry["created_at"] == mock_time.current
        assert entry["last_updated"] == mock_time.current

        # Verify variety-weighted confidence
        v = inference_engine.s2.orbit_cardinality[state_index]
        expected_conf = (1 / 6) * math.sqrt(v / inference_engine._v_max)
        assert entry["confidence"] == expected_conf

        # Validate the entry
        assert_phenotype_entry_valid(entry)


class TestKnowledgeIntegrityAndMaintenance:
    """Tests for knowledge integrity validation and maintenance."""

    def test_validate_knowledge_integrity(self, inference_engine):
        """Test validation with valid entries."""
        # Add some valid entries
        for i in range(5):
            entry = inference_engine.get_phenotype(i, i)
            inference_engine.learn(entry, i + 1)
        inference_engine.store.commit()
        report = inference_engine.validate_knowledge_integrity()

        assert report["total_entries"] == 5
        assert report["average_confidence"] > 0
        assert report["modified_entries"] == 0

    def test_apply_confidence_decay(self, inference_engine):
        """Test confidence decay."""
        # Add some entries
        entries = []
        for i in range(3):
            entry = inference_engine.get_phenotype(i, i)
            entries.append(entry)
        inference_engine.store.commit()
        # Store original confidences
        original_confidences = [entry["confidence"] for entry in entries]

        # Apply decay
        report = inference_engine.apply_confidence_decay(decay_factor=0.1)

        assert report["modified_entries"] == 3

        # Verify all entries were decayed and age incremented
        for i, entry in enumerate(entries):
            refreshed = inference_engine.store.get((i, i))
            assert refreshed["confidence"] < original_confidences[i]
            assert refreshed["age_counter"] == 1

    def test_prune_low_confidence_entries(self, inference_engine):
        """Test pruning low confidence entries."""
        # Add some entries with varying confidence
        state_indices = [10, 11, 12]

        # Create entries
        for i, state_index in enumerate(state_indices):
            entry = inference_engine.get_phenotype(state_index, 0)
            # Directly modify confidence
            entry["confidence"] = 0.01 if i == 0 else 0.1 if i == 1 else 0.2
            inference_engine.store.put(entry["context_signature"], entry)
        inference_engine.store.commit()
        # Prune entries below 0.05
        removed = inference_engine.prune_low_confidence_entries(confidence_threshold=0.05)

        assert removed == 1
        assert inference_engine.store.get((state_indices[0], 0)) is None
        assert inference_engine.store.get((state_indices[1], 0)) is not None
        assert inference_engine.store.get((state_indices[2], 0)) is not None


class TestAddressComputation:
    """Tests for semantic address computation."""

    def test_compute_semantic_address(self, inference_engine):
        """Test semantic address computation."""
        # Test deterministic generation
        addr1 = inference_engine._compute_semantic_address((10, 20))
        addr2 = inference_engine._compute_semantic_address((10, 20))
        assert addr1 == addr2

        # Test different contexts yield different addresses
        addr3 = inference_engine._compute_semantic_address((10, 21))
        assert addr1 != addr3

        # Test address is within modulus range
        assert 0 <= addr1 < inference_engine.endogenous_modulus


class TestKnowledgeStatistics:
    """Tests for knowledge statistics."""

    def test_get_knowledge_statistics(self, inference_engine):
        """Test knowledge statistics."""
        # Add some entries
        for i in range(5):
            entry = inference_engine.get_phenotype(i, i)
            inference_engine.learn(entry, i + 1)
        inference_engine.store.commit()
        stats = inference_engine.get_knowledge_statistics()

        assert stats["total_entries"] == 5
        assert 0 <= stats["average_confidence"] <= 1.0
        assert 0 <= stats["memory_utilization"] <= 1.0
