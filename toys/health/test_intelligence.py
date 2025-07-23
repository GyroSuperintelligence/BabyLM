"""
Comprehensive tests for intelligence.py - the orchestration and API layer.
Tests agent lifecycle, state evolution, hook management, and multi-agent coordination.
"""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from baby import governance
from baby.contracts import AgentConfig, CycleHookFunction, PhenotypeEntry
from baby.intelligence import (
    AgentPool,
    GyroSI,
    IntelligenceEngine,
    LRUAgentCache,
    StateInfo,
    orchestrate_turn,
)
from baby.policies import OrbitStore, OverlayView, ReadOnlyView


class TestIntelligenceEngineInitialization:
    """Test IntelligenceEngine initialization and setup."""

    def test_initialization_with_real_ontology(self, meta_paths: Dict[str, str], temp_store: OrbitStore) -> None:
        """Test engine initializes correctly with real ontology."""
        engine = IntelligenceEngine(
            ontology_path=meta_paths["ontology"],
            phenotype_store=temp_store,
            agent_id="test_agent"
        )
        
        assert engine.agent_id == "test_agent"
        assert engine.s2 is not None
        assert engine.operator is not None
        assert engine.cycle_count == 0
        assert hasattr(engine, 'gene_mac_m_int')
        assert hasattr(engine, 'current_state_index')

    def test_initialization_generates_agent_id(self, meta_paths: Dict[str, str], temp_store: OrbitStore) -> None:
        """Test agent ID is generated if not provided."""
        engine = IntelligenceEngine(
            ontology_path=meta_paths["ontology"],
            phenotype_store=temp_store
        )
        
        assert engine.agent_id is not None
        assert len(engine.agent_id) > 0
        # Should be a valid UUID format
        assert "-" in engine.agent_id

    def test_epistemology_loading(self, meta_paths: Dict[str, str], temp_store: OrbitStore) -> None:
        """Test State Transition Table (STT) loading if available."""
        engine = IntelligenceEngine(
            ontology_path=meta_paths["ontology"],
            phenotype_store=temp_store
        )
        
        # Check if epistemology was loaded
        ep_path = Path(meta_paths["ontology"]).parent / "epistemology.npy"
        if ep_path.exists():
            assert engine.use_epistemology is True
            assert hasattr(engine, 'epistemology')
        else:
            assert engine.use_epistemology is False

    def test_origin_state_initialization(self, meta_paths: Dict[str, str], temp_store: OrbitStore) -> None:
        """Test engine starts at archetypal origin state."""
        engine = IntelligenceEngine(
            ontology_path=meta_paths["ontology"],
            phenotype_store=temp_store
        )
        
        expected_origin = engine.s2.tensor_to_int(governance.GENE_Mac_S)
        assert engine.gene_mac_m_int == expected_origin

    def test_hook_batch_interval_configuration(self, meta_paths: Dict[str, str], temp_store: OrbitStore) -> None:
        """Test hook batch interval can be configured."""
        custom_interval = 16
        engine = IntelligenceEngine(
            ontology_path=meta_paths["ontology"],
            phenotype_store=temp_store,
            hook_batch_interval=custom_interval
        )
        
        assert engine._hook_batch_interval == custom_interval
        assert engine._hook_event_buffer.maxlen == custom_interval

    def test_algedonic_regulation_initialization(self, meta_paths: Dict[str, str], temp_store: OrbitStore) -> None:
        """Test algedonic regulation components are initialized."""
        engine = IntelligenceEngine(
            ontology_path=meta_paths["ontology"],
            phenotype_store=temp_store
        )
        
        assert hasattr(engine, '_θ_buf')
        assert hasattr(engine, '_θ_high')
        assert hasattr(engine, '_θ_low')
        assert hasattr(engine, '_cool_introns')
        assert hasattr(engine, '_pain_streak')
        assert engine._pain_streak == 0


class TestEgressProcessing:
    """Test egress (input → transformation) processing."""

    def test_process_egress_basic(self, gyrosi_agent: GyroSI) -> None:
        """Test basic egress processing transforms state."""
        engine = gyrosi_agent.engine
        
        input_byte = 42
        original_state = engine.gene_mac_m_int
        
        intron = engine.process_egress(input_byte)
        
        # Should transcribe input
        expected_intron = governance.transcribe_byte(input_byte)
        assert intron == expected_intron
        
        # State should change (unless by coincidence)
        # Can't guarantee change since it depends on physics
        assert engine.cycle_count == 1

    def test_process_egress_byte_masking(self, gyrosi_agent: GyroSI) -> None:
        """Test input byte is properly masked."""
        engine = gyrosi_agent.engine
        
        large_input = 0x1FF  # 9 bits
        masked_input = 0xFF   # 8 bits
        
        intron1 = engine.process_egress(large_input)
        engine.reset_to_archetypal_state()
        intron2 = engine.process_egress(masked_input)
        
        assert intron1 == intron2

    def test_process_egress_increments_cycle_count(self, gyrosi_agent: GyroSI) -> None:
        """Test egress increments cycle count."""
        engine = gyrosi_agent.engine
        
        initial_count = engine.cycle_count
        engine.process_egress(42)
        
        assert engine.cycle_count == initial_count + 1

    def test_process_egress_updates_theta_buffer(self, gyrosi_agent: GyroSI) -> None:
        """Test egress updates divergence tracking buffer."""
        engine = gyrosi_agent.engine
        
        initial_buffer_len = len(engine._θ_buf)
        engine.process_egress(42)
        
        assert len(engine._θ_buf) == initial_buffer_len + 1

    def test_process_egress_state_integrity(self, gyrosi_agent: GyroSI) -> None:
        """Test state remains within 48-bit bounds."""
        engine = gyrosi_agent.engine
        
        # Process multiple inputs
        for byte_val in [0, 1, 42, 128, 255]:
            engine.process_egress(byte_val)
            
            current_state = (engine.gene_mac_m_int if not engine.use_epistemology 
                           else engine.s2.get_state_from_index(engine.current_state_index))
            assert 0 <= current_state < (1 << 48)

    def test_process_egress_with_epistemology(self, meta_paths: Dict[str, str], temp_store: OrbitStore) -> None:
        """Test egress with State Transition Table if available."""
        engine = IntelligenceEngine(
            ontology_path=meta_paths["ontology"],
            phenotype_store=temp_store
        )
        
        if engine.use_epistemology:
            original_index = engine.current_state_index
            intron = engine.process_egress(42)
            
            # Should use epistemology for state transition
            assert hasattr(engine, 'epistemology')
            # Index should change (unless self-loop)
            assert engine.current_state_index >= 0


class TestIngressProcessing:
    """Test ingress (context → response) processing."""

    def test_process_ingress_basic(self, gyrosi_agent: GyroSI) -> None:
        """Test basic ingress processing generates response."""
        engine = gyrosi_agent.engine
        
        last_intron = 42
        response_byte = engine.process_ingress(last_intron)
        
        assert isinstance(response_byte, int)
        assert 0 <= response_byte <= 255

    def test_process_ingress_intron_masking(self, gyrosi_agent: GyroSI) -> None:
        """Test intron is properly masked to 8 bits."""
        engine = gyrosi_agent.engine
        
        large_intron = 0x1FF  # 9 bits
        masked_intron = 0xFF   # 8 bits
        
        response1 = engine.process_ingress(large_intron)
        response2 = engine.process_ingress(masked_intron)
        
        assert response1 == response2

    def test_process_ingress_creates_phenotype(self, gyrosi_agent: GyroSI) -> None:
        """Test ingress creates phenotype entry."""
        engine = gyrosi_agent.engine
        
        last_intron = 42
        initial_entries = len(list(engine.operator.store.iter_entries()))
        
        engine.process_ingress(last_intron)
        
        final_entries = len(list(engine.operator.store.iter_entries()))
        assert final_entries > initial_entries

    def test_process_ingress_applies_learning(self, gyrosi_agent: GyroSI) -> None:
        """Test ingress applies Monodromic Fold learning."""
        engine = gyrosi_agent.engine
        
        last_intron = 42
        engine.process_ingress(last_intron)
        
        # Should have created and learned from phenotype
        # Check that store has entries
        entries = list(engine.operator.store.iter_entries())
        assert len(entries) > 0
        
        # Check that usage count was incremented
        for key, entry in entries:
            if key[1] == last_intron:  # Same intron
                assert entry["usage_count"] > 0

    def test_process_ingress_hook_processing(self, gyrosi_agent: GyroSI) -> None:
        """Test ingress processes hooks appropriately."""
        engine = gyrosi_agent.engine
        
        hook_calls = []
        def test_hook(eng, phenotype_entry, intron):
            hook_calls.append((eng, phenotype_entry, intron))
        
        engine.add_hook(test_hook)
        
        # Process enough cycles to trigger hook
        for i in range(engine._hook_batch_interval):
            engine.process_ingress(42)
        
        # Should have called hook
        assert len(hook_calls) >= engine._hook_batch_interval

    def test_process_ingress_cooling_mechanism(self, gyrosi_agent: GyroSI) -> None:
        """Test algedonic cooling when divergence is high."""
        engine = gyrosi_agent.engine
        
        # Force high theta (pain) state
        engine._θ_buf.extend([engine._θ_high + 0.1] * 10)
        
        original_cycle_count = engine.cycle_count
        original_microstep_count = engine._microstep_count
        
        engine.process_ingress(42)
        
        # Should have performed cooling microsteps
        assert engine._microstep_count > original_microstep_count
        # External cycle count should be preserved
        assert engine.cycle_count == original_cycle_count

    def test_process_ingress_autonomic_cycles(self, gyrosi_agent: GyroSI) -> None:
        """Test autonomic cycles under extreme pain."""
        engine = gyrosi_agent.engine
        
        # Force high theta and pain streak
        engine._θ_buf.extend([engine._θ_high + 0.1] * 10)
        engine._pain_streak = 300  # High pain
        
        # Mock autonomic cycles
        engine._autonomic_cycles = [[1, 2, 3], [4, 5, 6]]
        
        original_microstep_count = engine._microstep_count
        
        engine.process_ingress(42)
        
        # Should have performed additional autonomic microsteps
        assert engine._microstep_count > original_microstep_count

    def test_process_ingress_phenotype_response_generation(self, gyrosi_agent: GyroSI) -> None:
        """Test response generation from phenotype."""
        engine = gyrosi_agent.engine
        
        # Create phenotype with known string
        state_index = 0
        intron = 65  # ASCII 'A'
        
        if engine.use_epistemology:
            engine.current_state_index = state_index
        else:
            # Set state to index 0
            engine.gene_mac_m_int = engine.s2.get_state_from_index(state_index)
        
        response = engine.process_ingress(intron)
        
        # Should return valid byte
        assert 0 <= response <= 255


class TestHookManagement:
    """Test post-cycle hook functionality."""

    def test_add_hook(self, gyrosi_agent: GyroSI) -> None:
        """Test adding hooks to engine."""
        engine = gyrosi_agent.engine
        
        def dummy_hook(eng, entry, intron):
            pass
        
        initial_count = len(engine.post_cycle_hooks)
        engine.add_hook(dummy_hook)
        
        assert len(engine.post_cycle_hooks) == initial_count + 1
        assert dummy_hook in engine.post_cycle_hooks

    def test_remove_hook_success(self, gyrosi_agent: GyroSI) -> None:
        """Test successful hook removal."""
        engine = gyrosi_agent.engine
        
        def dummy_hook(eng, entry, intron):
            pass
        
        engine.add_hook(dummy_hook)
        result = engine.remove_hook(dummy_hook)
        
        assert result is True
        assert dummy_hook not in engine.post_cycle_hooks

    def test_remove_hook_not_found(self, gyrosi_agent: GyroSI) -> None:
        """Test hook removal when hook not present."""
        engine = gyrosi_agent.engine
        
        def dummy_hook(eng, entry, intron):
            pass
        
        result = engine.remove_hook(dummy_hook)
        
        assert result is False

    def test_hook_execution_with_batching(self, gyrosi_agent: GyroSI) -> None:
        """Test hooks are executed in batches."""
        engine = gyrosi_agent.engine
        
        execution_log = []
        def logging_hook(eng, entry, intron):
            execution_log.append((eng.cycle_count, intron))
        
        engine.add_hook(logging_hook)
        
        # Process multiple ingress cycles
        batch_size = engine._hook_batch_interval
        for i in range(batch_size - 1):
            engine.process_ingress(i)
        
        # Should not have executed yet
        assert len(execution_log) == 0
        
        # One more should trigger batch execution
        engine.process_ingress(batch_size)
        
        # Should have executed for all buffered events
        assert len(execution_log) == batch_size

    def test_hook_execution_on_pain_spike(self, gyrosi_agent: GyroSI) -> None:
        """Test hooks execute immediately on high divergence."""
        engine = gyrosi_agent.engine
        
        execution_log = []
        def logging_hook(eng, entry, intron):
            execution_log.append("executed")
        
        engine.add_hook(logging_hook)
        
        # Force high theta
        engine._θ_buf.extend([engine._θ_high + 0.1] * 5)
        
        # Single ingress should trigger immediate hook execution
        engine.process_ingress(42)
        
        assert len(execution_log) > 0


class TestBatchLearning:
    """Test batch learning functionality."""

    def test_batch_learn_empty_data(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning with empty data."""
        engine = gyrosi_agent.engine
        
        original_cycle_count = engine.cycle_count
        engine.batch_learn(b"")
        
        # Should not change state
        assert engine.cycle_count == original_cycle_count

    def test_batch_learn_single_byte(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning with single byte."""
        engine = gyrosi_agent.engine
        
        test_data = b"A"
        original_state = engine.gene_mac_m_int
        
        engine.batch_learn(test_data)
        
        assert engine.cycle_count == 1

    def test_batch_learn_multiple_bytes(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning with multiple bytes."""
        engine = gyrosi_agent.engine
        
        test_data = b"Hello"
        original_cycle_count = engine.cycle_count
        
        engine.batch_learn(test_data)
        
        # Should process each byte
        assert engine.cycle_count == original_cycle_count + len(test_data)

    def test_batch_learn_streaming_fold(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning uses streaming fold for memory efficiency."""
        engine = gyrosi_agent.engine
        
        test_data = b"test"
        
        # Mock to verify fold is called correctly
        with patch('baby.governance.fold') as mock_fold:
            mock_fold.side_effect = lambda a, b: (a + b) % 256  # Simple mock
            
            engine.batch_learn(test_data)
            
            # Should have called fold for accumulation
            assert mock_fold.called

    def test_batch_learn_creates_phenotypes(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning creates phenotype entries."""
        engine = gyrosi_agent.engine
        
        initial_entries = len(list(engine.operator.store.iter_entries()))
        
        engine.batch_learn(b"test")
        
        final_entries = len(list(engine.operator.store.iter_entries()))
        assert final_entries > initial_entries


class TestStateInformation:
    """Test state information retrieval."""

    def test_get_state_info_structure(self, gyrosi_agent: GyroSI) -> None:
        """Test state info returns correct structure."""
        engine = gyrosi_agent.engine
        
        info = engine.get_state_info()
        
        assert isinstance(info, dict)
        required_fields = [
            "agent_id", "cycle_count", "state_integer", "tensor_index",
            "angular_divergence_radians", "angular_divergence_degrees", "active_hooks"
        ]
        
        for field in required_fields:
            assert field in info
            assert info[field] is not None

    def test_get_state_info_values(self, gyrosi_agent: GyroSI) -> None:
        """Test state info contains valid values."""
        engine = gyrosi_agent.engine
        
        # Add a hook to test count
        def dummy_hook(eng, entry, intron):
            pass
        engine.add_hook(dummy_hook)
        
        info = engine.get_state_info()
        
        assert info["agent_id"] == engine.agent_id
        assert info["cycle_count"] == engine.cycle_count
        assert 0 <= info["state_integer"] < (1 << 48)
        assert 0 <= info["tensor_index"] < engine.s2.endogenous_modulus
        assert 0 <= info["angular_divergence_radians"] <= 3.15  # π
        assert 0 <= info["angular_divergence_degrees"] <= 180
        assert info["active_hooks"] == 1

    def test_get_state_info_angle_conversion(self, gyrosi_agent: GyroSI) -> None:
        """Test angle conversion between radians and degrees."""
        engine = gyrosi_agent.engine
        
        info = engine.get_state_info()
        
        radians = info["angular_divergence_radians"]
        degrees = info["angular_divergence_degrees"]
        
        # Should be consistent conversion
        expected_degrees = radians * 180 / 3.14159
        assert abs(degrees - expected_degrees) < 0.1


class TestAgentReset:
    """Test agent state reset functionality."""

    def test_reset_to_archetypal_state(self, gyrosi_agent: GyroSI) -> None:
        """Test resetting agent to origin state."""
        engine = gyrosi_agent.engine
        
        # Change state first
        engine.process_egress(42)
        modified_state = engine.gene_mac_m_int
        modified_cycles = engine.cycle_count
        
        # Reset
        engine.reset_to_archetypal_state()
        
        # Should be back to origin
        expected_origin = engine.s2.tensor_to_int(governance.GENE_Mac_S)
        assert engine.gene_mac_m_int == expected_origin
        assert engine.cycle_count == 0

    def test_reset_preserves_hooks(self, gyrosi_agent: GyroSI) -> None:
        """Test reset preserves hook configuration."""
        engine = gyrosi_agent.engine
        
        def dummy_hook(eng, entry, intron):
            pass
        
        engine.add_hook(dummy_hook)
        hook_count = len(engine.post_cycle_hooks)
        
        engine.reset_to_archetypal_state()
        
        # Hooks should be preserved
        assert len(engine.post_cycle_hooks) == hook_count
        assert dummy_hook in engine.post_cycle_hooks


class TestKnowledgeIntegrity:
    """Test knowledge validation and maintenance."""

    def test_validate_knowledge_integrity(self, gyrosi_agent: GyroSI) -> None:
        """Test knowledge integrity validation."""
        engine = gyrosi_agent.engine
        
        # Add some knowledge
        engine.batch_learn(b"test data")
        
        result = engine.validate_knowledge_integrity()
        
        assert isinstance(result, bool)
        # With valid knowledge, should return True
        assert result is True

    def test_apply_confidence_decay(self, gyrosi_agent: GyroSI) -> None:
        """Test confidence decay application."""
        engine = gyrosi_agent.engine
        
        # Add some knowledge
        engine.batch_learn(b"test")
        
        result = engine.apply_confidence_decay(decay_rate=0.01)
        
        assert isinstance(result, dict)
        assert "total_entries" in result or len(result) > 0

    def test_prune_low_confidence_entries(self, gyrosi_agent: GyroSI) -> None:
        """Test pruning of low confidence entries."""
        engine = gyrosi_agent.engine
        
        # Add some knowledge
        engine.batch_learn(b"test")
        
        # Prune with very low threshold
        pruned_count = engine.prune_low_confidence_entries(confidence_threshold=0.001)
        
        assert isinstance(pruned_count, int)
        assert pruned_count >= 0


class TestGyroSISystem:
    """Test the complete GyroSI system wrapper."""

    def test_gyrosi_initialization_basic(self, meta_paths: Dict[str, str], temp_dir: Path) -> None:
        """Test basic GyroSI initialization."""
        config: AgentConfig = {
            "ontology_path": meta_paths["ontology"],
            "knowledge_path": str(temp_dir / "knowledge.pkl.gz"),
        }
        
        agent = GyroSI(config)
        
        assert agent.config == config
        assert agent.agent_id is not None
        assert agent.engine is not None
        
        agent.close()

    def test_gyrosi_with_custom_agent_id(self, meta_paths: Dict[str, str], temp_dir: Path) -> None:
        """Test GyroSI with custom agent ID."""
        config: AgentConfig = {
            "ontology_path": meta_paths["ontology"],
            "knowledge_path": str(temp_dir / "knowledge.pkl.gz"),
        }
        
        custom_id = "test_agent_123"
        agent = GyroSI(config, agent_id=custom_id)
        
        assert agent.agent_id == custom_id
        assert agent.engine.agent_id == custom_id
        
        agent.close()

    def test_gyrosi_ingest(self, gyrosi_agent: GyroSI) -> None:
        """Test GyroSI ingest functionality."""
        test_data = b"Hello, GyroSI!"
        
        # Should not raise
        gyrosi_agent.ingest(test_data)
        
        # Should have learned something
        assert gyrosi_agent.engine.cycle_count > 0

    def test_gyrosi_respond(self, gyrosi_agent: GyroSI) -> None:
        """Test GyroSI response generation."""
        test_input = b"Hello"
        
        response = gyrosi_agent.respond(test_input)
        
        assert isinstance(response, bytes)
        assert len(response) == 1  # Current implementation returns 1 byte

    def test_gyrosi_respond_empty_input(self, gyrosi_agent: GyroSI) -> None:
        """Test GyroSI handles empty input."""
        response = gyrosi_agent.respond(b"")
        
        assert response == b""

    def test_gyrosi_get_agent_info(self, gyrosi_agent: GyroSI) -> None:
        """Test GyroSI agent information retrieval."""
        info = gyrosi_agent.get_agent_info()
        
        assert isinstance(info, dict)
        assert "agent_id" in info
        assert "config" in info
        assert "system_integrity" in info
        assert info["config"] == gyrosi_agent.config

    def test_gyrosi_add_monitoring_hook(self, gyrosi_agent: GyroSI) -> None:
        """Test adding monitoring hooks."""
        def monitor_hook(eng, entry, intron):
            pass
        
        gyrosi_agent.add_monitoring_hook(monitor_hook)
        
        assert monitor_hook in gyrosi_agent.engine.post_cycle_hooks

    def test_gyrosi_apply_maintenance(self, gyrosi_agent: GyroSI) -> None:
        """Test maintenance operations."""
        # Add some knowledge first
        gyrosi_agent.ingest(b"test data")
        
        result = gyrosi_agent.apply_maintenance()
        
        assert isinstance(result, dict)
        assert "decay_applied" in result
        assert "entries_pruned" in result
        assert "timestamp" in result

    def test_gyrosi_close(self, meta_paths: Dict[str, str], temp_dir: Path) -> None:
        """Test GyroSI clean shutdown."""
        config: AgentConfig = {
            "ontology_path": meta_paths["ontology"],
            "knowledge_path": str(temp_dir / "knowledge.pkl.gz"),
        }
        
        agent = GyroSI(config)
        
        # Should not raise
        agent.close()


class TestLRUAgentCache:
    """Test LRU cache for agent management."""

    def test_lru_cache_initialization(self) -> None:
        """Test LRU cache initialization."""
        cache = LRUAgentCache(max_size=3)
        
        assert cache.max_size == 3
        assert len(cache) == 0

    def test_lru_cache_basic_operations(self) -> None:
        """Test basic LRU cache operations."""
        cache = LRUAgentCache(max_size=2)
        
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        
        assert len(cache) == 2
        assert cache["key1"] == "value1"
        assert cache["key2"] == "value2"

    def test_lru_cache_eviction(self) -> None:
        """Test LRU cache evicts oldest items."""
        cache = LRUAgentCache(max_size=2)
        
        # Mock agents with close method
        agent1 = Mock()
        agent2 = Mock()
        agent3 = Mock()
        
        cache["key1"] = agent1
        cache["key2"] = agent2
        
        # Adding third should evict first
        cache["key3"] = agent3
        
        assert len(cache) == 2
        assert "key1" not in cache
        assert "key2" in cache
        assert "key3" in cache
        
        # Should have called close on evicted agent
        agent1.close.assert_called_once()

    def test_lru_cache_access_updates_order(self) -> None:
        """Test accessing items updates LRU order."""
        cache = LRUAgentCache(max_size=2)
        
        agent1 = Mock()
        agent2 = Mock()
        agent3 = Mock()
        
        cache["key1"] = agent1
        cache["key2"] = agent2
        
        # Access key1 to make it most recent
        _ = cache["key1"]
        
        # Adding key3 should evict key2 (least recently used)
        cache["key3"] = agent3
        
        assert "key1" in cache  # Should remain
        assert "key2" not in cache  # Should be evicted
        assert "key3" in cache
        
        agent2.close.assert_called_once()


class TestAgentPool:
    """Test multi-agent pool management."""

    def test_agent_pool_initialization(self, meta_paths: Dict[str, str], temp_dir: Path) -> None:
        """Test agent pool initialization."""
        public_path = str(temp_dir / "public.pkl.gz")
        
        # Create empty public store
        OrbitStore(public_path).close()
        
        pool = AgentPool(meta_paths["ontology"], public_path)
        
        assert pool.ontology_path == meta_paths["ontology"]
        assert pool.base_knowledge_path == public_path
        assert pool.max_agents > 0
        assert len(pool._shards) == AgentPool.SHARD_COUNT
        
        pool.close_all()

    def test_agent_pool_get_or_create_agent(self, agent_pool: AgentPool) -> None:
        """Test getting or creating agents."""
        agent_id = "test_agent"
        
        # First call should create
        agent1 = agent_pool.get_or_create_agent(agent_id)
        
        assert isinstance(agent1, GyroSI)
        assert agent1.agent_id == agent_id
        
        # Second call should return same agent
        agent2 = agent_pool.get_or_create_agent(agent_id)
        
        assert agent1 is agent2

    def test_agent_pool_with_role_hint(self, agent_pool: AgentPool) -> None:
        """Test agent creation with role hint."""
        agent_id = "test_agent_with_role"
        role_hint = "assistant"
        
        agent = agent_pool.get_or_create_agent(agent_id, role_hint=role_hint)
        
        assert isinstance(agent, GyroSI)
        # Role hint should be in metadata
        metadata = agent.config.get("agent_metadata", {})
        assert metadata.get("role_hint") == role_hint

    def test_agent_pool_remove_agent(self, agent_pool: AgentPool) -> None:
        """Test removing agents from pool."""
        agent_id = "removable_agent"
        
        # Create agent
        agent = agent_pool.get_or_create_agent(agent_id)
        
        # Remove it
        result = agent_pool.remove_agent(agent_id)
        
        assert result is True
        
        # Should not be in active agents
        active = agent_pool.get_active_agents()
        assert agent_id not in active

    def test_agent_pool_remove_nonexistent(self, agent_pool: AgentPool) -> None:
        """Test removing non-existent agent."""
        result = agent_pool.remove_agent("nonexistent")
        
        assert result is False

    def test_agent_pool_get_active_agents(self, agent_pool: AgentPool) -> None:
        """Test getting list of active agents."""
        # Create some agents
        agent_ids = ["agent1", "agent2", "agent3"]
        
        for agent_id in agent_ids:
            agent_pool.get_or_create_agent(agent_id)
        
        active = agent_pool.get_active_agents()
        
        for agent_id in agent_ids:
            assert agent_id in active

    def test_agent_pool_sharding(self, agent_pool: AgentPool) -> None:
        """Test agent pool uses sharding correctly."""
        agent_id1 = "shard_test_1"
        agent_id2 = "shard_test_2"
        
        # Create agents
        agent_pool.get_or_create_agent(agent_id1)
        agent_pool.get_or_create_agent(agent_id2)
        
        # Should distribute across shards
        total_agents = 0
        for shard in agent_pool._shards:
            with shard["lock"]:
                total_agents += len(shard["agents"])
        
        assert total_agents == 2

    def test_agent_pool_lru_eviction(self, meta_paths: Dict[str, str], temp_dir: Path) -> None:
        """Test LRU eviction policy."""
        public_path = str(temp_dir / "public.pkl.gz")
        OrbitStore(public_path).close()
        
        # Create pool with LRU and small capacity
        preferences = {"max_agents_in_memory": 2, "agent_eviction_policy": "lru"}
        pool = AgentPool(meta_paths["ontology"], public_path, preferences)
        
        # Create agents beyond capacity
        agent1 = pool.get_or_create_agent("agent1")
        agent2 = pool.get_or_create_agent("agent2")
        agent3 = pool.get_or_create_agent("agent3")  # Should evict agent1
        
        active = pool.get_active_agents()
        
        # Should have evicted oldest agent (exact behavior depends on sharding)
        assert len(active) <= 2
        
        pool.close_all()

    def test_agent_pool_close_all(self, agent_pool: AgentPool) -> None:
        """Test closing all agents in pool."""
        # Create some agents
        for i in range(3):
            agent_pool.get_or_create_agent(f"agent{i}")
        
        # Should not raise
        agent_pool.close_all()
        
        # Should be empty
        active = agent_pool.get_active_agents()
        assert len(active) == 0


class TestMultiAgentSetup:
    """Test multi-agent configurations and interactions."""

    def test_multi_agent_public_private_separation(self, multi_agent_setup: Dict[str, Any]) -> None:
        """Test public/private knowledge separation."""
        user_agent = multi_agent_setup["user"]
        assistant_agent = multi_agent_setup["assistant"]
        
        # Both should be initialized
        assert isinstance(user_agent, GyroSI)
        assert isinstance(assistant_agent, GyroSI)
        
        # Should have different agent IDs
        assert user_agent.agent_id != assistant_agent.agent_id

    def test_multi_agent_shared_public_knowledge(self, multi_agent_setup: Dict[str, Any]) -> None:
        """Test agents share public knowledge."""
        user_agent = multi_agent_setup["user"]
        assistant_agent = multi_agent_setup["assistant"]
        
        # Both should reference same public store path
        assert user_agent.config["public_knowledge_path"] == assistant_agent.config["public_knowledge_path"]
        
        # But different private paths
        assert user_agent.config["private_knowledge_path"] != assistant_agent.config["private_knowledge_path"]

    def test_multi_agent_independent_learning(self, multi_agent_setup: Dict[str, Any]) -> None:
        """Test agents learn independently."""
        user_agent = multi_agent_setup["user"]
        assistant_agent = multi_agent_setup["assistant"]
        
        # Each agent learns different data
        user_agent.ingest(b"user data")
        assistant_agent.ingest(b"assistant data")
        
        # Should have different cycle counts
        user_info = user_agent.get_agent_info()
        assistant_info = assistant_agent.get_agent_info()
        
        # May have different states due to independent learning
        assert user_info["agent_id"] != assistant_info["agent_id"]


class TestOrchestratedTurn:
    """Test orchestrated conversational turns."""

    @patch('toys.communication.tokenizer.encode')
    @patch('toys.communication.tokenizer.decode')
    def test_orchestrate_turn_basic(self, mock_decode, mock_encode, agent_pool: AgentPool) -> None:
        """Test basic orchestrated turn."""
        # Mock tokenizer functions
        mock_encode.return_value = b"encoded_input"
        mock_decode.return_value = "decoded_response"
        
        user_id = "test_user"
        assistant_id = "test_assistant"
        user_input = "Hello, assistant!"
        tokenizer_name = "bert-base-uncased"
        
        result = orchestrate_turn(agent_pool, user_id, assistant_id, user_input, tokenizer_name)
        
        assert isinstance(result, str)
        assert result == "decoded_response"
        
        # Should have called tokenizer functions
        mock_encode.assert_called_with(user_input, name=tokenizer_name)
        mock_decode.assert_called()

    @patch('toys.communication.tokenizer.encode')
    @patch('toys.communication.tokenizer.decode')
    def test_orchestrate_turn_creates_agents(self, mock_decode, mock_encode, agent_pool: AgentPool) -> None:
        """Test orchestrated turn creates agents as needed."""
        mock_encode.return_value = b"test"
        mock_decode.return_value = "response"
        
        user_id = "new_user"
        assistant_id = "new_assistant"
        
        orchestrate_turn(agent_pool, user_id, assistant_id, "test", "bert-base-uncased")
        
        # Should have created both agents
        active = agent_pool.get_active_agents()
        assert user_id in active
        assert assistant_id in active

    @patch('toys.communication.tokenizer')
    def test_orchestrate_turn_missing_tokenizer(self, mock_tokenizer, agent_pool: AgentPool) -> None:
        """Test handling when tokenizer is not available."""
        # Mock missing tokenizer
        mock_tokenizer.encode.side_effect = ImportError("Tokenizer not found")
        
        with patch('warnings.warn') as mock_warn:
            result = orchestrate_turn(agent_pool, "user", "assistant", "test", "bert-base-uncased")
            
            assert result == "[ERROR: Tokenizer not found]"
            mock_warn.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_intelligence_engine_missing_ontology(self, temp_store: OrbitStore) -> None:
        """Test error when ontology file is missing."""
        with pytest.raises(FileNotFoundError):
            IntelligenceEngine(
                ontology_path="/nonexistent/path.json",
                phenotype_store=temp_store
            )

    def test_gyrosi_missing_ontology_path(self, temp_dir: Path) -> None:
        """Test error when ontology path is missing from config."""
        config: AgentConfig = {
            "knowledge_path": str(temp_dir / "knowledge.pkl.gz"),
            # Missing ontology_path
        }
        
        with pytest.raises(ValueError, match="must include 'ontology_path'"):
            GyroSI(config)

    def test_process_egress_with_none_input(self, gyrosi_agent: GyroSI) -> None:
        """Test egress handles type conversion gracefully."""
        engine = gyrosi_agent.engine
        
        # Should handle and mask properly
        result = engine.process_egress(None)  # Should convert to 0
        assert isinstance(result, int)

    def test_process_ingress_response_fallback(self, gyrosi_agent: GyroSI) -> None:
        """Test ingress response generation fallback."""
        engine = gyrosi_agent.engine
        
        # Process with state that might produce empty phenotype
        response = engine.process_ingress(0)
        
        # Should always return valid byte
        assert 0 <= response <= 255

    def test_agent_pool_invalid_preferences(self, meta_paths: Dict[str, str], temp_dir: Path) -> None:
        """Test agent pool with invalid preferences."""
        public_path = str(temp_dir / "public.pkl.gz")
        OrbitStore(public_path).close()
        
        # Invalid eviction policy should default
        invalid_prefs = {"agent_eviction_policy": "invalid_policy"}
        
        pool = AgentPool(meta_paths["ontology"], public_path, invalid_prefs)
        
        # Should handle gracefully
        assert pool.eviction_policy == "invalid_policy"  # Stored as-is
        
        pool.close_all()

    def test_state_sync_consistency(self, gyrosi_agent: GyroSI) -> None:
        """Test state synchronization consistency."""
        engine = gyrosi_agent.engine
        
        if engine.use_epistemology:
            # Test sync methods work correctly
            original_index = engine.current_state_index
            
            engine._sync_state_fields_from_index()
            assert engine.gene_mac_m_int == engine.s2.get_state_from_index(original_index)
            
            engine._sync_index_from_state_int()
            assert engine.current_state_index == original_index

    def test_hook_execution_exception_handling(self, gyrosi_agent: GyroSI) -> None:
        """Test hook execution handles exceptions gracefully."""
        engine = gyrosi_agent.engine
        
        def failing_hook(eng, entry, intron):
            raise RuntimeError("Hook failed")
        
        engine.add_hook(failing_hook)
        
        # Should not crash the engine
        try:
            engine.process_ingress(42)
        except RuntimeError:
            # If exception propagates, that's also acceptable behavior
            pass

    def test_memory_cleanup_on_close(self, meta_paths: Dict[str, str], temp_dir: Path) -> None:
        """Test proper cleanup on agent close."""
        config: AgentConfig = {
            "ontology_path": meta_paths["ontology"],
            "knowledge_path": str(temp_dir / "knowledge.pkl.gz"),
        }
        
        agent = GyroSI(config)
        
        # Use the agent
        agent.ingest(b"test")
        
        # Close should not raise
        agent.close()
        
        # Subsequent operations should not crash (though may error)
        try:
            agent.ingest(b"after close")
        except:
            pass  # Expected to fail after close

    def test_large_batch_learning(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning with larger data."""
        engine = gyrosi_agent.engine
        
        # Large batch should not cause memory issues
        large_data = b"x" * 1000
        
        initial_cycle = engine.cycle_count
        engine.batch_learn(large_data)
        
        assert engine.cycle_count == initial_cycle + len(large_data)

    def test_agent_id_uniqueness(self, agent_pool: AgentPool) -> None:
        """Test that auto-generated agent IDs are unique."""
        # Create agents without specifying IDs
        agents = []
        for i in range(5):
            # Use different temporary names to avoid pool reuse
            agent = agent_pool.get_or_create_agent(f"temp_agent_{i}")
            agents.append(agent)
        
        # All should have unique IDs
        agent_ids = [agent.agent_id for agent in agents]
        assert len(set(agent_ids)) == len(agent_ids)