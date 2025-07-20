"""
Tests for the GyroSI intelligence system.

This suite tests policies.py, contracts.py, and intelligence.py
using the provided fixtures and actual implementations.
"""

import os
import json
import pytest
from pathlib import Path
from typing import cast


class TestPolicies:
    """Test storage policies and persistence mechanisms."""

    def test_orbit_store_basic_operations(self, orbit_store, sample_phenotype_entry):
        """Test basic OrbitStore operations with proper fixtures."""
        # Create a test key
        key = (42, 123)
        # Use the store from the fixture
        orbit_store.put(key, sample_phenotype_entry)
        orbit_store.commit()
        # Verify data was stored and can be retrieved
        retrieved = orbit_store.get(key)
        assert retrieved is not None
        assert retrieved["phenotype"] == sample_phenotype_entry["phenotype"]
        assert retrieved["confidence"] == sample_phenotype_entry["confidence"]

    def test_orbit_store_updates(self, orbit_store, sample_phenotype_entry):
        """Test OrbitStore updates and overwrites."""
        key = (1, 1)

        # Initial write
        orbit_store.put(key, sample_phenotype_entry)
        orbit_store.commit()

        # Update with modified entry
        updated_entry = sample_phenotype_entry.copy()
        updated_entry["confidence"] = 0.9
        updated_entry["usage_count"] = 20

        orbit_store.put(key, updated_entry)
        orbit_store.commit()

        # Verify update was applied
        retrieved = orbit_store.get(key)
        assert retrieved["confidence"] == 0.9
        assert retrieved["usage_count"] == 20

    def test_orbit_store_delete(self, orbit_store, sample_phenotype_entry):
        """Test OrbitStore deletion."""
        key = (2, 2)

        # Add entry
        orbit_store.put(key, sample_phenotype_entry)
        orbit_store.commit()

        # Verify it exists
        assert orbit_store.get(key) is not None

        # Delete entry
        orbit_store.delete(key)

        # Verify it's gone
        assert orbit_store.get(key) is None

    def test_overlay_view(self, overlay_store, sample_phenotype_entry):
        """Test OverlayView with private and public stores."""
        # Public entry should exist from fixture setup
        public_key = (0, 0)
        public_store = (
            overlay_store.public_store.base_store
            if hasattr(overlay_store.public_store, "base_store")
            else overlay_store.public_store
        )
        print("DEBUG type(public_key):", type(public_key), public_key)
        print("DEBUG type(public_store.data.keys()):", [type(k) for k in public_store.data.keys()])
        print("DEBUG public_store.data.keys():", list(public_store.data.keys()))
        entry_private = overlay_store.private_store.get(public_key)
        print("DEBUG overlay private_store.get((0, 0)):", entry_private)
        print("DEBUG public store data:", public_store.data)
        entry = overlay_store.public_store.get(public_key)
        print("DEBUG overlay public_store.get((0, 0)):", entry)
        retrieved = overlay_store.get(public_key)
        print("DEBUG test_overlay_view: overlay_store.get((0, 0)) returned:", retrieved)
        assert retrieved is not None
        assert overlay_store.get(public_key)["phenotype"] == "public"

        # Add private entry with new key
        private_key = (1, 1)
        overlay_store.put(private_key, sample_phenotype_entry)

        # Verify it's retrievable
        retrieved = overlay_store.get(private_key)
        assert retrieved is not None
        assert retrieved["phenotype"] == sample_phenotype_entry["phenotype"]

        # Test private overrides public with same key
        overlay_store.put(public_key, sample_phenotype_entry)
        retrieved = overlay_store.get(public_key)
        assert retrieved["phenotype"] == sample_phenotype_entry["phenotype"]

    def test_readonly_view(self, temp_dir, orbit_store, sample_phenotype_entry):
        """Test ReadOnlyView protection."""
        from baby.policies import ReadOnlyView

        # Setup a read-only view of our store
        key = (3, 3)
        orbit_store.put(key, sample_phenotype_entry)
        orbit_store.commit()

        # Create read-only view
        readonly = ReadOnlyView(orbit_store)

        # Should be able to read
        retrieved = readonly.get(key)
        assert retrieved is not None
        assert retrieved["phenotype"] == sample_phenotype_entry["phenotype"]

        # Should not be able to write
        with pytest.raises(RuntimeError):
            readonly.put(key, {"phenotype": "should fail", "confidence": 0.5})


class TestContracts:
    """Test contracts and type specifications."""

    def test_phenotype_entry_validation(self, sample_phenotype_entry):
        """Test phenotype entry structure validation using fixture."""
        from baby.contracts import PhenotypeEntry

        # Use the utility function from conftest
        assert_phenotype_entry_valid(sample_phenotype_entry)

        # Create a valid typed dict
        entry: PhenotypeEntry = {
            "phenotype": "test",
            "confidence": 0.8,
            "memory_mask": 0b10101010,
            "context_signature": (42, 123),
            "usage_count": 5,
        }

        # Should pass validation
        assert_phenotype_entry_valid(entry)

    def test_agent_config_validation(self, agent_config):
        """Test AgentConfig validation using fixture."""
        from baby.contracts import AgentConfig

        # Verify required fields are present
        assert "ontology_path" in agent_config
        assert os.path.exists(agent_config["ontology_path"])

        # Add optional fields
        config = cast(
            AgentConfig, {**agent_config, "agent_metadata": {"role": "test"}, "enable_phenomenology_storage": True}
        )

        # Verify fields
        agent_metadata = config.get("agent_metadata")
        assert agent_metadata is not None and agent_metadata.get("role") == "test"
        assert config.get("enable_phenomenology_storage") is True

    def test_maintenance_report(self):
        """Test MaintenanceReport structure."""
        from baby.contracts import MaintenanceReport

        # Create a valid maintenance report
        report: MaintenanceReport = {
            "operation": "test_maintenance",
            "success": True,
            "entries_processed": 100,
            "entries_modified": 50,
            "elapsed_seconds": 1.5,
        }

        # Verify report structure
        assert report["operation"] == "test_maintenance"
        assert report["success"] is True
        assert report["entries_processed"] == 100
        assert report["entries_modified"] == 50
        assert report["elapsed_seconds"] == 1.5


class TestIntelligence:
    """Test intelligence engine components."""

    def test_intelligence_engine_initialization(self, agent_config, orbit_store):
        """Test IntelligenceEngine initialization."""
        from baby.intelligence import IntelligenceEngine

        # Create engine using actual code paths
        engine = IntelligenceEngine(ontology_path=agent_config["ontology_path"], phenotype_store=orbit_store)

        # Verify engine state
        assert engine.agent_id is not None
        assert engine.cycle_count == 0

        # Verify subsystems are initialized
        assert hasattr(engine, "s2")
        assert hasattr(engine, "operator")

    def test_intelligence_engine_cycle(self, agent_config, orbit_store):
        """Test intelligence engine's processing cycle."""
        from baby.intelligence import IntelligenceEngine

        engine = IntelligenceEngine(ontology_path=agent_config["ontology_path"], phenotype_store=orbit_store)

        # Process a simple input
        byte_value = 65  # ASCII 'A'
        intron = engine.process_egress(byte_value)

        # Verify basic processing occurred
        assert 0 <= intron <= 255
        assert engine.cycle_count == 1

        # Complete the cycle with ingress
        output = engine.process_ingress(intron)

        # Output should be a valid byte
        assert 0 <= output <= 255

    def test_intelligence_hooks(self, agent_config, orbit_store):
        """Test intelligence engine hooks."""
        from baby.intelligence import IntelligenceEngine

        engine = IntelligenceEngine(ontology_path=agent_config["ontology_path"], phenotype_store=orbit_store)

        # Track hook calls
        hook_calls = []

        def test_hook(engine, phenotype_entry, last_intron):
            hook_calls.append((engine.cycle_count, last_intron))

        # Add the hook
        engine.add_hook(test_hook)

        # Process a cycle
        intron = engine.process_egress(65)
        engine.process_ingress(intron)

        # Verify hook was called
        assert len(hook_calls) == 1
        assert hook_calls[0][0] == 1  # cycle count
        assert hook_calls[0][1] == intron

        # Remove hook and verify it's not called
        engine.remove_hook(test_hook)
        intron = engine.process_egress(66)
        engine.process_ingress(intron)

        # Hook count should not increase
        assert len(hook_calls) == 1

    def test_gyrosi_initialization(self, gyrosi_agent, agent_config):
        """Test GyroSI initialization using fixture."""
        # Verify agent was initialized correctly
        assert gyrosi_agent.agent_id is not None
        assert gyrosi_agent.config == agent_config

        # Check engine initialization
        assert hasattr(gyrosi_agent, "engine")
        assert gyrosi_agent.engine.cycle_count == 0

    def test_gyrosi_ingest_respond(self, gyrosi_agent):
        """Test GyroSI ingest and respond methods."""
        # Test data ingestion
        test_data = b"Hello, World!"
        gyrosi_agent.ingest(test_data)

        # Verify cycles were processed
        assert gyrosi_agent.engine.cycle_count == len(test_data)

        # Test response generation
        test_input = b"Test"
        response = gyrosi_agent.respond(test_input)

        # Verify response properties
        assert isinstance(response, bytes)
        assert len(response) == len(test_input)

        # Cycle count should have increased
        assert gyrosi_agent.engine.cycle_count == len(test_data) + len(test_input)

        # Verify empty input handling
        empty_response = gyrosi_agent.respond(b"")
        assert empty_response == b""

    def test_agent_info(self, gyrosi_agent):
        """Test agent info retrieval."""
        # Get agent info
        info = gyrosi_agent.get_agent_info()

        # Verify structure
        assert "agent_id" in info
        assert info["agent_id"] == gyrosi_agent.agent_id
        assert "cycle_count" in info
        assert "config" in info
        assert "knowledge_statistics" in info
        assert "system_integrity" in info


class TestAgentPool:
    """Test agent pool management."""

    def test_agent_pool_basics(self, agent_pool):
        """Test agent pool basic operations."""
        # Initially empty
        assert len(agent_pool.get_active_agents()) == 0

        # Create agent
        agent1 = agent_pool.get_or_create_agent("test_user")

        # Verify agent was created
        assert agent1.agent_id == "test_user"
        assert "test_user" in agent_pool.get_active_agents()

        # Get same agent again
        agent1_again = agent_pool.get_or_create_agent("test_user")

        # Should be the same instance
        assert agent1 is agent1_again

        # Create second agent
        agent2 = agent_pool.get_or_create_agent("test_assistant")

        # Should be different instance
        assert agent2 is not agent1
        assert "test_assistant" in agent_pool.get_active_agents()
        assert len(agent_pool.get_active_agents()) == 2

    def test_agent_removal(self, agent_pool):
        """Test agent removal from pool."""
        # Create agent
        agent_pool.get_or_create_agent("temp_agent")
        assert "temp_agent" in agent_pool.get_active_agents()

        # Remove agent
        result = agent_pool.remove_agent("temp_agent")

        # Verify removal
        assert result is True
        assert "temp_agent" not in agent_pool.get_active_agents()

        # Try to remove non-existent agent
        result = agent_pool.remove_agent("nonexistent")
        assert result is False

    def test_orchestrate_turn(self, agent_pool):
        """Test orchestration of a conversation turn."""
        # Test basic orchestration
        response = orchestrate_turn(
            agent_pool, "test_user_orchestrate", "test_assistant_orchestrate", "Hello, how are you?"
        )

        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify agents were created
        active_agents = agent_pool.get_active_agents()
        assert "test_user_orchestrate" in active_agents
        assert "test_assistant_orchestrate" in active_agents


class TestIntegration:
    """Integration tests for the full system."""

    def test_learn_and_respond(self, gyrosi_agent):
        """Test learning and response generation."""
        # Teach the agent something
        training_data = b"The quick brown fox jumps over the lazy dog."
        gyrosi_agent.ingest(training_data)

        # Get info after learning
        info_after_learning = gyrosi_agent.get_agent_info()

        # Test response after learning
        response = gyrosi_agent.respond(b"Hello")

        # Verify response
        assert isinstance(response, bytes)
        assert len(response) == 5  # "Hello" has 5 bytes

        # Get info after responding
        info_after_response = gyrosi_agent.get_agent_info()

        # Cycle count should have increased
        assert info_after_response["cycle_count"] > info_after_learning["cycle_count"]

    def test_multi_agent_interaction(self, agent_pool):
        """Test multi-agent interaction."""
        # Create two agents
        user_agent = agent_pool.get_or_create_agent("integration_user")
        assistant_agent = agent_pool.get_or_create_agent("integration_assistant")

        # Verify both agents exist
        assert user_agent.agent_id == "integration_user"
        assert assistant_agent.agent_id == "integration_assistant"

        # Test conversation (multiple turns)
        first_response = orchestrate_turn(agent_pool, "integration_user", "integration_assistant", "Hello!")
        assert isinstance(first_response, str)

        # Second turn
        second_response = orchestrate_turn(agent_pool, "integration_user", "integration_assistant", "How are you?")
        assert isinstance(second_response, str)

        # Agents should persist across turns
        assert user_agent.engine.cycle_count > 0
        assert assistant_agent.engine.cycle_count > 0

    def test_unicode_handling(self, agent_pool):
        """Test handling of Unicode text."""
        # Test with Unicode input
        unicode_input = "Hello 世界! 🌍"

        # Should not raise exceptions
        response = orchestrate_turn(agent_pool, "unicode_user", "unicode_assistant", unicode_input)

        # Response should be valid
        assert isinstance(response, str)

        # The response itself may not be meaningful, but should be valid UTF-8
        response.encode("utf-8")  # Should not raise exception


def assert_phenotype_entry_valid(entry):
    """Validate phenotype entry structure."""
    required_fields = ["phenotype", "memory_mask", "confidence", "context_signature"]
    for field in required_fields:
        assert field in entry, f"Missing required field: {field}"

    assert isinstance(entry["phenotype"], str)
    assert isinstance(entry["memory_mask"], int)
    assert 0 <= entry["memory_mask"] <= 255
    assert 0 <= entry["confidence"] <= 1.0
    assert isinstance(entry["context_signature"], tuple)
    assert len(entry["context_signature"]) == 2


def orchestrate_turn(pool, user_id, assistant_id, user_input):
    """Orchestrate a turn between agents (with proper import)."""
    from baby.intelligence import orchestrate_turn as actual_orchestrate_turn

    return actual_orchestrate_turn(pool, user_id, assistant_id, user_input)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
