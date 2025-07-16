"""
Tests for S4/5: Intelligence - Orchestration & API
"""

import pytest
import time
import threading
from baby import governance
from baby.intelligence import (
    IntelligenceEngine,
    GyroSI,
    AgentPool,
    orchestrate_turn,
)
from baby.types import AgentConfig


class TestIntelligenceEngine:
    """Test the intelligence engine orchestration."""
    
    @pytest.fixture
    def intelligence_engine(self, manifold_data, pickle_store):
        """Create an intelligence engine for testing."""
        manifold_path, _ = manifold_data
        return IntelligenceEngine(manifold_path, pickle_store)
    
    def test_initialization(self, intelligence_engine):
        """Test engine initialization."""
        assert intelligence_engine.agent_id is not None
        assert intelligence_engine.cycle_count == 0
        assert intelligence_engine.gene_mac_m_int == intelligence_engine.s2.tensor_to_int(governance.GENE_Mac_S)
    
    def test_process_egress(self, intelligence_engine):
        """Test egress processing (input transformation)."""
        initial_state = intelligence_engine.gene_mac_m_int
        
        # Process a byte
        input_byte = 42
        intron = intelligence_engine.process_egress(input_byte)
        
        # Check intron is transcribed correctly
        assert intron == governance.transcribe_byte(input_byte)
        
        # Check state was transformed
        assert intelligence_engine.gene_mac_m_int != initial_state
        assert intelligence_engine.cycle_count == 1
    
    def test_process_ingress(self, intelligence_engine):
        """Test ingress processing (learning and response)."""
        # First do egress to get an intron
        intron = intelligence_engine.process_egress(42)
        
        # Process ingress
        output_byte = intelligence_engine.process_ingress(intron)
        
        # Should return a valid byte
        assert 0 <= output_byte <= 255
        
        # Default phenotype "?" should return ord('?') = 63
        assert output_byte == ord('?')
    
    def test_batch_learn(self, intelligence_engine):
        """Test batch learning."""
        initial_cycles = intelligence_engine.cycle_count
        
        # Batch learn from text
        data = b"Hello, GyroSI!"
        intelligence_engine.batch_learn(data)
        
        # Check cycles increased
        assert intelligence_engine.cycle_count == initial_cycles + len(data)
    
    def test_hooks(self, intelligence_engine):
        """Test post-cycle hooks."""
        hook_calls = []
        
        def test_hook(engine, phenotype, intron):
            hook_calls.append((engine.cycle_count, intron))
        
        # Add hook
        intelligence_engine.add_hook(test_hook)
        
        # Process some cycles
        for i in range(3):
            intron = intelligence_engine.process_egress(i)
            intelligence_engine.process_ingress(intron)
        
        # Verify hook was called
        assert len(hook_calls) == 3
        assert hook_calls[0][0] == 1  # First cycle
        
        # Remove hook
        assert intelligence_engine.remove_hook(test_hook)
        assert not intelligence_engine.remove_hook(test_hook)  # Already removed
    
    def test_state_info(self, intelligence_engine):
        """Test state information retrieval."""
        info = intelligence_engine.get_state_info()
        
        assert info["agent_id"] == intelligence_engine.agent_id
        assert info["cycle_count"] == 0
        assert "state_integer" in info
        assert "tensor_index" in info
        assert 0 <= info["angular_divergence_radians"] <= 3.15
        assert info["active_hooks"] == 0
    
    def test_reset_to_archetypal(self, intelligence_engine):
        """Test resetting to archetypal state."""
        # Process some data to change state
        intelligence_engine.batch_learn(b"Some data")
        assert intelligence_engine.cycle_count > 0
        
        # Reset
        intelligence_engine.reset_to_archetypal_state()
        
        assert intelligence_engine.cycle_count == 0
        assert intelligence_engine.gene_mac_m_int == intelligence_engine.s2.tensor_to_int(governance.GENE_Mac_S)


class TestGyroSI:
    """Test the main GyroSI API."""
    
    def test_basic_initialization(self, agent_config):
        """Test basic GyroSI initialization."""
        agent = GyroSI(agent_config)
        
        assert agent.agent_id is not None
        assert agent.engine is not None
        
        agent.close()
    
    def test_ingest_and_respond(self, gyrosi_agent):
        """Test the main ingest/respond cycle."""
        # Ingest some training data
        training_data = b"The quick brown fox"
        gyrosi_agent.ingest(training_data)
        
        # Generate response
        input_data = b"jumps"
        response = gyrosi_agent.respond(input_data)
        
        assert isinstance(response, bytes)
        assert len(response) == len(input_data)
    
    def test_empty_input_handling(self, gyrosi_agent):
        """Test handling of empty input."""
        # Empty ingest
        gyrosi_agent.ingest(b"")
        
        # Empty respond
        response = gyrosi_agent.respond(b"")
        assert response == b""
    
    def test_agent_info(self, gyrosi_agent):
        """Test agent information retrieval."""
        info = gyrosi_agent.get_agent_info()
        
        assert "agent_id" in info
        assert "config" in info
        assert "knowledge_statistics" in info
        assert "system_integrity" in info
    
    def test_monitoring_hooks(self, gyrosi_agent):
        """Test adding monitoring hooks."""
        hook_data = []
        
        def monitor_hook(engine, phenotype, intron):
            hook_data.append(phenotype["phenotype"])
        
        gyrosi_agent.add_monitoring_hook(monitor_hook)
        gyrosi_agent.respond(b"test")
        
        assert len(hook_data) == 4  # One per byte
    
    def test_maintenance_operations(self, gyrosi_agent):
        """Test maintenance operations."""
        # Add some data first
        gyrosi_agent.ingest(b"Sample data for maintenance test")
        
        # Apply maintenance
        report = gyrosi_agent.apply_maintenance(
            decay_factor=0.99,
            confidence_threshold=0.01
        )
        
        assert "decay_applied" in report
        assert "entries_pruned" in report
        assert "timestamp" in report


class TestAgentPool:
    """Test agent pool management."""
    
    def test_get_or_create_agent(self, agent_pool):
        """Test agent creation and retrieval."""
        # Create new agent
        agent1 = agent_pool.get_or_create_agent("user1", role_hint="user")
        assert agent1 is not None
        
        # Retrieve same agent
        agent2 = agent_pool.get_or_create_agent("user1")
        assert agent2 is agent1  # Same instance
    
    def test_remove_agent(self, agent_pool):
        """Test agent removal."""
        # Create agent
        agent_pool.get_or_create_agent("temp_agent")
        assert "temp_agent" in agent_pool.get_active_agents()
        
        # Remove agent
        assert agent_pool.remove_agent("temp_agent")
        assert "temp_agent" not in agent_pool.get_active_agents()
        
        # Try to remove non-existent
        assert not agent_pool.remove_agent("temp_agent")
    
    def test_lru_eviction(self, manifold_data, temp_dir):
        """Test LRU eviction policy."""
        manifold_path, _ = manifold_data
        knowledge_path = os.path.join(temp_dir, "knowledge.pkl.gz")
        
        # Create pool with small capacity
        preferences = {"max_agents_in_memory": 3, "agent_eviction_policy": "lru"}
        pool = AgentPool(manifold_path, knowledge_path, preferences)
        
        # Create more agents than capacity
        for i in range(5):
            pool.get_or_create_agent(f"agent{i}")
        
        # Should have evicted oldest
        active = pool.get_active_agents()
        assert len(active) == 3
        assert "agent0" not in active
        assert "agent1" not in active
        assert "agent4" in active
        
        pool.close_all()
    
    def test_thread_safety(self, agent_pool):
        """Test thread-safe agent access."""
        errors = []
        results = {}
        
        def create_and_use_agent(agent_id):
            try:
                agent = agent_pool.get_or_create_agent(agent_id)
                response = agent.respond(b"test")
                results[agent_id] = len(response)
            except Exception as e:
                errors.append(e)
        
        # Launch multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_and_use_agent, args=(f"agent{i}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 10


class TestConversationOrchestration:
    """Test conversation orchestration between agents."""
    
    def test_orchestrate_turn(self, agent_pool):
        """Test basic conversation turn."""
        # Setup agents
        user_agent = agent_pool.get_or_create_agent("user", role_hint="user")
        assistant_agent = agent_pool.get_or_create_agent("assistant", role_hint="assistant")
        
        # Train assistant
        assistant_agent.ingest(b"I am a helpful assistant.")
        
        # Orchestrate turn
        response = orchestrate_turn(
            agent_pool,
            "user",
            "assistant", 
            "Hello!"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_multi_turn_conversation(self, agent_pool):
        """Test multi-turn conversation."""
        responses = []
        
        # Have a short conversation
        for user_input in ["Hello", "How are you?", "Goodbye"]:
            response = orchestrate_turn(
                agent_pool,
                "user1",
                "assistant1",
                user_input
            )
            responses.append(response)
        
        assert len(responses) == 3
        assert all(isinstance(r, str) for r in responses)
    
    def test_unicode_handling(self, agent_pool):
        """Test handling of unicode in conversations."""
        # Test various unicode inputs
        test_inputs = [
            "Hello! 👋",
            "Café ☕",
            "数学 πr²",
            "Москва",
        ]
        
        for input_text in test_inputs:
            response = orchestrate_turn(
                agent_pool,
                "unicode_user",
                "unicode_assistant",
                input_text
            )
            assert isinstance(response, str)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_learning_cycle(self, agent_config):
        """Test complete learning and response cycle."""
        agent = GyroSI(agent_config)
        
        # Train on some text
        training = b"The rain in Spain stays mainly in the plain."
        agent.ingest(training)
        
        # Test various inputs
        test_cases = [
            b"rain",
            b"Spain",
            b"plain"
        ]
        
        responses = []
        for test_input in test_cases:
            response = agent.respond(test_input)
            responses.append(response)
        
        # Responses should be consistent for same input
        response1 = agent.respond(b"rain")
        response2 = agent.respond(b"rain")
        assert response1 == response2
        
        agent.close()
    
    def test_persistence_across_sessions(self, agent_config, temp_dir):
        """Test that learning persists across agent sessions."""
        agent_id = "persistent_agent"
        
        # First session - train
        agent1 = GyroSI(agent_config, agent_id=agent_id)
        agent1.ingest(b"Knowledge to remember")
        response1 = agent1.respond(b"test")
        agent1.close()
        
        # Second session - should remember
        agent2 = GyroSI(agent_config, agent_id=agent_id)
        response2 = agent2.respond(b"test")
        agent2.close()
        
        # Should produce same response (demonstrating memory)
        assert response1 == response2
    
    @pytest.mark.slow
    def test_performance_at_scale(self, agent_config):
        """Test performance with larger workload."""
        from conftest import Timer
        
        agent = GyroSI(agent_config)
        
        # Generate test data
        test_size = 10000
        test_data = bytes(range(256)) * (test_size // 256)
        
        # Measure ingestion performance
        with Timer() as ingest_timer:
            agent.ingest(test_data)
        
        ingest_rate = len(test_data) / ingest_timer.elapsed
        assert ingest_rate > 100_000, f"Only {ingest_rate:.0f} bytes/sec ingestion"
        
        # Measure response performance
        with Timer() as respond_timer:
            response = agent.respond(test_data[:1000])
        
        respond_rate = 1000 / respond_timer.elapsed
        assert respond_rate > 10_000, f"Only {respond_rate:.0f} bytes/sec response"
        
        agent.close()