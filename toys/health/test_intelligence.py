"""
Tests for the front-line integration layer: intelligence.py, orchestration, and API adapters.
Tests focus on end-to-end flows while avoiding overlap with existing governance/information/inference tests.
"""

import concurrent.futures
import json
import os  # noqa: F401
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from baby.contracts import AgentConfig
from baby.intelligence import AgentPool, GyroSI, orchestrate_turn
from baby.policies import OrbitStore
from toys.communication import tokenizer as gyrotok
from toys.communication.external_adapter import app


# Parameterize tokenizer name for easier multi-tokenizer testing
TOKENIZER_NAMES = ["bert-base-uncased"]


def _count_entries(store: Any) -> int:
    """Helper to count entries in a store."""
    return sum(1 for _ in store.iter_entries())


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_adapter_pool() -> None:
    """
    Reset the global agent pool before and after each test for perfect isolation.
    
    This is necessary because the FastAPI adapter uses a module-level agent_pool 
    that persists across test runs. Without this reset, tests would interfere with
    each other through cumulative state.
    """
    from toys.communication.external_adapter import agent_pool as global_pool
    
    # Reset before test
    global_pool.close_all()
    global_pool.ensure_triad()
    
    yield
    
    # Reset after test (cleanup)
    global_pool.close_all()


class TestIntelligenceEngineOrchestration:
    """Test core intelligence orchestration between agents."""

    def test_batch_learning_stores_different_phenotypes(self, gyrosi_agent: GyroSI) -> None:
        """Test batch learning creates distinct phenotypes for different input sequences."""
        agent = gyrosi_agent
        
        # Learn two different sequences
        sequence1 = b"ABC"
        sequence2 = b"CBA"
        
        # Learn first sequence
        agent.ingest(sequence1)
        
        # Learn second sequence
        agent.ingest(sequence2)
        
        # Examine store directly to verify different phenotypes were created
        store = agent.engine.operator.store
        
        # Get all entries and extract their phenotypes
        phenotypes = []
        for _, entry in store.iter_entries():
            if "phenotype" in entry:
                phenotypes.append(entry["phenotype"])
        
        # Should have created at least two different phenotypes
        assert len(set(phenotypes)) >= 2

    def test_hook_system_via_respond_path_handles_batching(self, gyrosi_agent: GyroSI) -> None:
        """Test hooks are triggered through the respond pathway, accounting for batching."""
        agent = gyrosi_agent
        hook_calls = []
        
        def test_hook(engine: Any, phenotype_entry: Any, last_intron: int) -> None:
            hook_calls.append(True)
        
        agent.add_monitoring_hook(test_hook)
        
        # Make enough calls to ensure hook buffer is flushed
        # Hook batch interval is typically 8, so make more than that
        for _ in range(10):
            agent.respond(b"test")
        
        # Hook should have been called through respond path
        assert len(hook_calls) > 0


class TestGyroSIAgentLifecycle:
    """Test GyroSI agent lifecycle: creation, learning, response, and maintenance."""

    def test_agent_respond_produces_valid_output(self, gyrosi_agent: GyroSI) -> None:
        """Test agent generates valid bytes in response to input."""
        agent = gyrosi_agent
        
        # Test with various inputs
        test_inputs = [b"Hello", b"42", b"GyroSI"]
        
        for input_data in test_inputs:
            response = agent.respond(input_data)
            
            assert isinstance(response, bytes)
            assert len(response) > 0
            # Response should be valid bytes
            assert all(0 <= b <= 255 for b in response)

    def test_knowledge_persistence_across_sessions(self, multi_agent_setup: Dict[str, Any]) -> None:
        """Test knowledge persists across agent sessions by counting entries."""
        setup = multi_agent_setup
        user_agent = setup["user"]
        
        # Get initial entry count
        store = user_agent.engine.operator.store
        initial_count = _count_entries(store)
        
        # Train with specific knowledge
        training_data = b"GyroSI uses physics-grounded artificial intelligence."
        user_agent.ingest(training_data)
        
        # Get count after learning
        learned_count = _count_entries(store)
        assert learned_count > initial_count
        
        # Close agent
        user_agent.close()
        
        # Create new agent with same configuration
        new_config: AgentConfig = {
            "ontology_path": user_agent.config["ontology_path"],
            "public_knowledge_path": setup["public_path"],
            "private_knowledge_path": str(setup["temp_dir"] / "user_private.pkl.gz"),
            "enable_phenomenology_storage": True,
            "phenomenology_map_path": user_agent.config["phenomenology_map_path"],
        }
        
        new_agent = GyroSI(new_config, agent_id="test_user", base_path=setup["temp_dir"])
        
        # Count entries in reloaded agent
        reloaded_count = _count_entries(new_agent.engine.operator.store)
        
        # Should have persisted the learned entries
        assert reloaded_count >= learned_count
        
        new_agent.close()


class TestAgentPoolManagement:
    """Test AgentPool multi-agent coordination and management."""

    def test_pool_triad_creation(self, agent_pool: AgentPool) -> None:
        """Test agent pool creates required triad agents."""
        pool = agent_pool
        
        # Ensure triad exists
        pool.ensure_triad()
        
        active_agents = pool.get_active_agents()
        required_agents = {"user", "system", "assistant"}
        
        assert required_agents.issubset(set(active_agents))

    def test_agent_isolation_in_pool(self, agent_pool: AgentPool) -> None:
        """Test agents in pool maintain separate knowledge states."""
        pool = agent_pool
        
        # Get two different agents
        agent1 = pool.get_or_create_agent("test_agent_1")
        agent2 = pool.get_or_create_agent("test_agent_2")
        
        # Train them differently
        agent1.ingest(b"Agent 1 specific knowledge")
        agent2.ingest(b"Agent 2 different knowledge")
        
        # Verify different states
        state1 = agent1.get_agent_info()
        state2 = agent2.get_agent_info()
        
        assert state1["agent_id"] != state2["agent_id"]
        
        # Clean up
        pool.remove_agent("test_agent_1")
        pool.remove_agent("test_agent_2")

    def test_pool_lru_eviction_policy(self, temp_dir: Path, meta_paths: Dict[str, str]) -> None:
        """Test LRU eviction policy correctly removes the least recently used agent."""
        # Create temporary public knowledge store
        public_path = str(temp_dir / "pub_lru.pkl.gz")
        OrbitStore(public_path).close()
        
        # Create a tiny pool with max 2 agents and LRU policy
        tiny_pool = AgentPool(
            meta_paths["ontology"],
            public_path,
            preferences={"max_agents_in_memory": 2, "agent_eviction_policy": "lru"},
            base_path=temp_dir,
            allow_auto_create=True
        )
        
        try:
            # Create first agent (will be LRU after second agent created)
            agent_old = tiny_pool.get_or_create_agent("agent_old")
            
            # Create second agent
            agent_recent = tiny_pool.get_or_create_agent("agent_recent")
            
            # Create third agent - should evict the oldest (first) agent
            agent_newest = tiny_pool.get_or_create_agent("agent_newest")
            
            active_agents = tiny_pool.get_active_agents()
            
            # The oldest agent should be evicted
            assert "agent_old" not in active_agents
            # The two most recent should remain
            assert "agent_recent" in active_agents
            assert "agent_newest" in active_agents
        finally:
            tiny_pool.close_all()

    def test_pool_ttl_eviction_policy(self, temp_dir: Path, meta_paths: Dict[str, str]) -> None:
        """Test TTL eviction policy removes expired agents."""
        # Create temporary public knowledge store
        public_path = str(temp_dir / "pub_ttl.pkl.gz")
        OrbitStore(public_path).close()
        
        # Create pool with TTL eviction and very short timeout (0.001 minutes = 0.06 seconds)
        ttl_pool = AgentPool(
            meta_paths["ontology"],
            public_path,
            preferences={
                "max_agents_in_memory": 100,
                "agent_eviction_policy": "ttl",
                "agent_ttl_minutes": 0.001  # Very short TTL
            },
            base_path=temp_dir,
            allow_auto_create=True
        )
        
        try:
            # Create agent
            agent = ttl_pool.get_or_create_agent("ttl_test_agent")
            assert "ttl_test_agent" in ttl_pool.get_active_agents()
            
            # Wait for TTL to expire
            time.sleep(0.1)  # 100ms, should be more than enough
            
            # Trigger eviction by creating a new agent
            ttl_pool.get_or_create_agent("ttl_trigger_agent")
            
            # First agent should be evicted due to TTL expiry
            assert "ttl_test_agent" not in ttl_pool.get_active_agents()
        finally:
            ttl_pool.close_all()

    @pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
    def test_orchestrate_turn_with_tokenizer(
        self, agent_pool: AgentPool, tokenizer_name: str
    ) -> None:
        """Test orchestrated conversation turn with tokenizer integration."""
        pool = agent_pool
        pool.ensure_triad()
        
        user_input = "What is the meaning of life?"
        
        # Orchestrate conversation turn
        response = orchestrate_turn(
            pool=pool,
            user_id="user",
            assistant_id="assistant", 
            user_input=user_input,
            tokenizer_name=tokenizer_name
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    def test_orchestrate_turn_tokenizer_error(self, agent_pool: AgentPool) -> None:
        """Test orchestrate_turn raises error with invalid tokenizer name."""
        pool = agent_pool
        pool.ensure_triad()
        
        user_input = "Test input"
        
        # Try with non-existent tokenizer
        with pytest.raises(Exception):  # Could be FileNotFoundError, ValueError, etc.
            orchestrate_turn(
                pool=pool,
                user_id="user",
                assistant_id="assistant",
                user_input=user_input,
                tokenizer_name="non-existent-tokenizer"
            )


class TestTokenizerIntegration:
    """Test tokenizer integration with agent workflow."""

    @pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
    def test_tokenizer_roundtrip_preservation(self, tokenizer_name: str) -> None:
        """Test tokenizer round-trip preserves content using only public API."""
        # Test with text that has minimal tokenizer normalization
        test_text = "Hello world! This is a test of the tokenizer."
        
        # Encode to bytes
        encoded_bytes = gyrotok.encode(test_text, name=tokenizer_name)
        
        # Decode back to text
        decoded_text = gyrotok.decode(encoded_bytes, name=tokenizer_name)
        
        # Re-encode decoded text and compare byte sequences
        reencoded_bytes = gyrotok.encode(decoded_text, name=tokenizer_name)
        
        # Byte sequences should be identical (full reversibility using public API only)
        assert encoded_bytes == reencoded_bytes

    def test_tokenizer_utf8_fallback(self) -> None:
        """Test tokenizer's UTF-8 fallback for invalid LEB128 bytes."""
        # Invalid LEB128 bytes with high bit set on last byte
        invalid_bytes = b"\xFF\xFF"
        
        # Should fall back to UTF-8 decode without raising an exception
        result = gyrotok.decode(invalid_bytes, name="bert-base-uncased")
        
        # Should return a string (may be replacement chars)
        assert isinstance(result, str)


class TestExternalAdapterHTTP:
    """Test external API adapter endpoints."""

    def test_models_endpoint(self, client: TestClient) -> None:
        """Test /v1/models endpoint returns expected format."""
        response = client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "gyrosi-baby-0.9.6"

    def test_chat_completions_basic(self, client: TestClient) -> None:
        """Test chat completions generates valid response."""
        payload = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "user", "content": "Hello, GyroSI!"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify OpenAI-compatible structure
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_concurrent_chat_completions(self, client: TestClient) -> None:
        """Test concurrent requests don't interfere with each other."""
        payload1 = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [{"role": "user", "content": "Hello from thread 1"}]
        }
        
        payload2 = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [{"role": "user", "content": "Hello from thread 2"}]
        }
        
        def make_request(payload: Dict[str, Any]) -> int:
            response = client.post("/v1/chat/completions", json=payload)
            return response.status_code
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(make_request, payload1)
            future2 = executor.submit(make_request, payload2)
            
            # Both should succeed
            status1 = future1.result()
            status2 = future2.result()
        
        assert status1 == 200
        assert status2 == 200

    def test_chat_completions_with_system_message(self, client: TestClient) -> None:
        """Test chat completions incorporates system message."""
        payload = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What can you help me with?"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] is not None

    def test_chat_completions_streaming(self, client: TestClient) -> None:
        """Test streaming chat completions returns valid SSE format."""
        payload = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
        
        # Use streaming context manager for robust SSE testing
        with client.stream(
            "POST", 
            "/v1/chat/completions", 
            json=payload, 
            params={"stream": "true"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            
            # Collect all chunks
            chunks = []
            content_chunks = []
            
            for line in response.iter_lines():
                if line.strip() and line.startswith("data: "):
                    chunks.append(line)
                    
                    # Parse non-DONE chunks for content
                    if "data: [DONE]" not in line:
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            if ("choices" in data and data["choices"] and 
                                "delta" in data["choices"][0] and 
                                "content" in data["choices"][0]["delta"]):
                                content_chunks.append(data["choices"][0]["delta"]["content"])
                        except json.JSONDecodeError:
                            continue
            
            # Verify we have chunks and a DONE marker
            assert len(chunks) >= 1
            assert any("data: [DONE]" in chunk for chunk in chunks)
            
            # Should have extracted some content
            assert len(content_chunks) > 0
            
            # Joining content should produce a valid string
            full_content = "".join(content_chunks)
            assert len(full_content) > 0

    def test_hf_generate_endpoint(self, client: TestClient) -> None:
        """Test HuggingFace-compatible generate endpoint."""
        payload = {
            "inputs": "Generate a response about artificial intelligence."
        }
        
        response = client.post("/generate", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)
        assert len(data["generated_text"]) > 0

    def test_user_id_mapping_to_internal_agent(self, client: TestClient) -> None:
        """Test custom user ID in header maps to internal 'user' agent."""
        from toys.communication.external_adapter import agent_pool as global_pool
        
        # Get initial pool state
        pre_agents = set(global_pool.get_active_agents())
        
        payload = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "user", "content": "Hello from custom user!"}
            ]
        }
        
        # Use a custom user ID in header
        headers = {"X-User-ID": "custom-user-123"}
        
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 200
        
        # Verify pool state - no new agent should be created
        post_agents = set(global_pool.get_active_agents())
        assert "custom-user-123" not in post_agents  # No agent with custom ID
        assert "user" in post_agents  # Standard triad agent exists
        assert pre_agents == post_agents  # No new agents created
        
        # Verify a response was generated
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert len(data["choices"][0]["message"]["content"]) > 0


class TestEndToEndConversation:
    """Test complete conversation flows through all layers."""

    @pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
    def test_full_conversation_pipeline(
        self, agent_pool: AgentPool, tokenizer_name: str
    ) -> None:
        """Test complete conversation from input to response."""
        pool = agent_pool
        pool.ensure_triad()
        
        # Simulate a conversation turn
        user_input = "Tell me about the nature of intelligence."
        
        # Full pipeline: tokenize → agents → orchestrate → detokenize
        response = orchestrate_turn(
            pool=pool,
            user_id="user",
            assistant_id="assistant",
            user_input=user_input,
            tokenizer_name=tokenizer_name
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Response should be different from input (showing processing)
        assert response != user_input

    @pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
    def test_multi_turn_conversation(
        self, agent_pool: AgentPool, tokenizer_name: str
    ) -> None:
        """Test multi-turn conversation produces meaningful responses."""
        pool = agent_pool
        pool.ensure_triad()
        
        # First turn
        response1 = orchestrate_turn(
            pool=pool,
            user_id="user",
            assistant_id="assistant",
            user_input="I like cats.",
            tokenizer_name=tokenizer_name
        )
        
        # Second turn referencing first
        response2 = orchestrate_turn(
            pool=pool,
            user_id="user", 
            assistant_id="assistant",
            user_input="What do I like?",
            tokenizer_name=tokenizer_name
        )
        
        assert isinstance(response1, str)
        assert isinstance(response2, str)
        assert len(response1) > 0
        assert len(response2) > 0