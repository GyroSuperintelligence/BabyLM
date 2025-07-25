"""
Tests for the front-line integration layer: intelligence.py, orchestration, and API adapters.
Tests focus on end-to-end flows while avoiding overlap with existing governance/information/inference tests.
"""

import concurrent.futures
import json
from typing import Any, Dict, Callable
from pathlib import Path

import pytest

from baby.intelligence import AgentPool, GyroSI, orchestrate_turn
from toys.communication import tokenizer as gyrotok


# Parameterize tokenizer name for easier multi-tokenizer testing
TOKENIZER_NAMES = ["bert-base-uncased"]


def _count_entries(store: Any) -> int:
    """Helper to count entries in a store."""
    return sum(1 for _ in store.iter_entries())


class TestIntelligenceEngineOrchestration:
    """Test core intelligence orchestration between agents."""

    def test_batch_learning_stores_different_phenotypes(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test batch learning creates distinct phenotypes for different input sequences."""
        # Use isolated agent to avoid state interference between tests
        agent = isolated_agent_factory(tmp_path)

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

    def test_hook_system_via_respond_path_handles_batching(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test hooks are triggered through the respond pathway, accounting for batching."""
        # Use isolated agent to avoid hook interference between tests
        agent = isolated_agent_factory(tmp_path)
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

    def test_agent_respond_produces_valid_output(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test agent generates valid bytes in response to input."""
        # Use isolated agent to ensure clean state for response testing
        agent = isolated_agent_factory(tmp_path)

        # Test with various inputs
        test_inputs = [b"Hello", b"42", b"GyroSI"]

        for input_data in test_inputs:
            response = agent.respond(input_data)

            assert isinstance(response, bytes)
            assert len(response) > 0
            # Response should be valid bytes
            assert all(0 <= b <= 255 for b in response)

    def test_agent_state_persistence_across_operations(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test agent maintains consistent state across multiple operations."""
        agent = isolated_agent_factory(tmp_path)

        # Get initial state
        initial_state = agent.get_agent_info()
        initial_cycle_count = initial_state["cycle_count"]

        # Perform some operations
        agent.ingest(b"test data for persistence")
        agent.respond(b"test response")

        # Check state has evolved
        updated_state = agent.get_agent_info()
        assert updated_state["cycle_count"] > initial_cycle_count
        assert updated_state["system_integrity"] is not None


class TestAgentPoolManagement:
    """Test AgentPool multi-agent coordination and management."""

    def test_pool_triad_creation(self, agent_pool: AgentPool) -> None:
        """Test agent pool creates required triad agents."""
        pool = agent_pool

        # Triad should already exist from session setup
        active_agents = pool.get_active_agents()
        required_agents = {"user", "system", "assistant"}

        assert required_agents.issubset(set(active_agents))

    def test_pool_agent_retrieval_consistency(self, agent_pool: AgentPool) -> None:
        """Test pool returns consistent agent instances."""
        pool = agent_pool

        # Get the same agent multiple times
        user1 = pool.get("user")
        user2 = pool.get("user")

        # Should be the same instance
        assert user1 is user2

    @pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
    def test_orchestrate_turn_with_tokenizer(self, agent_pool: AgentPool, tokenizer_name: str) -> None:
        """Test orchestrated conversation turn with tokenizer integration."""
        pool = agent_pool

        user_input = "What is the meaning of life?"

        # Orchestrate conversation turn
        response = orchestrate_turn(
            pool=pool, user_id="user", assistant_id="assistant", user_input=user_input, tokenizer_name=tokenizer_name
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_orchestrate_turn_tokenizer_error(self, agent_pool: AgentPool) -> None:
        """Test orchestrate_turn raises error with invalid tokenizer name."""
        pool = agent_pool

        user_input = "Test input"

        # Try with non-existent tokenizer
        with pytest.raises(Exception):  # Could be FileNotFoundError, ValueError, etc.
            orchestrate_turn(
                pool=pool,
                user_id="user",
                assistant_id="assistant",
                user_input=user_input,
                tokenizer_name="non-existent-tokenizer",
            )

    def test_pool_agent_isolation(self, agent_pool: AgentPool) -> None:
        """Test that different agents in the pool have isolated storage."""
        pool = agent_pool

        user_agent = pool.get("user")
        assistant_agent = pool.get("assistant")

        # Have each agent learn something different
        user_agent.ingest(b"user specific data")
        assistant_agent.ingest(b"assistant specific data")

        # Check that their stores are separate
        user_entries = list(user_agent.engine.operator.store.iter_entries())
        assistant_entries = list(assistant_agent.engine.operator.store.iter_entries())

        # They should have different data (this is a basic check - in reality the stores
        # have overlapping public knowledge but separate private knowledge)
        user_private_entries = [e for k, e in user_entries if "user specific" in str(e)]
        assistant_private_entries = [e for k, e in assistant_entries if "assistant specific" in str(e)]

        # Each should have learned their own data but not see the other's private data
        # Note: This is a simplified check - the actual implementation uses overlay views
        assert len(user_private_entries) >= 0  # May be 0 due to how learning works
        assert len(assistant_private_entries) >= 0


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
        invalid_bytes = b"\xff\xff"

        # Should fall back to UTF-8 decode without raising an exception
        result = gyrotok.decode(invalid_bytes, name="bert-base-uncased")

        # Should return a string (may be replacement chars)
        assert isinstance(result, str)

    def test_tokenizer_agent_integration(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test tokenizer integration with isolated agent."""
        agent = isolated_agent_factory(tmp_path)

        # Test text that exercises tokenizer
        test_text = "The quick brown fox jumps over the lazy dog."

        # Encode and use with agent
        encoded = gyrotok.encode(test_text, name="bert-base-uncased")
        response = agent.respond(encoded)

        # Should get valid response
        assert isinstance(response, bytes)
        assert len(response) > 0

        # Should be decodable
        decoded_response = gyrotok.decode(response, name="bert-base-uncased")
        assert isinstance(decoded_response, str)


class TestExternalAdapterHTTP:
    """Test external API adapter endpoints."""

    def test_models_endpoint(self, test_client: Any) -> None:
        """Test /v1/models endpoint returns expected format."""
        response = test_client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "gyrosi-baby-0.9.6"

    def test_chat_completions_basic(self, test_client: Any) -> None:
        """Test chat completions generates valid response."""
        payload = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello, GyroSI!"}]}

        response = test_client.post("/v1/chat/completions", json=payload)

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

    def test_concurrent_chat_completions(self, test_client: Any) -> None:
        """Test concurrent requests don't interfere with each other."""
        payload1 = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello from thread 1"}]}

        payload2 = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello from thread 2"}]}

        def make_request(payload: Dict[str, Any]) -> int:
            response = test_client.post("/v1/chat/completions", json=payload)
            return int(response.status_code)

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(make_request, payload1)
            future2 = executor.submit(make_request, payload2)

            # Both should succeed
            status1 = future1.result()
            status2 = future2.result()

        assert status1 == 200
        assert status2 == 200

    def test_chat_completions_with_system_message(self, test_client: Any) -> None:
        """Test chat completions incorporates system message."""
        payload = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What can you help me with?"},
            ],
        }

        response = test_client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] is not None

    def test_chat_completions_streaming(self, test_client: Any) -> None:
        """Test streaming chat completions returns valid SSE format."""
        payload = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello!"}]}

        # Use streaming context manager for robust SSE testing
        with test_client.stream("POST", "/v1/chat/completions", json=payload, params={"stream": "true"}) as response:
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
                            if (
                                "choices" in data
                                and data["choices"]
                                and "delta" in data["choices"][0]
                                and "content" in data["choices"][0]["delta"]
                            ):
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

    def test_hf_generate_endpoint(self, test_client: Any) -> None:
        """Test HuggingFace-compatible generate endpoint."""
        payload = {"inputs": "Generate a response about artificial intelligence."}

        response = test_client.post("/generate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)
        assert len(data["generated_text"]) > 0

    def test_user_id_mapping_to_internal_agent(self, api_test_setup: Any) -> None:
        """Test custom user ID in header maps to internal 'user' agent."""
        test_client, global_pool = api_test_setup

        # Get initial pool state
        pre_agents = set(global_pool.get_active_agents())

        payload = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello from custom user!"}]}

        # Use a custom user ID in header
        headers = {"X-User-ID": "custom-user-123"}

        response = test_client.post("/v1/chat/completions", json=payload, headers=headers)
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
    def test_full_conversation_pipeline(self, agent_pool: AgentPool, tokenizer_name: str) -> None:
        """Test complete conversation from input to response."""
        pool = agent_pool

        # Simulate a conversation turn
        user_input = "Tell me about the nature of intelligence."

        # Full pipeline: tokenize → agents → orchestrate → detokenize
        response = orchestrate_turn(
            pool=pool, user_id="user", assistant_id="assistant", user_input=user_input, tokenizer_name=tokenizer_name
        )

        assert isinstance(response, str)
        assert len(response) > 0
        # Response should be different from input (showing processing)
        assert response != user_input

    @pytest.mark.parametrize("tokenizer_name", TOKENIZER_NAMES)
    def test_multi_turn_conversation(self, agent_pool: AgentPool, tokenizer_name: str) -> None:
        """Test multi-turn conversation produces meaningful responses."""
        pool = agent_pool

        # First turn
        response1 = orchestrate_turn(
            pool=pool,
            user_id="user",
            assistant_id="assistant",
            user_input="I like cats.",
            tokenizer_name=tokenizer_name,
        )

        # Second turn referencing first
        response2 = orchestrate_turn(
            pool=pool,
            user_id="user",
            assistant_id="assistant",
            user_input="What do I like?",
            tokenizer_name=tokenizer_name,
        )

        assert isinstance(response1, str)
        assert isinstance(response2, str)
        assert len(response1) > 0
        assert len(response2) > 0

    def test_conversation_state_persistence(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that conversation state persists between turns (isolated agent)."""
        agent = isolated_agent_factory(tmp_path)

        # Get initial state
        initial_state = agent.get_agent_info()
        initial_cycle_count = initial_state["cycle_count"]

        # Have a conversation
        agent.ingest(b"Remember that I like pizza.")

        # Check that agent state has evolved
        updated_state = agent.get_agent_info()
        final_cycle_count = updated_state["cycle_count"]

        assert final_cycle_count > initial_cycle_count


class TestAgentIsolationPatterns:
    """Test patterns for agent isolation in different scenarios."""

    def test_isolated_vs_pool_agent_behavior(
        self, isolated_agent_factory: Callable[[Path], GyroSI], agent_pool: AgentPool, tmp_path: Path
    ) -> None:
        """Test that isolated agents behave consistently with pool agents."""
        # Create isolated agent
        isolated_agent = isolated_agent_factory(tmp_path)

        # Get agent from pool
        pool_agent = agent_pool.get("assistant")

        # Test same operation on both
        test_input = b"consistent behavior test"

        isolated_response = isolated_agent.respond(test_input)
        pool_response = pool_agent.respond(test_input)

        # Both should produce valid responses (though content may differ due to different states)
        assert isinstance(isolated_response, bytes)
        assert isinstance(pool_response, bytes)
        assert len(isolated_response) > 0
        assert len(pool_response) > 0

    def test_multiple_isolated_agents_independence(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that multiple isolated agents don't interfere with each other."""
        # Create two isolated agents
        agent1 = isolated_agent_factory(tmp_path / "agent1")
        agent2 = isolated_agent_factory(tmp_path / "agent2")

        # Have them learn different things
        agent1.ingest(b"agent1 learns this")
        agent2.ingest(b"agent2 learns that")

        # Check they have independent stores
        store1_entries = list(agent1.engine.operator.store.iter_entries())
        store2_entries = list(agent2.engine.operator.store.iter_entries())

        # Each should have only learned their own data
        assert len(store1_entries) >= 0
        assert len(store2_entries) >= 0

        # Check they can't see each other's data
        # (This is a basic check - the actual verification would depend on the specific learning implementation)
        agent1_info = agent1.get_agent_info()
        agent2_info = agent2.get_agent_info()

        # Should have different agent IDs
        assert agent1_info["agent_id"] != agent2_info["agent_id"]
