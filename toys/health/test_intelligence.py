"""
Tests for the front-line integration layer: intelligence.py, orchestration, and API adapters.
Tests focus on end-to-end flows with token-aware architecture.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import patch

import pytest

from baby.contracts import PhenotypeEntry
from baby.intelligence import AgentPool, GyroSI, orchestrate_turn
from toys.communication import tokenizer as gyrotok

# Test configuration
TOKENIZER_NAME = "bert-base-uncased"
TEST_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "What is the nature of intelligence?",
    "I like cats and dogs",
    "This is a test of the token-aware system",
]


def _count_store_entries(store: Any) -> int:
    """Helper to count entries in a store."""
    return sum(1 for _ in store.iter_entries())


def _get_store_keys(store: Any) -> List[Tuple[int, int]]:
    """Helper to get all keys from a store."""
    return [key for key, _ in store.iter_entries()]


class TestTokenAwareIntelligence:
    """Test core token-aware intelligence engine behavior."""

    def test_token_aware_phenotype_creation(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that token-aware learning creates phenotypes with proper keys."""
        agent = isolated_agent_factory(tmp_path)

        # Use tokenized input
        test_text = "hello world test"
        tokenized_input = gyrotok.encode(test_text, name=TOKENIZER_NAME)
        agent.ingest(tokenized_input)

        # Verify phenotypes use (state_index, token_id) keys
        store = agent.engine.operator.store
        keys = _get_store_keys(store)

        assert len(keys) > 0, "Should create at least one phenotype"

        for state_idx, token_id in keys:
            assert isinstance(state_idx, int), f"State index should be int, got {type(state_idx)}"
            assert isinstance(token_id, int), f"Token ID should be int, got {type(token_id)}"
            assert 0 <= state_idx < 788_986, f"State index {state_idx} out of valid range"
            assert 0 <= token_id <= 30522, f"Token ID {token_id} outside BERT vocab range"

    def test_different_tokens_create_different_phenotypes(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that different token sequences create distinct phenotypes."""
        agent = isolated_agent_factory(tmp_path)

        # Process different texts
        texts = ["apple", "orange", "banana"]
        all_keys = set()

        for text in texts:
            tokenized = gyrotok.encode(text, name=TOKENIZER_NAME)
            agent.ingest(tokenized)

            # Collect keys after each ingestion
            current_keys = _get_store_keys(agent.engine.operator.store)
            all_keys.update(current_keys)

        # Should have multiple unique keys (different tokens create different phenotypes)
        assert len(all_keys) >= len(texts), "Different tokens should create distinct phenotypes"

    def test_token_boundary_detection(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test proper LEB128 token boundary detection."""
        agent = isolated_agent_factory(tmp_path)

        # Start with clean buffer
        agent.engine.reset_token_buffer()
        assert len(agent.engine._byte_buf) == 0
        assert agent.engine._last_token_id == 0

        # Process a complete token sequence
        test_text = "test"
        tokenized = gyrotok.encode(test_text, name=TOKENIZER_NAME)

        for byte_val in tokenized:
            agent.engine.process_egress(byte_val)

        # Buffer should be clear after complete token
        assert len(agent.engine._byte_buf) == 0, "Buffer should clear after complete token"

    def test_buffer_overflow_protection(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test buffer overflow protection mechanism."""
        agent = isolated_agent_factory(tmp_path)

        # Reset to clean state
        agent.engine.reset_token_buffer()

        # Try to overflow buffer with continuation bytes
        max_bytes = agent.engine.MAX_TOKEN_BYTES

        for i in range(max_bytes + 2):
            # Send bytes that will result in continuation bytes after transcription
            # transcribe_byte does byte ^ 0xAA, and we want the result to have bit 7 = 1 (continuation)
            # Since 0xAA has bit 7 = 1, we need byte to have bit 7 = 0 (because 0 ^ 1 = 1)
            byte_val = 0x00 | (i & 0x7F)  # This has bit 7 = 0
            # After transcription: byte_val ^ 0xAA = i ^ 0xAA
            # For i=0: 0 ^ 0xAA = 0xAA (bit 7 = 1, continuation)
            # For i=1: 1 ^ 0xAA = 0xAB (bit 7 = 1, continuation)
            # etc.
            agent.engine.process_egress(byte_val)

        # Buffer should have been cleared due to overflow protection
        assert len(agent.engine._byte_buf) <= max_bytes, "Buffer overflow protection should activate"
        # The buffer should be small after overflow protection (the intron causing overflow is discarded)
        assert len(agent.engine._byte_buf) < max_bytes, "Buffer should be below limit after overflow protection"


class TestTokenGeneration:
    """Test token-aware response generation."""

    def test_complete_token_generation(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test that responses contain complete LEB128 tokens."""
        agent = isolated_agent_factory(tmp_path)

        # Generate response with tokenized input
        input_text = "Generate a response"
        tokenized_input = gyrotok.encode(input_text, name=TOKENIZER_NAME)
        response_bytes = agent.respond(tokenized_input, max_new_tokens=5)

        # Verify response contains complete tokens
        assert isinstance(response_bytes, bytes), "Response should be bytes"
        assert len(response_bytes) > 0, "Response should not be empty"

        # Should be decodable as complete token sequence
        try:
            decoded_tokens = gyrotok.bytes_to_ids(response_bytes)
            assert len(decoded_tokens) > 0, "Should decode at least one token"
            assert all(isinstance(tok_id, int) for tok_id in decoded_tokens), "All tokens should be integers"
        except ValueError as e:
            pytest.fail(f"Response contains incomplete tokens: {e}")

    def test_generated_tokens_are_valid(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test that generated tokens are within valid vocabulary range."""
        agent = isolated_agent_factory(tmp_path)

        input_text = "Test vocabulary"
        tokenized_input = gyrotok.encode(input_text, name=TOKENIZER_NAME)
        response_bytes = agent.respond(tokenized_input, max_new_tokens=10)

        # Decode and validate token IDs
        decoded_tokens = gyrotok.bytes_to_ids(response_bytes)

        for token_id in decoded_tokens:
            assert 0 <= token_id <= 30522, f"Token ID {token_id} outside BERT vocabulary range"

    def test_response_roundtrip_consistency(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that generated responses can roundtrip through tokenizer."""
        agent = isolated_agent_factory(tmp_path)

        input_text = "Roundtrip test"
        tokenized_input = gyrotok.encode(input_text, name=TOKENIZER_NAME)
        response_bytes = agent.respond(tokenized_input, max_new_tokens=8)

        # Decode to text and re-encode
        response_text = gyrotok.decode(response_bytes, name=TOKENIZER_NAME)
        re_encoded = gyrotok.encode(response_text, name=TOKENIZER_NAME)

        # Check that decode-encode-decode is idempotent (second decode equals first decode)
        # This verifies that the tokenizer can roundtrip the text without loss
        # Note: Tokenizer may normalize spacing around special tokens, so we normalize for comparison
        roundtrip_text = gyrotok.decode(re_encoded, name=TOKENIZER_NAME)
        # Normalize by removing all spaces to handle tokenizer spacing differences
        normalized_response = response_text.replace(" ", "")
        normalized_roundtrip = roundtrip_text.replace(" ", "")
        assert normalized_roundtrip == normalized_response, "Tokenizer decode should be idempotent"


class TestAgentLifecycle:
    """Test GyroSI agent lifecycle with proper isolation."""

    def test_agent_initialization_state(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test agent initializes with correct default state."""
        agent = isolated_agent_factory(tmp_path)

        state_info = agent.get_agent_info()

        assert state_info["cycle_count"] == 0, "New agent should start with zero cycles"
        # The archetypal state is NOT at index 0; its index is determined by sorting all discovered states.
        assert state_info["tensor_index"] == 549871, "Should start at archetypal state (GENE_Mac_S)"
        assert state_info["system_integrity"] is not None, "Should report integrity status"
        assert "agent_id" in state_info, "Should have agent ID"

    def test_state_evolution_through_learning(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that agent state evolves through learning."""
        agent = isolated_agent_factory(tmp_path)

        initial_state = agent.get_agent_info()
        initial_cycles = initial_state["cycle_count"]

        # Learn from tokenized input
        for text in TEST_TEXTS[:3]:
            tokenized = gyrotok.encode(text, name=TOKENIZER_NAME)
            agent.ingest(tokenized)

        final_state = agent.get_agent_info()
        final_cycles = final_state["cycle_count"]

        assert final_cycles > initial_cycles, "Cycle count should increase through learning"

    def test_agent_responds_to_all_inputs(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test agent produces valid responses for various inputs."""
        agent = isolated_agent_factory(tmp_path)

        for test_text in TEST_TEXTS:
            tokenized_input = gyrotok.encode(test_text, name=TOKENIZER_NAME)
            response = agent.respond(tokenized_input, max_new_tokens=5)

            assert isinstance(response, bytes), f"Response should be bytes for input '{test_text}'"
            assert len(response) > 0, f"Response should not be empty for input '{test_text}'"

    def test_agent_cleanup_and_isolation(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test proper agent cleanup and isolation."""
        agent1 = isolated_agent_factory(tmp_path / "agent1")
        agent2 = isolated_agent_factory(tmp_path / "agent2")

        # Different agents should have different IDs
        info1 = agent1.get_agent_info()
        info2 = agent2.get_agent_info()
        assert info1["agent_id"] != info2["agent_id"], "Isolated agents should have different IDs"

        # Learn different things
        text1 = "Agent one learns this"
        text2 = "Agent two learns that"

        agent1.ingest(gyrotok.encode(text1, name=TOKENIZER_NAME))
        agent2.ingest(gyrotok.encode(text2, name=TOKENIZER_NAME))

        # Should have independent stores
        keys1 = set(_get_store_keys(agent1.engine.operator.store))
        keys2 = set(_get_store_keys(agent2.engine.operator.store))

        # May have some overlap due to shared ontology, but should have some differences
        assert len(keys1) > 0, "Agent 1 should have learned something"
        assert len(keys2) > 0, "Agent 2 should have learned something"


class TestHookSystem:
    """Test intelligence engine hook system."""

    def test_hooks_called_with_proper_context(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test hooks receive proper token-aware context."""
        agent = isolated_agent_factory(tmp_path)
        captured_calls: List[Dict[str, Any]] = []

        def test_hook(engine: Any, phenotype_entry: PhenotypeEntry, last_token_byte: int) -> None:
            captured_calls.append(
                {"phenotype": phenotype_entry, "last_byte": last_token_byte, "call_count": len(captured_calls) + 1}
            )

        agent.add_monitoring_hook(test_hook)

        # Process tokenized input that should trigger hooks
        test_input = gyrotok.encode("hook test", name=TOKENIZER_NAME)
        agent.respond(test_input, max_new_tokens=3)

        # Should have captured hook calls
        assert len(captured_calls) > 0, "Hooks should be called during response generation"

        for call_data in captured_calls:
            assert "phenotype" in call_data, "Hook should receive phenotype"
            assert "last_byte" in call_data, "Hook should receive last byte"
            assert isinstance(call_data["last_byte"], int), "Last byte should be integer"

    def test_hook_removal(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test hook removal functionality."""
        agent = isolated_agent_factory(tmp_path)
        call_count = 0

        def test_hook(engine: Any, phenotype_entry: PhenotypeEntry, last_token_byte: int) -> None:
            nonlocal call_count
            call_count += 1

        # Add and remove hook
        agent.add_monitoring_hook(test_hook)
        removed = agent.engine.remove_hook(test_hook)
        assert removed, "Hook removal should return True when hook found"

        # Process input - should not trigger removed hook
        test_input = gyrotok.encode("test", name=TOKENIZER_NAME)
        agent.respond(test_input)

        assert call_count == 0, "Removed hook should not be called"


class TestAgentPoolOrchestration:
    """Test multi-agent orchestration and pool management."""

    def test_pool_triad_exists(self, agent_pool: AgentPool) -> None:
        """Test agent pool maintains required triad."""
        active_agents = agent_pool.get_active_agents()
        required_agents = {"user", "system", "assistant"}

        assert required_agents.issubset(set(active_agents)), "Pool should maintain triad agents"

    def test_pool_agent_consistency(self, agent_pool: AgentPool) -> None:
        """Test pool returns consistent agent instances."""
        user1 = agent_pool.get("user")
        user2 = agent_pool.get("user")

        assert user1 is user2, "Pool should return same instance for same ID"

    def test_orchestrated_conversation_flow(self, agent_pool: AgentPool) -> None:
        """Test complete orchestrated conversation."""
        user_input = "What is artificial intelligence?"

        response = orchestrate_turn(
            pool=agent_pool,
            user_id="user",
            assistant_id="assistant",
            user_input=user_input,
            tokenizer_name=TOKENIZER_NAME,
        )

        assert isinstance(response, str), "Orchestrated response should be string"
        assert len(response) > 0, "Response should not be empty"
        assert response != user_input, "Response should differ from input"

    def test_multi_turn_conversation_memory(self, agent_pool: AgentPool) -> None:
        """Test multi-turn conversation maintains context."""
        # First turn
        response1 = orchestrate_turn(
            pool=agent_pool,
            user_id="user",
            assistant_id="assistant",
            user_input="My favorite color is blue.",
            tokenizer_name=TOKENIZER_NAME,
        )

        # Second turn referencing first
        response2 = orchestrate_turn(
            pool=agent_pool,
            user_id="user",
            assistant_id="assistant",
            user_input="What is my favorite color?",
            tokenizer_name=TOKENIZER_NAME,
        )

        assert isinstance(response1, str), "First response should be string"
        assert isinstance(response2, str), "Second response should be string"
        assert len(response1) > 0, "First response should not be empty"
        assert len(response2) > 0, "Second response should not be empty"

    def test_agent_isolation_in_pool(self, agent_pool: AgentPool) -> None:
        """Test agents in pool maintain isolation."""
        user_agent = agent_pool.get("user")
        assistant_agent = agent_pool.get("assistant")

        # Get initial states
        user_info = user_agent.get_agent_info()
        assistant_info = assistant_agent.get_agent_info()

        assert user_info["agent_id"] != assistant_info["agent_id"], "Pool agents should have different IDs"

        # Have each learn something different
        user_text = "User learns this information"
        assistant_text = "Assistant learns that information"

        user_agent.ingest(gyrotok.encode(user_text, name=TOKENIZER_NAME))
        assistant_agent.ingest(gyrotok.encode(assistant_text, name=TOKENIZER_NAME))

        # Verify independent learning
        user_final = user_agent.get_agent_info()
        assistant_final = assistant_agent.get_agent_info()

        assert user_final["cycle_count"] > user_info["cycle_count"], "User agent should learn"
        assert assistant_final["cycle_count"] > assistant_info["cycle_count"], "Assistant agent should learn"


class TestTokenizerIntegration:
    """Test tokenizer integration with agent system."""

    def test_tokenizer_roundtrip_consistency(self) -> None:
        """Test tokenizer roundtrip preserves content."""
        for test_text in TEST_TEXTS:
            # Encode to bytes
            encoded = gyrotok.encode(test_text, name=TOKENIZER_NAME)

            # Decode back to text
            decoded = gyrotok.decode(encoded, name=TOKENIZER_NAME)

            # Re-encode and compare
            re_encoded = gyrotok.encode(decoded, name=TOKENIZER_NAME)

            assert encoded == re_encoded, f"Roundtrip failed for text: '{test_text}'"

    def test_tokenizer_error_handling(self) -> None:
        """Test tokenizer handles malformed input gracefully."""
        # Invalid LEB128 bytes
        invalid_bytes = b"\xff\xff\xff"

        # Should not raise exception (falls back to UTF-8)
        result = gyrotok.decode(invalid_bytes, name=TOKENIZER_NAME)
        assert isinstance(result, str), "Should return string even for invalid input"

    def test_tokenizer_agent_compatibility(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test tokenizer works correctly with agent system."""
        agent = isolated_agent_factory(tmp_path)

        for test_text in TEST_TEXTS[:3]:  # Test subset to avoid long test
            # Encode and process with agent
            encoded = gyrotok.encode(test_text, name=TOKENIZER_NAME)
            response = agent.respond(encoded, max_new_tokens=3)

            # Should be decodable
            decoded_response = gyrotok.decode(response, name=TOKENIZER_NAME)
            assert isinstance(decoded_response, str), f"Response should decode to string for '{test_text}'"


class TestExternalAPIIntegration:
    """Test external API adapter functionality."""

    def test_models_endpoint_structure(self, test_client: Any) -> None:
        """Test /v1/models endpoint returns correct structure."""
        response = test_client.get("/v1/models")

        assert response.status_code == 200, "Models endpoint should return 200"
        data = response.json()

        assert "data" in data, "Response should have 'data' field"
        assert len(data["data"]) > 0, "Should list at least one model"
        assert data["data"][0]["id"] == "gyrosi-baby", "Should list correct model ID"

    def test_chat_completions_basic_functionality(self, test_client: Any) -> None:
        """Test basic chat completions functionality."""
        payload = {"model": "gyrosi-baby", "messages": [{"role": "user", "content": "Hello, GyroSI!"}]}

        response = test_client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200, "Chat completions should return 200"
        data = response.json()

        # Verify OpenAI-compatible structure
        required_fields = ["id", "object", "created", "model", "choices"]
        for field in required_fields:
            assert field in data, f"Response should have '{field}' field"

        assert data["object"] == "chat.completion", "Object type should be correct"
        assert len(data["choices"]) > 0, "Should have at least one choice"
        assert data["choices"][0]["message"]["role"] == "assistant", "Response should be from assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0, "Response should have content"

    def test_chat_completions_with_system_message(self, test_client: Any) -> None:
        """Test system message handling in chat completions."""
        payload = {
            "model": "gyrosi-baby",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What can you help with?"},
            ],
        }

        response = test_client.post("/v1/chat/completions", json=payload)

        assert response.status_code == 200, "System message handling should work"
        data = response.json()
        assert data["choices"][0]["message"]["content"] is not None, "Should generate response with system context"

    def test_streaming_chat_completions(self, test_client: Any) -> None:
        """Test streaming chat completions functionality."""
        payload = {"model": "gyrosi-baby", "messages": [{"role": "user", "content": "Stream test"}]}

        with test_client.stream("POST", "/v1/chat/completions", json=payload, params={"stream": "true"}) as response:
            assert response.status_code == 200, "Streaming should return 200"
            assert response.headers["content-type"].startswith("text/event-stream"), "Should use SSE content type"

            chunks = []
            content_parts = []

            for line in response.iter_lines():
                if line.strip() and line.startswith("data: "):
                    chunks.append(line)

                    if "data: [DONE]" not in line:
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            if (
                                "choices" in data
                                and data["choices"]
                                and "delta" in data["choices"][0]
                                and "content" in data["choices"][0]["delta"]
                            ):
                                content_parts.append(data["choices"][0]["delta"]["content"])
                        except json.JSONDecodeError:
                            continue

            assert len(chunks) > 0, "Should receive streaming chunks"
            assert any("data: [DONE]" in chunk for chunk in chunks), "Should end with DONE marker"
            assert len(content_parts) > 0, "Should receive content in chunks"

    def test_huggingface_generate_endpoint(self, test_client: Any) -> None:
        """Test HuggingFace-compatible generate endpoint."""
        payload = {"inputs": "Generate a response about AI"}

        with patch(
            "toys.communication.external_adapter.orchestrate_turn", return_value="artificial intelligence response"
        ):
            response = test_client.post("/generate", json=payload)

        assert response.status_code == 200, "Generate endpoint should return 200"
        data = response.json()

        assert "generated_text" in data, "Response should have generated_text field"
        assert isinstance(data["generated_text"], str), "Generated text should be string"
        assert len(data["generated_text"]) > 0, "Generated text should not be empty"


class TestMaintenanceOperations:
    """Test agent maintenance and lifecycle operations."""

    def test_agent_maintenance_operations(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test agent maintenance operations work correctly."""
        agent = isolated_agent_factory(tmp_path)

        # Learn some data first
        for text in TEST_TEXTS[:2]:
            tokenized = gyrotok.encode(text, name=TOKENIZER_NAME)
            agent.ingest(tokenized)

        # Run maintenance
        maintenance_result = agent.apply_maintenance(decay_rate=0.1, confidence_threshold=0.01)

        assert isinstance(maintenance_result, dict), "Maintenance should return dict"
        assert "timestamp" in maintenance_result, "Should include timestamp"

    def test_agent_state_reset(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test agent state reset functionality."""
        agent = isolated_agent_factory(tmp_path)

        # Learn something to change state
        test_input = gyrotok.encode("change state", name=TOKENIZER_NAME)
        agent.ingest(test_input)

        # Verify state changed
        changed_state = agent.get_agent_info()
        assert changed_state["cycle_count"] > 0, "State should have changed"

        # Reset to archetypal state
        agent.engine.reset_to_archetypal_state()

        # Verify reset
        reset_state = agent.get_agent_info()
        assert reset_state["cycle_count"] == 0, "Cycle count should reset"
        # NOTE: The archetypal state (GENE_Mac_S) is NOT at index 0 in the ontology;
        # its index is determined by its integer value after sorting all discovered states.
        assert reset_state["tensor_index"] == 549871, "Should return to archetypal state"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_input_handling(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test handling of edge case inputs."""
        agent = isolated_agent_factory(tmp_path)

        # Test with minimal input
        empty_response = agent.respond(b"", max_new_tokens=1)
        assert isinstance(empty_response, bytes), "Should handle empty input gracefully"

    def test_large_input_handling(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test handling of large inputs."""
        agent = isolated_agent_factory(tmp_path)

        # Create large input
        large_text = "This is a large input. " * 100
        large_input = gyrotok.encode(large_text, name=TOKENIZER_NAME)

        # Should handle without error
        response = agent.respond(large_input, max_new_tokens=5)
        assert isinstance(response, bytes), "Should handle large input"
        assert len(response) > 0, "Should generate response for large input"

    def test_invalid_tokenizer_name_handling(self, agent_pool: AgentPool) -> None:
        """Test error handling for invalid tokenizer names."""
        with pytest.raises((FileNotFoundError, ValueError)):
            orchestrate_turn(
                pool=agent_pool,
                user_id="user",
                assistant_id="assistant",
                user_input="test",
                tokenizer_name="nonexistent-tokenizer",
            )

    def test_concurrent_agent_access(self, agent_pool: AgentPool) -> None:
        """Test concurrent access to same agent doesn't cause issues."""
        agent = agent_pool.get("assistant")

        # Simulate concurrent operations
        responses = []
        for i in range(5):
            test_input = gyrotok.encode(f"concurrent test {i}", name=TOKENIZER_NAME)
            response = agent.respond(test_input, max_new_tokens=2)
            responses.append(response)

        # All responses should be valid
        assert len(responses) == 5, "Should handle concurrent requests"
        assert all(isinstance(r, bytes) and len(r) > 0 for r in responses), "All responses should be valid"


# Utility functions for test validation
def validate_phenotype_structure(entry: PhenotypeEntry) -> None:
    """Validate phenotype entry has correct structure."""
    required_fields = ["mask", "conf"]
    for field in required_fields:
        assert field in entry, f"Phenotype missing required field: {field}"

    assert isinstance(entry["mask"], int), "Phenotype mask should be int"
    assert 0 <= entry["mask"] <= 255, "Phenotype mask should be uint8"
    assert isinstance(entry["conf"], float), "Phenotype confidence should be float"
    assert 0.0 <= entry["conf"] <= 1.0, "Phenotype confidence should be in [0,1]"


def validate_token_sequence(token_ids: List[int]) -> None:
    """Validate token sequence is within vocabulary bounds."""
    for token_id in token_ids:
        assert isinstance(token_id, int), f"Token ID should be int, got {type(token_id)}"
        assert 0 <= token_id <= 30522, f"Token ID {token_id} outside BERT vocabulary"


def validate_agent_state_consistency(agent: GyroSI) -> None:
    """Validate agent state is internally consistent."""
    state_info = agent.get_agent_info()

    assert state_info["cycle_count"] >= 0, "Cycle count should be non-negative"
    assert 0 <= state_info["tensor_index"] < 788_986, "Tensor index should be in valid range"
    assert 0.0 <= state_info["angular_divergence_radians"] <= 3.15, "Angular divergence should be valid"
    assert state_info["system_integrity"] is not None, "System integrity should be checked"


# Export validation utilities for use in other test modules
__all__ = [
    "validate_phenotype_structure",
    "validate_token_sequence",
    "validate_agent_state_consistency",
    "TOKENIZER_NAME",
    "TEST_TEXTS",
]
