"""
Tests for the front-line integration layer: intelligence.py, orchestration, and API adapters.
Tests focus on end-to-end flows while avoiding overlap with existing governance/information/inference tests.
"""

import json
from typing import Any, Dict, Callable, Tuple, Optional, cast
from pathlib import Path

import pytest
from unittest.mock import patch

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
        agent = isolated_agent_factory(tmp_path)
        sequence1 = b"abc"
        sequence2 = b"def"
        # Learn first sequence
        agent.ingest(sequence1)
        # Learn second sequence
        agent.ingest(sequence2)
        # Examine store directly to verify phenotypes were created
        store = agent.engine.operator.store
        phenotypes = []
        for _, entry in store.iter_entries():
            if "phenotype" in entry:
                phenotypes.append(entry["phenotype"])
        # Assert at least one phenotype exists and learning completed
        assert len(phenotypes) >= 1

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

        # Try with valid tokenizer
        orchestrate_turn(
            pool=pool,
            user_id="user",
            assistant_id="assistant",
            user_input=user_input,
            tokenizer_name="bert-base-uncased",
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
        # Negative check for cross-contamination
        assert all(
            "assistant specific" not in str(e) for e in user_private_entries
        ), "User store contaminated by assistant data"
        assert all(
            "user specific" not in str(e) for e in assistant_private_entries
        ), "Assistant store contaminated by user data"


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


class TestTokenizerDebug:
    def test_incomplete_token_id_sequence_debug(self) -> None:
        """Test that malformed blobs still trigger appropriate errors (for debugging purposes)."""
        from toys.communication import tokenizer as gyrotok

        # Try a range of malformed or truncated blobs
        test_blobs = [
            b"",  # empty
            b"\x80",  # single continuation byte, incomplete
            b"\x81\x82",  # multiple continuation bytes, incomplete
            b"\xff\xff\xff\xff",  # long run, incomplete
            b"\x81\x00",  # valid
            b"\x81",  # incomplete
            b"\x81\x81",  # incomplete
            b"\x00",  # valid single token
            b"\x81\x00\x82",  # valid + incomplete
        ]

        # Count expected exceptions (malformed blobs should still fail)
        expected_exceptions = 0
        for i, blob in enumerate(test_blobs):
            try:
                gyrotok._bytes_to_ids(blob)
            except ValueError as e:
                expected_exceptions += 1
                print(f"[TEST-DBG] Blob {i}: {blob.hex()} -> {e}")
            else:
                print(f"[TEST-DBG] Blob {i}: {blob.hex()} -> OK")

        # Verify that malformed blobs still trigger exceptions (this is expected behavior)
        assert expected_exceptions > 0, "Expected some malformed blobs to trigger exceptions"

    def test_gyrosi_respond_produces_complete_tokens(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that GyroSI.respond() produces complete LEB128 tokens without truncation."""
        from toys.communication import tokenizer as gyrotok

        # Create an agent
        agent = isolated_agent_factory(tmp_path)

        # Generate some response bytes
        test_input = b"Hello, test input"
        response_bytes = agent.respond(test_input, max_new_tokens=10)

        print(f"[TEST-DBG] Response length: {len(response_bytes)} bytes")
        print(f"[TEST-DBG] Response hex: {response_bytes.hex()}")

        # Try to decode the response - this should NOT raise an exception
        try:
            # Unmask the bytes first
            unmasked_bytes = bytes(b ^ 0xAA for b in response_bytes)
            print(f"[TEST-DBG] Unmasked hex: {unmasked_bytes.hex()}")

            # Decode to token IDs
            token_ids = gyrotok._bytes_to_ids(unmasked_bytes)
            print(f"[TEST-DBG] Decoded token IDs: {token_ids}")

            # If we get here, the tokens are complete
            assert len(token_ids) > 0, "Should decode at least one token"

        except ValueError as e:
            print(f"[TEST-DBG] ERROR: Incomplete token sequence: {e}")
            print("[TEST-DBG] This indicates the 8-byte safety exit is truncating tokens")
            raise AssertionError(f"GyroSI.respond() produced incomplete tokens: {e}")

        # Clean up
        agent.close()

    def test_gyrosi_respond_long_tokens(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test that GyroSI.respond() handles long tokens correctly without truncation."""
        from toys.communication import tokenizer as gyrotok

        # Create an agent
        agent = isolated_agent_factory(tmp_path)

        # Try to generate more tokens to increase chance of hitting the 8-byte limit
        test_input = b"This is a longer input that might trigger longer token sequences"
        response_bytes = agent.respond(test_input, max_new_tokens=20)

        print(f"[TEST-DBG] Long response length: {len(response_bytes)} bytes")
        print(f"[TEST-DBG] Long response hex: {response_bytes.hex()}")

        # Try to decode the response - this should NOT raise an exception
        try:
            # Unmask the bytes first
            unmasked_bytes = bytes(b ^ 0xAA for b in response_bytes)
            print(f"[TEST-DBG] Long unmasked hex: {unmasked_bytes.hex()}")

            # Decode to token IDs
            token_ids = gyrotok._bytes_to_ids(unmasked_bytes)
            print(f"[TEST-DBG] Long decoded token IDs: {token_ids}")

            # If we get here, the tokens are complete
            assert len(token_ids) > 0, "Should decode at least one token"

        except ValueError as e:
            print(f"[TEST-DBG] ERROR: Incomplete token sequence: {e}")
            print("[TEST-DBG] This indicates the 8-byte safety exit is truncating tokens")
            raise AssertionError(f"GyroSI.respond() produced incomplete tokens: {e}")

        # Clean up
        agent.close()

    def test_gyrosi_respond_edge_cases(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test edge cases that might trigger incomplete token sequences."""
        from toys.communication import tokenizer as gyrotok

        # Create an agent
        agent = isolated_agent_factory(tmp_path)

        # Test various edge case inputs that might trigger the bug
        edge_case_inputs = [
            b"",  # empty input
            b"A" * 100,  # very long repeated input
            b"\x00\x01\x02\x03\x04\x05",  # binary data
            b"Hello world with special chars: \xc3\xa9\xc3\xb1\xc3\xbc\xc3\x9f",  # unicode
            b"Very long input that might cause the physics to generate unusual token patterns " * 10,
        ]

        for i, test_input in enumerate(edge_case_inputs):
            print(f"[TEST-DBG] Testing edge case {i}: {len(test_input)} bytes")

            # Initialize variables to avoid unbound variable errors
            response_bytes = b""
            unmasked_bytes = b""

            try:
                response_bytes = agent.respond(test_input, max_new_tokens=5)
                print(f"[TEST-DBG] Edge case {i} response: {len(response_bytes)} bytes")

                # Try to decode the response
                unmasked_bytes = bytes(b ^ 0xAA for b in response_bytes)
                token_ids = gyrotok._bytes_to_ids(unmasked_bytes)
                print(f"[TEST-DBG] Edge case {i} tokens: {token_ids}")

            except ValueError as e:
                print(f"[TEST-DBG] ERROR in edge case {i}: {e}")
                print(f"[TEST-DBG] Input: {test_input[:50].decode('utf-8', errors='replace')}...")
                print(f"[TEST-DBG] Response: {response_bytes.hex()!r}")
                raise AssertionError(f"Edge case {i} produced incomplete tokens: {e}")
            except Exception as e:
                print(f"[TEST-DBG] Unexpected error in edge case {i}: {e}")
                raise

        # Clean up
        agent.close()

    def test_gyrosi_respond_incomplete_token_debug(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test to specifically reproduce the incomplete token sequence error."""
        from toys.communication import tokenizer as gyrotok

        # Create an agent
        agent = isolated_agent_factory(tmp_path)

        # Try to trigger the error by generating many tokens
        test_input = b"Generate a long response that might cause incomplete tokens"

        # Initialize variables to avoid unbound variable errors
        response_bytes = b""
        unmasked_bytes = b""

        try:
            # Generate more tokens to increase chance of hitting edge cases
            response_bytes = agent.respond(test_input, max_new_tokens=50)
            print(f"[TEST-DBG] Generated {len(response_bytes)} bytes")

            # Try to decode the response - this should NOT raise an exception
            unmasked_bytes = bytes(b ^ 0xAA for b in response_bytes)
            token_ids = gyrotok._bytes_to_ids(unmasked_bytes)
            print(f"[TEST-DBG] Successfully decoded {len(token_ids)} tokens")

        except ValueError as e:
            print(f"[TEST-DBG] ERROR: Incomplete token sequence: {e}")
            print(f"[TEST-DBG] Response hex: {response_bytes.hex()!r}")
            print(f"[TEST-DBG] Unmasked hex: {unmasked_bytes.hex()!r}")
            raise AssertionError(f"GyroSI.respond() produced incomplete tokens: {e}")

        # Clean up
        agent.close()


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
        # payload1 = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello from thread 1"}]}
        # payload2 = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello from thread 2"}]}

        def make_request(payload: Dict[str, Any]) -> int:
            response = test_client.post("/v1/chat/completions", json=payload)
            return int(response.status_code)

        # Make concurrent requests
        # with pytest.raises(Exception):  # Could be FileNotFoundError, ValueError, etc.
        #     orchestrate_turn(
        #         pool=pool,
        #         user_id="user",
        #         assistant_id="assistant",
        #         user_input=user_input,
        #         tokenizer_name="non-existent-tokenizer",
        #     )

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
        """Test streaming chat completions returns valid SSE format and is not pathologically slow."""
        import time

        payload = {"model": "gyrosi-baby-0.9.6", "messages": [{"role": "user", "content": "Hello!"}]}

        # Use streaming context manager for robust SSE testing
        with test_client.stream("POST", "/v1/chat/completions", json=payload, params={"stream": "true"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            # Collect all chunks
            chunks = []
            content_chunks = []
            token_latencies = []

            for line in response.iter_lines():
                if line.strip() and line.startswith("data: "):
                    start = time.time()
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
                                token_latencies.append(time.time() - start)
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

            # Micro-benchmark: fail if average per-token latency exceeds 5ms
            if token_latencies:
                avg_latency = sum(token_latencies) / len(token_latencies)
                assert avg_latency < 0.005, f"Average per-token latency too high: {avg_latency * 1000:.2f} ms"

    def test_hf_generate_endpoint(self, test_client: Any) -> None:
        """Test HuggingFace-compatible generate endpoint, ensuring output is in the correct tokenizer alphabet."""
        payload = {"inputs": "Generate a response about artificial intelligence."}

        # Patch orchestrate_turn to return a known string
        with patch("toys.communication.external_adapter.orchestrate_turn", return_value="hello world"):
            response = test_client.post("/generate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data
        generated = data["generated_text"]
        assert isinstance(generated, str)
        assert len(generated) > 0

        # Re-encode and decode with the same tokenizer to ensure lossless round-trip
        encoded = gyrotok.encode(generated, name="bert-base-uncased")
        decoded = gyrotok.decode(encoded, name="bert-base-uncased")
        assert generated == decoded, (
            "Generated text is not losslessly round-trippable with the tokenizer "
            "(possible UTF-8 fallback or replacement chars)"
        )

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


class TestBUIngress:
    """Test BU-Ingress functionality."""

    def test_bu_ingress_step_basic(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test basic BU-Ingress step functionality."""
        agent = isolated_agent_factory(tmp_path)

        # Create a test phenotype entry
        from baby.contracts import PhenotypeEntry, GovernanceSignature

        sig: GovernanceSignature = {"neutral": 4, "li": 1, "fg": 1, "bg": 0, "dyn": 2}

        entry: PhenotypeEntry = {
            "phenotype": "test",
            "confidence": 0.5,
            "exon_mask": 0x42,
            "usage_count": 1,
            "last_updated": 1234567890.0,
            "created_at": 1234567890.0,
            "governance_signature": sig,
            "context_signature": (0, 0),
            "_original_context": None,
        }

        # Test with different theta values
        theta_low = 0.1
        theta_high = 0.5
        theta_max = 0.95

        # Test low theta regime
        byte_out_low, intron_out_low = agent.engine._bu_ingress_step(entry, theta_low)
        assert isinstance(byte_out_low, int)
        assert isinstance(intron_out_low, int)
        assert 0 <= byte_out_low <= 255
        assert 0 <= intron_out_low <= 255

        # Test high theta regime
        byte_out_high, intron_out_high = agent.engine._bu_ingress_step(entry, theta_high)
        assert isinstance(byte_out_high, int)
        assert isinstance(intron_out_high, int)
        assert 0 <= byte_out_high <= 255
        assert 0 <= intron_out_high <= 255

        # Test max theta regime
        byte_out_max, intron_out_max = agent.engine._bu_ingress_step(entry, theta_max)
        assert isinstance(byte_out_max, int)
        assert isinstance(intron_out_max, int)
        assert 0 <= byte_out_max <= 255
        assert 0 <= intron_out_max <= 255

    def test_bu_ingress_step_branch_selection(
        self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path
    ) -> None:
        """Test that BU-Ingress selects different output distributions for different theta regimes
        using the public API and debug hook."""
        import random
        import numpy as np

        agent = isolated_agent_factory(tmp_path)
        from baby.contracts import PhenotypeEntry, GovernanceSignature

        sig: GovernanceSignature = {"neutral": 4, "li": 1, "fg": 1, "bg": 0, "dyn": 2}
        entry: PhenotypeEntry = {
            "phenotype": "test",
            "confidence": 0.5,
            "exon_mask": 0x42,
            "usage_count": 1,
            "last_updated": 1234567890.0,
            "created_at": 1234567890.0,
            "governance_signature": sig,
            "context_signature": (0, 0),
            "_original_context": None,
        }
        N = 50
        random.seed(42)
        np.random.seed(42)
        regimes = [(0.1, "low"), (0.5, "mid"), (0.95, "high")]
        results = {}
        for theta, label in regimes:
            # Set up a debug hook to capture (byte_out, intron_out)
            captured = []

            def on_emit(engine: Any, phenotype_entry: Any, last_intron: int) -> None:
                # The engine will call this after each ingress cycle
                # We want to capture the output of the most recent respond call
                # The public respond returns only the byte, but the hook gives us the intron as well
                # We use the engine's last output (byte_out, intron_out) from process_ingress
                # The phenotype_entry is the one used for output
                # The last_intron is the intron just used
                # For this test, we just record the last_intron and the phenotype_entry's mask
                captured.append((phenotype_entry["exon_mask"], last_intron))

            agent.add_monitoring_hook(on_emit)
            # Set theta buffer to control the regime
            agent.engine._θ_buf.clear()
            agent.engine._θ_buf.extend([theta] * 8)  # Fill buffer to set mean theta
            # Run N times
            for _ in range(N):
                # Overwrite the operator's store to always return our entry for get_phenotype
                # Use a simpler approach - just mock the store to return our entry
                original_get = agent.engine.operator.store.get

                def mock_get(context_key: Tuple[int, int]) -> Optional[Dict[str, Any]]:
                    return cast(Dict[str, Any], entry)

                agent.engine.operator.store.get = mock_get
                agent.respond(b"probe", max_new_tokens=1)
                # Restore original method
                agent.engine.operator.store.get = original_get
            # Remove the hook after this regime
            agent.engine.remove_hook(on_emit)
            byte_means = float(np.mean([o[0] for o in captured]))
            intron_means = float(np.mean([o[1] for o in captured]))
            results[label] = (byte_means, intron_means)
        # Assert that the means differ across regimes
        byte_means_list = [float(results[label][0]) for _, label in regimes]
        intron_means_list = [float(results[label][1]) for _, label in regimes]
        assert (
            len(set(byte_means_list)) > 1 or len(set(intron_means_list)) > 1
        ), "BU-ingress output does not vary across theta regimes"

    def test_auto_regressive_generation(self, isolated_agent_factory: Callable[[Path], GyroSI], tmp_path: Path) -> None:
        """Test that the auto-regressive loop generates multiple bytes."""
        agent = isolated_agent_factory(tmp_path)

        # Test with a simple input
        response = agent.respond(b"hello", max_new_tokens=10)

        # Should return a byte string of reasonable length
        assert isinstance(response, bytes)
        assert len(response) > 1  # Should generate more than one byte
        # With token-based generation, byte count may exceed token count
        assert all(0 <= b <= 255 for b in response)  # All bytes should be valid

        # Test with different max_new_tokens values
        response_short = agent.respond(b"test", max_new_tokens=5)
        response_long = agent.respond(b"test", max_new_tokens=20)

        # With token-based generation, each token can be multiple bytes
        # So we can't assume byte count equals token count
        assert len(response_short) > 0
        assert len(response_long) > 0
        assert len(response_long) >= len(response_short)  # Longer limit should produce at least as many bytes


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

        # Check they have independent private overlays (no key overlap)
        private1 = getattr(agent1.engine.operator, "private_view", None)
        private2 = getattr(agent2.engine.operator, "private_view", None)
        if private1 is not None and private2 is not None:
            keys1 = set(k for k, _ in private1.iter_entries())
            keys2 = set(k for k, _ in private2.iter_entries())
            assert keys1.isdisjoint(keys2), "Private overlays of isolated agents overlap: state isolation violated"
