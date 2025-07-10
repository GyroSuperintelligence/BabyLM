"""
Tests for the intelligence engine, learning state, and thread processing in the BabyLM system.
Covers format management, input processing, response generation, and thread statistics.
"""

import os
import uuid
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from typing import cast

# Import modules from baby package
from baby.inference import InferenceEngine
from baby.information import assign_agent_uuid, shard_path
from baby.intelligence import initialize_intelligence_engine
from baby.governance import derive_canonical_patterns
from baby.types import ThreadMetadata, ChildRef
from baby.intelligence import weighted_choice


# Helper to coerce any value to a flat list of strings
def to_flat_str_list(val) -> list[str]:
    if isinstance(val, list):
        flat = []
        for v in val:
            if isinstance(v, str):
                flat.append(v)
            elif isinstance(v, list):
                flat.extend([x for x in v if isinstance(x, str)])
        return flat
    elif isinstance(val, str):
        return [val]
    else:
        return []


# ------------------------------------------------------------------------------
# Intelligence (S4) Tests
# ------------------------------------------------------------------------------


class TestIntelligence:
    """Tests for the Intelligence layer (orchestration)"""

    def test_intelligence_engine_init(self, initialized_intelligence_engine):
        """Test IntelligenceEngine initialization"""
        engine = initialized_intelligence_engine

        assert isinstance(engine.agent_uuid, str)
        assert isinstance(engine.format_uuid, str)
        assert engine.thread_uuid is None
        assert engine.thread_file_key is None
        assert isinstance(engine.M, dict)
        assert isinstance(engine.memory_prefs, dict)
        assert hasattr(engine, "pattern_index")
        assert engine.pattern_index is not None  # Should exist in private mode

    def test_load_or_init_formats_existing(self, initialized_intelligence_engine):
        """Test loading existing format metadata"""
        engine = initialized_intelligence_engine

        # Create format directory and file
        formats_dir = Path("memories/public/formats")
        format_shard = shard_path(formats_dir, engine.format_uuid)
        format_shard.mkdir(parents=True, exist_ok=True)

        format_path = format_shard / f"format-{engine.format_uuid}.json"
        mock_format = {
            "format_uuid": engine.format_uuid,
            "format_name": "test_format",
            "patterns": [],
        }

        with open(format_path, "w") as f:
            json.dump(mock_format, f)

        # Test loading
        result = engine._load_or_init_formats()

        # Verify loaded format
        assert result.get("format_uuid") == engine.format_uuid
        assert result.get("format_name") == "test_format"

    def test_load_or_init_formats_new(self, initialized_intelligence_engine):
        """Test initializing new format metadata"""
        engine = initialized_intelligence_engine

        # Ensure format file doesn't exist
        formats_dir = Path("memories/public/formats")
        format_shard = shard_path(formats_dir, engine.format_uuid)
        format_path = format_shard / f"format-{engine.format_uuid}.json"
        if format_path.exists():
            format_path.unlink()

        # Test initialization
        with patch.object(
            engine,
            "_initialize_format_metadata",
            return_value={"format_uuid": engine.format_uuid, "test": True},
        ):
            result = engine._load_or_init_formats()

            # Verify initialized format
            assert result.get("format_uuid") == engine.format_uuid
            assert result.get("test") is True

    def test_initialize_format_metadata(self, initialized_intelligence_engine):
        """Test format metadata initialization"""
        engine = initialized_intelligence_engine

        # Initialize new format metadata
        result = engine._initialize_format_metadata()

        # Verify structure
        assert result.get("format_uuid") == engine.format_uuid
        assert "patterns" in result
        assert len(result.get("patterns")) == 256
        assert "cgm_policies" in result

        # Check all required policies
        required_policies = ["governance", "information", "inference", "intelligence"]
        for policy in required_policies:
            assert policy in result.get("cgm_policies")

    def test_start_new_thread(self, initialized_intelligence_engine):
        """Test starting a new thread"""
        engine = initialized_intelligence_engine

        with patch.object(engine, "_derive_file_key", return_value=b"0" * 32):
            # Start a new thread
            thread_uuid = engine.start_new_thread()

            # Verify thread initialization
            assert thread_uuid == engine.thread_uuid
            assert engine.thread_file_key == b"0" * 32
            assert engine.current_thread_keys == []

            # Check that thread metadata file exists
            private_dir = Path("memories/private/agents")
            agent_shard = shard_path(private_dir, engine.agent_uuid)
            agent_dir = agent_shard / f"agent-{engine.agent_uuid}"
            threads_dir = agent_dir / "threads"
            thread_shard = shard_path(threads_dir, thread_uuid)
            thread_meta_path = thread_shard / f"thread-{thread_uuid}.json"
            assert thread_meta_path.exists()

            # Check that key file exists
            keys_dir = agent_dir / "keys"
            key_shard = shard_path(keys_dir, thread_uuid)
            key_path = key_shard / f"key-{thread_uuid}.bin.enc"
            assert key_path.exists()

    def test_process_input_stream(self, initialized_intelligence_engine):
        """Test processing an input stream"""
        engine = initialized_intelligence_engine
        test_input = b"Test input stream"
        expected_event = {"type": "input", "data": "VGVzdCBpbnB1dCBzdHJlYW0="}
        with (
            patch.object(
                engine.information_engine,
                "process_stream",
                return_value=(b"ciphertext", b"keystream"),
            ),
            patch.object(engine, "_append_to_thread") as mock_append,
        ):
            # Process input stream
            plaintext, ciphertext = engine.process_input_stream(test_input)
            # Verify results
            assert plaintext == test_input
            assert ciphertext == b"ciphertext"
            # Accept privacy kwarg
            mock_append.assert_called_once_with(expected_event, privacy="private")

    def test_generate_and_save_response(self, initialized_intelligence_engine):
        """Test generating and saving a response"""
        engine = initialized_intelligence_engine
        import base64

        with (
            patch.object(engine, "_generate_response_bytes", return_value=b"Generated response"),
            patch.object(engine, "_append_to_thread") as mock_append,
        ):
            # Generate response
            response = engine.generate_and_save_response(length=20)
            # Verify results
            assert response == b"Generated response"
            # Accept privacy kwarg
            expected_event = {"type": "output", "data": base64.b64encode(b"Generated response").decode("utf-8")}
            mock_append.assert_called_once_with(expected_event, privacy="private")

    def test_generate_response_bytes(self, initialized_intelligence_engine):
        """Test generating response bytes"""
        engine = initialized_intelligence_engine

        with (
            patch.object(engine, "_generate_response_byte", side_effect=[(97, 0), (98, 1), (99, 2)]),
            patch.object(engine.inference_engine, "process_byte", return_value=(0, 0.5)),
            patch.object(engine, "update_learning_state"),
        ):
            # Generate 3 bytes
            response = engine._generate_response_bytes(3)

            # Verify the output
            assert response == b"abc"

            # Check method calls
            assert engine._generate_response_byte.call_count == 3
            assert engine.inference_engine.process_byte.call_count == 3
            assert engine.update_learning_state.call_count == 3

    def test_update_learning_state(self, initialized_intelligence_engine):
        """Test updating learning state with new signature"""
        engine = initialized_intelligence_engine

        # Initialize pattern data
        pattern_index = 42
        engine.M["patterns"][pattern_index]["count"] = 5
        engine.M["patterns"][pattern_index]["first_cycle"] = 1
        engine.M["patterns"][pattern_index]["last_cycle"] = 5

        # Update learning state with the new signature
        engine.update_learning_state(source_byte=100, key_index=pattern_index, resonance=0.5, event_type="INPUT")

        # Verify pattern updates
        assert engine.M["patterns"][pattern_index]["count"] == 6
        assert engine.M["patterns"][pattern_index]["first_cycle"] == 1
        assert engine.M["patterns"][pattern_index]["last_cycle"] == engine.inference_engine.cycle_counter

        # Verify key recording
        assert len(engine.current_thread_keys) == 1
        gene_key = engine.current_thread_keys[0]
        assert gene_key["cycle"] == engine.inference_engine.cycle_counter
        assert gene_key["pattern_index"] == pattern_index
        assert gene_key["event_type"] == "INPUT"
        assert gene_key["source_byte"] == 100

    def test_update_learning_state_first_occurrence(self, initialized_intelligence_engine):
        """Test updating learning state for first pattern occurrence"""
        engine = initialized_intelligence_engine

        # Initialize pattern data
        pattern_index = 42
        engine.M["patterns"][pattern_index]["count"] = 0
        engine.M["patterns"][pattern_index]["first_cycle"] = None
        engine.M["patterns"][pattern_index]["last_cycle"] = None

        # Update learning state
        engine.update_learning_state(source_byte=200, key_index=pattern_index, resonance=0.3, event_type="OUTPUT")

        # Verify pattern updates
        assert engine.M["patterns"][pattern_index]["count"] == 1
        assert engine.M["patterns"][pattern_index]["first_cycle"] == engine.inference_engine.cycle_counter
        assert engine.M["patterns"][pattern_index]["last_cycle"] == engine.inference_engine.cycle_counter

    def test_encode_decode(self, initialized_intelligence_engine):
        """Test character encoding and decoding"""
        engine = initialized_intelligence_engine

        # Setup pattern with character label
        pattern_index = 42
        character_label = "test_character"
        engine.M["patterns"][pattern_index]["character"] = character_label

        # Test encode
        assert engine.encode(character_label) == pattern_index
        assert engine.encode("nonexistent") is None

        # Test decode
        assert engine.decode(pattern_index) == character_label
        assert engine.decode(99) is None

    def test_load_thread_content(self, initialized_intelligence_engine):
        """Test loading a thread's decrypted content as NDJSON events"""
        engine = initialized_intelligence_engine
        thread_uuid = "test-thread-uuid"
        import base64, json

        # Prepare a fake NDJSON thread with two events
        event1 = {"type": "input", "data": base64.b64encode(b"abc").decode("utf-8")}
        event2 = {"type": "output", "data": base64.b64encode(b"xyz").decode("utf-8")}
        ndjson = f"{json.dumps(event1)}\n{json.dumps(event2)}\n".encode("utf-8")
        # Patch the correct targets as used in baby/intelligence.py
        with (
            patch("baby.intelligence.load_thread", return_value=ndjson) as mock_load_thread,
            patch(
                "baby.intelligence.load_thread_key", return_value=bytes([i % 256 for i in range(256)])
            ) as mock_load_thread_key,
        ):
            # Patch agent_uuid and agent_secret to None to skip decryption
            engine.agent_uuid = None
            engine.agent_secret = None
            # Load thread content
            content = engine.load_thread_content(thread_uuid)
            # Should return a list of event dicts with decoded data, or None
            assert isinstance(content, list)
            assert len(content) == 2
            assert content[0]["type"] == "input"
            assert content[0]["data"] == b"abc"
            assert content[1]["type"] == "output"
            assert content[1]["data"] == b"xyz"
            # Check that the right functions were called
            mock_load_thread.assert_called_once_with(engine.agent_uuid, thread_uuid)
            mock_load_thread_key.assert_not_called()

    def test_get_thread_relationships(self, initialized_intelligence_engine):
        """Test getting thread relationships"""
        engine = initialized_intelligence_engine
        thread_uuid = "test-thread-uuid"
        parent_uuid = "parent-thread-uuid"
        child_uuid = "child-thread-uuid"

        with (
            patch("baby.intelligence.parent", return_value=parent_uuid) as mock_parent,
            patch("baby.intelligence.children", return_value=[child_uuid]) as mock_children,
        ):
            # Get relationships
            relationships = engine.get_thread_relationships(thread_uuid)

            # Verify result
            assert relationships["parent"] == parent_uuid
            assert relationships["children"] == [child_uuid]

            # Check function calls
            mock_parent.assert_called_once_with(engine.agent_uuid, thread_uuid)
            mock_children.assert_called_once_with(engine.agent_uuid, thread_uuid)

    def test_get_thread_chain(self, initialized_intelligence_engine):
        """Test getting a thread chain"""
        engine = initialized_intelligence_engine
        thread_uuid = "test-thread-uuid"
        parent_uuid = "parent-thread-uuid"
        grandparent_uuid = "grandparent-thread-uuid"

        # Mock parent function
        with patch("baby.intelligence.parent") as mock_parent:
            # Set up mock return values
            mock_parent.side_effect = lambda a, t: {
                "test-thread-uuid": parent_uuid,
                "parent-thread-uuid": grandparent_uuid,
                "grandparent-thread-uuid": None,
            }.get(t)

            # Get thread chain
            chain = engine.get_thread_chain(thread_uuid)

            # Verify chain order (root->leaf)
            assert grandparent_uuid in chain
            assert parent_uuid in chain
            assert thread_uuid in chain

            # Check parent calls
            assert mock_parent.call_count >= 2

    def test_get_thread_statistics(self, initialized_intelligence_engine):
        """Test getting thread statistics"""
        engine = initialized_intelligence_engine

        # Create test data in the actual directory structure
        agent_uuid = engine.agent_uuid
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, agent_uuid)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        threads_dir = agent_dir / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)

        # Use valid UUIDs for thread IDs
        thread_ids = [str(uuid.uuid4()) for _ in range(3)]
        registry = {"count": 3, "uuids": thread_ids}
        with open(threads_dir / "registry.json", "w") as f:
            json.dump(registry, f)

        # Create thread metadata files in the correct shard paths
        for thread_id, meta in zip(
            thread_ids,
            [
                cast(
                    ThreadMetadata,
                    {
                        "thread_uuid": None,
                        "parent_uuid": None,
                        "children": [{"uuid": str(thread_ids[1]), "name": None}],
                        "size_bytes": 1000,
                        "privacy": "public",
                    },
                ),
                cast(
                    ThreadMetadata,
                    {
                        "thread_uuid": None,
                        "parent_uuid": thread_ids[0],
                        "children": [{"uuid": str(thread_ids[2]), "name": None}],
                        "size_bytes": 2000,
                        "privacy": "public",
                    },
                ),
                cast(
                    ThreadMetadata,
                    {
                        "thread_uuid": None,
                        "parent_uuid": thread_ids[1],
                        "children": [],
                        "size_bytes": 3000,
                        "privacy": "public",
                    },
                ),
            ],
        ):
            meta["thread_uuid"] = thread_id
            # Remove to_flat_str_list; children is a list of ChildRef dicts
            thread_shard = shard_path(threads_dir, thread_id)
            thread_shard.mkdir(parents=True, exist_ok=True)
            with open(thread_shard / f"thread-{thread_id}.json", "w") as f:
                json.dump(meta, f)

        # Get statistics
        stats = engine.get_thread_statistics()

        # Verify statistics
        assert stats["total_threads"] == 3
        assert stats["total_size_bytes"] == 6000
        assert stats["relationship_stats"]["threads_with_parents"] == 2
        assert stats["relationship_stats"]["threads_with_children"] == 2
        assert stats["relationship_stats"]["isolated_threads"] == 0
        assert len(stats["thread_details"]) == 3

    # Remove or update test_thread_capacity_exceeded if thread rotation is not implemented/spec-compliant


# ------------------------------------------------------------------------------
# Test Utilities
# ------------------------------------------------------------------------------


def test_weighted_choice():
    """Test the weighted choice utility function"""
    items = ["a", "b", "c"]
    weights = [0.1, 0.8, 0.1]

    # Mock random to return specific values
    with patch("random.random") as mock_random:
        # Test selecting first item
        mock_random.return_value = 0.05
        result = weighted_choice(items, weights)
        assert result == "a"

        # Test selecting second item
        mock_random.return_value = 0.5
        result = weighted_choice(items, weights)
        assert result == "b"

        # Test selecting third item
        mock_random.return_value = 0.95
        result = weighted_choice(items, weights)
        assert result == "c"


def test_public_mode_initialization(public_intelligence_engine):
    """Test that public mode works correctly"""
    engine = public_intelligence_engine

    assert engine.agent_uuid is None
    assert engine.agent_secret is None
    assert engine.pattern_index is None  # Should be None in public mode
    assert isinstance(engine.format_uuid, str)
    assert isinstance(engine.M, dict)


def test_confidence_updates_on_learning(initialized_intelligence_engine):
    """
    Verify that update_learning_state correctly calculates and updates
    a pattern's confidence as a moving average.
    """
    engine = initialized_intelligence_engine
    pattern_index = 10
    # Precondition: Confidence should be 0.0 initially.
    initial_confidence = engine.M["patterns"][pattern_index]["confidence"]
    assert initial_confidence == 0.0, "Initial confidence should be 0.0"
    # Action 1: A perfect match (resonance = 0.0)
    engine.update_learning_state(source_byte=0, key_index=pattern_index, resonance=0.0, event_type="INPUT")
    first_confidence = engine.M["patterns"][pattern_index]["confidence"]
    assert np.isclose(first_confidence, 0.01), f"Confidence after perfect match is wrong: {first_confidence}"
    # Action 2: A perfect mismatch (resonance = np.pi)
    engine.update_learning_state(source_byte=0, key_index=pattern_index, resonance=np.pi, event_type="INPUT")
    second_confidence = engine.M["patterns"][pattern_index]["confidence"]
    assert np.isclose(second_confidence, 0.0099), f"Confidence after mismatch is wrong: {second_confidence}"


def test_intelligent_encode_prefers_stronger_pattern(initialized_intelligence_engine):
    """
    Verify that intelligent_encode uses count and confidence to select the
    best pattern for a character when multiple options exist.
    """
    engine = initialized_intelligence_engine
    # Setup: Map "A" to two different patterns with different strengths.
    engine.M["patterns"][65]["character"] = "A"
    engine.M["patterns"][65]["count"] = 10
    engine.M["patterns"][65]["confidence"] = 0.5  # Score = 10 * 0.5 = 5
    engine.M["patterns"][150]["character"] = "A"
    engine.M["patterns"][150]["count"] = 100
    engine.M["patterns"][150]["confidence"] = 0.9  # Score = 100 * 0.9 = 90
    chosen_index = engine.intelligent_encode("A")
    assert chosen_index == 150, "intelligent_encode failed to pick the stronger pattern."


def test_intelligent_generation_prefers_meaningful_pattern(initialized_intelligence_engine, monkeypatch):
    """
    Verify that _generate_response_byte prefers a semantically meaningful pattern
    over a more physically resonant but meaningless one.
    """
    engine = initialized_intelligence_engine
    # Setup: Create a scenario with two candidates.
    engine.M["patterns"][10]["character"] = None
    engine.M["patterns"][10]["confidence"] = 0.0
    engine.M["patterns"][20]["character"] = "B"
    engine.M["patterns"][20]["confidence"] = 0.95  # High semantic score
    mock_resonances = np.full(256, np.pi)  # All are invalid by default
    mock_resonances[10] = 0.1 * np.pi  # physical_score = 0.9
    mock_resonances[20] = 0.4 * np.pi  # physical_score = 0.6
    monkeypatch.setattr(engine.inference_engine, "compute_pattern_resonances", lambda: mock_resonances)
    _output_byte, chosen_index = engine._generate_response_byte()
    assert chosen_index == 20, "Generation failed to prioritize the semantically meaningful pattern."


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
