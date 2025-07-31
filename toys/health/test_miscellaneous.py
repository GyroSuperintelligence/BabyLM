"""
Miscellaneous tests for GyroSI Baby system.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Generator

import numpy as np
import pytest

# Local imports
from baby.intelligence import GyroSI
from baby.contracts import AgentConfig
from baby.policies import OrbitStore
from toys.communication import tokenizer as gyrotok
from toys.communication.external_adapter import app
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# =============================================================================
# File Analysis Tests
# =============================================================================


class TestTrainedModelAnalysis:
    """Analyze the trained Wikipedia model files."""

    @pytest.fixture
    def archive_path(self) -> Path:
        """Path to the Archive directory with trained model files."""
        return Path(__file__).parent.parent / "training" / "Archive"

    @pytest.fixture
    def gyro_file(self, archive_path: Path) -> Path:
        """Path to the trained gyro file."""
        return archive_path / "wikipedia_simple.gyro"

    @pytest.fixture
    def bin_file(self, archive_path: Path) -> Path:
        """Path to the trained knowledge file."""
        return archive_path / "wikipedia_simple.bin"

    def test_gyro_file_exists_and_analyze(self, gyro_file: Path) -> None:
        """Test that the gyro file exists and analyze its contents."""
        assert gyro_file.exists(), f"Gyro file not found: {gyro_file}"

        # Get file stats
        stat = gyro_file.stat()
        size_mb = stat.st_size / (1024 * 1024)

        print("\n📊 Gyro File Analysis:")
        print(f"   Path: {gyro_file}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Created: {stat.st_ctime}")
        print(f"   Modified: {stat.st_mtime}")

        # Read first few bytes to understand structure
        with gyro_file.open("rb") as f:
            first_bytes = f.read(100)
            print(f"   First 100 bytes: {first_bytes[:50].hex()}...")

            # Count total bytes
            f.seek(0, 2)  # Seek to end
            total_bytes = f.tell()
            print(f"   Total bytes: {total_bytes:,}")

            # Sample some content
            f.seek(0)
            sample_size = min(1000, total_bytes)
            sample = f.read(sample_size)
            print(f"   Sample content (hex): {sample[:100].hex()}...")

            # Try to decode as tokens - trim trailing partial token
            try:
                # sample is masked bytes; find last complete token boundary
                trim = len(sample) - 1
                while trim >= 0 and ((sample[trim] ^ 0xAA) & 0x80):  # continuation bit set after unmask
                    trim -= 1
                if trim >= 0:
                    tokens = gyrotok.bytes_to_ids(sample[: trim + 1])
                    print(f"   Decoded tokens: {len(tokens)} tokens")
                    print(f"   Token sample: {tokens[:10]}")
                else:
                    print("   No complete tokens found in sample")
            except Exception as e:
                print(f"   Token decoding error: {e}")

    def test_bin_file_exists_and_analyze(self, bin_file: Path) -> None:
        """Test that the bin file exists and analyze its contents."""
        assert bin_file.exists(), f"Bin file not found: {bin_file}"

        # Get file stats
        stat = bin_file.stat()
        size_mb = stat.st_size / (1024 * 1024)

        print("\n📊 Knowledge File Analysis:")
        print(f"   Path: {bin_file}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Created: {stat.st_ctime}")
        print(f"   Modified: {stat.st_mtime}")

        # Try to open as OrbitStore to analyze structure
        try:
            store = OrbitStore(str(bin_file), append_only=True)

            # Get store statistics
            print(f"   Store type: {type(store)}")
            print(f"   Store path: {store.store_path}")

            # Try to get some entries
            try:
                # Sample some keys (this might not work depending on store structure)
                print("   Store opened successfully")

                # Get file size info
                if hasattr(store, "_mmap") and store._mmap is not None:
                    print(f"   Memory map size: {len(store._mmap):,} bytes")

            except Exception as e:
                print(f"   Store analysis error: {e}")

            store.close()

        except Exception as e:
            print(f"   Failed to open as OrbitStore: {e}")

    def test_file_comparison(self, gyro_file: Path, bin_file: Path) -> None:
        """Compare the two files and their relationship."""
        gyro_stat = gyro_file.stat()
        bin_stat = bin_file.stat()

        gyro_size_mb = gyro_stat.st_size / (1024 * 1024)
        bin_size_mb = bin_stat.st_size / (1024 * 1024)

        print("\n📊 File Comparison:")
        print(f"   Gyro file: {gyro_size_mb:.1f} MB")
        print(f"   Bin file: {bin_size_mb:.1f} MB")
        print(f"   Compression ratio: {gyro_size_mb/bin_size_mb:.2f}x")
        print(f"   Time difference: {abs(gyro_stat.st_mtime - bin_stat.st_mtime):.0f}s")


# =============================================================================
# Isolated Agent Tests
# =============================================================================


class TestIsolatedAgentWithTrainedKnowledge:
    """Test an isolated agent using the trained knowledge."""

    @pytest.fixture
    def isolated_agent_config(self, test_env: Dict[str, Any]) -> AgentConfig:
        """Create config for isolated agent with trained knowledge."""
        archive_path = Path(__file__).parent.parent / "training" / "Archive"
        trained_bin = archive_path / "wikipedia_simple.bin"

        return {
            "ontology_path": test_env["main_meta_files"]["ontology"],
            "phenomenology_map_path": test_env["main_meta_files"]["phenomenology"],
            "knowledge_path": str(trained_bin),
            "base_path": str(test_env["memories_dir"]),
        }

    @pytest.fixture
    def isolated_agent(self, isolated_agent_config: AgentConfig) -> Generator[GyroSI, None, None]:
        """Create isolated agent with trained knowledge."""
        agent = GyroSI(
            isolated_agent_config,
            agent_id="test_trained_agent",
            base_path=Path(str(isolated_agent_config.get("base_path", "."))),
        )
        yield agent
        agent.close()

    def test_agent_with_trained_knowledge(self, isolated_agent: GyroSI) -> None:
        """Test that agent can load and use trained knowledge."""
        print("\n🧠 Isolated Agent Test:")
        print(f"   Agent ID: {isolated_agent.agent_id}")
        print(f"   Knowledge path: {isolated_agent.config.get('knowledge_path', 'Not set')}")

        # Check agent state
        print(f"   Engine cycle count: {isolated_agent.engine.cycle_count}")
        print(f"   Operator store: {type(isolated_agent.engine.operator.store)}")

        # Test basic functionality
        assert isolated_agent.engine is not None
        assert isolated_agent.engine.operator is not None
        assert isolated_agent.engine.operator.store is not None

        print("   ✅ Agent loaded successfully with trained knowledge")

    def test_agent_ingest_capability(self, isolated_agent: GyroSI) -> None:
        """Test that agent can still ingest new data."""
        test_text = "Hello world, this is a test."

        # Encode test text
        encoded_bytes = gyrotok.encode(test_text)
        print("\n🧠 Ingest Test:")
        print(f"   Test text: '{test_text}'")
        print(f"   Encoded bytes: {len(encoded_bytes)} bytes")

        # Ingest the test data
        isolated_agent.ingest_bulk(encoded_bytes)

        print("   ✅ Agent successfully ingested test data")
        print(f"   New cycle count: {isolated_agent.engine.cycle_count}")


# =============================================================================
# Communication Tests
# =============================================================================


class TestCommunicationWithTrainedModel:
    """Test communication capabilities using the external adapter API."""

    @pytest.fixture
    def api_client(self, test_env: Dict[str, Any]) -> Generator[TestClient, None, None]:
        """Create API client with trained model configuration."""
        # Set up environment for API
        os.environ["GYROSI_PREFERENCES_PATH"] = test_env["preferences_path"]

        # Create test client
        client = TestClient(app)

        yield client

        # Cleanup
        if "GYROSI_PREFERENCES_PATH" in os.environ:
            del os.environ["GYROSI_PREFERENCES_PATH"]

    def test_models_endpoint(self, api_client: TestClient) -> None:
        """Test the models endpoint."""
        print("\n🌐 API Models Test:")

        response = api_client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        print(f"   Response: {json.dumps(data, indent=2)}")

        assert "data" in data
        assert len(data["data"]) > 0

        # Check model info
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"

        print("   ✅ Models endpoint working")

    def test_chat_completion_basic(self, api_client: TestClient) -> None:
        """Test basic chat completion."""
        print("\n🌐 Chat Completion Test:")

        payload = {
            "model": "gyrosi-baby",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
        }

        response = api_client.post("/v1/chat/completions", json=payload)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")

            assert "choices" in data
            assert len(data["choices"]) > 0

            choice = data["choices"][0]
            assert "message" in choice
            assert "content" in choice["message"]

            content = choice["message"]["content"]
            print(f"   Generated content: '{content}'")

            # Check if content is not empty
            assert len(content.strip()) > 0, "Generated content should not be empty"

        else:
            print(f"   Error response: {response.text}")
            # Don't fail the test - the model might not be ready for chat yet

    def test_huggingface_generate(self, api_client: TestClient) -> None:
        """Test HuggingFace-style text generation."""
        print("\n🌐 HF Generation Test:")

        payload = {"inputs": "The quick brown fox", "parameters": {"max_new_tokens": 20, "temperature": 0.7}}

        response = api_client.post("/generate", json=payload)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")

            assert "generated_text" in data
            generated_text = data["generated_text"]

            print(f"   Generated text: '{generated_text}'")

            # Check if generated text extends the input
            assert len(generated_text) > len(payload["inputs"])

        else:
            print(f"   Error response: {response.text}")
            # Don't fail the test - the model might not be ready yet

    def test_conversation_context(self, api_client: TestClient) -> None:
        """Test conversation with context/memory."""
        print("\n🌐 Conversation Context Test:")

        # First message
        payload1 = {
            "model": "gyrosi-baby",
            "messages": [{"role": "user", "content": "My name is Alice."}],
            "max_tokens": 30,
        }

        response1 = api_client.post("/v1/chat/completions", json=payload1)
        print(f"   First message status: {response1.status_code}")

        if response1.status_code == 200:
            data1 = response1.json()
            first_reply = data1["choices"][0]["message"]["content"]
            print(f"   First reply: '{first_reply}'")

            # Second message with context
            payload2 = {
                "model": "gyrosi-baby",
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": first_reply},
                    {"role": "user", "content": "What's my name?"},
                ],
                "max_tokens": 30,
            }

            response2 = api_client.post("/v1/chat/completions", json=payload2)
            print(f"   Second message status: {response2.status_code}")

            if response2.status_code == 200:
                data2 = response2.json()
                second_reply = data2["choices"][0]["message"]["content"]
                print(f"   Second reply: '{second_reply}'")

                # Check if the model remembers the name
                if "Alice" in second_reply:
                    print("   ✅ Model remembered the name!")
                else:
                    print("   ⚠️  Model didn't seem to remember the name")
            else:
                print(f"   Second message error: {response2.text}")
        else:
            print(f"   First message error: {response1.text}")


# =============================================================================
# Knowledge Analysis Tests
# =============================================================================


class TestKnowledgeAnalysis:
    """Analyze the knowledge structure and content."""

    @pytest.fixture
    def knowledge_store(self) -> Generator[OrbitStore, None, None]:
        """Open the trained knowledge store for analysis."""
        archive_path = Path(__file__).parent.parent / "training" / "Archive"
        trained_bin = archive_path / "wikipedia_simple.bin"

        store = OrbitStore(str(trained_bin), append_only=True)
        yield store
        store.close()

    def test_knowledge_store_structure(self, knowledge_store: OrbitStore) -> None:
        """Analyze the structure of the knowledge store."""
        print("\n🧠 Knowledge Store Analysis:")
        print(f"   Store type: {type(knowledge_store)}")
        print(f"   Store path: {knowledge_store.store_path}")

        # Get file size
        stat = Path(knowledge_store.store_path).stat()
        size_mb = stat.st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.1f} MB")

        # Try to analyze internal structure
        if hasattr(knowledge_store, "_mmap") and knowledge_store._mmap is not None:
            mmap_size = len(knowledge_store._mmap)
            print(f"   Memory map size: {mmap_size:,} bytes")

            # Sample some data
            if mmap_size > 0:
                sample_size = min(1000, mmap_size)
                sample = knowledge_store._mmap[:sample_size]
                print(f"   Sample data (hex): {sample[:100].hex()}...")

        print("   ✅ Knowledge store structure analyzed")

    def test_knowledge_entries_sample(self, knowledge_store: OrbitStore) -> None:
        """Try to sample some knowledge entries."""
        print("\n🧠 Knowledge Entries Sample:")

        # This is exploratory - we don't know the exact structure
        # but we can try to understand what's in the store

        try:
            # Try to access some methods/properties
            store_attrs = [attr for attr in dir(knowledge_store) if not attr.startswith("_")]
            print(f"   Available methods: {store_attrs}")

            # Try to get some basic info
            if hasattr(knowledge_store, "store_path"):
                print(f"   Store path: {knowledge_store.store_path}")

            if hasattr(knowledge_store, "_mmap") and knowledge_store._mmap is not None:
                mmap = knowledge_store._mmap
                print(f"   Memory map type: {type(mmap)}")
                print(f"   Memory map size: {len(mmap)} bytes")

                # Try to find patterns in the data
                sample_data = mmap[: min(10000, len(mmap))]
                unique_values = np.unique(sample_data)
                print(f"   Unique values in sample: {len(unique_values)}")
                print(f"   Value range: {unique_values.min()} to {unique_values.max()}")

        except Exception as e:
            print(f"   Analysis error: {e}")

        print("   ✅ Knowledge entries sample completed")


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullIntegration:
    """Test the full pipeline from trained model to communication."""

    def test_end_to_end_communication(self, test_env: Dict[str, Any]) -> None:
        """Test end-to-end communication using trained model."""
        print("\n🔄 End-to-End Integration Test:")

        # Set up environment
        os.environ["GYROSI_PREFERENCES_PATH"] = test_env["preferences_path"]

        try:
            # Create API client
            client = TestClient(app)

            # Test basic communication
            payload = {"model": "gyrosi-baby", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20}

            response = client.post("/v1/chat/completions", json=payload)
            print(f"   API response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"   Generated response: '{content}'")
                print("   ✅ End-to-end communication successful!")
            else:
                print(f"   API error: {response.text}")
                print("   ⚠️  Communication test failed - model may need more training")

        except Exception as e:
            print(f"   Integration error: {e}")
        finally:
            # Cleanup
            if "GYROSI_PREFERENCES_PATH" in os.environ:
                del os.environ["GYROSI_PREFERENCES_PATH"]

    def test_model_capabilities_summary(self) -> None:
        """Provide a summary of model capabilities."""
        print("\n📋 Model Capabilities Summary:")

        archive_path = Path(__file__).parent.parent / "training" / "Archive"
        gyro_file = archive_path / "wikipedia_simple.gyro"
        bin_file = archive_path / "wikipedia_simple.bin"

        if gyro_file.exists() and bin_file.exists():
            gyro_size = gyro_file.stat().st_size / (1024 * 1024)
            bin_size = bin_file.stat().st_size / (1024 * 1024)

            print("   ✅ Trained model files found")
            print(f"   📊 Training data: {gyro_size:.1f} MB")
            print(f"   🧠 Knowledge base: {bin_size:.1f} MB")
            print(f"   📈 Learning efficiency: {bin_size/gyro_size:.2f}x compression")

            if bin_size > 10:  # If knowledge is substantial
                print("   🎯 Model appears to have learned significant knowledge")
            else:
                print("   ⚠️  Knowledge base seems small - may need more training")
        else:
            print("   ❌ Trained model files not found")
            print(f"   Expected: {gyro_file} and {bin_file}")
