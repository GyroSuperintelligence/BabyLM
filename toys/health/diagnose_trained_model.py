#!/usr/bin/env python3
"""
Standalone Diagnostic Script for GyroSI Trained Model

This script launches the external adapter with the trained Wikipedia model
and runs a series of diagnostic tests to verify its structure and functionality.

Usage:
    python toys/health/diagnose_trained_model.py
"""

import json
import sys
import time
import struct
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from fastapi.testclient import TestClient  # noqa: E402


# Add timestamped debug function
def debug_print(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] DEBUG: {message}")


def build_and_save_bloom_filter() -> None:
    """Build and save the Bloom filter for the public knowledge store."""
    from baby.policies import OrbitStore
    from pathlib import Path

    # Get the path to the public knowledge store
    preferences_path = Path(__file__).resolve().parents[2] / "memories/memory_preferences.json"
    with open(preferences_path) as f:
        preferences = json.load(f)

    knowledge_path = str(Path(__file__).resolve().parents[2] / preferences["public_knowledge"]["path"])
    bloom_path = knowledge_path + ".bloom"

    print_info(f"Building Bloom filter for: {knowledge_path}")
    print_info(f"Bloom filter will be saved to: {bloom_path}")

    # Create the store and build the Bloom filter
    store = OrbitStore(store_path=knowledge_path, use_mmap=True)

    # Force a commit to save the Bloom filter
    store.commit()
    store.close()

    print_info("Bloom filter built and saved successfully.")

    # Verify the Bloom filter was created
    if Path(bloom_path).exists():
        print_info(f"Bloom filter file created: {Path(bloom_path).stat().st_size} bytes")
    else:
        print_warn("Bloom filter file was not created!")


# Don't import the adapter immediately - it will load the full knowledge file
# from toys.communication.external_adapter import app as fastapi_app  # noqa: E402


# --- Configuration ---
PREFERENCES_PATH = PROJECT_ROOT / "memories/memory_preferences.json"
ARCHIVE_PATH = PROJECT_ROOT / "toys/training/Archive"
GYRO_FILE_NAME = "wikipedia_simple.gyro"
BIN_FILE_NAME = "wikipedia_simple.bin"


# --- Helper Functions for Pretty Printing ---
def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title.upper()} ".center(60, "="))
    print("=" * 60)


def print_pass(message: str) -> None:
    print(f"✅ PASS: {message}")


def print_fail(message: str) -> None:
    print(f"❌ FAIL: {message}")


def print_warn(message: str) -> None:
    print(f"⚠️  WARN: {message}")


def print_info(message: str) -> None:
    print(f" {message}")


# --- Diagnostic Test Classes ---


class FileDiagnostics:
    """Fast analysis of knowledge files without loading whole files."""

    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = knowledge_dir
        # Look for knowledge files in the directory
        self.bin_files = list(self.knowledge_dir.glob("*.bin"))
        self.bloom_files = list(self.knowledge_dir.glob("*.bloom"))
        self.idx_files = list(self.knowledge_dir.glob("*.idx"))

    def run(self) -> None:
        print_header("File Diagnostics")

        if not self._check_files_exist():
            return

        self._check_file_sizes()
        self._check_bin_structure()

    def _check_files_exist(self) -> bool:
        print_info(f"Checking for knowledge files in: {self.knowledge_dir}")

        if not self.bin_files:
            print_fail(f"No .bin files found in: {self.knowledge_dir}")
            return False

        print_pass(f"Found {len(self.bin_files)} knowledge file(s).")
        if self.bloom_files:
            print_pass(f"Found {len(self.bloom_files)} Bloom filter file(s).")
        if self.idx_files:
            print_pass(f"Found {len(self.idx_files)} index file(s).")

        return True

    def _check_file_sizes(self) -> None:
        for bin_file in self.bin_files:
            bin_mb = bin_file.stat().st_size / (1024 * 1024)
            print_info(f"Knowledge file size: {bin_file.name} - {bin_mb:.1f} MB")

            if bin_mb < 1:
                print_warn(f"File {bin_file.name} is very small (<1MB), may be incomplete.")
            else:
                print_pass(f"File {bin_file.name} size is reasonable.")

        for bloom_file in self.bloom_files:
            bloom_kb = bloom_file.stat().st_size / 1024
            print_info(f"Bloom filter size: {bloom_file.name} - {bloom_kb:.1f} KB")

        for idx_file in self.idx_files:
            idx_kb = idx_file.stat().st_size / 1024
            print_info(f"Index file size: {idx_file.name} - {idx_kb:.1f} KB")

    def _check_bin_structure(self) -> None:
        print_info("Analyzing knowledge file phenotype structure...")

        for bin_file in self.bin_files:
            file_size = bin_file.stat().st_size

            if file_size % 12 != 0:
                print_fail(f"File {bin_file.name} size ({file_size}) is not divisible by 12. Structure is corrupted.")
                continue

            print_info(f"Estimated phenotypes in {bin_file.name}: {file_size // 12:,}")

            # Sample records from start and middle of file
            sample_positions = [0, min(file_size - 120, file_size // 2)]
            valid_count = 0
            fmt = "<IIf"  # Updated format to match _unpack_phenotype

            with bin_file.open("rb") as f:
                for pos in sample_positions:
                    f.seek(pos)
                    chunk = f.read(120)  # 10 records
                    for i in range(0, len(chunk), 12):
                        if i + 12 > len(chunk):
                            break
                        try:
                            state, token, mask = struct.unpack(fmt, chunk[i : i + 12])
                            if 0 <= mask <= 1.0:  # mask is now a float
                                valid_count += 1
                        except struct.error:
                            continue

            if valid_count < 10:
                print_fail(f"Failed to validate phenotype records in {bin_file.name}. Structure may be corrupt.")
            else:
                print_pass(f"Validated {valid_count} phenotype records in {bin_file.name} successfully.")


class APIDiagnostics:
    """Tests the API endpoints with the trained model."""

    def __init__(self, client: TestClient):
        self.client = client

    def run(self) -> None:
        print_header("API Diagnostics")
        self._test_connectivity()
        self._test_simple_queries()
        self._test_knowledge_queries()
        self._test_conversation_memory()

    def _test_connectivity(self) -> None:
        print_info("Testing API connectivity...")
        response = self.client.get("/v1/models")
        if response.status_code == 200 and "gyrosi-baby" in response.text:
            print_pass("API is responsive and lists the correct model.")
        else:
            print_fail(f"API connectivity failed. Status: {response.status_code}, Response: {response.text}")

    def _test_simple_queries(self) -> None:
        print_info("Testing simple chat queries...")
        prompts = ["Hello, how are you?", "What is your purpose?"]
        for prompt in prompts:
            print(f"   Query: '{prompt}'")
            response = self._chat(prompt)
            print(f"   Reply: '{response}'")
            if not response.strip():
                print_fail("Received an empty reply.")
                return
        print_pass("Model responds to simple queries.")

    def _test_knowledge_queries(self) -> None:
        print_info("Testing for signs of Wikipedia knowledge...")
        prompts = {
            "What is Python?": ["programming", "language", "code"],
            "Tell me about machine learning.": ["algorithm", "data", "model"],
        }
        learned_count = 0
        for prompt, keywords in prompts.items():
            print(f"   Query: '{prompt}'")
            response = self._chat(prompt).lower()
            print(f"   Reply: '{response}'")
            found_keywords = [kw for kw in keywords if kw in response]
            if found_keywords:
                learned_count += 1
                print(f"   ✅ Found keywords: {found_keywords}")
            else:
                print("   ⚠️  No expected keywords found.")

        if learned_count == len(prompts):
            print_pass("Model shows signs of learned knowledge in all queries.")
        elif learned_count > 0:
            print_pass("Model shows some signs of learned knowledge.")
        else:
            print_warn("Model did not show clear signs of learned knowledge.")

    def _test_conversation_memory(self) -> None:
        print_info("Testing conversation memory...")
        history = [{"role": "user", "content": "My name is Basil and I like to test language models."}]

        # First turn
        response1 = self._chat_with_history(history)
        print(f"   User: '{history[0]['content']}'")
        print(f"   Assistant: '{response1}'")
        history.append({"role": "assistant", "content": response1})

        # Second turn
        history.append({"role": "user", "content": "What is my name and what do I like to do?"})
        response2 = self._chat_with_history(history).lower()
        print(f"   User: '{history[2]['content']}'")
        print(f"   Assistant: '{response2}'")

        found_basil = "basil" in response2
        found_test = "test" in response2

        if found_basil and found_test:
            print_pass("Model remembered context from previous turn.")
        elif found_basil or found_test:
            print_warn("Model showed partial memory.")
        else:
            print_fail("Model did not show signs of conversation memory.")

    def _chat(self, prompt: str) -> str:
        return self._chat_with_history([{"role": "user", "content": prompt}])

    def _chat_with_history(self, messages: list[dict[str, str]]) -> str:
        payload = {"model": "gyrosi-baby", "messages": messages}  # Remove max_tokens limit
        response = self.client.post("/v1/chat/completions", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return str(result["choices"][0]["message"]["content"])


def main() -> int:
    """Main execution function."""

    # Add timeout for the entire script
    import signal

    def timeout_handler(signum: int, frame: Any) -> None:
        print_warn("Script timeout reached. Exiting...")
        sys.exit(1)

    # Set 5 minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5 minutes

    try:
        # 1. Verify that preferences are pointing to a valid knowledge file
        print_info("Loading preferences from: " + str(PREFERENCES_PATH))
        try:
            with open(PREFERENCES_PATH) as f:
                prefs = json.load(f)
            public_path = prefs.get("public_knowledge", {}).get("path", "")
            if not public_path:
                print_fail("No public knowledge path found in preferences.")
                return 1

            # Check if the file exists
            full_path = PROJECT_ROOT / public_path
            if not full_path.exists():
                print_fail(f"Knowledge file not found: {full_path}")
                return 1

            print_pass(f"Preferences point to valid knowledge file: {public_path}")
        except Exception as e:
            print_fail(f"Could not load or parse preferences.json: {e}")
            return 1

        # 2. Run file diagnostics
        print_header("FILE DIAGNOSTICS")
        # Use the knowledge file path from preferences instead of Archive
        knowledge_path = PROJECT_ROOT / public_path
        file_checker = FileDiagnostics(knowledge_path.parent)
        file_checker.run()

        # Build and save Bloom filter for faster subsequent runs
        print_info("Building Bloom filter for faster API diagnostics...")
        build_and_save_bloom_filter()

        # 3. Run API diagnostics (optional - can be slow with large knowledge files)
        print_warn("API diagnostics require loading the full knowledge file (77MB).")
        print_warn("This may take several minutes to build the Bloom filter.")
        print_info("Press Ctrl+C to skip API diagnostics and run file diagnostics only.")

        try:
            # Import the adapter only when needed
            from toys.communication.external_adapter import app as fastapi_app

            client = TestClient(fastapi_app)

            api_checker = APIDiagnostics(client)

            api_checker.run()
        except KeyboardInterrupt:
            print_warn("API diagnostics skipped by user.")
            print_info("File diagnostics completed successfully.")
        except Exception as e:
            # This will catch the slow-loading issue if it happens
            if "SystemExit" in str(e):
                print_fail("API startup stalled. This is likely due to the large knowledge file.")
                print_info("The system is trying to build a Bloom filter, which is slow for large files.")
                print_info("This is expected behavior for this architecture with large stores.")
            else:
                print_fail(f"An unexpected error occurred during API tests: {e}")
            return 1

        print_header("Diagnostics Complete")
        return 0

    finally:
        # Cancel the alarm
        signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
