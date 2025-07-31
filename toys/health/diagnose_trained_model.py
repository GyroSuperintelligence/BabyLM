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
import struct
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from fastapi.testclient import TestClient  # noqa: E402

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
    print(f"   ℹ️  {message}")


# --- Diagnostic Test Classes ---


class FileDiagnostics:
    """Fast analysis of training outputs without loading whole files."""

    def __init__(self, archive_path: Path):
        self.archive_path = archive_path
        self.gyro_file = self.archive_path / GYRO_FILE_NAME
        self.bin_file = self.archive_path / BIN_FILE_NAME

    def run(self) -> None:
        print_header("File Diagnostics")

        if not self._check_files_exist():
            return

        self._check_file_sizes()
        self._check_gyro_structure()
        self._check_bin_structure()

    def _check_files_exist(self) -> bool:
        print_info(f"Checking for training files in: {self.archive_path}")

        if not self.gyro_file.exists():
            print_fail(f"Gyro file not found: {self.gyro_file}")
            return False
        if not self.bin_file.exists():
            print_fail(f"Bin file not found: {self.bin_file}")
            return False

        print_pass("Both training files exist.")
        return True

    def _check_file_sizes(self) -> None:
        gyro_mb = self.gyro_file.stat().st_size / (1024 * 1024)
        bin_mb = self.bin_file.stat().st_size / (1024 * 1024)

        print_info(f"Gyro file size: {gyro_mb:.1f} MB")
        print_info(f"Bin file size: {bin_mb:.1f} MB")

        if gyro_mb < 1 or bin_mb < 1:
            print_warn("Files are very small (<1MB), training may have been incomplete.")
        else:
            print_pass("File sizes are reasonable.")

    def _check_gyro_structure(self) -> None:
        print_info("Analyzing gyro file token structure (first 5KB)...")
        sample = self.gyro_file.read_bytes()[:5120]

        tokens_found = 0
        pos = 0
        while pos < len(sample):
            while pos < len(sample):
                intron = sample[pos] ^ 0xAA
                pos += 1
                if (intron & 0x80) == 0:
                    tokens_found += 1
                    break

        if tokens_found < 10:
            print_fail(f"Found only {tokens_found} tokens in sample. File may be corrupted.")
        else:
            print_pass(f"Found {tokens_found} complete tokens in sample. Structure looks valid.")

    def _check_bin_structure(self) -> None:
        print_info("Analyzing bin file phenotype structure...")
        file_size = self.bin_file.stat().st_size

        if file_size % 12 != 0:
            print_fail(f"Bin file size ({file_size}) is not divisible by 12. Structure is corrupted.")
            return

        print_info(f"Estimated phenotypes: {file_size // 12:,}")

        # Sample records from start and middle of file
        sample_positions = [0, min(file_size - 120, file_size // 2)]
        valid_count = 0
        fmt = "<IIBHx"

        with self.bin_file.open("rb") as f:
            for pos in sample_positions:
                f.seek(pos)
                chunk = f.read(120)  # 10 records
                for i in range(0, len(chunk), 12):
                    if i + 12 > len(chunk):
                        break
                    state, token, mask, conf_u16 = struct.unpack(fmt, chunk[i:i + 12])
                    conf = struct.unpack("e", struct.pack("H", conf_u16))[0]
                    if 0 <= conf <= 1.0 and 0 <= mask <= 255:
                        valid_count += 1

        if valid_count < 10:
            print_fail("Failed to validate phenotype records in sample. Structure may be corrupt.")
        else:
            print_pass(f"Validated {valid_count} phenotype records successfully.")


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
            print(f"   Reply: '{response[:80]}...'")
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
            print(f"   Reply: '{response[:80]}...'")
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
        print(f"   Assistant: '{response1[:80]}...'")
        history.append({"role": "assistant", "content": response1})

        # Second turn
        history.append({"role": "user", "content": "What is my name and what do I like to do?"})
        response2 = self._chat_with_history(history).lower()
        print(f"   User: '{history[2]['content']}'")
        print(f"   Assistant: '{response2[:80]}...'")

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
        payload = {"model": "gyrosi-baby", "messages": messages}
        response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


def main() -> int:
    """Main execution function."""

    # 1. Verify that preferences are pointing to the trained model
    print_info("Loading preferences from: " + str(PREFERENCES_PATH))
    try:
        with open(PREFERENCES_PATH) as f:
            prefs = json.load(f)
        public_path = prefs.get("public_knowledge", {}).get("path", "")
        if BIN_FILE_NAME not in public_path:
            print_fail(f"Your 'memory_preferences.json' does not point to '{BIN_FILE_NAME}'.")
            print_info(f"It currently points to: {public_path}")
            print_info("Please update it to: 'toys/training/Archive/wikipedia_simple.bin'")
            return 1
        print_pass("Preferences correctly point to the trained model.")
    except Exception as e:
        print_fail(f"Could not load or parse preferences.json: {e}")
        return 1

    # 2. Run file diagnostics
    file_checker = FileDiagnostics(ARCHIVE_PATH)
    file_checker.run()

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


if __name__ == "__main__":
    sys.exit(main())
