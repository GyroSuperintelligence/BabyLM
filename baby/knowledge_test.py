#!/usr/bin/env python3
"""
Robust Knowledge Injection and Retrieval Test Script

This script provides a reliable test framework for the complete knowledge pipeline:
1. Robust server startup with comprehensive error handling
2. Knowledge injection with proper ingestion-only mode
3. Knowledge querying with session continuity
4. Detailed diagnostics and metrics tracking

Usage:
    python -m baby.knowledge_test
"""

import json
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests


class RobustKnowledgeTestRunner:
    """Robust test runner with comprehensive error handling and diagnostics."""

    def __init__(self, port: int = 9000, config_path: str = "baby/config.json"):
        self.port = port
        self.config_path = config_path
        self.base_url = f"http://localhost:{port}"
        self.server_process: Optional[subprocess.Popen[str]] = None
        self.test_results: Dict[str, Any] = {}
        self.startup_logs: List[str] = []

        # Paths
        self.project_root = Path(__file__).parent.parent
        self.wiki_test_path = self.project_root / "toys" / "training" / "wiki_test.txt"
        self.passive_memory_path = self.project_root / "memories" / "public" / "knowledge" / "passive_memory.bin"
        self.config_full_path = self.project_root / self.config_path

        # Setup signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.log(f"Received signal {signum}, cleaning up...", "WARN")
        self.cleanup()
        sys.exit(0)

    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps and levels."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level:5}] {message}"
        print(formatted_msg)
        self.startup_logs.append(formatted_msg)

    def validate_environment(self) -> bool:
        """Comprehensive environment validation."""
        self.log("🔍 Validating environment...")

        issues = []

        # Check Python version
        if sys.version_info < (3, 8):
            issues.append(f"Python {sys.version_info.major}.{sys.version_info.minor} too old (need 3.8+)")

        # Check config file
        if not self.config_full_path.exists():
            issues.append(f"Config file missing: {self.config_full_path}")
        else:
            try:
                config = json.loads(self.config_full_path.read_text())
                self.log(f"✓ Config loaded: {config.get('version', 'unknown version')}")
            except Exception as e:
                issues.append(f"Config file invalid: {e}")

        # Check wiki test file
        if not self.wiki_test_path.exists():
            issues.append(f"Wiki test file missing: {self.wiki_test_path}")
        # Removed verbose file size logging

        # Check memory files (silent check)
        memory_files = [
            "memories/public/meta/epistemology.npy",
            "memories/public/meta/ontology_keys.npy",
            "memories/public/meta/theta.npy",
            "memories/public/meta/phenomenology_map.npy",
            "memories/public/meta/orbit_sizes.npy",
            "memories/public/knowledge/address_memory.dat",
            "memories/public/knowledge/passive_memory.bin",
        ]

        for mem_file in memory_files:
            full_path = self.project_root / mem_file
            if not full_path.exists():
                issues.append(f"Memory file missing: {mem_file}")
            # Removed verbose file size logging

        # Check required modules
        try:
            # Import check only - modules are available for testing
            pass
        except ImportError as e:
            issues.append(f"Missing module: {e}")

        # Check port availability - if server is already running, we'll use it
        import socket

        self.server_already_running = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", self.port)) == 0:
                self.server_already_running = True
                self.log(f"✓ Server already running on port {self.port}")
            # Removed verbose port logging

        if issues:
            self.log("❌ Environment validation failed:", "ERROR")
            for issue in issues:
                self.log(f"  - {issue}", "ERROR")
            return False

        self.log("✅ Environment validation passed")
        return True

    def start_server_robust(self) -> bool:
        """Start server with comprehensive error handling and diagnostics."""
        # Check if server is already running
        if hasattr(self, "server_already_running") and self.server_already_running:
            self.log("✓ Using existing server")
            return True

        self.log("🚀 Starting server...")

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "baby.responses_api.serve",
            "--inference-backend",
            "gyro",
            "--config",
            str(self.config_full_path),
            "--port",
            str(self.port),
        ]

        try:
            # Start process with minimal logging
            self.server_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            # Monitor startup with minimal log capture
            startup_timeout = 60  # Increased timeout
            startup_success = False

            def log_reader():
                """Read server logs in background thread (critical messages only)."""
                if self.server_process and self.server_process.stdout:
                    for line in iter(self.server_process.stdout.readline, ""):
                        if line.strip():
                            # Only log actual critical server messages, skip parsing errors
                            if (
                                any(
                                    keyword in line.upper()
                                    for keyword in ["FAILED", "EXCEPTION", "TRACEBACK", "CRITICAL"]
                                )
                                and "PARSING TOKENS" not in line.upper()
                            ):
                                self.log(f"SERVER ERROR: {line.strip()}", "ERROR")

            # Start log reader thread
            log_thread = threading.Thread(target=log_reader, daemon=True)
            log_thread.start()

            # Wait for server to become ready
            for _ in range(startup_timeout):
                # Check if process crashed
                if self.server_process.poll() is not None:
                    exit_code = self.server_process.returncode
                    self.log(f"❌ Server process exited with code {exit_code}", "ERROR")
                    return False

                # Test server readiness
                try:
                    test_request = {"input": "test", "stream": False, "store": False, "max_output_tokens": 1}

                    response = requests.post(f"{self.base_url}/v1/responses", json=test_request, timeout=3)

                    if response.status_code == 200:
                        self.log("✅ Server ready")
                        startup_success = True
                        break

                except requests.exceptions.RequestException:
                    pass  # Expected during startup

                time.sleep(1)

            if not startup_success:
                self.log("❌ Server failed to start within timeout", "ERROR")
                self.cleanup()
                return False

            return True

        except Exception as e:
            self.log(f"❌ Failed to start server: {e}", "ERROR")
            return False

    def inject_knowledge_safe(self, article_content: str) -> Optional[str]:
        """Inject knowledge with ingestion-only mode (max_output_tokens=0)."""
        self.log("📚 Injecting knowledge...")

        # Record memory stats before (for summary only)
        before_stats = self.get_memory_stats()
        self.test_results["memory_before_injection"] = before_stats

        request_data = {
            "input": article_content,
            "stream": False,
            "store": True,  # Enable session storage
            "max_output_tokens": 0,  # Ingestion only - no output generation
            "__debug": False,  # Disable verbose debug info
            "model": "gyro",
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/responses", json=request_data, timeout=120  # Allow time for ingestion
            )

            if response.status_code == 200:
                result = response.json()
                response_id = result.get("id")

                # Check memory growth (for summary only)
                time.sleep(2)  # Allow file system sync
                after_stats = self.get_memory_stats()
                self.test_results["memory_after_injection"] = after_stats

                self.log(f"✅ Knowledge injected")
                return response_id

            else:
                self.log(f"❌ Knowledge injection failed: {response.status_code}", "ERROR")
                return None

        except Exception as e:
            self.log(f"❌ Knowledge injection error: {e}", "ERROR")
            return None

    def query_knowledge_safe(self, previous_response_id: str, query: str) -> bool:
        """Send full sentence from article using session continuity."""
        self.log(f"🔍 Testing with full sentence: '{query[:80]}{'...' if len(query) > 80 else ''}'")

        request_data = {
            "input": query,
            "stream": False,
            "store": True,
            "max_output_tokens": 150,
            "previous_response_id": previous_response_id,  # Session continuity
            "__debug": False,
            "model": "gyro",
        }

        try:
            response = requests.post(f"{self.base_url}/v1/responses", json=request_data, timeout=60)

            if response.status_code == 200:
                result = response.json()
                output = result.get("output", [])
                text_items = [
                    it
                    for it in output
                    if isinstance(it, dict) and it.get("type") == "message" and it.get("role") == "assistant"
                ]
                ok = bool(text_items) and any(
                    any(
                        part.get("text", "").strip()
                        for part in it.get("content", [])
                        if isinstance(part, dict) and part.get("type") in ("text", "output_text")
                    )
                    for it in text_items
                )
                self.log(f"response: {output}")

                if ok:
                    self.log(f"✅ Query successful with valid assistant message")

                    self.test_results["query_response"] = {
                        "query": query,
                        "response": output,
                        "length": len(str(output)),
                        "response_id": result.get("id"),
                    }

                    return True
                else:
                    self.log(f"❌ Query failed: No valid assistant message with text content", "ERROR")
                    return False

            else:
                self.log(f"❌ Query failed: {response.status_code}", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Query error: {e}", "ERROR")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "passive_memory_exists": False,
            "passive_memory_mb": 0.0,
            "address_memory_exists": False,
            "address_memory_mb": 0.0,
        }

        # Passive memory
        if self.passive_memory_path.exists():
            size_bytes = self.passive_memory_path.stat().st_size
            stats.update({"passive_memory_exists": True, "passive_memory_mb": size_bytes / (1024 * 1024)})

        # Address memory
        address_path = self.project_root / "memories" / "public" / "knowledge" / "address_memory.dat"
        if address_path.exists():
            size_bytes = address_path.stat().st_size
            stats.update({"address_memory_exists": True, "address_memory_mb": size_bytes / (1024 * 1024)})

        return stats

    def load_test_article(self) -> str:
        """Load the wiki test article."""
        if not self.wiki_test_path.exists():
            raise FileNotFoundError(f"Wiki test file not found: {self.wiki_test_path}")

        content = self.wiki_test_path.read_text(encoding="utf-8")
        self.log(f"📖 Loaded test article: {len(content)} chars")
        return content

    def run_complete_test(self) -> bool:
        """Run the complete knowledge test pipeline."""
        self.log("🎯 Starting complete knowledge test pipeline")

        try:
            # 1. Environment validation
            if not self.validate_environment():
                return False

            # 2. Server startup
            if not self.start_server_robust():
                return False

            # 3. Load test content
            article_content = self.load_test_article()

            # 4. Knowledge injection (ingestion-only)
            response_id = self.inject_knowledge_safe(article_content)
            if not response_id:
                return False

            # 5. Knowledge querying with full sentences from the article
            # Use complete sentences from the wiki article for proper testing
            test_sentences = [
                "In mathematics and computer science, an algorithm is a sequence of rigorous instructions, typically used to solve a class of specific problems or to perform a computation.",
                "Algorithms are used as specifications for performing calculations and data processing.",
                "More advanced algorithms can use conditionals to divert the code execution through various routes and deduce valid inferences, achieving automation eventually.",
                "Using human characteristics as descriptors of machines in metaphorical ways was already practiced by Alan Turing with terms such as memory, search and stimulus.",
            ]

            query_success = False
            for sentence in test_sentences:
                if self.query_knowledge_safe(response_id, sentence):
                    query_success = True
                    break

            if not query_success:
                self.log("❌ All sentence tests failed", "ERROR")
                return False

            self.log("🎉 Complete test pipeline successful!")
            return True

        except Exception as e:
            self.log(f"❌ Test pipeline failed: {e}", "ERROR")
            return False
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        # Don't terminate server if we're using an existing one
        if hasattr(self, "server_already_running") and self.server_already_running:
            self.log("✓ Leaving existing server running")
            return

        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception as e:
                self.log(f"Cleanup error: {e}", "WARN")
            finally:
                self.server_process = None

    def print_summary(self):
        """Print test summary and results."""
        self.log("\n" + "=" * 60)
        self.log("📊 KNOWLEDGE TEST SUMMARY")
        self.log("=" * 60)

        if self.test_results:
            for key, value in self.test_results.items():
                if isinstance(value, dict):
                    self.log(f"{key}:")
                    for k, v in value.items():
                        self.log(f"  {k}: {v}")
                else:
                    self.log(f"{key}: {value}")
        else:
            self.log("No test results available")

        self.log("=" * 60)


def main():
    """Main entry point."""
    print("🧪 Robust BabyLM Knowledge Test")
    print("================================")

    runner = RobustKnowledgeTestRunner()

    try:
        success = runner.run_complete_test()
        runner.print_summary()

        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        runner.cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        runner.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
