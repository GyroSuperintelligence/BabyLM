#!/usr/bin/env python3
import requests
import subprocess
import sys
import time
import signal
import socket
import json
from pathlib import Path


class GyroChat:
    def __init__(self, port=9000):
        self.port = port
        self.url = f"http://localhost:{port}/v1/responses"
        self.server_process = None
        self.project_root = Path(__file__).parent.parent
        
        # Setup signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def is_server_running(self):
        """Check if server is already running on the port."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", self.port)) == 0
        except:
            return False

    def display_config_flags(self):
        """Display the current configuration flags."""
        try:
            config_path = self.project_root / "baby" / "config.json"
            with open(config_path) as f:
                config = json.load(f)
            
            runtime = config.get("runtime", {})
            print("🔧 Configuration Flags:")
            
            # Core physics switches
            flags = [
                ("Slab routing", runtime.get("enable_slab_routing", "not set")),
                ("DoF jitter", runtime.get("enable_dof_jitter", "not set")),
                ("Egress mask", runtime.get("enable_egress_mask", "not set")),
                ("Refractory gates", runtime.get("enable_refractory_gates", "not set")),
            ]
            
            for flag_name, value in flags:
                status = "✅ ENABLED" if value else "❌ DISABLED"
                print(f"  {flag_name}: {status}")
            
            print()
        except Exception as e:
            print(f"⚠️  Could not load config flags: {e}")
            print()

    def start_server(self):
        """Start the server if not already running."""
        if self.is_server_running():
            print("✓ Server already running")
            self.display_config_flags()
            return True

        print("🚀 Starting server...")
        
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "baby.responses_api.serve",
            "--inference-backend",
            "gyro",
            "--config",
            str(self.project_root / "baby" / "config.json"),
            "--port",
            str(self.port),
        ]

        try:
            # Start process
            self.server_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.DEVNULL,   # ⟵ do not pipe without draining
                stderr=subprocess.STDOUT,
            )

            # Wait for server to become ready
            startup_timeout = 60
            for _ in range(startup_timeout):
                # Check if process crashed
                if self.server_process.poll() is not None:
                    print(f"❌ Server process exited with code {self.server_process.returncode}")
                    return False

                # Test server readiness
                try:
                    test_request = {"input": "test", "stream": False, "store": False, "max_output_tokens": 1}
                    response = requests.post(self.url, json=test_request, timeout=3)
                    if response.status_code == 200:
                        print("✅ Server ready")
                        self.display_config_flags()
                        return True
                except requests.exceptions.RequestException:
                    pass  # Expected during startup

                time.sleep(1)

            print("❌ Server failed to start within timeout")
            self.cleanup()
            return False

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def cleanup(self):
        """Clean up server process."""
        if self.server_process:
            try:
                # On Windows, terminate() is the right call (sends CTRL-BREAK for new process groups).
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception as e:
                print(f"Cleanup error: {e}")
            finally:
                self.server_process = None

    def chat(self):
        """Main chat loop."""
        print("🤖 GyroSI BabyLM Chat Interface")
        print("=" * 50)
        print("Type 'quit' to exit, 'clear' to start new conversation")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    print("Starting new conversation...")
                    continue
                elif not user_input:
                    continue
                
                print("GyroSI: ", end="", flush=True)
                
                payload = {
                    "input": user_input,
                    "stream": False,
                    "store": False,
                    "max_output_tokens": 50,
                }
                
                response = requests.post(self.url, json=payload, timeout=(5, 300))
                
                if response.status_code == 200:
                    data = response.json()

                    # 1) Prefer top-level convenience field if present
                    if isinstance(data, dict) and data.get("output_text"):
                        print(data["output_text"])
                        continue

                    # 2) OpenAI Responses-style: data["output"] list of items
                    if isinstance(data, dict) and isinstance(data.get("output"), list):
                        for item in data["output"]:
                            if item.get("type") == "message" and item.get("role") == "assistant":
                                for part in item.get("content", []) or []:
                                    if part.get("type") == "output_text":
                                        print(part.get("text", ""))
                                        break
                                else:
                                    continue
                                break
                        else:
                            print("(No assistant message in output)")
                        continue

                    # 3) Raw message list (some older servers)
                    if isinstance(data, list):
                        for item in data:
                            if item.get("type") == "message" and item.get("role") == "assistant":
                                for part in item.get("content", []) or []:
                                    if part.get("type") == "output_text":
                                        print(part.get("text", ""))
                                        break
                        continue

                    print("(Unrecognised response shape)")
                else:
                    print(f"Error: HTTP {response.status_code}")
                    print(response.text)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
            
            print()


def main():
    chat = GyroChat()
    
    try:
        if not chat.start_server():
            print("Failed to start server. Exiting.")
            return
        
        chat.chat()
    finally:
        chat.cleanup()


if __name__ == "__main__":
    main()
