#!/usr/bin/env python3
"""
Knowledge Injection and Retrieval Test Script

This script tests the complete knowledge pipeline:
1. Starts the GyroSI server with gyro backend
2. Injects knowledge from wiki_test.txt article
3. Queries the knowledge with sentence continuation
4. Monitors passive memory, caps, admissibility, and recovery stats

Usage:
    python -m baby.knowledge_test
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests


class KnowledgeTestRunner:
    """Comprehensive test runner for knowledge ingestion and retrieval."""
    
    def __init__(self, port: int = 8001, config_path: str = "baby/config.json"):
        self.port = port
        self.config_path = config_path
        self.base_url = f"http://localhost:{port}"
        self.server_process: Optional[subprocess.Popen] = None
        self.test_results: Dict[str, Any] = {}
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.wiki_test_path = self.project_root / "toys" / "training" / "wiki_test.txt"
        self.passive_memory_path = self.project_root / "memories" / "public" / "knowledge" / "passive_memory.bin"
        
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def check_dependencies(self) -> bool:
        """Check if all required files and dependencies exist."""
        self.log("Checking dependencies...")
        
        # Check wiki test file
        if not self.wiki_test_path.exists():
            self.log(f"ERROR: Wiki test file not found: {self.wiki_test_path}", "ERROR")
            return False
            
        # Check config file
        config_full_path = self.project_root / self.config_path
        if not config_full_path.exists():
            self.log(f"ERROR: Config file not found: {config_full_path}", "ERROR")
            return False
            
        # Check if memories directory exists, create if needed
        memories_dir = self.project_root / "memories" / "public" / "knowledge"
        memories_dir.mkdir(parents=True, exist_ok=True)
        
        self.log("✓ All dependencies checked")
        return True
        
    def get_passive_memory_stats(self) -> Dict[str, Any]:
        """Get passive memory file statistics."""
        stats = {
            "exists": False,
            "size_bytes": 0,
            "size_mb": 0.0,
            "modified_time": None
        }
        
        if self.passive_memory_path.exists():
            stat_info = self.passive_memory_path.stat()
            stats.update({
                "exists": True,
                "size_bytes": stat_info.st_size,
                "size_mb": round(stat_info.st_size / (1024 * 1024), 3),
                "modified_time": time.ctime(stat_info.st_mtime)
            })
            
        return stats
        
    def start_server(self) -> bool:
        """Start the GyroSI server with gyro backend."""
        self.log("Starting GyroSI server...")
        
        # Start server command (use correct CLI flag --inference-backend)
        cmd = [
            sys.executable, "-m", "baby.responses_api.serve",
            "--inference-backend", "gyro",
            "--config", self.config_path,
            "--port", str(self.port)
        ]
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start by probing /v1/responses with a minimal request
            max_wait = 45  # seconds
            for i in range(max_wait):
                # If the process crashed, abort early
                if self.server_process.poll() is not None:
                    # Read stderr to provide context
                    try:
                        stderr_tail = self.server_process.stderr.read() if self.server_process.stderr else ""
                    except Exception:
                        stderr_tail = ""
                    self.log("ERROR: Server process exited prematurely", "ERROR")
                    if stderr_tail:
                        self.log(stderr_tail[-5000:], "ERROR")
                    return False
                try:
                    probe_body = {
                        "input": "ping",
                        "stream": False,
                        "store": False,
                        "max_output_tokens": 1
                    }
                    response = requests.post(f"{self.base_url}/v1/responses", json=probe_body, timeout=2)
                    if response.status_code == 200:
                        self.log(f"✓ Server started successfully on port {self.port}")
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
                self.log(f"Waiting for server... ({i+1}/{max_wait})")
                
            self.log("ERROR: Server failed to start within timeout", "ERROR")
            return False
            
        except Exception as e:
            self.log(f"ERROR: Failed to start server: {e}", "ERROR")
            return False
            
    def load_wiki_article(self) -> str:
        """Load the wiki test article content."""
        try:
            with open(self.wiki_test_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            self.log(f"✓ Loaded wiki article ({len(content)} characters)")
            return content
        except Exception as e:
            self.log(f"ERROR: Failed to load wiki article: {e}", "ERROR")
            return ""
            
    def inject_knowledge(self, article_content: str) -> Optional[str]:
        """Inject knowledge via POST to /v1/responses."""
        self.log("Injecting knowledge into GyroSI...")
        
        # Record passive memory stats before injection
        before_stats = self.get_passive_memory_stats()
        self.test_results["passive_memory_before"] = before_stats
        self.log(f"Passive memory before: {before_stats['size_mb']} MB")
        
        # Prepare request
        request_data = {
            "input": article_content,
            "stream": False,  # Use non-streaming for easier handling
            "store": True,    # Enable storage for chaining
            "temperature": 0.0,
            "max_output_tokens": 100,  # Limit output for knowledge injection
            "metadata": {
                "__debug": True  # Enable debug mode
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/responses",
                json=request_data,
                timeout=180  # Increase timeout from 60 to 180 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                response_id = result.get("id")
                
                # Record passive memory stats after injection
                time.sleep(1)  # Give time for file system to update
                after_stats = self.get_passive_memory_stats()
                self.test_results["passive_memory_after"] = after_stats
                
                self.log(f"✓ Knowledge injected successfully")
                self.log(f"Response ID: {response_id}")
                self.log(f"Passive memory after: {after_stats['size_mb']} MB")
                self.log(f"Memory growth: {after_stats['size_mb'] - before_stats['size_mb']:.3f} MB")
                
                # Log debug information if available
                if "metadata" in result and "__debug" in result["metadata"]:
                    debug_info = result["metadata"]["__debug"]
                    self.log(f"Debug info: {json.dumps(debug_info, indent=2)}")
                
                return response_id
            else:
                self.log(f"ERROR: Knowledge injection failed: {response.status_code}", "ERROR")
                self.log(f"Response: {response.text}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"ERROR: Knowledge injection request failed: {e}", "ERROR")
            return None
            
    def query_knowledge(self, previous_response_id: str, article_content: str) -> bool:
        """Query the injected knowledge with sentence continuation."""
        self.log("Querying injected knowledge...")
        
        # Extract first sentence from article for continuation test
        sentences = article_content.split('. ')
        if len(sentences) > 1:
            # Use first part of first sentence for continuation
            first_sentence = sentences[0]
            words = first_sentence.split()
            if len(words) > 3:
                prompt = ' '.join(words[:3])  # Use first 3 words
                self.log(f"Using prompt for continuation: '{prompt}'")
            else:
                prompt = "Algorithm"  # Fallback
        else:
            prompt = "Algorithm"  # Fallback
            
        request_data = {
            "previous_response_id": previous_response_id,
            "input": prompt,
            "stream": False,
            "temperature": 0.0,
            "max_output_tokens": 200,
            "metadata": {
                "__debug": True
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/responses",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response text
                output_text = ""
                if "output" in result:
                    for item in result["output"]:
                        if item.get("type") == "message" and item.get("role") == "assistant":
                            content = item.get("content", [])
                            if isinstance(content, list):
                                for content_item in content:
                                    if content_item.get("type") in ["text", "output_text"]:
                                        output_text += content_item.get("text", "")
                
                self.log(f"✓ Knowledge query successful")
                self.log(f"Query prompt: '{prompt}'")
                self.log(f"Response length: {len(output_text)} characters")
                self.log(f"Response preview: {output_text[:200]}...")
                
                # Store results
                self.test_results["query_prompt"] = prompt
                self.test_results["query_response"] = output_text
                self.test_results["query_response_length"] = len(output_text)
                
                # Check if response seems related to the article
                article_keywords = ["algorithm", "mathematics", "computer", "science", "computation"]
                response_lower = output_text.lower()
                keyword_matches = [kw for kw in article_keywords if kw in response_lower]
                
                if keyword_matches:
                    self.log(f"✓ Response contains relevant keywords: {keyword_matches}")
                    self.test_results["keyword_relevance"] = True
                    self.test_results["matched_keywords"] = keyword_matches
                else:
                    self.log("⚠ Response may not be related to injected knowledge", "WARN")
                    self.test_results["keyword_relevance"] = False
                
                # Log debug information
                if "metadata" in result and "__debug" in result["metadata"]:
                    debug_info = result["metadata"]["__debug"]
                    self.log(f"Query debug info: {json.dumps(debug_info, indent=2)}")
                
                return True
            else:
                self.log(f"ERROR: Knowledge query failed: {response.status_code}", "ERROR")
                self.log(f"Response: {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"ERROR: Knowledge query request failed: {e}", "ERROR")
            return False
            
    def run_comprehensive_test(self) -> bool:
        """Run the complete knowledge test pipeline."""
        self.log("=" * 60)
        self.log("STARTING COMPREHENSIVE KNOWLEDGE TEST")
        self.log("=" * 60)
        
        try:
            # Step 1: Check dependencies
            if not self.check_dependencies():
                return False
                
            # Step 2: Start server
            if not self.start_server():
                return False
                
            # Step 3: Load article
            article_content = self.load_wiki_article()
            if not article_content:
                return False
                
            # Step 4: Inject knowledge
            response_id = self.inject_knowledge(article_content)
            if not response_id:
                return False
                
            # Step 5: Query knowledge
            if not self.query_knowledge(response_id, article_content):
                return False
                
            # Step 6: Final summary
            self.log("=" * 60)
            self.log("TEST COMPLETED SUCCESSFULLY")
            self.log("=" * 60)
            
            self.print_test_summary()
            return True
            
        except KeyboardInterrupt:
            self.log("Test interrupted by user", "WARN")
            return False
        except Exception as e:
            self.log(f"ERROR: Unexpected error during test: {e}", "ERROR")
            return False
        finally:
            self.cleanup()
            
    def print_test_summary(self):
        """Print a comprehensive test summary."""
        self.log("\n📊 TEST SUMMARY:")
        self.log("-" * 40)
        
        # Passive memory stats
        before = self.test_results.get("passive_memory_before", {})
        after = self.test_results.get("passive_memory_after", {})
        
        self.log(f"📁 Passive Memory:")
        self.log(f"   Before: {before.get('size_mb', 0):.3f} MB")
        self.log(f"   After:  {after.get('size_mb', 0):.3f} MB")
        self.log(f"   Growth: {after.get('size_mb', 0) - before.get('size_mb', 0):.3f} MB")
        
        # Query results
        self.log(f"\n🔍 Knowledge Query:")
        self.log(f"   Prompt: '{self.test_results.get('query_prompt', 'N/A')}'")
        self.log(f"   Response Length: {self.test_results.get('query_response_length', 0)} chars")
        self.log(f"   Keyword Relevance: {'✓' if self.test_results.get('keyword_relevance') else '✗'}")
        
        if self.test_results.get("matched_keywords"):
            self.log(f"   Matched Keywords: {self.test_results['matched_keywords']}")
            
        # Response preview
        response = self.test_results.get("query_response", "")
        if response:
            self.log(f"\n📝 Response Preview:")
            self.log(f"   {response[:150]}{'...' if len(response) > 150 else ''}")
            
    def cleanup(self):
        """Clean up resources."""
        self.log("Cleaning up...")
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.log("✓ Server stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.log("✓ Server force-killed")
            except Exception as e:
                self.log(f"Warning: Error stopping server: {e}", "WARN")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GyroSI Knowledge Test Runner")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--config", default="baby/config.json", help="Config file path")
    
    args = parser.parse_args()
    
    runner = KnowledgeTestRunner(port=args.port, config_path=args.config)
    success = runner.run_comprehensive_test()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()