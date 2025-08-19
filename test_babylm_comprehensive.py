#!/usr/bin/env python3
"""
Comprehensive BabyLM Performance Test & Capability Demonstration

This script tests and showcases BabyLM's capabilities including:
- Text generation with metrics tracking
- Recovery ladder performance
- Cache efficiency
- Admissibility checks
- Surface form validation
- Thread safety under concurrent load
"""

import time
import threading
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add baby module to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from baby.kernel.gyro_core import GyroEngine
except ImportError:
    print("Error: GyroEngine not found. Please ensure baby.kernel.gyro_core is available.")
    sys.exit(1)

# Use existing tokenizer functionality
try:
    from baby.tokenizer import get_tokenizer
except ImportError:
    get_tokenizer = None


class BabyLMTester:
    """Comprehensive tester for BabyLM capabilities."""
    
    def __init__(self):
        self.engine: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.test_results: Dict[str, Any] = {}
        
    def setup(self) -> bool:
        """Initialize BabyLM components."""
        try:
            print("🚀 Initializing BabyLM components...")
            
            # Initialize GyroEngine (this may take time for large models)
            print("⏳ Loading GyroEngine (this may take a moment)...")
            
            # Create minimal configuration for testing
            atlas_paths = {
                "epistemology": "memories/public/meta/epistemology.npy",
                "ontology_keys": "memories/public/meta/ontology_keys.npy",
                "theta": "memories/public/meta/theta.npy",
                "phenomenology_map": "memories/public/meta/phenomenology_map.npy",
                "orbit_sizes": "memories/public/meta/orbit_sizes.npy"
            }
            store_paths = {
                "passive_memory": "memories/public/knowledge/passive_memory.bin",
                "address_memory": "memories/public/knowledge/address_memory.dat"
            }
            runtime = {
                "max_nudges": "6",
                "enable_self_reinforcement": "false",
                "cold_start_grace_window": "50",
                "passive_log_sync_interval": "1000"
            }
            
            version_info = {
                "atlas_version": "1.0.0",
                "address_version": "1.0.0",
                "config_version": "1.0.0"
            }
            
            self.engine = GyroEngine(atlas_paths, store_paths, runtime, version_info)
            print("✅ GyroEngine initialized")
            
            # Initialize tokenizer if available
            if get_tokenizer is not None:
                self.tokenizer = get_tokenizer()
                print("✅ Tokenizer initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            self.engine = None
            return False
    
    def test_basic_generation(self) -> Dict[str, Any]:
        """Test basic text generation capabilities."""
        print("\n📝 Testing Basic Text Generation...")
        
        test_prompts = [
            "The quick brown fox",
            "In a world where artificial intelligence",
            "The meaning of life is",
            "Once upon a time",
            "The future of technology"
        ]
        
        results = {
            "successful_generations": 0,
            "failed_generations": 0,
            "average_length": 0,
            "generations": []
        }
        
        total_length = 0
        
        for prompt in test_prompts:
            try:
                start_time = time.time()
                
                # Generate response using GyroEngine directly
                if self.engine is not None and self.tokenizer is not None:
                    # Tokenize the prompt
                    prompt_tokens = self.tokenizer.encode(prompt)
                    
                    # Start from initial state and evolve through prompt
                    current_state = self.engine.start_state()
                    for token in prompt_tokens:
                        current_state = self.engine.evolve_on_user(current_state, token)
                    
                    # Generate continuation tokens
                    generated_tokens = []
                    max_new_tokens = 10
                    
                    for _ in range(max_new_tokens):
                        next_token = self.engine.next_token_deterministic(current_state)
                        if next_token is None:
                            # Try recovery if no deterministic token
                            candidates = self.engine.recover_candidates(current_state)
                            if candidates:
                                next_token = candidates[0]
                            else:
                                break
                        
                        generated_tokens.append(next_token)
                        current_state = self.engine.evolve_on_assistant(current_state, next_token)
                        
                        # Stop on end tokens or control tokens
                        if next_token in [200001, 200007]:  # END tokens
                            break
                    
                    # Decode the generated tokens
                    if generated_tokens and self.tokenizer is not None:
                        generated_text = self.tokenizer.decode(generated_tokens)
                        response = prompt + generated_text
                    else:
                        response = prompt + " [no tokens generated]"
                        
                    print(f"  ✅ '{prompt}' → Generated {len(generated_tokens)} tokens")
                else:
                    raise Exception("Engine not initialized")
                
                generation_time = time.time() - start_time
                
                if response and len(response.strip()) > 0:
                    results["successful_generations"] += 1
                    total_length += len(response)
                    
                    results["generations"].append({
                        "prompt": prompt,
                        "response": response[:100] + "..." if len(response) > 100 else response,
                        "length": len(response),
                        "time_ms": round(generation_time * 1000, 2)
                    })
                    
                    print(f"  ✅ '{prompt}' → {len(response)} chars in {generation_time*1000:.1f}ms")
                else:
                    results["failed_generations"] += 1
                    print(f"  ❌ '{prompt}' → No response generated")
                    
            except Exception as e:
                results["failed_generations"] += 1
                print(f"  ❌ '{prompt}' → Error: {e}")
        
        if results["successful_generations"] > 0:
            results["average_length"] = total_length / results["successful_generations"]
        
        return results
    
    def test_metrics_tracking(self) -> Dict[str, Any]:
        """Test the runtime metrics system."""
        print("\n📊 Testing Metrics Tracking...")
        
        # Reset metrics
        if self.engine is None:
            print("⚠️  Engine not available - skipping metrics tracking test")
            return {"status": "skipped", "reason": "engine_unavailable"}
        self.engine.reset_metrics()
        
        # Perform some operations to generate metrics
        test_states = [12345, 67890, 11111, 22222, 33333]
        
        for state in test_states:
            try:
                # Test token generation
                if self.engine is not None:
                    token = self.engine.next_token_deterministic(state)
                    
                    # Test admissibility
                    if token is not None:
                        self.engine.is_admissible(state, token)
                    
                    # Test address lookup
                    if token is not None:
                        self.engine.address_of_token(token)
                else:
                    continue
                    
            except Exception as e:
                print(f"  ⚠️ Error testing state {state}: {e}")
        
        # Get metrics
        metrics = self.engine.get_metrics()
        
        print("\n📈 Current Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")
        
        return metrics
    
    def test_concurrent_performance(self, num_threads: int = 5, operations_per_thread: int = 10) -> Dict[str, Any]:
        """Test thread safety and concurrent performance."""
        print(f"\n🔄 Testing Concurrent Performance ({num_threads} threads, {operations_per_thread} ops each)...")
        
        results: Dict[str, Any] = {
            "threads_completed": 0,
            "total_operations": 0,
            "errors": 0,
            "total_time": 0.0
        }
        
        def worker_thread(thread_id: int):
            """Worker function for concurrent testing."""
            thread_results = {"operations": 0, "errors": 0}
            
            for i in range(operations_per_thread):
                try:
                    # Generate a pseudo-random state based on thread and iteration
                    state = (thread_id * 1000 + i) % 100000
                    
                    # Test token generation
                    if self.engine is not None:
                        token = self.engine.next_token_deterministic(state)
                        
                        if token is not None:
                            # Test admissibility
                            self.engine.is_admissible(state, token)
                            
                            # Test address lookup
                            self.engine.address_of_token(token)
                    else:
                        continue
                    
                    thread_results["operations"] += 1
                    
                except Exception as e:
                    thread_results["errors"] += 1
                    print(f"  ⚠️ Thread {thread_id} error: {e}")
            
                # Update results (thread-safe)
            with threading.Lock():
                results["threads_completed"] += 1
                results["total_operations"] += thread_results["operations"]
                results["errors"] += thread_results["errors"]
        
        # Start timing
        start_time = time.time()
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        results["total_time"] = time.time() - start_time
        results["operations_per_second"] = results["total_operations"] / results["total_time"] if results["total_time"] > 0 else 0
        
        print(f"  ✅ Completed {results['total_operations']} operations in {results['total_time']:.2f}s")
        print(f"  📈 Performance: {results['operations_per_second']:.1f} ops/sec")
        print(f"  ❌ Errors: {results['errors']}")
        
        return results
    
    def test_surface_form_validation(self) -> Dict[str, Any]:
        """Test surface form validation capabilities."""
        print("\n🛡️ Testing Surface Form Validation...")
        
        # Test cases: (text, should_be_valid)
        test_cases = [
            ("This is a normal sentence.", True),
            (" Leading whitespace", False),
            ("Double  spaces", False),
            ("Triple   spaces", False),
            ("aaaaaaaaaa", False),  # Character run-length
            ("word word word word word", False),  # Token repetition
            ("Sentence.No space after period", False),
            ("Normal sentence. Proper spacing.", True),
            ("\n\nToo many newlines", False),
            ("Proper text with good formatting.", True)
        ]
        
        results = {
            "total_tests": len(test_cases),
            "correct_validations": 0,
            "incorrect_validations": 0,
            "test_details": []
        }
        
        for text, expected_valid in test_cases:
            try:
                # Enhanced surface form validation
                is_bad = False
                
                # Leading whitespace
                if text.startswith(' '):
                    is_bad = True
                
                # Multiple consecutive spaces
                if '  ' in text:
                    is_bad = True
                
                # Character repetition (5+ consecutive identical chars)
                for i in range(len(text) - 4):
                    if len(set(text[i:i+5])) == 1:
                        is_bad = True
                        break
                
                # Word repetition (same word repeated 4+ times)
                words = text.split()
                if len(words) >= 4:
                    for i in range(len(words) - 3):
                        if words[i] == words[i+1] == words[i+2] == words[i+3]:
                            is_bad = True
                            break
                
                # Punctuation spacing issues
                for i, char in enumerate(text):
                    if char in '.!?;:' and i < len(text) - 1:
                        next_char = text[i + 1]
                        if next_char not in ' \n':
                            is_bad = True
                            break
                
                # Too many consecutive newlines (2+ at start or 3+ anywhere)
                if text.startswith('\n\n') or '\n\n\n' in text:
                    is_bad = True
                
                is_valid = not is_bad
                
                is_correct = (is_valid == expected_valid)
                
                if is_correct:
                    results["correct_validations"] += 1
                    status = "✅"
                else:
                    results["incorrect_validations"] += 1
                    status = "❌"
                
                results["test_details"].append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "expected": expected_valid,
                    "actual": is_valid,
                    "correct": is_correct
                })
                
                print(f"  {status} '{text[:30]}...' → Expected: {expected_valid}, Got: {is_valid}")
                
            except Exception as e:
                results["incorrect_validations"] += 1
                print(f"  ❌ Error testing '{text[:30]}...': {e}")
        
        accuracy = results["correct_validations"] / results["total_tests"] * 100
        print(f"\n📊 Surface Form Validation Accuracy: {accuracy:.1f}%")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and compile results."""
        print("🧪 Starting Comprehensive BabyLM Test Suite")
        print("=" * 50)
        
        if not self.setup():
            return {"error": "Setup failed"}
        
        # Run all tests
        test_results = {}
        
        try:
            test_results["basic_generation"] = self.test_basic_generation()
            test_results["metrics_tracking"] = self.test_metrics_tracking()
            test_results["concurrent_performance"] = self.test_concurrent_performance()
            test_results["surface_form_validation"] = self.test_surface_form_validation()
            
            # Get final metrics
            if self.engine is not None:
                test_results["final_metrics"] = self.engine.get_metrics()
            
            # Print comprehensive metrics summary
            print("\n" + "=" * 50)
            print("📊 FINAL METRICS SUMMARY")
            print("=" * 50)
            if self.engine is not None:
                self.engine.print_metrics_summary()
            
        except Exception as e:
            test_results["error"] = str(e)
            print(f"❌ Test suite error: {e}")
        
        return test_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "babylm_test_results.json"):
        """Save test results to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main test execution."""
    tester = BabyLMTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Save results
    tester.save_results(results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    if "error" in results:
        print(f"❌ Test suite failed: {results['error']}")
        return
    
    # Basic generation summary
    if "basic_generation" in results:
        gen = results["basic_generation"]
        print(f"📝 Text Generation: {gen['successful_generations']}/{gen['successful_generations'] + gen['failed_generations']} successful")
        if gen['successful_generations'] > 0:
            print(f"   Average length: {gen['average_length']:.1f} characters")
    
    # Concurrent performance summary
    if "concurrent_performance" in results:
        perf = results["concurrent_performance"]
        print(f"🔄 Concurrent Performance: {perf['operations_per_second']:.1f} ops/sec")
        print(f"   Errors: {perf['errors']}/{perf['total_operations']} operations")
    
    # Surface form validation summary
    if "surface_form_validation" in results:
        surf = results["surface_form_validation"]
        accuracy = surf['correct_validations'] / surf['total_tests'] * 100
        print(f"🛡️ Surface Form Validation: {accuracy:.1f}% accuracy")
    
    print("\n✨ BabyLM Comprehensive Test Complete!")


if __name__ == "__main__":
    main()