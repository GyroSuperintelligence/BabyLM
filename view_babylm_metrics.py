#!/usr/bin/env python3
"""
BabyLM Metrics Viewer

A simple script to view current runtime metrics from BabyLM GyroEngine.
This can be run independently to check metrics without running full tests.
"""

import sys
import time
from pathlib import Path

# Add baby module to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from baby.kernel.gyro_core import GyroEngine
except ImportError as e:
    print(f"❌ Failed to import BabyLM modules: {e}")
    print("Please ensure you're running from the BabyLM root directory.")
    sys.exit(1)


def main():
    """Load GyroEngine and display current metrics."""
    print("🚀 Loading BabyLM GyroEngine...")
    
    try:
        # Initialize GyroEngine with minimal configuration
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
        
        engine = GyroEngine(atlas_paths, store_paths, runtime, version_info)
        print("✅ GyroEngine loaded successfully!")
        
        # Display current metrics
        print("\n" + "=" * 50)
        print("📊 CURRENT BABYLM METRICS")
        print("=" * 50)
        
        # Use the built-in metrics summary
        engine.print_metrics_summary()
        
        # Also show raw metrics data
        metrics = engine.get_metrics()
        
        print("\n📋 Raw Metrics Data:")
        print("-" * 30)
        for key, value in metrics.items():
            if type(value) is float:
                print(f"  {key}: {value:.4f}")
            elif type(value) is int:
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
        
        # Perform a few test operations to generate some metrics
        print("\n🔄 Performing test operations to generate metrics...")
        
        test_operations = 0
        start_time = time.time()
        
        # Test some token generations
        for i in range(10):
            try:
                state = 12345 + i * 1000
                token = engine.next_token_deterministic(state)
                
                if token is not None:
                    # Test admissibility
                    engine.is_admissible(state, token)
                    
                    # Test address lookup
                    engine.address_of_token(token)
                
                test_operations += 1
                
            except Exception as e:
                print(f"  ⚠️ Error in test operation {i}: {e}")
        
        elapsed = time.time() - start_time
        print(f"✅ Completed {test_operations} test operations in {elapsed:.3f}s")
        
        # Show updated metrics
        print("\n" + "=" * 50)
        print("📊 UPDATED METRICS AFTER TEST OPERATIONS")
        print("=" * 50)
        
        engine.print_metrics_summary()
        
        # Show metrics that changed
        new_metrics = engine.get_metrics()
        
        print("\n📈 Metrics Changes:")
        print("-" * 30)
        
        for key in sorted(metrics.keys()):
            old_val = metrics[key]
            new_val = new_metrics[key]
            
            if old_val != new_val:
                change = new_val - old_val if (type(old_val) in (int, float)) else "N/A"
                print(f"{key:25}: {old_val:>8} → {new_val:>8} (Δ{change})")
        
        print("\n✨ Metrics viewing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()