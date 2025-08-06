#!/usr/bin/env python3
"""
Test script to analyze what we're actually storing vs what we should store.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from baby.intelligence import GyroSI
from baby.policies import OrbitStore
from baby import governance


def analyze_compression():
    """Analyze what we're storing and why compression isn't working."""

    print("🔍 COMPRESSION ANALYSIS")
    print("=" * 60)

    # 1. Check tape size
    tape_path = PROJECT_ROOT / "toys/training/knowledge/wikipedia_simple.gyro"
    if tape_path.exists():
        tape_size = tape_path.stat().st_size
        print(f"📼 Tape size: {tape_size:,} bytes ({tape_size/1024/1024:.1f} MB)")
    else:
        print("❌ No tape found")
        return

    # 2. Check knowledge store size
    knowledge_path = PROJECT_ROOT / "toys/training/knowledge/wikipedia_simple.bin"
    if knowledge_path.exists():
        knowledge_size = knowledge_path.stat().st_size
        print(f"🧠 Knowledge store size: {knowledge_size:,} bytes ({knowledge_size/1024/1024:.1f} MB)")

        if knowledge_size > 0:
            compression_ratio = tape_size / knowledge_size
            print(f"📊 Current compression ratio: {compression_ratio:.2f}x")

            if compression_ratio < 1.0:
                expansion_ratio = knowledge_size / tape_size
                print(f"💥 EXPANSION (not compression): {expansion_ratio:.2f}x BIGGER!")
        else:
            print("❌ Knowledge store is empty (0 bytes)")
    else:
        print("❌ No knowledge store found")
        return

    print("\n🧬 THEORETICAL ANALYSIS")
    print("=" * 60)

    # 3. Load the theoretical maps
    try:
        # Load phenomenology map
        phen_map_path = PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"
        if phen_map_path.exists():
            phenomenology_map = np.load(phen_map_path)
            print(f"📋 Phenomenology map: {len(phenomenology_map):,} states -> 256 orbits")

            # Calculate theoretical compression from 788K states to 256 orbits
            theoretical_compression = len(phenomenology_map) / 256
            print(f"🎯 Theoretical state->orbit compression: {theoretical_compression:.0f}x")
        else:
            print("❌ No phenomenology map found")

        # Load ontology
        ontology_path = PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"
        if ontology_path.exists():
            ontology = np.load(ontology_path)
            print(f"📚 Ontology size: {len(ontology):,} states")
        else:
            print("❌ No ontology found")

    except Exception as e:
        print(f"❌ Error loading theoretical maps: {e}")

    print("\n💾 ACTUAL STORAGE ANALYSIS")
    print("=" * 60)

    # 4. Analyze what's actually in the knowledge store
    if knowledge_path.exists() and knowledge_size > 0:
        try:
            # Try to load and analyze the orbit store
            config = {
                "ontology_path": str(PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"),
                "epistemology_path": str(PROJECT_ROOT / "memories/public/meta/epistemology.npy"),
                "phenomenology_map_path": str(PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"),
                "private_knowledge_path": str(knowledge_path),
                "public_knowledge_path": str(PROJECT_ROOT / "memories/public/meta"),
                "learn_batch_size": 50,
                "enable_phenomenology_storage": True,
                "preferences": {},
            }

            agent = GyroSI(config, agent_id="compression_test", base_path=PROJECT_ROOT)
            store = agent.engine.operator.store

            # Count what's actually stored
            if hasattr(store, "iter_entries"):
                entries = list(store.iter_entries())
                print(f"📦 Total entries in store: {len(entries)}")

                if len(entries) > 0:
                    # Sample a few entries to see the structure
                    print("\n🔬 Sample entries:")
                    for i, (key, entry) in enumerate(entries[:5]):
                        print(f"  {i+1}. Key: {key}, Entry: {entry}")

                    # Check what orbits are represented
                    unique_keys = set()
                    for key, entry in entries:
                        unique_keys.add(key)

                    print(f"🎯 Unique keys/orbits: {len(unique_keys)}")
                    print(f"📈 Average entries per key: {len(entries)/len(unique_keys):.2f}")

                    # Calculate actual storage efficiency
                    bytes_per_entry = knowledge_size / len(entries)
                    print(f"💾 Bytes per entry: {bytes_per_entry:.1f}")

            agent.close()

        except Exception as e:
            print(f"❌ Error analyzing store: {e}")

    print("\n🎯 DIAGNOSIS")
    print("=" * 60)

    # The problem analysis
    print("Based on the theory:")
    print("✅ Should store: ~256 orbit entries (phenomenological classes)")
    print("✅ Should achieve: ~3000x compression (788K states -> 256 orbits)")
    print("✅ Should result: Tape larger than knowledge store")
    print()

    if knowledge_size > tape_size:
        print("❌ ACTUAL PROBLEM: We're storing MORE than the input!")
        print("❌ This suggests we're storing individual (state, token) pairs")
        print("❌ Instead of compressed orbit-level information")
        print()
        print("🔧 FIX NEEDED:")
        print("1. Store by ORBIT (0-255) not by STATE (0-788K)")
        print("2. Multiple tokens should map to same orbit entry")
        print("3. Use holographic fold to compress within orbits")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_compression()
