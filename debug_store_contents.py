#!/usr/bin/env python3
"""
Debug script to examine the actual contents of the knowledge store.
"""

import sys
import struct
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from baby.policies import _unpack_phenotype


def debug_store_contents():
    """Debug what's actually in the store file."""

    print("🔍 KNOWLEDGE STORE DEBUG")
    print("=" * 60)

    knowledge_path = PROJECT_ROOT / "toys/training/knowledge/wikipedia_simple.bin"

    if not knowledge_path.exists():
        print("❌ Knowledge store file not found")
        return

    file_size = knowledge_path.stat().st_size
    print(f"📁 File size: {file_size:,} bytes")

    if file_size == 0:
        print("❌ File is empty")
        return

    try:
        with open(knowledge_path, "rb") as f:
            data = f.read()

        print(f"📊 Read {len(data)} bytes")

        # Try to unpack records
        offset = 0
        entries = []

        while offset < len(data):
            try:
                entry, new_offset = _unpack_phenotype(memoryview(data), offset)
                entries.append((offset, new_offset - offset, entry))
                print(f"📦 Entry {len(entries)}: offset={offset}, size={new_offset-offset}, entry={entry}")
                offset = new_offset

                # Stop after first 10 entries to avoid spam
                if len(entries) >= 10:
                    print(f"... (showing first 10 entries out of estimated {file_size//20} total)")
                    break

            except Exception as e:
                print(f"❌ Failed to unpack at offset {offset}: {e}")
                break

        print(f"\n📈 ANALYSIS:")
        print(f"  Total entries found: {len(entries)}")

        if entries:
            avg_size = sum(size for _, size, _ in entries) / len(entries)
            estimated_total = file_size // avg_size
            print(f"  Average entry size: {avg_size:.1f} bytes")
            print(f"  Estimated total entries: {estimated_total}")

            # Analyze entry structure
            orbit_keys = set()
            has_orbit_field = 0
            has_key_field = 0

            for _, _, entry in entries:
                if "orbit" in entry:
                    has_orbit_field += 1
                    orbit_keys.add(entry["orbit"])
                if "key" in entry:
                    has_key_field += 1

            print(f"  Entries with 'orbit' field: {has_orbit_field}/{len(entries)}")
            print(f"  Entries with 'key' field: {has_key_field}/{len(entries)}")
            print(f"  Unique orbits seen: {len(orbit_keys)}")
            print(f"  Orbits: {sorted(orbit_keys)[:20]}{'...' if len(orbit_keys) > 20 else ''}")

            # Check if we're storing efficiently
            if len(orbit_keys) > 0:
                compression_efficiency = estimated_total / len(orbit_keys)
                print(f"  Entries per orbit: {compression_efficiency:.1f}")

                if compression_efficiency > 10:
                    print("⚠️  HIGH REDUNDANCY: Many entries per orbit!")
                    print("⚠️  This suggests we're storing individual tokens instead of compressed orbits")
                elif compression_efficiency < 2:
                    print("✅ GOOD: Low redundancy, close to orbit-level storage")

    except Exception as e:
        print(f"❌ Error reading store: {e}")


def debug_phenotype_format():
    """Show the current phenotype format."""
    print("\n🔬 PHENOTYPE FORMAT")
    print("=" * 60)

    from baby.policies import _STRUCT_FMT, _STRUCT_SIZE

    print(f"Struct format: {_STRUCT_FMT}")
    print(f"Fixed header size: {_STRUCT_SIZE} bytes")

    # Show what each field means
    print("Field breakdown:")
    print("  B orbit        (uint8)   # phenomenological orbit (0-255)")
    print("  B mask         (uint8)   # holographic compression mask")
    print("  H conf_f16     (uint16)  # confidence weighted by orbit cardinality")
    print("  I tokens_count (uint32)  # number of tokens in this orbit")
    print("  + variable-length token list")


if __name__ == "__main__":
    debug_store_contents()
    debug_phenotype_format()
