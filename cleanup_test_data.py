#!/usr/bin/env python3
"""
Cleanup script for BabyLM test data.
Removes test-generated files while preserving essential system files.
"""

import os
import shutil
import glob
from pathlib import Path


def cleanup_test_data(base_path: str = "s2_information"):
    """Clean up test-generated data while preserving essential files."""

    base_path = str(base_path)

    print("🧹 Starting cleanup of test data...")

    # Files/directories to preserve (essential system files)
    preserve_patterns = [
        "s2_manifest.json",
        "agency/g2_information/g2_information.dat",  # Epigenome projection
    ]

    # Directories to clean (test-generated data)
    clean_dirs = [
        "agency/g1_information",  # Genome packs
        "agency/g4_information",  # Curriculum files
        "agency/g5_information",  # Session files
        "agents",  # Agent data
    ]

    # Count files before cleanup
    total_files_before = 0
    for clean_dir in clean_dirs:
        clean_path = os.path.join(base_path, clean_dir)
        if os.path.exists(clean_path):
            total_files_before += len(glob.glob(os.path.join(clean_path, "**/*"), recursive=True))

    print(f"📊 Found {total_files_before} test-generated files to clean")

    # Clean each directory
    for clean_dir in clean_dirs:
        clean_path = os.path.join(base_path, clean_dir)
        if os.path.exists(clean_path):
            print(f"🗑️  Cleaning {clean_dir}...")
            shutil.rmtree(clean_path)
            print(f"✅ Cleaned {clean_dir}")

    # Recreate empty directories
    for clean_dir in clean_dirs:
        clean_path = os.path.join(base_path, clean_dir)
        os.makedirs(clean_path, exist_ok=True)
        print(f"📁 Recreated empty directory: {clean_dir}")

    # Count files after cleanup
    total_files_after = 0
    for clean_dir in clean_dirs:
        clean_path = os.path.join(base_path, clean_dir)
        if os.path.exists(clean_path):
            total_files_after += len(glob.glob(os.path.join(clean_path, "**/*"), recursive=True))

    print(f"📊 Cleanup complete! Removed {total_files_before - total_files_after} files")
    print("✅ Essential system files preserved:")
    for pattern in preserve_patterns:
        preserve_path = os.path.join(base_path, pattern)
        if os.path.exists(preserve_path):
            print(f"   - {pattern}")


def analyze_sharding(base_path: str = "s2_information"):
    """Analyze the sharding distribution to understand why so many shards exist."""

    base_path = str(base_path)
    g1_path = os.path.join(base_path, "agency/g1_information")
    if not os.path.exists(g1_path):
        print("❌ No g1_information directory found")
        return

    shards = [d for d in os.listdir(g1_path) if os.path.isdir(os.path.join(g1_path, d))]
    print(f"🔍 Found {len(shards)} shard directories")

    # Analyze shard distribution
    shard_counts = {}
    for shard in shards:
        shard_path = os.path.join(g1_path, shard)
        file_count = len(os.listdir(shard_path))
        shard_counts[shard] = file_count

    # Show top 10 shards by file count
    sorted_shards = sorted(shard_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n📈 Top 10 shards by file count:")
    for i, (shard, count) in enumerate(sorted_shards[:10]):
        print(f"   {i+1:2d}. Shard {shard}: {count} files")

    # Show shard distribution
    print(f"\n📊 Shard distribution:")
    print(f"   - Shards with 0 files: {sum(1 for count in shard_counts.values() if count == 0)}")
    print(f"   - Shards with 1 file: {sum(1 for count in shard_counts.values() if count == 1)}")
    print(f"   - Shards with 2+ files: {sum(1 for count in shard_counts.values() if count >= 2)}")


if __name__ == "__main__":
    print("🚀 BabyLM Test Data Cleanup")
    print("=" * 40)

    # Analyze Current Decoded Genome Cycle
    analyze_sharding()

    print("\n" + "=" * 40)

    # Ask for confirmation
    response = input("Do you want to proceed with cleanup? (y/N): ").strip().lower()
    if response in ["y", "yes"]:
        cleanup_test_data()
    else:
        print("❌ Cleanup cancelled")
