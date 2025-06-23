#!/usr/bin/env python3
"""
Migration script to update existing knowledge packages to mechanical engine format.
"""
import sys
from pathlib import Path


def migrate_knowledge_packages():
    """Migrate existing knowledge packages to support encryption"""
    data_dir = Path("data/knowledge")

    for knowledge_dir in data_dir.iterdir():
        if not knowledge_dir.is_dir():
            continue

        print(f"Migrating {knowledge_dir.name}...")

        # Rename unencrypted files
        old_log = knowledge_dir / "navigation_log" / "genome.log"
        if old_log.exists():
            print(f"  - Found unencrypted navigation log: {old_log}")
            # Keep as-is for now, will be encrypted on next save

        old_gene = knowledge_dir / "gene.dat"
        if old_gene.exists():
            print(f"  - Found unencrypted gene: {old_gene}")
            # Keep as-is for now, will be encrypted on next save


def verify_operator_matrix():
    """Verify operator matrix exists"""
    matrix_path = Path("src/core/operator_matrix.dat")

    if not matrix_path.exists():
        print("ERROR: operator_matrix.dat not found!")
        print("Run: python gyro_tools/build_operator_matrix.py")
        return False

    print(f"✓ Operator matrix found ({matrix_path.stat().st_size} bytes)")
    return True


def main():
    print("GyroSI Mechanical Engine Migration")
    print("=" * 40)

    # Check operator matrix
    if not verify_operator_matrix():
        sys.exit(1)

    # Migrate knowledge packages
    migrate_knowledge_packages()

    print("\nMigration complete!")
    print("\nNext steps:")
    print("1. Set GYROSI_USER_KEY environment variable")
    print("2. Run tests: pytest tests/")
    print("3. Start using the mechanical engine")


if __name__ == "__main__":
    main()
