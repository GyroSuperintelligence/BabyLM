"""
s2_manifest.py - S2 Information Layer Manifest

Defines the storage structure and constants for the S2 layer.
"""

import json
import os

MANIFEST = {
    "version": "1.0",
    "pack_size": 64 * 1024 * 1024,  # 64MB
    "shard_prefix_length": 2,  # hex digits for sharding by first byte
    "archive_shard_limit": 10000,  # max records per archive file
}


def get_manifest():
    """Return a copy of the S2 manifest."""
    return MANIFEST.copy()


def get_shard_from_uuid(uuid_str: str) -> str:
    """
    Extract shard prefix from UUID string.

    Args:
        uuid_str: UUID string (with or without hyphens)

    Returns:
        Two-character hex shard prefix
    """
    # Remove hyphens if present
    clean_uuid = uuid_str.replace("-", "")
    return clean_uuid[:2].lower()


def ensure_s2_structure(base_path: str = "s2_information"):
    """
    Ensure the S2 directory structure exists.

    Args:
        base_path: Base path for S2 information storage
    """
    # Create base structure
    os.makedirs(base_path, exist_ok=True)

    # Write manifest
    manifest_path = os.path.join(base_path, "s2_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(MANIFEST, f, indent=2)

    # Create agency directories (shards will be created on-demand by IntelligenceEngine)
    agency_path = os.path.join(base_path, "agency")
    os.makedirs(os.path.join(agency_path, "g1_information"), exist_ok=True)
    os.makedirs(os.path.join(agency_path, "g2_information"), exist_ok=True)
    os.makedirs(os.path.join(agency_path, "g4_information"), exist_ok=True)
    os.makedirs(os.path.join(agency_path, "g5_information"), exist_ok=True)

    # Create agents directory (shards will be created on-demand by IntelligenceEngine)
    agents_path = os.path.join(base_path, "agents")
    os.makedirs(agents_path, exist_ok=True)


if __name__ == "__main__":
    ensure_s2_structure()
