"""
information.py - S2 Information processing for GyroSI Baby LM

This module implements the Information Engine for processing input streams
and generating output streams, representing the Information (S2) layer
of the Common Governance Model. It's responsible for all persistent storage
operations and registry management.
"""

import os
import uuid
import numpy as np
import fcntl
from pathlib import Path
from typing import Tuple, Callable, List, Dict, Optional, Union
from datetime import datetime
import re

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from baby.inference import InferenceEngine
from baby.types import FormatMetadata, ThreadMetadata, GeneKeysMetadata

# pyright: reportMissingModuleSource=false
try:
    import orjson as json

    # orjson API: loads(bytes), dumps(obj) -> bytes
    def json_loads(s):
        if isinstance(s, str):
            s = s.encode("utf-8")
        return json.loads(s)

    def json_dumps(obj):
        return json.dumps(obj).decode("utf-8")

except ImportError:
    try:
        import ujson as json

        json_loads = json.loads
        json_dumps = json.dumps
    except ImportError:
        import json as std_json

        json_loads = std_json.loads
        json_dumps = std_json.dumps


# ====================================================================
# Persistent Storage Helpers - Sharding and Registry
# ====================================================================


def get_memory_preferences() -> Dict:
    """
    Load memory preferences from file or create with defaults if not exists.

    Returns:
        Dict: Memory preferences including sharding configuration
    """
    prefs_path = Path("memories/memory_preferences.json")
    prefs_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(prefs_path, "r") as f:
            prefs = json_loads(f.read())
    except (FileNotFoundError, ValueError):
        # Create default preferences
        prefs = {
            "sharding": {"width": 2, "max_files": 30000, "second_level": True},
            "storage_config": {"max_thread_size_mb": 64, "encryption_algorithm": "AES-256-GCM"},
            "format_config": {"default_cgm_version": "1.0.0", "max_character_label_length": 128},
        }

        with open(prefs_path, "w") as f:
            f.write(json_dumps(prefs))

    return prefs


def shard_path(root: Path, uuid_: str, width=2, limit=30_000) -> Path:
    """
    Calculate the appropriate shard path for a UUID.

    Args:
        root: Root directory where shards are stored
        uuid_: UUID to calculate shard for
        width: Number of hex characters to use for first-level shard
        limit: Maximum files in a first-level shard before using second level

    Returns:
        Path: Complete path to the shard directory
    """
    if not uuid_ or len(uuid_) < 8:
        raise ValueError(f"Invalid UUID format: {uuid_}")

    # Get first-level shard (first `width` characters)
    first_level = uuid_[:width]
    first_path = root / first_level

    # Check if we need second-level sharding
    prefs = get_memory_preferences()
    second_level_enabled = prefs["sharding"]["second_level"]

    if second_level_enabled and first_path.exists():
        # Count ALL files and directories in first-level shard, not just registry entries
        count = 0
        registry_path = first_path / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        registry = json_loads(f.read())
                        count = registry.get("count", 0)
                    except (json.JSONDecodeError, IOError):
                        pass
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except Exception:
                pass
        # If registry count is unreliable, count files physically
        if count == 0:
            count = sum(1 for _ in first_path.iterdir())
        # Use second-level sharding if over limit
        if count > limit:
            second_level = uuid_[width : width * 2]
            return first_path / second_level
    return first_path


def atomic_write(path: Path, data: bytes) -> None:
    """
    Write data to a file atomically using a temporary file and rename.

    Args:
        path: Path to write to
        data: Bytes to write
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file in same directory
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "wb") as f:
        # Lock the file for exclusive access
        fcntl.flock(f, fcntl.LOCK_EX)

        # Write data and ensure it's flushed to disk
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    # Atomic rename (this is atomic on most file systems)
    os.rename(tmp_path, path)

    # Sync the directory to ensure rename is durable
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def update_registry(dirpath: Path, uuid_: str, update_parent: bool = True) -> None:
    """
    Update a registry file with a new UUID, optionally updating parent registry.

    Args:
        dirpath: Directory containing the registry
        uuid_: UUID to add to registry
        update_parent: Whether to update parent registry if this is a second-level shard
    """
    registry_path = dirpath / "registry.json"

    # Use file locking instead of sentinel file
    with open(registry_path, "a+") as f:
        # Lock the file for exclusive access
        fcntl.flock(f, fcntl.LOCK_EX)

        # Seek to beginning to read contents
        f.seek(0)

        # Read existing registry or create new one
        try:
            registry = json_loads(f.read())
        except (json.JSONDecodeError, ValueError):
            registry = {"count": 0, "uuids": []}

        # Add UUID if not already present and not empty
        if uuid_ and uuid_ not in registry["uuids"]:
            registry["uuids"].append(uuid_)
            registry["count"] = len(registry["uuids"])

            # Truncate file and write updated registry
            f.seek(0)
            f.truncate()
            f.write(json_dumps(registry))
            f.flush()
            os.fsync(f.fileno())

    # Update parent registry if requested and this is a second-level shard
    if update_parent and len(dirpath.parts) >= 2:
        parent_dir = dirpath.parent
        parent_uuid = dirpath.name  # The second-level shard name

        # If parent exists and this is a second-level shard
        if parent_dir.exists() and len(parent_uuid) == 2:  # 2-char hex is first-level shard
            update_registry(parent_dir, parent_uuid, False)  # Don't recurse again


def rebuild_registry(dirpath: Path) -> None:
    """
    Rebuild a registry by scanning the directory for objects.

    Args:
        dirpath: Directory to scan for rebuilding registry
    """
    # Remove any temporary files
    for tmp_file in dirpath.glob("*.tmp"):
        tmp_file.unlink()

    # Get all files/directories with proper naming pattern
    valid_objects = []

    uuid_pattern = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")

    # Check for agent directories
    for item in dirpath.glob("agent-*"):
        if item.is_dir():
            m = uuid_pattern.search(item.name)
            if m:
                valid_objects.append(m.group(1))

    # Check for thread files
    for item in dirpath.glob("thread-*.json"):
        if item.is_file():
            m = uuid_pattern.search(item.name)
            if m:
                valid_objects.append(m.group(1))

    # Check for thread encrypted files
    for item in dirpath.glob("thread-*.enc"):
        if item.is_file():
            m = uuid_pattern.search(item.name)
            if m:
                valid_objects.append(m.group(1))

    # Check for key files
    for item in dirpath.glob("key-*.bin.enc"):
        if item.is_file():
            m = uuid_pattern.search(item.name)
            if m:
                valid_objects.append(m.group(1))

    # Check for format files
    for item in dirpath.glob("format-*.json"):
        if item.is_file():
            m = uuid_pattern.search(item.name)
            if m:
                valid_objects.append(m.group(1))

    # Check for second-level shards (directories with 2-char hex names)
    for item in dirpath.glob("[0-9a-f][0-9a-f]"):
        if item.is_dir():
            valid_objects.append(item.name)

    # Create new registry
    registry = {"count": len(valid_objects), "uuids": valid_objects}

    # Write registry
    registry_path = dirpath / "registry.json"
    with open(registry_path, "w") as f:
        f.write(json_dumps(registry))


# ====================================================================
# Agent UUID Management
# ====================================================================


def ensure_agent_uuid() -> str:
    """
    Get the current agent UUID, or create and register a new one if missing.

    Returns:
        str: Agent UUID
    """
    private_dir = Path("memories/private")
    private_dir.mkdir(parents=True, exist_ok=True)

    agents_dir = private_dir / "agents"
    agents_dir.mkdir(exist_ok=True)

    # Look for any shard directories
    for shard_dir in agents_dir.glob("*"):
        if shard_dir.is_dir():
            # Check for agent directories at first level
            for agent_dir in shard_dir.glob("agent-*"):
                if agent_dir.is_dir():
                    # Extract UUID from directory name
                    agent_uuid = "-".join(agent_dir.name.split("-")[1:])
                    return agent_uuid

            # Check for second-level shards
            for second_shard in shard_dir.glob("[0-9a-f][0-9a-f]"):
                if second_shard.is_dir():
                    # Check for agent directories at second level
                    for agent_dir in second_shard.glob("agent-*"):
                        if agent_dir.is_dir():
                            # Extract UUID from directory name
                            agent_uuid = "-".join(agent_dir.name.split("-")[1:])
                            return agent_uuid

    # No agent found, create a new one
    return assign_agent_uuid(str(uuid.uuid4()))


def assign_agent_uuid(new_uuid: str) -> str:
    """
    Assign a new UUID to the agent, creating all necessary directories.

    Args:
        new_uuid: New UUID to assign to agent

    Returns:
        str: The new agent UUID
    """
    private_dir = Path("memories/private/agents")
    private_dir.mkdir(parents=True, exist_ok=True)

    # Calculate shard path
    shard = shard_path(private_dir, new_uuid)
    shard.mkdir(parents=True, exist_ok=True)

    # Create agent directory
    agent_dir = shard / f"agent-{new_uuid}"
    agent_dir.mkdir(exist_ok=True)

    # Create threads and keys directories
    (agent_dir / "threads").mkdir(exist_ok=True)
    (agent_dir / "keys").mkdir(exist_ok=True)

    # Initialize empty registries
    update_registry(shard, new_uuid)
    update_registry(agent_dir / "threads", "")
    update_registry(agent_dir / "keys", "")

    return new_uuid


# ====================================================================
# Thread Lifecycle Management
# ====================================================================


def _get_thread_path(thread_uuid: str, agent_uuid: Optional[str]) -> Path:
    if agent_uuid:
        root_dir = Path("memories/private/agents")
        agent_shard = shard_path(root_dir, agent_uuid)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        return agent_dir / "threads"
    else:
        return Path("memories/public/threads")


def create_thread(
    privacy: str,
    parent_uuid: Optional[str],
    format_uuid: str,
    thread_name: Optional[str] = None,
    curriculum: Optional[str] = None,
    tags: Optional[list] = None,
) -> str:
    """
    Create a new thread (public or private).
    Args:
        privacy: 'public' or 'private'
        parent_uuid: Optional parent thread UUID
        format_uuid: Format UUID to use for the thread
        thread_name: Optional human-friendly name for the thread
        curriculum: Optional curriculum label
        tags: Optional list of tags
    Returns:
        str: New thread UUID
    """
    thread_uuid = str(uuid.uuid4())
    agent_uuid = None if privacy == "public" else ensure_agent_uuid()
    threads_dir = _get_thread_path(thread_uuid, agent_uuid)
    thread_shard = shard_path(threads_dir, thread_uuid)
    thread_shard.mkdir(parents=True, exist_ok=True)
    parent_name = None
    if parent_uuid:
        parent_shard = shard_path(threads_dir, parent_uuid)
        parent_meta_path = parent_shard / f"thread-{parent_uuid}.json"
        if parent_meta_path.exists():
            with open(parent_meta_path, "r") as f:
                parent_meta = json_loads(f.read())
            parent_name = parent_meta.get("thread_name")
    now = datetime.now().isoformat()
    thread_meta: ThreadMetadata = {
        "thread_uuid": thread_uuid,
        "thread_name": thread_name,
        "agent_uuid": agent_uuid,  # Deprecated
        "parent_uuid": parent_uuid,
        "parent_name": parent_name,
        "children": [],
        "format_uuid": format_uuid,
        "curriculum": curriculum,
        "tags": tags,
        "created_at": now,
        "last_updated": now,
        "size_bytes": 0,
        "privacy": privacy,
    }
    thread_meta_path = thread_shard / f"thread-{thread_uuid}.json"
    with open(thread_meta_path, "w") as f:
        f.write(json_dumps(thread_meta))
    update_registry(threads_dir, thread_uuid)
    update_registry(thread_shard, thread_uuid)
    return thread_uuid


def save_thread(thread_uuid: str, content: bytes, privacy: str = "private") -> None:
    """
    Save thread content to disk, encrypted or plaintext based on privacy.
    Args:
        thread_uuid (str): The thread UUID.
        content (bytes): The thread content to save.
        privacy (str): 'private' or 'public'.
    """
    # Use the same path calculation as load_thread for consistency
    agent_uuid = None if privacy == "public" else ensure_agent_uuid()
    threads_dir = _get_thread_path(thread_uuid, agent_uuid)
    thread_shard = shard_path(threads_dir, thread_uuid)
    ext = ".enc" if agent_uuid else ".ndjson"
    thread_path = thread_shard / f"thread-{thread_uuid}{ext}"
    thread_shard.mkdir(parents=True, exist_ok=True)
    with open(thread_path, "wb") as f:
        f.write(content)


def load_thread(agent_uuid: Optional[str], thread_uuid: str) -> Optional[bytes]:
    """
    Load thread content from disk (public or private).
    Args:
        agent_uuid: Agent UUID (None for public thread)
        thread_uuid: Thread UUID
    Returns:
        bytes: Thread content or None if not found
    """
    threads_dir = _get_thread_path(thread_uuid, agent_uuid)
    thread_shard = shard_path(threads_dir, thread_uuid)
    ext = ".enc" if agent_uuid else ".ndjson"
    thread_path = thread_shard / f"thread-{thread_uuid}{ext}"
    if not thread_path.exists():
        return None
    with open(thread_path, "rb") as f:
        return f.read()


# ====================================================================
# Key Storage
# ====================================================================


def store_thread_key(agent_uuid: str, thread_uuid: str, key: bytes, agent_secret: str) -> None:
    """
    Store an encryption key for a thread.

    Args:
        agent_uuid: Agent UUID
        thread_uuid: Thread UUID
        key: Key to store (must be exactly 32 bytes for AES-256)
        agent_secret: Agent secret for encryption
    """
    if len(key) != 32:
        raise ValueError(f"Thread key must be exactly 32 bytes for AES-256, got {len(key)}")

    # Get agent directory
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, agent_uuid)
    agent_dir = agent_shard / f"agent-{agent_uuid}"

    # Calculate key shard (using thread UUID)
    keys_dir = agent_dir / "keys"
    key_shard = shard_path(keys_dir, thread_uuid)
    key_shard.mkdir(parents=True, exist_ok=True)

    # Derive encryption key using PBKDF2-HMAC-SHA256
    salt = agent_uuid.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key for AES-256
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    agent_key = kdf.derive(agent_secret.encode("utf-8"))

    # Encrypt with AES-256-GCM
    nonce = os.urandom(12)
    cipher = Cipher(algorithms.AES(agent_key), modes.GCM(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(key) + encryptor.finalize()

    # Store as: nonce (12) + tag (16) + ciphertext
    encrypted_blob = nonce + encryptor.tag + ciphertext

    # Write key file
    key_path = key_shard / f"key-{thread_uuid}.bin.enc"
    atomic_write(key_path, encrypted_blob)

    # Update registries
    update_registry(keys_dir, thread_uuid)
    update_registry(key_shard, thread_uuid)


def store_gene_keys(
    thread_uuid: str,
    gene_keys: list[GeneKeysMetadata],
    privacy: str,
    agent_secret: Optional[str] = None,
    agent_uuid: Optional[str] = None,
) -> None:
    """
    Store gene keys (pattern observation logs) in the appropriate location (public or private).
    If privacy is 'private' and agent_secret is provided, encrypt and store privately. 
    Otherwise, store unencrypted in public.
    """
    import struct

    if privacy == "private" and agent_secret:
        if agent_uuid is None:
            raise ValueError("agent_uuid must not be None for private gene key storage")
        # PRIVATE (encrypted, append-only per-record)
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, agent_uuid)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        keys_dir = agent_dir / "keys"
        key_shard = shard_path(keys_dir, thread_uuid)
        key_shard.mkdir(parents=True, exist_ok=True)
        gene_keys_path = key_shard / f"gene-{thread_uuid}.ndjson.enc"
        salt = (agent_uuid + thread_uuid + "gene_keys").encode("utf-8")
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
        encryption_key = kdf.derive(agent_secret.encode("utf-8"))
        with open(gene_keys_path, "ab") as f:
            for gk in gene_keys:
                ndjson_line = json_dumps({**gk, "privacy": privacy, "agent_uuid": agent_uuid})
                ndjson_bytes = ndjson_line.encode("utf-8")
                nonce = os.urandom(12)
                cipher = Cipher(algorithms.AES(encryption_key), modes.GCM(nonce), backend=default_backend())
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(ndjson_bytes) + encryptor.finalize()
                encrypted_blob = nonce + encryptor.tag + ciphertext
                # Write length prefix (4 bytes, big-endian)
                f.write(struct.pack(">I", len(encrypted_blob)))
                f.write(encrypted_blob)
        update_registry(key_shard, thread_uuid)
    else:
        # PUBLIC (unencrypted)
        keys_dir = Path("memories/public/keys")
        key_shard = shard_path(keys_dir, thread_uuid)
        key_shard.mkdir(parents=True, exist_ok=True)
        gene_keys_path = key_shard / f"gene-{thread_uuid}.ndjson"
        # Change mode from "w" to "a" for appending
        with open(gene_keys_path, "a", encoding="utf-8") as f:
            for gk in gene_keys:
                f.write(json_dumps({**gk, "privacy": privacy, "agent_uuid": None}) + "\n")
        update_registry(key_shard, thread_uuid)


def load_gene_keys(
    thread_uuid: str,
    agent_uuid: Optional[str] = None,
    agent_secret: Optional[str] = None,
) -> list[GeneKeysMetadata]:
    """
    Load gene keys from the appropriate location (public or private).
    If agent_uuid and agent_secret are provided, decrypt and load privately. Otherwise, load unencrypted from public.
    """
    import struct

    if agent_uuid and agent_secret:
        # PRIVATE (encrypted, per-record)
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, agent_uuid)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        keys_dir = agent_dir / "keys"
        key_shard = shard_path(keys_dir, thread_uuid)
        private_path = key_shard / f"gene-{thread_uuid}.ndjson.enc"
        if not private_path.exists():
            # Fallback: Try public path
            public_dir = Path("memories/public/keys")
            public_shard = shard_path(public_dir, thread_uuid)
            public_path = public_shard / f"gene-{thread_uuid}.ndjson"
            if public_path.exists():
                with open(public_path, "r", encoding="utf-8") as f:
                    return [json_loads(line) for line in f if line.strip()]
            return []
        salt = (agent_uuid + thread_uuid + "gene_keys").encode("utf-8")
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
        decryption_key = kdf.derive(agent_secret.encode("utf-8"))
        gene_keys = []
        with open(private_path, "rb") as f:
            while True:
                len_bytes = f.read(4)
                if not len_bytes or len(len_bytes) < 4:
                    break
                (blob_len,) = struct.unpack(">I", len_bytes)
                encrypted_blob = f.read(blob_len)
                if len(encrypted_blob) != blob_len:
                    break  # Corrupt or truncated file
                nonce = encrypted_blob[:12]
                tag = encrypted_blob[12:28]
                ciphertext = encrypted_blob[28:]
                cipher = Cipher(algorithms.AES(decryption_key), modes.GCM(nonce, tag), backend=default_backend())
                decryptor = cipher.decryptor()
                try:
                    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
                    ndjson_line = decrypted_data.decode("utf-8")
                    gene_keys.append(json_loads(ndjson_line))
                except Exception:
                    pass  # Skip corrupt record
        return gene_keys
    else:
        # PUBLIC (unencrypted)
        keys_dir = Path("memories/public/keys")
        key_shard = shard_path(keys_dir, thread_uuid)
        gene_keys_path = key_shard / f"gene-{thread_uuid}.ndjson"
        if not gene_keys_path.exists():
            return []
        with open(gene_keys_path, "r", encoding="utf-8") as f:
            return [json_loads(line) for line in f if line.strip()]


def load_thread_key(
    agent_uuid: str,
    thread_uuid: str,
    agent_secret: str,
) -> Optional[bytes]:
    """
    Load and decrypt the thread key for a private thread.
    Returns the 32-byte key, or None if not found or decryption fails.
    """
    # Use top-level imports, remove redundant imports
    # Get agent directory
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, agent_uuid)
    agent_dir = agent_shard / f"agent-{agent_uuid}"
    keys_dir = agent_dir / "keys"
    key_shard = shard_path(keys_dir, thread_uuid)
    key_path = key_shard / f"key-{thread_uuid}.bin.enc"
    if not key_path.exists():
        return None
    with open(key_path, "rb") as f:
        encrypted_blob = f.read()
    # Derive encryption key using PBKDF2-HMAC-SHA256
    salt = agent_uuid.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key for AES-256
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    agent_key = kdf.derive(agent_secret.encode("utf-8"))
    # Decrypt with AES-256-GCM
    nonce = encrypted_blob[:12]
    tag = encrypted_blob[12:28]
    ciphertext = encrypted_blob[28:]
    cipher = Cipher(algorithms.AES(agent_key), modes.GCM(nonce, tag), backend=default_backend())
    decryptor = cipher.decryptor()
    try:
        key = decryptor.update(ciphertext) + decryptor.finalize()
        if len(key) != 32:
            return None
        return key
    except Exception:
        return None


# ====================================================================
# Relationship Traversal
# ====================================================================


def parent(agent_uuid: str, thread_uuid: str) -> Optional[str]:
    """
    Get the parent thread UUID for a thread.

    Args:
        agent_uuid: Agent UUID
        thread_uuid: Thread UUID

    Returns:
        str: Parent UUID or None if not found
    """
    # Get agent directory
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, agent_uuid)
    agent_dir = agent_shard / f"agent-{agent_uuid}"

    # Calculate thread shard
    threads_dir = agent_dir / "threads"
    thread_shard = shard_path(threads_dir, thread_uuid)

    # Check thread metadata
    meta_path = thread_shard / f"thread-{thread_uuid}.json"

    if not meta_path.exists():
        return None

    with open(meta_path, "r") as f:
        meta = json_loads(f.read())

    return meta.get("parent_uuid")


def children(agent_uuid: str, thread_uuid: str) -> List[str]:
    """
    Get the child thread UUIDs for a thread.
    Args:
        agent_uuid: Agent UUID
        thread_uuid: Thread UUID
    Returns:
        list[str]: List of child UUIDs
    """
    # Get agent directory
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, agent_uuid)
    agent_dir = agent_shard / f"agent-{agent_uuid}"
    # Calculate thread shard
    threads_dir = agent_dir / "threads"
    thread_shard = shard_path(threads_dir, thread_uuid)
    # Check thread metadata
    meta_path = thread_shard / f"thread-{thread_uuid}.json"
    if not meta_path.exists():
        return []
    with open(meta_path, "r") as f:
        meta = json_loads(f.read())
    val = [c["uuid"] for c in meta.get("children", [])]
    # Always return a flat list of strings (write path enforces this)
    return list(val) if isinstance(val, list) else []


# ====================================================================
# Format Management
# ====================================================================


def list_formats() -> List[str]:
    """
    List all available format UUIDs.

    Returns:
        list[str]: List of format UUIDs
    """
    formats_dir = Path("memories/public/formats")
    formats_dir.mkdir(parents=True, exist_ok=True)

    # Get all format registries
    format_uuids = []

    # Use recursive glob to find all format files at any level
    for format_file in formats_dir.rglob("format-*.json"):
        # Extract the full UUID from the filename
        name = format_file.name
        if name.startswith("format-") and name.endswith(".json"):
            uuid_ = name[len("format-") : -len(".json")]
            format_uuids.append(uuid_)

    return format_uuids


def load_format(format_uuid: str) -> Optional[FormatMetadata]:
    """
    Load a format from disk.
    Args:
        format_uuid: Format UUID
    Returns:
        FormatMetadata: Format data or None if not found
    """
    formats_dir = Path("memories/public/formats")
    format_shard = shard_path(formats_dir, format_uuid)
    format_path = format_shard / f"format-{format_uuid}.json"
    if not format_path.exists():
        return None
    with open(format_path, "r") as f:
        format_data = json_loads(f.read())
    return format_data  # type: ignore


def store_format(format_data: FormatMetadata) -> str:
    """
    Store a format to disk.
    Args:
        format_data: FormatMetadata (must contain format_uuid)
    Returns:
        str: Format UUID
    """
    format_uuid = format_data.get("format_uuid")
    if not format_uuid:
        format_uuid = str(uuid.uuid4())
        format_data["format_uuid"] = format_uuid
    formats_dir = Path("memories/public/formats")
    format_shard = shard_path(formats_dir, format_uuid)
    format_shard.mkdir(parents=True, exist_ok=True)
    # Remove any pattern_distances matrix if present (store separately)
    if "pattern_distances" in format_data:
        pattern_distances = format_data["pattern_distances"]
        distances_path = format_shard / f"pattern-distances-{format_uuid}.dat"
        if isinstance(pattern_distances, dict) and "matrix" in pattern_distances:
            matrix = np.array(pattern_distances["matrix"], dtype=np.float32)
            with open(distances_path, "wb") as f:
                matrix.tofile(f)
            format_data["pattern_distances"] = {
                "computed_at": pattern_distances.get("computed_at", datetime.now().isoformat()),
                "cgm_version": pattern_distances.get("cgm_version", "1.0.0"),
                "path": str(distances_path.relative_to(formats_dir)),
            }
    format_path = format_shard / f"format-{format_uuid}.json"
    with open(format_path, "w") as f:
        f.write(json_dumps(format_data))
    update_registry(formats_dir, format_uuid)
    update_registry(format_shard, format_uuid)
    return format_uuid


def load_pattern_distances(format_uuid: str) -> Optional[np.ndarray]:
    """
    Load pattern distance matrix for a format.

    Args:
        format_uuid: Format UUID

    Returns:
        np.ndarray: Matrix of pattern distances or None if not found
    """
    formats_dir = Path("memories/public/formats")
    format_shard = shard_path(formats_dir, format_uuid)
    distances_path = format_shard / f"pattern-distances-{format_uuid}.dat"

    if not distances_path.exists():
        return None

    # Load matrix from file
    try:
        matrix = np.fromfile(distances_path, dtype=np.float32)
        return matrix.reshape((256, 256))
    except Exception:
        return None


# ====================================================================
# Generic Object Storage
# ====================================================================


def store_object(obj_type: str, payload: Union[bytes, Dict], ext: str = "dat") -> str:
    """
    Store a generic object with proper UUID generation and sharding.

    Args:
        obj_type: Object type (agent, thread, key, format)
        payload: Data to store (bytes for binary, dict for JSON)
        ext: File extension (default: dat)

    Returns:
        str: UUID of stored object
    """
    # Validate object type
    if obj_type not in ["agent", "thread", "key", "format"]:
        raise ValueError(f"Invalid object type: {obj_type}")

    # Generate UUID
    obj_uuid = str(uuid.uuid4())

    # Determine root directory
    if obj_type == "format":
        root_dir = Path("memories/public/formats")
    elif obj_type == "agent":
        root_dir = Path("memories/private/agents")
    else:
        raise ValueError(f"Direct storage not supported for {obj_type}, use specific helpers")

    # Calculate shard
    shard = shard_path(root_dir, obj_uuid)
    shard.mkdir(parents=True, exist_ok=True)

    # Create filename
    filename = f"{obj_type}-{obj_uuid}.{ext}"
    file_path = shard / filename

    # Write data
    if isinstance(payload, dict):
        with open(file_path, "w") as f:
            f.write(json_dumps(payload))
    else:
        atomic_write(file_path, payload)

    # Update registry
    update_registry(root_dir, obj_uuid)
    update_registry(shard, obj_uuid)

    return obj_uuid


# ====================================================================
# Passive-Active Intelligence Components
# ====================================================================


class ThreadChainCache:
    """
    Cache thread relationships and patterns for faster inference-time access.
    This enables the "passive" thread history to actively influence inference.
    """

    def __init__(self, agent_uuid: str, agent_secret: str, cache_size: int = 100):
        self.agent_uuid = agent_uuid
        self.agent_secret = agent_secret
        self.cache_size = cache_size
        self._pattern_cache = {}  # pattern_index -> [(thread_uuid, position), ...]
        self._thread_cache = {}  # thread_uuid -> {metadata, patterns, relationships}
        self._access_order = []  # LRU tracking

    def get_context_window(self, thread_uuid: str, window_size: int = 3) -> List[Dict]:
        """
        Get surrounding threads for context (like attention mechanism).
        Returns threads in temporal/relational proximity.
        """
        if thread_uuid in self._thread_cache:
            self._update_access(thread_uuid)
            cached = self._thread_cache[thread_uuid]

            # Get parent and children threads
            context_threads = []

            # Add parent context
            if cached.get("parent_uuid"):
                parent_data = self._load_thread_patterns(cached["parent_uuid"])
                if parent_data:
                    context_threads.append(parent_data)

            # Add current thread
            context_threads.append(cached)

            # Add child contexts (up to window_size)
            for child_uuid in cached.get("children", [])[: window_size - len(context_threads)]:
                child_data = self._load_thread_patterns(child_uuid)
                if child_data:
                    context_threads.append(child_data)

            return context_threads

        return []

    def find_pattern_contexts(self, pattern_index: int, max_contexts: int = 10) -> List[Dict]:
        """
        Find historical contexts where a specific pattern appeared.
        This allows the system to "remember" successful pattern usages.
        """
        if pattern_index in self._pattern_cache:
            contexts = self._pattern_cache[pattern_index][:max_contexts]
            return [
                {
                    "thread_uuid": thread_uuid,
                    "position": position,
                    "surrounding_patterns": self._get_surrounding_patterns(thread_uuid, position),
                }
                for thread_uuid, position in contexts
            ]
        return []

    def _load_thread_patterns(self, thread_uuid: str) -> Optional[Dict]:
        """Load thread metadata and pattern history"""
        if thread_uuid in self._thread_cache:
            self._update_access(thread_uuid)
            return self._thread_cache[thread_uuid]

        # Load from disk
        try:
            # Get thread metadata
            private_dir = Path("memories/private/agents")
            agent_shard = shard_path(private_dir, self.agent_uuid)
            agent_dir = agent_shard / f"agent-{self.agent_uuid}"
            threads_dir = agent_dir / "threads"
            thread_shard = shard_path(threads_dir, thread_uuid)
            meta_path = thread_shard / f"thread-{thread_uuid}.json"

            with open(meta_path, "r") as f:
                meta = json_loads(f.read())

            # Load gene keys
            gene_keys = load_gene_keys(thread_uuid, self.agent_uuid, self.agent_secret)

            # Add gene keys to metadata (in memory only)
            meta["gene_keys"] = gene_keys

            # Cache the data
            self._cache_thread(thread_uuid, meta)
            return meta

        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _cache_thread(self, thread_uuid: str, metadata: Dict):
        """Add thread to cache with LRU eviction"""
        if len(self._thread_cache) >= self.cache_size:
            # Evict least recently used
            lru_uuid = self._access_order.pop(0)
            del self._thread_cache[lru_uuid]

        self._thread_cache[thread_uuid] = metadata
        self._access_order.append(thread_uuid)

        # Update pattern index
        if "gene_keys" in metadata:
            for i, gene_key in enumerate(metadata["gene_keys"]):
                pattern_idx = gene_key["pattern_index"]
                if pattern_idx not in self._pattern_cache:
                    self._pattern_cache[pattern_idx] = []
                self._pattern_cache[pattern_idx].append((thread_uuid, i))

    def _update_access(self, thread_uuid: str):
        """Update LRU order"""
        if thread_uuid in self._access_order:
            self._access_order.remove(thread_uuid)
        self._access_order.append(thread_uuid)

    def _get_surrounding_patterns(self, thread_uuid: str, position: int, window: int = 5) -> List[int]:
        """Get patterns surrounding a specific position in a thread"""
        if thread_uuid in self._thread_cache:
            metadata = self._thread_cache[thread_uuid]
            gene_keys = metadata.get("gene_keys", [])

            if gene_keys:
                start = max(0, position - window)
                end = min(len(gene_keys), position + window + 1)
                return [gene_keys[i]["pattern_index"] for i in range(start, end)]

        return []


class PatternIndex:
    """
    Index patterns to thread locations for O(1) lookup during inference.
    Makes the "passive" thread storage "active" by enabling fast pattern retrieval.
    """

    def __init__(self, agent_uuid: str, agent_secret: str):
        self.agent_uuid = agent_uuid
        self.agent_secret = agent_secret
        self.pattern_locations = {}  # pattern_index -> [(thread_uuid, offset, cycle), ...]
        self.pattern_sequences = {}  # (pattern_a, pattern_b) -> frequency
        self.pattern_contexts = {}  # pattern_index -> {before: Counter, after: Counter}
        self._thread_gene_keys_cache = {}  # thread_uuid -> gene_keys

    def update_from_thread(self, thread_uuid: str, gene_keys: List[Dict]):
        """Update index when a thread is saved"""
        # Cache gene keys to avoid repeated loading
        self._thread_gene_keys_cache[thread_uuid] = gene_keys

        for i, gene_key in enumerate(gene_keys):
            pattern_idx = gene_key["pattern_index"]
            cycle = gene_key["cycle"]

            # Record location
            if pattern_idx not in self.pattern_locations:
                self.pattern_locations[pattern_idx] = []
            self.pattern_locations[pattern_idx].append((thread_uuid, i, cycle))

            # Record sequences
            if i > 0:
                prev_pattern = gene_keys[i - 1]["pattern_index"]
                seq_key = (prev_pattern, pattern_idx)
                self.pattern_sequences[seq_key] = self.pattern_sequences.get(seq_key, 0) + 1

                # Update contexts
                if pattern_idx not in self.pattern_contexts:
                    self.pattern_contexts[pattern_idx] = {"before": {}, "after": {}}

                before_count = self.pattern_contexts[pattern_idx]["before"]
                before_count[prev_pattern] = before_count.get(prev_pattern, 0) + 1

            if i < len(gene_keys) - 1:
                next_pattern = gene_keys[i + 1]["pattern_index"]
                if pattern_idx not in self.pattern_contexts:
                    self.pattern_contexts[pattern_idx] = {"before": {}, "after": {}}
                after_count = self.pattern_contexts[pattern_idx]["after"]
                after_count[next_pattern] = after_count.get(next_pattern, 0) + 1

    def get_likely_next_patterns(self, current_pattern: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get the most likely next patterns based on historical sequences"""
        if current_pattern not in self.pattern_contexts:
            return []

        after_patterns = self.pattern_contexts[current_pattern]["after"]
        total = sum(after_patterns.values())

        if total == 0:
            return []

        # Calculate probabilities
        probabilities = [(pattern, count / total) for pattern, count in after_patterns.items()]

        # Sort by probability and return top k
        probabilities.sort(key=lambda x: x[1], reverse=True)
        return probabilities[:top_k]

    def find_similar_contexts(self, recent_patterns: List[int], k: int = 5) -> List[Dict]:
        """
        Find historical contexts where a sequence of recent patterns appeared.
        If the most recent pattern is very common, sample or cap the number of locations checked for performance.
        """
        if not recent_patterns:
            return []
        most_recent = recent_patterns[-1]
        locations = self.pattern_locations.get(most_recent, [])
        # Cap the number of locations checked for performance
        MAX_LOCATIONS = 100
        if len(locations) > MAX_LOCATIONS:
            import random

            locations = random.sample(locations, MAX_LOCATIONS)
        results = []
        for thread_uuid, offset, cycle in locations:
            # Calculate similarity score based on preceding patterns
            score = self._calculate_context_similarity(thread_uuid, offset, recent_patterns)
            if score > 0:
                results.append({"thread_uuid": thread_uuid, "offset": offset, "cycle": cycle, "similarity": score})

        # Sort by similarity and return top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def _calculate_context_similarity(self, thread_uuid: str, offset: int, recent_patterns: List[int]) -> float:
        """Calculate similarity between recent patterns and historical context"""
        # Get gene keys for this thread (from cache if available)
        if thread_uuid in self._thread_gene_keys_cache:
            gene_keys = self._thread_gene_keys_cache[thread_uuid]
        else:
            gene_keys = load_gene_keys(thread_uuid, self.agent_uuid, self.agent_secret)
            self._thread_gene_keys_cache[thread_uuid] = gene_keys

        # Check surrounding patterns
        context_length = min(len(recent_patterns), 10)  # Use up to 10 recent patterns

        if offset < context_length:
            return 0  # Not enough preceding context

        # Get preceding patterns from this thread
        historical_patterns = [gene_keys[i]["pattern_index"] for i in range(offset - context_length, offset)]

        # Count matching patterns
        matches = sum(1 for i in range(context_length) if historical_patterns[i] == recent_patterns[i])

        return matches / context_length


class InformationEngine:
    """
    Information Engine for processing streams of data

    Manages stream processing, tracking position, and buffering output.
    This engine doesn't make intelligent decisions - it merely processes
    streams according to instructions from the Intelligence layer.
    """

    def __init__(self):
        """Initialize the Information Engine"""
        # Current position in active thread
        self.stream_pointer = 0

        # Accumulator for generated bytes
        self.output_buffer = bytearray()

    def process_stream(
        self,
        inference_engine: InferenceEngine,
        update_callback: Callable[[int, int, float, str], None],
        input_stream: bytes,
    ) -> Tuple[bytes, bytes]:
        """
        Process an entire stream of input bytes

        Args:
            inference_engine: The InferenceEngine to use for processing
            update_callback: Function to call for each processed byte (from IntelligenceEngine)
            input_stream: Bytes to process

        Returns:
            Tuple containing:
            - intermediate_ciphertext: Encrypted form of input
            - dynamic_keystream: Generated keystream
        """
        intermediate_ciphertext = bytearray()
        dynamic_keystream = bytearray()
        self.stream_pointer = 0
        for P_n in input_stream:
            key_index, resonance = inference_engine.process_byte(P_n)
            update_callback(P_n, key_index, resonance, "INPUT")
            keystream_byte = inference_engine.G[key_index]
            C_n = P_n ^ keystream_byte
            intermediate_ciphertext.append(C_n)
            dynamic_keystream.append(keystream_byte)
            self.stream_pointer += 1
        return bytes(intermediate_ciphertext), bytes(dynamic_keystream)

    def process_generated_bytes(
        self,
        inference_engine: InferenceEngine,
        update_callback: Callable[[int, int, float, str], None],
        bytes_to_process: bytes,
    ) -> None:
        """
        Process a sequence of generated bytes (from Intelligence layer)

        This method is used to feed back output bytes into the system,
        ensuring they affect the Epigenome state just like input bytes would.

        Args:
            inference_engine: The InferenceEngine to use
            update_callback: Function to call for each processed byte
            bytes_to_process: Sequence of bytes to process
        """
        for byte in bytes_to_process:
            key_index, resonance = inference_engine.process_byte(byte)
            update_callback(byte, key_index, resonance, "OUTPUT")
            self.stream_pointer += 1
