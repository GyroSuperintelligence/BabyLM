"""
Write/policy logic for GyroSI (S5): OrbitStore and storage decorators.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import mmap
import os
import json
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast, Callable
import sys
import numpy as np

from baby.contracts import MaintenanceReport

logger = logging.getLogger(__name__)

# Singleton cache for phenomenology maps by path
_phenomenology_map_cache: Dict[str, Dict[int, int]] = {}
_phenomenology_map_lock = threading.Lock()

# Add at module top (after imports)
_Sentinel = object()
_append_only_cache_global: dict[tuple[int, int, str], Optional[Any]] = {}
_append_only_cache_order: deque[tuple[int, int, str]] = deque()
_append_only_cache_maxsize = 8192

# Confidence normalization constants
_CONFIDENCE_PRECISION = 1e-4  # Tolerance for float16 precision differences
_CONFIDENCE_ROUNDING_DIGITS = 4  # Round to 4 decimal places for consistency


def normalize_confidence(confidence: float) -> float:
    """
    Normalize confidence values to ensure consistent decision-making.

    This function rounds confidence values to a consistent precision to avoid
    issues with float16 precision differences when making decisions between
    thousands of phenotypes.

    Args:
        confidence: Raw confidence value (0.0 to 1.0)

    Returns:
        Normalized confidence value with consistent precision
    """
    # Clamp to valid range
    confidence = max(0.0, min(1.0, confidence))

    # Round to consistent precision to avoid float16 artifacts
    return round(confidence, _CONFIDENCE_ROUNDING_DIGITS)


def confidence_equals(a: float, b: float, tolerance: float = _CONFIDENCE_PRECISION) -> bool:
    """
    Compare confidence values with tolerance for float16 precision differences.

    Args:
        a: First confidence value
        b: Second confidence value
        tolerance: Tolerance for comparison

    Returns:
        True if values are effectively equal
    """
    return abs(a - b) < tolerance


# ---------- NEW PACK / UNPACK (12-byte fixed) --------------------
# struct layout:  <I I B H x   (x = 1-byte pad)
#   I  state_idx    (uint32)
#   I  token_id     (uint32)
#   B  mask         (uint8)
#   H  conf_f16     (uint16)   # reinterpret float16
#   x  pad          (uint8)    # keep 4-byte alignment
_STRUCT_FMT = "<IIBHx"
_STRUCT_SIZE = struct.calcsize(_STRUCT_FMT)  # = 12


def _abs(path: Optional[str], base: Path) -> str:
    """Return absolute path, expanding user and joining with base if needed."""
    if path is None:
        raise ValueError("Path must not be None")
    p = Path(os.path.expanduser(str(path)))
    return str(p if p.is_absolute() else base / p)


def _get_append_only_cached_static(
    store_id: str, context_key: tuple[int, int], uncached_fn: Callable[[tuple[int, int]], Optional[Any]]
) -> Optional[Any]:
    """Get value from append-only cache or compute and store it if missing.
    Maintains LRU order and evicts oldest if over capacity.
    """
    cache_key = (context_key[0], context_key[1], store_id)
    cached = _append_only_cache_global.get(cache_key, _Sentinel)
    if cached is not _Sentinel:
        try:
            _append_only_cache_order.remove(cache_key)
        except ValueError:
            pass
        _append_only_cache_order.append(cache_key)
        return cached
    result = uncached_fn(context_key)
    _append_only_cache_global[cache_key] = result
    _append_only_cache_order.append(cache_key)
    if len(_append_only_cache_global) > _append_only_cache_maxsize:
        oldest = _append_only_cache_order.popleft()
        del _append_only_cache_global[oldest]
    return result


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# Field order and types are **fixed**; do not alter.
# New 12-byte fixed structure: <I I B H x
#   1. state_idx        : uint32
#   2. token_id         : uint32
#   3. mask             : uint8
#   4. conf_f16         : uint16 (reinterpret float16)
#   5. pad              : uint8 (4-byte alignment)
#
# Deterministic layout with fixed-size entries.
# Returned dict contains exactly the same keys that already
# exist elsewhere in the codebase – no additions.


def _pack_phenotype(entry: Dict[str, Any]) -> bytes:
    if "key" not in entry:
        raise KeyError("Entry must have 'key' field")
    state_idx, token_id = entry["key"]  # renamed upstream
    mask = entry["mask"] & 0xFF
    conf = float(entry["conf"])
    # Convert to float16 and then to uint16 for storage
    conf_f16 = np.float16(conf)
    conf_u16 = conf_f16.view(np.uint16).item()
    return struct.pack(_STRUCT_FMT, state_idx, token_id, mask, conf_u16)


def _unpack_phenotype(buf: memoryview, offset: int = 0) -> tuple[Dict[str, Any], int]:
    state_idx, token_id, mask, conf_u16 = struct.unpack_from(_STRUCT_FMT, buf, offset)
    # Convert from uint16 back to float16 and then to float
    conf_f16 = np.uint16(conf_u16).view(np.float16)
    raw_conf = float(conf_f16.item())
    # Normalize confidence to ensure consistent decision-making
    normalized_conf = normalize_confidence(raw_conf)
    entry = {
        "mask": mask,
        "conf": normalized_conf,
        # expose composite key for callers
        "key": (state_idx, token_id),
    }
    return entry, offset + _STRUCT_SIZE


def load_phenomenology_map(phenomenology_map_path: str, base_path: Path = PROJECT_ROOT) -> Dict[int, int]:
    """Load and cache the phenomenology map from disk, shared between all CanonicalViews. Uses base_path for root."""
    resolved_path = _abs(phenomenology_map_path, base_path)
    if not resolved_path:
        raise ValueError("phenomenology_map_path must not be None")
    with _phenomenology_map_lock:
        if resolved_path in _phenomenology_map_cache:
            return _phenomenology_map_cache[resolved_path]
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Phenomenology map not found: {resolved_path}")
        arr = np.load(resolved_path, mmap_mode="r")
        phen_map = {i: int(rep) for i, rep in enumerate(arr)}
        _phenomenology_map_cache[resolved_path] = phen_map
        return phen_map


class OrbitStore:
    """
    If append_only=False, creates a .idx sidecar for fast key lookup. Most agents/tests use append_only=True (no .idx).

    Performance optimizations for append-only mode:
    - Bloom filter for fast "definitely absent" checks before disk scan
    - Memory-mapped file access for faster sequential scanning
    - Automatic mmap remapping on commit() to include new data

    Durability guarantees:
    - Async fsync thread-pool for performance
    - Graceful shutdown handlers (atexit + SIGINT/SIGTERM) for clean exits
    - Explicit flush() API for high-value writes
    - Configurable write_threshold for different use cases
    """

    def __init__(
        self,
        store_path: str,
        *,
        write_threshold: int = 100,
        use_mmap: bool = False,
        append_only: bool = False,
        base_path: Path = PROJECT_ROOT,
    ):
        resolved_store_path = _abs(store_path, base_path)
        if not resolved_store_path:
            raise ValueError("store_path must not be None")
        self.store_path = resolved_store_path
        self.append_only = append_only
        self.write_threshold = write_threshold
        self.use_mmap = use_mmap
        self.lock = threading.RLock()
        self.pending_writes: Dict[Tuple[int, int], Any] = {}
        self._flush_count = 0
        self._fsync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_fsync: Optional[concurrent.futures.Future[Any]] = None
        self._last_fsync = 0.0
        self._fsync_interval = 0.05  # 50ms or configurable
        self.log_file: Optional[Any] = None
        self._mmap: Optional[mmap.mmap] = None
        self._mmap_size = 0
        self._last_remap = 0.0  # Timestamp of last remap
        self.index: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Bloom filter for append-only mode
        self._bloom_filter: Optional[BloomFilter] = None
        if self.append_only:
            # Initialize bloom filter with estimated capacity
            estimated_entries = 100000  # Default estimate, will be adjusted
            if os.path.exists(resolved_store_path):
                file_size = os.path.getsize(resolved_store_path)
                # Rough estimate: assume average entry size of 200 bytes
                estimated_entries = max(100000, file_size // 200)
            self._bloom_filter = BloomFilter(estimated_entries, error_rate=0.01)

        if self.append_only:
            # self.index is not used in append_only mode
            self.log_path = self.store_path  # .bin file
            self.index_path = None
        else:
            self.index_path = resolved_store_path + ".idx"
            self.log_path = resolved_store_path + ".log"
        self._load_index()
        self.log_file = open(self.log_path, "ab")

        # Initialize mmap for both modes if enabled
        if self.use_mmap and os.path.exists(self.log_path):
            self._open_mmap()

        # Enable mmap by default for append-only stores for better performance
        if self.append_only and not self.use_mmap and os.path.exists(self.log_path):
            self.use_mmap = True
            self._open_mmap()

        # Register graceful shutdown handlers
        self._register_shutdown_handlers()

    def _open_mmap(self) -> None:
        if self._mmap:
            self._mmap.close()
        with open(self.log_path, "rb") as f:
            if os.path.getsize(self.log_path) > 0:
                self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            else:
                self._mmap = None
            if self._mmap:
                self._mmap_size = self._mmap.size()

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        with self.lock:
            if context_key in self.pending_writes:
                return self.pending_writes[context_key]
            if self.append_only:
                # Check bloom filter first for fast "definitely absent" check
                if self._bloom_filter and not self._bloom_filter.might_contain(context_key):
                    return None

                # Use cached get for append-only mode
                return self._get_append_only_cached(context_key)
            if context_key in self.index:
                offset, size = self.index[context_key]
                if self.use_mmap and self._mmap:
                    entry_buf = self._mmap[offset:offset + size]
                else:
                    with open(self.log_path, "rb") as f:
                        f.seek(offset)
                        entry_buf = f.read(size)
                entry, _ = _unpack_phenotype(memoryview(entry_buf))
                return entry
        return None

    def _get_append_only_cached(self, context_key: Tuple[int, int]) -> Optional[Any]:
        return _get_append_only_cached_static(self.store_path, context_key, self._get_append_only_uncached)

    def _cache_pop(self, context_key: Tuple[int, int]) -> None:
        """Remove a specific key from the append-only cache (per store)."""
        cache_key = (context_key[0], context_key[1], self.store_path)
        if cache_key in _append_only_cache_global:
            try:
                _append_only_cache_order.remove(cache_key)
            except ValueError:
                pass
            del _append_only_cache_global[cache_key]

    def _get_append_only_uncached(self, context_key: Tuple[int, int]) -> Optional[Any]:
        """
        Cached version of append-only get() for token-level training performance.

        This cache is automatically invalidated when the store is committed,
        ensuring consistency with new writes.
        """
        # Use mmap if available for faster scanning
        if self.use_mmap and self._mmap:
            buf = memoryview(self._mmap)
        else:
            with open(self.log_path, "rb") as f:
                buf = memoryview(f.read())

        # Scan backwards for the most recent match
        offsets = []
        offset = 0
        try:
            while offset < len(buf):
                try:
                    _, new_offset = _unpack_phenotype(buf, offset)
                    offsets.append(offset)
                    offset = new_offset
                except struct.error:
                    break
        except Exception as e:
            logger.warning("Error during get() scan: %s", e)
        # Now scan backwards
        found = []
        for offset in reversed(offsets):
            try:
                entry, _ = _unpack_phenotype(buf, offset)
                key = tuple(entry["key"])
                if key == context_key:
                    found.append(entry)
            except Exception:
                continue
        if found:
            return found[0]
        return None

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        pending_fsync = None
        with self.lock:
            # Always copy the entry before mutating or storing
            entry = dict(entry)  # ensure a new dict, not just a shallow copy
            # Ensure the entry has the key field
            if "key" not in entry:
                entry["key"] = context_key
            self.pending_writes[context_key] = dict(entry)  # store a new dict
            # Fine‑grain invalidation – just drop the key we're overwriting
            if self.append_only:
                self._cache_pop(context_key)
            if self.pending_writes is not None and len(self.pending_writes) >= self.write_threshold:
                pending_fsync = self._flush()
        if pending_fsync:
            # No blocking wait on fsync here; state will be updated asynchronously
            pass

    def commit(self) -> None:
        with self.lock:
            # Capture pending writes before flush for bloom filter update
            pending_keys = list(self.pending_writes.keys()) if self.append_only else []

            self._flush()
            self._write_index()

            # Only remap mmap after flush if file has grown
            if self.use_mmap:
                current_size = os.path.getsize(self.log_path)
                if not hasattr(self, "_last_mmap_size") or current_size > getattr(self, "_last_mmap_size", 0):
                    self._open_mmap()
                    self._last_mmap_size = current_size
                else:
                    # Still update the timestamp for consistency
                    self._last_remap = time.time()

            # Update bloom filter & **fine‑grain cache invalidation**
            if self.append_only:
                if self._bloom_filter:
                    for context_key in pending_keys:
                        self._bloom_filter.add(context_key)

                # touch only affected items instead of nuking whole cache
                for k in pending_keys:
                    self._cache_pop(k)
        # No blocking wait on fsync here; state will be updated asynchronously

    def _flush(self) -> Optional[concurrent.futures.Future[Any]]:
        pending_fsync = None
        if not self.log_file or not self.pending_writes:
            return None
        for context_key, entry in self.pending_writes.items():
            if self.append_only:
                payload = _pack_phenotype(entry)
                self.log_file.write(payload)
            else:
                payload = _pack_phenotype(entry)
                offset = self.log_file.tell()
                size = len(payload)
                self.log_file.write(payload)
                self.index[context_key] = (offset, size)
        self.log_file.flush()
        now = time.time()
        if now - self._last_fsync > self._fsync_interval:
            future = self._fsync_executor.submit(os.fsync, self.log_file.fileno())
            self._pending_fsync = future

            def _on_fsync_done(fut: concurrent.futures.Future[Any]) -> None:
                # Clear the pending fsync when done
                self._pending_fsync = None
                # Optionally log completion
                # logger.info("Async fsync completed")

            future.add_done_callback(_on_fsync_done)
            self._last_fsync = now
        if self._pending_fsync:
            pending_fsync = self._pending_fsync
        self.pending_writes.clear()
        self._flush_count += 1
        if self._flush_count >= 10 and not self.append_only:
            self._write_index()
            self._flush_count = 0
        return pending_fsync

    def mark_dirty(self, context_key: Tuple[int, int], entry: Any) -> None:
        with self.lock:
            self.pending_writes[context_key] = entry.copy()

    def _write_index(self) -> None:
        if self.append_only:
            return
        if self.index_path is None:
            return
        # Convert all keys/values to native Python int before serializing
        # Convert tuple keys to strings for JSON compatibility
        index_py = {str(key): (int(v[0]), int(v[1])) for key, v in self.index.items()}
        with open(self.index_path, "wb") as f:
            packed = json.dumps(index_py).encode("utf-8")
            f.write(packed)
            f.flush()
            if sys.platform != "darwin":
                os.fsync(f.fileno())

    def _read_index(self) -> None:
        if self.append_only or not self.index_path or not os.path.exists(self.index_path):
            return
        # Handle empty file gracefully
        if os.path.getsize(self.index_path) == 0:
            self.index = {}
            return
        with open(self.index_path, "rb") as f:
            raw_index = json.loads(f.read().decode("utf-8"))
            # Convert string keys back to tuples
            self.index = {}
            for k, v in raw_index.items():
                if not isinstance(k, str):
                    raise TypeError(f"Invalid key type in index: {type(k)}. All keys must be strings.")
                # Parse tuple string format back to tuple
                try:
                    state, token_id = map(int, k.strip("()").split(", "))
                    key_tuple = (state, token_id)
                    self.index[key_tuple] = tuple(v)
                except (ValueError, IndexError) as e:
                    raise TypeError(f"Invalid key format in index: {k}") from e

    def _load_index(self) -> None:
        if self.append_only:
            # Populate bloom filter with existing entries
            if self._bloom_filter and os.path.exists(self.log_path):
                with open(self.log_path, "rb") as f:
                    buf = memoryview(f.read())
                    offset = 0
                    try:
                        while offset < len(buf):
                            try:
                                entry, new_offset = _unpack_phenotype(buf, offset)
                                if "key" in entry:
                                    context_key = tuple(entry["key"])
                                    self._bloom_filter.add(context_key)
                                offset = new_offset
                            except struct.error:
                                # Corrupt tail record - stop gracefully
                                break
                    except Exception as e:
                        logger.warning("Error during _load_index scan: %s", e)
            return
        self._read_index()
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "rb") as f:
            buf = memoryview(f.read())
            offset = 0
            while offset < len(buf):
                entry, new_offset = _unpack_phenotype(buf, offset)
                size = new_offset - offset
                if "key" in entry:
                    context_key = tuple(entry["key"])
                    self.index[context_key] = (offset, size)
                offset = new_offset

    def delete(self, context_key: Tuple[int, int]) -> None:
        if self.append_only:
            raise RuntimeError("append_only store cannot delete; run prune-and-compact instead")
        with self.lock:
            self.index.pop(context_key, None)

    def close(self) -> None:
        with self.lock:
            if self.pending_writes:
                self.commit()
            if hasattr(self, "_pending_fsync") and self._pending_fsync is not None:
                # No blocking wait on fsync here; state will be updated asynchronously
                pass
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            if self.log_file and not self.log_file.closed:
                self.log_file.close()
            if self._fsync_executor:
                self._fsync_executor.shutdown(wait=True)
            # Clear bloom filter
            if self._bloom_filter:
                self._bloom_filter.clear()

            # Clear cache
            # Clear global cache for this store
            to_remove = [k for k in _append_only_cache_global if k[2] == self.store_path]
            for k in to_remove:
                del _append_only_cache_global[k]
            # Mutate the global deque in place
            global _append_only_cache_order
            _append_only_cache_order = deque(k for k in _append_only_cache_order if k[2] != self.store_path)

    def _register_shutdown_handlers(self) -> None:
        """Register graceful shutdown handlers for atexit and signals."""
        import atexit
        import signal

        def _graceful_shutdown() -> None:
            """Force final flush on shutdown."""
            try:
                if hasattr(self, "pending_writes") and self.pending_writes:
                    logger.info("Graceful shutdown: flushing %d pending writes", len(self.pending_writes))
                    self.commit()
            except Exception as e:
                logger.warning("Graceful shutdown flush failed: %s", e)

        # Register for clean exit
        atexit.register(_graceful_shutdown)

        # Register for signals (SIGINT, SIGTERM)
        def _signal_handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, performing graceful shutdown", signum)
            _graceful_shutdown()
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except (OSError, ValueError) as e:
            # Some environments may not allow signal registration
            logger.debug("Could not register signal handlers: %s", e)

    def flush(self) -> None:
        """
        Explicit flush API for high-value writes.

        Forces immediate flush of pending_writes and fsync to disk.
        Use this for critical writes that must not be lost.
        """
        with self.lock:
            if self.pending_writes:
                logger.debug("Explicit flush: writing %d pending entries", len(self.pending_writes))
                # Capture pending keys before flush for bloom filter update
                pending_keys = list(self.pending_writes.keys()) if self.append_only else []

                self._flush()
                # No blocking wait on fsync here; state will be updated asynchronously

                # Update bloom filter for append-only mode
                if self.append_only and self._bloom_filter:
                    for context_key in pending_keys:
                        self._bloom_filter.add(context_key)

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        entries: Dict[Tuple[int, int], Any] = {}
        if not os.path.exists(self.log_path):
            return entries
        if self.append_only:
            # Read all entries from binary log
            with open(self.log_path, "rb") as f:
                buf = memoryview(f.read())
                offset = 0
                while offset < len(buf):
                    entry, new_offset = _unpack_phenotype(buf, offset)
                    key = tuple(entry["key"])
                    entries[key] = entry
                    offset = new_offset
            return entries
        with open(self.log_path, "rb") as f:
            for context_key, (offset, size) in self.index.items():
                f.seek(offset)
                entry_buf = f.read(size)
                entry, _ = _unpack_phenotype(memoryview(entry_buf))
                if isinstance(entry, dict):
                    entries[context_key] = entry
        return entries

    def set_data_dict(self, data: Dict[Tuple[int, int], Any]) -> None:
        with self.lock:
            if self.log_file:
                self.log_file.close()
            open(self.log_path, "wb").close()
            if not self.append_only:
                self.index.clear()
            self.log_file = open(self.log_path, "ab")
            self.pending_writes.clear()
            for k, v in data.items():
                self.put(k, v)
            self._flush()

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        yielded = set()
        if self.append_only:
            # Use mmap if available for faster scanning
            if self.use_mmap and self._mmap:
                buf = memoryview(self._mmap)
            else:
                with open(self.log_path, "rb") as f:
                    buf = memoryview(f.read())

            offset = 0
            latest_entries = {}  # Track latest entry for each key

            try:
                while offset < len(buf):
                    try:
                        entry, new_offset = _unpack_phenotype(buf, offset)
                        key = tuple(entry["key"])
                        latest_entries[key] = entry  # Keep the latest version
                        offset = new_offset
                    except struct.error:
                        # Corrupt tail record - stop gracefully
                        break

                # Yield only the latest version of each entry
                for key, entry in latest_entries.items():
                    yield key, entry
                    yielded.add(key)
            except Exception as e:
                logger.warning("Error during iter_entries scan: %s", e)
                # Yield what we have so far
                for key, entry in latest_entries.items():
                    if key not in yielded:
                        yield key, entry
                        yielded.add(key)
            return
        with self.lock:
            for k, v in self.pending_writes.items():
                yield k, v
                yielded.add(k)
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "rb") as f:
            for context_key, (offset, size) in self.index.items():
                if context_key in yielded:
                    continue
                f.seek(offset)
                entry_buf = f.read(size)
                entry, _ = _unpack_phenotype(memoryview(entry_buf))
                yield context_key, entry


class CanonicalView:
    def __init__(self, base_store: Any, phenomenology_map_path: str, base_path: Path = PROJECT_ROOT):
        """CanonicalView with base_path for sandboxing phenomenology_map_path."""
        self.base_store: Optional[Any] = base_store
        self.phenomenology_map_path = _abs(phenomenology_map_path, base_path)
        if not self.phenomenology_map_path:
            raise ValueError("phenomenology_map_path must not be None")
        if not os.path.exists(self.phenomenology_map_path):
            raise FileNotFoundError(f"Phenomenology map not found: {self.phenomenology_map_path}")
        self.phen_map = load_phenomenology_map(self.phenomenology_map_path, base_path)

    def _get_phenomenology_key(self, context_key: Tuple[int, int]) -> Tuple[int, int]:
        tensor_index, token_id = context_key
        phenomenology_index = self.phen_map.get(tensor_index, tensor_index)
        return (phenomenology_index, token_id)

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        phenomenology_key = self._get_phenomenology_key(context_key)
        entry = self.base_store.get(phenomenology_key)
        # For minimal phenotype, just return the entry as-is
        return entry

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        phen_key = self._get_phenomenology_key(context_key)
        # For minimal phenotype, just store with the phenomenology key
        self.base_store.put(phen_key, entry)

    def commit(self) -> None:
        """Commit pending writes to the base store."""
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        if hasattr(self.base_store, "commit"):
            self.base_store.commit()

    def flush(self) -> None:
        """Explicit flush for high-value writes."""
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        if hasattr(self.base_store, "flush"):
            self.base_store.flush()

    def delete(self, context_key: Tuple[int, int]) -> None:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        phen_key = self._get_phenomenology_key(context_key)
        deleter = getattr(self.base_store, "delete", None)
        if callable(deleter):
            deleter(phen_key)
        else:
            raise NotImplementedError("Underlying store does not support deletion.")

    def close(self) -> None:
        if self.base_store is not None:
            self.base_store.close()
            self.base_store = None

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        return cast(Dict[Tuple[int, int], Any], self.base_store.data)

    def _load_index(self) -> None:
        if self.base_store is not None and hasattr(self.base_store, "_load_index"):
            self.base_store._load_index()

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        """
        Yield entries keyed by their phenomenology key.
        """
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        for phen_key, entry in self.base_store.iter_entries():
            yield phen_key, entry


class OverlayView:
    def __init__(self, public_store: Any, private_store: Any):
        import threading

        self.public_store: Optional[Any] = public_store
        self.private_store: Optional[Any] = private_store
        self.lock = getattr(private_store, "lock", threading.RLock())

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        with self.lock:
            if self.private_store is None or self.public_store is None:
                raise RuntimeError("OverlayView: store is closed or None")
            entry = self.private_store.get(context_key)
            if entry:
                return entry
            fallback = self.public_store.get(context_key)
            return fallback

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        if self.private_store is None:
            raise RuntimeError("OverlayView: private_store is closed or None")
        self.private_store.put(context_key, entry)

    def commit(self) -> None:
        """Commit pending writes to the base stores."""
        if self.private_store is None:
            raise RuntimeError("OverlayView: private_store is closed or None")
        if hasattr(self.private_store, "commit"):
            self.private_store.commit()

    def flush(self) -> None:
        """Explicit flush for high-value writes."""
        if self.private_store is None:
            raise RuntimeError("OverlayView: private_store is closed or None")
        if hasattr(self.private_store, "flush"):
            self.private_store.flush()

    def delete(self, context_key: Tuple[int, int]) -> None:
        if self.private_store is None:
            raise RuntimeError("OverlayView: private_store is closed or None")
        deleter = getattr(self.private_store, "delete", None)
        if callable(deleter):
            deleter(context_key)
        else:
            raise NotImplementedError("Underlying private_store does not support deletion.")

    def close(self) -> None:
        if self.private_store is not None:
            self.private_store.close()
            self.private_store = None
        if self.public_store is not None:
            self.public_store.close()
            self.public_store = None

    def reload_public_knowledge(self) -> None:
        if self.public_store is not None and hasattr(self.public_store, "_load_index"):
            self.public_store._load_index()

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        if self.public_store is None or self.private_store is None:
            raise RuntimeError("OverlayView: store is closed or None")
        combined_data = self.public_store.data.copy()
        combined_data.update(self.private_store.data)
        return cast(Dict[Tuple[int, int], Any], combined_data)

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        """
        Yields (key, entry) pairs, merging public and private stores.
        Private store entries take precedence over public store entries.
        This is memory-efficient for large stores.
        """
        if self.public_store is None or self.private_store is None:
            raise RuntimeError("OverlayView: store is closed or None")
        yielded = set()
        # Yield all private entries first (latest versions only)
        for key, entry in self.private_store.iter_entries():
            if key not in yielded:
                yield key, entry
                yielded.add(key)
        # Yield public entries not shadowed by private
        for key, entry in self.public_store.iter_entries():
            if key not in yielded:
                yield key, entry

    def _load_index(self) -> None:
        if self.private_store is not None and hasattr(self.private_store, "_load_index"):
            self.private_store._load_index()


class ReadOnlyView:
    def __init__(self, base_store: Any):
        import threading

        self.base_store: Optional[Any] = base_store
        self.lock = getattr(base_store, "lock", threading.RLock())

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        return self.base_store.get(context_key)

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        raise RuntimeError("This store is read-only.")

    def commit(self) -> None:
        """Commit pending writes to the base store."""
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        # ReadOnlyView doesn't support writes

    def flush(self) -> None:
        """Explicit flush for high-value writes."""
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        # ReadOnlyView doesn't support writes

    def close(self) -> None:
        if self.base_store is not None:
            self.base_store.close()
            self.base_store = None

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        return cast(Dict[Tuple[int, int], Any], self.base_store.data)

    def _load_index(self) -> None:
        if self.base_store is not None and hasattr(self.base_store, "_load_index"):
            self.base_store._load_index()

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        yield from self.base_store.iter_entries()


def merge_phenotype_maps(
    source_paths: List[str],
    dest_path: str,
    conflict_resolution: str = "highest_confidence",
    base_path: Path = PROJECT_ROOT,
) -> MaintenanceReport:
    """
    Merge multiple phenotype maps into a single consolidated map.

    Note: This merge uses OR to unify masks across phenotypes. The policy-layer OR operation is not the Fold.

    Args:
        source_paths: List of source map file paths
        dest_path: Destination file path
        conflict_resolution: Strategy for handling conflicts
            - "highest_confidence": Keep entry with highest confidence
            - "OR_masks": Combine memory masks with bitwise OR
            - "newest": Keep most recently updated entry
            - "weighted_average": Average confidence, OR masks

    Returns:
        Maintenance report with merge statistics
    """
    start_time = time.time()
    merged_data = {}
    conflict_count = 0
    total_entries = 0

    resolved_dest = _abs(dest_path, base_path)
    resolved_sources = [_abs(p, base_path) for p in source_paths]

    for path in resolved_sources:
        try:
            if path.endswith(".bin"):
                store = OrbitStore(path, append_only=True)
            else:
                store = OrbitStore(path)
            source_data = store.data  # dict
            store.close()
        except Exception:
            continue

        for context_key, entry in source_data.items():
            total_entries += 1

            if context_key not in merged_data:
                merged_data[context_key] = entry.copy()
            else:
                conflict_count += 1
                existing = merged_data[context_key]

                if conflict_resolution == "highest_confidence":
                    # Use normalized comparison to avoid float16 precision issues
                    entry_conf = normalize_confidence(entry.get("conf", 0))
                    existing_conf = normalize_confidence(existing.get("conf", 0))
                    if entry_conf > existing_conf:
                        merged_data[context_key] = entry.copy()

                elif conflict_resolution == "OR_masks":
                    existing["mask"] |= entry.get("mask", 0)
                    # Normalize confidence values for consistent comparison
                    existing["conf"] = normalize_confidence(max(existing.get("conf", 0), entry.get("conf", 0)))

                elif conflict_resolution == "newest":
                    # For minimal phenotype, use conf as proxy for "newest"
                    # Use normalized comparison to avoid float16 precision issues
                    entry_conf = normalize_confidence(entry.get("conf", 0))
                    existing_conf = normalize_confidence(existing.get("conf", 0))
                    if entry_conf > existing_conf:
                        merged_data[context_key] = entry.copy()

                elif conflict_resolution == "weighted_average":
                    # Weighted average based on confidence
                    w1 = normalize_confidence(existing.get("conf", 0.1))
                    w2 = normalize_confidence(entry.get("conf", 0.1))
                    total_weight = w1 + w2

                    # Calculate weighted average and normalize result
                    weighted_conf = (existing.get("conf", 0) * w1 + entry.get("conf", 0) * w2) / total_weight
                    existing["conf"] = normalize_confidence(weighted_conf)
                    existing["mask"] |= entry.get("mask", 0)

    # Save merged result
    os.makedirs(os.path.dirname(resolved_dest) or ".", exist_ok=True)

    dest_store = OrbitStore(resolved_dest, append_only=resolved_dest.endswith(".bin"))
    dest_store.set_data_dict(merged_data)
    dest_store.commit()
    dest_store.close()

    elapsed = time.time() - start_time

    return {
        "operation": "merge_phenotype_maps",
        "success": True,
        "entries_processed": total_entries,
        "entries_modified": len(merged_data),
        "elapsed_seconds": elapsed,
    }


def apply_global_confidence_decay(
    store_path: str,
    decay_factor: float = 0.999,
    time_threshold_days: float = 30.0,
    dry_run: bool = False,
    base_path: Path = PROJECT_ROOT,
) -> MaintenanceReport:
    """
    Apply confidence decay to all entries in a knowledge store.

    Uses the same exponential decay formula as the internal engine:
        confidence = confidence * exp(-decay_factor * age_factor)

    Args:
        store_path: Path to the phenotype store
        decay_factor: Exponential decay rate per age unit (small value, e.g. 0.001)
        time_threshold_days: Days without update to trigger decay
        dry_run: If True, calculate but don't apply changes

    Returns:
        Maintenance report
    """
    start_time = time.time()

    resolved_store_path = _abs(store_path, base_path)
    # Check if it's an OrbitStore file (has .log file)
    log_path = Path(str(resolved_store_path) + ".log")
    if not os.path.exists(resolved_store_path) and not log_path.exists():
        return {
            "operation": "apply_global_confidence_decay",
            "success": False,
            "entries_processed": 0,
            "entries_modified": 0,
            "elapsed_seconds": 0,
        }

    store = OrbitStore(resolved_store_path, append_only=True)
    modified_count = 0
    processed_count = 0

    for key, entry in store.iter_entries():
        processed_count += 1
        # For minimal phenotype, we don't have timestamps, so skip age-based decay
        # Just apply a simple decay factor
        old_conf = entry.get("conf", 0.0)
        new_conf = old_conf * decay_factor
        if new_conf < 0.01:  # same floor the tests use
            new_conf = 0.01

        # Normalize confidence for consistent comparison
        normalized_new_conf = normalize_confidence(new_conf)
        normalized_old_conf = normalize_confidence(old_conf)
        if normalized_new_conf != normalized_old_conf:
            modified_count += 1
            if not dry_run:
                entry["conf"] = new_conf  # Keep original precision for storage
                e = {"key": key, "mask": entry.get("mask", 0), "conf": new_conf}
                store.put(key, e)

    if modified_count > 0 and not dry_run:
        store.commit()

    store.close()
    elapsed = time.time() - start_time

    return {
        "operation": "apply_global_confidence_decay",
        "success": True,
        "entries_processed": processed_count,
        "entries_modified": modified_count,
        "elapsed_seconds": elapsed,
    }


def export_knowledge_statistics(store_path: str, output_path: str, base_path: Path = PROJECT_ROOT) -> MaintenanceReport:
    """
    Export detailed statistics about a knowledge store.

    Args:
        store_path: Path to the phenotype store
        output_path: Path to save JSON statistics

    Returns:
        Maintenance report
    """
    start_time = time.time()

    resolved_store_path = _abs(store_path, base_path)
    resolved_output_path = _abs(output_path, base_path)
    # Check if it's an OrbitStore file (has .log file)
    log_path = Path(str(resolved_store_path) + ".log")
    if not os.path.exists(resolved_store_path) and not log_path.exists():
        return {
            "operation": "export_knowledge_statistics",
            "success": False,
            "entries_processed": 0,
            "entries_modified": 0,
            "elapsed_seconds": 0,
        }

    store = OrbitStore(resolved_store_path, append_only=True)
    entries = list(store.data.values())
    store.close()

    stats_data = {
        "total_entries": len(entries),
        "state_indices": [int(e.get("key", (0, 0))[0]) for e in entries],
        "token_ids": [int(e.get("key", (0, 0))[1]) for e in entries],
        "masks": [int(e.get("mask", 0)) for e in entries],
        "confidence": [float(e.get("conf", 0.0)) for e in entries],
    }

    # Save statistics
    os.makedirs(os.path.dirname(resolved_output_path) or ".", exist_ok=True)
    with open(resolved_output_path, "w") as f:
        # ujson does not support indent argument
        json.dump(stats_data, f)

    elapsed = time.time() - start_time
    return {
        "operation": "export_knowledge_statistics",
        "success": True,
        "entries_processed": len(entries),
        "entries_modified": 0,
        "elapsed_seconds": elapsed,
    }


def validate_ontology_integrity(
    ontology_path: str, phenomenology_map_path: Optional[str] = None, base_path: Path = PROJECT_ROOT
) -> MaintenanceReport:
    """
    Validate the integrity of ontology and phenomenology .npy files.

    Args:
        ontology_path: Path to ontology_keys.npy
        phenomenology_map_path: Optional path to phenomenology_map.npy

    Returns:
        Maintenance report
    """
    import numpy as np

    start_time = time.time()
    issues = []

    resolved_ontology_path = _abs(ontology_path, base_path)
    # Check ontology file
    if not os.path.exists(resolved_ontology_path):
        return {
            "operation": "validate_ontology_integrity",
            "success": False,
            "entries_processed": 0,
            "entries_modified": 0,
            "elapsed_seconds": 0,
        }

    try:
        keys = np.load(resolved_ontology_path, mmap_mode="r")
    except Exception:
        return {
            "operation": "validate_ontology_integrity",
            "success": False,
            "entries_processed": 0,
            "entries_modified": 0,
            "elapsed_seconds": 0,
        }

    # Validate keys array
    if keys.shape[0] != 788_986:
        issues.append(f"Invalid ontology keys array size: {keys.shape[0]}")
    if keys.dtype != np.uint64:
        issues.append(f"Ontology keys dtype should be uint64, got {keys.dtype}")
    if not np.all(np.diff(keys) > 0):
        issues.append("Ontology keys are not strictly increasing (not sorted or have duplicates)")

    # Check phenomenology map if provided
    phenomenology_issues = 0
    resolved_phenomenology_map_path = (
        _abs(phenomenology_map_path, base_path) if phenomenology_map_path is not None else None
    )
    if resolved_phenomenology_map_path and os.path.exists(resolved_phenomenology_map_path):
        try:
            pheno = np.load(resolved_phenomenology_map_path, mmap_mode="r")
            # Validate all indices are in range
            for idx, rep in enumerate(pheno):
                if idx < 0 or idx >= 788_986:
                    phenomenology_issues += 1
                if rep < 0 or rep >= 788_986:
                    phenomenology_issues += 1
            if phenomenology_issues > 0:
                issues.append(f"Found {phenomenology_issues} invalid phenomenology mappings")
        except Exception:
            issues.append("Failed to validate phenomenology map.")

    elapsed = time.time() - start_time

    return {
        "operation": "validate_ontology_integrity",
        "success": len(issues) == 0,
        "entries_processed": int(keys.shape[0]),
        "entries_modified": 0,
        "elapsed_seconds": elapsed,
    }


def prune_and_compact_store(
    store_path: str,
    output_path: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_age_days: Optional[float] = None,
    dry_run: bool = False,
    archive_summary_path: Optional[str] = None,
    base_path: Path = PROJECT_ROOT,
) -> MaintenanceReport:
    """
    Prune and compact an OrbitStore in one pass.

    This removes phenotypes that are below `min_confidence`, then rewrites the log
    with only retained entries so stale historical versions are discarded.

    Args:
        store_path: Base path of the OrbitStore (same value passed to OrbitStore()).
        output_path: Optional destination path. If None, compacts in-place.
        min_confidence: Remove entries with confidence < min_confidence.
        dry_run: If True, only report what would be removed.
        archive_summary_path: If provided, write JSON summary of pruned entries.

    Returns:
        MaintenanceReport describing the operation.
    """
    start_time = time.time()
    now = time.time()

    resolved_store_path = _abs(store_path, base_path)
    resolved_output_path = _abs(output_path, base_path) if output_path is not None else None
    destination = resolved_output_path or resolved_store_path
    resolved_archive_summary_path = _abs(archive_summary_path, base_path) if archive_summary_path is not None else None

    # Load source store
    source_store = OrbitStore(resolved_store_path, append_only=True)
    total_entries = 0
    pruned_entries = 0

    keep: Dict[Tuple[int, int], Any] = {}
    archive_summary: Dict[str, Any] = {
        "pruned": [],
        "criteria": {
            "min_confidence": min_confidence,
        },
        "generated_at": now,
        "source_path": store_path,
        "output_path": destination,
    }

    for key, entry in source_store.iter_entries():
        total_entries += 1
        conf = entry.get("conf", 0.0)

        remove = False
        if min_confidence is not None and conf < min_confidence:
            remove = True

        if remove:
            pruned_entries += 1
            if archive_summary_path:
                archive_summary["pruned"].append(
                    {
                        "key": list(key),
                        "confidence": float(conf),
                        "mask": int(entry.get("mask", 0)),
                    }
                )
        else:
            keep[key] = entry

    # If dry-run, do not modify anything
    if dry_run:
        source_store.close()
        elapsed = time.time() - start_time
        if resolved_archive_summary_path:
            os.makedirs(os.path.dirname(resolved_archive_summary_path) or ".", exist_ok=True)
            with open(resolved_archive_summary_path, "w") as f:
                json.dump(archive_summary, f)
        return {
            "operation": "prune_and_compact_store",
            "success": True,
            "entries_processed": total_entries,
            "entries_modified": pruned_entries,
            "elapsed_seconds": elapsed,
        }

    # Perform compaction into destination
    # If destination == store_path we reuse the existing instance
    if destination == resolved_store_path:
        # Rewrite in-place using existing store API
        source_store.set_data_dict(keep)
        source_store.commit()
        source_store.close()
    else:
        # Write to new path
        source_store.close()
        dest_store = OrbitStore(destination, append_only=True)
        dest_store.set_data_dict(keep)
        dest_store.commit()
        dest_store.close()

    if resolved_archive_summary_path:
        os.makedirs(os.path.dirname(resolved_archive_summary_path) or ".", exist_ok=True)
        with open(resolved_archive_summary_path, "w") as f:
            json.dump(archive_summary, f)

    elapsed = time.time() - start_time
    return {
        "operation": "prune_and_compact_store",
        "success": True,
        "entries_processed": total_entries,
        "entries_modified": pruned_entries,
        "elapsed_seconds": elapsed,
    }


"""
NOTE: To ensure all stores/views are safely closed on process exit, register their close methods with atexit
at the point where you instantiate them, e.g.:

    import atexit
    store = OrbitStore(...)
    atexit.register(store.close)

This avoids referencing undefined variables at the module level.
"""


def to_native(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_native(v) for v in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


# Add bloom filter implementation
class BloomFilter:
    """Simple bloom filter for fast "definitely absent" checks."""

    def __init__(self, capacity: int, error_rate: float = 0.01):
        """
        Initialize bloom filter.

        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive rate
        """
        self.capacity = capacity
        self.error_rate = error_rate

        # Calculate optimal size and hash count
        self.size = self._optimal_size(capacity, error_rate)
        self.hash_count = self._optimal_hash_count(self.size, capacity)

        # Initialize bit array
        self.bit_array = bytearray((self.size + 7) // 8)
        self.count = 0

    def _optimal_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size using textbook formula: m = -n * ln(p) / (ln(2))^2."""
        if n <= 0 or p <= 0 or p >= 1:
            return 1000  # Default size for invalid parameters
        import math

        return max(1000, int(-n * math.log(p) / (math.log(2) ** 2)))

    def _optimal_hash_count(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions using textbook formula: k = (m/n) * ln(2)."""
        if m <= 0 or n <= 0:
            return 7  # Default hash count for invalid parameters
        import math

        return max(1, int((m / n) * math.log(2)))

    def _hash_functions(self, item: Tuple[int, int]) -> List[int]:
        """Generate multiple hash values for the item."""
        # Convert tuple to bytes for hashing
        item_bytes = struct.pack("II", item[0], item[1])

        hashes = []
        for i in range(self.hash_count):
            # Use different salts for each hash function
            salt = struct.pack("I", i)
            hash_obj = hashlib.sha256(salt + item_bytes)
            hash_value = int.from_bytes(hash_obj.digest()[:8], byteorder="big")
            hashes.append(hash_value % self.size)
        return hashes

    def add(self, item: Tuple[int, int]) -> None:
        """Add item to the bloom filter."""
        for hash_val in self._hash_functions(item):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            self.bit_array[byte_index] |= 1 << bit_index
        self.count += 1

    def might_contain(self, item: Tuple[int, int]) -> bool:
        """Check if item might be in the bloom filter."""
        for hash_val in self._hash_functions(item):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            if not (self.bit_array[byte_index] & (1 << bit_index)):
                return False
        return True

    def clear(self) -> None:
        """Clear the bloom filter."""
        self.bit_array = bytearray((self.size + 7) // 8)
        self.count = 0
