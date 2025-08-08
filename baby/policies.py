"""
Phenotype storage and views.

Append-only varint layout with in-RAM index rebuilt at startup.
Record format per entry:
  <ULEB128 state_idx> <ULEB128 n_pairs=1> <ULEB128 token_id> <uint8 mask>
"""

from __future__ import annotations

import logging
import mmap
import os
from threading import RLock
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast, Callable, Set
import sys
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Varint state-block format (single pair)
# ------------------------------------------------------------


def _abs(path: Optional[str], base: Path) -> str:
    """Return absolute path, expanding user and joining with base if needed."""
    if path is None:
        raise ValueError("Path must not be None")
    p = Path(os.path.expanduser(str(path)))
    return str(p if p.is_absolute() else base / p)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _encode_uleb128(value: int) -> bytes:
    """Encode unsigned integer as LEB128 bytes."""
    if value < 0:
        raise ValueError("LEB128 encoding requires non-negative integers")

    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value == 0:
            result.append(byte)
            break
        else:
            result.append(byte | 0x80)

    return bytes(result)


def _decode_uleb128(data: bytes, offset: int = 0) -> tuple[int, int]:
    """Decode unsigned LEB128 integer from bytes.

    Returns:
        (value, bytes_consumed)
    """
    result = 0
    shift = 0
    bytes_consumed = 0

    for i in range(offset, len(data)):
        byte = data[i]
        result |= (byte & 0x7F) << shift
        bytes_consumed += 1

        if (byte & 0x80) == 0:
            break

        shift += 7
        if shift >= 32:
            raise ValueError("LEB128 value too large")

    return result, bytes_consumed


def _pack_phenotype(entry: Dict[str, Any]) -> bytes:
    """Pack phenotype entry to varint state-block format."""
    if "key" not in entry:
        raise KeyError("Entry must have 'key' field")
    state_idx, token_id = entry["key"]
    mask = entry["mask"]

    # Encode as varint state-block: [state_idx][n_pairs=1][token_id][mask]
    state_bytes = _encode_uleb128(int(state_idx))
    n_pairs_bytes = _encode_uleb128(1)  # Single pair
    token_bytes = _encode_uleb128(int(token_id))
    mask_bytes = bytes([int(mask) & 0xFF])

    return state_bytes + n_pairs_bytes + token_bytes + mask_bytes


def _unpack_phenotype(buf: memoryview, offset: int = 0) -> tuple[Dict[str, Any], int]:
    """Unpack phenotype entry from varint state-block format.

    Returns:
        (entry, bytes_consumed_from_offset)
    """
    data = bytes(buf[offset:])

    # Decode state_idx
    state_idx, state_bytes = _decode_uleb128(data, 0)

    # Decode n_pairs
    n_pairs, n_bytes = _decode_uleb128(data, state_bytes)

    # Validate n_pairs - currently only support n_pairs=1
    if n_pairs != 1:
        raise ValueError(f"Unsupported n_pairs value: {n_pairs} (only 1 is supported)")

    # Decode token_id
    token_id, token_bytes = _decode_uleb128(data, state_bytes + n_bytes)

    # Decode mask (single byte)
    end_pos = state_bytes + n_bytes + token_bytes
    if end_pos >= len(data):
        raise ValueError("Incomplete phenotype data")
    mask = data[end_pos]

    entry = {"mask": int(mask) & 0xFF, "key": (int(state_idx), int(token_id))}
    total_bytes = state_bytes + n_bytes + token_bytes + 1

    # Return bytes consumed relative to the provided offset
    return entry, total_bytes


def load_phenomenology_map(phenomenology_map_path: str, base_path: Path = PROJECT_ROOT) -> NDArray[Any]:
    """Load and cache the phenomenology map from disk as numpy array for memory efficiency."""
    resolved_path = _abs(phenomenology_map_path, base_path)
    if not resolved_path:
        raise ValueError("phenomenology_map_path must not be None")
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Phenomenology map not found: {resolved_path}")
    arr = cast(NDArray[Any], np.load(resolved_path, mmap_mode="r"))
    return arr


class OrbitStore:
    """
    Append-only binary log with in-RAM index rebuilt at startup. Thread-safe via RLock.
    """

    def __init__(
        self,
        store_path: str,
        *,
        write_threshold: int = 100,
        use_mmap: bool = True,
        preferences: Optional[Dict[str, Any]] = None,
        base_path: Path = PROJECT_ROOT,
    ):
        resolved_store_path = _abs(store_path, base_path)
        if not resolved_store_path:
            raise ValueError("store_path must not be None")
        self.store_path = resolved_store_path
        prefs = preferences or {}
        # Wire from preferences with sane defaults
        self.write_threshold = int(prefs.get("write_batch_size", write_threshold))
        storage_backend = str(prefs.get("storage_backend", "binary_struct"))
        self.use_mmap = bool(prefs.get("use_mmap", (storage_backend == "binary_struct"))) if preferences is not None else bool(use_mmap)
        self.max_size_mb = int(prefs.get("max_file_size_mb", 512))
        self._max_size_bytes = int(self.max_size_mb) * 1024 * 1024
        self.lock = RLock()
        self.pending_writes: Dict[Tuple[int, int], Any] = {}
        self._flush_count = 0

        # In-RAM index from scan
        self.log_file: Optional[Any] = None
        self._mmap: Optional[mmap.mmap] = None
        self._mmap_file: Optional[Any] = None  # File handle for mmap
        self._mmap_size = 0
        self._last_remap = 0.0  # Timestamp of last remap
        self.index: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # O(1) candidate listing for a given state
        self.index_by_state: Dict[int, Set[int]] = {}

        # Single file storage
        self.log_path = self.store_path  # .bin file

        # Build index from one-pass scan at startup
        self._build_index_from_scan()

        # Ensure the directory exists and create the file if it doesn't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_file = open(self.log_path, "ab")

        # Initialize mmap
        if self.use_mmap and os.path.exists(self.log_path):
            self._open_mmap()

        # Register graceful shutdown handlers
        self._register_shutdown_handlers()

    # ---------- One-pass index scan ----------
    def _build_index_from_scan(self) -> None:
        """Build in-RAM index from one-pass scan of the data file."""
        if not os.path.exists(self.log_path):
            return  # No data file to scan

        try:
            with open(self.log_path, "rb") as f:
                data = f.read()

            offset = 0
            while offset < len(data):
                try:
                    # Parse varint state-block format (single pair)
                    entry, consumed = _unpack_phenotype(memoryview(data), offset)
                    key = cast(Tuple[int, int], entry["key"])

                    # Record offset and size for this entry
                    self.index[key] = (offset, consumed)

                    # Update state index
                    state_idx = key[0]
                    self.index_by_state.setdefault(state_idx, set()).add(key[1])

                    offset += consumed

                except (ValueError, IndexError) as e:
                    # Skip corrupted data and stop scan
                    logger.warning("Corrupted data at offset %d: %s", offset, e)
                    break

            logger.info(
                "Built index from scan: %d entries, %d states",
                len(self.index),
                len(self.index_by_state),
            )

        except Exception as e:
            logger.warning("Failed to build index from scan: %s", e)
            # Continue with empty index

    def _open_mmap(self) -> None:
        if self._mmap:
            self._mmap.close()
        # Keep file open for mmap lifetime
        if os.path.getsize(self.log_path) > 0:
            self._mmap_file = open(self.log_path, "rb")
            self._mmap = mmap.mmap(self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
            self._mmap_size = self._mmap.size()
        else:
            self._mmap = None
            self._mmap_file = None

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        with self.lock:
            if context_key in self.pending_writes:
                return self.pending_writes[context_key]

            # Use index for O(1) lookup
            if context_key in self.index:
                offset, size = self.index[context_key]
                if self.use_mmap and self._mmap:
                    entry_buf = self._mmap[offset : offset + size]
                else:
                    with open(self.log_path, "rb") as f:
                        f.seek(offset)
                        entry_buf = f.read(size)

                if len(entry_buf) == 0:
                    return None

                entry, _ = _unpack_phenotype(memoryview(entry_buf), 0)
                return entry

            return None

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        with self.lock:
            entry = dict(entry)
            if "key" not in entry:
                entry["key"] = context_key
            self.pending_writes[context_key] = dict(entry)
            if self.pending_writes is not None and len(self.pending_writes) >= self.write_threshold:
                self._flush()

    def commit(self) -> None:
        with self.lock:
            self._flush()

            if self.use_mmap:
                if os.path.exists(self.log_path):
                    current_size = os.path.getsize(self.log_path)
                    if not hasattr(self, "_last_mmap_size") or current_size > getattr(self, "_last_mmap_size", 0):
                        self._open_mmap()
                        self._last_mmap_size = current_size
                    else:
                        self._last_remap = time.time()
                else:
                    self._last_remap = time.time()

            for k in list(self.pending_writes.keys()):
                pass

    def _flush(self) -> None:
        if not self.log_file or not self.pending_writes:
            return None
        for context_key, entry in self.pending_writes.items():
            payload = _pack_phenotype(entry)
            # Rollover if writing this payload would exceed size cap
            try:
                if self._max_size_bytes > 0:
                    current_offset = self.log_file.tell()
                    if (current_offset + len(payload)) > self._max_size_bytes:
                        self._rollover()
            except Exception:
                # Non-fatal rollover check failure; continue writing
                pass
            offset = self.log_file.tell()
            size = len(payload)
            self.log_file.write(payload)
            self.index[context_key] = (offset, size)
            s_idx, tok_id = context_key
            self.index_by_state.setdefault(s_idx, set()).add(tok_id)
        self.log_file.flush()
        fdatasync_func = getattr(os, "fdatasync", None)
        if fdatasync_func is not None:
            fdatasync_func(self.log_file.fileno())
        else:
            os.fsync(self.log_file.fileno())
        self.pending_writes.clear()
        return None

    def _rollover(self) -> None:
        """Rotate the current log to knowledge_YYYYMMDD.bin and start a new one."""
        if not self.log_file:
            return
        # Close mmap and file handles cleanly
        try:
            self.log_file.flush()
            fdatasync_func = getattr(os, "fdatasync", None)
            if fdatasync_func is not None:
                fdatasync_func(self.log_file.fileno())
            else:
                os.fsync(self.log_file.fileno())
        except Exception:
            pass
        try:
            if self._mmap:
                self._mmap.close()
            if self._mmap_file:
                self._mmap_file.close()
        except Exception:
            pass
        # Compute destination path
        dir_path = os.path.dirname(self.log_path)
        base = os.path.basename(self.log_path)
        date_tag = time.strftime("%Y%m%d")
        if base.startswith("knowledge") and base.endswith(".bin"):
            dst = os.path.join(dir_path, f"knowledge_{date_tag}.bin")
        else:
            root, ext = os.path.splitext(base)
            dst = os.path.join(dir_path, f"{root}_{date_tag}{ext}")
        # Ensure unique destination
        counter = 1
        final_dst = dst
        while os.path.exists(final_dst):
            final_dst = dst.replace(".bin", f"_{counter}.bin") if dst.endswith(".bin") else f"{dst}.{counter}"
            counter += 1
        # Rename current log
        try:
            self.log_file.close()
        except Exception:
            pass
        os.rename(self.log_path, final_dst)
        # Start new empty log
        Path(self.log_path).touch(exist_ok=True)
        self.log_file = open(self.log_path, "ab")
        # Reset in-RAM indices for the fresh file
        self.index.clear()
        self.index_by_state.clear()
        # Reset mmap
        self._mmap = None
        self._mmap_file = None
        self._mmap_size = 0
        # No need to _open_mmap on empty file
        logger.info("Rolled over knowledge log to %s; new log at %s", final_dst, self.log_path)

    def mark_dirty(self, context_key: Tuple[int, int], entry: Any) -> None:
        with self.lock:
            self.pending_writes[context_key] = entry.copy()

    def _load_index(self) -> None:
        """Load index from one-pass scan (replaces persistent index)."""
        self._build_index_from_scan()

    def delete(self, context_key: Tuple[int, int]) -> None:
        with self.lock:
            self.index.pop(context_key, None)
            s_idx, tok_id = context_key
            st = self.index_by_state.get(s_idx)
            if st:
                st.discard(tok_id)

    def close(self) -> None:
        with self.lock:
            if self.pending_writes:
                self.commit()
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            if hasattr(self, "_mmap_file") and self._mmap_file:
                self._mmap_file.close()
                self._mmap_file = None
            if self.log_file and not self.log_file.closed:
                self.log_file.close()

    def _register_shutdown_handlers(self) -> None:
        """Register graceful shutdown handlers for atexit and signals."""
        import atexit
        import signal

        def _graceful_shutdown() -> None:
            try:
                if hasattr(self, "pending_writes") and self.pending_writes:
                    logger.info("Graceful shutdown: flushing %d pending writes", len(self.pending_writes))
                    self.commit()
            except Exception as e:
                logger.warning("Graceful shutdown flush failed: %s", e)

        atexit.register(_graceful_shutdown)

        def _signal_handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, performing graceful shutdown", signum)
            _graceful_shutdown()
            sys.exit(0)

        if not any("pytest" in arg for arg in sys.argv) and not any("uvicorn" in arg for arg in sys.argv):
            try:
                signal.signal(signal.SIGINT, _signal_handler)
                signal.signal(signal.SIGTERM, _signal_handler)
            except (OSError, ValueError) as e:
                logger.debug("Could not register signal handlers: %s", e)

    def flush(self) -> None:
        """
        Explicit flush API for high-value writes.
        """
        with self.lock:
            if self.pending_writes:
                pending_keys = list(self.pending_writes.keys())
                self._flush()
                for k in pending_keys:
                    pass

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        """Get all data as a dictionary."""
        entries: Dict[Tuple[int, int], Any] = {}
        for k, v in self.iter_entries():
            entries[k] = v
        return entries

    def set_data_dict(self, data: Dict[Tuple[int, int], Any]) -> None:
        with self.lock:
            if self.log_file:
                self.log_file.close()
            Path(self.log_path).touch(exist_ok=True)
            self.index.clear()
            self.log_file = open(self.log_path, "ab")
            self.pending_writes.clear()
            for k, v in data.items():
                self.put(k, v)
            self._flush()

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        """Iterate over all entries in the store."""
        with self.lock:
            yielded: set[Tuple[int, int]] = set()

            # 1) yield pending (latest) first
            for k, v in self.pending_writes.items():
                yield k, dict(v)
                yielded.add(k)

            # 2) yield committed last versions via index
            if not os.path.exists(self.log_path):
                return

            if self.use_mmap and self._mmap:
                mv = memoryview(self._mmap)
                for context_key, (offset, size) in self.index.items():
                    if context_key in yielded:
                        continue
                    entry_buf_mv = mv[offset : offset + size]
                    entry, _ = _unpack_phenotype(entry_buf_mv, 0)
                    yield context_key, entry
            else:
                with open(self.log_path, "rb") as f:
                    for context_key, (offset, size) in self.index.items():
                        if context_key in yielded:
                            continue
                        f.seek(offset)
                        entry_buf: bytes = f.read(size)
                        entry, _ = _unpack_phenotype(memoryview(entry_buf), 0)
                        yield context_key, entry

    # Fast helper: iterate keys for a given state using the in-memory index.
    def iter_keys_for_state(self, state_idx: int) -> Iterator[Tuple[int, int]]:
        with self.lock:
            for s, t in self.pending_writes.keys():
                if s == state_idx:
                    yield (s, t)
            for tok_id in self.index_by_state.get(state_idx, ()): 
                yield (state_idx, tok_id)


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
        phen_idx = int(self.phen_map[tensor_index])
        if phen_idx == tensor_index:
            return context_key
        return (phen_idx, token_id)

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        phenomenology_key = self._get_phenomenology_key(context_key)
        entry = self.base_store.get(phenomenology_key)
        return entry

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        phen_key = self._get_phenomenology_key(context_key)
        if hasattr(entry, "copy"):
            entry = entry.copy()
        elif isinstance(entry, dict):
            entry = dict(entry)
        entry["key"] = phen_key
        self.base_store.put(phen_key, entry)

    def commit(self) -> None:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        if hasattr(self.base_store, "commit"):
            self.base_store.commit()

    def flush(self) -> None:
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
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        for phen_key, entry in self.base_store.iter_entries():
            yield phen_key, entry

    def iter_keys_for_state(self, state_idx: int) -> Iterator[Tuple[int, int]]:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        rep = int(self.phen_map[state_idx])
        if hasattr(self.base_store, "iter_keys_for_state"):
            yield from self.base_store.iter_keys_for_state(rep)
            return
        if hasattr(self.base_store, "public_store") and hasattr(self.base_store, "private_store"):
            seen = set()
            if self.base_store.private_store and hasattr(self.base_store.private_store, "iter_keys_for_state"):
                for key in self.base_store.private_store.iter_keys_for_state(rep):
                    if key not in seen:
                        seen.add(key)
                        yield key
            if self.base_store.public_store and hasattr(self.base_store.public_store, "iter_keys_for_state"):
                for key in self.base_store.public_store.iter_keys_for_state(rep):
                    if key not in seen:
                        seen.add(key)
                        yield key
            return
        raise RuntimeError(f"Base store {type(self.base_store)} does not support iter_keys_for_state")


class OverlayView:
    def __init__(self, public_store: Any, private_store: Any):
        self.public_store: Optional[Any] = public_store
        self.private_store: Optional[Any] = private_store
        self.lock = getattr(private_store, "lock", RLock())

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
        if self.private_store is None:
            raise RuntimeError("OverlayView: private_store is closed or None")
        if hasattr(self.private_store, "commit"):
            self.private_store.commit()

    def flush(self) -> None:
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

    def close_public(self) -> None:
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
        combined = self.public_store.data.copy()
        combined.update(self.private_store.data)
        return cast(Dict[Tuple[int, int], Any], combined)

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        if self.public_store is None or self.private_store is None:
            raise RuntimeError("OverlayView: store is closed or None")
        yielded = set()
        for key, entry in self.private_store.iter_entries():
            if key not in yielded:
                yield key, entry
                yielded.add(key)
        for key, entry in self.public_store.iter_entries():
            if key not in yielded:
                yield key, entry

    def _load_index(self) -> None:
        if self.private_store is not None and hasattr(self.private_store, "_load_index"):
            self.private_store._load_index()

    def iter_keys_for_state(self, state_idx: int) -> Iterator[Tuple[int, int]]:
        if self.private_store is None or self.public_store is None:
            raise RuntimeError("OverlayView: store is closed or None")
        seen: set[Tuple[int, int]] = set()
        it_priv: Optional[Callable[[int], Iterator[Tuple[int, int]]]] = getattr(
            self.private_store, "iter_keys_for_state", None
        )
        if callable(it_priv):
            for k in it_priv(state_idx):
                seen.add(k)
                yield k
        it_pub: Optional[Callable[[int], Iterator[Tuple[int, int]]]] = getattr(
            self.public_store, "iter_keys_for_state", None
        )
        if callable(it_pub):
            for k in it_pub(state_idx):
                if k not in seen:
                    yield k


class ReadOnlyView:
    def __init__(self, base_store: Any):
        self.base_store: Optional[Any] = base_store
        self.lock = getattr(base_store, "lock", RLock())

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        return self.base_store.get(context_key)

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        raise RuntimeError("This store is read-only.")

    def commit(self) -> None:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")

    def flush(self) -> None:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")

    def close(self) -> None:
        if self.base_store is not None:
            self.base_store.close()
            self.base_store = None

    @property
    def data(self) -> Dict[Tuple[Tuple[int, int], int], Any]:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        return cast(Dict[Tuple[Tuple[int, int], int], Any], self.base_store.data)

    def _load_index(self) -> None:
        if self.base_store is not None and hasattr(self.base_store, "_load_index"):
            self.base_store._load_index()

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        yield from self.base_store.iter_entries()

    def iter_keys_for_state(self, state_idx: int) -> Iterator[Tuple[int, int]]:
        if self.base_store is None:
            raise RuntimeError("ReadOnlyView: base_store is closed or None")
        it: Optional[Callable[[int], Iterator[Tuple[int, int]]]] = getattr(self.base_store, "iter_keys_for_state", None)
        if callable(it):
            yield from it(state_idx)
            return
        raise RuntimeError(f"Base store {type(self.base_store)} does not support iter_keys_for_state")


class MultiKnowledgeView:
    """View that can read from multiple knowledge files."""

    def __init__(self, knowledge_files: List[str], base_path: Path = PROJECT_ROOT):
        import threading

        self.knowledge_files = knowledge_files
        self.base_path = base_path
        self.stores: List[Any] = []
        self.lock = threading.RLock()

        for file_path in knowledge_files:
            resolved_path = _abs(file_path, base_path)
            if os.path.exists(resolved_path):
                try:
                    store = OrbitStore(resolved_path, use_mmap=True)
                    self.stores.append(store)
                    logger.info(
                        "Loaded knowledge file: %s (%d entries)", os.path.basename(file_path), len(store.data)
                    )
                except Exception as e:
                    logger.warning("Failed to load %s: %s", file_path, e)
            else:
                logger.warning("Knowledge file not found: %s", resolved_path)

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        with self.lock:
            for store in self.stores:
                entry = store.get(context_key)
                if entry:
                    return entry
            return None

    def iter_entries(self) -> Iterator[Tuple[Tuple[int, int], Any]]:
        with self.lock:
            yielded = set()
            for store in self.stores:
                for key, entry in store.iter_entries():
                    if key not in yielded:
                        yield key, entry
                        yielded.add(key)

    def iter_keys_for_state(self, state_idx: int) -> Iterator[Tuple[int, int]]:
        with self.lock:
            seen = set()
            for store in self.stores:
                for key in store.iter_keys_for_state(state_idx):
                    if key not in seen:
                        yield key
                        seen.add(key)

    def close(self) -> None:
        with self.lock:
            for store in self.stores:
                store.close()
            self.stores.clear()

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        with self.lock:
            combined = {}
            for store in self.stores:
                combined.update(store.data)
            return combined


def create_multi_knowledge_view(
    knowledge_dir: str,
    pattern: str = "knowledge_*.bin",
    base_path: Path = PROJECT_ROOT,
) -> MultiKnowledgeView:
    """Create a MultiKnowledgeView from all knowledge files matching a pattern."""
    import glob

    resolved_dir = _abs(knowledge_dir, base_path)
    if not os.path.exists(resolved_dir):
        logger.warning("Knowledge directory not found: %s", resolved_dir)
        return MultiKnowledgeView([], base_path)

    search_pattern = os.path.join(resolved_dir, pattern)
    knowledge_files = glob.glob(search_pattern)

    if not knowledge_files:
        return MultiKnowledgeView([], base_path)

    knowledge_files.sort()

    logger.info(
        "Found %d knowledge files: %s",
        len(knowledge_files),
        [os.path.basename(f) for f in knowledge_files],
    )

    return MultiKnowledgeView(knowledge_files, base_path)


# Public API surface
PhenotypeStore = OrbitStore

__all__ = [
    "PhenotypeStore",
    "CanonicalView",
    "OverlayView",
    "ReadOnlyView",
    "MultiKnowledgeView",
    "create_multi_knowledge_view",
]
