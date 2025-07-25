"""
Write/policy logic for GyroSI (S5): OrbitStore and storage decorators.
"""

import concurrent.futures
import logging
import math
import mmap
import os
import json
import msgpack
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
import sys

from baby.contracts import MaintenanceReport

logger = logging.getLogger(__name__)

# Singleton cache for phenomenology maps by path
_phenomenology_map_cache: Dict[str, Dict[int, int]] = {}
_phenomenology_map_lock = threading.Lock()


def _abs(path: Optional[str], base: Path) -> str:
    if path is None:
        raise ValueError("Path must not be None")
    p = Path(os.path.expanduser(str(path)))
    return str(p if p.is_absolute() else base / p)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        with open(resolved_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                raw_map = loaded
            elif isinstance(loaded, dict) and "phenomenology_map" in loaded:
                raw_map = loaded["phenomenology_map"]
            else:
                raise ValueError("Unrecognized phenomenology map format")
            if isinstance(raw_map, list):
                phen_map = {i: rep for i, rep in enumerate(raw_map)}
            else:
                phen_map = {int(k): int(v) for k, v in raw_map.items()}
            _phenomenology_map_cache[resolved_path] = phen_map
            return phen_map


class OrbitStore:
    """
    Unified OrbitStore: supports both advanced (indexed, mmap, deletion) and append-only (msgpack, .mpk) modes.
    Use append_only=True for append-only msgpack mode (V2 semantics).
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
        self._remap_counter = 0
        self._last_remap_size = 0
        self._remap_interval = 10
        self._remap_min_size_increase = 1024 * 1024
        self.index: Dict[Tuple[int, int], Tuple[int, int]] = {}
        if self.append_only:
            # self.index is not used in append_only mode
            self.log_path = self.store_path  # .mpk file
            self.index_path = None
            self._append_only_unpacker = None
        else:
            self.index_path = resolved_store_path + ".idx"
            self.log_path = resolved_store_path + ".log"
        self._load_index()
        self.log_file = open(self.log_path, "ab")
        if self.use_mmap and not self.append_only and os.path.exists(self.log_path):
            self._open_mmap()

    def _open_mmap(self) -> None:
        if self.append_only:
            return
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
                # Linear scan for append-only mode
                for k, v in self.iter_entries():
                    if k == context_key:
                        return v
                return None
            if context_key in self.index:
                offset, size = self.index[context_key]
                if self.use_mmap and self._mmap:
                    entry_data = self._mmap[offset : offset + size]
                else:
                    with open(self.log_path, "rb") as f:
                        f.seek(offset)
                        entry_data = f.read(size)
                return msgpack.unpackb(entry_data, raw=False)
        return None

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        pending_fsync = None
        with self.lock:
            if "context_signature" not in entry:
                entry = entry.copy()
                entry["context_signature"] = context_key
            self.pending_writes[context_key] = entry.copy()
            if self.pending_writes is not None and len(self.pending_writes) >= self.write_threshold:
                pending_fsync = self._flush()
        if pending_fsync:
            pending_fsync.result()

    def commit(self) -> None:
        with self.lock:
            self._flush()
            self._write_index()
            if self.use_mmap and not self.append_only:
                current_size = os.path.getsize(self.log_path)
                remap_needed = False
                if self._mmap is None:
                    remap_needed = True
                else:
                    self._remap_counter += 1
                    size_increase = current_size - self._last_remap_size
                    if (
                        self._remap_counter >= self._remap_interval
                        or size_increase > self._remap_min_size_increase
                        or current_size != self._mmap_size
                    ):
                        remap_needed = True
                if remap_needed:
                    self._open_mmap()
                    self._remap_counter = 0
                    self._last_remap_size = os.path.getsize(self.log_path)
        if self._pending_fsync:
            self._pending_fsync.result()

    def _flush(self) -> Optional[concurrent.futures.Future[Any]]:
        pending_fsync = None
        if not self.log_file or not self.pending_writes:
            return None
        for context_key, entry in self.pending_writes.items():
            data: bytes = cast(bytes, msgpack.packb(to_native(entry), use_bin_type=True))
            offset = self.log_file.tell()
            size = len(data)
            self.log_file.write(data)
            if not self.append_only:
                self.index[context_key] = (offset, size)
        self.log_file.flush()
        now = time.time()
        if now - self._last_fsync > self._fsync_interval:
            self._pending_fsync = self._fsync_executor.submit(os.fsync, self.log_file.fileno())
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
        index_py = {tuple(int(x) for x in k): (int(v[0]), int(v[1])) for k, v in self.index.items()}
        with open(self.index_path, "wb") as f:
            packed = msgpack.packb(index_py, use_bin_type=True)
            if packed is not None:
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
            raw_index = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
            # Convert any list keys to tuples for compatibility
            self.index = {tuple(k) if isinstance(k, list) else k: v for k, v in raw_index.items()}

    def _load_index(self) -> None:
        if self.append_only:
            return
        self._read_index()
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "rb") as f:
            offset = 0
            unpacker = msgpack.Unpacker(use_list=False, raw=False)
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                unpacker.feed(chunk)
                for entry in unpacker:
                    packed = msgpack.packb(entry, use_bin_type=True)
                    if not isinstance(packed, bytes):
                        continue
                    size = len(packed)
                    if "context_signature" in entry:
                        context_key = entry["context_signature"]
                        self.index[context_key] = (offset, size)
                    offset += size

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
                try:
                    self._pending_fsync.result()
                except Exception as e:
                    logger.warning("Deferred fsync failed: %s", e)
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            if self.log_file and not self.log_file.closed:
                self.log_file.close()
            if self._fsync_executor:
                self._fsync_executor.shutdown(wait=True)

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        entries: Dict[Tuple[int, int], Any] = {}
        if not os.path.exists(self.log_path):
            return entries
        if self.append_only:
            # Read all entries from .mpk
            with open(self.log_path, "rb") as f:
                unpacker = msgpack.Unpacker(f, use_list=False, raw=False)
                for entry in unpacker:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        k, v = entry
                        entries[tuple(k)] = v
            return entries
        with open(self.log_path, "rb") as f:
            for context_key, (offset, size) in self.index.items():
                f.seek(offset)
                entry_data = f.read(size)
                entry = msgpack.unpackb(entry_data, raw=False)
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
            if self._append_only_unpacker is None:
                f = open(self.log_path, "rb")
                self._append_only_unpacker = msgpack.Unpacker(f, use_list=False, raw=False)
            unpacker = self._append_only_unpacker
            if unpacker is not None:
                for entry in unpacker:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        k, v = entry
                        yield tuple(k), v
                        yielded.add(tuple(k))
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
                entry_data = f.read(size)
                entry = msgpack.unpackb(entry_data, raw=False)
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
        tensor_index, intron = context_key
        phenomenology_index = self.phen_map.get(tensor_index, tensor_index)
        return (phenomenology_index, intron)

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        phenomenology_key = self._get_phenomenology_key(context_key)
        entry = self.base_store.get(phenomenology_key)
        if entry and "_original_context" in entry:
            # Strip metadata added by put() method
            clean_entry = entry.copy()
            orig_ctx = clean_entry.pop("_original_context", None)
            if orig_ctx is not None:
                clean_entry["context_signature"] = orig_ctx
            return clean_entry
        return entry

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        phen_key = self._get_phenomenology_key(context_key)
        if entry.get("context_signature") != phen_key:
            e = entry.copy()
            e["_original_context"] = context_key
            e["context_signature"] = phen_key
            entry = e
        self.base_store.put(phen_key, entry)

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
        Yield entries keyed by their original context (if present), ensuring a single logical
        entry per phenotype. Also strip _original_context from the yielded entry.
        """
        if self.base_store is None:
            raise RuntimeError("CanonicalView: base_store is closed or None")
        for phen_key, entry in self.base_store.iter_entries():
            orig = entry.get("_original_context")
            if orig is not None:
                clean = entry.copy()
                clean.pop("_original_context", None)
                clean["context_signature"] = orig
                yield orig, clean
            else:
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
        # Yield all private entries first
        for key, entry in self.private_store.iter_entries():
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
        log_path = Path(str(path) + ".log")
        if log_path.exists() or path.endswith(".mpk"):
            try:
                if path.endswith(".mpk"):
                    store = OrbitStore(path, append_only=True)
                else:
                    store = OrbitStore(path)
                source_data = store.data  # dict
                store.close()
            except Exception:
                continue
        else:
            # Only support OrbitStore; skip unsupported files
            continue

        for context_key, entry in source_data.items():
            total_entries += 1

            if context_key not in merged_data:
                merged_data[context_key] = entry.copy()
            else:
                conflict_count += 1
                existing = merged_data[context_key]

                if conflict_resolution == "highest_confidence":
                    if entry.get("confidence", 0) > existing.get("confidence", 0):
                        merged_data[context_key] = entry.copy()

                elif conflict_resolution == "OR_masks":
                    existing["exon_mask"] |= entry.get("exon_mask", 0)
                    existing["usage_count"] += entry.get("usage_count", 0)
                    existing["confidence"] = max(existing.get("confidence", 0), entry.get("confidence", 0))
                    existing["last_updated"] = max(existing.get("last_updated", 0), entry.get("last_updated", 0))

                elif conflict_resolution == "newest":
                    if entry.get("last_updated", 0) > existing.get("last_updated", 0):
                        merged_data[context_key] = entry.copy()

                elif conflict_resolution == "weighted_average":
                    # Weighted average based on usage count
                    w1 = existing.get("usage_count", 1)
                    w2 = entry.get("usage_count", 1)
                    total_weight = w1 + w2

                    existing["confidence"] = (
                        existing.get("confidence", 0) * w1 + entry.get("confidence", 0) * w2
                    ) / total_weight
                    existing["exon_mask"] |= entry.get("exon_mask", 0)
                    existing["usage_count"] = total_weight
                    existing["last_updated"] = max(existing.get("last_updated", 0), entry.get("last_updated", 0))

    # Save merged result
    os.makedirs(os.path.dirname(resolved_dest) or ".", exist_ok=True)

    dest_store = OrbitStore(resolved_dest, append_only=resolved_dest.endswith(".mpk"))
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
    current_time = time.time()

    for key, entry in store.iter_entries():
        processed_count += 1

        last_updated = entry.get("last_updated", current_time)

        # Calculate aging factors
        time_since_update = current_time - last_updated
        days_since_update = time_since_update / (24 * 3600)

        # Use only the time-based aging factor
        age_factor = days_since_update - time_threshold_days if days_since_update > time_threshold_days else 0

        if age_factor > 0:
            if not dry_run:
                # Apply exponential decay (same as internal engine)
                entry["confidence"] = max(0.01, entry.get("confidence", 0.0) * math.exp(-decay_factor * age_factor))
                # Persist the mutation
                store.put(key, entry)

            modified_count += 1

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
        "confidence": [float(e.get("confidence", 0.0)) for e in entries],
        "memory": [int(e.get("usage_count", 0)) for e in entries],
        "created_at": [float(e.get("created_at", 0.0)) for e in entries],
        "last_updated": [float(e.get("last_updated", 0.0)) for e in entries],
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
    Validate the integrity of ontology data files.

    Args:
        ontology_path: Path to genotype map
        phenomenology_map_path: Optional path to phenomenology map

    Returns:
        Maintenance report
    """
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
        with open(resolved_ontology_path, "r") as f:
            ontology_data = json.load(f)
    except Exception:
        return {
            "operation": "validate_ontology_integrity",
            "success": False,
            "entries_processed": 0,
            "entries_modified": 0,
            "elapsed_seconds": 0,
        }

    # Validate ontology structure
    expected_keys = ["schema_version", "ontology_map", "endogenous_modulus", "ontology_diameter", "total_states"]

    for key in expected_keys:
        if key not in ontology_data:
            issues.append(f"Missing required key: {key}")

    # Validate constants
    if ontology_data.get("endogenous_modulus") != 788_986:
        issues.append(f"Invalid endogenous modulus: {ontology_data.get('endogenous_modulus')}")

    if ontology_data.get("ontology_diameter") != 6:
        issues.append(f"Invalid ontology diameter: {ontology_data.get('ontology_diameter')}")

    ontology_map = ontology_data.get("ontology_map", {})
    if len(ontology_map) != 788_986:
        issues.append(f"Invalid ontology map size: {len(ontology_map)}")

    # Check phenomenology map if provided
    phenomenology_issues = 0
    resolved_phenomenology_map_path = (
        _abs(phenomenology_map_path, base_path) if phenomenology_map_path is not None else None
    )
    if resolved_phenomenology_map_path and os.path.exists(resolved_phenomenology_map_path):
        try:
            with open(resolved_phenomenology_map_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_map = data
            elif isinstance(data, dict) and "phenomenology_map" in data:
                raw_map = data["phenomenology_map"]
            else:
                raise ValueError("Unrecognized phenomenology map format")

            if isinstance(raw_map, list):
                phenomenology_map = {i: rep for i, rep in enumerate(raw_map)}
            else:
                phenomenology_map = {int(k): int(v) for k, v in raw_map.items()}

            # Validate all indices are in range
            for idx, phenomenology_idx in phenomenology_map.items():
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= 788_986:
                    phenomenology_issues += 1
                if phenomenology_idx < 0 or phenomenology_idx >= 788_986:
                    phenomenology_issues += 1

            if phenomenology_issues > 0:
                issues.append(f"Found {phenomenology_issues} invalid phenomenology mappings")

        except Exception as e:
            issues.append(f"Failed to validate phenomenology map: {e}")

    elapsed = time.time() - start_time

    return {
        "operation": "validate_ontology_integrity",
        "success": len(issues) == 0,
        "entries_processed": len(ontology_map),
        "entries_modified": 0,
        "elapsed_seconds": elapsed,
    }


def prune_and_compact_store(
    store_path: str,
    output_path: Optional[str] = None,
    max_age_days: Optional[float] = None,
    min_confidence: Optional[float] = None,
    dry_run: bool = False,
    archive_summary_path: Optional[str] = None,
    base_path: Path = PROJECT_ROOT,
) -> MaintenanceReport:
    """
    Prune and compact an OrbitStore in one pass.

    This removes phenotypes that are older than `max_age_days` and/or below
    `min_confidence`, then rewrites the log with only retained entries so
    stale historical versions are discarded.

    Args:
        store_path: Base path of the OrbitStore (same value passed to OrbitStore()).
        output_path: Optional destination path. If None, compacts in-place.
        max_age_days: Remove entries whose last_updated is older than now - max_age_days.
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
            "max_age_days": max_age_days,
            "min_confidence": min_confidence,
        },
        "generated_at": now,
        "source_path": store_path,
        "output_path": destination,
    }

    # Thresholds
    age_cutoff_ts: Optional[float] = None
    if max_age_days is not None and max_age_days > 0:
        age_cutoff_ts = now - (max_age_days * 24 * 3600)

    for key, entry in source_store.iter_entries():
        total_entries += 1
        conf = entry.get("confidence", 0.0)
        last_updated = entry.get("last_updated", now)

        remove = False
        if min_confidence is not None and conf < min_confidence:
            remove = True
        if age_cutoff_ts is not None and last_updated < age_cutoff_ts:
            remove = True

        if remove:
            pruned_entries += 1
            if archive_summary_path:
                archive_summary["pruned"].append(
                    {
                        "key": list(key),
                        "confidence": float(conf),
                        "last_updated": float(last_updated),
                        "usage_count": int(entry.get("usage_count", 0)),
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
