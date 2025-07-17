"""
Write/policy logic for GyroSI (S5): OrbitStore and storage decorators.
"""

import os
import gzip
import pickle
import json
import time
from typing import List, Optional, Dict, Tuple, Any

from baby.contracts import MaintenanceReport
import threading
import mmap
import concurrent.futures
import atexit


class OrbitStore:
    """S5: Core storage primitive - handles file-based phenotype storage (write/policy)."""

    def __init__(self, store_path: str, write_threshold: int = 100, use_mmap: bool = False):
        self.store_path = store_path
        self.index_path = store_path + ".idx"
        self.log_path = store_path + ".log"
        self.write_threshold = write_threshold
        self.use_mmap = use_mmap
        self.lock = threading.RLock()
        self.index: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.log_file = None
        self.pending_writes: list[tuple[tuple[int, int], Any]] = []
        self._mmap = None
        self._mmap_size = 0
        self._load_index()
        self.log_file = open(self.log_path, "ab")
        if use_mmap and os.path.exists(self.log_path):
            self._open_mmap()
        self._flush_count = 0
        self._fsync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_fsync: Optional[concurrent.futures.Future] = None

    def _open_mmap(self):
        if self._mmap:
            self._mmap.close()
        with open(self.log_path, "rb") as f:
            self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._mmap_size = os.path.getsize(self.log_path)

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        with self.lock:
            if context_key in self.index:
                offset, size = self.index[context_key]
                if self.use_mmap and self._mmap:
                    entry_data = self._mmap[offset : offset + size]
                else:
                    with open(self.log_path, "rb") as f:
                        f.seek(offset)
                        entry_data = f.read(size)
                return pickle.loads(entry_data)
        return None

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        pending_fsync = None
        with self.lock:
            if "context_signature" not in entry:
                entry = entry.copy()
                entry["context_signature"] = context_key
            self.pending_writes.append((context_key, entry.copy()))
            if len(self.pending_writes) >= self.write_threshold:
                pending_fsync = self._flush()
        if pending_fsync:
            pending_fsync.result()

    def commit(self) -> None:
        pending_fsync = None
        with self.lock:
            if self.pending_writes:
                pending_fsync = self._flush()
            with gzip.open(self.index_path, "wb") as f:
                pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.use_mmap:
                current_size = os.path.getsize(self.log_path)
                if self._mmap is None or current_size != self._mmap_size:
                    self._open_mmap()
        if pending_fsync:
            pending_fsync.result()

    def _flush(self):
        pending_fsync = None
        if not self.log_file or not self.pending_writes:
            return None
        for context_key, entry in self.pending_writes:
            data = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
            offset = self.log_file.tell()
            size = len(data)
            self.log_file.write(data)
            self.index[context_key] = (offset, size)
        self.log_file.flush()
        if self._pending_fsync:
            pending_fsync = self._pending_fsync
        self._pending_fsync = self._fsync_executor.submit(os.fsync, self.log_file.fileno())
        self.pending_writes.clear()
        self._flush_count += 1
        if self._flush_count >= 10:
            with gzip.open(self.index_path, "wb") as f:
                pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._flush_count = 0
        return pending_fsync

    def _load_index(self) -> None:
        if os.path.exists(self.index_path):
            with gzip.open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            return
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "rb") as f:
            offset = 0
            while True:
                try:
                    entry = pickle.load(f)
                    size = f.tell() - offset
                    if "context_signature" in entry:
                        context_key = entry["context_signature"]
                        self.index[context_key] = (offset, size)
                    offset = f.tell()
                except EOFError:
                    break

    def close(self) -> None:
        with self.lock:
            if self.pending_writes:
                self.commit()
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            if self.log_file:
                self.log_file.close()
            if self._pending_fsync:
                self._pending_fsync.result()
            self._fsync_executor.shutdown(wait=True)

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        entries = {}
        if not os.path.exists(self.log_path):
            return entries
        with open(self.log_path, "rb") as f:
            while True:
                try:
                    entry = pickle.load(f)
                    if "context_signature" in entry:
                        context_key = entry["context_signature"]
                        entries[context_key] = entry
                except EOFError:
                    break
        return entries

    def set_data_dict(self, data: Dict[Tuple[int, int], Any]) -> None:
        with self.lock:
            if self.log_file:
                self.log_file.close()
            open(self.log_path, "wb").close()
            self.index.clear()
            self.log_file = open(self.log_path, "ab")
            self.pending_writes.clear()
            for k, v in data.items():
                self.put(k, v)
            self._flush()

    def iter_entries(self):
        """
        Yields (key, entry) pairs for only the latest entry for each key, using self.index.
        Mutating the entry requires calling store.put(key, entry) to persist.
        """
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "rb") as f:
            for context_key, (offset, size) in self.index.items():
                f.seek(offset)
                entry = pickle.load(f)
                if "context_signature" in entry:
                    yield context_key, entry


class CanonicalView:
    def __init__(self, base_store: Any, canonical_map_path: str):
        import threading

        self.base_store = base_store
        self.lock = getattr(base_store, "lock", threading.RLock())
        with open(canonical_map_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                self.canonical_map = dict(enumerate(loaded))
            else:
                self.canonical_map = {int(k): v for k, v in loaded.items()}

    def _get_canonical_key(self, context_key: Tuple[int, int]) -> Tuple[int, int]:
        tensor_index, intron = context_key
        canonical_index = self.canonical_map.get(tensor_index, tensor_index)
        return (canonical_index, intron)

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        canonical_key = self._get_canonical_key(context_key)
        return self.base_store.get(canonical_key)

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        canonical_key = self._get_canonical_key(context_key)
        if "context_signature" not in entry:
            entry = entry.copy()
            entry["context_signature"] = context_key
        self.base_store.put(canonical_key, entry)

    def close(self) -> None:
        self.base_store.close()

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        return self.base_store.data

    def _load_index(self) -> None:
        if hasattr(self.base_store, "_load_index"):
            self.base_store._load_index()


class OverlayView:
    def __init__(self, public_store: Any, private_store: Any):
        import threading

        self.public_store = public_store
        self.private_store = private_store
        self.lock = getattr(private_store, "lock", threading.RLock())

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        with self.lock:
            entry = self.private_store.get(context_key)
            if entry:
                return entry
            return self.public_store.get(context_key)

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        self.private_store.put(context_key, entry)

    def close(self) -> None:
        self.private_store.close()
        self.public_store.close()

    def reload_public_knowledge(self) -> None:
        if hasattr(self.public_store, "_load_index"):
            self.public_store._load_index()

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        combined = self.public_store.data.copy()
        combined.update(self.private_store.data)
        return combined

    def _load_index(self) -> None:
        if hasattr(self.private_store, "_load_index"):
            self.private_store._load_index()


class ReadOnlyView:
    def __init__(self, base_store: Any):
        import threading

        self.base_store = base_store
        self.lock = getattr(base_store, "lock", threading.RLock())

    def get(self, context_key: Tuple[int, int]) -> Optional[Any]:
        return self.base_store.get(context_key)

    def put(self, context_key: Tuple[int, int], entry: Any) -> None:
        raise RuntimeError("This store is read-only.")

    def close(self) -> None:
        self.base_store.close()

    @property
    def data(self) -> Dict[Tuple[int, int], Any]:
        return self.base_store.data

    def _load_index(self) -> None:
        if hasattr(self.base_store, "_load_index"):
            self.base_store._load_index()


def merge_phenotype_maps(
    source_paths: List[str], dest_path: str, conflict_resolution: str = "highest_confidence"
) -> MaintenanceReport:
    """
    Merge multiple phenotype maps into a single consolidated map.

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

    for source_path in source_paths:
        if not os.path.exists(source_path):
            print(f"Warning: Source file not found: {source_path}")
            continue

        try:
            with gzip.open(source_path, "rb") as f:
                source_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {source_path}: {e}")
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
                    existing["memory_mask"] |= entry.get("memory_mask", 0)
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
                    existing["memory_mask"] |= entry.get("memory_mask", 0)
                    existing["usage_count"] = total_weight
                    existing["last_updated"] = max(existing.get("last_updated", 0), entry.get("last_updated", 0))

    # Save merged result
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    dest_store = OrbitStore(dest_path)
    dest_store.set_data_dict(merged_data)
    dest_store.commit()
    dest_store.close()

    elapsed = time.time() - start_time

    return MaintenanceReport(
        operation="merge_phenotype_maps",
        success=True,
        entries_processed=total_entries,
        entries_modified=len(merged_data),
        elapsed_seconds=elapsed,
    )


def apply_global_confidence_decay(
    store_path: str,
    decay_factor: float = 0.999,
    age_threshold: int = 100,
    time_threshold_days: float = 30.0,
    dry_run: bool = False,
) -> MaintenanceReport:
    """
    Apply confidence decay to all entries in a knowledge store.

    Args:
        store_path: Path to the phenotype store
        decay_factor: Multiplicative decay factor
        age_threshold: Minimum age counter to trigger decay
        time_threshold_days: Days without update to trigger decay
        dry_run: If True, calculate but don't apply changes

    Returns:
        Maintenance report
    """
    start_time = time.time()

    if not os.path.exists(store_path):
        return MaintenanceReport(
            operation="apply_global_confidence_decay",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
        )

    store = OrbitStore(store_path)
    modified_count = 0
    processed_count = 0
    current_time = time.time()

    for key, entry in store.iter_entries():
        processed_count += 1

        age_counter = entry.get("age_counter", 0)
        last_updated = entry.get("last_updated", current_time)

        # Calculate aging factors
        time_since_update = current_time - last_updated
        days_since_update = time_since_update / (24 * 3600)

        # Use maximum of the two aging factors
        age_factor = max(
            age_counter - age_threshold if age_counter > age_threshold else 0,
            days_since_update - time_threshold_days if days_since_update > time_threshold_days else 0,
        )

        if age_factor > 0:
            if not dry_run:
                # Apply decay
                old_mask = entry.get("memory_mask", 0)
                decay_strength = decay_factor**age_factor
                decay_mask = int(255 * decay_strength)
                entry["memory_mask"] = old_mask & decay_mask
                entry["confidence"] = max(0.01, entry.get("confidence", 0.0) * decay_strength)
                # Persist the mutation
                store.put(key, entry)

            modified_count += 1

    if modified_count > 0 and not dry_run:
        store.commit()

    store.close()
    elapsed = time.time() - start_time

    return MaintenanceReport(
        operation="apply_global_confidence_decay",
        success=True,
        entries_processed=processed_count,
        entries_modified=modified_count,
        elapsed_seconds=elapsed,
    )


def export_knowledge_statistics(store_path: str, output_path: str) -> MaintenanceReport:
    """
    Export detailed statistics about a knowledge store.

    Args:
        store_path: Path to the phenotype store
        output_path: Path to save JSON statistics

    Returns:
        Maintenance report
    """
    start_time = time.time()

    if not os.path.exists(store_path):
        return MaintenanceReport(
            operation="export_knowledge_statistics",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
        )

    store = OrbitStore(store_path)
    entries = list(store.data.values())
    store.close()

    if not entries:
        stats = {"total_entries": 0, "generated_at": time.time()}
    else:
        # Calculate comprehensive statistics
        confidences = [e.get("confidence", 0.0) for e in entries]
        memory_masks = [e.get("memory_mask", 0) for e in entries]
        age_counters = [e.get("age_counter", 0) for e in entries]
        usage_counts = [e.get("usage_count", 0) for e in entries]

        # Memory utilization
        total_bits = sum(bin(mask).count("1") for mask in memory_masks)
        max_possible_bits = len(memory_masks) * 8
        memory_utilization = total_bits / max_possible_bits if max_possible_bits > 0 else 0

        # Temporal analysis
        current_time = time.time()
        ages_days = []
        for entry in entries:
            last_updated = entry.get("last_updated", current_time)
            age_days = (current_time - last_updated) / (24 * 3600)
            ages_days.append(age_days)

        # Phenotype diversity
        phenotypes = {}
        for entry in entries:
            phenotype = entry.get("phenotype", "?")
            phenotypes[phenotype] = phenotypes.get(phenotype, 0) + 1

        stats = {
            "total_entries": len(entries),
            "confidence": {
                "average": sum(confidences) / len(confidences),
                "median": sorted(confidences)[len(confidences) // 2],
                "min": min(confidences),
                "max": max(confidences),
                "high_confidence_count": sum(1 for c in confidences if c > 0.8),
                "low_confidence_count": sum(1 for c in confidences if c < 0.2),
            },
            "memory": {
                "utilization": memory_utilization,
                "total_bits_set": total_bits,
                "average_bits_per_entry": total_bits / len(entries),
            },
            "usage": {
                "total_usage": sum(usage_counts),
                "average_usage": sum(usage_counts) / len(usage_counts),
                "max_usage": max(usage_counts) if usage_counts else 0,
            },
            "age": {
                "average_age_counter": sum(age_counters) / len(age_counters),
                "average_days_since_update": sum(ages_days) / len(ages_days),
                "oldest_entry_days": max(ages_days) if ages_days else 0,
            },
            "phenotype_diversity": {
                "unique_phenotypes": len(phenotypes),
                "top_phenotypes": sorted(phenotypes.items(), key=lambda x: x[1], reverse=True)[:10],
            },
            "generated_at": time.time(),
            "store_path": store_path,
        }

    # Save statistics
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - start_time

    return MaintenanceReport(
        operation="export_knowledge_statistics",
        success=True,
        entries_processed=len(entries),
        entries_modified=0,
        elapsed_seconds=elapsed,
    )


def validate_manifold_integrity(manifold_path: str, canonical_map_path: Optional[str] = None) -> MaintenanceReport:
    """
    Validate the integrity of manifold data files.

    Args:
        manifold_path: Path to genotype map
        canonical_map_path: Optional path to canonical map

    Returns:
        Maintenance report
    """
    start_time = time.time()
    issues = []

    # Check manifold file
    if not os.path.exists(manifold_path):
        return MaintenanceReport(
            operation="validate_manifold_integrity",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
        )

    try:
        with open(manifold_path, "r") as f:
            manifold_data = json.load(f)
    except Exception:
        return MaintenanceReport(
            operation="validate_manifold_integrity",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
        )

    # Validate manifold structure
    expected_keys = ["schema_version", "genotype_map", "endogenous_modulus", "manifold_diameter", "total_states"]

    for key in expected_keys:
        if key not in manifold_data:
            issues.append(f"Missing required key: {key}")

    # Validate constants
    if manifold_data.get("endogenous_modulus") != 788_986:
        issues.append(f"Invalid endogenous modulus: {manifold_data.get('endogenous_modulus')}")

    if manifold_data.get("manifold_diameter") != 6:
        issues.append(f"Invalid manifold diameter: {manifold_data.get('manifold_diameter')}")

    genotype_map = manifold_data.get("genotype_map", {})
    if len(genotype_map) != 788_986:
        issues.append(f"Invalid genotype map size: {len(genotype_map)}")

    # Check canonical map if provided
    canonical_issues = 0
    if canonical_map_path and os.path.exists(canonical_map_path):
        try:
            with open(canonical_map_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                canonical_map = dict(enumerate(data))
            else:
                canonical_map = {int(k): v for k, v in data.items()}

            # Validate all indices are in range
            for idx, canonical_idx in canonical_map.items():
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= 788_986:
                    canonical_issues += 1
                if canonical_idx < 0 or canonical_idx >= 788_986:
                    canonical_issues += 1

            if canonical_issues > 0:
                issues.append(f"Found {canonical_issues} invalid canonical mappings")

        except Exception as e:
            issues.append(f"Failed to validate canonical map: {e}")

    elapsed = time.time() - start_time

    return MaintenanceReport(
        operation="validate_manifold_integrity",
        success=len(issues) == 0,
        entries_processed=len(genotype_map),
        entries_modified=0,
        elapsed_seconds=elapsed,
    )

"""
NOTE: To ensure all stores/views are safely closed on process exit, register their close methods with atexit
at the point where you instantiate them, e.g.:

    import atexit
    store = OrbitStore(...)
    atexit.register(store.close)

This avoids referencing undefined variables at the module level.
"""
