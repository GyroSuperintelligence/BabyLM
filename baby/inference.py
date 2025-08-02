"""
S3: Inference - Interpretation & Meaning Management

This module provides the InferenceEngine class responsible for
converting physical state indices into semantic meanings and managing
the learning process through Monodromic Fold.
"""

import hashlib
import math
from typing import Any, Dict, Tuple, cast, Optional, List

import numpy as np

from baby import governance
from baby.contracts import PhenotypeEntry, ValidationReport
from baby.information import InformationEngine


class InferenceEngine:
    """
    S3: Inference - Interpretation & Meaning Management.

    The regulatory center that converts physical states into semantic meanings.
    Contains the endogenous inference operator that bridges physical and semantic worlds.
    """

    def __init__(
        self,
        s2_engine: InformationEngine,
        phenotype_store: Any,
        phenomenology_map: Optional[np.ndarray[Any, np.dtype[np.integer[Any]]]] = None,
        s_buffer: Optional[List[int]] = None,
    ):
        """
        Initialize inference operator with measurement engine and storage.

        Args:
            s2_engine: Information engine for state measurement
            phenotype_store: Storage interface for phenotype data
            phenomenology_map: Optional map from state indices to orbit IDs (0-255)
            s_buffer: Optional reference to S-buffer for mask synchronization
        """
        self.s2 = s2_engine
        self.store = phenotype_store
        self.s_buffer = s_buffer

        # Fix orbit-ID lookup - map canonical indices to 0-255 range
        if phenomenology_map is not None:
            reps = np.unique(phenomenology_map)  # 256 canonical reps
            self._orbit_of: Optional[np.ndarray[Any, np.dtype[np.unsignedinteger[Any]]]] = np.zeros_like(
                phenomenology_map, dtype=np.uint8
            )
            for i, rep in enumerate(reps):
                self._orbit_of[phenomenology_map == rep] = i  # 0‥255
        else:
            self._orbit_of = None

        self.phenomenology_map = phenomenology_map

        assert self.s2._keys is not None
        self.endogenous_modulus = len(self.s2._keys)

        # Cache the maximum variety to avoid recomputation
        oc = self.s2.orbit_cardinality
        if hasattr(oc, "max"):
            self._v_max = oc.max()
        else:
            self._v_max = max(oc)
        if self._v_max == 0:
            raise ValueError("S2 orbit cardinality cannot be zero.")

        # Initialize TokenSTT for efficient token-level state transitions
        # Note: TokenSTT is not currently used - keeping for future integration
        self.token_stt = None

    def get_phenotype(self, state_index: int, token_id: int) -> PhenotypeEntry:
        """
        Retrieve or create a phenotype entry for the given state and token.

        Args:
            state_index: Canonical index of the physical state
            token_id: Token ID from the tokenizer

        Returns:
            Phenotype entry (created if not exists)

        Raises:
            IndexError: If state_index is out of bounds
        """
        # Validate state_index is within bounds
        if state_index < 0 or state_index >= len(self.s2.orbit_cardinality):
            raise IndexError(f"state_index {state_index} out of bounds [0, {len(self.s2.orbit_cardinality)})")

        # Always store with the true physical state index; CanonicalView will canonicalize.
        storage_key = (state_index, token_id)
        context_key = (state_index, token_id)  # Preserve original state_index

        entry = self.store.get(storage_key)

        if entry is None:
            entry = self._create_default_phenotype(context_key)
            self.store.put(storage_key, entry)
        else:
            # Copy to avoid mutating shared dict references
            entry = dict(entry)

        # Ensure 'key' is always present with original state_index
        if "key" not in entry:
            entry["key"] = context_key
        return cast(PhenotypeEntry, entry)

    def learn(self, phenotype_entry: PhenotypeEntry, last_intron: int, state_index: int) -> PhenotypeEntry:
        """
        Update memory via the Monodromic Fold.

        This is where all learning occurs in the GyroSI system. Uses the
        path-dependent Monodromic Fold operation to accumulate experience.

        Args:
            phenotype_entry: Entry to update with new experience
            last_intron: Last intron to integrate
            state_index: Current state index for learning rate calculation

        Returns:
            Updated phenotype entry

        Raises:
            IndexError: If state_index is out of bounds
        """
        # Validate state_index is within bounds
        if state_index < 0 or state_index >= len(self.s2.orbit_cardinality):
            raise IndexError(f"state_index {state_index} out of bounds [0, {len(self.s2.orbit_cardinality)})")

        # Clamp last_intron to 8 bits
        last_intron = last_intron & 0xFF

        # Get old mask and ensure it's clamped
        old_mask = phenotype_entry.get("mask", 0) & 0xFF

        # Use Monodromic Fold and clamp result
        new_mask = governance.fold(old_mask, last_intron) & 0xFF

        # Calculate learning rate and confidence
        novelty = bin(old_mask ^ new_mask).count("1") / 8.0

        v = self.s2.orbit_cardinality[state_index]
        alpha = (1 / 6) * math.sqrt(v / self._v_max)
        current_confidence = phenotype_entry.get("conf", 0.1)
        new_confidence = min(1.0, current_confidence + (1 - current_confidence) * alpha * novelty)

        # Short-circuit: if mask and confidence unchanged, return early
        # Use normalized comparison to avoid float16 precision issues
        if new_mask == old_mask and abs(round(new_confidence, 4) - round(current_confidence, 4)) < 1e-4:
            return phenotype_entry

        # Assertions before updating
        assert 0 <= new_mask <= 255
        assert 0 <= new_confidence <= 1

        # Update the entry with minimal structure
        phenotype_entry["mask"] = new_mask
        phenotype_entry["conf"] = new_confidence

        # Persist mutations using the original key
        key = phenotype_entry["key"]  # (state_idx, token_id)
        storage_key = key
        self.store.put(storage_key, phenotype_entry)  # persist

        return phenotype_entry

    def learn_token(self, token_id: int, state_index: int, last_intron: int) -> PhenotypeEntry:
        """
        Token-level learning using LEB128 physics.

        This method learns at the token level, applying the full token's
        intron sequence to the state transition and learning the final state.

        Args:
            token_id: Token ID from tokenizer
            state_index: Current state index
            last_intron: Last intron from the token's sequence

        Returns:
            Updated phenotype entry

        Raises:
            IndexError: If state_index is out of bounds
        """
        # Validate state_index is within bounds
        if state_index < 0 or state_index >= len(self.s2.orbit_cardinality):
            raise IndexError(f"state_index {state_index} out of bounds [0, {len(self.s2.orbit_cardinality)})")

        # Always store with the true physical state index; CanonicalView will canonicalize.
        storage_key = (state_index, token_id)
        context_key = (state_index, token_id)  # Preserve original state_index

        entry = self.store.get(storage_key)

        if entry is None:
            entry = self._create_default_phenotype(context_key)
            self.store.put(storage_key, entry)
        else:
            # Copy to avoid mutating shared dict references
            entry = dict(entry)

        # Ensure 'key' is always present with original state_index
        if "key" not in entry:
            entry["key"] = context_key

        # Learn the token using LEB128 physics
        return self.learn(entry, last_intron, state_index)

    def validate_knowledge_integrity(self) -> ValidationReport:
        def _flush_store(s: Any) -> None:
            if hasattr(s, "_flush"):
                s._flush()
            if hasattr(s, "commit"):
                s.commit()
            for attr in ("private_store", "public_store", "base_store"):
                if hasattr(s, attr):
                    _flush_store(getattr(s, attr))

        _flush_store(self.store)

        total_entries = 0
        confidence_sum = 0.0
        anomaly_count = 0
        seen: set[tuple[int, int]] = set()

        # Walk down to the lowest OrbitStore (or anything that has an 'index') to inspect raw keys
        def _unwrap_raw_store(s: Any) -> Any:
            for attr in ("base_store", "private_store", "public_store"):
                if hasattr(s, attr):
                    return _unwrap_raw_store(getattr(s, attr))
            return s

        raw_store = _unwrap_raw_store(self.store)
        raw_pairs: list[tuple[tuple[int, int], Any]] = []
        if hasattr(raw_store, "iter_entries"):
            # raw_store.iter_entries() on OrbitStore yields the key used in its index (true raw key)
            try:
                raw_pairs = list(raw_store.iter_entries())
            except Exception:
                raw_pairs = []

        def _norm_key(store_key: tuple[int, int], entry: Any) -> tuple[int, int]:
            # Used only for deduplication, not for anomaly detection
            # For token-aware keys, the key is already (state_index, token_id)
            return store_key

        for store_key, entry in raw_pairs:
            norm_key = _norm_key(store_key, entry)
            if norm_key in seen:
                anomaly_count += 1
            seen.add(norm_key)
            total_entries += 1
            confidence_sum += float(entry["conf"])

        return cast(
            ValidationReport,
            {
                "total_entries": len(seen),
                "average_confidence": confidence_sum / max(total_entries, 1),
                "store_type": type(self.store).__name__,
                "modified_entries": anomaly_count,
            },
        )

    def apply_confidence_decay(self, decay_factor: float = 0.001) -> Dict[str, Any]:
        """
        Apply confidence decay to all phenotype entries.

        Args:
            decay_factor: Decay factor (small value, e.g. 0.001)

        Returns:
            Decay report
        """

        def _flush_store(s: Any) -> None:
            if hasattr(s, "_flush"):
                s._flush()
            if hasattr(s, "commit"):
                s.commit()
            for attr in ("private_store", "public_store", "base_store"):
                if hasattr(s, attr):
                    _flush_store(getattr(s, attr))

        _flush_store(self.store)

        decayed_count = 0
        total_entries = 0

        for key, entry in self.store.iter_entries():
            total_entries += 1
            current_confidence = entry.get("conf", 0.0)
            if current_confidence > 0.0:
                new_confidence = max(0.01, current_confidence - decay_factor)  # Apply minimum floor of 0.01
                # Normalize confidence for consistent comparison
                normalized_new_conf = round(new_confidence, 4)  # Same precision as normalize_confidence
                normalized_current_conf = round(current_confidence, 4)
                if normalized_new_conf != normalized_current_conf:
                    entry["conf"] = new_confidence  # Keep original precision for storage
                    # Persist the change back to the store
                    self.store.put(key, entry)
                    decayed_count += 1

        return {
            "total_entries": total_entries,
            "decayed_entries": decayed_count,
            "decay_factor": decay_factor,
        }

    def prune_low_confidence_entries(
        self,
        *,
        confidence_threshold: float,
    ) -> int:
        """
        Remove all phenotypes whose *current* confidence is below the threshold.

        NOTE: This only mutates the in‑memory dict of the underlying
        OrbitStore (which is safe even for OverlayView).  Persisting the
        effect to disk is left to the caller via baby.policies.prune_and_compact_store.
        """
        if confidence_threshold < 0.0:
            raise ValueError("confidence_threshold must be ≥ 0.0")

        # First, flush any pending writes to ensure we see all entries
        if hasattr(self.store, "commit"):
            self.store.commit()

        # `iter_entries()` gives canonicalised keys even under the view layer.
        # Use normalized comparison to avoid float16 precision issues
        to_remove: list[tuple[int, int]] = []
        for key, entry in self.store.iter_entries():
            # Normalize confidence for consistent comparison
            normalized_conf = round(entry["conf"], 4)  # Same precision as normalize_confidence
            if normalized_conf < confidence_threshold:
                to_remove.append(key)

        for key in to_remove:
            # Use the store's delete method if available, otherwise try to delete from data
            if hasattr(self.store, "delete"):
                try:
                    self.store.delete(key)
                except (NotImplementedError, RuntimeError) as e:
                    # OverlayView may not support deletion for public entries
                    # Append-only stores don't support deletion
                    # Log but continue with other entries
                    if getattr(self, "debug_mode", False):
                        print(f"Could not delete entry {key}: {e}")
            elif hasattr(self.store, "data") and not (type(self.store).__name__ in ("OverlayView", "CanonicalView")):
                # Only try data deletion for non-view stores
                if key in self.store.data:
                    del self.store.data[key]

        # Commit changes to persist deletions
        if hasattr(self.store, "commit"):
            self.store.commit()

        return len(to_remove)

    def _compute_semantic_address(self, context_key: Tuple[int, int]) -> int:
        """
        Compute deterministic semantic address for context.

        Uses hash-based mapping to endogenous modulus for consistent
        address assignment across restarts.

        Args:
            context_key: (state_index, token_id) tuple

        Returns:
            Semantic address within endogenous modulus
        """
        # Create deterministic hash of context
        context_bytes = f"{context_key[0]}:{context_key[1]}".encode("utf-8")
        hash_digest = hashlib.sha256(context_bytes).digest()

        # Map to endogenous modulus with explicit bit handling
        hash_int = int.from_bytes(hash_digest[:8], "big") & ((1 << 64) - 1)
        return int(hash_int % self.endogenous_modulus)

    def _create_default_phenotype(self, context_key: Tuple[int, int]) -> PhenotypeEntry:
        """
        Create a new PhenotypeEntry with default values.
        
        Args:
            context_key: (state_index, token_id) tuple
            
        Returns:
            PhenotypeEntry with default values
        """
        # Start with a neutral mask; learning will move it via Monodromic Fold.
        return cast(
            PhenotypeEntry,
            {
                "mask": 0x00,
                "conf": 0.1,
                "key": context_key,
            },
        )
