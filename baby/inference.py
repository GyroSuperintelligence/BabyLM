"""
S3: Inference - Interpretation & Meaning Management

This module provides the InferenceEngine class responsible for
converting physical state indices into semantic meanings and managing
the learning process through Monodromic Fold.
"""

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
    ):
        """
        Initialize inference operator with measurement engine and storage.

        Args:
            s2_engine: Information engine for state measurement
            phenotype_store: Storage interface for phenotype data
            phenomenology_map: Optional map from state indices to orbit IDs (0-255)

        """
        self.s2 = s2_engine
        self.store = phenotype_store

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

        # Store with canonical keys (learn_token_preonly handles canonicalization).
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

        Only the mask is stored - confidence is computed at runtime from physics.
        """
        if state_index < 0 or state_index >= len(self.s2.orbit_cardinality):
            raise IndexError(f"state_index {state_index} out of bounds [0, {len(self.s2.orbit_cardinality)})")

        last_intron = last_intron & 0xFF
        old_mask = phenotype_entry.get("mask", 0)

        # Use Monodromic Fold
        new_mask = governance.fold(old_mask, last_intron)

        # Check if this is a brand-new entry
        is_new = bool(phenotype_entry.pop("_new", False))

        # Only write if mask changed or this is a new entry
        if new_mask == old_mask and not is_new:
            return phenotype_entry

        assert 0 <= new_mask <= 0xFF

        phenotype_entry["mask"] = new_mask

        key = phenotype_entry.get("key")
        if key is None:
            raise ValueError("PhenotypeEntry missing required 'key' field")
        storage_key = key
        self.store.put(storage_key, phenotype_entry)

        return phenotype_entry

    def learn_token_preonly(self, token_id: int, state_index_pre: int, last_intron: int) -> None:
        """
        Learn pre-state association for a token.
        """
        # Canonicalize to representative state
        rep_pre = state_index_pre
        if self.phenomenology_map is not None:
            rep_pre = int(self.phenomenology_map[state_index_pre])

        key = (rep_pre, token_id)
        entry = self.store.get(key)

        entry_new = entry is None
        if entry is None:
            entry = self._create_default_phenotype(key)
        else:
            entry = dict(entry)

        entry["key"] = key
        entry["_new"] = entry_new
        self.learn(cast(PhenotypeEntry, entry), last_intron & 0xFF, rep_pre)

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

        return cast(
            ValidationReport,
            {
                "total_entries": len(seen),
                "average_confidence": 0.0,  # Confidence no longer stored
                "store_type": type(self.store).__name__,
                "modified_entries": anomaly_count,
            },
        )

    def apply_confidence_decay(self, decay_factor: float = 0.99) -> Dict[str, Any]:
        """
        Legacy method for compatibility - confidence is now computed at runtime.

        Returns a report indicating no entries were modified since confidence
        is no longer stored in phenotype entries.
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

        total_entries = 0
        for _ in self.store.iter_entries():
            total_entries += 1

        return {
            "total_entries": total_entries,
            "decayed_entries": 0,  # No decay applied since confidence not stored
            "decay_factor": decay_factor,
        }

    def manage_orbit_entropy(self, max_tokens_per_orbit: int = 64) -> int:
        """
        Manage phenotype storage by orbit entropy instead of raw confidence.
        Keeps the most informative tokens per orbit.
        """
        # Group entries by orbit
        orbit_tokens: Dict[int, List[Tuple[int, Any]]] = {}  # orbit_id -> [(token_id, entry)]

        for key, entry in self.store.iter_entries():
            state_idx, token_id = key

            # Get orbit representative
            orbit_id = state_idx
            if self.phenomenology_map is not None:
                orbit_id = int(self.phenomenology_map[state_idx])

            if orbit_id not in orbit_tokens:
                orbit_tokens[orbit_id] = []
            orbit_tokens[orbit_id].append((token_id, entry))

        removed_count = 0

        for orbit_id, tokens in orbit_tokens.items():
            if len(tokens) <= max_tokens_per_orbit:
                continue

            # Score by mask entropy * uniqueness (confidence no longer stored)
            scored_tokens = []
            for token_id, entry in tokens:
                # Use mask bit diversity as a proxy for information content
                mask = entry.get("mask", 0)
                mask_entropy = bin(mask).count("1") / 8.0  # Normalized bit count

                # Higher score for rare tokens (anti-log frequency)
                # Calculate actual frequency from store
                token_frequency = 1  # Default frequency
                try:
                    # Count occurrences of this token across all states
                    token_count = 0
                    for _, entry_check in self.store.iter_entries():
                        if entry_check and entry_check.get("key") and entry_check["key"][1] == token_id:
                            token_count += 1
                    token_frequency = max(1, token_count)
                except Exception:
                    pass
                uniqueness = 1.0 / (1.0 + math.log(token_frequency))
                score = mask_entropy * uniqueness
                scored_tokens.append((score, token_id, entry))

            # Keep top K by score
            scored_tokens.sort(reverse=True)

            # Remove excess tokens
            for _, token_id, entry in scored_tokens[max_tokens_per_orbit:]:
                key = entry["key"]
                if hasattr(self.store, "delete"):
                    try:
                        self.store.delete(key)
                        removed_count += 1
                    except Exception:
                        pass

        if hasattr(self.store, "commit"):
            self.store.commit()

        return removed_count

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
                "key": context_key,
            },
        )

    def prune_low_confidence_entries(self, confidence_threshold: float = 0.05) -> int:
        """
        Legacy method for compatibility - confidence is now computed at runtime.

        Since confidence is no longer stored, this method returns 0.
        Use manage_orbit_entropy() for actual entry pruning based on physics.
        """
        return 0
