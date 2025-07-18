"""
S3: Inference - Interpretation & Meaning Management

This module provides the InferenceEngine class responsible for
converting physical state indices into semantic meanings and managing
the learning process through gyrogroup coaddition.
"""

import time
import hashlib
from typing import Dict, Any, Tuple

from baby.contracts import PhenotypeEntry, ValidationReport
from baby.information import InformationEngine
from baby import governance
import numpy as np


class InferenceEngine:
    """
    S3: Interpretation & Meaning Management.

    The regulatory center that converts physical states into semantic meanings.
    Contains the endogenous inference operator that bridges physical and semantic worlds.
    """

    def __init__(self, s2_engine: InformationEngine, phenotype_store: Any):
        """
        Initialize inference operator with measurement engine and storage.

        Args:
            s2_engine: Information engine for state measurement
            phenotype_store: Storage interface for phenotype data
        """
        self.s2 = s2_engine
        self.store = phenotype_store
        self.endogenous_modulus = self.s2.endogenous_modulus

    def get_phenotype(self, state_index: int, intron: int) -> PhenotypeEntry:
        """
        Convert physical state identity to semantic meaning.

        Creates or retrieves the phenotype entry that represents the semantic
        meaning of a specific physical state in a specific context.

        Args:
            state_index: Canonical index of the physical state
            intron: 8-bit context instruction

        Returns:
            Phenotype entry with semantic information
        """
        context_key = (state_index, intron)

        # Try to retrieve existing phenotype
        phenotype_entry = self.store.get(context_key)

        if not phenotype_entry:
            # Create new phenotype entry for unknown context
            semantic_address = self._compute_semantic_address(context_key)
            phenotype_entry = self._create_default_phenotype(context_key, semantic_address)
            self.store.put(context_key, phenotype_entry)

        return phenotype_entry

    def learn(self, phenotype_entry: PhenotypeEntry, intron: int) -> None:
        """
        Update memory via true gyrogroup coaddition.

        This is where all learning occurs in the GyroSI system. Uses the
        path-dependent coaddition operation to accumulate experience.

        Args:
            phenotype_entry: Entry to update with new experience
            intron: Learning signal to integrate
        """
        old_mask = phenotype_entry.get("memory_mask", 0)

        # Use S1's true gyrogroup coaddition
        new_mask = governance.coadd(old_mask, intron)

        if new_mask != old_mask:
            phenotype_entry["memory_mask"] = new_mask
            phenotype_entry["usage_count"] = phenotype_entry.get("usage_count", 0) + 1
            phenotype_entry["last_updated"] = time.time()

            # Periodic confidence boost for frequently used entries
            if phenotype_entry["usage_count"] % 1000 == 0:
                phenotype_entry["age_counter"] = min(255, phenotype_entry.get("age_counter", 0) + 1)
                # Variety-weighted confidence update
                state_index = phenotype_entry.get("context_signature", (0, 0))[0]
                variety = self.s2.orbit_cardinality[state_index]
                phenotype_entry["confidence"] = min(
                    1.0,
                    phenotype_entry.get("confidence", 0.1) * 1.1 * np.log2(variety + 1) / 8.0
                )

            # Persist the updated entry
            self.store.put(phenotype_entry.get("context_signature", (0, 0)), phenotype_entry)

    def validate_knowledge_integrity(self) -> ValidationReport:
        """
        Validates the integrity of the knowledge base.

        Returns:
            Report with integrity statistics
        """
        total_entries = 0
        confidence_sum = 0.0

        # Direct access for stores that support it
        if hasattr(self.store, "data"):
            for entry in self.store.data.values():  # type: ignore[attr-defined]
                total_entries += 1
                confidence_sum += entry.get("confidence", 0.0)

        return ValidationReport(
            total_entries=total_entries,
            average_confidence=confidence_sum / total_entries if total_entries > 0 else 0,
            store_type=type(self.store).__name__,
        )  # type: ignore

    def apply_confidence_decay(
        self, decay_factor: float = 0.999, age_threshold: int = 100, time_threshold_days: float = 30.0
    ) -> ValidationReport:
        """
        Applies temporal decay to aging knowledge entries.

        Fixed: Now uses more reasonable decay logic based on maximum of
        age counter and time-based aging, not their sum.

        Args:
            decay_factor: Multiplicative decay (0.999 = 0.1% decay per unit)
            age_threshold: Minimum age counter to trigger decay
            time_threshold_days: Days without update to trigger decay

        Returns:
            Report with number of modified entries
        """
        if not hasattr(self.store, "data"):
            raise NotImplementedError("Decay only supported for stores with direct data access")

        modified_count = 0
        current_time = time.time()

        for key, entry in getattr(self.store, "iter_entries", lambda: self.store.data.items())():
            age_counter = entry.get("age_counter", 0)
            last_updated = entry.get("last_updated", current_time)

            # Calculate time-based aging
            time_since_update = current_time - last_updated
            days_since_update = time_since_update / (24 * 3600)

            # Use maximum of the two aging factors, not sum
            age_factor = max(
                age_counter - age_threshold if age_counter > age_threshold else 0,
                days_since_update - time_threshold_days if days_since_update > time_threshold_days else 0,
            )

            if age_factor > 0:
                # Apply decay to memory mask (probabilistically clear bits)
                old_mask = entry.get("memory_mask", 0)
                # Smooth decay based on age factor
                decay_strength = decay_factor**age_factor
                decay_mask = int(255 * decay_strength)
                entry["memory_mask"] = old_mask & decay_mask

                # Apply confidence decay
                entry["confidence"] = entry.get("confidence", 0.1) * decay_strength

                # Prevent complete confidence loss
                entry["confidence"] = max(0.01, entry["confidence"])

                self.store.put(key, entry)
                modified_count += 1

        if hasattr(self.store, "commit"):
            self.store.commit()
        return ValidationReport(
            total_entries=len(self.store.data) if hasattr(self.store, "data") else 0,  # type: ignore[attr-defined]
            average_confidence=0.0,  # Would need full recalculation
            store_type=type(self.store).__name__,
            modified_entries=modified_count,
        )  # type: ignore

    def prune_low_confidence_entries(self, confidence_threshold: float = 0.05) -> int:
        """
        Remove entries below confidence threshold.

        Args:
            confidence_threshold: Minimum confidence to retain

        Returns:
            Number of entries removed
        """
        if not hasattr(self.store, "data"):
            raise NotImplementedError("Pruning only supported for stores with direct data access")

        keys_to_remove = []

        for key, entry in getattr(self.store, "iter_entries", lambda: self.store.data.items())():
            if entry.get("confidence", 1.0) < confidence_threshold:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            # Persist deletion by putting a tombstone dict
            self.store.put(key, {"context_signature": key, "__deleted__": True})
        if hasattr(self.store, "commit"):
            self.store.commit()

        return len(keys_to_remove)

    def _compute_semantic_address(self, context_key: Tuple[int, int]) -> int:
        """
        Compute deterministic semantic address for context.

        Uses hash-based mapping to endogenous modulus for consistent
        address assignment across restarts.

        Args:
            context_key: (tensor_index, intron) tuple

        Returns:
            Semantic address within endogenous modulus
        """
        # Create deterministic hash of context
        context_bytes = f"{context_key[0]}:{context_key[1]}".encode("utf-8")
        hash_digest = hashlib.sha256(context_bytes).digest()

        # Map to endogenous modulus
        hash_int = int.from_bytes(hash_digest[:8], "big")
        return hash_int % self.endogenous_modulus

    def _create_default_phenotype(self, context_key: Tuple[int, int], semantic_address: int) -> PhenotypeEntry:
        """
        Create default phenotype entry for unknown context.

        Args:
            context_key: (tensor_index, intron) tuple
            semantic_address: Computed semantic address

        Returns:
            Initialized phenotype entry
        """
        current_time = time.time()

        return PhenotypeEntry(
            phenotype="?",  # Unknown meaning initially
            memory_mask=0,  # No learned associations yet
            confidence=0.1,  # Low initial confidence
            context_signature=context_key,
            semantic_address=semantic_address,
            usage_count=0,
            age_counter=0,
            created_at=current_time,
            last_updated=current_time,
        )

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive knowledge base statistics.

        Returns:
            Dictionary with various statistics
        """
        if not hasattr(self.store, "data"):
            return {"error": "Statistics not available for this store type"}

        entries = list(self.store.data.values())  # type: ignore[attr-defined]

        if not entries:
            return {"total_entries": 0, "average_confidence": 0.0, "memory_utilization": 0.0, "age_distribution": {}}

        confidences = [e.get("confidence", 0.0) for e in entries]
        memory_masks = [e.get("memory_mask", 0) for e in entries]
        age_counters = [e.get("age_counter", 0) for e in entries]

        # Calculate memory utilization (average bits set in masks)
        total_bits = sum(bin(mask).count("1") for mask in memory_masks)
        max_possible_bits = len(memory_masks) * 8  # 8 bits per mask
        memory_utilization = total_bits / max_possible_bits if max_possible_bits > 0 else 0

        # Age distribution
        age_distribution = {}
        for age in age_counters:
            age_bucket = (age // 10) * 10  # Group by decades
            age_distribution[age_bucket] = age_distribution.get(age_bucket, 0) + 1

        return {
            "total_entries": len(entries),
            "average_confidence": sum(confidences) / len(confidences),
            "median_confidence": sorted(confidences)[len(confidences) // 2],
            "memory_utilization": memory_utilization,
            "age_distribution": age_distribution,
            "high_confidence_entries": sum(1 for c in confidences if c > 0.8),
            "low_confidence_entries": sum(1 for c in confidences if c < 0.2),
        }
