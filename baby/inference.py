"""
S3: Inference - Interpretation & Meaning Management

This module provides the InferenceEngine class responsible for
converting physical state indices into semantic meanings and managing
the learning process through Monodromic Fold.
"""

import hashlib
import math
import time
from typing import Any, Dict, Iterable, Optional, Tuple, cast

from baby.contracts import PhenotypeEntry, ValidationReport
from baby.governance import compute_governance_signature, fold, fold_sequence
from baby.information import InformationEngine


class InferenceEngine:
    """
    S3: Inference - Interpretation & Meaning Management.

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

        # Cache the maximum variety to avoid recomputation
        oc = self.s2.orbit_cardinality
        if hasattr(oc, "max"):
            self._v_max = oc.max()
        elif isinstance(oc, dict):
            self._v_max = max(oc.values())
        else:
            self._v_max = max(oc)
        if self._v_max == 0:
            raise ValueError("S2 orbit cardinality cannot be zero.")

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
        # Assert canonicality
        assert 0 <= state_index < self.endogenous_modulus

        # Clamp intron to 8 bits
        intron = intron & 0xFF

        # Form context key
        context_key = (state_index, intron)

        # Try to retrieve existing phenotype
        phenotype_entry = self.store.get(context_key)

        if not phenotype_entry:
            # Create new phenotype entry for unknown context
            phenotype_entry = self._create_default_phenotype(context_key)
            self.store.put(context_key, phenotype_entry)

        return cast(PhenotypeEntry, phenotype_entry)

    def learn_by_key(self, state_index: int, intron: int) -> PhenotypeEntry:
        """
        Retrieve or create a phenotype entry and apply the Monodromic Fold.

        Learning is path-dependent and non-associative: fold(x, x) = 0 ensures
        exon_mask toggles (x ↔ 0) rather than accumulating state. This is
        intentional for Agnostic learning. Long-term structure is stored in
        metadata (confidence, usage_count, etc.), not in exon_mask.

        Args:
            state_index: Canonical index of the physical state
            intron: 8-bit context instruction

        Returns:
            Updated phenotype entry
        """
        entry = self.get_phenotype(state_index, intron)
        self.learn(entry, intron)
        return entry

    def batch_learn(self, state_index: int, introns: Iterable[int]) -> Optional[PhenotypeEntry]:
        """
        Apply a sequence of introns in order via the Monodromic Fold.

        This preserves path dependence by reducing the intron sequence through
        the ordered, non-associative `fold_sequence` function before a single
        learn operation on the final accumulated intron.
        """
        introns_list = list(introns)
        if not introns_list:
            return None

        # Reduce the sequence into a single path-dependent intron.
        # We start the fold from 0 (the left identity of the Fold operation).
        path_intron = fold_sequence([i & 0xFF for i in introns_list], start_state=0)

        # learn_by_key will correctly call the single 'fold' operation
        return self.learn_by_key(state_index, path_intron)

    def learn(self, phenotype_entry: PhenotypeEntry, intron: int) -> PhenotypeEntry:
        """
        Update memory via the Monodromic Fold.

        This is where all learning occurs in the GyroSI system. Uses the
        path-dependent Monodromic Fold operation to accumulate experience.

        Note: Only novel updates (changing masks) reset the age counter.

        Args:
            phenotype_entry: Entry to update with new experience
            intron: Learning signal to integrate

        Returns:
            Updated phenotype entry
        """
        # Clamp intron to 8 bits
        intron = intron & 0xFF
        # Use direct key access for context_signature, but handle missing key explicitly
        context_key = phenotype_entry.get("context_signature")
        if context_key is None:
            raise KeyError("Phenotype entry is missing required 'context_signature' key.")
        state_index = context_key[0]
        # Canonical index assertion
        assert 0 <= state_index < self.endogenous_modulus
        # Get old mask and ensure it's clamped
        old_mask = phenotype_entry.get("exon_mask", 0) & 0xFF

        # Always increment usage count for any learn attempt
        phenotype_entry["usage_count"] = phenotype_entry.get("usage_count", 0) + 1

        # Use Monodromic Fold and clamp result
        new_mask = fold(old_mask, intron) & 0xFF
        # Calculate learning rate and confidence
        novelty = bin(old_mask ^ new_mask).count("1") / 8.0
        v = self.s2.orbit_cardinality[state_index]
        alpha = (1 / 6) * math.sqrt(v / self._v_max)
        current_confidence = phenotype_entry.get("confidence", 0.1)
        new_confidence = min(1.0, current_confidence + (1 - current_confidence) * alpha * novelty)
        # Short-circuit: if mask and confidence unchanged, only update usage_count/last_updated in memory
        if new_mask == old_mask and abs(new_confidence - current_confidence) < 1e-9:
            phenotype_entry["last_updated"] = time.time()
            if hasattr(self.store, "mark_dirty"):
                self.store.mark_dirty(context_key, phenotype_entry)
            return phenotype_entry
        # Assertions before updating
        assert 0 <= new_mask <= 255
        assert 0 <= new_confidence <= 1
        # Update the entry
        phenotype_entry["exon_mask"] = new_mask
        sig = compute_governance_signature(new_mask)
        phenotype_entry["governance_signature"] = {
            "neutral": sig[0],
            "li": sig[1],
            "fg": sig[2],
            "bg": sig[3],
            "dyn": sig[4],
        }
        phenotype_entry["confidence"] = new_confidence
        # Write to the correct key
        self.store.put(context_key, phenotype_entry)
        return phenotype_entry

    def validate_knowledge_integrity(self) -> ValidationReport:
        """
        Validates the integrity of the knowledge base.

        Verifies critical invariants for all entries:
        - Context signatures match keys
        - Memory masks within valid range
        - Confidence values within valid range
        - State indices canonical

        Returns:
            Report with integrity statistics and anomaly count.
            Note: The 'modified_entries' field in the returned ValidationReport is the anomaly count.
        """
        total_entries = 0
        confidence_sum = 0.0
        anomaly_count = 0

        for key, entry in self.store.iter_entries():
            total_entries += 1
            confidence_sum += entry.get("confidence", 0.0)

            # Check invariants
            try:
                # Key matches context signature
                if entry.get("context_signature") != (key):
                    anomaly_count += 1

                # Memory mask in valid range
                mask = entry.get("exon_mask", 0)
                if not (0 <= mask <= 255):
                    anomaly_count += 1

                # Confidence in valid range
                conf = entry.get("confidence", 0.0)
                if not (0 <= conf <= 1.0):
                    anomaly_count += 1

                # State index canonical
                state_idx = entry.get("context_signature", (0, 0))[0]
                if not (0 <= state_idx < self.endogenous_modulus):
                    anomaly_count += 1

                # Timestamps monotonic
                created = entry.get("created_at", 0)
                updated = entry.get("last_updated", 0)
                if updated < created:
                    anomaly_count += 1
            except Exception:
                # Any other exception during validation
                anomaly_count += 1

        return ValidationReport(
            total_entries=total_entries,
            average_confidence=confidence_sum / total_entries if total_entries > 0 else 0,
            store_type=type(self.store).__name__,
            modified_entries=anomaly_count,
        )

    def apply_confidence_decay(self, decay_factor: float = 0.001) -> Dict[str, Any]:
        modified_count = 0
        total_entries = 0
        for key, entry in self.store.iter_entries():
            # Apply simple exponential decay to confidence only
            new_conf = max(0.01, entry.get("confidence", 0.1) * math.exp(-decay_factor))
            assert 0 <= new_conf <= 1.0
            entry["confidence"] = new_conf
            self.store.put(key, entry)
            modified_count += 1
            total_entries += 1
        if hasattr(self.store, "commit"):
            self.store.commit()
        report = ValidationReport(
            total_entries=total_entries,
            average_confidence=0.0,  # Would need full recalculation
            store_type=type(self.store).__name__,
            modified_entries=modified_count,
        )
        # Convert ValidationReport to dict
        try:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(report):
                return asdict(report)
        except ImportError:
            pass
        if hasattr(report, "__dict__"):
            return vars(report)
        return {
            "total_entries": total_entries,
            "average_confidence": 0.0,
            "store_type": type(self.store).__name__,
            "modified_entries": modified_count,
        }

    def prune_low_confidence_entries(self, confidence_threshold: float = 0.05) -> int:
        """
        Remove entries below confidence threshold.

        Args:
            confidence_threshold: Minimum confidence to retain

        Returns:
            Number of entries removed
        """
        keys_to_remove = []

        for key, entry in self.store.iter_entries():
            if entry.get("confidence", 1.0) < confidence_threshold:
                keys_to_remove.append(key)

        # Directly delete entries instead of using tombstones
        for key in keys_to_remove:
            if hasattr(self.store, "delete"):
                self.store.delete(key)
            else:
                raise NotImplementedError("Store does not support deletion via 'delete' method.")

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

        # Map to endogenous modulus with explicit bit handling
        hash_int = int.from_bytes(hash_digest[:8], "big") & ((1 << 64) - 1)
        return int(hash_int % self.endogenous_modulus)

    def _create_default_phenotype(self, context_key: Tuple[int, int]) -> PhenotypeEntry:
        """
        Create default phenotype entry for unknown context.

        Args:
            context_key: (tensor_index, intron) tuple

        Returns:
            Initialized phenotype entry
        """
        current_time = time.time()
        state_index, intron = context_key
        v = self.s2.orbit_cardinality[state_index]
        confidence = (1 / 6) * math.sqrt(v / self._v_max)
        phenotype = f"P[{state_index}:{intron}]"
        init_mask = intron & 0xFF
        sig = compute_governance_signature(init_mask)
        return {
            "phenotype": phenotype,
            "exon_mask": init_mask,  # Initial mask is the intron
            "confidence": confidence,
            "governance_signature": {"neutral": sig[0], "li": sig[1], "fg": sig[2], "bg": sig[3], "dyn": sig[4]},
            "context_signature": context_key,
            "usage_count": 0,
            "created_at": current_time,
            "last_updated": current_time,
            "_original_context": None,
        }
