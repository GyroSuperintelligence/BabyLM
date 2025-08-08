"""
S3: Inference - Interpretation & Meaning Management

This module provides the InferenceEngine class responsible for
converting physical state indices into semantic meanings and managing
the learning process through Monodromic Fold.
"""

import math
from typing import Any, Tuple, cast, Optional

import numpy as np

from baby import governance
from baby.contracts import PhenotypeEntry
from baby.information import InformationEngine

import logging

logger = logging.getLogger(__name__)

# ---------- EXON PRODUCT COMPUTATION --------------------


def exon_product_from_state(state_index: int, theta: float, orbit_size: int) -> int:
    """
    Physics-native projection of 48-bit state to 8-bit exon product via bit-family extraction.

    The 8-bit product encodes (LI | FG | BG) as (2 | 3 | 3) bits taken from the
    state's canonical index. This preserves family structure without heuristic scaling.

    Args:
        state_index: Canonical index of the physical state (48-bit manifold index)
        theta: Unused here (retained for signature compatibility)
        orbit_size: Unused here (retained for signature compatibility)

    Returns:
        8-bit exon product encoding the state's family composition
    """
    # Extract family components deterministically from the state index
    li_component = (state_index >> 6) & 0x03   # 2 bits
    fg_component = (state_index >> 12) & 0x07  # 3 bits
    bg_component = (state_index >> 21) & 0x07  # 3 bits

    exon_product = ((li_component << 6) | (fg_component << 3) | bg_component) & 0xFF
    return int(exon_product)


def orbit(intron: int) -> list[int]:
    """
    Compute the deterministic orbit of an 8-bit intron under (LI, FG, BG) masks.

    This provides endogenous variety through the group action of CGM on one byte.

    Args:
        intron: 8-bit intron value

    Returns:
        List of introns in the orbit (up to 4 elements, no RNG)
    """
    from baby.governance import EXON_LI_MASK, EXON_FG_MASK, EXON_BG_MASK

    seen = {intron}
    out = [intron]
    for m in (EXON_LI_MASK, EXON_FG_MASK, EXON_BG_MASK):
        v = intron ^ m
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------- COMPACT PACK / UNPACK (9-byte fixed) --------------------


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

        # Check if mask would change before calling learn()
        old_mask = entry.get("mask", 0)
        new_mask = governance.fold(old_mask, last_intron & 0xFF)

        # Early return if mask unchanged and not a new entry
        if new_mask == old_mask and not entry_new:
            return

        self.learn(cast(PhenotypeEntry, entry), last_intron & 0xFF, rep_pre)

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
