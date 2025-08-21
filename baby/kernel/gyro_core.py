# baby/kernel/gyro_core.py

import numpy as np
from typing import Optional, Dict, List, Set

from baby.constants.frozen_channels import FROZEN_CHANNELS


class GyroEngine:
    """
    GyroSI Core Engine (irreducible physics lifecycle).

    The engine is not an inference machine with heuristics bolted on.
    It is a direct navigator of the 5 canonical maps, which are the
    physical expression of the lifecycle:

        CS      → theta map
        UNA     → ontology (all discoverable states)
        ONA     → phenomenology (orbit representatives)
        BU-Eg   → epistemology (transitions under introns)
        BU-In   → orbit sizes (scaling of ingress retrieval)

    This class provides the minimal operations needed to step through
    this lifecycle and store/retrieve learned tokens faithfully.
    """

    def __init__(
        self,
        atlas_paths: Dict[str, str],
        store_paths: Dict[str, str],
        runtime: Dict[str, str],
        version_info: Optional[Dict[str, str]] = None,
        vocab_size: Optional[int] = None,
    ):
        """Load atlas maps and initialize minimal state."""
        self.runtime = runtime

        # Load the 5 maps (the physics)
        self._load_atlas_maps(atlas_paths)

        # Reverse index: state_id (48-bit) → row index
        self._build_reverse_index()

        # Learned associations (state_idx → set of tokens)
        self.state_to_tokens: Dict[int, Set[int]] = {}
        self.last_user_address: Optional[int] = None

    # --- Map loading ---

    def _load_atlas_maps(self, atlas_paths: Dict[str, str]):
        """Load the 5 canonical physics maps."""
        self.epistemology = np.load(atlas_paths["epistemology"], mmap_mode="r", allow_pickle=False)
        self.ontology_keys = np.load(atlas_paths["ontology_keys"], mmap_mode="r", allow_pickle=False)
        self.theta = np.load(atlas_paths["theta"], mmap_mode="r", allow_pickle=False)
        self.phenomenology_map = np.load(atlas_paths["phenomenology_map"], mmap_mode="r", allow_pickle=False)
        self.orbit_sizes = np.load(atlas_paths["orbit_sizes"], mmap_mode="r", allow_pickle=False)

    def _build_reverse_index(self):
        """Build reverse index: 48-bit state ID → row index."""
        self.state_to_index = {int(state): int(i) for i, state in enumerate(self.ontology_keys)}
        print(f"Built reverse index with {len(self.state_to_index)} states")

    # --- ψ transformation (token ↔ intron) ---

    @staticmethod
    def byte_to_intron(b: int) -> int:
        """ψ(b) = b ⊕ 0xAA"""
        return (b & 0xFF) ^ 0xAA

    @staticmethod
    def intron_to_byte(i: int) -> int:
        """ψ⁻¹(i) = i ⊕ 0xAA"""
        return (i & 0xFF) ^ 0xAA

    @staticmethod
    def token_to_introns(token_id: int) -> List[int]:
        """Encode token_id into introns via ψ transformation."""
        return [GyroEngine.byte_to_intron(token_id & 0xFF)]

    # --- Core physics operations ---

    def apply_intron(self, state_id: int, intron: int) -> int:
        """
        Apply intron to state via epistemology (BU-Eg).
        This is the physics of state transition.
        """
        if state_id not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state_id:012X}")

        state_idx = self.state_to_index[state_id]
        next_idx = self.epistemology[state_idx, intron]
        return int(self.ontology_keys[next_idx])

    def micro_path(self, start_state: int, introns: List[int]) -> List[int]:
        """
        Trace the micro-path of a sequence of introns from a start state.
        Pure epistemology unfolding.
        """
        states = [start_state]
        cur = start_state
        for intron in introns:
            cur = self.apply_intron(cur, intron)
            states.append(cur)
        return states

    def start_state(self) -> int:
        """
        Common Source (CS): the state with minimal theta.
        """
        min_idx = int(np.argmin(self.theta))
        return int(self.ontology_keys[min_idx])

    # --- Lifecycle: BU-Eg / BU-In ---

    def evolve_on_user(self, state: int, token_id: int) -> int:
        """
        BU-Egress (store):
        Store token at its canonical address, but do not evolve the state.
        """
        addr = self.address_of_token(token_id)
        addr_idx = self.state_to_index[addr]

        if addr_idx not in self.state_to_tokens:
            self.state_to_tokens[addr_idx] = set()
        self.state_to_tokens[addr_idx].add(token_id)

        self.last_user_address = addr
        return state  # unchanged

    def evolve_on_assistant(self, state: int, token_id: int) -> int:
        """
        BU-Ingress (generate):
        Apply token introns to evolve the current state forward.
        """
        introns = self.token_to_introns(token_id)
        cur = state
        for intron in introns:
            cur = self.apply_intron(cur, intron)
        return cur

    def next_token_deterministic(self, state: int) -> Optional[int]:
        """
        BU-Ingress:
        Deterministically recall a token stored at this state.
        Uses orbit size as a weight, but minimally here just returns first.
        """
        state_idx = self.state_to_index[state]
        tokens = self.state_to_tokens.get(state_idx, set())
        if not tokens:
            return None
        return list(tokens)[0]

    # --- Map query utilities (to inspect geometry) ---

    def get_theta(self, state: int) -> float:
        """Return θ of a state (CS orientation)."""
        idx = self.state_to_index[state]
        return float(self.theta[idx])

    def get_orbit_representative(self, state: int) -> int:
        """Return canonical orbit representative (ONA)."""
        idx = self.state_to_index[state]
        rep_idx = self.phenomenology_map[idx]
        return int(self.ontology_keys[rep_idx])

    def get_orbit_size(self, state: int) -> int:
        """Return orbit cardinality (BU-In scaling)."""
        idx = self.state_to_index[state]
        return int(self.orbit_sizes[idx])

    # --- Address computation ---

    def address_of_token(self, token_id: int) -> int:
        """
        Canonical address of a token: use ontology index modulo size.
        This is the fold from token space into ontology (UNA).
        """
        idx = token_id % len(self.ontology_keys)
        return int(self.ontology_keys[idx])
