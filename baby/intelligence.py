"""
S5: Intelligence - Semantic Memory & Learning

This module provides the IntelligenceEngine class responsible for
coordinating the entire GyroSI learning process.
"""

import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, cast, Iterator, Tuple

import numpy as np
from numpy.typing import NDArray

from baby import governance
from baby.contracts import AgentConfig, CycleHookFunction, PreferencesConfig, PhenotypeEntry
from baby.inference import InferenceEngine
from baby.information import InformationEngine, SEP_ID
from baby.policies import CanonicalView, PhenotypeStore, OverlayView, ReadOnlyView


class _TokBridge:
    def id_to_bytes(self, tok_id: int) -> bytes:
        from baby.information import token_id_to_bytes

        return token_id_to_bytes(tok_id)

    def bytes_to_id(self, bs: bytes) -> int:
        from baby.information import bytes_to_token_id

        return bytes_to_token_id(bs)

    def bytes_to_ids(self, bs: bytes) -> List[int]:
        from baby.information import bytes_to_token_ids

        return bytes_to_token_ids(bs)


TOK = _TokBridge()

if TYPE_CHECKING:
    pass


class StateInfo(TypedDict):
    agent_id: str
    cycle_count: int
    state_integer: int
    tensor_index: int
    angular_divergence_radians: float
    angular_divergence_degrees: float
    active_hooks: int
    cycle_step: str


def _abs(path: Optional[str], base: Path) -> str:
    if path is None:
        raise ValueError("Path must not be None")
    p = Path(os.path.expanduser(str(path)))
    return str(p if p.is_absolute() else base / p)


# --- JIT batch function for epistemology ---


class IntelligenceEngine:
    """
    S4: Strategic Operations & Environment Interface.

    Manages agent state evolution, orchestrates the egress/ingress cycle,
    and implements operational strategies. Handles adaptation to external demands.
    """

    preferences: Dict[str, Any]
    phenomenology_map: Optional[NDArray[Any]]
    epistemology: NDArray[Any]
    s2: Any
    operator: Any
    phenotype_store: Any
    current_state_index: int
    _store_mutation_epoch: int
    _state_buf: NDArray[np.int32]
    _raw_leb_buf: List[int]
    MAX_TOKEN_BYTES: int
    _S: List[int]
    post_cycle_hooks: List[CycleHookFunction]
    cycle_count: int
    _pain_streak: int
    agent_id: str
    hook_batch_interval: int
    base_path: Path
    ontology_path: str
    gene_mac_m_int: int
    _last_token_id: int
    learning_enabled: bool  # New flag to control learning

    def __init__(
        self,
        ontology_path: str,
        phenotype_store: Any,
        agent_id: Optional[str] = None,
        hook_batch_interval: int = 8,
        epistemology_path: Optional[str] = None,
        phenomenology_map_path: Optional[str] = None,
        base_path: Path = Path(__file__).resolve().parents[2],
        preferences: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize intelligence engine.
        All file paths are resolved with respect to base_path unless absolute.
        """
        self.base_path = base_path
        self.preferences = preferences or {}
        self.ontology_path = _abs(
            ontology_path if ontology_path is not None else "memories/public/meta/ontology_keys.npy", self.base_path
        )
        self.phenotype_store = phenotype_store
        self.agent_id = agent_id or str(uuid.uuid4())
        self.hook_batch_interval = hook_batch_interval
        self.post_cycle_hooks: List[CycleHookFunction] = []
        self.cycle_count = 0
        self._pain_streak: int = 0
        self.learning_enabled = True  # Enable learning by default

        self.current_state_index = 0

        # --- phenomenology setup ----------------------------------------------
        if phenomenology_map_path is not None:
            phenomenology_map_path = _abs(phenomenology_map_path, self.base_path)
            self.phenomenology_map = np.load(phenomenology_map_path)
        else:
            self.phenomenology_map = None

        # --- information setup -----------------------------------------------
        ep_path_resolved = _abs(epistemology_path or "memories/public/meta/epistemology.npy", self.base_path)
        self.s2 = InformationEngine(
            keys_path=self.ontology_path,
            ep_path=ep_path_resolved,
            phenomap_path=_abs(phenomenology_map_path or "memories/public/meta/phenomenology_map.npy", self.base_path),
            # theta is emitted alongside the epistemology
            theta_path=str(Path(ep_path_resolved).with_name("theta.npy")),
        )

        # Load epistemology directly from file with memory mapping to avoid RAM duplication
        self.epistemology = np.load(ep_path_resolved, mmap_mode="r")

        # --- S-buffer for intron selection ----------------------------------
        self._S = [0] * 8

        # --- operator setup ---------------------------------------------------
        self.operator = InferenceEngine(
            s2_engine=self.s2,
            phenotype_store=self.phenotype_store,
            phenomenology_map=self.phenomenology_map,
        )

        # --- token buffer for egress processing: stores introns (internal bytes) ---
        self._raw_leb_buf: List[int] = []
        self.MAX_TOKEN_BYTES = 1024  # Maximum token buffer size

        # --- state tracking --------------------------------------------------
        # Start at archetypal state from ontology via STT
        self.current_state_index = self.s2.get_index_from_state(self.s2.tensor_to_int(governance.GENE_Mac_S))
        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
        self._last_token_id = 0

        # --- vectorized epistemology buffer ----------------------------------
        # Reusable buffer for state trajectory computation (max 64K to prevent RAM explosion)
        self._state_buf = np.empty(65536, dtype=np.int32)

        # --- hot-path caches ---------------------------------------------------
        self._neigh_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self._cache_full_mask: Dict[int, int] = {}
        self._cache_tok_first_intron: Dict[int, int] = {}
        self._cache_tokens_by_tail: Dict[int, Tuple[int, ...]] = {}
        self._cache_token_bytes: Dict[int, bytes] = {}

    def _action_value(self, state_idx: int, intron: int, mask: int) -> float:
        """
        Pure-physics utility for choosing the next token.
        No temperature, no stochasticity.
        
        Args:
            state_idx: Current state index
            intron: Intron byte that would be applied
            mask: Learned mask for the candidate token
            
        Returns:
            Action value (higher is better)
        """
        next_state = int(self.epistemology[state_idx, intron])
        
        # 1) cooling term (Δθ = θ - θ')
        θ_now = self.s2._theta_table[state_idx]
        θ_next = self.s2._theta_table[next_state]
        dθ = θ_now - θ_next  # we want positive (cooling)

        # 2) information injected by Fold
        current_mask = self._state_byte_projection(state_idx)
        fold_result = governance.fold(current_mask, mask)
        H_before = bin(current_mask).count("1")
        H_after = bin(fold_result).count("1")
        fold_entropy = H_after - H_before

        return float(dθ + fold_entropy)

    def _neighbourhood(self, state_idx: int, max_theta: float = 0.30) -> List[int]:
        """
        Return representative indices whose θ-distance to the current state
        is ≤ max_theta.

        This implements spectral neighborhood retrieval for accessing learned
        patterns from nearby states in the manifold topology.

        Args:
            state_idx: Current state index
            max_theta: Maximum angular distance (radians) for neighborhood inclusion

        Returns:
            List of representative state indices in the local neighborhood
        """
        if self.s2._theta_table is None:
            # Fallback to single representative if theta table is missing
            if self.operator.phenomenology_map is not None:
                return [int(self.operator.phenomenology_map[state_idx])]
            return [state_idx]

        # Get reference value for current state
        θ0 = self.s2._theta_table[state_idx]

        # Vectorized filter over the whole manifold
        θ = self.s2._theta_table
        mask = np.abs(θ - θ0) <= max_theta

        # Apply phenomenology projection to get representatives
        if self.operator.phenomenology_map is not None:
            candidate_indices = np.nonzero(mask)[0]
            reps = self.operator.phenomenology_map[candidate_indices]
            return list(np.unique(reps).tolist())
        else:
            return list(np.nonzero(mask)[0].tolist())

    def _state_byte_projection(self, state_index: int) -> int:
        """Stable 8-bit projection of 48-bit state via XOR-fold; invariant under bit shifts that preserve parity."""
        try:
            s = self.s2.get_state_from_index(state_index)  # 48-bit int
        except Exception:
            s = state_index
        # XOR-fold 48→8 bits (6 bytes)
        b0 = (s >> 0) & 0xFF
        b1 = (s >> 8) & 0xFF
        b2 = (s >> 16) & 0xFF
        b3 = (s >> 24) & 0xFF
        b4 = (s >> 32) & 0xFF
        b5 = (s >> 40) & 0xFF
        return int(b0 ^ b1 ^ b2 ^ b3 ^ b4 ^ b5)

    def _emit_token_with_feedback(self, state_idx: int, theta: float) -> Tuple[int, bytes]:
        """Select one token, emit its full LEB128 byte sequence, and advance physics.
        Returns (token_id, emitted_bytes)."""
        tok_id = self.generate_token_exon(state_idx)

        from baby.information import token_id_to_bytes

        token_bytes = token_id_to_bytes(tok_id)
        # Advance physics unconditionally. Learning writes are gated by `learning_enabled`.
        self.process_egress_bulk(token_bytes)

        return tok_id, token_bytes

    # ---------------- hot-path helpers (cached) ----------------
    def _get_token_bytes(self, tok_id: int) -> bytes:
        bs = self._cache_token_bytes.get(tok_id)
        if bs is None:
            bs = TOK.id_to_bytes(int(tok_id))
            # simple cap to avoid unbounded growth
            if len(self._cache_token_bytes) > 16384:
                self._cache_token_bytes.clear()
            self._cache_token_bytes[tok_id] = bs
        return bs

    def _get_token_first_intron(self, tok_id: int) -> int:
        v = self._cache_tok_first_intron.get(tok_id)
        if v is None:
            data = self._get_token_bytes(tok_id)
            v = governance.transcribe_byte(data[0]) & 0xFF if data else 0
            if len(self._cache_tok_first_intron) > 32768:
                self._cache_tok_first_intron.clear()
            self._cache_tok_first_intron[tok_id] = v
        return v

    def _get_full_mask(self, tok_id: int) -> int:
        m = self._cache_full_mask.get(tok_id)
        if m is None:
            from baby.information import token_to_introns

            m = governance.fold_sequence(token_to_introns(int(tok_id))) & 0xFF
            if len(self._cache_full_mask) > 32768:
                self._cache_full_mask.clear()
            self._cache_full_mask[tok_id] = m
        return m

    def _get_tokens_by_tail(self, intron: int) -> Tuple[int, ...]:
        out = self._cache_tokens_by_tail.get(intron)
        if out is None:
            from baby.information import find_tokens_by_last_intron

            out = tuple(find_tokens_by_last_intron(int(intron)))
            if len(self._cache_tokens_by_tail) > 1024:
                self._cache_tokens_by_tail.clear()
            self._cache_tokens_by_tail[intron] = out
        return out

    def process_egress(self, input_byte: int) -> int:
        input_byte &= 0xFF
        intron = governance.transcribe_byte(input_byte)
        intron &= 0xFF

        if len(self._raw_leb_buf) >= self.MAX_TOKEN_BYTES:
            self._raw_leb_buf.clear()
            self._last_token_id = 0
            return intron

        self._raw_leb_buf.append(intron)

        # Cache pre-state before transition
        pre_state_index = self.current_state_index

        # Apply state transition
        new_state_index = self.epistemology[self.current_state_index, intron]
        self.current_state_index = new_state_index
        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)

        assert self.gene_mac_m_int < (1 << 48)
        self.cycle_count += 1

        if (intron & 0x80) == 0:  # Token complete
            try:
                tok_bytes = bytes(b ^ 0xAA for b in self._raw_leb_buf)
                token_id = TOK.bytes_to_id(tok_bytes)

                if token_id == SEP_ID:
                    # Learn SEP as a pre-state association so it can be chosen later
                    self.operator.learn_token_preonly(token_id, pre_state_index, intron)
                    self._last_token_id = token_id
                    self._raw_leb_buf.clear()
                    return intron
                self._prev_token_id = token_id

                # Learn at post-state using the token's last intron (Genetics §7.1/§5.5.4)
                from baby.governance import token_last_intron

                token_last_intron_byte = token_last_intron(token_id)
                post_state_index = self.current_state_index
                self.operator.learn_token_postonly(token_id, post_state_index, token_last_intron_byte)

                # Get the post-state entry for hooks
                post_entry = self.operator.store.get((post_state_index, token_id))
                for hook in self.post_cycle_hooks:
                    try:
                        hook(self, cast(PhenotypeEntry, post_entry), intron, token_id, post_state_index)
                    except Exception:
                        pass

                self._last_token_id = token_id
            except Exception:
                pass
            finally:
                self._raw_leb_buf.clear()

        return intron

    def reset_token_buffer(self) -> None:
        """
        Reset token buffer - call when starting new stream.
        Clears accumulated bytes and resets last token ID.
        """
        self._raw_leb_buf.clear()
        self._last_token_id = 0

    def process_egress_bulk(self, blob: bytes) -> None:
        """
        Vectorised replacement for repeated process_egress().
        Handles state updates in C/NumPy, then walks the token
        boundaries just once per byte for learning.
        """
        if not blob:
            return

        # Thread safety: use store lock to prevent data races on _state_buf
        store_lock = getattr(self.operator.store, "lock", None)
        if store_lock is not None:
            with store_lock:
                self._process_egress_bulk_internal(blob)
        else:
            # Fallback if no lock available
            self._process_egress_bulk_internal(blob)

    def _process_egress_bulk_internal(self, blob: bytes) -> None:
        """Internal implementation of process_egress_bulk without thread safety."""
        arr = np.frombuffer(blob, dtype=np.uint8)  # masked bytes (external)
        introns = arr ^ 0xAA  # unmask once → true introns

        # --- 1. state evolution ------------------------------------------------
        # Fully vectorized STT processing for high performance

        # --- 1. build full state trajectory in one shot
        intr = introns  # alias for brevity (np.ndarray[uint8])
        n = intr.shape[0]

        # Allocate state buffer with memory bounds (max 64K to prevent RAM explosion)
        max_chunk_size = 65536  # 64K introns max per chunk
        if n > max_chunk_size:
            # Process in chunks to maintain memory bounds
            for chunk_start in range(0, n, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, n)
                chunk_introns = intr[chunk_start:chunk_end]
                self._process_epistemology_chunk(chunk_introns, arr[chunk_start:chunk_end])
        else:
            # Process full chunk if within bounds
            self._process_epistemology_chunk(intr, arr)

        # Update cycle count
        self.cycle_count += len(arr)

    def _process_epistemology_chunk(
        self, introns: np.ndarray[Any, Any], masked_arr: Optional[np.ndarray[Any, Any]] = None
    ) -> None:
        """
        Process a chunk of introns using fully vectorized STT.

        Args:
            introns: Unmasked intron array (uint8)
            masked_arr: Original masked byte array for token extraction
        """
        ep = self.epistemology

        n = introns.shape[0]
        if n == 0:
            return

        # --- 1. build full state trajectory in one shot using vectorized operations
        # Allocate buffer slice - reuse existing buffer to avoid allocations
        if self._state_buf.shape[0] < n:
            # Expand buffer if needed (should be rare with 64K max chunks)
            self._state_buf = np.empty(max(n, 65536), dtype=np.int32)

        st = self._state_buf[:n]
        # Initialize the entire buffer to avoid garbage values
        st.fill(0)
        st[0] = self.current_state_index

        # Bounds checking to prevent out-of-bounds access
        epistemology_size = ep.shape[0]
        if st[0] >= epistemology_size:
            raise RuntimeError(
                f"Initial state index {st[0]} is out of bounds for epistemology matrix (size {epistemology_size})"
            )

        # Optimized vectorized state transitions: compute all transitions in one operation
        # Use NumPy advanced indexing for better performance
        if n > 1:
            # Vectorized state transitions: st[1:] = ep[st[:-1], introns[:-1]]
            prev_states = st[:-1]
            prev_introns = introns[:-1]

            # Bounds checking for all transitions at once
            if np.any(prev_states >= epistemology_size):
                raise RuntimeError("State indices out of bounds for epistemology matrix")

            # Compute all transitions in one operation
            new_states = ep[prev_states, prev_introns]

            # All transitions in the epistemology matrix are valid
            # State index 0 is a valid state, not an invalid one

            # Bounds checking for new states
            if np.any(new_states >= epistemology_size):
                raise RuntimeError("Transition results out of bounds for epistemology matrix")

            # Assign all new states at once
            st[1:] = new_states

        # Process tokens with correct phase
        token_start_idx = 0
        for i, intron in enumerate(introns):
            if len(self._raw_leb_buf) >= self.MAX_TOKEN_BYTES:
                self._raw_leb_buf.clear()
                self._last_token_id = 0
                token_start_idx = i + 1  # Reset token start

            self._raw_leb_buf.append(intron)

            if (intron & 0x80) == 0:  # Token complete
                tok_bytes = bytes(b ^ 0xAA for b in self._raw_leb_buf)
                try:
                    token_id = TOK.bytes_to_id(tok_bytes)

                    if token_id == SEP_ID:
                        # Get pre-state (start of this token)
                        pre_state = int(st[token_start_idx])
                        # Only learn if learning is enabled
                        if self.learning_enabled:
                            # Learn post-state association at token closure
                            post_state = int(st[i])  # state after applying this intron
                            self.operator.learn_token_postonly(token_id, post_state, int(intron))
                        # Drop cached candidates for this representative state
                        rep = (
                            int(self.phenomenology_map[pre_state]) if self.phenomenology_map is not None else pre_state
                        )
                        # Drop cached candidates for this representative state (if cache exists)
                        cand_cache = getattr(self, "_cand_cache", None)
                        if cand_cache is not None:
                            cand_cache.pop(rep, None)
                        self._last_token_id = token_id
                        self._raw_leb_buf.clear()
                        token_start_idx = i + 1
                        continue
                    self._prev_token_id = token_id

                    # Get pre-state (where token started)
                    pre_state = int(st[token_start_idx])

                    # Post-state storage (only if learning is enabled)
                    if self.learning_enabled:
                        post_state = int(st[i])
                        self.operator.learn_token_postonly(token_id, post_state, int(intron))

                    # Get the post-state entry for hooks
                    post_entry = self.operator.store.get((post_state, token_id))
                    for hook in self.post_cycle_hooks:
                        try:
                            hook(self, cast(PhenotypeEntry, post_entry), int(intron), token_id, post_state)
                        except Exception:
                            pass

                    self._last_token_id = token_id
                except Exception:
                    pass
                finally:
                    self._raw_leb_buf.clear()
                    token_start_idx = i + 1  # Next token starts here

        # --- 4. persist new head state
        if n > 0:
            final_pre_state = int(st[n - 1])  # state *before* applying the last intron
            final_intron = int(introns[-1])
            if final_pre_state >= epistemology_size:
                raise RuntimeError(
                    f"Final pre-state index {final_pre_state} is out of bounds "
                    f"for epistemology matrix (size {epistemology_size})"
                )
            new_state = self.epistemology[final_pre_state, final_intron]

            # All transitions in the epistemology matrix are valid
            # State index 0 is a valid state, not an invalid one

            if new_state >= epistemology_size:
                raise RuntimeError(
                    f"Final transition to state {new_state} is out of bounds "
                    f"for epistemology matrix (size {epistemology_size})"
                )

            self.current_state_index = new_state
            # keep state integer in sync with index
            try:
                self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
            except Exception:
                pass

    def process_ingress(self) -> tuple[int, int]:
        """Emit exactly one token and return (last_byte, last_intron) of that token."""
        state_idx = self.current_state_index
        theta = 0.0  # Removed theta tracking
        tok_id, token_bytes = self._emit_token_with_feedback(state_idx, theta)
        # Derive last intron from last byte via ψ
        last_byte = token_bytes[-1]
        last_intron = last_byte ^ 0xAA
        return last_byte, last_intron

    def add_hook(self, hook: CycleHookFunction) -> None:
        """
        Add a post-cycle hook for monitoring or maintenance.

        Args:
            hook: Function called after each ingress cycle
        """
        self.post_cycle_hooks.append(hook)

    def remove_hook(self, hook: CycleHookFunction) -> bool:
        """
        Remove a previously added hook.

        Args:
            hook: Hook function to remove

        Returns:
            True if hook was found and removed
        """
        try:
            self.post_cycle_hooks.remove(hook)
            return True
        except ValueError:
            return False

    def get_state_info(self) -> StateInfo:
        """
        Get comprehensive information about current agent state.

        Returns:
            Dictionary with state information
        """
        self._sync_state_fields_from_index()
        angular_divergence = self.s2.measure_state_divergence_index(self.current_state_index)
        tensor_index = self.current_state_index
        agent_id: str = self.agent_id
        cycle_count: int = self.cycle_count
        state_integer: int = self.gene_mac_m_int
        tensor_index_val: int = tensor_index
        angular_divergence_radians: float = float(angular_divergence)
        angular_divergence_degrees: float = float(angular_divergence * 180 / 3.14159)
        active_hooks: int = len(self.post_cycle_hooks)
        cycle_step: str = "CS"  # Default since we removed cycle tracking
        info: StateInfo = {
            "agent_id": agent_id,
            "cycle_count": cycle_count,
            "state_integer": state_integer,
            "tensor_index": tensor_index_val,
            "angular_divergence_radians": angular_divergence_radians,
            "angular_divergence_degrees": angular_divergence_degrees,
            "active_hooks": active_hooks,
            "cycle_step": cycle_step,
        }
        return info

    def reset_to_archetypal_state(self) -> None:
        """Reset agent to the archetypal state (GENE_Mac_S)."""
        self.gene_mac_m_int = self.s2.tensor_to_int(governance.GENE_Mac_S)
        self._sync_index_from_state_int()
        self.cycle_count = 0

    def _sync_state_fields_from_index(self) -> None:
        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)

    def _sync_index_from_state_int(self) -> None:
        self.current_state_index = self.s2.get_index_from_state(self.gene_mac_m_int)

    def generate_token_exon(self, state_index: int) -> int:
        """
        Generate next token using spectral neighborhood retrieval and physics.

        Implements the theoretical approach:
        1. Compute exon product from current state physics
        2. Use spectral neighborhood retrieval for learned patterns
        3. Integrate physics-native action values with interference terms
        4. Select token using constructive overlap between state and memory
        """
        import math
        import numpy as np
        from functools import lru_cache

        from baby.inference import orbit, exon_product_from_state
        from baby import governance  # for fold in entropy term

        # --- Cache helpers for performance ---
        @lru_cache(maxsize=1 << 15)
        def _tok_first_intron(tok_id: int) -> int:
            """Return the intron corresponding to the first emitted byte of the token.
            This aligns scoring with the immediate STT update used during egress.
            """
            from baby.information import token_id_to_bytes
            from baby.governance import transcribe_byte

            data = token_id_to_bytes(tok_id)
            if data:
                return transcribe_byte(data[0]) & 0xFF
            return 0

        @lru_cache(maxsize=1 << 15)
        def _tok_full_mask(tok_id: int) -> int:
            from baby.information import token_to_introns
            from baby.governance import fold_sequence

            return fold_sequence(token_to_introns(tok_id))

        @lru_cache(maxsize=256)
        def _tokens_by_tail(intron: int) -> tuple[int, ...]:
            from baby.information import find_tokens_by_last_intron

            return tuple(find_tokens_by_last_intron(intron))

        # Step 1: Compute exon product from current state physics
        theta = self.s2.measure_state_divergence_index(state_index)
        orbit_size = self.s2.get_orbit_cardinality(state_index)
        exon_product = exon_product_from_state(state_index, theta, orbit_size)

        # Step 2: Get spectral neighborhood for learned pattern retrieval (cached per-instance)
        if not hasattr(self, "_neigh_cache"):
            self._neigh_cache = {}
        neigh_key = (int(state_index), 10)  # 10 => 0.10 scaled
        neighborhood_reps = self._neigh_cache.get(neigh_key)
        if neighborhood_reps is None:
            neighborhood_reps = tuple(self._neighbourhood(state_index, max_theta=0.10))
            # simple capacity control
            if len(self._neigh_cache) > 4096:
                self._neigh_cache.clear()
            self._neigh_cache[neigh_key] = neighborhood_reps
        # Neighborhood computed; no debug output

        # Step 3: Collect candidate tokens from both physics and learned patterns
        candidate_tokens: set[int] = set()

        # Physics-derived candidates from resonant introns (expanded search)
        resonant_introns = orbit(exon_product)

        # Add primary resonant introns using even-spread deterministic sampling
        for intron in resonant_introns:
            tokens = self._get_tokens_by_tail(intron)
            n = len(tokens)
            k = 30
            if n:
                stride = max(1, n // k)
                sampled = tokens[::stride][:k]
            else:
                sampled = ()
            candidate_tokens.update(sampled)

        # Expand search to nearby exon products for vocabulary diversity
        nearby_exons = [(exon_product + i) % 256 for i in [-2, -1, 1, 2]]  # Nearby exon products
        for nearby_exon in nearby_exons:
            nearby_introns = orbit(nearby_exon)
            for intron in nearby_introns[:2]:  # Just first 2 introns from each nearby exon
                tokens = self._get_tokens_by_tail(intron)
                n = len(tokens)
                k = 10
                if n:
                    stride = max(1, n // k)
                    sampled = tokens[::stride][:k]
                else:
                    sampled = ()
                candidate_tokens.update(sampled)

        # Physics expansion complete

        # Learned candidates from spectral neighborhood
        for rep_state in neighborhood_reps:
            try:
                for _, token_id in self.operator.store.iter_keys_for_state(rep_state):
                    candidate_tokens.add(token_id)
                    if len(candidate_tokens) > 100:  # Reasonable limit
                        break
            except Exception:
                continue

        # Candidate collection complete

        # Ensure we have at least some candidates
        if not candidate_tokens:
            return SEP_ID

        # Step 4: Vectorized evaluation of all candidates
        # Convert to list for indexing and filter out tokenizer-reserved [unusedX] ids
        from functools import lru_cache
        from baby.information import decode_text

        @lru_cache(maxsize=1 << 15)
        def _is_unused(tok_id: int) -> bool:
            try:
                txt = decode_text(self._get_token_bytes(int(tok_id)))
                return txt.startswith("[unused")
            except Exception:
                return False

        filtered = [t for t in candidate_tokens if not _is_unused(t)]
        candidates = filtered if filtered else list(candidate_tokens)

        # Special-case at CS: drop candidate tokens whose intron sequence is entirely standing
        # (no FG/BG drive bits). This enforces PCE: emergence to UNA under driving introns.
        try:
            state_int = int(self.s2.get_state_from_index(state_index))
        except Exception:
            state_int = -1
        if state_int == 0:  # governance.CS_INT, using integer value 0
            driven: list[int] = []
            for tok in candidates:
                bs = self._get_token_bytes(int(tok))
                has_drive = False
                for b in bs:
                    intr = governance.transcribe_byte(int(b)) & 0xFF
                    if (intr & (governance.EXON_FG_MASK | governance.EXON_BG_MASK)) != 0:
                        has_drive = True
                        break
                if has_drive:
                    driven.append(int(tok))
            if driven:
                candidates = driven

        # Pre-gather data for physics computation using full token sequences
        full_masks = np.array([self._get_full_mask(t) for t in candidates], dtype=np.uint8)

        # Successor states after applying all introns of the token (sequence-aware)
        succ_list: list[int] = []
        for tok in candidates:
            s = int(state_index)
            data = self._get_token_bytes(int(tok))
            # Apply each byte's intron in order, matching egress
            for b in data:
                intr = governance.transcribe_byte(int(b)) & 0xFF
                s = int(self.epistemology[s, intr])
            succ_list.append(s)
        succ_indices = np.array(succ_list, dtype=np.int64)

        # Vectorized θ computation
        θ_now = self.s2._theta_table[state_index]
        θ_next = self.s2._theta_table[succ_indices]
        dθ = θ_now - θ_next

        # Fold-entropy computed against current state via Monodromic Fold
        current_mask = self._state_byte_projection(state_index) & 0xFF
        H_before = bin(current_mask).count("1")
        # Apply fold(current_mask, candidate_full_mask) per candidate
        folded_masks = [(governance.fold(current_mask, int(m) & 0xFF)) for m in full_masks]
        H_after = np.array([bin(m & 0xFF).count("1") for m in folded_masks])
        fold_entropy = H_after - H_before

        # Vectorized action values: physics using full sequence path-dependency
        # Prefer proximity to UNA threshold (π/4) rather than collapse to CS.
        una = math.pi / 4.0
        theta_alignment = -np.square(θ_next - una)
        A_physics = theta_alignment + fold_entropy

        # Physically exclude standing emissions (no successor movement) if any
        same_state = (succ_indices == int(state_index))
        if np.any(~same_state):
            A_physics = np.where(same_state, -1e9, A_physics)

        # Deprioritize transitions that collapse to CS (θ_next ~ 0) when viable alternatives exist
        if np.any(θ_next > 1e-3):
            A_physics = np.where(θ_next <= 1e-3, A_physics - 1e6, A_physics)

        # Spectral memory interference: constructive overlap with learned masks
        # for neighborhood representatives. This is physics-native (mask overlap),
        # not a heuristic, and mirrors how learning stores the monodromic fold.
        # Read-only spectral interference evaluated at candidate post-states (rep of succ)
        if neighborhood_reps:
            interference = np.zeros_like(A_physics, dtype=np.float32)
            fm_int = full_masks.astype(int)
            for i, tok in enumerate(candidates):
                succ_idx = int(succ_indices[i])
                rep_succ = succ_idx
                if self.operator.phenomenology_map is not None:
                    rep_succ = int(self.operator.phenomenology_map[succ_idx])
                entry = self.operator.store.get((rep_succ, int(tok)))
                if entry is None:
                    continue
                learned_mask = int(entry.get("mask", 0)) & 0xFF
                if learned_mask:
                    overlap = bin((fm_int[i] & learned_mask) & 0xFF).count("1") / 8.0
                    interference[i] = float(overlap)
            A_physics = A_physics + interference * 2.0

        # Find best candidate preferring successors that advance the state
        order = np.argsort(A_physics)[::-1]
        best_idx = int(order[0])
        for idx in order:
            if int(succ_indices[int(idx)]) != int(state_index):
                best_idx = int(idx)
                break
        best_tok = candidates[best_idx]
        best_A = float(A_physics[best_idx])

        return int(best_tok)


# ---------- Intelligence Engine ----------


class GyroSI:
    """
    S5: Whole System Identity & Policy.

    The outermost viable system boundary that encapsulates the entire VSM stack.
    Manages configuration, agent identity, and provides the stable external API.
    """

    preferences: Dict[str, Any]
    config: AgentConfig
    agent_id: str
    base_path: Path
    engine: IntelligenceEngine

    def __init__(
        self,
        config: AgentConfig,
        agent_id: Optional[str] = None,
        phenotype_store: Optional[Any] = None,
        base_path: Path = Path(__file__).resolve().parents[2],
    ):
        """
        Initialize GyroSI agent.
        All file paths are resolved with respect to base_path unless absolute.
        """
        self.base_path = base_path.resolve()
        self.config = config.copy()
        self.preferences = cast(Dict[str, Any], config.get("preferences", {}))
        # Patch only allowed AgentConfig path keys to be absolute if not already
        if (
            "ontology_path" in self.config
            and self.config["ontology_path"]
            and not os.path.isabs(str(self.config["ontology_path"]))
        ):
            # Resolve ontology path relative to base_path
            self.config["ontology_path"] = str(Path(self.base_path) / str(self.config["ontology_path"]))
        if (
            "knowledge_path" in self.config
            and self.config["knowledge_path"]
            and not os.path.isabs(str(self.config["knowledge_path"]))
        ):
            self.config["knowledge_path"] = str(self.base_path / str(self.config["knowledge_path"]))
        if (
            "public_knowledge_path" in self.config
            and self.config["public_knowledge_path"]
            and not os.path.isabs(str(self.config["public_knowledge_path"]))
        ):
            self.config["public_knowledge_path"] = str(self.base_path / str(self.config["public_knowledge_path"]))
        # Don't modify private_knowledge_path - let PhenotypeStore resolve it with its own base_path
        # if (
        #     "private_knowledge_path" in self.config
        #     and self.config["private_knowledge_path"]
        #     and not os.path.isabs(str(self.config["private_knowledge_path"]))
        # ):
        #     self.config["private_knowledge_path"] = str(self.base_path / str(self.config["private_knowledge_path"]))
        if (
            "phenomenology_map_path" in self.config
            and self.config["phenomenology_map_path"]
            and not os.path.isabs(str(self.config["phenomenology_map_path"]))
        ):
            self.config["phenomenology_map_path"] = str(self.base_path / str(self.config["phenomenology_map_path"]))
        if (
            "private_agents_base_path" in self.config
            and self.config["private_agents_base_path"]
            and not os.path.isabs(str(self.config["private_agents_base_path"]))
        ):
            self.config["private_agents_base_path"] = str(self.base_path / str(self.config["private_agents_base_path"]))
        if "base_path" in self.config and self.config["base_path"] and not os.path.isabs(str(self.config["base_path"])):
            self.config["base_path"] = str(self.base_path / str(self.config["base_path"]))
        if "private_agents_base_path" not in self.config or not self.config["private_agents_base_path"]:
            self.config["private_agents_base_path"] = str(self.base_path / "memories/private/agents")
        # Only assign keys that are valid for AgentConfig
        if "phenomenology_map_path" not in self.config or not self.config["phenomenology_map_path"]:
            onto = self.config.get("ontology_path")
            assert isinstance(onto, str)
            self.config["phenomenology_map_path"] = str(self.base_path / "memories/public/meta/phenomenology_map.npy")
        if "base_path" not in self.config or not self.config["base_path"]:
            self.config["base_path"] = str(self.base_path / "memories")
        self.agent_id = agent_id or str(uuid.uuid4())
        if phenotype_store is None:
            phenotype_store = self._create_default_store()

        # Use local variables for extra paths
        onto = self.config.get("ontology_path")
        if onto is None:
            raise ValueError("ontology_path must be set in config")
        assert isinstance(onto, str)

        # Use epistemology_path from config if provided, otherwise derive from ontology path
        epistemology_path = self.config.get("epistemology_path")
        if epistemology_path is None:
            epistemology_path = str(Path(onto).with_name("epistemology.npy"))
        else:
            epistemology_path = str(epistemology_path)

        self.engine = IntelligenceEngine(
            ontology_path=onto,
            phenotype_store=phenotype_store,
            agent_id=self.agent_id,
            epistemology_path=epistemology_path,
            phenomenology_map_path=self.config["phenomenology_map_path"],
            base_path=self.base_path,
            preferences=self.preferences,  # Pass preferences to the agent
        )

    def ingest(self, data: bytes) -> None:
        """
        Learn from a batch of data using ordered Monodromic Fold.

        This is the primary learning interface. Data is processed as a batch
        with the final state determining the learning context.

        Args:
            data: Bytes to learn from
        """
        self.engine.learning_enabled = True  # Enable learning for ingest
        for b in data:
            self.engine.process_egress(b)

        # Ensure pending writes are flushed
        store = self.engine.operator.store
        if hasattr(store, "commit"):
            store.commit()
        # Removed automatic close()

    def ingest_bulk(self, blob: bytes, *, autoclose: bool = False) -> None:
        """
        Vectorized bulk ingestion for high-performance learning.

        Args:
            blob: Bytes to learn from (processed in bulk)
            autoclose: Whether to commit changes immediately (default: False)
        """
        self.engine.learning_enabled = True  # Enable learning for ingest
        self.engine.process_egress_bulk(blob)

        # Only commit if explicitly requested
        if autoclose:
            store = self.engine.operator.store
            if hasattr(store, "commit"):
                store.commit()

    def respond(self, data: bytes, max_new_tokens: Optional[int] = None) -> bytes:
        """Generate complete response and return as bytes."""
        # 0. let physics ingest the stimulus (no learning while responding)
        self.engine.learning_enabled = False
        self.engine.reset_token_buffer()  # flush any partial token
        self.engine.process_egress_bulk(data)  # the missing call - process stimulus through physics

        # 1. now generate
        result = bytearray()
        for token_bytes in self.respond_stream(max_new_tokens):  # no data needed now
            result.extend(token_bytes)
        return bytes(result)

    def respond_stream(self, max_new_tokens: Optional[int] = None) -> Iterator[bytes]:
        """
        Generate response token by token, yielding each token's bytes as it's generated.
        This enables real streaming without pre-computing the entire response.
        """

        # Disable learning during generation to prevent self-talk
        self.engine.learning_enabled = False

        tokens_done = 0

        while True:
            state_idx = self.engine.current_state_index
            theta = 0.0  # Removed theta tracking
            tok_id, token_bytes = self.engine._emit_token_with_feedback(state_idx, theta)

            # Yield token bytes immediately for real streaming
            yield token_bytes
            tokens_done += 1

            # (1) Physical stop
            if tok_id == SEP_ID:
                break

            # (2) Caller-provided limit (kept working for tests; you can pass None)
            if max_new_tokens is not None and tokens_done >= max_new_tokens:
                break

        self._commit_if_needed()

    def _commit_if_needed(self) -> None:
        store = self.engine.operator.store
        if hasattr(store, "commit"):
            store.commit()

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive agent information.

        Returns:
            Dictionary with agent metadata and state
        """
        state_info = self.engine.get_state_info()

        return {
            **state_info,
            "config": self.config,
        }

    def get_knowledge_store_size(self) -> int:
        """Get the current size of the knowledge store in bytes."""
        store = self.engine.operator.store
        if hasattr(store, "data"):
            return len(store.data)
        return 0

    def add_monitoring_hook(self, hook: CycleHookFunction) -> None:
        """Add a monitoring hook to the intelligence engine."""
        self.engine.add_hook(hook)

    def apply_maintenance(
        self, decay_rate: float = 0.001, confidence_threshold: float = 0.05, max_tokens_per_orbit: int = 64
    ) -> Dict[str, Any]:
        """
        Apply maintenance operations to the knowledge base.

        Args:
            decay_rate: Confidence decay rate for aging entries (small value, e.g. 0.001)
            confidence_threshold: Minimum confidence for entry retention
            max_tokens_per_orbit: Maximum tokens to keep per orbit (orbit entropy management)

        Returns:
            Maintenance report
        """
        # Maintenance operations removed - focus on core physics
        return {"timestamp": time.time()}

    def close(self) -> None:
        """Clean shutdown of the agent."""
        self.engine.operator.store.close()

    def _create_default_store(self) -> Any:
        """Create the default knowledge store for this agent."""
        learn_batch_size = 100
        knowledge_path: Optional[str] = None

        # Check if we should use multi-agent overlay
        if self.config.get("private_knowledge_path"):
            # Multi-agent setup with public/private knowledge
            private_path = self.config.get("private_knowledge_path")
            if private_path is None:
                private_agents_base_path = self.config.get("private_agents_base_path")
                if private_agents_base_path:
                    private_path = os.path.join(str(private_agents_base_path), f"{self.agent_id}/knowledge.bin")
                else:
                    raise ValueError(
                        "private_agents_base_path must be set in config if private_knowledge_path is provided"
                    )
            public_knowledge_path = self.config.get("public_knowledge_path")
            if public_knowledge_path is None:
                raise ValueError("public_knowledge_path must be set in config for multi-agent setup")
            public_store = ReadOnlyView(
                PhenotypeStore(public_knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path)
            )
            private_store = PhenotypeStore(private_path, write_threshold=learn_batch_size, base_path=self.base_path)
            return OverlayView(public_store, private_store)
        else:
            # Single-agent setup
            knowledge_path = self.config.get("knowledge_path")
            if knowledge_path is None:
                # Use preferences if available, otherwise fallback to default
                prefs_dict = self.preferences
                if "public_knowledge" in prefs_dict:
                    knowledge_path = prefs_dict["public_knowledge"]["path"]
                else:
                    base_path = self.config.get("base_path") or str(self.base_path / "memories/public/knowledge")
                    knowledge_path = os.path.join(base_path, "knowledge.bin")  # binary_struct-based fallback path
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)

        if knowledge_path is None:
            raise RuntimeError("Failed to determine knowledge path")
        base_store = PhenotypeStore(knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path)

        # CanonicalView: enable if flag is True, or autodetect if None and file exists
        phenomenology_map_path = self.config.get("phenomenology_map_path")
        # Ensure phenomenology_map_path is always a str before use
        if phenomenology_map_path is None:
            phenomenology_map_path = str(self.base_path / "memories/public/meta/phenomenology_map.npy")
        if self.config.get("enable_phenomenology_storage") or (
            self.config.get("enable_phenomenology_storage") is None
            and phenomenology_map_path
            and os.path.exists(phenomenology_map_path)
        ):
            if phenomenology_map_path and os.path.exists(phenomenology_map_path):
                return CanonicalView(base_store, phenomenology_map_path, base_path=self.base_path)
            # Phenomenology map not found; proceed without canonical view

        return base_store


class LRUAgentCache(OrderedDict[Any, Any]):
    """LRU cache for agent instances with size limit."""

    def __init__(self, max_size: int, *args: Any, **kwargs: Any):
        self.max_size = max_size
        super().__init__(*args, **kwargs)
        self._lock = RLock()

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            # Move to end on access
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self:
                # Update existing
                self.move_to_end(key)
            super().__setitem__(key, value)

            # Evict oldest if over limit
            if len(self) > self.max_size:
                oldest_key = next(iter(self))
                oldest_agent = self[oldest_key]
                oldest_agent.close()  # Clean shutdown
                del self[oldest_key]


class AgentPool:
    """Manages a collection of independent GyroSI agents with eviction policy, using sharded pools for concurrency."""

    SHARD_COUNT = 16  # Power of two for fast masking

    def __init__(
        self,
        ontology_path: str,
        base_knowledge_path: str,
        preferences: Optional[Dict[str, Any]] = None,
        *,
        allowed_ids: Optional[set[str]] = None,
        allow_auto_create: bool = False,
        private_agents_base_path: Optional[str] = None,
        base_path: Path = Path(__file__).resolve().parents[2],
        meta_dir: Optional[str] = None,
    ):
        self.base_path = base_path.resolve()
        self.meta_dir = meta_dir
        if private_agents_base_path is None:
            private_agents_base_path = str(self.base_path / "memories/private/agents")
        self.private_agents_base_path = private_agents_base_path
        self.ontology_path = _abs(
            ontology_path if ontology_path is not None else "memories/public/meta/ontology_keys.npy", self.base_path
        )
        self.base_knowledge_path = _abs(base_knowledge_path, self.base_path)
        self.preferences = preferences or {}
        # cast preferences to a dict so .get is allowed
        self.max_agents = self.preferences.get("max_agents_in_memory", 1000)
        self.eviction_policy = self.preferences.get("agent_eviction_policy", "lru")

        # Initialize TTL settings
        self.agent_ttl_minutes = 0
        if self.eviction_policy == "ttl":
            self.agent_ttl_minutes = self.preferences.get("agent_ttl_minutes", 30)

        self.allowed_ids = allowed_ids or {"user", "system", "assistant"}
        self.allow_auto_create = allow_auto_create
        self.agent_access_times: Dict[str, float] = {}
        self.agent_created_at: Dict[str, float] = {}
        self._shards: List[Dict[str, Any]] = []
        for _ in range(self.SHARD_COUNT):
            if self.eviction_policy == "lru":
                shard_size = max(1, self.max_agents // self.SHARD_COUNT)
                self._shards.append({"agents": LRUAgentCache(shard_size), "lock": RLock()})
            else:
                self._shards.append({"agents": OrderedDict(), "lock": RLock()})
        # Check if we should use multi-knowledge view
        from baby.policies import create_multi_knowledge_view

        # Check if there are multiple knowledge_*.bin files in the directory
        knowledge_dir = os.path.dirname(self.base_knowledge_path)
        multi_view = create_multi_knowledge_view(knowledge_dir, "knowledge_*.bin", self.base_path)

        if multi_view.stores:  # If we found multiple knowledge files
            # Informational: using multiple knowledge files
            base_store: Any = multi_view
        else:
            # Fall back to single file
            single_store: Any = PhenotypeStore(self.base_knowledge_path, write_threshold=100, base_path=self.base_path)
            base_store = single_store

        self._public_store: Optional[ReadOnlyView] = ReadOnlyView(
            CanonicalView(
                base_store,
                phenomenology_map_path=str(Path(self.ontology_path).with_name("phenomenology_map.npy")),
                base_path=self.base_path,
            )
        )

    def _shard_index(self, agent_id: str) -> int:
        # Use hash for even distribution; mask for power-of-two
        return (hash(agent_id) if isinstance(agent_id, str) else agent_id) & (self.SHARD_COUNT - 1)

    def get_or_create_agent(self, agent_id: str, role_hint: Optional[str] = None) -> "GyroSI":
        if not self.allow_auto_create and agent_id not in self.allowed_ids:
            raise PermissionError(f"Auto-create disabled and agent_id '{agent_id}' is not allowed.")
        return self._get_or_create(agent_id, role_hint)

    # NEW: internal impl so we can reuse it
    def _get_or_create(self, agent_id: str, role_hint: Optional[str]) -> "GyroSI":
        idx = self._shard_index(agent_id)
        shard = self._shards[idx]
        with shard["lock"]:
            agents = shard["agents"]
            # Update access time for TTL-based eviction
            if self.eviction_policy == "ttl":
                now_mono = time.monotonic()
                self.agent_access_times[agent_id] = now_mono
                self._evict_expired_agents()

            if agent_id not in agents:
                # Check if we need to evict before creating new
                if not isinstance(agents, LRUAgentCache):
                    self._maybe_evict_agent(shard)

                # Create private knowledge path
                private_path_rel = os.path.join(self.private_agents_base_path, f"{agent_id}/knowledge.bin")
                private_path = _abs(private_path_rel, self.base_path)
                os.makedirs(os.path.dirname(private_path), exist_ok=True)

                # Multi-agent overlay using decorators, reuse cached public store
                public_store = self._public_store
                private_store = PhenotypeStore(private_path, write_threshold=100, base_path=self.base_path)
                agent_store: Any = OverlayView(public_store, private_store)

                # Create agent config
                config: AgentConfig = {
                    "ontology_path": self.ontology_path,
                    "public_knowledge_path": self.base_knowledge_path,
                    "private_knowledge_path": private_path,
                    "phenomenology_map_path": str(
                        self.preferences.get("phenomenology_map_path")
                        or str(Path(self.ontology_path).with_name("phenomenology_map.npy"))
                    ),
                    "private_agents_base_path": str(self.private_agents_base_path),
                    "base_path": str(self.base_path),
                    "enable_phenomenology_storage": bool(self.preferences.get("enable_phenomenology_storage", False)),
                    "preferences": cast(PreferencesConfig, self.preferences),  # Pass preferences to the agent
                }

                # Add role hint to metadata if provided
                if role_hint:
                    config["agent_metadata"] = {"role_hint": role_hint}

                agents[agent_id] = GyroSI(
                    config=config,
                    agent_id=agent_id,
                    phenotype_store=agent_store,
                    base_path=self.base_path,
                )
                # Track creation time for TTL eviction
                if self.eviction_policy == "ttl":
                    now_mono = time.monotonic()
                    self.agent_created_at[agent_id] = now_mono

            return cast(GyroSI, agents[agent_id])

    # NEW: get without creating
    def get(self, agent_id: str) -> "GyroSI":
        idx = self._shard_index(agent_id)
        shard = self._shards[idx]
        with shard["lock"]:
            agents = shard["agents"]
            if agent_id not in agents:
                raise KeyError(f"Agent '{agent_id}' not found in pool.")
            return cast(GyroSI, agents[agent_id])

    # NEW: explicit creation API (ignores allowed_ids, but still respects base path etc.)
    def create_agent(self, agent_id: str, role_hint: Optional[str] = None) -> "GyroSI":
        onto = self.ontology_path
        phenomap = str(Path(onto).with_name("phenomenology_map.npy"))
        private_rel = os.path.join(self.private_agents_base_path, f"{agent_id}/knowledge.bin")
        private_path = _abs(private_rel, self.base_path)
        config: AgentConfig = {
            "ontology_path": onto,
            "public_knowledge_path": self.base_knowledge_path,
            "private_knowledge_path": private_path,
            "phenomenology_map_path": str(phenomap),
            "private_agents_base_path": str(self.private_agents_base_path),
            "base_path": str(self.base_path),
            "enable_phenomenology_storage": bool(self.preferences.get("enable_phenomenology_storage", False)),
        }
        return GyroSI(config, agent_id=agent_id, base_path=self.base_path)

    # NEW: bootstrap trio
    def ensure_triad(self) -> None:
        for aid, role in (("user", "user"), ("system", "system"), ("assistant", "assistant")):
            if aid not in self.get_active_agents():
                self._get_or_create(aid, role)

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent from pool.

        Args:
            agent_id: Agent to remove

        Returns:
            True if agent was found and removed
        """
        idx = self._shard_index(agent_id)
        shard = self._shards[idx]
        with shard["lock"]:
            agents = shard["agents"]
            if agent_id in agents:
                agents[agent_id].close()
                del agents[agent_id]
                if self.eviction_policy != "lru" and agent_id in self.agent_access_times:
                    del self.agent_access_times[agent_id]
                return True
            return False

    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        result = []
        for shard in self._shards:
            with shard["lock"]:
                result.extend(list(shard["agents"].keys()))
        return result

    def close_all(self) -> None:
        """Close all agents in the pool."""
        for shard in self._shards:
            with shard["lock"]:
                for agent in shard["agents"].values():
                    agent.close()
                shard["agents"].clear()
            if self.eviction_policy != "lru":
                self.agent_access_times.clear()
        # Only close the public store once here
        if hasattr(self, "_public_store") and self._public_store is not None:
            self._public_store.close()
            self._public_store = None

    def _maybe_evict_agent(self, shard: Dict[str, Any]) -> None:
        """Evict agent if at capacity (non-LRU policies) for a shard."""
        agents = shard["agents"]
        if len(agents) >= self.max_agents // self.SHARD_COUNT:
            if self.eviction_policy == "lfu":
                oldest_id = next(iter(agents))
            elif self.eviction_policy == "ttl":
                oldest_id = min(self.agent_access_times, key=lambda k: self.agent_access_times[k])
            else:
                oldest_id = next(iter(agents))
            self.remove_agent(oldest_id)

    def _evict_expired_agents(self) -> None:
        """Evict agents that have exceeded their TTL."""
        if not self.agent_ttl_minutes:
            return
        now = time.monotonic()
        ttl = self.agent_ttl_minutes * 60.0

        for shard in self._shards:
            with shard["lock"]:
                expired = []
                for agent_id in list(shard["agents"].keys()):
                    last = self.agent_access_times.get(agent_id, self.agent_created_at.get(agent_id, now))
                    if now - last > ttl:
                        expired.append(agent_id)

                for agent_id in expired:
                    try:
                        agent = shard["agents"].pop(agent_id)
                        if hasattr(agent, "close"):
                            agent.close()
                    except Exception:
                        pass
                    finally:
                        self.agent_access_times.pop(agent_id, None)
                        self.agent_created_at.pop(agent_id, None)


def orchestrate_turn(
    pool: AgentPool,
    user_id: str,
    assistant_id: str,
    user_input: str,
    tokenizer_name: str,  # Make this mandatory
    max_new_tokens: Optional[int] = None,  # No artificial limit
) -> str:
    """
    Orchestrate a single conversational turn between agents using a tokenizer.

    Args:
        pool: Agent pool containing the agents
        user_id: User agent identifier
        assistant_id: Assistant agent identifier
        user_input: User's input text
        tokenizer_name: Name of the tokenizer to use (e.g., "bert-base-uncased")

    Returns:
        Assistant's response text
    """
    try:
        user_agent = pool.get(user_id)
        assistant_agent = pool.get(assistant_id)
    except KeyError as e:
        raise RuntimeError(f"Missing required agent: {e}. Call pool.ensure_triad() or pool.create_agent() first.")

    # 1. Encode input using the specified tokenizer. No fallback.
    from baby.information import encode_text, decode_text, sep_bytes

    in_bytes = encode_text(user_input, name=tokenizer_name)

    # 2. User agent learns from the input (no generation)
    user_agent.ingest_bulk(in_bytes)

    # Physically mark end-of-user-turn to align the assistant's physics
    stimulus = in_bytes + sep_bytes()

    # 3. Assistant responds to stimulus
    response = assistant_agent.respond(stimulus, max_new_tokens=max_new_tokens)

    # 4. Decode response using the same tokenizer.
    #    The `decode` function already has a UTF-8 fallback for robustness.
    result = decode_text(response, name=tokenizer_name)
    return result


def stream_turn(
    pool: AgentPool,
    user_id: str,
    assistant_id: str,
    user_input: str,
    tokenizer_name: str,
    max_new_tokens: Optional[int] = None,
) -> Iterator[bytes]:
    """Stream a response token-by-token as raw bytes.

    This mirrors `orchestrate_turn` but yields each token's bytes for SSE.
    It performs the same steps:
    - encode user text
    - learn on the user agent
    - append SEP to form the stimulus
    - prime assistant physics without learning
    - stream tokens from the assistant
    """
    try:
        user_agent = pool.get(user_id)
        assistant_agent = pool.get(assistant_id)
    except KeyError as e:
        raise RuntimeError(f"Missing required agent: {e}. Call pool.ensure_triad() or pool.create_agent() first.")

    from baby.information import encode_text, sep_bytes

    # Encode and learn on user
    in_bytes = encode_text(user_input, name=tokenizer_name)
    user_agent.ingest_bulk(in_bytes)

    # Prime assistant physics without learning
    stimulus = in_bytes + sep_bytes()
    eng = assistant_agent.engine
    eng.learning_enabled = False
    eng.reset_token_buffer()
    eng.process_egress_bulk(stimulus)

    # Stream tokens (derive default length if needed)
    if max_new_tokens is None:
        try:
            tp = cast(Dict[str, Any], pool.preferences.get("turn_policy", {}))
            max_new_tokens = int(tp.get("max_reply_tokens", 96))
        except Exception:
            max_new_tokens = 96
    yield from assistant_agent.respond_stream(max_new_tokens=max_new_tokens)
