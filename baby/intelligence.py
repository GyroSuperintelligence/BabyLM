"""
S5: Intelligence - Semantic Memory & Learning

This module provides the IntelligenceEngine class responsible for
coordinating the entire GyroSI learning process.
"""

import os
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import RLock
import math
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, TypedDict, cast, Iterator, Tuple

import numpy as np
from numpy.typing import NDArray

from baby import governance
from baby.contracts import AgentConfig, CycleHookFunction, PreferencesConfig, PhenotypeEntry
from baby.inference import InferenceEngine
from baby.information import InformationEngine, SEP_ID
from baby.policies import CanonicalView, OrbitStore, OverlayView, ReadOnlyView



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
    _θ_buf: Deque[float]
    _θ_low: float
    _θ_high: float
    _cand_cache: Dict[int, List[Tuple[int, float, int]]]
    _cand_cache_limit: int
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

        # --- theta tracking --------------------------------------------------
        self._θ_buf: Deque[float] = deque(maxlen=8)
        self._θ_low = float(cast(float, self.preferences.get("theta_low", 0.05)))
        self._θ_high = float(cast(float, self.preferences.get("theta_high", 0.6)))

        # --- auto-prune setup -----------------------------------------------
        self._register_auto_prune_hook()

        # --- vectorized epistemology buffer ----------------------------------
        # Reusable buffer for state trajectory computation (max 64K to prevent RAM explosion)
        self._state_buf = np.empty(65536, dtype=np.int32)

        # --- candidate cache for O(1) lookup ----------------------------------
        self._cand_cache: Dict[int, List[Tuple[int, float, int]]] = {}
        self._cand_cache_limit = 65536
        self._store_mutation_epoch = 0

    def _temperature_from_theta(self, theta: float) -> float:
        """
        Smooth sigmoid temperature schedule.
        Maps theta from [0, π] to temperature [floor, cap].
        """
        temp_floor = float(cast(float, self.preferences.get("temperature_floor", 0.1)))
        temp_cap = float(cast(float, self.preferences.get("temperature_cap", 1.0)))

        # Sigmoid centered at θ_low with steep transition
        sigmoid = 1.0 / (1.0 + math.exp(-(theta - self._θ_low) * 10))

        return temp_floor + (temp_cap - temp_floor) * sigmoid

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
        """Select one token, emit its full LEB128 byte sequence, feed each byte through Egress.
        Returns (token_id, emitted_bytes)."""
        temperature = self._temperature_from_theta(theta)

        # Optional: very noisy; never on by default to avoid stdout stalls
        if getattr(self, "debug_mode", False):
            # Print only the first few times per request/agent to avoid floods
            if not hasattr(self, "_dbg_seen"):
                self._dbg_seen = 0
            if self._dbg_seen < 3:
                self._dbg_seen += 1
                self.debug_candidates(state_idx)

        tok_id = self.generate_token_exon(state_idx, temperature)
        from baby.information import token_id_to_bytes

        token_bytes = token_id_to_bytes(tok_id)
        # Use bulk application instead of per-byte feedback
        if not hasattr(self, "_probe_emit"):
            self._probe_emit = 0
        t0 = time.perf_counter()
        self.process_egress_bulk(token_bytes)
        t1 = time.perf_counter()
        if self._probe_emit < 5:
            self._probe_emit += 1
            print(f"[probe] feedback bulk for {len(token_bytes)} bytes took {(t1-t0)*1e3:.2f} ms")
        return tok_id, token_bytes

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
        self.current_state_index = self.epistemology[self.current_state_index, intron]
        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
        self._cached_state_int = self.gene_mac_m_int

        assert self.gene_mac_m_int < (1 << 48)
        self.cycle_count += 1

        div = self.s2.measure_state_divergence_index(self.current_state_index)
        self._θ_buf.append(div)

        if (intron & 0x80) == 0:  # Token complete
            try:
                tok_bytes = bytes(b ^ 0xAA for b in self._raw_leb_buf)
                token_id = TOK.bytes_to_id(tok_bytes)

                if token_id == SEP_ID:
                    # Learn SEP as a pre-state association so it can be chosen later
                    self.operator.learn_token_preonly(token_id, pre_state_index, intron)
                    # Drop cached candidates for this representative state
                    rep = (
                        int(self.phenomenology_map[pre_state_index])
                        if self.phenomenology_map is not None
                        else pre_state_index
                    )
                    self._cand_cache.pop(rep, None)
                    self._last_token_id = token_id
                    self._raw_leb_buf.clear()
                    return intron
                self._prev_token_id = token_id

                # Learn with pre-only storage
                self.operator.learn_token_preonly(token_id, pre_state_index, intron)
                # Drop cached candidates for this representative state
                rep = (
                    int(self.phenomenology_map[pre_state_index])
                    if self.phenomenology_map is not None
                    else pre_state_index
                )
                self._cand_cache.pop(rep, None)

                # Get the pre-state entry for hooks
                pre_entry = self.operator.store.get((pre_state_index, token_id))
                for hook in self.post_cycle_hooks:
                    try:
                        hook(self, cast(PhenotypeEntry, pre_entry), intron, token_id, pre_state_index)
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
        import numpy as np

        if not blob:
            return

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
        st[0] = self.current_state_index

        # Bounds checking to prevent out-of-bounds access
        epistemology_size = ep.shape[0]
        if st[0] >= epistemology_size:
            raise RuntimeError(
                f"Initial state index {st[0]} is out of bounds for epistemology matrix (size {epistemology_size})"
            )

        # True vectorization: compute all state transitions in one operation
        # Add bounds checking for each transition
        for i in range(1, n):
            prev_state = st[i - 1]
            intron = introns[i - 1]
            if prev_state >= epistemology_size:
                raise RuntimeError(
                    f"State index {prev_state} is out of bounds for epistemology matrix (size {epistemology_size})"
                )
            new_state = ep[prev_state, intron]
            if new_state >= epistemology_size:
                raise RuntimeError(
                    f"Transition to state {new_state} is out of bounds "
                f"for epistemology matrix (size {epistemology_size})"
            )
            st[i] = new_state

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
                        self.operator.learn_token_preonly(token_id, pre_state, int(intron))
                        # Drop cached candidates for this representative state
                        rep = (
                            int(self.phenomenology_map[pre_state]) if self.phenomenology_map is not None else pre_state
                        )
                        self._cand_cache.pop(rep, None)
                        self._last_token_id = token_id
                        self._raw_leb_buf.clear()
                        token_start_idx = i + 1
                        continue
                    self._prev_token_id = token_id

                    # Get pre-state (where token started)
                    pre_state = int(st[token_start_idx])

                    # Pre-only storage
                    self.operator.learn_token_preonly(token_id, pre_state, int(intron))
                    # Drop cached candidates for this representative state
                    rep = int(self.phenomenology_map[pre_state]) if self.phenomenology_map is not None else pre_state
                    self._cand_cache.pop(rep, None)

                    # Get the pre-state entry for hooks
                    pre_entry = self.operator.store.get((pre_state, token_id))
                    for hook in self.post_cycle_hooks:
                        try:
                            hook(self, cast(PhenotypeEntry, pre_entry), int(intron), token_id, pre_state)
                        except Exception:
                            pass

                    self._last_token_id = token_id
                except Exception:
                    pass
                finally:
                    self._raw_leb_buf.clear()
                    token_start_idx = i + 1  # Next token starts here

        # --- 4. persist new head state + θ updates
        if n > 0:
            final_pre_state = int(st[n - 1])  # state *before* applying the last intron
            final_intron = int(introns[-1])
            if final_pre_state >= epistemology_size:
                raise RuntimeError(
                    f"Final pre-state index {final_pre_state} is out of bounds "
                    f"for epistemology matrix (size {epistemology_size})"
                )
            new_state = self.epistemology[final_pre_state, final_intron]
            if new_state >= epistemology_size:
                raise RuntimeError(
                    f"Final transition to state {new_state} is out of bounds "
                    f"for epistemology matrix (size {epistemology_size})"
                )
            self.current_state_index = new_state
            # keep state integer in sync with index
            try:
                self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
                self._cached_state_int = self.gene_mac_m_int
            except Exception:
                pass

            # θ update: push at least the final divergence (cheap and keeps temperature sane)
            try:
                div = self.s2.measure_state_divergence_index(self.current_state_index)
                self._θ_buf.append(div)
            except Exception:
                pass

            # OPTIONAL: for very long chunks, add a mid-sample to smooth temperature changes
            if n > 1024:
                mid_i = n // 2
                try:
                    mid_state = int(st[mid_i])
                    div_mid = self.s2.measure_state_divergence_index(mid_state)
                    self._θ_buf.append(div_mid)
                except Exception:
                    pass

    def process_ingress(self) -> tuple[int, int]:
        """Emit exactly one token and return (last_byte, last_intron) of that token."""
        state_idx = self.current_state_index
        theta = self._θ_buf[-1] if self._θ_buf else 0.0
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
        info: StateInfo = {
            "agent_id": agent_id,
            "cycle_count": cycle_count,
            "state_integer": state_integer,
            "tensor_index": tensor_index_val,
            "angular_divergence_radians": angular_divergence_radians,
            "angular_divergence_degrees": angular_divergence_degrees,
            "active_hooks": active_hooks,
        }
        return info

    def reset_to_archetypal_state(self) -> None:
        """Reset agent to the archetypal state (GENE_Mac_S)."""
        self.gene_mac_m_int = self.s2.tensor_to_int(governance.GENE_Mac_S)
        self._sync_index_from_state_int()
        self.cycle_count = 0

    def _sync_state_fields_from_index(self) -> None:
        self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
        self._cached_state_int = self.gene_mac_m_int

    def _sync_index_from_state_int(self) -> None:
        self.current_state_index = self.s2.get_index_from_state(self.gene_mac_m_int)

    def validate_knowledge_integrity(self) -> bool:
        """
        Validate the integrity of the knowledge base.

        Returns:
            True if integrity is maintained, False otherwise
        """
        result = self.operator.validate_knowledge_integrity()
        if isinstance(result, bool):
            return result
        if hasattr(result, "success"):
            return bool(getattr(result, "success"))
        # If result is a Mapping (e.g., ValidationReport TypedDict), treat as True if total_entries > 0
        try:
            from collections.abc import Mapping

            if isinstance(result, Mapping):
                # mypy requires the cast on the same line as .get
                return bool(cast(Mapping[str, Any], result).get("total_entries", 0) > 0)
        except ImportError:
            pass
        return False

    def apply_confidence_decay(self, decay_rate: float = 0.001) -> Dict[str, Any]:
        result = self.operator.apply_confidence_decay(decay_rate)
        try:
            return dict(result)
        except Exception:
            if is_dataclass(result) and not isinstance(result, type):
                return asdict(result)
            if hasattr(result, "__dict__"):
                return cast(Dict[str, Any], vars(result))
            return {}



    def debug_candidates(self, state_index: int) -> None:
        """Debug candidate generation for a given state"""
        rep = state_index
        if self.phenomenology_map is not None:
            rep = int(self.phenomenology_map[state_index])

        ks_rep = list(self.operator.store.iter_keys_for_state(rep))[:10]
        ks_raw = list(self.operator.store.iter_keys_for_state(state_index))[:10]

        print(f"[cand] state={state_index} rep={rep} ks_rep={len(ks_rep)} ks_raw={len(ks_raw)}")
        if ks_rep:
            print("[cand] sample entry:", self.operator.store.get(ks_rep[0]))

    def generate_token_exon(self, state_index: int, temperature: float = 1.0) -> int:
        """
        Generate next token using physics-driven exon product sieve.
        
        Implements the core generation algorithm:
        1. Get baseline exon product from state physics
        2. Specialize with learned masks (if any exist)
        3. Use exon product to generate candidate intron bytes
        4. Find tokens matching those introns via tokenizer trie
        5. Score and sample based on physics-derived confidence

        Args:
            state_index: Current state index
            temperature: Generation temperature (0.0 = deterministic, 1.0 = random)

        Returns:
            Generated token_id
        """
        # Get physics parameters for this state
        try:
            theta = self.s2.measure_state_divergence_index(state_index)
            orbit_size = self.s2.get_orbit_cardinality(state_index)
        except (AttributeError, IndexError):
            # Fallback values if physics data unavailable
            theta = 0.5
            orbit_size = 100
        
        # Step 1: Get baseline exon product from state physics
        baseline_exon_product = governance.exon_product_from_state(state_index, theta, orbit_size)
        
        # Step 2: Check for learned specializations for this state
        specialized_candidates = []
        
        # Look for learned entries that override the baseline
        for s_idx, token_id in getattr(self.operator.store, "iter_keys_for_state", lambda _s: [])(state_index):
            entry = self.operator.store.get((s_idx, token_id))
            if entry and "mask" in entry:
                # Use learned mask instead of baseline
                learned_mask = entry["mask"]
                specialized_candidates.append((token_id, learned_mask))
        
        # Step 3: Generate candidate intron bytes using exon product sieve
        if specialized_candidates:
            # Use specialized masks for known tokens
            candidate_tokens = []
            for token_id, mask in specialized_candidates:
                # Convert mask to intron candidates
                intron_candidates = governance.propose_resonant_introns(mask)
                
                # Verify this token's last intron matches one of our candidates
                token_last_intron = governance.token_last_intron(token_id)
                if token_last_intron in intron_candidates:
                    # Score based on physics alignment
                    confidence = self._compute_runtime_confidence(state_index, mask, theta, orbit_size)
                    resonance = self._calculate_resonance(state_index, mask)
                    score = confidence * 0.6 + resonance * 0.4
                    candidate_tokens.append((token_id, score, mask))
        else:
            # No learned specializations, use baseline exon product
            intron_candidates = governance.propose_resonant_introns(baseline_exon_product)
            candidate_tokens = self._find_tokens_by_intron_sieve(intron_candidates, state_index, baseline_exon_product)
        
        if not candidate_tokens:
            # No physics-aligned candidates found, emit SEP to end turn
            return SEP_ID
        
        # Step 4: Score and sample based on temperature
        candidate_tokens.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        
        if temperature < 0.1:
            # Deterministic: return highest-scoring token
            return int(candidate_tokens[0][0])
        else:
            # Probabilistic sampling
            scores = np.array([score for _, score, _ in candidate_tokens])
            
            # Apply temperature scaling with softmax
            if len(scores) > 1:
                scores = scores - scores.min()  # Normalize to prevent overflow
                scaled_scores = scores / temperature
                exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
                probs = exp_scores / np.sum(exp_scores)
                
                # Sample from distribution
                chosen_idx = int(np.random.choice(len(candidate_tokens), p=probs))
                return int(candidate_tokens[chosen_idx][0])
            else:
                return int(candidate_tokens[0][0])

    def _compute_runtime_confidence(self, state_index: int, mask: int, theta: float, orbit_size: int) -> float:
        """
        Compute confidence at runtime from physics parameters.
        
        Replaces stored confidence with physics-based calculation using
        theta divergence, orbit cardinality, and mask entropy.
        """
        # Theta factor: higher divergence reduces confidence
        theta_factor = max(0.1, 1.0 - (theta / math.pi))
        
        # Orbit factor: larger orbits provide more stability
        max_orbit_size = getattr(self.s2, "_v_max", 1000)
        orbit_factor = min(1.0, orbit_size / max_orbit_size)
        
        # Mask entropy: more diverse masks indicate higher information content
        mask_entropy = bin(mask).count("1") / 8.0
        
        # Combine factors
        confidence = (theta_factor * 0.4) + (orbit_factor * 0.3) + (mask_entropy * 0.3)
        return min(1.0, max(0.1, confidence))
    
    def _find_tokens_by_intron_sieve(self, intron_candidates: List[int], state_index: int, exon_product: int) -> List[Tuple[int, float, int]]:
        """
        Find tokens whose last intron matches one of the candidate introns.
        
        Uses the tokenizer trie lookup functionality from information module.
        """
        from baby.information import find_tokens_by_last_intron
        
        candidate_tokens = []
        
        for intron in intron_candidates:
            # Find tokens matching this intron
            matching_tokens = find_tokens_by_last_intron(intron)
            
            for token_id in matching_tokens:
                # Score based on physics alignment
                confidence = self._compute_runtime_confidence(state_index, exon_product, 0.5, 100)
                resonance = self._calculate_resonance(state_index, exon_product)
                score = confidence * 0.6 + resonance * 0.4
                candidate_tokens.append((token_id, score, exon_product))
        
        return candidate_tokens

    def _calculate_resonance(self, state_index: int, mask: int) -> float:
        """Calculate token-level resonance between state and phenotype mask."""
        # Cache key for resonance calculation
        cache_key = (state_index, mask)

        # Check if we have a resonance cache
        if not hasattr(self, "_resonance_cache"):
            self._resonance_cache: Dict[Tuple[int, int], float] = {}

        # Return cached result if available
        if cache_key in self._resonance_cache:
            return self._resonance_cache[cache_key]

        # Use stable 8-bit projection for resonance calculation
        low_byte = self._state_byte_projection(state_index)
        hd = bin((low_byte ^ (mask & 0xFF)) & 0xFF).count("1")
        base_resonance = 1.0 - (hd / 8.0)

        # Add orbit-based resonance if phenomenology is available
        if hasattr(self, "s2") and hasattr(self.s2, "orbit_cardinality"):
            try:
                # Use representative state index for orbit factor calculation
                rep_state_idx = state_index
                if self.phenomenology_map is not None:
                    rep_state_idx = int(self.phenomenology_map[state_index])
                orbit_size = self.s2.orbit_cardinality[rep_state_idx]
                max_orbit_size = getattr(self.s2, "_v_max", int(np.max(self.s2.orbit_cardinality))) or 1
                orbit_factor = min(1.0, orbit_size / max_orbit_size)
                # Combine base resonance with orbit factor
                result = (base_resonance * 0.7) + (orbit_factor * 0.3)
            except (IndexError, AttributeError):
                result = base_resonance
        else:
            result = base_resonance

        # Cache the result (limit cache size to prevent memory issues)
        if len(self._resonance_cache) < 10000:  # Max 10K cached results
            self._resonance_cache[cache_key] = result

        return float(result)










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
        # Don't modify private_knowledge_path - let OrbitStore resolve it with its own base_path
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
        self.engine.process_egress_bulk(blob)

        # Only commit if explicitly requested
        if autoclose:
            store = self.engine.operator.store
            if hasattr(store, "commit"):
                store.commit()

    def respond(self, data: bytes, max_new_tokens: Optional[int] = None) -> bytes:
        """Generate complete response and return as bytes."""
        result = bytearray()
        for token_bytes in self.respond_stream(data, max_new_tokens):
            result.extend(token_bytes)
        return bytes(result)

    def respond_stream(self, data: bytes, max_new_tokens: Optional[int] = None) -> Iterator[bytes]:
        """
        Generate response token by token, yielding each token's bytes as it's generated.
        This enables real streaming without pre-computing the entire response.
        """
        import time
        import numpy as np
        from baby.information import bytes_to_token_ids, decode_text, token_id_to_bytes, SEP_ID

        # 1) Ingest prompt bytes
        self.engine.process_egress_bulk(data)

        # Policy knobs (not "caps")
        prefs = self.preferences if isinstance(self.preferences, dict) else {}
        turn_cfg = prefs.get("turn_policy", {})
        ratio = float(turn_cfg.get("reply_len_ratio", 1.2))
        min_reply = int(turn_cfg.get("min_reply_tokens", 8))
        max_reply = int(turn_cfg.get("max_reply_tokens", 96))
        wall_s = float(turn_cfg.get("max_wall_time_s", 2.0))
        theta_eps = float(turn_cfg.get("theta_std_epsilon", 0.01))
        tok_name = prefs.get("tokenizer", {}).get("name", "bert-base-uncased")

        in_tok_len = 1
        try:
            in_tok_len = max(1, len(bytes_to_token_ids(data)))
        except Exception:
            pass
        target = int(np.clip(int(ratio * in_tok_len + 2), min_reply, max_reply))

        tokens_done = 0
        t0 = time.perf_counter()

        while True:
            state_idx = self.engine.current_state_index
            theta = self.engine._θ_buf[-1] if self.engine._θ_buf else 0.0
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

            # (3) Natural stop: enough tokens AND the text looks complete
            if tokens_done >= target:
                try:
                    # Decode current token to check for completion
                    token_text = decode_text(token_bytes, name=tok_name)
                    if token_text and token_text[-1:] in ".!?…\n":
                        break
                except Exception:
                    pass
                # Respect physics: force a SEP if we must stop
                sep = token_id_to_bytes(SEP_ID)
                self.engine.process_egress_bulk(sep)
                yield sep
                break

            # (4) Soft time guard (for your 2015 MBP edge case)
            if time.perf_counter() - t0 > wall_s:
                sep = token_id_to_bytes(SEP_ID)
                self.engine.process_egress_bulk(sep)
                yield sep
                break

            # (5) θ-settling (state stopped changing meaningfully)
            if len(self.engine._θ_buf) >= 8 and tokens_done >= min_reply:
                try:
                    import numpy as _np

                    rec = _np.array(list(self.engine._θ_buf)[-8:], dtype=_np.float32)
                    if float(rec.std()) < theta_eps:
                        sep = token_id_to_bytes(SEP_ID)
                        self.engine.process_egress_bulk(sep)
                        yield sep
                        break
                except Exception:
                    pass

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
            "system_integrity": self.engine.validate_knowledge_integrity(),
        }

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
        # Apply confidence decay
        decay_report = self.engine.apply_confidence_decay(decay_rate)

        # Manage orbit entropy (replaces confidence-based pruning)
        orbit_entropy_removed = self.engine.operator.manage_orbit_entropy(max_tokens_per_orbit)

        # Commit changes to persist maintenance operations
        store = self.engine.operator.store
        if hasattr(store, "commit"):
            store.commit()

        return {"decay_applied": decay_report, "orbit_entropy_removed": orbit_entropy_removed, "timestamp": time.time()}

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
                OrbitStore(public_knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path)
            )
            private_store = OrbitStore(private_path, write_threshold=learn_batch_size, base_path=self.base_path)
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
        base_store = OrbitStore(knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path)

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
            else:
                print(f"Warning: phenomenology map not found at {phenomenology_map_path}")
                pass

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
            print(f"Using MultiKnowledgeView with {len(multi_view.stores)} knowledge files")
            base_store: Any = multi_view
        else:
            # Fall back to single file
            print(f"Using single knowledge file: {os.path.basename(self.base_knowledge_path)}")
            single_store: Any = OrbitStore(self.base_knowledge_path, write_threshold=100, base_path=self.base_path)
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
                private_store = OrbitStore(private_path, write_threshold=100, base_path=self.base_path)
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
                    except Exception as e:
                        print(f"Warning: Failed to evict agent {agent_id}: {e}")
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
