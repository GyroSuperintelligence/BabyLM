"""
S5: Intelligence - Semantic Memory & Learning

This module provides the IntelligenceEngine class responsible for
coordinating the entire GyroSI learning process.
"""

import math
import os
import random
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import asdict, is_dataclass
from functools import cached_property
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, TypedDict, cast

import numpy as np

from baby import governance
from baby.contracts import AgentConfig, CycleHookFunction, PreferencesConfig, PhenotypeEntry
from baby.inference import InferenceEngine
from baby.information import InformationEngine
from baby.policies import CanonicalView, OrbitStore, OverlayView, ReadOnlyView
from toys.communication import tokenizer as tokmod


class _TokBridge:
    @cached_property
    def mod(self) -> Any:
        from toys.communication import tokenizer as _tok

        return _tok

    def id_to_bytes(self, tok_id: int) -> bytes:
        return cast(bytes, self.mod.id_to_bytes(tok_id))

    def bytes_to_id(self, bs: bytes) -> int:
        return cast(int, self.mod.bytes_to_id(bs))

    def bytes_to_ids(self, bs: bytes) -> List[int]:
        return cast(List[int], self.mod.bytes_to_ids(bs))


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

    def __init__(
        self,
        ontology_path: str,
        phenotype_store: Any,
        agent_id: Optional[str] = None,
        hook_batch_interval: int = 8,
        epistemology_path: Optional[str] = None,
        phenomenology_map_path: Optional[str] = None,
        base_path: Path = Path(__file__).resolve().parents[2],
        preferences: Optional[PreferencesConfig] = None,
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
        # Buffer for emission of the current token as *unmasked introns*
        self._emit_buf: list[int] = []

        # --- epistemology setup ------------------------------------------------
        self.use_epistemology = epistemology_path is not None
        if self.use_epistemology:
            # For epistemology paths, always resolve relative to project root
            project_root = Path(__file__).resolve().parents[2] / "BabyLM"
            epistemology_path = _abs(epistemology_path, project_root)
            self.epistemology = np.load(epistemology_path)
            self.current_state_index = 0  # Will be set to archetypal state after s2 is created
        else:
            self.epistemology = None
            self.current_state_index = 0  # Always initialize to 0, not None

        # --- phenomenology setup ----------------------------------------------
        self.use_phenomenology = phenomenology_map_path is not None
        if self.use_phenomenology:
            # For phenomenology paths, always resolve relative to project root
            project_root = Path(__file__).resolve().parents[2] / "BabyLM"
            phenomenology_map_path = _abs(phenomenology_map_path, project_root)
            self.phenomenology_map = np.load(phenomenology_map_path)
        else:
            self.phenomenology_map = None

        # --- information setup -----------------------------------------------
        self.s2 = InformationEngine(
            keys_path=self.ontology_path,
            ep_path=_abs(epistemology_path or "memories/public/meta/epistemology.npy", self.base_path),
            phenomap_path=_abs(phenomenology_map_path or "memories/public/meta/phenomenology_map.npy", self.base_path),
            theta_path=str(Path(self.ontology_path).with_name("theta.npy")),
        )

        # --- S-buffer for intron selection ----------------------------------
        self._S = [0] * 8

        # --- operator setup ---------------------------------------------------
        self.operator = InferenceEngine(
            s2_engine=self.s2,
            phenotype_store=self.phenotype_store,
            phenomenology_map=self.phenomenology_map,
            s_buffer=self._S,
        )

        # --- token buffer for egress processing ------------------------------
        self._byte_buf: List[int] = []
        self.MAX_TOKEN_BYTES = 1024  # Maximum token buffer size

        # --- state tracking --------------------------------------------------
        if self.use_epistemology:
            # Start at archetypal state when using epistemology
            self.current_state_index = self.s2.get_index_from_state(self.s2.tensor_to_int(governance.GENE_Mac_S))
            self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
        else:
            self.gene_mac_m_int = self.s2.get_state_from_index(0)  # Use first state as archetypal
        self._last_token_id = 0

        # --- theta tracking --------------------------------------------------
        self._θ_buf: Deque[float] = deque(maxlen=8)
        self._θ_low = -0.1
        self._θ_high = 0.1

        # --- auto-prune setup -----------------------------------------------
        self._register_auto_prune_hook()

        # --- vectorized epistemology buffer ----------------------------------
        # Reusable buffer for state trajectory computation (max 64K to prevent RAM explosion)
        self._state_buf = np.empty(65536, dtype=np.int32)

    def process_egress(self, input_byte: int) -> int:
        """
        Process Intelligence Egress: Transform input into action.

        This is the "outward" phase where external input transforms the
        system's internal physical state according to gyroscopic physics.

        Args:
            input_byte: Input byte (0-255) from external environment

        Returns:
            Transcribed intron instruction
        """
        input_byte &= 0xFF  # Defensive masking
        # S1: Transcribe input through holographic topology
        intron = governance.transcribe_byte(input_byte)
        intron &= 0xFF  # Defensive masking (optional, for symmetry)

        # Check buffer size limit to prevent runaway growth
        if len(self._byte_buf) >= self.MAX_TOKEN_BYTES:
            self._byte_buf.clear()  # Prevent runaway buffer growth
            self._last_token_id = 0  # Reset token ID to prevent stale context
            print("Warning: Token buffer overflow cleared")
            return intron  # EARLY-RETURN to discard the intron causing overflow

        # Append intron to current token buffer
        self._byte_buf.append(intron)

        # S1: Apply gyroscopic transformation to physical state
        if self.use_epistemology:
            assert self.epistemology is not None
            self.current_state_index = self.epistemology[self.current_state_index, intron]
            self._cached_state_int = self.s2.get_state_from_index(self.current_state_index)
            # No eager sync here
        else:
            self.gene_mac_m_int = governance.apply_gyration_and_transform(self.gene_mac_m_int, intron)
            self._sync_index_from_state_int()

        # State integrity assertion
        assert (self.gene_mac_m_int if not self.use_epistemology else self._cached_state_int) < (1 << 48)

        self.cycle_count += 1

        # Record divergence in θ buffer
        div = self.s2.measure_state_divergence(
            self.gene_mac_m_int if not self.use_epistemology else self._cached_state_int
        )
        self._θ_buf.append(div)

        # Check if token is complete (bit 7 == 0)
        if (intron & 0x80) == 0:
            try:
                # Token is complete - unmask back to byte stream
                tok_bytes = bytes(b ^ 0xAA for b in self._byte_buf)
                token_id = TOK.bytes_to_id(tok_bytes)

                # Fix: ignore [SEP] token (ID 102) when setting _last_token_id
                if token_id == 102:  # WordPiece [SEP]
                    # grab the *previous* real token instead
                    token_id = getattr(self, "_prev_token_id", 102)
                self._prev_token_id = token_id

                # Knowledge update once per token using token-level learning
                state_idx = (
                    self.current_state_index
                    if self.use_epistemology
                    else self.s2.get_index_from_state(self.gene_mac_m_int)
                )
                # Use the actual intron that finished the token for learning
                phenotype_entry = self.operator.learn_token(token_id, state_idx, intron)

                # Call post-cycle hooks for monitoring
                for hook in self.post_cycle_hooks:
                    try:
                        hook(self, phenotype_entry, intron)
                    except Exception:
                        # Log hook errors but don't fail the cycle
                        print("Warning: Hook error occurred")

                # Prepare for ingress
                self._last_token_id = token_id
            except Exception:
                # Handle malformed token sequences gracefully
                print("Warning: Malformed token sequence skipped")
            finally:
                # Always clear buffer to prevent accumulation
                self._byte_buf.clear()
        else:
            pass

        return intron

    def reset_token_buffer(self) -> None:
        """
        Reset token buffer - call when starting new stream.
        Clears accumulated bytes and resets last token ID.
        """
        self._byte_buf.clear()
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
        if self.use_epistemology:
            # Fully vectorized epistemology processing for high performance
            if self.epistemology is None:
                raise RuntimeError("Epistemology not loaded")

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
        else:
            # Fallback to individual processing for non-epistemology mode
            for intron in arr:
                self.process_egress(intron)

        # Update cycle count
        self.cycle_count += len(arr)

    def _process_epistemology_chunk(self, introns: np.ndarray[Any, Any], masked_arr: np.ndarray[Any, Any]) -> None:
        """
        Process a chunk of introns using fully vectorized epistemology.

        Args:
            introns: Unmasked intron array (uint8)
            masked_arr: Original masked byte array for token extraction
        """
        if self.epistemology is None:
            raise RuntimeError("Epistemology not loaded")

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
        epistemology_size = self.epistemology.shape[0]
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
            new_state = self.epistemology[prev_state, intron]
            if new_state >= epistemology_size:
                raise RuntimeError(
                    f"Transition to state {new_state} is out of bounds "
                    f"for epistemology matrix (size {epistemology_size})"
                )
            st[i] = new_state

        # --- 2. process tokens properly (like the original process_egress)
        token_buffer = []
        for i, intron in enumerate(introns):
            token_buffer.append(intron)

            # Check if token is complete (bit 7 == 0)
            if (intron & 0x80) == 0:
                # Token is complete - unmask back to byte stream
                tok_bytes = bytes(b ^ 0xAA for b in token_buffer)
                try:
                    token_id = TOK.bytes_to_id(tok_bytes)
                    state_idx = int(st[i])  # final state *after* this token
                    # Use the actual intron that finished the token for learning
                    phenotype_entry = self.operator.learn_token(token_id, state_idx, int(introns[i]))

                    # Call post-cycle hooks for monitoring
                    for hook in self.post_cycle_hooks:
                        try:
                            hook(self, phenotype_entry, int(introns[i]))
                        except Exception:
                            # Log hook errors but don't fail the cycle
                            print("Warning: Hook error occurred")
                except Exception:
                    pass
                finally:
                    # Always clear buffer to prevent accumulation
                    token_buffer.clear()

        # --- 4. persist new head state
        if n > 0:
            final_intron = introns[-1]
            if self.current_state_index >= epistemology_size:
                raise RuntimeError(
                    f"Final state index {self.current_state_index} is out of bounds "
                    f"for epistemology matrix (size {epistemology_size})"
                )
            new_state = self.epistemology[self.current_state_index, final_intron]
            if new_state >= epistemology_size:
                raise RuntimeError(
                    f"Final transition to state {new_state} is out of bounds "
                    f"for epistemology matrix (size {epistemology_size})"
                )
            self.current_state_index = new_state

    def process_ingress(self) -> tuple[int, int]:
        """
        Generate **one token** (may be 1-3 bytes). Returns:
            (byte_out, intron_out)  of the *last* byte emitted.
        """
        # --- 1. resolve current state ---
        state_idx = (
            self.current_state_index if self.use_epistemology else self.s2.get_index_from_state(self.gene_mac_m_int)
        )

        # Prepare a fresh token when the buffer is empty
        if not self._emit_buf:
            # --- 0. resolve orbit & θ
            theta = self._θ_buf[-1] if self._θ_buf else 0.0

            # --- 1. Get phenotype using orbit-based lookup
            phe = self.operator.get_phenotype(state_idx, self._last_token_id)

            if phe:
                # For now, use the last token ID as fallback since we're storing introns
                tok_id = self._last_token_id
                conf = phe.get("conf", 0.1)
                mask = phe.get("mask", 0)

                # --- 2. Theta-gated exploration (as per Genetics.md §5.5.4)
                # Algedonic selection based on angular divergence
                if theta < self._θ_low:
                    # Calm mode: use newest aligned intron (S[5])
                    current_s = self._S[-1] if hasattr(self, "_S") and self._S else 0
                    sim = 1.0 - (bin(current_s ^ mask).count("1") / 8.0)
                    # High similarity → stick to learned token
                    if sim < 0.5:
                        vocab_size = tokmod.vocab_size()
                        tok_id = random.randint(0, vocab_size - 1)
                elif theta < self._θ_high:
                    # Cautious mode: use less recent intron (S[4])
                    # Moderate exploration
                    if random.random() < 0.3:
                        vocab_size = tokmod.vocab_size()
                        tok_id = random.randint(0, vocab_size - 1)
                else:
                    # Corrective mode: use raw exon-product for stability
                    # High divergence → direct corrective action
                    vocab_size = tokmod.vocab_size()
                    tok_id = random.randint(0, vocab_size - 1)

                # --- 3. Orbit-based learning rate adjustment
                # Get orbit cardinality for learning rate
                orbit_size = self.s2.orbit_cardinality[state_idx] if hasattr(self.s2, "orbit_cardinality") else 1
                max_variety = getattr(self.s2, "max_variety", 1)
                learning_rate = math.sqrt(orbit_size / max_variety) if max_variety > 0 else 0.1

                # Confidence-weighted exploration
                if random.random() < (1.0 - conf) * learning_rate:
                    vocab_size = tokmod.vocab_size()
                    tok_id = random.randint(0, vocab_size - 1)
            else:
                # Fallback to CLS token
                tok_id = 101

            # --- 4. emit as before
            masked = TOK.id_to_bytes(tok_id)
            self._emit_buf = [b ^ 0xAA for b in masked]  # store as unmasked introns

        intron_out = self._emit_buf.pop(0)
        byte_out = intron_out ^ 0xAA

        # ---------- 3. feed back through egress -----------
        self.process_egress(byte_out)  # will update state + learn

        return byte_out, intron_out

    def _choose_intron(self, phe: PhenotypeEntry, theta: float, state_index: int) -> int:
        """
        Choose the next intron based on phenotype and theta using LEB128 physics.

        Args:
            phe: Phenotype entry with minimal structure (mask, conf)
            theta: Current theta value
            state_index: Current state index for orbit cardinality calculation

        Returns:
            Chosen intron value
        """
        # Import LEB128 physics functions
        try:
            from toys.experiments.leb128_physics import token_to_introns
        except ImportError:
            # Fallback to original byte-level approach if LEB128 not available
            return self._choose_intron_byte_level(phe, theta, state_index)

        # For LEB128 physics, we need to choose a token, not an intron
        # This is a simplified approach - in practice, we'd want to choose from learned tokens
        mask = phe["mask"]
        conf = phe["conf"]
        v = self.s2.orbit_cardinality[state_index]
        p = governance.exon_product_from_metadata(mask, conf, v, self.s2._v_max)

        self._S = self._S[1:] + [governance.fold(self._S[0], p)]

        # For now, use the original logic but with LEB128-aware intron selection
        if theta < self._θ_low:
            chosen_intron = self._S[5]
        elif theta < self._θ_high:
            chosen_intron = self._S[4]
        else:
            chosen_intron = p

        return chosen_intron

    def _choose_intron_byte_level(self, phe: PhenotypeEntry, theta: float, state_index: int) -> int:
        """
        Original byte-level intron selection (fallback method).
        """
        mask = phe["mask"]
        conf = phe["conf"]
        v = self.s2.orbit_cardinality[state_index]
        p = governance.exon_product_from_metadata(mask, conf, v, self.s2._v_max)

        self._S = self._S[1:] + [governance.fold(self._S[0], p)]

        if theta < self._θ_low:
            return self._S[5]
        elif theta < self._θ_high:
            return self._S[4]
        return p

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
        angular_divergence = self.s2.measure_state_divergence(self.gene_mac_m_int)
        tensor_index = (
            self.current_state_index if self.use_epistemology else self.s2.get_index_from_state(self.gene_mac_m_int)
        )
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
        if self.use_epistemology:
            self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)
            self._cached_state_int = self.gene_mac_m_int

    def _sync_index_from_state_int(self) -> None:
        if self.use_epistemology:
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
            if is_dataclass(result):
                return asdict(result)
            if hasattr(result, "__dict__"):
                return vars(result)
            return {}

    def prune_low_confidence_entries(self, confidence_threshold: float = 0.05) -> int:
        """
        Prune entries from the knowledge base with confidence below a threshold.

        Args:
            confidence_threshold: Minimum confidence for entry retention

        Returns:
            Number of entries pruned
        """
        return self.operator.prune_low_confidence_entries(confidence_threshold=confidence_threshold)

    def _register_auto_prune_hook(self) -> None:
        """
        Registers a post-cycle hook to automatically prune low-confidence entries
        based on preferences configuration.
        """
        # Check if auto-pruning is enabled in preferences
        pruning_cfg = self.preferences.get("pruning", {})
        if pruning_cfg.get("enable_auto_decay", False):
            self.add_hook(self._auto_prune_hook)

    def _auto_prune_hook(
        self, engine: "IntelligenceEngine", phenotype_entry: PhenotypeEntry, last_token_byte: int
    ) -> None:
        """
        Post-cycle hook to automatically prune low-confidence entries.

        This hook is called after each cycle and removes entries below the confidence threshold.
        If more than 10,000 entries are removed, it triggers background compaction.

        Rate limiting: Only runs every 1000 cycles to avoid performance issues with large stores.
        """
        # Rate limiting: only run every 1000 cycles to avoid performance issues
        if self.cycle_count % 1000 != 0:
            return

        # Skip pruning for now since we're using index-based mode
        return

        # Get pruning configuration from preferences
        pruning_cfg = self.preferences.get("pruning", {})
        thr = pruning_cfg.get("confidence_threshold", 0.05)

        # Prune low-confidence entries
        try:
            removed = self.operator.prune_low_confidence_entries(confidence_threshold=thr)
        except RuntimeError as e:
            if "append_only store cannot delete" in str(e):
                # For append-only stores, we can't do in-memory pruning
                # The pruning will happen during the next compaction cycle
                removed = 0
            else:
                # Re-raise other RuntimeErrors
                raise
        except Exception:
            removed = 0

        # Optional: trigger background compaction if many entries vanished
        if removed > 10_000:
            # Get the store path for compaction
            store_path = None
            if hasattr(self.operator.store, "store_path"):
                store_path = self.operator.store.store_path
            elif hasattr(self.operator.store, "base_store") and hasattr(self.operator.store.base_store, "store_path"):
                store_path = self.operator.store.base_store.store_path

            if store_path:
                try:
                    # Import here to avoid circular imports
                    from baby.policies import prune_and_compact_store

                    prune_and_compact_store(
                        store_path=store_path,
                        output_path=None,  # in-place
                        min_confidence=thr,
                        dry_run=False,
                    )
                except Exception:
                    # Log the error but don't fail the hook
                    pass

        if removed > 0:
            # Only print this message if in debug mode
            if getattr(self, "debug_mode", False):
                print(f"Auto-pruned {removed} low-confidence entries (threshold: {thr})")


class GyroSI:
    """
    S5: Whole System Identity & Policy.

    The outermost viable system boundary that encapsulates the entire VSM stack.
    Manages configuration, agent identity, and provides the stable external API.
    """

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
        self.preferences = config.get("preferences", {})
        # Patch only allowed AgentConfig path keys to be absolute if not already
        if (
            "ontology_path" in self.config
            and self.config["ontology_path"]
            and not os.path.isabs(str(self.config["ontology_path"]))
        ):
            # For ontology paths, always resolve relative to project root
            project_root = Path(__file__).resolve().parents[2] / "BabyLM"
            self.config["ontology_path"] = str(project_root / str(self.config["ontology_path"]))
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
            # For phenomenology paths, always resolve relative to project root
            project_root = Path(__file__).resolve().parents[2] / "BabyLM"
            self.config["phenomenology_map_path"] = str(project_root / str(self.config["phenomenology_map_path"]))
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
            # For phenomenology paths, always resolve relative to project root
            project_root = Path(__file__).resolve().parents[2] / "BabyLM"
            self.config["phenomenology_map_path"] = str(project_root / "memories/public/meta/phenomenology_map.npy")
        if "base_path" not in self.config or not self.config["base_path"]:
            self.config["base_path"] = str(self.base_path / "memories")
        self.agent_id = agent_id or str(uuid.uuid4())
        if phenotype_store is None:
            phenotype_store = self._create_default_store()

        # Extract preferences from config
        preferences = self.config.get("preferences", {})

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
            preferences=preferences,
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

    def respond(self, data: bytes, max_new_tokens: int = 64) -> bytes:
        """
        Generate an intelligent response.  Guarantees that every emitted
        LEB128 token is complete (no dangling continuation bit).
        """

        # ---------- 1. ingest user prompt -------------------------------
        for i, b in enumerate(data):
            self.engine.process_egress(b)

        # ---------- 2. generate -----------------------------------------
        out = bytearray()
        tokens_done = 0

        while tokens_done < max_new_tokens:
            # keep the start offset so we can EOS-check later
            token_start = len(out)

            # -------- generate ONE token ----------------------------
            while True:
                byte_out, _ = self.engine.process_ingress()
                out.append(byte_out)

                # Did the *intron* side say "done"?  (bit-7 = 0)
                if ((byte_out ^ 0xAA) & 0x80) == 0:
                    tokens_done += 1
                    break  # token closed – leave byte-loop

            # -------- optional EOS check ---------------------------------
            try:
                # Import tokenizer locally to avoid top-level sys.path hack
                import sys as _sys
                from pathlib import Path

                _sys.path.append(str(Path(__file__).resolve().parents[2] / "toys" / "communication"))
                from toys.communication import tokenizer as tok

                # unmask just this token and decode its ID
                intron_bytes = bytes(b ^ 0xAA for b in out[token_start:])
                if tok._bytes_to_ids(intron_bytes)[0] == 102:  # [SEP]
                    break
            except ValueError:
                # should never happen now, but we keep the guard
                pass

        self._commit_if_needed()
        return bytes(out)

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

    def apply_maintenance(self, decay_rate: float = 0.001, confidence_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Apply maintenance operations to the knowledge base.

        Args:
            decay_rate: Confidence decay rate for aging entries (small value, e.g. 0.001)
            confidence_threshold: Minimum confidence for entry retention

        Returns:
            Maintenance report
        """
        # Apply confidence decay
        decay_report = self.engine.apply_confidence_decay(decay_rate)

        # Prune low-confidence entries
        pruned_count = self.engine.prune_low_confidence_entries(confidence_threshold)

        # Commit changes to persist maintenance operations
        store = self.engine.operator.store
        if hasattr(store, "commit"):
            store.commit()

        return {"decay_applied": decay_report, "entries_pruned": pruned_count, "timestamp": time.time()}

    def close(self) -> None:
        """Clean shutdown of the agent."""
        self.engine.operator.store.close()

    def _create_default_store(self) -> Any:
        """Create default storage based on configuration."""
        # --- Honor store_options for binary_struct/append-only store ---
        # cast self.config to a dict so .get is allowed
        # Always use index-based mode for faster lookups
        pass
        # Canonicalisation enablement consistency: autodetect if flag is None or missing
        enable_phenomenology = cast(Dict[str, Any], self.config).get("enable_phenomenology_storage", None)

        # Create base store
        public_knowledge_path = self.config.get("public_knowledge_path")
        learn_batch_size = self.config.get("learn_batch_size", 100)
        if learn_batch_size is None:
            learn_batch_size = 100
        phenomenology_map_path: Optional[str] = None  # Ensure always defined
        private_root = self.config.get("private_agents_base_path", str(self.base_path / "memories/private/agents"))
        if public_knowledge_path is not None:
            # Multi-agent setup with public/private knowledge
            private_path = self.config.get("private_knowledge_path")
            if private_path is None:
                if private_root is None:
                    raise ValueError("private_agents_base_path must not be None")
                private_path = os.path.join(str(private_root), f"{self.agent_id}/knowledge.bin")
            # Multi-agent overlay using decorators
            public_store = ReadOnlyView(
                OrbitStore(public_knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path)
            )
            private_store = OrbitStore(private_path, write_threshold=learn_batch_size, base_path=self.base_path)
            base_store: Any = OverlayView(public_store, private_store)
            phenomenology_map_path = self.config.get("phenomenology_map_path")
        else:
            # Single-agent setup
            knowledge_path = self.config.get("knowledge_path")
            if knowledge_path is None:
                # Use preferences if available, otherwise fallback to default
                if self.preferences and isinstance(self.preferences, dict):
                    prefs_dict = cast(Dict[str, Any], self.preferences)
                    if "public_knowledge" in prefs_dict:
                        knowledge_path = prefs_dict["public_knowledge"]["path"]
                    else:
                        base_path = self.config.get("base_path") or str(self.base_path / "memories")
                        knowledge_path = os.path.join(base_path, "knowledge.bin")  # binary_struct-based fallback path
                else:
                    base_path = self.config.get("base_path") or str(self.base_path / "memories")
                    knowledge_path = os.path.join(base_path, "knowledge.bin")  # binary_struct-based fallback path
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)
            base_store = OrbitStore(knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path)

            # CanonicalView: enable if flag is True, or autodetect if None and file exists
            phenomenology_map_path = self.config.get("phenomenology_map_path")
        # Ensure phenomenology_map_path is always a str before use
        if phenomenology_map_path is None:
            phenomenology_map_path = str(self.base_path / "memories/public/meta/phenomenology_map.npy")
        if enable_phenomenology or (
            enable_phenomenology is None and phenomenology_map_path and os.path.exists(phenomenology_map_path)
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
        preferences: Optional[PreferencesConfig] = None,
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
        self.max_agents = cast(Dict[str, Any], self.preferences).get("max_agents_in_memory", 1000)
        self.eviction_policy = cast(Dict[str, Any], self.preferences).get("agent_eviction_policy", "lru")
        self.allowed_ids = allowed_ids or {"user", "system", "assistant"}
        self.allow_auto_create = allow_auto_create
        self.agent_access_times: Dict[str, float] = {}
        self._shards: List[Dict[str, Any]] = []
        for _ in range(self.SHARD_COUNT):
            if self.eviction_policy == "lru":
                shard_size = max(1, self.max_agents // self.SHARD_COUNT)
                self._shards.append({"agents": LRUAgentCache(shard_size), "lock": RLock()})
            else:
                self._shards.append({"agents": OrderedDict(), "lock": RLock()})
        self._public_store: Optional[ReadOnlyView] = ReadOnlyView(
            CanonicalView(
                OrbitStore(self.base_knowledge_path, write_threshold=100, base_path=self.base_path),
                phenomenology_map_path=str(Path(self.ontology_path).with_name("phenomenology_map.npy")),
                base_path=self.base_path,
            )
        )

    def _shard_index(self, agent_id: str) -> int:
        # Use hash for even distribution; mask for power-of-two
        return (hash(agent_id) if isinstance(agent_id, str) else agent_id) & (self.SHARD_COUNT - 1)

    def get_or_create_agent(self, agent_id: str, role_hint: Optional[str] = None) -> "GyroSI":
        # legacy path – still here, but now obeys policy
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
                self.agent_access_times[agent_id] = time.time()
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
                base_store: Any = OverlayView(public_store, private_store)

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
                    "preferences": self.preferences,  # Pass preferences to the agent
                }

                # Add role hint to metadata if provided
                if role_hint:
                    config["agent_metadata"] = {"role_hint": role_hint}

                agents[agent_id] = GyroSI(
                    config=config,
                    agent_id=agent_id,
                    phenotype_store=base_store,
                    base_path=self.base_path,
                )

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
        """Remove agents that haven't been accessed recently (TTL policy)."""
        if self.eviction_policy != "ttl":
            return
        ttl_minutes = self.preferences.get("agent_ttl_minutes", 60)
        ttl_seconds = ttl_minutes * 60
        current_time = time.time()
        expired_agents = [
            agent_id
            for agent_id, last_access in self.agent_access_times.items()
            if current_time - last_access > ttl_seconds
        ]
        for agent_id in expired_agents:
            self.remove_agent(agent_id)


def orchestrate_turn(
    pool: AgentPool,
    user_id: str,
    assistant_id: str,
    user_input: str,
    tokenizer_name: str,  # Make this mandatory
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
    from toys.communication import tokenizer as tok

    in_bytes = tok.encode(user_input, name=tokenizer_name)

    # 2. User agent processes input, creating stimulus
    stimulus = user_agent.respond(in_bytes)

    # 3. Assistant responds to stimulus
    response = assistant_agent.respond(stimulus)

    # 4. Decode response using the same tokenizer.
    #    The `decode` function already has a UTF-8 fallback for robustness.
    result = tok.decode(response, name=tokenizer_name)
    return result
