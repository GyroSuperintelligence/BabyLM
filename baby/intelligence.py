"""
S4/5: Intelligence – Orchestration & API
…your module docstring…
"""

from __future__ import annotations

from pathlib import Path

# Import tokenizer for token-aware generation
import os
import time
import uuid
from collections import OrderedDict, deque
from functools import cached_property
from threading import RLock
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, TypedDict, cast

from dataclasses import asdict, is_dataclass

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
        base_path: Path = Path(__file__).resolve().parents[1],
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
            epistemology_path = _abs(epistemology_path, self.base_path)
            self.epistemology = np.load(epistemology_path)
            self.current_state_index = 0
        else:
            self.epistemology = None
            self.current_state_index = 0  # Always initialize to 0, not None

        # --- phenomenology setup ----------------------------------------------
        self.use_phenomenology = phenomenology_map_path is not None
        if self.use_phenomenology:
            phenomenology_map_path = _abs(phenomenology_map_path, self.base_path)
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

        # --- operator setup ---------------------------------------------------
        self.operator = InferenceEngine(
            s2_engine=self.s2,
            phenotype_store=self.phenotype_store,
        )

        # --- state tracking --------------------------------------------------
        self.gene_mac_m_int = self.s2.get_state_from_index(0)  # Use first state as archetypal
        self._last_token_id = 0

        # --- theta tracking --------------------------------------------------
        self._θ_buf: Deque[float] = deque(maxlen=8)
        self._θ_low = -0.1
        self._θ_high = 0.1

        # --- S-buffer for intron selection ----------------------------------
        self._S = [0] * 8

        # --- token buffer for egress processing ------------------------------
        self._byte_buf: List[int] = []
        self.MAX_TOKEN_BYTES = 1024  # Maximum token buffer size

        # --- auto-prune setup -----------------------------------------------
        self._register_auto_prune_hook()

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

                # Knowledge update once per token
                state_idx = (
                    self.current_state_index
                    if self.use_epistemology
                    else self.s2.get_index_from_state(self.gene_mac_m_int)
                )
                phe = self.operator.get_phenotype(state_idx, token_id)
                # Fold only with the final intron of the token
                self.operator.learn(phe, intron, state_idx)

                # Call post-cycle hooks after learning
                for hook in self.post_cycle_hooks:
                    hook(self, phe, intron)

                # Prepare for ingress
                self._last_token_id = token_id
            except Exception as e:
                # Handle malformed token sequences gracefully
                print(f"Warning: Malformed token sequence skipped: {e}")
            finally:
                # Always clear buffer to prevent accumulation
                self._byte_buf.clear()

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
            # Process each intron individually to maintain state consistency
            start = 0
            assert self.epistemology is not None
            for i, intron in enumerate(introns.tolist()):  # use unmasked intron
                self.current_state_index = self.epistemology[self.current_state_index, intron]
                state_idx = self.current_state_index

                # Token completes when the *intron* continuation bit is 0
                if (intron & 0x80) == 0:
                    # token_bytes must be the original *masked* slice from the blob
                    token_bytes = bytes(arr[start:i + 1])
                    try:
                        token_id = TOK.bytes_to_id(token_bytes)  # expects masked bytes
                        phe = self.operator.get_phenotype(state_idx, token_id)
                        self.operator.learn(phe, int(intron), state_idx)
                    except Exception:
                        pass
                    start = i + 1
        else:
            # Fallback to individual processing for non-epistemology mode
            for intron in arr:
                self.process_egress(intron)

        # Update cycle count
        self.cycle_count += len(arr)

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
            phe = self.operator.get_phenotype(state_idx, self._last_token_id)
            # Choose a deterministic token id from context; map into tokenizer vocab
            vocab_size = tokmod.vocab_size()
            tok_id = self.operator._compute_semantic_address(phe["key"]) % vocab_size
            # id_to_bytes returns *masked* bytes; we need unmasked introns for the physics step
            masked = TOK.id_to_bytes(int(tok_id))
            self._emit_buf = [b ^ 0xAA for b in masked]  # store as unmasked introns

        intron_out = self._emit_buf.pop(0)
        byte_out = intron_out ^ 0xAA

        # ---------- 3. feed back through egress -----------
        self.process_egress(byte_out)  # will update state + learn

        return byte_out, intron_out

    def _choose_intron(self, phe: PhenotypeEntry, theta: float, state_index: int) -> int:
        """
        Choose the next intron based on phenotype and theta.

        Args:
            phe: Phenotype entry with minimal structure (mask, conf)
            theta: Current theta value
            state_index: Current state index for orbit cardinality calculation

        Returns:
            Chosen intron value
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
        """
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
                except Exception as e:
                    # Log the error but don't fail the hook
                    print(f"Auto-pruning compaction failed: {e}")

        if removed > 0:
            # Only print this message if in debug mode
            # print(f"Auto-pruned {removed} low-confidence entries (threshold: {thr})")
            if getattr(self, "debug_mode", False):
                print(f"Auto-pruned {removed} low-confidence entries (threshold: {thr})")
            # else: suppress in normal operation


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
        base_path: Path = Path(__file__).resolve().parents[1],
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
            self.config["ontology_path"] = str(self.base_path / str(self.config["ontology_path"]))
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
        if (
            "private_knowledge_path" in self.config
            and self.config["private_knowledge_path"]
            and not os.path.isabs(str(self.config["private_knowledge_path"]))
        ):
            self.config["private_knowledge_path"] = str(self.base_path / str(self.config["private_knowledge_path"]))
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
            self.config["phenomenology_map_path"] = str(Path(onto).with_name("phenomenology_map.npy"))
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
        epistemology_path = str(Path(onto).with_name("epistemology.npy"))
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
        for b in data:
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
                import sys
                from pathlib import Path

                sys.path.append(str(Path(__file__).resolve().parents[1] / "toys" / "communication"))
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
        opts = cast(Dict[str, Any], self.config).get("store_options", {}) or {}
        if opts.get("append_only"):
            knowledge_path = cast(Dict[str, Any], self.config).get("knowledge_path")
            if knowledge_path is None:
                raise ValueError("knowledge_path must be set in config")
            return OrbitStore(
                knowledge_path,
                append_only=opts.get("append_only", True),
            )
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
                OrbitStore(
                    public_knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path, append_only=True
                )
            )
            private_store = OrbitStore(
                private_path, write_threshold=learn_batch_size, base_path=self.base_path, append_only=True
            )
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
            base_store = OrbitStore(
                knowledge_path, write_threshold=learn_batch_size, base_path=self.base_path, append_only=True
            )

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
        base_path: Path = Path(__file__).resolve().parents[1],
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
            OrbitStore(self.base_knowledge_path, write_threshold=100, base_path=self.base_path, append_only=True)
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
                private_store = OrbitStore(
                    private_path, write_threshold=100, base_path=self.base_path, append_only=True
                )
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
                print(f"DEBUG: Created agent {agent_id}. Current agents: {list(agents.keys())}")

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
    return tok.decode(response, name=tokenizer_name)
