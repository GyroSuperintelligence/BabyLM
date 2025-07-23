"""
S4/5: Intelligence - Orchestration & API

This module provides the IntelligenceEngine and GyroSI classes responsible for
orchestrating the complete system and providing the external API.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict, deque
from threading import RLock
from typing import Any, Dict, List, Optional, TypedDict, cast, TYPE_CHECKING, Deque, Tuple

import numpy as np
from dataclasses import asdict, is_dataclass
from pathlib import Path

from baby import governance
from baby.contracts import AgentConfig, CycleHookFunction, PreferencesConfig
from baby.inference import InferenceEngine
from baby.information import InformationEngine
from baby.policies import CanonicalView, OrbitStore, OverlayView, ReadOnlyView


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
    ):
        """
        Initialize intelligence engine.
        All file paths are resolved with respect to base_path unless absolute.
        """
        self.base_path = base_path
        self.ontology_path = _abs(ontology_path, self.base_path)
        self.epistemology_path = _abs(
            epistemology_path if epistemology_path is not None else str(
                Path(self.ontology_path).with_name("epistemology.npy")
            ),
            self.base_path,
        )
        self.phenomenology_map_path = _abs(
            phenomenology_map_path if phenomenology_map_path is not None else str(
                Path(self.ontology_path).with_name("phenomenology_map.json")
            ),
            self.base_path,
        )
        # Initialize subsystem engines
        self.s2: InformationEngine = InformationEngine(self._load_ontology(self.ontology_path))
        self.operator: InferenceEngine = InferenceEngine(self.s2, phenotype_store)

        # Agent state
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.use_epistemology: bool = False
        self.current_state_index: int
        self.gene_mac_m_int: int
        self._cached_state_int: int = 0  # Only used if use_epistemology is True
        if self.epistemology_path and os.path.exists(self.epistemology_path):
            try:
                import numpy as np

                self.epistemology = np.load(self.epistemology_path, mmap_mode="r")
                self.use_epistemology = True
                print("INFO: State Transition Table (STT) loaded. Using optimized state transitions.")
            except Exception as e:
                print(
                    f"WARNING: Could not load STT from {self.epistemology_path}. "
                    f"Error: {e}. Falling back to dynamic physics."
                )

        origin_int: int = self.s2.tensor_to_int(governance.GENE_Mac_S)
        if self.use_epistemology:
            self.current_state_index = self.s2.get_index_from_state(origin_int)
            self.gene_mac_m_int = origin_int  # Ensure always defined
        else:
            self.gene_mac_m_int = origin_int
            self.current_state_index = 0  # Default if not using epistemology
        self.cycle_count: int = 0
        self._microstep_count: int = 0  # Track internal cooling/autonomic steps
        # Extension points
        self.post_cycle_hooks: List[CycleHookFunction] = []
        self._hook_event_buffer: Deque[Tuple[Any, ...]] = deque(maxlen=hook_batch_interval)
        self._hook_batch_interval = hook_batch_interval

        # Algedonic regulation and autonomic cycles
        self._θ_buf: deque[float] = deque(maxlen=128)
        self._θ_high: float = 0.9  # radians
        self._θ_low: float = 0.3
        self._cool_introns: tuple[int, ...] = (0b01000010,)
        try:
            if self.phenomenology_map_path and os.path.exists(self.phenomenology_map_path):
                with open(self.phenomenology_map_path) as f:
                    pheno_data = json.load(f)
                    self._autonomic_cycles: list[Any] = pheno_data.get("autonomic_cycles", [])
                    self._autonomic_cycles_curated: dict[str, Any] = pheno_data.get("autonomic_cycles_curated", {})
            else:
                self._autonomic_cycles = []
                self._autonomic_cycles_curated = {}
        except Exception:
            self._autonomic_cycles = []
            self._autonomic_cycles_curated = {}
        self._pain_streak: int = 0

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

        # S1: Apply gyroscopic transformation to physical state
        if self.use_epistemology:
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
        return intron

    def process_ingress(self, last_intron: int) -> int:
        last_intron &= 0xFF  # Defensive masking
        # S3: Get semantic meaning of current state + context
        # state_index is physical index; canonicalisation (if enabled) is applied at storage layer (CanonicalView)
        if self.use_epistemology:
            state_index = self.current_state_index
        else:
            state_index = self.s2.get_index_from_state(self.gene_mac_m_int)
        # Ingress: complete monodromic loop by folding with the same intron used for addressing.
        # This causes exon_mask to collapse (x → 0), expressing closure—not accumulation.
        # Phenotype metadata tracks recurrence; output is drawn from the stored value, not from the mask.
        phenotype_entry = self.operator.get_phenotype(state_index, last_intron)

        # S3: Learn through Monodromic Fold
        self.operator.learn(phenotype_entry, last_intron)

        # Buffer hook events and process hooks every N cycles, unless pain spike
        self._hook_event_buffer.append((self, phenotype_entry, last_intron))
        process_hooks_now = False
        θ = np.mean(self._θ_buf) if self._θ_buf else 0.0  # Cache θ once
        if θ > self._θ_high or len(self._hook_event_buffer) >= self._hook_batch_interval:
            process_hooks_now = True
        if process_hooks_now and self.post_cycle_hooks:
            for event in list(self._hook_event_buffer):
                for hook in self.post_cycle_hooks:
                    hook(*event)
            self._hook_event_buffer.clear()

        # Store original phenotype_entry for output
        output_phenotype_entry = phenotype_entry

        # Algedonic decision at start (must run before output)
        # Use cached θ
        if θ > self._θ_high:
            self._pain_streak += 1
            cooling_intron = self._cool_introns[self.cycle_count % len(self._cool_introns)]
            saved = self.cycle_count
            self.process_egress(cooling_intron)  # does all the usual work
            self.cycle_count = saved  # restore the external-cycle count
            self._microstep_count += 1  # track internal steps separately
            # State integrity assertion after cooling micro-step
            assert (
                self.gene_mac_m_int
                if not self.use_epistemology
                else self.s2.get_state_from_index(self.current_state_index)
            ) < (1 << 48)
            if self.use_epistemology:
                state_index_cool = self.current_state_index
            else:
                state_index_cool = self.s2.get_index_from_state(self.gene_mac_m_int)
            cool_entry = self.operator.get_phenotype(state_index_cool, cooling_intron)
            self.operator.learn(cool_entry, cooling_intron)
            if self._pain_streak > 256 and self._autonomic_cycles:
                for intr in self._autonomic_cycles[self.cycle_count % len(self._autonomic_cycles)]:
                    saved = self.cycle_count
                    self.process_egress(intr)
                    self.cycle_count = saved
                    self._microstep_count += 1
                    # State integrity assertion after autonomic cycle micro-step
                    assert (
                        self.gene_mac_m_int
                        if not self.use_epistemology
                        else self.s2.get_state_from_index(self.current_state_index)
                    ) < (1 << 48)
                    if self.use_epistemology:
                        si_aut = self.current_state_index
                    else:
                        si_aut = self.s2.get_index_from_state(self.gene_mac_m_int)
                    aut_entry = self.operator.get_phenotype(si_aut, intr)
                    self.operator.learn(aut_entry, intr)
                self._pain_streak = 0
        elif θ < self._θ_low:
            self._pain_streak = 0

        # Generate response from original phenotype_entry (not cooling intron)
        phenotype = output_phenotype_entry.get("phenotype")
        if isinstance(phenotype, str) and phenotype:
            return ord(phenotype[0])
        elif isinstance(phenotype, str):
            return ord("?")
        elif phenotype is not None:
            return int(phenotype) & 0xFF  # Ensure byte range
        else:
            return ord("?")

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

    def batch_learn(self, data: bytes) -> None:
        """
        Learn from a batch of data using streaming Monodromic Fold.

        This method allows efficient batch learning while preserving the
        path-dependent nature of the Fold operation. Uses O(1) memory
        by streaming the Fold instead of collecting all introns.

        Args:
            data: Batch of bytes to learn from
        """
        if not data:
            return

        # Streaming Fold: O(1) memory instead of O(N)
        acc = 0
        for byte in data:
            # Use the optimized process_egress method (STT-aware)
            intron = self.process_egress(byte)
            acc = governance.fold(acc, intron)
            # self.cycle_count is incremented in process_egress

        # Learn from the final accumulated intron
        if acc != 0:
            # Use the correct state index depending on STT
            if self.use_epistemology:
                state_index = self.current_state_index
            else:
                state_index = self.s2.get_index_from_state(self.gene_mac_m_int)
            phenotype_entry = self.operator.get_phenotype(state_index, acc)
            self.operator.learn(phenotype_entry, acc)
        # Ensure all writes are flushed/committed
        store = self.operator.store
        if hasattr(store, "commit"):
            store.commit()

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

    def _load_ontology(self, ontology_path: str) -> Dict[str, Any]:
        """Loads the ontology data from a JSON file as ManifoldData."""
        with open(ontology_path, "r") as f:
            data = json.load(f)
        return dict(data)

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
                return bool(result.get("total_entries", 0) > 0)
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
        return self.operator.prune_low_confidence_entries(confidence_threshold)


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
        # Patch only allowed AgentConfig path keys to be absolute if not already
        if (
            "ontology_path" in self.config
            and self.config["ontology_path"]
            and not os.path.isabs(str(self.config["ontology_path"]))
        ):
            self.config["ontology_path"] = str(
                self.base_path / str(self.config["ontology_path"])
            )
        if (
            "knowledge_path" in self.config
            and self.config["knowledge_path"]
            and not os.path.isabs(str(self.config["knowledge_path"]))
        ):
            self.config["knowledge_path"] = str(
                self.base_path / str(self.config["knowledge_path"])
            )
        if (
            "public_knowledge_path" in self.config
            and self.config["public_knowledge_path"]
            and not os.path.isabs(str(self.config["public_knowledge_path"]))
        ):
            self.config["public_knowledge_path"] = str(
                self.base_path / str(self.config["public_knowledge_path"])
            )
        if (
            "private_knowledge_path" in self.config
            and self.config["private_knowledge_path"]
            and not os.path.isabs(str(self.config["private_knowledge_path"]))
        ):
            self.config["private_knowledge_path"] = str(
                self.base_path / str(self.config["private_knowledge_path"])
            )
        if (
            "phenomenology_map_path" in self.config
            and self.config["phenomenology_map_path"]
            and not os.path.isabs(str(self.config["phenomenology_map_path"]))
        ):
            self.config["phenomenology_map_path"] = str(
                self.base_path / str(self.config["phenomenology_map_path"])
            )
        if (
            "private_agents_base_path" in self.config
            and self.config["private_agents_base_path"]
            and not os.path.isabs(str(self.config["private_agents_base_path"]))
        ):
            self.config["private_agents_base_path"] = str(
                self.base_path / str(self.config["private_agents_base_path"])
            )
        if "base_path" in self.config and self.config["base_path"] and not os.path.isabs(str(self.config["base_path"])):
            self.config["base_path"] = str(
                self.base_path / str(self.config["base_path"])
            )
        if "private_agents_base_path" not in self.config or not self.config["private_agents_base_path"]:
            self.config["private_agents_base_path"] = str(
                self.base_path / "memories/private/agents"
            )
        # Only assign keys that are valid for AgentConfig
        if "phenomenology_map_path" not in self.config or not self.config["phenomenology_map_path"]:
            onto = self.config.get("ontology_path")
            if onto is not None:
                self.config["phenomenology_map_path"] = str(
                    Path(onto).with_name("phenomenology_map.json")
                )
            else:
                raise ValueError(
                    "ontology_path must be set in config"
                )
        if "base_path" not in self.config or not self.config["base_path"]:
            self.config["base_path"] = str(
                self.base_path / "memories"
            )
        self.agent_id = agent_id or str(uuid.uuid4())
        if phenotype_store is None:
            phenotype_store = self._create_default_store()
        # Use local variables for extra paths
        onto = self.config.get("ontology_path")
        if onto is None:
            raise ValueError("ontology_path must be set in config")
        epistemology_path = str(Path(onto).with_name("epistemology.npy"))
        self.engine = IntelligenceEngine(
            ontology_path=onto,
            phenotype_store=phenotype_store,
            agent_id=self.agent_id,
            epistemology_path=epistemology_path,
            phenomenology_map_path=self.config["phenomenology_map_path"],
            base_path=self.base_path,
        )

    def ingest(self, data: bytes) -> None:
        """
        Learn from a batch of data using ordered Monodromic Fold.

        This is the primary learning interface. Data is processed as a batch
        with the final state determining the learning context.

        Args:
            data: Bytes to learn from
        """
        self.engine.batch_learn(data)
        # Ensure pending writes are flushed
        store = self.engine.operator.store
        if hasattr(store, "commit"):
            store.commit()
        # Removed automatic close()

    def respond(self, data: bytes) -> bytes:
        """
        Generate an intelligent response to input data.

        Args:
            data: Input bytes to respond to

        Returns:
            Intelligent response bytes
        """
        if not data:
            return b""

        introns = []
        for byte in data:
            intron = self.engine.process_egress(byte)
            introns.append(intron)

        # Option 1: classic per-byte ingress (status quo)
        # response = bytearray()
        # for intron in introns:
        #     output_byte = self.engine.process_ingress(intron)
        #     response.append(output_byte)
        # return bytes(response)

        # Option 2: fold to one ingress (batched)
        acc = 0
        for intron in introns:
            acc = governance.fold(acc, intron)

        output_byte = self.engine.process_ingress(acc)
        response = bytes([output_byte])  # one-byte response

        self._commit_if_needed()
        return response

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

        return {"decay_applied": decay_report, "entries_pruned": pruned_count, "timestamp": time.time()}

    def close(self) -> None:
        """Clean shutdown of the agent."""
        self.engine.operator.store.close()

    def _create_default_store(self) -> Any:
        """Create default storage based on configuration."""
        # Canonicalisation enablement consistency: autodetect if flag is None or missing
        enable_phenomenology = self.config.get("enable_phenomenology_storage", None)

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
                private_path = os.path.join(str(private_root), f"{self.agent_id}/knowledge.pkl.gz")
            # Multi-agent overlay using decorators
            public_store = ReadOnlyView(OrbitStore(public_knowledge_path, write_threshold=learn_batch_size))
            private_store = OrbitStore(private_path, write_threshold=learn_batch_size)
            base_store: Any = OverlayView(public_store, private_store)
            phenomenology_map_path = self.config.get("phenomenology_map_path")
        else:
            # Single-agent setup
            knowledge_path = self.config.get("knowledge_path")
            if knowledge_path is None:
                base_path = self.config.get("base_path") or str(self.base_path / "memories")
                knowledge_path = os.path.join(base_path, "knowledge.pkl.gz")
            base_store = OrbitStore(knowledge_path, write_threshold=learn_batch_size)

            # CanonicalView: enable if flag is True, or autodetect if None and file exists
            phenomenology_map_path = self.config.get("phenomenology_map_path")
        # Ensure phenomenology_map_path is always a str before use
        if phenomenology_map_path is None:
            phenomenology_map_path = str(self.base_path / "memories/public/meta/phenomenology_map.json")
        if enable_phenomenology or (
            enable_phenomenology is None and phenomenology_map_path and os.path.exists(phenomenology_map_path)
        ):
            if phenomenology_map_path and os.path.exists(phenomenology_map_path):
                return CanonicalView(base_store, phenomenology_map_path)
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
        self.ontology_path = _abs(ontology_path, self.base_path)
        self.base_knowledge_path = _abs(base_knowledge_path, self.base_path)
        self.preferences = preferences or {}
        self.max_agents = self.preferences.get("max_agents_in_memory", 1000)
        self.eviction_policy = self.preferences.get("agent_eviction_policy", "lru")
        self.allowed_ids = allowed_ids or {"user", "system", "assistant"}
        self.allow_auto_create = allow_auto_create
        self.agent_access_times: Dict[str, float] = {}
        self._shards: List[Dict[str, Any]] = []
        for _ in range(self.SHARD_COUNT):
            if self.eviction_policy == "lru":
                self._shards.append({"agents": LRUAgentCache(self.max_agents // self.SHARD_COUNT), "lock": RLock()})
            else:
                self._shards.append({"agents": OrderedDict(), "lock": RLock()})
        self._public_store: Optional[ReadOnlyView] = ReadOnlyView(
            OrbitStore(self.base_knowledge_path, write_threshold=100)
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
                private_path = os.path.join(self.private_agents_base_path, f"{agent_id}/knowledge.pkl.gz")

                # Ensure directory exists
                os.makedirs(os.path.dirname(private_path), exist_ok=True)

                # Multi-agent overlay using decorators, reuse cached public store
                public_store = self._public_store
                private_store = OrbitStore(private_path, write_threshold=100)
                base_store: Any = OverlayView(public_store, private_store)

                # Create agent config
                config: AgentConfig = {
                    "ontology_path": self.ontology_path,
                    "public_knowledge_path": self.base_knowledge_path,
                    "private_knowledge_path": private_path,
                    "phenomenology_map_path": str(
                        self.preferences.get("phenomenology_map_path") or str(
                            Path(self.ontology_path).with_name("phenomenology_map.json")
                        )
                    ),
                    "private_agents_base_path": str(self.private_agents_base_path),
                    "base_path": str(self.base_path),
                    "enable_phenomenology_storage": bool(self.preferences.get("enable_phenomenology_storage", False)),
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
        phenomap = str(Path(onto).with_name("phenomenology_map.json"))
        if self.private_agents_base_path is None:
            raise ValueError("private_agents_base_path must not be None")
        private_path = os.path.join(str(self.private_agents_base_path), f"{agent_id}/knowledge.pkl.gz")
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
