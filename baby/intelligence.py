"""
S4/5: Intelligence - Orchestration & API

This module provides the IntelligenceEngine and GyroSI classes responsible for
orchestrating the complete system and providing the external API.
"""

# Try to use ujson for speed, fall back to standard json if unavailable
try:
    import ujson as json  # type: ignore[import]
except ImportError:
    import json  # type: ignore
import uuid
import os
import time
from typing import Dict, Any, Optional, List, cast
from collections import OrderedDict
from threading import RLock
from collections import deque
import numpy as np

from baby import governance
from baby.information import InformationEngine
from baby.inference import InferenceEngine
from baby.contracts import CycleHookFunction, AgentConfig, PreferencesConfig
from baby.policies import OrbitStore, CanonicalView, OverlayView, ReadOnlyView


class IntelligenceEngine:
    """
    S4: Strategic Operations & Environment Interface.

    Manages agent state evolution, orchestrates the egress/ingress cycle,
    and implements operational strategies. Handles adaptation to external demands.
    """

    def __init__(self, ontology_path: str, phenotype_store: Any, agent_id: Optional[str] = None):
        """
        Initialize intelligence engine.

        Args:
            ontology_path: Path to discovered ontology data
            phenotype_store: Storage interface for learned knowledge
            agent_id: Unique identifier for this agent instance
        """
        # Initialize subsystem engines
        self.s2 = InformationEngine(self._load_ontology(ontology_path))
        self.operator = InferenceEngine(self.s2, phenotype_store)

        # Agent state
        self.agent_id = agent_id or str(uuid.uuid4())
        self.use_epistemology = False
        epistemology_path = ontology_path.replace("ontology_map.json", "epistemology.npy")
        if os.path.exists(epistemology_path):
            try:
                import numpy as np

                self.epistemology = np.load(epistemology_path, mmap_mode="r")
                self.use_epistemology = True
                print("INFO: State Transition Table (STT) loaded. Using optimized state transitions.")
            except Exception as e:
                print(
                    f"WARNING: Could not load STT from {epistemology_path}. Error: {e}. Falling back to dynamic physics."
                )

        origin_int = self.s2.tensor_to_int(governance.GENE_Mac_S)
        if self.use_epistemology:
            self.current_state_index = self.s2.get_index_from_state(origin_int)
            self.gene_mac_m_int = origin_int  # Ensure always defined
        else:
            self.gene_mac_m_int = origin_int
        self.cycle_count = 0

        # Extension points
        self.post_cycle_hooks: List[CycleHookFunction] = []

        # Algedonic regulation and autonomic cycles
        self._θ_buf = deque(maxlen=128)
        self._θ_high = 0.9   # radians
        self._θ_low  = 0.3
        self._cool_introns = (0b01000010,)
        phenomap = ontology_path.replace("ontology_map.json","phenomenology_map.json")
        try:
            with open(phenomap) as f:
                self._autonomic_cycles = json.load(f).get("autonomic_cycles", [])
        except Exception:
            self._autonomic_cycles = []
        self._pain_streak = 0

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
        # S1: Transcribe input through holographic topology
        intron = governance.transcribe_byte(input_byte)

        # S1: Apply gyroscopic transformation to physical state
        if self.use_epistemology:
            self.current_state_index = self.epistemology[self.current_state_index, intron]
            # No eager sync here
        else:
            self.gene_mac_m_int = governance.apply_gyration_and_transform(self.gene_mac_m_int, intron)
            self._sync_index_from_state_int()

        self.cycle_count += 1

        # Record divergence in θ buffer
        div = self.s2.measure_state_divergence(
            self.gene_mac_m_int if not self.use_epistemology
            else self.s2.get_state_from_index(self.current_state_index))
        self._θ_buf.append(div)
        return intron

    def process_ingress(self, last_intron: int) -> int:
        """
        Process Intelligence Ingress: Learn and respond.

        This is the "inward" phase where the system integrates experience
        and generates intelligent response based on accumulated knowledge.

        Args:
            last_intron: Intron from the previous egress phase

        Returns:
            Output byte representing intelligent response
        """
        # S3: Get semantic meaning of current state + context
        if self.use_epistemology:
            state_index = self.current_state_index
        else:
            state_index = self.s2.get_index_from_state(self.gene_mac_m_int)
        phenotype_entry = self.operator.get_phenotype(state_index, last_intron)

        # S3: Learn through gyrogroup coaddition
        self.operator.learn(phenotype_entry, last_intron)

        # Execute post-cycle hooks for monitoring/maintenance
        for hook in self.post_cycle_hooks:
            hook(self, phenotype_entry, last_intron)

        # Generate response from phenotype
        phenotype = phenotype_entry.get("phenotype")
        if isinstance(phenotype, str) and phenotype:
            return ord(phenotype[0])
        elif isinstance(phenotype, str):
            return ord("?")
        elif phenotype is not None:
            return int(phenotype) & 0xFF  # Ensure byte range
        else:
            return ord("?")

        # Algedonic decision at start
        θ = np.mean(self._θ_buf) if self._θ_buf else 0.0
        if θ > self._θ_high:
            self._pain_streak += 1
            last_intron = self._cool_introns[self.cycle_count % len(self._cool_introns)]
            if self._pain_streak > 256 and self._autonomic_cycles:
                for intr in self._autonomic_cycles[self.cycle_count % len(self._autonomic_cycles)]:
                    self.process_egress(intr)
                    self.operator.learn(
                        self.operator.get_phenotype(
                            self.current_state_index if self.use_epistemology
                            else self.s2.get_index_from_state(self.gene_mac_m_int),
                            intr),
                        intr)
                self._pain_streak = 0
        elif θ < self._θ_low:
            self._pain_streak = 0

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
        Learn from a batch of data using streaming gyrogroup coaddition.

        This method allows efficient batch learning while preserving the
        path-dependent nature of the coaddition operation. Uses O(1) memory
        by streaming the coaddition instead of collecting all introns.

        Args:
            data: Batch of bytes to learn from
        """
        if not data:
            return

        # Streaming coaddition: O(1) memory instead of O(N)
        acc = 0
        for byte in data:
            # Use the optimized process_egress method (STT-aware)
            intron = self.process_egress(byte)
            acc = governance.coadd(acc, intron)
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

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current agent state.

        Returns:
            Dictionary with state information
        """
        self._sync_state_fields_from_index()
        angular_divergence = self.s2.measure_state_divergence(self.gene_mac_m_int)

        return {
            "agent_id": self.agent_id,
            "cycle_count": self.cycle_count,
            "state_integer": self.gene_mac_m_int,
            "tensor_index": self.s2.get_index_from_state(self.gene_mac_m_int),
            "angular_divergence_radians": float(angular_divergence),
            "angular_divergence_degrees": float(angular_divergence * 180 / 3.14159),
            "active_hooks": len(self.post_cycle_hooks),
        }

    def reset_to_archetypal_state(self) -> None:
        """Reset agent to the archetypal state (GENE_Mac_S)."""
        self.gene_mac_m_int = self.s2.tensor_to_int(governance.GENE_Mac_S)
        self._sync_index_from_state_int()
        self.cycle_count = 0

    def _load_ontology(self, ontology_path: str) -> Dict[str, Any]:
        """Loads the ontology data from a JSON file as ManifoldData."""
        with open(ontology_path, "r") as f:
            data = json.load(f)
        return data  # type: ignore

    def _sync_state_fields_from_index(self):
        if self.use_epistemology:
            self.gene_mac_m_int = self.s2.get_state_from_index(self.current_state_index)

    def _sync_index_from_state_int(self):
        if self.use_epistemology:
            self.current_state_index = self.s2.get_index_from_state(self.gene_mac_m_int)


class GyroSI:
    """
    S5: Whole System Identity & Policy.

    The outermost viable system boundary that encapsulates the entire VSM stack.
    Manages configuration, agent identity, and provides the stable external API.
    """

    def __init__(self, config: AgentConfig, agent_id: Optional[str] = None, phenotype_store: Optional[Any] = None):
        """
        Initialize GyroSI with configuration.

        Args:
            config: Agent configuration dictionary
            agent_id: Unique agent identifier
            phenotype_store: Optional custom storage implementation
        """
        self.config = config
        self.agent_id = agent_id or str(uuid.uuid4())

        # Initialize storage if not provided
        if phenotype_store is None:
            phenotype_store = self._create_default_store()

        # Initialize core engine
        ontology_path = config.get("ontology_path")
        if not ontology_path:
            raise ValueError("AgentConfig must include 'ontology_path'.")
        self.engine = IntelligenceEngine(
            ontology_path=ontology_path, phenotype_store=phenotype_store, agent_id=self.agent_id
        )

    def ingest(self, data: bytes) -> None:
        """
        Learn from a batch of data using ordered gyrogroup coaddition.

        This is the primary learning interface. Data is processed as a batch
        with the final state determining the learning context.

        Args:
            data: Bytes to learn from
        """
        self.engine.batch_learn(data)
        # Ensure pending writes are flushed
        store = self.engine.operator.store
        if isinstance(store, OrbitStore):
            cast(OrbitStore, store).commit()
        elif hasattr(store, "close"):
            store.close()

    def respond(self, data: bytes) -> bytes:
        """
        Generate an intelligent response to input data.

        Each byte of input triggers an egress→ingress cycle, producing
        one byte of intelligent output based on accumulated knowledge.

        Args:
            data: Input bytes to respond to

        Returns:
            Intelligent response bytes
        """
        response = bytearray()

        if not data:
            return bytes(response)

        for byte in data:
            # Egress: Transform input into internal action
            intron = self.engine.process_egress(byte)

            # Ingress: Learn and generate intelligent response
            output_byte = self.engine.process_ingress(intron)
            response.append(output_byte)

        # Ensure pending writes are flushed
        store = self.engine.operator.store
        if isinstance(store, OrbitStore):
            cast(OrbitStore, store).commit()
        elif hasattr(store, "close"):
            store.close()
        return bytes(response)

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
            "knowledge_statistics": self.engine.operator.get_knowledge_statistics(),
            "system_integrity": self.engine.operator.validate_knowledge_integrity(),
        }

    def add_monitoring_hook(self, hook: CycleHookFunction) -> None:
        """Add a monitoring hook to the intelligence engine."""
        self.engine.add_hook(hook)

    def apply_maintenance(self, decay_factor: float = 0.999, confidence_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Apply maintenance operations to the knowledge base.

        Args:
            decay_factor: Confidence decay factor for aging entries
            confidence_threshold: Minimum confidence for entry retention

        Returns:
            Maintenance report
        """
        # Apply confidence decay
        decay_report = self.engine.operator.apply_confidence_decay(decay_factor)

        # Prune low-confidence entries
        pruned_count = self.engine.operator.prune_low_confidence_entries(confidence_threshold)

        return {"decay_applied": decay_report, "entries_pruned": pruned_count, "timestamp": time.time()}

    def close(self) -> None:
        """Clean shutdown of the agent."""
        self.engine.operator.store.close()

    def _create_default_store(self) -> Any:
        """Create default storage based on configuration."""
        # Check if phenomenology storage is enabled
        enable_phenomenology = self.config.get("enable_phenomenology_storage", False)

        # Create base store
        public_knowledge_path = self.config.get("public_knowledge_path")
        batch_size = self.config.get("batch_size", 100)
        if public_knowledge_path is not None:
            # Multi-agent setup with public/private knowledge
            private_path = self.config.get("private_knowledge_path")
            if private_path is None:
                private_path = f"memories/private/agents/{self.agent_id}/knowledge.pkl.gz"
            # Multi-agent overlay using decorators
            public_store = ReadOnlyView(OrbitStore(public_knowledge_path, write_threshold=batch_size))
            private_store = OrbitStore(private_path, write_threshold=batch_size)
            base_store = OverlayView(public_store, private_store)
        else:
            # Single-agent setup
            knowledge_path = self.config.get("knowledge_path")
            if knowledge_path is None:
                knowledge_path = "memories/knowledge.pkl.gz"
            base_store = OrbitStore(knowledge_path, write_threshold=batch_size)

        # Wrap with phenomenology decorator if enabled
        if enable_phenomenology:
            phenomenology_map_path = self.config.get("phenomenology_map_path")
            if phenomenology_map_path is None:
                phenomenology_map_path = "memories/public/meta/phenomenology_map.json"
            if os.path.exists(phenomenology_map_path):
                return CanonicalView(base_store, phenomenology_map_path)
            else:
                print(f"Warning: phenomenology map not found at {phenomenology_map_path}")

        return base_store


class LRUAgentCache(OrderedDict):
    """LRU cache for agent instances with size limit."""

    def __init__(self, max_size: int, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)
        self._lock = RLock()

    def __getitem__(self, key):
        with self._lock:
            # Move to end on access
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

    def __setitem__(self, key, value):
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
    """Manages a collection of independent GyroSI agents with eviction policy."""

    def __init__(self, ontology_path: str, base_knowledge_path: str, preferences: Optional[PreferencesConfig] = None):
        """
        Initialize agent pool.

        Args:
            ontology_path: Path to physical ontology data
            base_knowledge_path: Path to shared knowledge base
            preferences: Optional preferences configuration
        """
        self.ontology_path = ontology_path
        self.base_knowledge_path = base_knowledge_path

        # Load preferences
        self.preferences = preferences or {}
        max_agents = self.preferences.get("max_agents_in_memory", 1000)
        self.eviction_policy = self.preferences.get("agent_eviction_policy", "lru")

        # Initialize agent storage based on eviction policy
        if self.eviction_policy == "lru":
            self.agents = LRUAgentCache(max_agents)
        else:
            # Simple dict with manual eviction
            self.agents: Dict[str, GyroSI] = {}
            self.agent_access_times: Dict[str, float] = {}
            self.max_agents = max_agents

        self._lock = RLock()

    def get_or_create_agent(self, agent_id: str, role_hint: Optional[str] = None) -> GyroSI:
        """
        Retrieve existing agent or create new one.

        Args:
            agent_id: Unique agent identifier
            role_hint: Optional role metadata (not used by physics)

        Returns:
            GyroSI agent instance
        """
        with self._lock:
            # Update access time for TTL-based eviction
            if self.eviction_policy == "ttl":
                self.agent_access_times[agent_id] = time.time()
                self._evict_expired_agents()

            if agent_id not in self.agents:
                # Check if we need to evict before creating new
                if not isinstance(self.agents, LRUAgentCache):
                    self._maybe_evict_agent()

                # Create private knowledge path
                private_path = f"memories/private/agents/{agent_id}/knowledge.pkl.gz"

                # Ensure directory exists
                os.makedirs(os.path.dirname(private_path), exist_ok=True)

                # Multi-agent overlay using decorators
                public_store = ReadOnlyView(OrbitStore(self.base_knowledge_path, write_threshold=100))
                private_store = OrbitStore(private_path, write_threshold=100)
                store = OverlayView(public_store, private_store)

                # Create agent config
                config: AgentConfig = {
                    "ontology_path": self.ontology_path,
                    "public_knowledge_path": self.base_knowledge_path,
                    "private_knowledge_path": private_path,
                    "enable_phenomenology_storage": self.preferences.get("enable_phenomenology_storage", False),
                }

                # Add role hint to metadata if provided
                if role_hint:
                    config["agent_metadata"] = {"role_hint": role_hint}

                self.agents[agent_id] = GyroSI(config=config, agent_id=agent_id, phenotype_store=store)

            return self.agents[agent_id]

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent from pool.

        Args:
            agent_id: Agent to remove

        Returns:
            True if agent was found and removed
        """
        with self._lock:
            if agent_id in self.agents:
                self.agents[agent_id].close()
                del self.agents[agent_id]
                if agent_id in self.agent_access_times:
                    del self.agent_access_times[agent_id]
                return True
            return False

    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        with self._lock:
            return list(self.agents.keys())

    def close_all(self) -> None:
        """Close all agents in the pool."""
        with self._lock:
            for agent in self.agents.values():
                agent.close()
            self.agents.clear()
            self.agent_access_times.clear()

    def _maybe_evict_agent(self) -> None:
        """Evict agent if at capacity (non-LRU policies)."""
        if len(self.agents) >= self.max_agents:
            if self.eviction_policy == "lfu":
                # Evict least frequently used (would need usage tracking)
                # For now, just evict oldest
                oldest_id = next(iter(self.agents))
            elif self.eviction_policy == "ttl":
                # Evict oldest by access time
                oldest_id = min(self.agent_access_times, key=lambda k: self.agent_access_times[k])
            else:
                # Default: evict oldest
                oldest_id = next(iter(self.agents))

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


def orchestrate_turn(pool: AgentPool, user_id: str, assistant_id: str, user_input: str) -> str:
    """
    Orchestrate a single conversational turn between agents.

    Args:
        pool: Agent pool containing the agents
        user_id: User agent identifier
        assistant_id: Assistant agent identifier
        user_input: User's input text

    Returns:
        Assistant's response text
    """
    # Get participating agents
    user_agent = pool.get_or_create_agent(user_id, role_hint="user")
    assistant_agent = pool.get_or_create_agent(assistant_id, role_hint="assistant")

    # User agent processes input, creating stimulus
    stimulus = user_agent.respond(user_input.encode("utf-8"))

    # Assistant responds to stimulus
    response = assistant_agent.respond(stimulus)

    # Decode response, handling potential encoding issues
    try:
        return response.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback for non-UTF8 responses
        return response.decode("utf-8", errors="replace")
