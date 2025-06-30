"""
g3_inference.py - Inference Engine / GyroCodec

Pattern detection, compression, and key derivation engine.
Operates on accepted op-pairs to identify cycles and patterns.
"""

import torch
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, Counter
import s1_governance


@dataclass
class CompressedBlock:
    """Compressed representation of a cycle or pattern."""

    block_type: str  # "cycle_repeat" or "full_cycle"
    data: Any  # Either repeat count or full op-pair list


@dataclass
class PatternPromotion:
    """New pattern discovered and assigned a short code."""

    pattern_id: str
    op_sequence: List[Tuple[int, int]]
    frequency: int


@dataclass
class GeneSnapshot:
    """Current gene state fingerprint."""

    snapshot_hash: bytes
    step_count: int
    gene_state: Dict[str, torch.Tensor]


class InferenceEngine:
    """
    Pattern recognition and compression engine.
    Learns repeated sequences and provides key derivation.
    """

    def __init__(self, pattern_threshold: int = 3, agent_uuid: Optional[str] = None):
        """
        Initialize the inference engine.

        Args:
            pattern_threshold: Minimum occurrences before pattern promotion
            agent_uuid: Optional agent UUID for unique gene state initialization
        """
        self.pattern_threshold = pattern_threshold
        self.cycle_history = deque(maxlen=100)  # Keep last 100 cycles
        self.pattern_candidates = Counter()
        self.promoted_patterns = {}
        self.next_pattern_id = 0
        self.step_count = 0
        self.agent_uuid = agent_uuid

        # Gene state tracking for key derivation
        self.current_gene_state: Optional[Dict[str, torch.Tensor]] = None
        self._initialize_gene_state()

        # Safe pruning configuration - VERY conservative thresholds
        self.HORIZON_CUT = 0.01  # Only prune if very close to 50/50 (was 0.02)
        self.ENTROPY_CUT = 0.03  # Only prune if very low diversity (was 0.05)
        self.prune_stats = {"pruned": 0, "total": 0}  # Track pruning statistics

    def _initialize_gene_state(self):
        """Initialize the gene state from S1 governance (shared, constant for all agents)."""
        self.current_gene_state = s1_governance.get_gene_tensors()
        # No agent-specific mutation. The gene is a universal topological space.

    def apply_op_pair(self, op_pair: Tuple[int, int]):
        """
        Apply a single operation pair to the current gene state.

        Args:
            op_pair: (op_code, tensor_id) to apply
        """
        op_code, tensor_id = op_pair
        tensor_key = f"id_{tensor_id}"
        if self.current_gene_state is None:
            raise RuntimeError("Gene state not initialized")
        self.current_gene_state[tensor_key] = s1_governance.gyration_op(
            self.current_gene_state[tensor_key],
            op_code,
            clone=False,  # Mutate in place for efficiency
        )
        self.step_count += 1

    def process_cycle_complete(self, cycle_ops: List[Tuple[int, int]]) -> List:
        """
        Process a completed 48-op cycle.

        Args:
            cycle_ops: List of 48 operation pairs from the cycle

        Returns:
            List of emitted events
        """
        events = []

        # Apply all operations to gene state
        for op_pair in cycle_ops:
            self.apply_op_pair(op_pair)

        # Convert to hashable representation
        cycle_tuple = tuple(cycle_ops)

        # Check if this cycle is a repeat
        is_repeat = cycle_tuple in [tuple(c) for c in self.cycle_history]

        if is_repeat:
            # Find how many times it repeated
            repeat_count = sum(1 for c in self.cycle_history if tuple(c) == cycle_tuple)
            events.append(
                CompressedBlock(
                    block_type="cycle_repeat",
                    data={
                        "count": repeat_count,
                        "hash": hashlib.sha256(str(cycle_tuple).encode()).hexdigest()[:8],
                    },
                )
            )
        else:
            # New cycle - store full data
            events.append(CompressedBlock(block_type="full_cycle", data=cycle_ops))

        # Add to history
        self.cycle_history.append(cycle_ops)

        # Look for patterns within the cycle
        pattern_events = self._detect_patterns(cycle_ops)
        events.extend(pattern_events)

        # Add a gene snapshot after processing the cycle
        events.append(self.get_gene_snapshot())

        return events

    def _detect_patterns(self, ops: List[Tuple[int, int]]) -> List[PatternPromotion]:
        """
        Detect repeated patterns of 2-16 operations.

        Args:
            ops: List of operation pairs to analyze

        Returns:
            List of newly promoted patterns
        """
        promotions = []

        # Check patterns of different lengths
        for length in range(2, min(17, len(ops) + 1)):
            for start in range(len(ops) - length + 1):
                pattern = tuple(ops[start : start + length])
                self.pattern_candidates[pattern] += 1

                if self.pattern_candidates[pattern] >= self.pattern_threshold:
                    # Always emit a promotion event if threshold is met
                    if pattern not in self.promoted_patterns.values():
                        # First time promotion: assign a new pattern_id
                        pattern_id = f"P{self.next_pattern_id:04d}"
                        self.next_pattern_id += 1
                        self.promoted_patterns[pattern_id] = pattern
                    else:
                        # Find the existing pattern_id
                        pattern_id = next(
                            pid for pid, seq in self.promoted_patterns.items() if seq == pattern
                        )

                    promotions.append(
                        PatternPromotion(
                            pattern_id=pattern_id,
                            op_sequence=list(pattern),
                            frequency=self.pattern_candidates[pattern],
                        )
                    )

        return promotions

    def get_gene_snapshot(self) -> GeneSnapshot:
        """
        Generate a fingerprint of the current gene traversal state.

        Returns:
            GeneSnapshot with hash and step count
        """
        if self.current_gene_state is None:
            raise RuntimeError("Gene state not initialized")
        hasher = hashlib.sha256()
        hasher.update(self.current_gene_state["id_0"].numpy().tobytes())
        hasher.update(self.current_gene_state["id_1"].numpy().tobytes())
        hasher.update(str(self.step_count).encode())
        return GeneSnapshot(
            snapshot_hash=hasher.digest(),
            step_count=self.step_count,
            gene_state={
                "id_0": self.current_gene_state["id_0"].clone(),
                "id_1": self.current_gene_state["id_1"].clone(),
            },
        )

    def predict_next_operation(self, curriculum: Dict) -> Tuple[int, int]:
        """
        Predict the next operation based on learned patterns.

        Args:
            curriculum: Dictionary of learned patterns

        Returns:
            Predicted next operation pair (op_code, tensor_id)
        """
        # Check if we have any patterns in our history
        if not self.cycle_history or not curriculum.get("patterns"):
            # Default to identity operation if no patterns available
            return (0, 0)

        # Get recent history (last few operations)
        recent_history = []
        for cycle in reversed(list(self.cycle_history)):
            recent_history = list(cycle) + recent_history
            if len(recent_history) >= 32:  # Look at last 32 ops max
                break

        recent_history = recent_history[-32:]

        # Look for the longest matching pattern
        best_match = None
        best_match_length = 0
        next_op = None

        for pattern_id, pattern_info in curriculum.get("patterns", {}).items():
            pattern_sequence = [(op["op"], op["tensor"]) for op in pattern_info.get("sequence", [])]
            if not pattern_sequence:
                continue

            pattern_length = len(pattern_sequence)

            # Skip if pattern is longer than our history
            if pattern_length > len(recent_history):
                continue

            # Check if the end of our history matches this pattern
            if tuple(recent_history[-pattern_length:]) == tuple(pattern_sequence):
                if pattern_length > best_match_length:
                    best_match = pattern_id
                    best_match_length = pattern_length

                    # If this is a recurring pattern, predict it continues
                    if pattern_length < len(pattern_sequence):
                        next_op = pattern_sequence[0]  # First op of pattern

        # If we found a matching pattern, predict it continues
        if best_match and next_op:
            return next_op

        # Default to identity operation if no pattern matches
        return (0, 0)

    def get_state(self) -> Dict:
        """Return current engine state."""
        return {
            "cycle_history_size": len(self.cycle_history),
            "pattern_candidates": len(self.pattern_candidates),
            "promoted_patterns": len(self.promoted_patterns),
            "step_count": self.step_count,
        }

    def analyse_cycle(
        self, op_pairs: List[Tuple[int, int]], resonance_flags: List[bool]
    ) -> Dict[str, Any]:
        """
        Analyze a cycle for pruning decisions using safe, conservative thresholds.

        Args:
            op_pairs: List of 48 (op_code, tensor_id) pairs
            resonance_flags: List of 48 boolean resonance flags

        Returns:
            Dict with analysis metrics and prune decision
        """
        if len(op_pairs) != 48 or len(resonance_flags) != 48:
            raise ValueError(
                f"Expected 48 op_pairs and 48 resonance_flags, got {len(op_pairs)} and {len(resonance_flags)}"
            )

        # Calculate horizon distance (how close to 50/50 resonance)
        aligned_count = sum(1 for flag in resonance_flags if flag)
        horizon_distance = abs(aligned_count / 48.0 - 0.5)

        # Calculate pattern entropy (diversity of op-pairs)
        unique_ops = len(set(op_pairs))
        pattern_entropy = unique_ops / 48.0

        # Safe pruning decision - only prune if BOTH metrics are below thresholds
        # This is more conservative than A2's "either" logic
        horizon_prune = horizon_distance < self.HORIZON_CUT
        entropy_prune = pattern_entropy < self.ENTROPY_CUT

        # Only prune if BOTH conditions are met (extra safety)
        should_prune = horizon_prune and entropy_prune

        # Update statistics
        self.prune_stats["total"] += 1
        if should_prune:
            self.prune_stats["pruned"] += 1

        # Log detailed information for monitoring
        if (
            should_prune or self.prune_stats["total"] % 100 == 0
        ):  # Log every 100th cycle or when pruning
            print(
                f"Cycle analysis: horizon={horizon_distance:.3f} (cut={self.HORIZON_CUT}), "
                f"entropy={pattern_entropy:.3f} (cut={self.ENTROPY_CUT}), "
                f"prune={should_prune}, "
                f"stats={self.prune_stats['pruned']}/{self.prune_stats['total']}"
            )

        return {
            "horizon_distance": horizon_distance,
            "pattern_entropy": pattern_entropy,
            "aligned_count": aligned_count,
            "unique_ops": unique_ops,
            "horizon_prune": horizon_prune,
            "entropy_prune": entropy_prune,
            "prune": should_prune,
            "prune_reason": "both_metrics_low" if should_prune else "none",
        }
