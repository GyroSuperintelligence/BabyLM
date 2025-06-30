"""
g3_inference.py - Inference Engine / GyroCodec

Pattern detection, compression, and key derivation engine.
Operates on accepted op-pairs to identify cycles and patterns.

Device logic: All tensors are created on the selected device (GPU if available, else CPU).
"""

import torch

# Select device for all tensors and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import deque, Counter


@dataclass
class CompressedBlock:
    """
    Event for a compressed block of information.

    Attributes:
        block_type: Type of block ("full_cycle", "cycle_repeat", "pruned_cycle")
        data: Dictionary containing block-specific data
    """

    block_type: str
    data: Dict[str, Any]


@dataclass
class PatternPromotion:
    """
    Event for when a new pattern is identified and promoted.

    Attributes:
        pattern_hash: Unique hash identifying the pattern
        pattern: List of operation pairs constituting the pattern
        frequency: Number of times the pattern was observed
        cycle_ops: Complete cycle operations where pattern was found
    """

    pattern_hash: str
    pattern: List[Tuple[int, int]]
    frequency: int
    cycle_ops: List[Tuple[int, int]]


class InferenceEngine:
    """
    Analyzes sequences of operation pairs for patterns and compression opportunities.
    This engine is purely analytical and does not maintain a mutable gene state.

    Key responsibilities:
    1. Detect repeated cycles and emit compressed blocks
    2. Identify recurring patterns and promote them to the curriculum
    3. Analyze resonance patterns for pruning decisions
    4. Generate predictions based on learned patterns
    """

    # Constants for pruning analysis
    HORIZON_CUT = 0.01  # Prune if resonance is within 1% of the 50/50 horizon
    ENTROPY_CUT = 0.03  # Prune if pattern diversity is less than 3%

    # Constants for pattern recognition
    MIN_PATTERN_LENGTH = 2  # Shortest pattern to recognize
    MAX_PATTERN_LENGTH = 16  # Longest pattern to recognize

    def __init__(
        self,
        pattern_threshold: int = 5,
        agent_uuid: Optional[str] = None,
        min_pattern_length: Optional[int] = None,
        max_pattern_length: Optional[int] = None,
    ):
        """
        Initialize the Inference Engine.

        Args:
            pattern_threshold: Min frequency for a pattern to be promoted
            agent_uuid: Optional agent UUID for context
            min_pattern_length: Optional override for minimum pattern length
            max_pattern_length: Optional override for maximum pattern length
        """
        self.pattern_threshold = pattern_threshold
        self.agent_uuid = agent_uuid
        self.MIN_PATTERN_LENGTH = (
            min_pattern_length
            if min_pattern_length is not None
            else InferenceEngine.MIN_PATTERN_LENGTH
        )
        self.MAX_PATTERN_LENGTH = (
            max_pattern_length
            if max_pattern_length is not None
            else InferenceEngine.MAX_PATTERN_LENGTH
        )

        # Pattern storage
        self.promoted_patterns = {}  # Hash -> pattern
        self.pattern_counts = Counter()  # Pattern -> count

        # Cycle history for compression
        self.cycle_history = {}  # Hash -> {ops, count}

        # Statistics
        self.prune_stats = {"total": 0, "pruned": 0, "reasons": {}}

        # Sliding window for pattern detection
        self.sliding_window = deque(maxlen=self.MAX_PATTERN_LENGTH * 2)

    def process_cycle_complete(
        self, op_pairs: List[Tuple[int, int]], resonance_flags: List[bool]
    ) -> List[Any]:
        """
        Process a completed cycle of op-pairs for pattern detection and compression.

        Args:
            op_pairs: List of op-pairs from the completed cycle
            resonance_flags: List of resonance flags for pruning analysis

        Returns:
            List of generated events (CompressedBlock, PatternPromotion)
        """
        events = []

        # More flexible validation for tests
        if len(op_pairs) != len(resonance_flags):
            raise ValueError(
                f"Op pairs and resonance flags must match, got {len(op_pairs)} and {len(resonance_flags)}"
            )

        # Create cycle hash for detection
        cycle_str = "".join([f"{op[0]},{op[1]};" for op in op_pairs])
        cycle_hash = hashlib.sha256(cycle_str.encode("utf-8")).hexdigest()[:8]

        # Check for cycle repetition
        if cycle_hash in self.cycle_history:
            # This is a repeated cycle
            self.cycle_history[cycle_hash]["count"] += 1
            repeat_block = CompressedBlock(
                block_type="cycle_repeat",
                data={"hash": cycle_hash, "count": self.cycle_history[cycle_hash]["count"]},
            )
            events.append(repeat_block)
        else:
            # This is a new cycle
            self.cycle_history[cycle_hash] = {
                "ops": op_pairs.copy(),
                "count": 1,
                "index": len(self.cycle_history),
            }
            full_cycle_block = CompressedBlock(
                block_type="full_cycle", data={"hash": cycle_hash, "ops": op_pairs.copy()}
            )
            events.append(full_cycle_block)

        # Pattern detection - look for recurring patterns
        for pattern_len in range(
            self.MIN_PATTERN_LENGTH, min(self.MAX_PATTERN_LENGTH, len(op_pairs) // 2)
        ):
            for i in range(len(op_pairs) - pattern_len + 1):
                # Extract potential pattern
                pattern = tuple(op_pairs[i : i + pattern_len])

                # Update pattern count
                self.pattern_counts[pattern] += 1

                # Check if pattern meets threshold for promotion
                if self.pattern_counts[pattern] >= self.pattern_threshold:
                    pattern_list = list(pattern)
                    pattern_str = "".join([f"{op[0]},{op[1]};" for op in pattern_list])
                    pattern_hash = hashlib.sha256(pattern_str.encode("utf-8")).hexdigest()[:8]

                    # Only promote if not already promoted
                    if pattern_hash not in self.promoted_patterns:
                        self.promoted_patterns[pattern_hash] = pattern_list
                        events.append(
                            PatternPromotion(
                                pattern_hash=pattern_hash,
                                pattern=pattern_list,
                                frequency=self.pattern_counts[pattern],
                                cycle_ops=op_pairs.copy(),
                            )
                        )

        return events

    def analyse_cycle(
        self, op_pairs: List[Tuple[int, int]], resonance_flags: List[bool]
    ) -> Dict[str, Any]:
        """
        Perform pruning analysis on a completed cycle.

        Args:
            op_pairs: List of 48 op-pairs
            resonance_flags: List of 48 boolean resonance flags

        Returns:
            Dictionary with analysis metrics and pruning decision
        """
        self.prune_stats["total"] += 1

        # 1. Horizon distance (how far from 50/50 resonance)
        aligned_count = sum(resonance_flags)
        horizon_dist = abs(aligned_count / 48 - 0.5)

        # 2. Pattern entropy (diversity of op-pairs)
        unique_ops = len(set(op_pairs))
        pattern_entropy = unique_ops / 48.0

        # Pruning decision
        prune = False
        prune_reason = "none"

        if horizon_dist < self.HORIZON_CUT:
            prune = True
            prune_reason = "horizon_proximity"
        elif pattern_entropy < self.ENTROPY_CUT:
            prune = True
            prune_reason = "low_entropy"

        if prune:
            self.prune_stats["pruned"] += 1
            self.prune_stats["reasons"][prune_reason] = (
                self.prune_stats["reasons"].get(prune_reason, 0) + 1
            )

        return {
            "prune": prune,
            "prune_reason": prune_reason,
            "horizon_distance": horizon_dist,
            "pattern_entropy": pattern_entropy,
            "aligned_count": aligned_count,
            "unique_ops": unique_ops,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return current engine state."""
        return {
            "promoted_patterns": len(self.promoted_patterns),
            "cycle_history_size": len(self.cycle_history),
            "prune_stats": self.prune_stats,
            "top_patterns": dict(self.pattern_counts.most_common(5)),
        }

    def predict_next_operation(self, curriculum: Dict[str, Any]) -> Tuple[int, int]:
        """
        Predict the next operation based on learned patterns.

        Args:
            curriculum: Dictionary of learned patterns

        Returns:
            Predicted next operation pair (op_code, tensor_id)
        """
        # Default to identity operation if no patterns
        if not curriculum or "patterns" not in curriculum or not curriculum["patterns"]:
            return (7, 0)

        # Find the most frequent pattern
        best_pattern = None
        highest_freq = 0

        for pattern_id, pattern_info in curriculum["patterns"].items():
            if pattern_info.get("frequency", 0) > highest_freq:
                highest_freq = pattern_info["frequency"]
                pattern_seq = pattern_info.get("sequence", [])
                if pattern_seq:
                    # Extract first op-pair from the sequence
                    best_pattern = (pattern_seq[0]["op"], pattern_seq[0]["tensor"])

        return best_pattern if best_pattern else (7, 0)

    def reset(self) -> None:
        """Reset the engine to its initial state."""
        self.promoted_patterns = {}
        self.pattern_counts = Counter()
        self.cycle_history = {}
        self.prune_stats = {"total": 0, "pruned": 0, "reasons": {}}
        self.sliding_window.clear()
