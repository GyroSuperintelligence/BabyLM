"""
ext_phase_controller.py - Phase Advancement Strategies

This extension provides alternative phase advancement strategies and
phase-based control mechanisms.
"""

from typing import Dict, Any, Deque, List, Callable
from collections import deque
import math

from extensions.base import GyroExtension


class ext_PhaseController(GyroExtension):
    """
    Phase advancement strategies.
    FOOTPRINT: 4 bytes (phase state)
    MAPPING: Controls transition logic between phases
    """

    def __init__(self):
        """Initialize phase controller."""
        # Phase control state (4 bytes)
        self._phase_state: Dict[str, Any] = {
            "strategy": "linear",  # Current advancement strategy
            "modifier": 1,  # Phase advancement modifier
            "skip_count": 0,  # Phases skipped
            "hold_count": 0,  # Phases held
        }

        # Available strategies
        self._strategies: Dict[str, Callable[[int, bool], int]] = {
            "linear": self._linear_advance,
            "accelerated": self._accelerated_advance,
            "decelerated": self._decelerated_advance,
            "harmonic": self._harmonic_advance,
            "quantum": self._quantum_advance,
            "adaptive": self._adaptive_advance,
        }

        # Phase history for adaptive strategies
        self._phase_history: Deque[int] = deque(maxlen=48)

        # Resonance history for adaptation
        self._resonance_history: Deque[bool] = deque(maxlen=48)

        # Phase transition rules
        self._transition_rules: Dict[str, bool] = {
            "skip_non_resonant": False,
            "hold_on_resonance": False,
            "accelerate_on_pattern": False,
            "decelerate_on_boundary": False,
        }

        # Custom phase mappings
        self._phase_mappings: Dict[int, int] = {}

        # Statistics
        self._phase_stats: Dict[str, int] = {
            "total_advances": 0,
            "strategy_changes": 0,
            "phases_skipped": 0,
            "phases_held": 0,
            "effective_cycle_length": 48,
        }

    def ext_advance_phase(self, current_phase: int, resonated: bool) -> int:
        """
        Advance phase using current strategy.

        Args:
            current_phase: Current phase (0-47)
            resonated: Whether resonance occurred

        Returns:
            New phase value
        """
        # Record history
        self._phase_history.append(current_phase)
        self._resonance_history.append(resonated)
        self._phase_stats["total_advances"] += 1

        # Apply transition rules
        if self._should_hold_phase(current_phase, resonated):
            self._phase_state["hold_count"] += 1
            self._phase_stats["phases_held"] += 1
            return current_phase

        if self._should_skip_phase(current_phase, resonated):
            skip_amount = self._calculate_skip_amount(current_phase)
            self._phase_state["skip_count"] += skip_amount
            self._phase_stats["phases_skipped"] += skip_amount
            return (current_phase + skip_amount + 1) % 48

        # Apply current strategy
        strategy_func = self._strategies.get(self._phase_state["strategy"], self._linear_advance)
        new_phase = strategy_func(current_phase, resonated)

        # Apply custom mappings if any
        if new_phase in self._phase_mappings:
            new_phase = self._phase_mappings.get(new_phase, new_phase)

        return new_phase % 48

    def _should_hold_phase(self, phase: int, resonated: bool) -> bool:
        """Determine if phase should be held."""
        if self._transition_rules["hold_on_resonance"] and resonated:
            # Don't hold indefinitely
            return self._phase_state["hold_count"] < 3
        return False

    def _should_skip_phase(self, phase: int, resonated: bool) -> bool:
        """Determine if phase should be skipped."""
        if self._transition_rules["skip_non_resonant"] and not resonated:
            # Skip to next boundary phase
            return phase not in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
        return False

    def _calculate_skip_amount(self, current_phase: int) -> int:
        """Calculate how many phases to skip."""
        # Find next boundary phase
        boundaries = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
        for boundary in boundaries:
            if boundary > current_phase:
                return boundary - current_phase - 1
        # Wrap around to phase 0
        return 48 - current_phase - 1

    # --- Phase Advancement Strategies ---

    def _linear_advance(self, phase: int, resonated: bool) -> int:
        """Standard linear advancement."""
        return phase + 1

    def _accelerated_advance(self, phase: int, resonated: bool) -> int:
        """Accelerated advancement (2x speed)."""
        return phase + 2

    def _decelerated_advance(self, phase: int, resonated: bool) -> int:
        """Decelerated advancement (0.5x speed)."""
        if self._phase_stats["total_advances"] % 2 == 0:
            return phase + 1
        return phase

    def _harmonic_advance(self, phase: int, resonated: bool) -> int:
        """Harmonic advancement based on sine wave."""
        # Advance rate varies harmonically through the cycle
        position = phase / 48.0
        advance_rate = 1 + math.sin(2 * math.pi * position)

        if advance_rate > 1.5:
            return phase + 2
        elif advance_rate < 0.5:
            return phase
        else:
            return phase + 1

    def _quantum_advance(self, phase: int, resonated: bool) -> int:
        """Quantum advancement with discrete jumps."""
        # Jump to specific quantum states
        quantum_states = [0, 6, 12, 18, 24, 30, 36, 42]

        for state in quantum_states:
            if state > phase:
                return state

        return 0  # Wrap to beginning

    def _adaptive_advance(self, phase: int, resonated: bool) -> int:
        """Adaptive advancement based on resonance patterns."""
        # Calculate recent resonance rate
        if len(self._resonance_history) < 10:
            return phase + 1

        recent_resonances = list(self._resonance_history)[-10:]
        resonance_rate = sum(recent_resonances) / len(recent_resonances)

        # Adapt based on resonance rate
        if resonance_rate > 0.7:
            # High resonance - slow down to exploit
            if self._phase_stats["total_advances"] % 3 != 0:
                return phase + 1
            return phase
        elif resonance_rate < 0.3:
            # Low resonance - speed up to find better region
            return phase + 2
        else:
            # Normal advancement
            return phase + 1

    def ext_set_strategy(self, strategy: str) -> bool:
        """
        Set phase advancement strategy.

        Args:
            strategy: Strategy name

        Returns:
            True if strategy was set successfully
        """
        if strategy not in self._strategies:
            return False

        if strategy != self._phase_state["strategy"]:
            self._phase_state["strategy"] = strategy
            self._phase_stats["strategy_changes"] += 1

        return True

    def ext_set_transition_rule(self, rule: str, enabled: bool) -> bool:
        """
        Enable or disable a transition rule.

        Args:
            rule: Rule name
            enabled: Whether to enable the rule

        Returns:
            True if rule was set successfully
        """
        if rule not in self._transition_rules:
            return False

        self._transition_rules[rule] = enabled
        return True

    def ext_add_phase_mapping(self, from_phase: int, to_phase: int) -> None:
        """
        Add custom phase mapping.

        Args:
            from_phase: Source phase
            to_phase: Target phase
        """
        self._phase_mappings[from_phase] = to_phase

    def ext_get_phase_analysis(self) -> Dict[str, Any]:
        """Get comprehensive phase analysis."""
        # Calculate effective cycle length
        if self._phase_history:
            # Look for cycle completion
            cycle_lengths = []
            for i in range(len(self._phase_history) - 1):
                if self._phase_history[i] == 47 and self._phase_history[i + 1] == 0:
                    # Found cycle boundary
                    cycle_start = i - 47
                    if cycle_start >= 0:
                        cycle_lengths.append(i - cycle_start + 1)

            if cycle_lengths:
                self._phase_stats["effective_cycle_length"] = int(
                    sum(cycle_lengths) / len(cycle_lengths)
                )

        return {
            "current_strategy": self._phase_state["strategy"],
            "phase_statistics": self._phase_stats.copy(),
            "transition_rules": self._transition_rules.copy(),
            "resonance_correlation": self._calculate_resonance_correlation(),
            "phase_distribution": self._calculate_phase_distribution(),
        }

    def _calculate_resonance_correlation(self) -> float:
        """Calculate correlation between phase and resonance."""
        if len(self._phase_history) < 10:
            return 0.0

        # Simple correlation: do certain phases have higher resonance?
        phase_resonance_map: Dict[int, float] = {}

        for phase, resonated in zip(self._phase_history, self._resonance_history):
            if phase not in phase_resonance_map:
                phase_resonance_map[phase] = 0.0
            phase_resonance_map[phase] += 1.0 if resonated else 0.0

        # Calculate variance in resonance rates across phases
        resonance_rates = []
        for phase, resonances in phase_resonance_map.items():
            if resonances > 0:
                resonance_rates.append(resonances)

        if len(resonance_rates) > 1:
            mean_rate = sum(resonance_rates) / len(resonance_rates)
            variance = sum((r - mean_rate) ** 2 for r in resonance_rates) / len(resonance_rates)
            return min(variance * 10, 1.0)  # Normalize to 0-1

        return 0.0

    def _calculate_phase_distribution(self) -> Dict[str, int]:
        """Calculate distribution of visited phases."""
        distribution = {f"phase_{i}": 0 for i in range(48)}
        for phase in self._phase_history:
            distribution[f"phase_{phase}"] += 1
        return distribution

    def ext_optimize_strategy(self) -> str:
        """
        Suggest an optimal strategy based on current conditions.

        Returns:
            Selected strategy name
        """
        if len(self._resonance_history) < 48:
            return self._phase_state["strategy"]

        # Try each strategy for one cycle and measure performance
        # (In practice, this would need to be done over time)

        # For now, use simple heuristics
        resonance_rate = sum(self._resonance_history) / len(self._resonance_history)

        if resonance_rate > 0.6:
            # High resonance - use decelerated to exploit
            optimal = "decelerated"
        elif resonance_rate < 0.2:
            # Low resonance - use accelerated to search
            optimal = "accelerated"
        elif self._calculate_resonance_correlation() > 0.5:
            # High correlation - use adaptive
            optimal = "adaptive"
        else:
            # Default to linear
            optimal = "linear"

        self.ext_set_strategy(optimal)
        return optimal

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_phase_controller"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Phase state: 4 bytes as specified
        return 4

    def get_learning_state(self) -> Dict[str, Any]:
        """Phase patterns and optimal strategies."""
        return {
            "phase_statistics": self._phase_stats.copy(),
            "optimal_strategy": self.ext_optimize_strategy(),
            "phase_mappings": self._phase_mappings.copy(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Current phase control state."""
        return {
            "phase_state": self._phase_state.copy(),
            "transition_rules": self._transition_rules.copy(),
            "recent_phases": list(self._phase_history)[-20:],
        }

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore learned patterns."""
        if "phase_statistics" in state:
            self._phase_stats.update(state["phase_statistics"])

        if "phase_mappings" in state:
            self._phase_mappings = state["phase_mappings"]

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore phase control state."""
        if "phase_state" in state:
            self._phase_state.update(state["phase_state"])

        if "transition_rules" in state:
            self._transition_rules.update(state["transition_rules"])
