"""
ext_resonance_processor.py - Structural Resonance Processing

This extension analyzes and processes structural resonance patterns,
detecting cyclic patterns and modulating resonance response.
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np

from extensions.base import GyroExtension


class ext_ResonanceProcessor(GyroExtension):
    """
    Structural resonance processing.
    FOOTPRINT: 8-12 bytes (resonance cache)
    MAPPING: Detects and processes structural resonance events
    """

    def __init__(self):
        """Initialize resonance processor."""
        # Resonance cache (8-12 bytes)
        self._resonance_cache = {
            "last_resonance_phase": -1,  # 2 bytes
            "resonance_strength": 0.0,  # 4 bytes
            "pattern_id": 0,  # 2 bytes
            "modulation_factor": 1.0,  # 4 bytes
        }

        # Resonance history for pattern detection
        self._resonance_events = deque(maxlen=144)  # 3 full cycles

        # Detected resonance patterns
        self._resonance_patterns = {}
        self._pattern_counter = 0

        # Resonance field map (phase -> resonance probability)
        self._resonance_field = np.zeros(48, dtype=np.float32)
        self._field_samples = np.zeros(48, dtype=np.int32)

        # Harmonic analysis
        self._harmonic_components = {
            "fundamental": {"frequency": 1 / 48, "amplitude": 0.0, "phase": 0.0},
            "second": {"frequency": 2 / 48, "amplitude": 0.0, "phase": 0.0},
            "third": {"frequency": 3 / 48, "amplitude": 0.0, "phase": 0.0},
        }

        # Resonance statistics
        self._resonance_stats = {
            "total_resonances": 0,
            "unique_patterns": 0,
            "average_strength": 0.0,
            "peak_phase": -1,
            "peak_strength": 0.0,
        }

        # Cyclic pattern detection
        self._cycle_detector = {
            "buffer": deque(maxlen=96),  # 2 cycles
            "detected_periods": set(),
            "confidence": {},
        }

    def ext_process_resonance(
        self, phase: int, resonated: bool, input_byte: int, ops: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Process a resonance event.

        Args:
            phase: Current phase
            resonated: Whether resonance occurred
            input_byte: Input that caused the event
            ops: Operator codes if resonance occurred

        Returns:
            Resonance analysis results
        """
        # Record event
        event = {
            "phase": phase,
            "resonated": resonated,
            "input_byte": input_byte,
            "ops": ops,
            "strength": 0.0,
        }

        # Calculate resonance strength
        if resonated and ops:
            event["strength"] = self._calculate_resonance_strength(phase, input_byte, ops)

        self._resonance_events.append(event)

        # Update resonance field
        self._update_resonance_field(phase, event["strength"])

        # Update cache
        if resonated:
            self._resonance_cache["last_resonance_phase"] = phase
            self._resonance_cache["resonance_strength"] = event["strength"]
            self._resonance_stats["total_resonances"] += 1

        # Detect patterns
        patterns = self._detect_resonance_patterns()

        # Update harmonics
        if len(self._resonance_events) >= 48:
            self._analyze_harmonics()

        # Detect cycles
        self._update_cycle_detection(phase, resonated)

        return {
            "resonated": resonated,
            "strength": event["strength"],
            "field_value": self._resonance_field[phase],
            "patterns": patterns,
            "harmonics": self._get_harmonic_summary(),
            "modulation": self._resonance_cache["modulation_factor"],
        }

    def _calculate_resonance_strength(
        self, phase: int, input_byte: int, ops: Tuple[int, int]
    ) -> float:
        """Calculate strength of resonance based on multiple factors."""
        # Base strength from operator types
        op_0, op_1 = ops
        op_type_0 = (op_0 >> 1) & 0x07
        op_type_1 = (op_1 >> 1) & 0x07

        # Operator strength weights
        op_weights = [0.5, 1.0, 0.8, 0.9]  # Identity, Inverse, Forward, Backward
        base_strength = (op_weights[op_type_0] + op_weights[op_type_1]) / 2

        # Phase position factor (boundaries have higher strength)
        phase_factor = 1.0
        if phase in [0, 12, 24, 36]:  # CS boundaries
            phase_factor = 1.5
        elif phase in [3, 9, 15, 21, 27, 33, 39, 45]:  # UNA/ONA
            phase_factor = 1.3
        elif phase in [6, 18, 30, 42]:  # Nesting
            phase_factor = 1.2

        # Input byte complexity factor
        bit_count = bin(input_byte).count("1")
        complexity_factor = 0.5 + (bit_count / 8.0)  # 0.5 to 1.5

        # Historical field strength at this phase
        field_factor = 0.8 + (self._resonance_field[phase] * 0.4)  # 0.8 to 1.2

        # Combine factors
        strength = base_strength * phase_factor * complexity_factor * field_factor

        # Normalize to 0-1 range
        return min(strength / 3.0, 1.0)

    def _update_resonance_field(self, phase: int, strength: float) -> None:
        """Update the resonance field map."""
        # Exponential moving average
        alpha = 0.1
        self._resonance_field[phase] = alpha * strength + (1 - alpha) * self._resonance_field[phase]
        self._field_samples[phase] += 1

        # Update peak tracking
        if strength > self._resonance_stats["peak_strength"]:
            self._resonance_stats["peak_strength"] = strength
            self._resonance_stats["peak_phase"] = phase

    def _detect_resonance_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in resonance events."""
        if len(self._resonance_events) < 16:
            return []

        patterns = []

        # Look for repeating phase sequences
        recent_events = list(self._resonance_events)[-48:]  # Last cycle
        resonant_phases = [e["phase"] for e in recent_events if e["resonated"]]

        if len(resonant_phases) >= 4:
            # Check for arithmetic progressions
            for start in range(len(resonant_phases) - 3):
                seq = resonant_phases[start : start + 4]
                if len(set(seq[i + 1] - seq[i] for i in range(3))) == 1:
                    # Found arithmetic progression
                    pattern = {
                        "type": "arithmetic_progression",
                        "sequence": seq,
                        "step": seq[1] - seq[0],
                        "confidence": 0.8,
                    }
                    patterns.append(pattern)

        # Look for harmonic patterns (phases that are multiples)
        if resonant_phases:
            for divisor in [2, 3, 4, 6, 8, 12]:
                harmonic_phases = [p for p in resonant_phases if p % divisor == 0]
                if len(harmonic_phases) >= 3:
                    pattern = {
                        "type": "harmonic",
                        "divisor": divisor,
                        "phases": harmonic_phases,
                        "confidence": len(harmonic_phases) / len(resonant_phases),
                    }
                    patterns.append(pattern)

        # Store unique patterns
        for pattern in patterns:
            pattern_key = f"{pattern['type']}_{hash(str(pattern))}"
            if pattern_key not in self._resonance_patterns:
                self._resonance_patterns[pattern_key] = pattern
                self._pattern_counter += 1

        self._resonance_stats["unique_patterns"] = len(self._resonance_patterns)

        return patterns

    def _analyze_harmonics(self) -> None:
        """Analyze harmonic components of resonance field."""
        # Extract resonance strengths over last cycle
        recent_events = list(self._resonance_events)[-48:]
        strengths = np.zeros(48)

        for event in recent_events:
            strengths[event["phase"]] = max(strengths[event["phase"]], event["strength"])

        # Simple DFT for fundamental harmonics
        N = 48
        for harmonic_name, harmonic in self._harmonic_components.items():
            freq = harmonic["frequency"]

            # Calculate amplitude and phase
            real_sum = sum(strengths[n] * np.cos(2 * np.pi * freq * n) for n in range(N))
            imag_sum = sum(strengths[n] * np.sin(2 * np.pi * freq * n) for n in range(N))

            amplitude = np.sqrt(real_sum**2 + imag_sum**2) / N
            phase = np.arctan2(imag_sum, real_sum)

            harmonic["amplitude"] = amplitude
            harmonic["phase"] = phase

    def _update_cycle_detection(self, phase: int, resonated: bool) -> None:
        """Update cyclic pattern detection."""
        self._cycle_detector["buffer"].append((phase, resonated))

        # Look for repeating patterns
        if len(self._cycle_detector["buffer"]) >= 48:
            buffer = list(self._cycle_detector["buffer"])

            # Check for periods from 2 to 24
            for period in range(2, 25):
                if self._check_period(buffer, period):
                    self._cycle_detector["detected_periods"].add(period)

                    # Calculate confidence
                    if period not in self._cycle_detector["confidence"]:
                        self._cycle_detector["confidence"][period] = 0
                    self._cycle_detector["confidence"][period] += 0.1
                    self._cycle_detector["confidence"][period] = min(
                        self._cycle_detector["confidence"][period], 1.0
                    )

    def _check_period(self, buffer: List[Tuple[int, bool]], period: int) -> bool:
        """Check if buffer contains a repeating pattern with given period."""
        if len(buffer) < 2 * period:
            return False

        # Compare last two periods
        last_period = buffer[-period:]
        prev_period = buffer[-2 * period : -period]

        # Check for similarity (allow some variation)
        matches = sum(1 for a, b in zip(last_period, prev_period) if a[1] == b[1])
        similarity = matches / period

        return similarity >= 0.8  # 80% similarity threshold

    def ext_get_resonance_analysis(self) -> Dict[str, Any]:
        """Get comprehensive resonance analysis."""
        # Calculate average strength
        if self._resonance_stats["total_resonances"] > 0:
            total_strength = sum(e["strength"] for e in self._resonance_events if e["resonated"])
            self._resonance_stats["average_strength"] = (
                total_strength / self._resonance_stats["total_resonances"]
            )

        return {
            "statistics": self._resonance_stats.copy(),
            "field_map": self._resonance_field.tolist(),
            "patterns": list(self._resonance_patterns.values()),
            "harmonics": self._harmonic_components.copy(),
            "cycles": {
                "detected_periods": list(self._cycle_detector["detected_periods"]),
                "confidence": self._cycle_detector["confidence"].copy(),
            },
            "cache": self._resonance_cache.copy(),
        }

    def _get_harmonic_summary(self) -> Dict[str, float]:
        """Get summary of harmonic components."""
        return {name: harmonic["amplitude"] for name, harmonic in self._harmonic_components.items()}

    def ext_predict_resonance(self, phase: int, input_byte: int) -> float:
        """
        Predict resonance probability for given phase and input.

        Args:
            phase: Target phase
            input_byte: Input byte

        Returns:
            Predicted resonance probability (0-1)
        """
        # Base probability from field
        base_prob = self._resonance_field[phase]

        # Adjust based on input byte patterns
        if self._resonance_events:
            # Find similar input bytes that resonated at this phase
            similar_events = [
                e
                for e in self._resonance_events
                if e["phase"] == phase and abs(e["input_byte"] - input_byte) <= 16
            ]

            if similar_events:
                resonance_rate = sum(1 for e in similar_events if e["resonated"]) / len(
                    similar_events
                )
                base_prob = (base_prob + resonance_rate) / 2

        # Apply harmonic modulation
        harmonic_factor = 1.0
        for harmonic in self._harmonic_components.values():
            harmonic_value = harmonic["amplitude"] * np.cos(
                2 * np.pi * harmonic["frequency"] * phase + harmonic["phase"]
            )
            harmonic_factor += harmonic_value * 0.1

        return min(base_prob * harmonic_factor, 1.0)

    def ext_modulate_resonance(self, factor: float) -> None:
        """
        Set resonance modulation factor.

        Args:
            factor: Modulation factor (0.1 to 2.0)
        """
        self._resonance_cache["modulation_factor"] = max(0.1, min(factor, 2.0))

    def ext_get_field_visualization(self) -> Dict[str, Any]:
        """Get data for visualizing the resonance field."""
        return {
            "field_values": self._resonance_field.tolist(),
            "sample_counts": self._field_samples.tolist(),
            "phase_labels": [f"Phase {i}" for i in range(48)],
            "peak_phase": self._resonance_stats["peak_phase"],
            "peak_value": self._resonance_stats["peak_strength"],
        }

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_resonance_processor"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Resonance cache: 8-12 bytes as specified
        return 12

    def get_learning_state(self) -> Dict[str, Any]:
        """Resonance patterns and field map."""
        return {
            "resonance_patterns": self._resonance_patterns.copy(),
            "field_map": self._resonance_field.tolist(),
            "harmonic_components": self._harmonic_components.copy(),
            "statistics": self._resonance_stats.copy(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Current resonance state."""
        return {
            "resonance_cache": self._resonance_cache.copy(),
            "recent_events": list(self._resonance_events)[-20:],
            "cycle_detector": {
                "detected_periods": list(self._cycle_detector["detected_periods"]),
                "confidence": self._cycle_detector["confidence"].copy(),
            },
        }

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore learned patterns."""
        if "resonance_patterns" in state:
            self._resonance_patterns = state["resonance_patterns"]

        if "field_map" in state:
            self._resonance_field = np.array(state["field_map"], dtype=np.float32)

        if "harmonic_components" in state:
            self._harmonic_components = state["harmonic_components"]

        if "statistics" in state:
            self._resonance_stats = state["statistics"]

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore current state."""
        if "resonance_cache" in state:
            self._resonance_cache = state["resonance_cache"]

        if "cycle_detector" in state:
            self._cycle_detector["detected_periods"] = set(
                state["cycle_detector"]["detected_periods"]
            )
            self._cycle_detector["confidence"] = state["cycle_detector"]["confidence"]

    def ext_on_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """Process navigation events for resonance analysis."""
        # Extract operator codes
        op_0 = nav_event & 0x0F
        op_1 = (nav_event >> 4) & 0x0F

        # This would need phase information from the system
        # For now, estimate based on event patterns
        if input_byte is not None:
            # Use current cache phase as estimate
            estimated_phase = (self._resonance_cache["last_resonance_phase"] + 1) % 48
            self.ext_process_resonance(estimated_phase, True, input_byte, (op_0, op_1))
