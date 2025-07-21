#!/usr/bin/env python
"""
ontology.py

This script is the definitive implementation for discovering the GyroSI physical
ontology. It uses a 48-bit integer representation for high performance while
perfectly preserving the state-dependent, broadcasting gyration physics of the
original tensor-based model.
"""
import time
from collections import deque
import numpy as np

# ───────────────────────────────────────────────────────────────────
# 1. ARCHETYPAL STATE AND MASKS (The Invariant Physics)
# ───────────────────────────────────────────────────────────────────

# The canonical 48-element GENE_Mac_S tensor.
GENE_MAC_S_TENSOR = np.array(
    [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
    ],
    dtype=np.int8,
)


def _build_masks() -> tuple[int, int, int, list[int]]:
    """Private helper to pre-compute masks and broadcast patterns."""
    fg_mask, bg_mask = 0, 0
    for layer in range(4):
        for frame in range(2):
            for row in range(3):
                for col in range(2):
                    bit_index = ((layer * 2 + frame) * 3 + row) * 2 + col
                    if layer in (0, 2):
                        fg_mask |= 1 << bit_index
                    if layer in (1, 3):
                        bg_mask |= 1 << bit_index

    full_mask = (1 << 48) - 1

    # CORRECTED: A 48-bit integer requires repeating the 8-bit intron 6 times.
    intron_broadcast_masks = [int.from_bytes(i.to_bytes(1, "little") * 6, "little") for i in range(256)]
    return fg_mask, bg_mask, full_mask, intron_broadcast_masks


# Define physics constants at the module level
FG_MASK, BG_MASK, FULL_MASK, INTRON_BROADCAST_MASKS = _build_masks()


def tensor_to_int(T: np.ndarray) -> int:
    """Encodes a {-1, 1} tensor into a 48-bit integer (bit 1 = -1, bit 0 = +1)."""
    bits = 0
    # C-order flattening must match the mask construction order
    for i, val in enumerate(T.flatten(order="C")):
        if val == -1:
            bits |= 1 << i
    return bits


# ───────────────────────────────────────────────────────────────────
# 2. THE CORE PHYSICAL OPERATION
# ───────────────────────────────────────────────────────────────────


def apply_gyration_and_transform(state_int: int, intron: int) -> int:
    """
    Applies the full gyroscopic physics: a gyro-addition (XOR flips)
    followed by a state-dependent Thomas gyration (carry term).
    """
    temp_state = state_int
    if intron & 0b01000010:  # LI: Left Inverse
        temp_state ^= FULL_MASK
    if intron & 0b00100100:  # FG: Forward Gyration
        temp_state ^= FG_MASK
    if intron & 0b00011000:  # BG: Backward Gyration
        temp_state ^= BG_MASK

    # The Thomas gyration term, where state is modified by its own structure.
    intron_pattern = INTRON_BROADCAST_MASKS[intron]
    gyration = temp_state & intron_pattern
    final_state = temp_state ^ gyration

    return final_state


# ───────────────────────────────────────────────────────────────────
# 3. THE EXPLORATION ENGINE
# ───────────────────────────────────────────────────────────────────


def explore_ontology(start_state_int: int) -> None:
    """
    Performs a breadth-first search to discover the entire state space,
    recording the minimal depth each state is first reached.
    """
    discovered = {start_state_int}
    state_depth = {start_state_int: 0}
    frontier = deque([start_state_int])
    total_time = 0

    print("Starting ontology exploration with depth recording.")
    print("-" * 68)
    print(f"{'Depth':>5} | {'New States':>12} | {'Total States':>14} | {'Time (s)':>10}")
    print("-" * 68)

    while frontier:
        t_start = time.perf_counter()
        newly_discovered = []
        current_depth = int(min(state_depth[s] for s in frontier))
        for current_state in frontier:
            for intron in range(256):
                next_state = apply_gyration_and_transform(current_state, intron)
                if next_state not in discovered:
                    discovered.add(next_state)
                    state_depth[next_state] = current_depth + 1
                    newly_discovered.append(next_state)
        t_end = time.perf_counter()
        depth_time = t_end - t_start
        total_time += depth_time

        new_states_count = len(newly_discovered)
        print(f"{current_depth + 1:>5} | {new_states_count:>12,} | {len(discovered):>14,} | {depth_time:>9.3f}")

        if new_states_count == 0:
            break

        frontier = deque(newly_discovered)

    print("-" * 68)
    max_depth = max(state_depth.values()) if state_depth else 0
    print(f"Closure reached. Max minimal path length to any state: {max_depth}")
    print(f"Exploration finished. Total states found: {len(discovered):,}")
    print(f"Total time elapsed: {total_time:.3f}s")
    print("The ontology is fully explored and finite.")


# ───────────────────────────────────────────────────────────────────
# 4. MAIN EXECUTION
# ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initial_state_int = tensor_to_int(GENE_MAC_S_TENSOR)
    explore_ontology(initial_state_int)
