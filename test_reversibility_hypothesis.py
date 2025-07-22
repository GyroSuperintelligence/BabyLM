# test_reversibility_hypothesis.py

import time
from typing import Any, Dict, Tuple

import numpy as np

# --- Configuration ---
# You can adjust this, but 10,000 is a good balance of speed and rigor.
SAMPLE_SIZE = 10000
# The "magic 16" set proposed by A3 for direct testing.
MAGIC_16_INTRONS = {0, 24, 36, 60, 66, 90, 102, 126, 170, 194, 206, 218, 234, 236, 242, 254}

# --- Core Physics (Copied directly from baby/governance.py for standalone testing) ---


def build_masks_and_constants() -> Tuple[int, np.ndarray, np.ndarray]:
    """Pre-computes transformation masks based on layer-based physics."""
    FG, BG = 0, 0
    for layer in range(4):
        for frame in range(2):
            for row in range(3):
                for col in range(2):
                    bit_index = ((layer * 2 + frame) * 3 + row) * 2 + col
                    if layer in (0, 2):
                        FG |= 1 << bit_index
                    if layer in (1, 3):
                        BG |= 1 << bit_index
    FG_MASK = FG
    BG_MASK = BG
    FULL_MASK = (1 << 48) - 1
    INTRON_BROADCAST_MASKS = np.empty(256, dtype=np.uint64)
    for i in range(256):
        mask = 0
        for j in range(6):
            mask |= i << (8 * j)
        INTRON_BROADCAST_MASKS[i] = mask
    XFORM_MASK = np.empty(256, dtype=np.uint64)
    for i in range(256):
        m = 0
        if i & 0b01000010:
            m ^= FULL_MASK
        if i & 0b00100100:
            m ^= FG_MASK
        if i & 0b00011000:
            m ^= BG_MASK
        XFORM_MASK[i] = m
    return FULL_MASK, XFORM_MASK, INTRON_BROADCAST_MASKS


FULL_MASK, XFORM_MASK, INTRON_BROADCAST_MASKS = build_masks_and_constants()


def apply_gyration_and_transform_batch(states: np.ndarray, intron: int) -> "np.ndarray[np.uint64, Any]":
    """Vectorised transform for a batch of states (uint64)."""
    mask = XFORM_MASK[intron]
    pattern = INTRON_BROADCAST_MASKS[intron]
    temp = states ^ mask
    return (temp ^ (temp & pattern)).astype(np.uint64)


# --- Test Logic ---


def run_reversibility_test(sample_size: int) -> Dict[str, Any]:
    """
    Tests the self-inverse property F_i(F_i(s)) == s for all 256 introns.

    Returns a dictionary with detailed results.
    """
    print("--- Running Reversibility Test ---")
    print(f"Generating a random sample of {sample_size} 48-bit states...")

    # Generate a random sample of 48-bit states as uint64
    max_state = 1 << 48
    sample_states = np.random.randint(0, max_state, size=sample_size, dtype=np.uint64)

    results = {}

    print("Testing each of the 256 introns for the self-inverse property...")
    print("Progress: ", end="", flush=True)
    start_time = time.time()

    for i in range(256):
        # Apply the transformation twice: F_i(F_i(s))
        s_prime = apply_gyration_and_transform_batch(sample_states, i)
        s_double_prime = apply_gyration_and_transform_batch(s_prime, i)

        # Count how many states returned to their original value
        reversal_count = np.count_nonzero(s_double_prime == sample_states)
        results[i] = reversal_count

        if i > 0 and i % 16 == 0:
            print(".", end="", flush=True)

    end_time = time.time()
    print("\nTest complete.")

    # Process results
    perfectly_reversible = {i for i, count in results.items() if count == sample_size}
    partially_reversible = {i: count / sample_size for i, count in results.items() if 0 < count < sample_size}

    return {
        "sample_size": sample_size,
        "duration_seconds": end_time - start_time,
        "perfectly_reversible": perfectly_reversible,
        "partially_reversible": partially_reversible,
        "raw_counts": results,
    }


def print_report(results: dict) -> None:
    """Prints a clear, human-readable report of the test results."""

    print("\n\n--- Reversibility Test Report ---")
    print(f"Tested on {results['sample_size']} random states in {results['duration_seconds']:.2f} seconds.")

    print("\n[1] Perfectly Self-Inverse Introns (Reversible on 100% of the sample):")
    perfect_set = results["perfectly_reversible"]
    if perfect_set:
        print(f"  Found {len(perfect_set)} intron(s): {sorted(list(perfect_set))}")
    else:
        print("  None found.")

    print("\n[2] Partially Self-Inverse Introns (Reversible on some, but not all, states):")
    partial_dict = results["partially_reversible"]
    if partial_dict:
        # Sort by reversal rate, descending
        sorted_partial = sorted(partial_dict.items(), key=lambda item: item[1], reverse=True)
        print(f"  Found {len(partial_dict)} partially reversible introns. Top 5:")
        for i, rate in sorted_partial[:5]:
            print(f"    - Intron {i:<3}: {rate:.2%} reversal rate")
    else:
        print("  None found.")

    print("\n[3] Analysis of the 'Magic 16' Intron Set:")
    magic_set_perfect = MAGIC_16_INTRONS.intersection(perfect_set)
    magic_set_partial = {
        i: results["raw_counts"][i] / results["sample_size"] for i in MAGIC_16_INTRONS if i in partial_dict
    }
    magic_set_failed = MAGIC_16_INTRONS - magic_set_perfect - set(magic_set_partial.keys())

    print(f"  - {len(magic_set_perfect)} of the 16 were 100% reversible: {sorted(list(magic_set_perfect))}")
    if magic_set_partial:
        print(f"  - {len(magic_set_partial)} were partially reversible:")
        for i, rate in magic_set_partial.items():
            print(f"      Intron {i:<3}: {rate:.2%} reversal rate")
    if magic_set_failed:
        print(f"  - {len(magic_set_failed)} failed completely (0% reversible): {sorted(list(magic_set_failed))}")

    print("\n--- Conclusion ---")
    if perfect_set == {0}:
        print("✅ Verdict: A1's hypothesis is STRONGLY SUPPORTED.")
        print("   Only intron 0 (the identity operation) was found to be globally self-inverse.")
        print("   This implies that the path-dependent memory term is non-zero for all other introns,")
        print("   making them irreversible under the system's full physics.")
    elif len(perfect_set) > 1:
        print("⚠️ Verdict: A1's hypothesis is CONTRADICTED.")
        print(
            f"   Found {len(perfect_set)} introns that appear to be globally self-inverse: {sorted(list(perfect_set))}"
        )
        print("   This suggests the algebraic analysis may need refinement based on these empirical results.")
    else:
        print("❓ Verdict: Unexpected result. No introns, not even 0, were perfectly reversible.")
        print("   This could indicate an issue with the test or the physics implementation itself.")


if __name__ == "__main__":
    test_results = run_reversibility_test(SAMPLE_SIZE)
    print_report(test_results)
