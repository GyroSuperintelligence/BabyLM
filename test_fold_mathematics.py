#!/usr/bin/env python3
"""
A Definitive Experiment to Measure the Physical Non-Associativity of GyroSI.

This script directly tests the core physical axiom of the Common Governance Model:
that the order of operations (path) is preserved in the physical state of the
system. It moves beyond testing the 8-bit intron algebra and measures the actual
state transitions encoded in the epistemology map.

Hypothesis:
    For a given state S and a sequence of introns {i1, i2, ...}, the final
    physical state depends on the grouping of the introns. Specifically, the
    path ((S → i1) → i2) will lead to a different physical state than the
    path (S → (i1 ⋄ i2)), where '⋄' is the Monodromic Fold.

Methodology:
1.  Load the complete, authoritative epistemology and ontology maps.
2.  Sample thousands of random initial states and random intron triplets.
3.  For each sample, compute two distinct physical paths:
    a) Left-Associated Path: Apply introns sequentially to the state.
       S_final_L = apply(apply(S_initial, i1), i2)
    b) Right-Associated Path: Pre-compute the fold of the last two introns,
       then apply the sequence. i_combined = fold(i1, i2).
       S_final_R = apply(S_initial, i_combined)
4.  Compare the final state indices. A mismatch is a "witness" to physical
    non-associativity.
5.  Report the statistical prevalence of this phenomenon.
"""
import sys
import os
import numpy as np
import random
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from baby.governance import fold
    import ujson as json
except ImportError:
    print("Error: Required modules not found. Ensure you are in the correct environment.")
    sys.exit(1)


def load_maps(ontology_path: str, epistemology_path: str) -> tuple[dict, np.ndarray]:
    """Loads the necessary map files."""
    print("Loading authoritative maps...")
    if not os.path.exists(ontology_path) or not os.path.exists(epistemology_path):
        raise FileNotFoundError("Ensure ontology_map.json and epistemology.npy are present.")

    with open(ontology_path, "r") as f:
        ontology_data = json.load(f)

    ep = np.load(epistemology_path, mmap_mode="r")

    print(f"  Ontology states: {ontology_data['endogenous_modulus']:,}")
    print(f"  Epistemology shape: {ep.shape}")
    print("...maps loaded.\n")
    return ontology_data, ep


def run_experiment(ep: np.ndarray, num_samples: int = 10000) -> dict[str, int | float | list]:
    """
    Performs the main experiment to measure physical non-associativity.
    """
    print(f"Running experiment with {num_samples:,} random samples...")

    N = ep.shape[0]
    non_associative_count = 0
    witnesses: list[dict[str, int | tuple[int, int]]] = []

    for _ in tqdm(range(num_samples), desc="Measuring Path Divergence"):
        # 1. Select a random initial state and two random introns
        s_initial_idx = random.randrange(N)
        i1 = random.randrange(256)
        i2 = random.randrange(256)

        # 2. Compute the Left-Associated Path: (S → i1) → i2
        s_intermediate_idx = ep[s_initial_idx, i1]
        s_final_L_idx = ep[s_intermediate_idx, i2]

        # 3. Compute the Right-Associated Path: S → (i1 ⋄ i2)
        i_combined = fold(i1, i2)
        s_final_R_idx = ep[s_initial_idx, i_combined]

        # 4. Compare the final physical states
        if s_final_L_idx != s_final_R_idx:
            non_associative_count += 1
            if len(witnesses) < 5:  # Store a few examples
                witnesses.append(
                    {
                        "initial_state_idx": s_initial_idx,
                        "introns": (i1, i2),
                        "path_L_final_idx": int(s_final_L_idx),
                        "path_R_final_idx": int(s_final_R_idx),
                        "i_combined": i_combined,
                    }
                )

    return {
        "num_samples": num_samples,
        "non_associative_count": non_associative_count,
        "associative_count": num_samples - non_associative_count,
        "non_associativity_ratio": non_associative_count / num_samples,
        "witnesses": witnesses,
    }


def report_results(results: dict) -> None:
    """Prints a formatted report of the experimental findings."""
    print("\n" + "=" * 60)
    print("    EXPERIMENTAL RESULTS: PHYSICAL NON-ASSOCIATIVITY")
    print("=" * 60)

    ratio = results["non_associativity_ratio"]
    print(f"\nTotal Samples Tested: {results['num_samples']:,}")
    print(f"  - Non-Associative Events: {results['non_associative_count']:,}")
    print(f"  -   Associative Events: {results['associative_count']:,}")
    print(f"\nMeasured Physical Non-Associativity: {ratio:.2%}")

    print("\nSample Witnesses (cases where path grouping mattered):")
    for i, witness in enumerate(results["witnesses"]):
        i1, i2 = witness["introns"]
        print(f"  Witness {i+1}:")
        print(f"    - Initial State Index: {witness['initial_state_idx']}")
        print(f"    - Introns: (0x{i1:02x}, 0x{i2:02x})")
        print(f"    - Path ((S→i1)→i2) leads to State Index: {witness['path_L_final_idx']}")
        path_r_info = (
            f"    - Path (S→(i1⋄i2)) leads to State Index: {witness['path_R_final_idx']} "
            f"(using i_combined=0x{witness['i_combined']:02x})"
        )
        print(path_r_info)

    print("\n" + "=" * 25 + " CONCLUSION " + "=" * 25)
    if ratio > 0.85:
        print("The system demonstrates strong physical non-associativity. The grouping of")
        print("operations fundamentally alters the final physical state, confirming that")
        print("path-dependence is a core law of the system's physics, not just its")
        print("learning algebra. The Common Source axiom is physically upheld.")
    elif ratio > 0.1:
        print("The system exhibits significant physical non-associativity. While not")
        print("universal, the path of experience frequently alters the physical outcome.")
    else:
        print("The system shows weak physical non-associativity. The underlying physics")
        print("is more associative than predicted by the learning algebra. This suggests")
        print("a more complex relationship between the intron algebra and the state manifold.")


def main() -> None:
    """Main execution function."""
    ontology_path = "memories/public/meta/ontology_map.json"
    epistemology_path = "memories/public/meta/epistemology.npy"

    try:
        ontology_data, ep_table = load_maps(ontology_path, epistemology_path)
        results = run_experiment(ep_table)
        report_results(results)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure you have run the build scripts for the ontology and epistemology maps.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
