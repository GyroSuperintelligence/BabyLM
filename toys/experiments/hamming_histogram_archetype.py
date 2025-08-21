#!/usr/bin/env python3
"""
Generate Hamming distance histogram vs. UNA archetype.
Analyzes bit-level differences between all ontology states and the primary phenomenal archetype.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from baby.governance import tensor_to_int
from baby.information import InformationEngine


def hamming_distance(a: int, b: int) -> int:
    """Calculate Hamming distance between two integers."""
    return bin(a ^ b).count("1")


def generate_hamming_histogram():
    """Generate Hamming distance histogram vs. UNA archetype."""

    # Load ontology and related data
    meta_path = Path("memories/public/meta")
    ontology_path = meta_path / "ontology_keys.npy"
    theta_path = meta_path / "theta.npy"

    if not ontology_path.exists() or not theta_path.exists():
        print("❌ Ontology files not found. Please regenerate maps first.")
        return

    ontology = np.load(ontology_path, mmap_mode="r")
    theta = np.load(theta_path, mmap_mode="r")

    print(f"📊 Analyzing {len(ontology)} states for Hamming distances")

    # Find UNA archetype (θ ≈ π/4)
    target_theta = np.pi / 4
    diff = np.abs(theta - target_theta)
    una_index = int(np.argmin(diff))
    una_state = int(ontology[una_index])
    una_theta = float(theta[una_index])

    print(f"🎯 UNA Archetype:")
    print(f"   Index: {una_index}")
    print(f"   State: 0x{una_state:012X}")
    print(f"   Theta: {una_theta:.6f} (target: {target_theta:.6f})")
    print(f"   Difference: {diff[una_index]:.6f}")

    # Calculate Hamming distances for all states
    print(f"\n🔢 Computing Hamming distances...")
    hamming_distances = []

    # Process in chunks to manage memory
    chunk_size = 10000
    for i in range(0, len(ontology), chunk_size):
        end_idx = min(i + chunk_size, len(ontology))
        chunk = ontology[i:end_idx]

        # Calculate Hamming distances for this chunk
        chunk_distances = [hamming_distance(int(state), una_state) for state in chunk]
        hamming_distances.extend(chunk_distances)

        if (i // chunk_size + 1) % 10 == 0:
            print(f"   Processed {end_idx:,} / {len(ontology):,} states")

    hamming_distances = np.array(hamming_distances)

    # Generate statistics
    print(f"\n📈 Hamming Distance Statistics:")
    print(f"   Min distance: {hamming_distances.min()}")
    print(f"   Max distance: {hamming_distances.max()}")
    print(f"   Mean distance: {hamming_distances.mean():.2f}")
    print(f"   Std deviation: {hamming_distances.std():.2f}")
    print(f"   Median distance: {np.median(hamming_distances):.1f}")

    # Count distances
    unique_distances, counts = np.unique(hamming_distances, return_counts=True)
    print(f"\n🔍 Distance Distribution:")
    for dist, count in zip(unique_distances[:10], counts[:10]):  # Show first 10
        percentage = (count / len(hamming_distances)) * 100
        print(f"   Distance {dist:2d}: {count:6,} states ({percentage:5.2f}%)")

    if len(unique_distances) > 10:
        print(f"   ... and {len(unique_distances) - 10} more distances")

    # Create histogram
    plt.figure(figsize=(12, 8))

    # Main histogram
    plt.subplot(2, 1, 1)
    bins = np.arange(0, hamming_distances.max() + 2) - 0.5
    plt.hist(hamming_distances, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
    plt.xlabel("Hamming Distance from UNA Archetype")
    plt.ylabel("Number of States")
    plt.title(
        f"Hamming Distance Distribution vs. UNA Archetype\n"
        f"UNA State: 0x{una_state:012X} (θ={una_theta:.4f}, index={una_index})"
    )
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Total States: {len(ontology):,}\n"
    stats_text += f"Mean Distance: {hamming_distances.mean():.2f}\n"
    stats_text += f"Std Dev: {hamming_distances.std():.2f}\n"
    stats_text += f"Range: [{hamming_distances.min()}, {hamming_distances.max()}]"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Cumulative distribution
    plt.subplot(2, 1, 2)
    sorted_distances = np.sort(hamming_distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
    plt.plot(sorted_distances, cumulative, color="darkred", linewidth=2)
    plt.xlabel("Hamming Distance from UNA Archetype")
    plt.ylabel("Cumulative Percentage (%)")
    plt.title("Cumulative Distribution of Hamming Distances")
    plt.grid(True, alpha=0.3)

    # Add percentile markers
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(hamming_distances, p)
        plt.axvline(value, color="red", linestyle="--", alpha=0.7)
        plt.text(value, p + 2, f"P{p}={value:.1f}", rotation=90, verticalalignment="bottom", fontsize=8)

    plt.tight_layout()

    # Save the plot
    output_path = "hamming_histogram_vs_una_archetype.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n💾 Histogram saved to: {output_path}")

    # Show the plot
    plt.show()

    # Additional analysis: Find states with specific Hamming distances
    print(f"\n🔍 Notable States:")

    # States with distance 0 (should be just the archetype itself)
    distance_0 = np.where(hamming_distances == 0)[0]
    print(f"   Distance 0 (identical): {len(distance_0)} states")
    if len(distance_0) > 0:
        for idx in distance_0[:5]:  # Show first 5
            state = int(ontology[idx])
            print(f"     Index {idx}: 0x{state:012X} (θ={theta[idx]:.6f})")

    # States with distance 1 (single bit flip)
    distance_1 = np.where(hamming_distances == 1)[0]
    print(f"   Distance 1 (1-bit flip): {len(distance_1)} states")
    if len(distance_1) > 0:
        for idx in distance_1[:5]:  # Show first 5
            state = int(ontology[idx])
            print(f"     Index {idx}: 0x{state:012X} (θ={theta[idx]:.6f})")

    # States with maximum distance
    max_distance = hamming_distances.max()
    max_distance_indices = np.where(hamming_distances == max_distance)[0]
    print(f"   Distance {max_distance} (maximum): {len(max_distance_indices)} states")
    if len(max_distance_indices) > 0:
        for idx in max_distance_indices[:5]:  # Show first 5
            state = int(ontology[idx])
            print(f"     Index {idx}: 0x{state:012X} (θ={theta[idx]:.6f})")

    return {
        "una_index": una_index,
        "una_state": una_state,
        "una_theta": una_theta,
        "hamming_distances": hamming_distances,
        "stats": {
            "min": int(hamming_distances.min()),
            "max": int(hamming_distances.max()),
            "mean": float(hamming_distances.mean()),
            "std": float(hamming_distances.std()),
            "median": float(np.median(hamming_distances)),
        },
    }


if __name__ == "__main__":
    result = generate_hamming_histogram()
    if result:
        print(f"\n✅ Analysis complete. Hamming histogram generated.")
