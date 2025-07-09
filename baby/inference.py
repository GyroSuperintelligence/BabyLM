"""
inference.py - S3 Pattern recognition for GyroSI Baby LM

This module implements the Inference Engine for pattern recognition and learning,
representing the Inference (S3) layer of the Common Governance Model.
"""

import numpy as np
from typing import List, Tuple
import os
from baby.governance import (
    apply_operation,
    gene_add,
    gene_stateless,
    gyrodistance,
    derive_canonical_patterns,
    classify_pattern_resonance,
)


class InferenceEngine:
    """
    Inference Engine for pattern recognition and learning

    Manages the Epigenome tensor and performs pattern matching against
    canonical patterns.
    """

    def __init__(self):
        """Initialize the Inference Engine with zero-state tensor and load patterns"""
        # Initialize Epigenome tensor with zeros (shape [4, 2, 3, 2])
        self.T = np.zeros((4, 2, 3, 2), dtype=np.float32)

        # Initialize cycle counter
        self.cycle_counter = 0

        # Load or generate canonical patterns
        self.F, self.gyration_featurees = self._load_patterns()

        # Load genome mask (output byte mappings)
        self.G = self._load_genome_mask()

        # Perform initial stateless mutation
        self._initialize_epigenome()

        # Track recent pattern indices (up to 20)
        self.recent_patterns = []

    def _load_patterns(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load canonical patterns from file or generate if not present

        Returns:
            Tuple containing:
            - patterns: Array of shape [256, 48]
            - gyration_featurees: List of 256 class labels
        """
        pattern_file = "memories/public/masks/epigenome.dat"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(pattern_file), exist_ok=True)

        try:
            # Try to load patterns from file
            patterns = np.fromfile(pattern_file, dtype=np.float32)
            patterns = patterns.reshape((256, 48))

            # Regenerate resonance classes (these are deterministic)
            gyration_featurees = [classify_pattern_resonance(i) for i in range(256)]

            return patterns, gyration_featurees

        except (FileNotFoundError, ValueError, IOError):
            # Generate patterns if file doesn't exist or is corrupted
            patterns, gyration_featurees = derive_canonical_patterns()

            # Save patterns to file
            patterns.tofile(pattern_file)

            return patterns, gyration_featurees

    def _load_genome_mask(self) -> np.ndarray:
        """
        Load genome mask from file or generate if not present

        Returns:
            Array of 256 bytes mapping pattern indices to output bytes
        """
        genome_file = "memories/public/masks/genome.dat"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(genome_file), exist_ok=True)

        try:
            # Try to load genome mask from file
            genome_mask = np.fromfile(genome_file, dtype=np.uint8)

            # Validate size
            if genome_mask.size != 256:
                raise ValueError("Invalid genome mask size")

            return genome_mask

        except (FileNotFoundError, ValueError, IOError):
            # Generate identity mapping if file doesn't exist
            genome_mask = np.arange(256, dtype=np.uint8)

            # Save genome mask to file
            genome_mask.tofile(genome_file)

            return genome_mask

    def _initialize_epigenome(self) -> None:
        """
        Initialize Epigenome tensor with one cycle of the stateless gene

        This simulates one full inference cycle without user input,
        establishing the initial state.
        """
        # Start with gene_add instead of zeros
        self.T = gene_add.copy().astype(np.float32)

        # Use the imported gene_stateless constant
        gene_mutated = gene_stateless
        for i in range(8):
            if gene_mutated & (1 << i):
                apply_operation(self.T, i)
        # Reset cycle counter after initialization
        self.cycle_counter = 0

    def find_closest_pattern_index(self) -> Tuple[int, float]:
        """
        Find index of canonical pattern closest to current tensor state

        Returns:
            Tuple[int, float]: (Index of closest matching pattern (0-255), resonance/gyrodistance)
        """
        # Flatten current tensor for comparison
        flat_T = self.T.flatten()

        # Calculate distances to all patterns
        distances = []
        for pattern in self.F:
            dist = gyrodistance(flat_T, pattern)
            distances.append(dist)

        # Find index of minimum distance
        closest_index = int(np.argmin(distances))
        min_distance = float(distances[closest_index])

        return closest_index, min_distance

    def process_byte(self, P_n: int) -> Tuple[int, float]:
        """
        Process a single input byte

        Args:
            P_n: Input byte (0-255)

        Returns:
            Tuple[int, float]: (Index of the closest matching pattern (0-255), resonance/gyrodistance)
        """
        # 1. Compute gene_mutated = P_n ^ gene_stateless
        gene_mutated = P_n ^ gene_stateless

        # 2. Apply gyroscopic operations to tensor T
        for i in range(8):
            if gene_mutated & (1 << i):
                apply_operation(self.T, i)

        # 3. Find matching canonical pattern and resonance
        key_index, resonance = self.find_closest_pattern_index()

        # Track recent patterns
        if len(self.recent_patterns) >= 20:  # Keep last 20 patterns
            self.recent_patterns.pop(0)
        self.recent_patterns.append(key_index)

        # 4. Increment cycle counter
        self.cycle_counter += 1

        return key_index, resonance

    def compute_pattern_resonances(self) -> List[float]:
        """
        Compute resonance values between current tensor and all patterns

        Returns:
            List of 256 resonance values (distances) between current tensor
            and each canonical pattern
        """
        # Flatten current tensor for comparison
        flat_T = self.T.flatten()

        # Calculate distances to all patterns
        resonances = []
        for pattern in self.F:
            dist = gyrodistance(flat_T, pattern)
            resonances.append(dist)

        return resonances
