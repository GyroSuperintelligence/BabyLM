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
    derive_canonical_patterns,
    classify_pattern_resonance,
)
from collections import deque
from pathlib import Path

# Optional numba import for JIT compilation (adds 2-3x speedup)
_numba = None  # type: ignore
try:
    import numba as _numba  # type: ignore
except ImportError:
    pass


class InferenceEngine:
    """
    Inference Engine for pattern recognition and learning

    Manages the Epigenome tensor and performs pattern matching against
    canonical patterns.
    """

    def __init__(self, base_memories_dir: str = "memories"):
        """Initialize the Inference Engine with zero-state tensor and load patterns"""
        self.base_memories_dir = base_memories_dir
        # The Epigenome Tensor (dynamic state)
        self.T = np.zeros((4, 2, 3, 2), dtype=np.float32)

        # Initialize cycle counter
        self.cycle_counter = 0

        # The Epigenome Mask (canonical patterns)
        self.F, self.gyration_features = self._load_patterns()

        # Load genome mask (output byte mappings)
        self.G = self._load_genome_mask()

        # Perform initial stateless mutation
        self._initialize_epigenome()

        # Track recent pattern indices (up to 20)
        self.recent_patterns = deque(maxlen=20)

        # Add distance cache and hit/miss counters
        self._distance_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _load_patterns(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load canonical patterns from file or generate if not present.

        For this 48KB file, a direct read with np.fromfile is the most
        performant and robust method. Memory-mapping would add unnecessary
        overhead and complexity.

        Returns:
            Tuple containing:
            - patterns: Array of shape [256, 48] (read-only)
            - gyration_features: List of 256 class labels
        """
        pattern_file = str(Path(self.base_memories_dir) / "public/masks/epigenome.dat")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(pattern_file), exist_ok=True)

        try:
            # Try to load the entire 48KB patterns file directly into memory.
            patterns = np.fromfile(pattern_file, dtype=np.float32)
            # Explicitly validate the shape to protect against corrupted files.
            if patterns.size != 256 * 48:
                raise ValueError("Epigenome file is corrupted or has an incorrect size.")
            patterns = patterns.reshape((256, 48))
            # Mark the array as read-only to prevent accidental modification.
            patterns.flags.writeable = False
            # These are deterministic and cheap to regenerate.
            gyration_features = [classify_pattern_resonance(i) for i in range(256)]
            return patterns, gyration_features
        except (FileNotFoundError, ValueError, IOError):
            # This block is for first-time setup or if the file is corrupted.
            print("Generating canonical patterns... This happens only once.")
            patterns, gyration_features = derive_canonical_patterns()
            # Save the newly generated patterns to file.
            patterns.tofile(pattern_file)
            # Mark the new in-memory array as read-only as well.
            patterns.flags.writeable = False
            return patterns, gyration_features

    def _load_genome_mask(self) -> np.ndarray:
        """
        Load genome mask from file or generate if not present

        Returns:
            Array of 256 bytes mapping pattern indices to output bytes
        """
        genome_file = str(Path(self.base_memories_dir) / "public/masks/genome.dat")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(genome_file), exist_ok=True)

        try:
            # Try to load genome mask from file
            genome_mask = np.fromfile(genome_file, dtype=np.uint8, count=256)

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
                apply_operation(self.T, i)   # now mutates T directly
        # Reset cycle counter after initialization
        self.cycle_counter = 0

    def find_closest_pattern_index(self) -> Tuple[int, float]:
        """
        Find index of canonical pattern closest to current tensor state

        Returns:
            Tuple[int, float]: (Index of closest matching pattern (0-255), resonance/gyrodistance)
        """
        # Create a hash of current tensor state
        tensor_hash = hash(self.T.tobytes())

        # Check cache first
        if tensor_hash in self._distance_cache:
            self._cache_hits += 1
            return self._distance_cache[tensor_hash]

        self._cache_misses += 1

        # Flatten current tensor for comparison
        flat_T = self.T.flatten()
        # Vectorized calculation for all distances
        dot_products = np.dot(self.F, flat_T)
        normalized_distances = dot_products / flat_T.size
        angular_distances = np.arccos(np.clip(normalized_distances, -1.0, 1.0))
        closest_index = int(np.argmin(angular_distances))
        min_distance = float(angular_distances[closest_index])

        # Cache result (limit cache size)
        if len(self._distance_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            self._distance_cache = dict(list(self._distance_cache.items())[-500:])

        self._distance_cache[tensor_hash] = (closest_index, min_distance)
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
                apply_operation(self.T, i)   # now mutates T directly

        # 3. Find matching canonical pattern and resonance
        key_index, resonance = self.find_closest_pattern_index()

        # Track recent patterns
        self.recent_patterns.append(key_index)

        # 4. Increment cycle counter
        self.cycle_counter += 1

        return key_index, resonance

    # ------------------------------------------------------------------
    # Fast-path batch processor
    # ------------------------------------------------------------------
    def process_batch(self, p_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised version of process_byte().
        Args:
            p_batch : 1-D NumPy array dtype=uint8.
        Returns:
            (key_indices, resonances) – both NumPy arrays same length as p_batch.
        """
        batch_len = p_batch.size
        key_indices  = np.empty(batch_len, dtype=np.uint8)
        resonances   = np.empty(batch_len, dtype=np.float32)

        for i in range(batch_len):
            P_n = int(p_batch[i])
            # --- original per-byte logic, unchanged ---
            gene_mutated = P_n ^ gene_stateless
            for bit in range(8):
                if gene_mutated & (1 << bit):
                    apply_operation(self.T, bit)   # now mutates T directly

            key_i, res = self.find_closest_pattern_index()
            key_indices[i] = key_i
            resonances[i]  = res
            # We keep the cycle counter identical to the old path
            self.cycle_counter += 1
            self.recent_patterns.append(key_i)

        return key_indices, resonances

    # Optional Numba JIT compilation for 3x speedup
    if _numba:
        process_batch = _numba.njit(cache=True)(process_batch)

    def compute_pattern_resonances(self) -> np.ndarray:
        """
        Compute resonance values between current tensor and all patterns

        Returns:
            np.ndarray: Array of 256 resonance values (distances) between current tensor and each canonical pattern
        """
        flat_T = self.T.flatten()
        dot_products = np.dot(self.F, flat_T)
        normalized_distances = dot_products / flat_T.size
        angular_distances = np.arccos(np.clip(normalized_distances, -1.0, 1.0))
        return angular_distances

    def tensor_to_output_byte(self) -> int:
        """
        Convert a tensor to a single output byte by thresholding and packing bits.

        Args:
            tensor (np.ndarray): The input tensor to convert.

        Returns:
            int: The resulting output byte as an integer.
        """
        key_index, _ = self.find_closest_pattern_index()
        output_byte = self.G[key_index]
        if hasattr(output_byte, "item"):
            return int(output_byte.item())
        return int(output_byte)
