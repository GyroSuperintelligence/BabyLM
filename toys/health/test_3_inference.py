"""
Tests for the inference engine and related tensor operations in the BabyLM system.
Includes pattern loading, genome mask handling, epigenome initialization, and pattern matching logic.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

# Import modules from baby package
from baby.inference import InferenceEngine
from baby.governance import derive_canonical_patterns, apply_operation, gene_add


# ------------------------------------------------------------------------------
# Inference (S3) Tests
# ------------------------------------------------------------------------------


class TestInference:
    """Tests for the Inference layer (pattern recognition)"""

    def test_inference_engine_init(self):
        """Test InferenceEngine initialization"""
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_load_genome_mask", return_value=np.arange(256, dtype=np.uint8)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):

            engine = InferenceEngine()

            assert engine.T.shape == (4, 2, 3, 2)
            assert engine.cycle_counter == 0
            assert len(engine.gyration_features) == 256
            assert engine.F.shape == (256, 48)
            assert engine.G.shape == (256,)

    def test_load_patterns_file_exists(self, mock_env):
        """Test loading patterns when file exists"""
        # Create a mock pattern file
        pattern_file = Path(mock_env) / "memories/public/masks/epigenome.dat"
        pattern_file.parent.mkdir(parents=True, exist_ok=True)

        # Create test patterns
        test_patterns = np.ones((256, 48), dtype=np.float32)
        with open(pattern_file, "wb") as f:
            test_patterns.tofile(f)

        # Test loading
        engine = InferenceEngine()

        # Verify loaded patterns
        assert engine.F.shape == (256, 48)
        assert len(engine.gyration_features) == 256

        # Clean up
        pattern_file.unlink()

    def test_load_patterns_generate_new(self, mock_env):
        """Test pattern generation when file doesn't exist"""
        # Ensure pattern file doesn't exist
        pattern_file = Path(mock_env) / "memories/public/masks/epigenome.dat"
        if pattern_file.exists():
            pattern_file.unlink()

        # Use actual implementation to load/generate patterns
        with patch("baby.inference.derive_canonical_patterns", return_value=(np.ones((256, 48)), ["test"] * 256)):
            engine = InferenceEngine()

            # Verify generated patterns
            assert engine.F.shape == (256, 48)
            assert len(engine.gyration_features) == 256

    def test_load_genome_mask_file_exists(self, mock_env):
        """Test loading genome mask when file exists"""
        # Create a mock genome file
        genome_file = Path(mock_env) / "memories/public/masks/genome.dat"
        genome_file.parent.mkdir(parents=True, exist_ok=True)

        # Create test genome mask
        test_genome = np.arange(256, dtype=np.uint8)
        with open(genome_file, "wb") as f:
            test_genome.tofile(f)

        # Test loading
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):
            engine = InferenceEngine()

            # Verify loaded genome mask
            assert engine.G.shape == (256,)
            np.testing.assert_array_equal(engine.G, test_genome)

        # Clean up
        genome_file.unlink()

    def test_load_genome_mask_generate_new(self, mock_env):
        """Test genome mask generation when file doesn't exist"""
        # Ensure genome file doesn't exist
        genome_file = Path(mock_env) / "memories/public/masks/genome.dat"
        if genome_file.exists():
            genome_file.unlink()

        # Test loading/generation
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):
            engine = InferenceEngine()

            # Verify generated genome mask
            assert engine.G.shape == (256,)
            np.testing.assert_array_equal(engine.G, np.arange(256, dtype=np.uint8))

    def test_initialize_epigenome(self):
        """Test Epigenome initialization"""
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_load_genome_mask", return_value=np.arange(256, dtype=np.uint8)),
        ):

            engine = InferenceEngine()

            # Capture initial tensor state after initialization
            initial_T = engine.T.copy()

            # Reset tensor to zeros
            engine.T.fill(0.0)

            # Manually initialize epigenome
            engine._initialize_epigenome()

            # Verify initialization effect
            np.testing.assert_array_equal(engine.T, initial_T)
            assert engine.cycle_counter == 0

    def test_process_byte(self, mock_env):
        """Test processing a byte with explicit mutation logic"""
        # Use a real InferenceEngine for this test
        engine = InferenceEngine()
        engine.T = np.arange(4 * 2 * 3 * 2, dtype=np.float32).reshape(4, 2, 3, 2)
        initial_T = engine.T.copy()
        input_byte = 0x55
        gene_mutated = input_byte ^ 0xAA

        # Manually apply the expected operations
        expected_T = initial_T.copy()
        for i in range(8):
            if gene_mutated & (1 << i):
                expected_T = apply_operation(expected_T, i)

        # Call the method under test and UNPACK the tuple
        key_index, resonance = engine.process_byte(input_byte)

        # Assert the tensor matches the expected result
        np.testing.assert_array_equal(engine.T, expected_T)

        # Check that cycle counter incremented
        assert engine.cycle_counter == 1

        # Check return values are valid
        assert isinstance(key_index, int)
        assert 0 <= key_index < 256
        assert isinstance(resonance, float)

    def test_find_closest_pattern_index(self, mock_env):
        """Test finding the closest pattern"""
        engine = InferenceEngine()
        # Set tensor to all ones
        engine.T.fill(1.0)
        # Set all patterns to zeros except index 42, which matches the tensor
        engine.F = np.zeros((256, 48))
        engine.F[42] = np.ones(48)  # Make pattern 42 closest

        closest_index, min_distance = engine.find_closest_pattern_index()
        assert closest_index == 42
        assert min_distance == 0.0

    def test_canonical_pattern_matching(self):
        # Generate canonical patterns
        patterns, _ = derive_canonical_patterns()
        # Test several masks for robustness
        for mask in [0, 1, 37, 128, 255]:
            # Generate the tensor by applying the mask to the base tensor
            base_tensor = gene_add.copy().astype(np.float32)
            T = base_tensor.copy()
            for i in range(8):
                if mask & (1 << i):
                    apply_operation(T, i)
            flat_T = T.flatten().astype(np.float32)
            # The pattern at index 'mask' should match the tensor we generated
            assert np.allclose(flat_T, patterns[mask]), f"Pattern mismatch for mask {mask}"

    def test_compute_pattern_resonances(self, mock_env):
        """Test computing pattern resonances"""
        engine = InferenceEngine()
        # Create a controlled set of patterns for testing
        engine.F = np.zeros((256, 48))
        engine.F[0] = np.ones(48)  # First pattern has all 1s
        engine.F[1] = -np.ones(48)  # Second pattern has all -1s

        # Set current tensor to all 1s
        engine.T.fill(1.0)

        # Compute resonances
        resonances = engine.compute_pattern_resonances()

        # Check that first pattern has zero distance (perfect match)
        assert len(resonances) == 256
        assert resonances[0] == 0.0

        # Check that second pattern has maximum distance (opposite)
        assert np.isclose(resonances[1], np.pi)

        # Check that other patterns have intermediate distances
        for i in range(2, 256):
            assert 0.0 < resonances[i] <= np.pi


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
