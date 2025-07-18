"""
Tests for S1: Governance - Physics & Primitives

Tests the fundamental constants and pure functions that define
the physical laws of the GyroSI system.
"""

import pytest
import numpy as np
from typing import List

from baby import governance


class TestConstants:
    """Test the fundamental constants of the system."""

    def test_gene_mic_s_value(self):
        """Test that GENE_Mic_S has the correct binary pattern."""
        assert governance.GENE_Mic_S == 0xAA
        assert governance.GENE_Mic_S == 0b10101010
        assert governance.GENE_Mic_S == 170

    def test_gene_mac_s_structure(self):
        """Test the structure and properties of GENE_Mac_S tensor."""
        tensor = governance.GENE_Mac_S

        # Check shape
        assert tensor.shape == (4, 2, 3, 2)

        # Check dtype
        assert tensor.dtype == np.int8

        # Check all values are ±1
        unique_values = np.unique(tensor)
        assert np.array_equal(unique_values, np.array([-1, 1]))

        # Check total size
        assert tensor.size == 48

        # Verify alternating pattern at each layer
        for layer in range(4):
            if layer % 2 == 0:  # Layers 0 and 2
                assert np.all(tensor[layer, 0, :, 0] == -1)
                assert np.all(tensor[layer, 0, :, 1] == 1)
                assert np.all(tensor[layer, 1, :, 0] == 1)
                assert np.all(tensor[layer, 1, :, 1] == -1)
            else:  # Layers 1 and 3
                assert np.all(tensor[layer, 0, :, 0] == 1)
                assert np.all(tensor[layer, 0, :, 1] == -1)
                assert np.all(tensor[layer, 1, :, 0] == -1)
                assert np.all(tensor[layer, 1, :, 1] == 1)


class TestMaskGeneration:
    """Test mask generation and pre-computed masks."""

    def test_build_masks_and_constants(self):
        """Test that build_masks_and_constants generates correct masks."""
        fg, bg, full, intron_masks = governance.build_masks_and_constants()

        # Test FULL_MASK is all 48 bits set
        assert full == (1 << 48) - 1
        assert full == 0xFFFFFFFFFFFF

        # Test that FG and BG are complementary (cover all bits)
        assert (fg | bg) == full
        assert (fg & bg) == 0  # No overlap

        # Test intron broadcast masks
        assert len(intron_masks) == 256

        # Test specific intron patterns
        assert intron_masks[0] == 0  # All zeros
        assert intron_masks[255] == 0xFFFFFFFFFFFF  # All ones

        # Test a middle value (e.g., 0xAA = 10101010)
        expected_aa = 0xAAAAAAAAAAAA  # 0xAA repeated 6 times
        assert intron_masks[0xAA] == expected_aa

    def test_precomputed_masks(self):
        """Test that module-level masks are correctly initialized."""
        assert governance.FULL_MASK == (1 << 48) - 1
        assert isinstance(governance.FG_MASK, int)
        assert isinstance(governance.BG_MASK, int)
        assert len(governance.INTRON_BROADCAST_MASKS) == 256


class TestTransformations:
    """Test the physics transformation functions."""

    def test_transcribe_byte(self):
        """Test byte transcription through XOR."""
        # Test identity property
        assert governance.transcribe_byte(0) == 0xAA
        assert governance.transcribe_byte(0xAA) == 0

        # Test involution property (applying twice returns original)
        for byte in [0, 1, 127, 128, 255]:
            transcribed = governance.transcribe_byte(byte)
            double_transcribed = governance.transcribe_byte(transcribed)
            assert double_transcribed == byte

        # Test specific values
        assert governance.transcribe_byte(0xFF) == 0x55  # 11111111 XOR 10101010 = 01010101
        assert governance.transcribe_byte(0x00) == 0xAA  # 00000000 XOR 10101010 = 10101010

    def test_apply_gyration_and_transform_no_transform(self):
        """Test gyration with intron that triggers no transformations."""
        # Intron 0x00 should not trigger any bit pattern transformations
        state = 0x123456789ABC
        result = governance.apply_gyration_and_transform(state, 0x00)
        assert result == state  # No change expected

    def test_apply_gyration_and_transform_global_parity(self):
        """Test global parity flip transformation."""
        # Intron with bits 1,6 set (0b01000010 = 0x42)
        state = 0x123456789ABC
        result = governance.apply_gyration_and_transform(state, 0x42)

        # Should flip all bits, then apply gyration
        expected_after_flip = state ^ governance.FULL_MASK
        intron_pattern = governance.INTRON_BROADCAST_MASKS[0x42]
        expected_final = expected_after_flip ^ (expected_after_flip & intron_pattern)

        assert result == expected_final

    def test_apply_gyration_and_transform_combined(self):
        """Test combined transformations."""
        # Intron with all transform bits set: 0b01111110 = 0x7E
        state = 0xABCDEF123456
        result = governance.apply_gyration_and_transform(state, 0x7E)

        # Verify result is different from input
        assert result != state

        # Verify result is within valid range
        assert 0 <= result <= governance.FULL_MASK


class TestGyrogroupOperations:
    """Test gyrogroup algebraic operations."""

    def test_coadd_basic(self):
        """Test basic coadd operations."""
        # Test identity
        assert governance.coadd(0, 0) == 0

        # Test with specific values
        result = governance.coadd(0xAA, 0x55)
        assert isinstance(result, int)
        assert 0 <= result <= 255

    def test_coadd_non_commutative(self):
        """Test that coadd is non-commutative."""
        a, b = 0x12, 0x34
        assert governance.coadd(a, b) != governance.coadd(b, a)

        # Test with multiple pairs
        pairs = [(0xAA, 0x55), (0xFF, 0x00), (0x0F, 0xF0)]
        for a, b in pairs:
            if a != b:  # Only test when a != b
                assert governance.coadd(a, b) != governance.coadd(b, a), f"coadd({a}, {b}) was commutative!"

    def test_coadd_properties(self):
        """Test mathematical properties of coadd."""
        # Test the actual formula: a ⊞ b = a ⊕ gyr[a, ¬b](b)
        a, b = 0x3C, 0xC3
        not_b = b ^ 0xFF
        gyration_of_b = b ^ (a & not_b)
        expected = a ^ gyration_of_b
        assert governance.coadd(a, b) == expected

    def test_batch_introns_coadd_ordered_empty(self):
        """Test batch coadd with empty list."""
        assert governance.batch_introns_coadd_ordered([]) == 0

    def test_batch_introns_coadd_ordered_single(self):
        """Test batch coadd with single element."""
        assert governance.batch_introns_coadd_ordered([42]) == 42
        assert governance.batch_introns_coadd_ordered([0xFF]) == 0xFF

    def test_batch_introns_coadd_ordered_multiple(self):
        """Test batch coadd with multiple elements."""
        introns = [0x12, 0x34, 0x56]
        result = governance.batch_introns_coadd_ordered(introns)

        # Manually compute expected result
        expected = governance.coadd(governance.coadd(0x12, 0x34), 0x56)
        assert result == expected

    def test_batch_introns_coadd_order_matters(self):
        """Test that order matters in batch coadd for at least some inputs."""
        test_cases = [
            ([0xAA, 0x55, 0xFF], [0xFF, 0x55, 0xAA]),
            ([1, 2, 3], [3, 2, 1]),
            ([10, 20, 30], [30, 20, 10]),
            ([42, 137, 91], [91, 137, 42]),
            ([5, 7, 11], [11, 7, 5]),
        ]
        found_difference = False
        for a, b in test_cases:
            if governance.batch_introns_coadd_ordered(a) != governance.batch_introns_coadd_ordered(b):
                found_difference = True
                break
        assert found_difference, "Order did not affect result for any tested input"


class TestValidation:
    """Test validation functions."""

    def test_validate_tensor_consistency(self):
        """Test that the tensor validation passes for the correct tensor."""
        assert governance.validate_tensor_consistency() is True

    def test_validate_tensor_consistency_mock_failure(self, monkeypatch):
        """Test tensor validation with invalid tensors."""
        # Test wrong shape
        wrong_shape = np.ones((3, 2, 3, 2), dtype=np.int8)
        monkeypatch.setattr(governance, "GENE_Mac_S", wrong_shape)
        assert governance.validate_tensor_consistency() is False

        # Reset for next test
        monkeypatch.undo()

        # Test wrong dtype
        wrong_dtype = np.ones((4, 2, 3, 2), dtype=np.float32)
        monkeypatch.setattr(governance, "GENE_Mac_S", wrong_dtype)
        assert governance.validate_tensor_consistency() is False

        # Reset for next test
        monkeypatch.undo()

        # Test wrong values (not ±1)
        wrong_values = np.full((4, 2, 3, 2), 2, dtype=np.int8)
        monkeypatch.setattr(governance, "GENE_Mac_S", wrong_values)
        assert governance.validate_tensor_consistency() is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_state_transformations(self):
        """Test transformations with maximum state values."""
        max_state = governance.FULL_MASK

        # Test with various introns
        for intron in [0x00, 0xFF, 0xAA, 0x55]:
            result = governance.apply_gyration_and_transform(max_state, intron)
            assert 0 <= result <= governance.FULL_MASK

    def test_transcribe_all_bytes(self):
        """Test transcribe_byte for all possible input values."""
        results = set()
        for byte in range(256):
            result = governance.transcribe_byte(byte)
            assert 0 <= result <= 255
            results.add(result)

        # Should produce 256 unique outputs (bijection)
        assert len(results) == 256

    def test_coadd_all_combinations_sample(self):
        """Test coadd with a sample of all possible combinations."""
        # Testing all 256*256 combinations would be slow, so sample
        import random

        random.seed(42)  # Deterministic sampling

        for _ in range(100):  # Sample 100 random pairs
            a = random.randint(0, 255)
            b = random.randint(0, 255)
            result = governance.coadd(a, b)
            assert 0 <= result <= 255
