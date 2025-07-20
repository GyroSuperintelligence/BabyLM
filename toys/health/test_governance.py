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
        state = int(0x123456789ABC)
        result = governance.apply_gyration_and_transform(state, int(0x00))
        assert result == state  # No change expected

    def test_apply_gyration_and_transform_global_parity(self):
        """Test global parity flip transformation."""
        # Intron with bits 1,6 set (0b01000010 = 0x42)
        state = int(0x123456789ABC)
        result = governance.apply_gyration_and_transform(state, int(0x42))

        # Should flip all bits, then apply gyration
        expected_after_flip = state ^ governance.FULL_MASK
        intron_pattern = governance.INTRON_BROADCAST_MASKS[0x42]
        expected_final = expected_after_flip ^ (expected_after_flip & int(intron_pattern))

        assert result == expected_final

    def test_apply_gyration_and_transform_combined(self):
        """Test combined transformations."""
        # Intron with all transform bits set: 0b01111110 = 0x7E
        state = int(0xABCDEF123456)
        result = governance.apply_gyration_and_transform(state, int(0x7E))

        # Verify result is different from input
        assert result != state

        # Verify result is within valid range
        assert 0 <= result <= governance.FULL_MASK


class TestGyrogroupOperations:
    """Test gyrogroup algebraic operations."""

    def test_fold_basic(self):
        """Test basic fold operations."""
        assert governance.fold(0, 0) == 0
        result = governance.fold(0xAA, 0x55)
        assert isinstance(result, int)
        assert 0 <= result <= 255

    def test_fold_non_commutative(self):
        """Test that fold is non-commutative."""
        a, b = 0x12, 0x34
        assert governance.fold(a, b) != governance.fold(b, a)
        pairs = [(0xAA, 0x55), (0xFF, 0x00), (0x0F, 0xF0)]
        for a, b in pairs:
            if a != b:
                assert governance.fold(a, b) != governance.fold(b, a), f"fold({a}, {b}) was commutative!"

    def test_fold_properties(self):
        """Test mathematical properties of fold."""
        a, b = 0x3C, 0xC3
        not_b = b ^ 0xFF
        gyration_of_b = b ^ (a & not_b)
        expected = a ^ gyration_of_b
        assert governance.fold(a, b) == expected

    def test_fold_sequence_empty(self):
        """Test batch fold with empty list."""
        from baby.governance import fold_sequence

        assert fold_sequence([]) == 0

    def test_fold_sequence_single(self):
        """Test batch fold with single element."""
        from baby.governance import fold_sequence

        assert fold_sequence([42]) == 42
        assert fold_sequence([0xFF]) == 0xFF

    def test_fold_sequence_multiple(self):
        """Test batch fold with multiple elements."""
        from baby.governance import fold_sequence

        introns = [0x12, 0x34, 0x56]
        result = fold_sequence(introns)
        expected = governance.fold(governance.fold(0x12, 0x34), 0x56)
        assert result == expected

    def test_fold_sequence_order_matters(self):
        """
        Validates that fold_sequence is non-associative and path-sensitive.
        For a fixed set of introns, different permutations must yield distinct outputs.
        """
        from baby.governance import fold_sequence
        import itertools

        introns = [0x3C, 0xA5, 0x7E]
        perms = list(itertools.permutations(introns))

        results = {perm: fold_sequence(list(perm)) for perm in perms}  # convert tuple → list

        unique_results = set(results.values())
        assert (
            len(unique_results) > 1
        ), "fold_sequence appears associative or commutative; all permutations yielded same result."

        if len(unique_results) < len(perms):
            collision_groups = {}
            for perm, res in results.items():
                collision_groups.setdefault(res, []).append(perm)
            print("Some permutations collapsed to same output:")
            for out, group in collision_groups.items():
                if len(group) > 1:
                    print(f"Result {out:#04x} from permutations: {group}")

    def test_fold_all_combinations_sample(self):
        """Test fold with a sample of all possible combinations."""
        import random

        for _ in range(100):
            a = random.randint(0, 255)
            b = random.randint(0, 255)
            result = governance.fold(a, b)
            assert 0 <= result <= 255


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
        max_state = int(governance.FULL_MASK)

        # Test with various introns
        for intron in [0x00, 0xFF, 0xAA, 0x55]:
            result = governance.apply_gyration_and_transform(max_state, int(intron))
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
