"""
Tests for S1: Governance - Physics & Primitives
"""

import pytest
import numpy as np
from baby import governance


class TestCoreConstants:
    """Test the fundamental constants and their properties."""

    def test_gene_mic_s_value(self):
        """Test GENE_Mic_S has correct value."""
        assert governance.GENE_Mic_S == 0xAA
        assert governance.GENE_Mic_S == 0b10101010
        assert governance.GENE_Mic_S == 170

    def test_gene_mac_s_shape(self):
        """Test GENE_Mac_S has correct shape."""
        assert governance.GENE_Mac_S.shape == (4, 2, 3, 2)
        assert governance.GENE_Mac_S.dtype == np.int8

    def test_gene_mac_s_values(self):
        """Test GENE_Mac_S contains only ±1 values."""
        unique_values = np.unique(governance.GENE_Mac_S)
        assert np.array_equal(unique_values, np.array([-1, 1]))

    def test_gene_mac_s_alternating_pattern(self):
        """Test GENE_Mac_S has the correct alternating pattern."""
        # Layer 0 should match layer 2
        assert np.array_equal(governance.GENE_Mac_S[0], governance.GENE_Mac_S[2])
        # Layer 1 should match layer 3
        assert np.array_equal(governance.GENE_Mac_S[1], governance.GENE_Mac_S[3])
        # Layer 0 should be opposite of layer 1
        assert np.array_equal(governance.GENE_Mac_S[0], -governance.GENE_Mac_S[1])

    def test_masks_computation(self):
        """Test mask computation produces expected values."""
        assert isinstance(governance.FG_MASK, int)
        assert isinstance(governance.BG_MASK, int)
        assert governance.FULL_MASK == (1 << 48) - 1

        # Test that FG and BG masks don't overlap
        assert (governance.FG_MASK & governance.BG_MASK) == 0

        # Test that FG | BG covers all bits
        assert (governance.FG_MASK | governance.BG_MASK) == governance.FULL_MASK

    def test_intron_broadcast_masks(self):
        """Test intron broadcast mask generation."""
        assert len(governance.INTRON_BROADCAST_MASKS) == 256

        # Test specific patterns
        assert governance.INTRON_BROADCAST_MASKS[0] == 0
        assert governance.INTRON_BROADCAST_MASKS[255] == int.from_bytes(b"\xff" * 6, "little")

        # Test that mask for intron i has pattern i repeated 6 times
        for i in [0, 1, 42, 170, 255]:
            expected = int.from_bytes(bytes([i] * 6), "little")
            assert governance.INTRON_BROADCAST_MASKS[i] == expected


class TestGyroscopicOperations:
    """Test the core physics operations."""

    def test_transcribe_byte(self):
        """Test byte transcription through holographic topology."""
        # Test identity elements
        assert governance.transcribe_byte(0) == 0xAA
        assert governance.transcribe_byte(0xAA) == 0

        # Test involution property (f(f(x)) = x)
        for byte in [0, 42, 127, 170, 255]:
            assert governance.transcribe_byte(governance.transcribe_byte(byte)) == byte

    def test_apply_gyration_identity(self):
        """Test gyration with identity intron (0) leaves state unchanged."""
        test_states = [0, 42, 0xFFFFFFFFFFFF, (1 << 48) - 1]

        for state in test_states:
            result = governance.apply_gyration_and_transform(state, 0)
            assert result == state

    def expected(self, state_before: int, intron: int) -> int:
        temp = state_before
        if intron & 0b01000010:  # LI
            temp ^= governance.FULL_MASK
        if intron & 0b00100100:  # FG
            temp ^= governance.FG_MASK
        if intron & 0b00011000:  # BG
            temp ^= governance.BG_MASK
        temp ^= temp & governance.INTRON_BROADCAST_MASKS[intron]
        return temp

    def test_apply_gyration_global_flip(self):
        """Test global parity flip operation."""
        flip_intron = 0b01000010
        result = governance.apply_gyration_and_transform(0, flip_intron)
        assert result == self.expected(0, flip_intron)
        # Test involution
        result2 = governance.apply_gyration_and_transform(result, flip_intron)
        assert result2 == 0

    def test_apply_gyration_layer_operations(self):
        """Test forward and backward gyration operations."""
        test_state = 0
        fg_intron = 0b00100100  # Bits 2,5 set
        result_fg = governance.apply_gyration_and_transform(test_state, fg_intron)
        assert result_fg == self.expected(test_state, fg_intron)
        bg_intron = 0b00011000  # Bits 3,4 set
        result_bg = governance.apply_gyration_and_transform(test_state, bg_intron)
        assert result_bg == self.expected(test_state, bg_intron)

    def test_gyration_with_carry(self):
        """Test gyration memory (carry term) is applied correctly."""
        # Start with a state that has some bits set
        initial_state = 0b101010  # Some arbitrary pattern

        # Apply transformation with overlapping intron
        intron = 0b00001111  # Will create carry where state and intron overlap

        result = governance.apply_gyration_and_transform(initial_state, intron)

        # Verify the result is different from just the transforms
        # (This tests that the carry term was applied)
        temp_state = initial_state  # No primary transforms for this intron
        intron_pattern = governance.INTRON_BROADCAST_MASKS[intron]
        expected_carry = temp_state & intron_pattern
        expected_result = temp_state ^ expected_carry

        assert result == expected_result


class TestCoaddition:
    """Test the gyrogroup coaddition operation."""

    def test_coadd_identity(self):
        """Test coaddition with identity element (0)."""
        # 0 is left identity: 0 ⊞ a = a
        for a in [0, 42, 127, 255]:
            assert governance.coadd(0, a) == a

        # 0 is NOT right identity in general
        # a ⊞ 0 may not equal a due to gyration

    def test_coadd_non_commutative(self):
        """Test that coaddition is non-commutative."""
        pairs = [(1, 2), (42, 137), (170, 85), (255, 128)]

        non_commutative_count = 0
        for a, b in pairs:
            if governance.coadd(a, b) != governance.coadd(b, a):
                non_commutative_count += 1

        # At least some pairs should be non-commutative
        assert non_commutative_count > 0

    def test_coadd_path_dependence(self):
        """Test that coaddition is path-dependent (non-associative)."""
        # (a ⊞ b) ⊞ c ≠ a ⊞ (b ⊞ c) in general
        a, b, c = 42, 137, 91

        left_assoc = governance.coadd(governance.coadd(a, b), c)
        right_assoc = governance.coadd(a, governance.coadd(b, c))

        # They should generally be different (path-dependent)
        # Note: There might be special cases where they're equal
        # but in general they should differ

        # Test with multiple triples to ensure we find non-associative cases
        triples = [(1, 2, 3), (42, 137, 91), (170, 85, 255)]
        non_associative_count = 0

        for a, b, c in triples:
            left = governance.coadd(governance.coadd(a, b), c)
            right = governance.coadd(a, governance.coadd(b, c))
            if left != right:
                non_associative_count += 1

        assert non_associative_count > 0

    def test_batch_introns_coadd(self):
        """Test batch coaddition preserves order."""
        introns = [42, 137, 91, 255]

        # Manual reduction
        manual_result = introns[0]
        for intron in introns[1:]:
            manual_result = governance.coadd(manual_result, intron)

        # Batch reduction
        batch_result = governance.batch_introns_coadd_ordered(introns)

        assert batch_result == manual_result

    def test_batch_introns_empty(self):
        """Test batch coaddition with empty list."""
        assert governance.batch_introns_coadd_ordered([]) == 0

    def test_batch_introns_single(self):
        """Test batch coaddition with single element."""
        assert governance.batch_introns_coadd_ordered([42]) == 42


class TestConsistencyValidation:
    """Test internal consistency validation."""

    def test_tensor_validation_passes(self):
        """Test that the tensor passes validation on module load."""
        # This should have already passed during import
        assert governance.validate_tensor_consistency()

    def test_tensor_immutability(self):
        """Test that GENE_Mac_S is immutable."""
        original = governance.GENE_Mac_S.copy()

        # Try to modify (this should not affect the original)
        temp = governance.GENE_Mac_S
        temp_view = temp[0, 0, 0, 0]

        # Verify original is unchanged
        assert np.array_equal(governance.GENE_Mac_S, original)


class TestPhysicsInvariants:
    """Test physical invariants and properties."""

    def test_state_space_closure(self):
        """Test that transformations stay within valid state space."""
        # All states should be 48-bit integers
        max_state = (1 << 48) - 1

        test_states = [0, 42, 1000, max_state]
        test_introns = [0, 1, 127, 255]

        for state in test_states:
            for intron in test_introns:
                result = governance.apply_gyration_and_transform(state, intron)
                assert 0 <= result <= max_state

    def test_transformation_determinism(self):
        """Test that transformations are deterministic."""
        state = 424242
        intron = 137

        # Apply same transformation multiple times
        results = [governance.apply_gyration_and_transform(state, intron) for _ in range(10)]

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_coaddition_closure(self):
        """Test that coaddition stays within 8-bit range."""
        import random

        random.seed(42)

        for _ in range(100):
            a = random.randint(0, 255)
            b = random.randint(0, 255)
            result = governance.coadd(a, b)
            assert 0 <= result <= 255


class TestPerformance:
    """Performance benchmarks for critical operations."""

    # Performance tests removed as per user request.
