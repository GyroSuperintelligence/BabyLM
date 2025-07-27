"""
Comprehensive tests for governance.py - the physics layer of GyroSI.
Tests core mathematical operations, transformations, and invariants.
"""

import numpy as np
import itertools
import random
from baby import governance


class TestConstants:
    """Test fundamental constants and their properties."""

    def test_gene_mic_s_constant(self) -> None:
        """Test GENE_Mic_S is the expected holographic constant."""
        assert governance.GENE_Mic_S == 0xAA
        assert governance.GENE_Mic_S == 0b10101010
        assert bin(governance.GENE_Mic_S).count("1") == 4  # Balanced parity

    def test_gene_mac_s_tensor_structure(self) -> None:
        """Test GENE_Mac_S tensor has correct structure."""
        tensor = governance.GENE_Mac_S

        # Shape validation
        assert tensor.shape == (4, 2, 3, 2)
        assert tensor.dtype == np.int8

        # Value validation - only ±1 allowed
        unique_vals = np.unique(tensor)
        assert np.array_equal(unique_vals, np.array([-1, 1]))

        # Total element count
        assert tensor.size == 48

    def test_exon_masks(self) -> None:
        """Test exon mask constants have correct bit patterns."""
        # Verify bit positions
        assert governance.EXON_LI_MASK == 0b01000010  # bits 1, 6
        assert governance.EXON_FG_MASK == 0b00100100  # bits 2, 5
        assert governance.EXON_BG_MASK == 0b00011000  # bits 3, 4

        # Verify they're disjoint
        assert (governance.EXON_LI_MASK & governance.EXON_FG_MASK) == 0
        assert (governance.EXON_LI_MASK & governance.EXON_BG_MASK) == 0
        assert (governance.EXON_FG_MASK & governance.EXON_BG_MASK) == 0

        # Verify dynamic mask is union
        expected_dynamic = governance.EXON_LI_MASK | governance.EXON_FG_MASK | governance.EXON_BG_MASK
        assert governance.EXON_DYNAMIC_MASK == expected_dynamic

    def test_precomputed_masks(self) -> None:
        """Test precomputed transformation masks are valid."""
        # Verify arrays exist and have correct size
        assert len(governance.INTRON_BROADCAST_MASKS) == 256
        assert len(governance.XFORM_MASK) == 256

        # Verify they're within expected ranges
        assert all(0 <= mask < (1 << 48) for mask in governance.XFORM_MASK)


class TestGovernanceSignature:
    """Test governance signature computation."""

    def test_empty_mask(self) -> None:
        """Test signature of empty mask."""
        sig = governance.compute_governance_signature(0)
        assert sig == (6, 0, 0, 0, 0)  # (neutral_reserve, li, fg, bg, dyn)

    def test_full_dynamic_mask(self) -> None:
        """Test signature when all dynamic bits are set."""
        full_mask = governance.EXON_DYNAMIC_MASK
        sig = governance.compute_governance_signature(full_mask)
        assert sig == (0, 2, 2, 2, 6)  # All dynamic bits set

    def test_li_only(self) -> None:
        """Test signature with only LI bits set."""
        sig = governance.compute_governance_signature(governance.EXON_LI_MASK)
        assert sig == (4, 2, 0, 0, 2)

    def test_fg_only(self) -> None:
        """Test signature with only FG bits set."""
        sig = governance.compute_governance_signature(governance.EXON_FG_MASK)
        assert sig == (4, 0, 2, 0, 2)

    def test_bg_only(self) -> None:
        """Test signature with only BG bits set."""
        sig = governance.compute_governance_signature(governance.EXON_BG_MASK)
        assert sig == (4, 0, 0, 2, 2)

    def test_mask_overflow_handling(self) -> None:
        """Test that masks are properly masked to 8 bits."""
        large_mask = 0x1AA  # 9 bits
        sig = governance.compute_governance_signature(large_mask)
        # Should be equivalent to 0xAA
        expected_sig = governance.compute_governance_signature(0xAA)
        assert sig == expected_sig

    def test_signature_invariants(self) -> None:
        """Test mathematical invariants of governance signatures."""
        for mask in range(256):
            neutral, li, fg, bg, dyn = governance.compute_governance_signature(mask)

            # Dynamic population consistency
            assert dyn == li + fg + bg

            # Neutral reserve consistency
            assert neutral == 6 - dyn
            assert 0 <= neutral <= 6

            # Individual counters in valid range
            assert 0 <= li <= 2
            assert 0 <= fg <= 2
            assert 0 <= bg <= 2


class TestMonodromicFold:
    """Test the Monodromic Fold operation - core of learning."""

    def test_left_identity(self) -> None:
        """Test fold(0, b) = b (CS Emergence)."""
        for b in range(256):
            result = governance.fold(0, b)
            assert result == b, f"Left identity failed for b={b}"

    def test_right_absorber(self) -> None:
        """Test fold(a, 0) = 0 (Return to CS)."""
        for a in range(256):
            result = governance.fold(a, 0)
            assert result == 0, f"Right absorber failed for a={a}"

    def test_self_annihilation(self) -> None:
        """Test fold(a, a) = 0 (BU Closure)."""
        for a in range(256):
            result = governance.fold(a, a)
            assert result == 0, f"Self annihilation failed for a={a}"

    def test_non_commutativity(self) -> None:
        """Test that fold is non-commutative."""
        non_commutative_pairs = []
        for a in range(1, 16):  # Sample subset
            for b in range(1, 16):
                if a != b:
                    if governance.fold(a, b) != governance.fold(b, a):
                        non_commutative_pairs.append((a, b))

        # Should find many non-commutative pairs
        assert len(non_commutative_pairs) > 10

    def test_non_associativity(self) -> None:
        """Test that fold is non-associative."""
        non_associative_triples = []
        for a in range(1, 8):  # Small sample
            for b in range(1, 8):
                for c in range(1, 8):
                    left_assoc = governance.fold(governance.fold(a, b), c)
                    right_assoc = governance.fold(a, governance.fold(b, c))
                    if left_assoc != right_assoc:
                        non_associative_triples.append((a, b, c))

        # Should find non-associative triples
        assert len(non_associative_triples) > 5

    def test_fold_preserves_byte_range(self) -> None:
        """Test fold always returns values in [0, 255]."""
        for a in range(256):
            for b in [0, 1, 42, 128, 255]:  # Sample b values
                result = governance.fold(a, b)
                assert 0 <= result <= 255

    def test_fold_sequence_empty(self) -> None:
        """Test fold_sequence with empty list."""
        result = governance.fold_sequence([])
        assert result == 0

        result = governance.fold_sequence([], start_state=42)
        assert result == 42

    def test_fold_sequence_single(self) -> None:
        """Test fold_sequence with single element."""
        for start in [0, 42, 255]:
            for intron in [0, 1, 128, 255]:
                result = governance.fold_sequence([intron], start)
                expected = governance.fold(start, intron)
                assert result == expected

    def test_fold_sequence_ordering(self) -> None:
        """Test that fold_sequence respects order."""
        introns = [1, 2, 3]

        # Manual sequential fold
        result = 0
        for intron in introns:
            result = governance.fold(result, intron)

        # fold_sequence should match
        seq_result = governance.fold_sequence(introns)
        assert result == seq_result

    def test_fold_sequence_different_orders(self) -> None:
        """Test that different orders give different results (non-commutativity)."""
        introns = [random.randint(0, 255) for _ in range(5)]
        base = governance.fold_sequence(introns)
        found_diff = any(
            governance.fold_sequence(list(p)) != base for p in itertools.permutations(introns) if list(p) != introns
        )
        assert found_diff, "All permutations collapsed to the same value – investigate fold implementation."


class TestDualOperation:
    """Test the Global Duality Operator."""

    def test_dual_involution(self) -> None:
        """Test dual(dual(x)) = x (involution property)."""
        for x in range(256):
            result = governance.dual(governance.dual(x))
            assert result == x

    def test_dual_complement(self) -> None:
        """Test dual is bitwise complement."""
        for x in range(256):
            result = governance.dual(x)
            expected = (x ^ 0xFF) & 0xFF
            assert result == expected

    def test_dual_boundary_values(self) -> None:
        """Test dual on boundary values."""
        assert governance.dual(0) == 255
        assert governance.dual(255) == 0
        assert governance.dual(0xAA) == 0x55


class TestGyrationTransform:
    """Test gyroscopic transformations."""

    def test_transform_preserves_48_bit(self) -> None:
        """Test transforms stay within 48-bit bounds."""
        test_states = [0, 1, (1 << 47) - 1, (1 << 48) - 1]

        for state in test_states:
            for intron in [0, 1, 42, 128, 255]:
                result = governance.apply_gyration_and_transform(state, intron)
                assert 0 <= result < (1 << 48)

    def test_transform_intron_masking(self) -> None:
        """Test that intron is properly masked to 8 bits."""
        state = 12345
        intron_large = 0x1FF  # 9 bits
        intron_masked = 0xFF  # 8 bits

        result1 = governance.apply_gyration_and_transform(state, intron_large)
        result2 = governance.apply_gyration_and_transform(state, intron_masked)
        assert result1 == result2

    def test_batch_consistency(self) -> None:
        """Test batch transform matches individual transforms."""
        states = np.array([0, 1, 12345, (1 << 24)], dtype=np.uint64)
        intron = 42

        # Individual transforms
        individual_results = []
        for state in states:
            result = governance.apply_gyration_and_transform(int(state), intron)
            individual_results.append(result)

        # Batch transform
        batch_results = governance.apply_gyration_and_transform_batch(states, intron)

        # Should match
        assert np.array_equal(batch_results, np.array(individual_results, dtype=np.uint64))

    def test_all_introns_transform(self) -> None:
        """Test all-introns transform produces correct shape."""
        states = np.array([0, 1, 12345], dtype=np.uint64)
        results = governance.apply_gyration_and_transform_all_introns(states)

        assert results.shape == (len(states), 256)
        assert results.dtype == np.uint64

        # Verify first state matches individual transforms
        for intron in range(256):
            expected = governance.apply_gyration_and_transform(int(states[0]), intron)
            assert results[0, intron] == expected

    def test_origin_state_transforms(self) -> None:
        from baby.information import InformationEngine

        origin = InformationEngine.tensor_to_int(governance.GENE_Mac_S)
        # All introns should produce valid states
        results = set()
        for intron in range(256):
            result = governance.apply_gyration_and_transform(origin, intron)
            assert 0 <= result < (1 << 48)
            results.add(result)
        # Should produce multiple distinct states
        assert len(results) > 1


class TestTranscription:
    """Test byte transcription through holographic topology."""

    def test_transcribe_boundary_values(self) -> None:
        """Test transcription of boundary values."""
        assert governance.transcribe_byte(0) == (0 ^ governance.GENE_Mic_S)
        assert governance.transcribe_byte(255) == (255 ^ governance.GENE_Mic_S)

    def test_transcribe_preserves_range(self) -> None:
        """Test transcription preserves byte range."""
        for byte in range(256):
            result = governance.transcribe_byte(byte)
            assert 0 <= result <= 255

    def test_transcribe_is_involution(self) -> None:
        """Test transcribe(transcribe(x)) = x."""
        for byte in range(256):
            double_transcribed = governance.transcribe_byte(governance.transcribe_byte(byte))
            assert double_transcribed == byte

    def test_transcribe_xor_property(self) -> None:
        """Test transcription is XOR with GENE_Mic_S."""
        for byte in range(256):
            result = governance.transcribe_byte(byte)
            expected = byte ^ governance.GENE_Mic_S
            assert result == expected


class TestTensorValidation:
    """Test tensor consistency validation."""

    def test_validate_tensor_consistency_passes(self) -> None:
        """Test that GENE_Mac_S passes validation."""
        assert governance.validate_tensor_consistency() is True

    def test_validation_catches_wrong_shape(self) -> None:
        """Test validation catches incorrect tensor shape."""
        # Temporarily modify tensor shape (this is a bit hacky but tests the validation)
        original_tensor = governance.GENE_Mac_S.copy()

        # Create malformed tensor for testing
        wrong_tensor = np.ones((3, 2, 3, 2), dtype=np.int8)

        # Replace temporarily
        governance.GENE_Mac_S = wrong_tensor
        try:
            result = governance.validate_tensor_consistency()
            assert result is False
        finally:
            # Restore original
            governance.GENE_Mac_S = original_tensor

    def test_validation_catches_wrong_values(self) -> None:
        """Test validation catches values other than ±1."""
        original_tensor = governance.GENE_Mac_S.copy()

        # Wrong values
        wrong_tensor = np.zeros((4, 2, 3, 2), dtype=np.int8)

        governance.GENE_Mac_S = wrong_tensor
        try:
            result = governance.validate_tensor_consistency()
            assert result is False
        finally:
            governance.GENE_Mac_S = original_tensor


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_state_handling(self) -> None:
        """Test handling of states near 48-bit boundary."""
        max_48_bit = (1 << 48) - 1

        # Should handle maximum 48-bit value
        result = governance.apply_gyration_and_transform(max_48_bit, 0)
        assert 0 <= result < (1 << 48)

    def test_negative_inputs_handled(self) -> None:
        """Test that negative inputs are handled gracefully."""
        # apply_gyration_and_transform should handle negative inputs
        # by treating them as large positive integers (Python int behavior)
        result = governance.apply_gyration_and_transform(-1, 0)
        assert isinstance(result, int)

    def test_fold_extreme_values(self) -> None:
        """Test fold with extreme input values."""
        # Test with maximum byte values
        result = governance.fold(255, 255)
        assert result == 0  # Self-annihilation

        result = governance.fold(255, 0)
        assert result == 0  # Right absorber

    def test_governance_signature_comprehensive(self) -> None:
        """Test governance signature for all possible masks."""
        signatures = set()
        for mask in range(256):
            sig = governance.compute_governance_signature(mask)
            signatures.add(sig)

            # Verify signature is valid
            neutral, li, fg, bg, dyn = sig
            assert 0 <= neutral <= 6
            assert 0 <= li <= 2
            assert 0 <= fg <= 2
            assert 0 <= bg <= 2
            assert dyn == li + fg + bg
            assert neutral + dyn == 6

        # Should have multiple distinct signatures
        assert len(signatures) > 10


class TestExonProduct:
    """Test exon_product_from_metadata function."""

    def test_exon_product_from_metadata_basic(self) -> None:
        """Test basic exon_product_from_metadata functionality."""
        from baby.contracts import GovernanceSignature

        # Test with a simple signature
        sig: GovernanceSignature = {"neutral": 4, "li": 1, "fg": 1, "bg": 0, "dyn": 2}

        confidence = 0.5
        orbit_v = 100
        v_max = 200

        product = governance.exon_product_from_metadata(sig, confidence, orbit_v, v_max)

        # Should return an 8-bit value
        assert 0 <= product <= 255

        # Test that fold(p, p) == 0 for the returned product
        assert governance.fold(product, product) == 0

    def test_exon_product_from_metadata_edge_cases(self) -> None:
        """Test exon_product_from_metadata with edge cases."""
        from baby.contracts import GovernanceSignature

        # Test with zero confidence
        sig: GovernanceSignature = {"neutral": 6, "li": 0, "fg": 0, "bg": 0, "dyn": 0}

        product = governance.exon_product_from_metadata(sig, 0.0, 100, 200)
        assert product == 0

        # Test with full confidence and maximum values
        sig_full: GovernanceSignature = {"neutral": 0, "li": 2, "fg": 2, "bg": 2, "dyn": 6}

        product = governance.exon_product_from_metadata(sig_full, 1.0, 200, 200)
        assert 0 <= product <= 255
        assert governance.fold(product, product) == 0


class TestMathematicalProperties:
    """Test mathematical properties and invariants."""

    def test_fold_closure(self) -> None:
        """Test that fold is closed over the byte domain."""
        for a in range(0, 256, 17):  # Sample every 17th value
            for b in range(0, 256, 23):  # Sample every 23rd value
                result = governance.fold(a, b)
                assert 0 <= result <= 255

    def test_transform_deterministic(self) -> None:
        """Test that transforms are deterministic."""
        state = 12345
        intron = 42

        # Multiple calls should give same result
        result1 = governance.apply_gyration_and_transform(state, intron)
        result2 = governance.apply_gyration_and_transform(state, intron)
        result3 = governance.apply_gyration_and_transform(state, intron)

        assert result1 == result2 == result3

    def test_fold_sequence_associativity_failure(self) -> None:
        """Demonstrate that fold_sequence is not associative due to fold properties."""
        a, b, c = 3, 5, 7

        # Standard left fold (fold_sequence)
        sequential = governance.fold_sequence([a, b, c])
        left_grouped = governance.fold(governance.fold(a, b), c)
        right_grouped = governance.fold(a, governance.fold(b, c))

        # fold_sequence should be a left fold
        assert sequential == left_grouped
        # Demonstrate non-associativity
        assert left_grouped != right_grouped

    def test_broadcast_masks_coverage(self) -> None:
        """Test that broadcast masks cover expected patterns."""
        # Verify first few broadcast masks
        assert governance.INTRON_BROADCAST_MASKS[0] == 0  # All zeros
        assert governance.INTRON_BROADCAST_MASKS[1] == 0x010101010101  # Repeated 0x01
        assert governance.INTRON_BROADCAST_MASKS[255] == 0xFFFFFFFFFFFF  # All ones

    def test_xform_mask_identity(self) -> None:
        """Test transformation mask for identity (zero intron)."""
        identity_mask = governance.XFORM_MASK[0]

        # For intron 0, no LI/FG/BG bits set, so no transformation
        assert identity_mask == 0

    def test_xform_mask_full_activation(self) -> None:
        """Test transformation mask for full LI+FG+BG activation."""
        full_intron = governance.EXON_LI_MASK | governance.EXON_FG_MASK | governance.EXON_BG_MASK
        full_mask = governance.XFORM_MASK[full_intron]

        # Should combine all transformation effects
        expected = governance.FULL_MASK ^ governance.FG_MASK ^ governance.BG_MASK
        assert full_mask == expected
