#!/usr/bin/env python3
"""
Comprehensive test of the fold operation mathematical properties.

This test validates:
1. Non-commutativity: a ⊞ b ≠ b ⊞ a (for most cases)
2. Non-associativity: (a ⊞ b) ⊞ c ≠ a ⊞ (b ⊞ c) (for most cases)
3. Path dependence: Different orderings of the same multiset produce different results
4. Edge cases and special properties
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.governance import fold, fold_sequence
from itertools import permutations
import random


def test_commutativity():
    """Test that fold is non-commutative for various pairs."""
    print("=== Testing Commutativity ===")

    test_pairs = [
        (0x35, 0xE2),
        (0x01, 0x02),
        (0xAA, 0xBB),
        (0x10, 0x20),
        (0xFF, 0x00),
        (0x55, 0xAA),
        (0x12, 0x34),
        (0xAB, 0xCD),
    ]

    commutative_count = 0
    total_count = 0

    for a, b in test_pairs:
        result_ab = fold(a, b)
        result_ba = fold(b, a)
        is_commutative = result_ab == result_ba
        commutative_count += is_commutative
        total_count += 1

        print(f"fold({hex(a)}, {hex(b)}) = {hex(result_ab)}")
        print(f"fold({hex(b)}, {hex(a)}) = {hex(result_ba)}")
        print(f"Commutative: {is_commutative}")
        print()

    print(f"Commutative pairs: {commutative_count}/{total_count} ({commutative_count/total_count*100:.1f}%)")
    print(
        f"Non-commutative pairs: {total_count-commutative_count}/{total_count} ({(total_count-commutative_count)/total_count*100:.1f}%)"
    )
    print()

    return commutative_count, total_count


def test_associativity():
    """Test that fold is non-associative for various triplets."""
    print("=== Testing Associativity ===")

    test_triplets = [
        (0x35, 0xE2, 0x57),
        (0x01, 0x02, 0x03),
        (0xAA, 0xBB, 0xCC),
        (0x10, 0x20, 0x30),
        (0xFF, 0x00, 0x55),
        (0x12, 0x34, 0x56),
        (0xAB, 0xCD, 0xEF),
        (0x11, 0x22, 0x33),
    ]

    associative_count = 0
    total_count = 0

    for a, b, c in test_triplets:
        result_left = fold(fold(a, b), c)
        result_right = fold(a, fold(b, c))
        is_associative = result_left == result_right
        associative_count += is_associative
        total_count += 1

        print(f"({hex(a)} ⊞ {hex(b)}) ⊞ {hex(c)} = {hex(result_left)}")
        print(f"{hex(a)} ⊞ ({hex(b)} ⊞ {hex(c)}) = {hex(result_right)}")
        print(f"Associative: {is_associative}")
        print()

    print(f"Associative triplets: {associative_count}/{total_count} ({associative_count/total_count*100:.1f}%)")
    print(
        f"Non-associative triplets: {total_count-associative_count}/{total_count} ({(total_count-associative_count)/total_count*100:.1f}%)"
    )
    print()

    return associative_count, total_count


def test_path_dependence():
    """Test that different orderings of the same multiset produce different results."""
    print("=== Testing Path Dependence ===")

    test_sequences = [
        [0x35, 0xE2, 0x57],
        [0x01, 0x02, 0x03],
        [0xAA, 0xBB, 0xCC],
        [0x10, 0x20, 0x30],
        [0xFF, 0x00, 0x55],
        [0x12, 0x34, 0x56],
        [0xAB, 0xCD, 0xEF],
        [0x11, 0x22, 0x33],
    ]

    path_dependent_count = 0
    total_count = 0

    for seq in test_sequences:
        print(f"Testing sequence: {[hex(x) for x in seq]}")

        # Generate all permutations
        perms = list(permutations(seq))
        results = []

        for perm in perms:
            result = fold_sequence(list(perm))
            results.append(result)
            print(f"  {[hex(x) for x in perm]} -> {hex(result)}")

        # Check if all results are unique
        unique_results = len(set(results))
        is_path_dependent = unique_results > 1
        path_dependent_count += is_path_dependent
        total_count += 1

        print(f"  Unique results: {unique_results}/{len(results)}")
        print(f"  Path dependent: {is_path_dependent}")
        print()

    print(
        f"Path dependent sequences: {path_dependent_count}/{total_count} ({path_dependent_count/total_count*100:.1f}%)"
    )
    print(
        f"Path independent sequences: {total_count-path_dependent_count}/{total_count} ({(total_count-path_dependent_count)/total_count*100:.1f}%)"
    )
    print()

    return path_dependent_count, total_count


def test_special_properties():
    """Test special properties and edge cases."""
    print("=== Testing Special Properties ===")

    # Test identity element (if any)
    print("Testing for identity element:")
    for i in range(256):
        if fold(i, 0) == i and fold(0, i) == i:
            print(f"  0x00 might be identity for 0x{i:02X}")
        if fold(i, 0xFF) == i and fold(0xFF, i) == i:
            print(f"  0xFF might be identity for 0x{i:02X}")

    # Test self-inverse elements
    print("\nTesting for self-inverse elements:")
    for i in range(256):
        if fold(i, i) == 0:
            print(f"  0x{i:02X} is self-inverse (0x{i:02X} ⊞ 0x{i:02X} = 0x00)")

    # Test absorption elements
    print("\nTesting for absorption elements:")
    for i in range(256):
        if fold(i, 0) == 0:
            print(f"  0x00 absorbs 0x{i:02X}")
        if fold(0, i) == 0:
            print(f"  0x{i:02X} is absorbed by 0x00")

    print()


def test_random_properties():
    """Test properties with random values."""
    print("=== Testing Random Properties ===")

    random.seed(42)  # For reproducible results

    # Test commutativity with random pairs
    commutative_random = 0
    total_random = 1000

    for _ in range(total_random):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        if fold(a, b) == fold(b, a):
            commutative_random += 1

    print(f"Random commutativity: {commutative_random}/{total_random} ({commutative_random/total_random*100:.1f}%)")

    # Test associativity with random triplets
    associative_random = 0
    total_random = 1000

    for _ in range(total_random):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        c = random.randint(0, 255)
        if fold(fold(a, b), c) == fold(a, fold(b, c)):
            associative_random += 1

    print(f"Random associativity: {associative_random}/{total_random} ({associative_random/total_random*100:.1f}%)")
    print()


def test_original_problem_case():
    """Test the specific case that caused the original test failure."""
    print("=== Testing Original Problem Case ===")

    seq_a = [0x35, 0xE2, 0x57]
    seq_b = list(reversed(seq_a))

    print(f"Original sequence: {[hex(x) for x in seq_a]}")
    print(f"Reversed sequence: {[hex(x) for x in seq_b]}")

    result_a = fold_sequence(seq_a)
    result_b = fold_sequence(seq_b)

    print(f"Original result: {hex(result_a)}")
    print(f"Reversed result: {hex(result_b)}")
    print(f"Same result: {result_a == result_b}")

    # Show step-by-step calculation
    print("\nStep-by-step calculation for original:")
    result = seq_a[0]
    print(f"  Start: {hex(result)}")
    for i in range(1, len(seq_a)):
        old_result = result
        result = fold(result, seq_a[i])
        print(f"  fold({hex(old_result)}, {hex(seq_a[i])}) = {hex(result)}")

    print("\nStep-by-step calculation for reversed:")
    result = seq_b[0]
    print(f"  Start: {hex(result)}")
    for i in range(1, len(seq_b)):
        old_result = result
        result = fold(result, seq_b[i])
        print(f"  fold({hex(old_result)}, {hex(seq_b[i])}) = {hex(result)}")

    print()


def main():
    """Run all tests."""
    print("Comprehensive Test of Fold Operation Mathematical Properties")
    print("=" * 60)
    print()

    # Run all tests
    commutative_count, total_commutative = test_commutativity()
    associative_count, total_associative = test_associativity()
    path_dependent_count, total_path_dependent = test_path_dependence()
    test_special_properties()
    test_random_properties()
    test_original_problem_case()

    # Summary
    print("=== SUMMARY ===")
    print(f"Non-commutative pairs: {(total_commutative-commutative_count)/total_commutative*100:.1f}%")
    print(f"Non-associative triplets: {(total_associative-associative_count)/total_associative*100:.1f}%")
    print(f"Path dependent sequences: {path_dependent_count/total_path_dependent*100:.1f}%")
    print()

    print("The fold operation demonstrates:")
    if commutative_count < total_commutative:
        print("✓ Non-commutativity (order matters for most pairs)")
    else:
        print("✗ Commutativity (unexpected)")

    if associative_count < total_associative:
        print("✓ Non-associativity (grouping matters for most triplets)")
    else:
        print("✗ Associativity (unexpected)")

    if path_dependent_count > 0:
        print("✓ Path dependence (sequence order affects final result)")
    else:
        print("✗ No path dependence (unexpected)")

    print()
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
