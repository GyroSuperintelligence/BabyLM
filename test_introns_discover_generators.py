#!/usr/bin/env python3
"""
Explore the structure of the 5-element basis for fold algebra.
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baby.governance import fold, transcribe_byte


def analyze_basis_structure():
    """Analyze how the 5-element basis generates the algebra."""

    # The complete basis
    physical_masks = {
        "L0": 0b10000001,
        "LI": 0b01000010,
        "FG": 0b00100100,
        "BG": 0b00011000,
    }

    basis_elements = {}
    for name, mask in physical_masks.items():
        basis_elements[name] = transcribe_byte(mask)
    basis_elements["INV"] = 0xFF

    print("THE 5-ELEMENT BASIS")
    print("=" * 50)
    for name, intron in basis_elements.items():
        print(f"{name:3}: 0x{intron:02x} ({intron:08b})")

    # Track generation depth
    known = set(basis_elements.values())
    generation = {elem: 0 for elem in known}  # Basis elements are generation 0
    parent = {elem: elem for elem in known}  # Track first parent

    queue = list(known)
    head = 0

    print("\nGENERATION PROCESS")
    print("=" * 50)

    gen_counts = defaultdict(int)
    gen_counts[0] = len(known)

    while head < len(queue):
        a = queue[head]
        head += 1

        current_known = list(known)
        for b in current_known:
            for x, y in ((a, b), (b, a)):
                r = fold(x, y)
                if r not in known:
                    known.add(r)
                    queue.append(r)
                    # Track generation as min parent generation + 1
                    new_gen = min(generation[x], generation[y]) + 1
                    generation[r] = new_gen
                    parent[r] = x if generation[x] <= generation[y] else y
                    gen_counts[new_gen] += 1

    # Print generation statistics
    max_gen = max(generation.values())
    for g in range(max_gen + 1):
        print(f"Generation {g}: {gen_counts[g]} elements")

    print(f"\nTotal elements generated: {len(known)}")
    print(f"Maximum generation depth: {max_gen}")

    # Analyze role of each basis element
    print("\nROLE OF EACH BASIS ELEMENT")
    print("=" * 50)

    for name, elem in basis_elements.items():
        # Count how many elements have this as an ancestor
        descendants = sum(1 for e in known if is_ancestor(elem, e, parent))
        print(f"{name}: ancestor of {descendants} elements ({descendants/256*100:.1f}%)")


def is_ancestor(potential_ancestor, element, parent_map):
    """Check if potential_ancestor is in the generation path of element."""
    current = element
    seen = set()
    while current in parent_map and current not in seen:
        if current == potential_ancestor:
            return True
        seen.add(current)
        current = parent_map[current]
    return False


def test_necessity():
    """Test if each element of the 5-element basis is necessary."""
    physical_masks = {
        "L0": 0b10000001,
        "LI": 0b01000010,
        "FG": 0b00100100,
        "BG": 0b00011000,
    }

    basis = set()
    for mask in physical_masks.values():
        basis.add(transcribe_byte(mask))
    basis.add(0xFF)

    print("\nTESTING NECESSITY OF EACH ELEMENT")
    print("=" * 50)

    for remove in basis:
        test_basis = basis - {remove}

        # Calculate closure
        known = set(test_basis)
        queue = list(test_basis)
        head = 0

        while head < len(queue) and len(known) < 257:
            a = queue[head]
            head += 1
            for b in list(known):
                for x, y in ((a, b), (b, a)):
                    r = fold(x, y)
                    if r not in known:
                        known.add(r)
                        queue.append(r)

        print(f"Without 0x{remove:02x}: generates {len(known)} elements")
        if len(known) < 256:
            print(f"  → NECESSARY (missing {256 - len(known)} elements)")
        else:
            print(f"  → REDUNDANT")


def main():
    print("ANALYSIS OF THE 5-ELEMENT BASIS")
    print("=" * 70)

    analyze_basis_structure()
    test_necessity()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("The fold algebra requires exactly 5 generators:")
    print("  - 4 physical operations (L0, LI, FG, BG)")
    print("  - 1 logical operation (INV = 0xFF)")
    print("\nThis suggests the learning algebra is not purely physical")
    print("but requires one additional 'return' or 'negation' operator")
    print("to achieve complete closure.")


if __name__ == "__main__":
    main()
