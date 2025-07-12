#!/usr/bin/env python3
"""
Test script to verify the three proposed optimizations produce identical results.
"""

import numpy as np
import json
import sys
import os

# Add the baby module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baby'))

# Import the json functions from baby.information
from baby.information import json_loads, json_dumps

print("=== TESTING THREE PROPOSED OPTIMIZATIONS ===")

# Test 1: _L0, _LI, _FG, _BG optimization
print("\n1. Testing _L0, _LI, _FG, _BG optimization...")

# Current implementation
_L0_current = [0, 7]  # Identity
_LI_current = [1, 6]  # Inverse
_FG_current = [2, 5]  # Forward Gyration
_BG_current = [3, 4]  # Backward Gyration

# Proposed optimization
ops = np.array([[0,7],[1,6],[2,5],[3,4]], np.int8)
_L0_optimized, _LI_optimized, _FG_optimized, _BG_optimized = ops

print(f"Current _L0: {_L0_current}")
print(f"Optimized _L0: {_L0_optimized.tolist()}")
print(f"Current _LI: {_LI_current}")
print(f"Optimized _LI: {_LI_optimized.tolist()}")
print(f"Current _FG: {_FG_current}")
print(f"Optimized _FG: {_FG_optimized.tolist()}")
print(f"Current _BG: {_BG_current}")
print(f"Optimized _BG: {_BG_optimized.tolist()}")

# Test functionality
test_bit = 1
print(f"\nTesting bit {test_bit}:")
print(f"Current: {test_bit in _LI_current}")
print(f"Optimized: {test_bit in _LI_optimized}")
print(f"Results match: {test_bit in _LI_current == test_bit in _LI_optimized}")

# Test 2: update_registry JSON sanitize optimization
print("\n2. Testing update_registry JSON sanitize optimization...")

# Simulate the current implementation
def current_json_sanitize(f_content):
    try:
        registry = json_loads(f_content)
        if not isinstance(registry, dict):
            registry = {"count": 0, "uuids": []}
    except (json.JSONDecodeError, ValueError):
        registry = {"count": 0, "uuids": []}
    return registry

# Simulate the proposed optimization
def optimized_json_sanitize(f_content):
    registry = json_loads(f_content) if f_content else {"count":0,"uuids":[]}
    return registry

# Test cases
test_cases = [
    '{"count": 2, "uuids": ["a", "b"]}',  # Valid JSON
    '{"count": 2}',  # Missing uuids
    'invalid json',  # Invalid JSON
    '',  # Empty string
]

for i, test_case in enumerate(test_cases):
    try:
        current_result = current_json_sanitize(test_case)
        optimized_result = optimized_json_sanitize(test_case)
        print(f"Case {i+1}: {current_result == optimized_result}")
    except Exception as e:
        print(f"Case {i+1}: Error - {e}")

# Test 3: gene_keys matches counting optimization
print("\n3. Testing gene_keys matches counting optimization...")

# Current implementation
def current_count_matches(historical_patterns, recent_patterns):
    matches = 0
    for i in range(len(historical_patterns)):
        if historical_patterns[i] == recent_patterns[i]:
            matches += 1
    return matches

# Proposed optimization
def optimized_count_matches(historical_patterns, recent_patterns):
    if len(recent_patterns) < len(historical_patterns):
        return 0
    matches = sum(h==r for h,r in zip(historical_patterns, recent_patterns[-len(historical_patterns):]))
    return matches

# Test cases
test_cases = [
    ([1, 2, 3], [1, 2, 3]),  # Perfect match
    ([1, 2, 3], [4, 5, 6]),  # No match
    ([1, 2, 3], [1, 5, 3]),  # Partial match
    ([1, 2], [1, 2, 3, 4]),  # Historical shorter
    ([1, 2, 3, 4], [1, 2]),  # Recent shorter
    ([], [1, 2, 3]),  # Empty historical
    ([1, 2, 3], []),  # Empty recent
]

for i, (historical, recent) in enumerate(test_cases):
    try:
        current_result = current_count_matches(historical, recent)
        optimized_result = optimized_count_matches(historical, recent)
        print(f"Case {i+1}: {current_result == optimized_result} (current: {current_result}, optimized: {optimized_result})")
    except Exception as e:
        print(f"Case {i+1}: Error - {e}")

print("\n=== LINE COUNT COMPARISON ===")
print("1. _L0, _LI, _FG, _BG: 6 lines → 1 line (83% reduction)")
print("2. update_registry JSON: 7 lines → 1 line (86% reduction)")
print("3. gene_keys matches: ~5 lines → 1 line (80% reduction)")
print("Total: ~18 lines → 3 lines (83% reduction)") 