#!/usr/bin/env python3
"""
Extract real physics masks from governance.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'baby-legacy'))

import numpy as np
from governance import XFORM_MASK, INTRON_BROADCAST_MASKS

print("Real XFORM_MASK values:")
print("XFORM_MASK = np.array([")
for i in range(0, 256, 16):
    values = [str(int(XFORM_MASK[j])) for j in range(i, min(i+16, 256))]
    print(f"    {', '.join(values)},  # {i}-{min(i+15, 255)}")
print("], dtype=np.uint64)")

print("\nReal INTRON_BROADCAST_MASKS values:")
print("INTRON_BROADCAST_MASKS = np.array([")
for i in range(0, 256, 16):
    values = [str(int(INTRON_BROADCAST_MASKS[j])) for j in range(i, min(i+16, 256))]
    print(f"    {', '.join(values)},  # {i}-{min(i+15, 255)}")
print("], dtype=np.uint64)")
