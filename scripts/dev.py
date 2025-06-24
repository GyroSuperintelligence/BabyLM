#!/usr/bin/env python3
"""
Development entry point for GyroSI Baby LM CLI.
"""
import sys
import os
from gyro_tools.gyro_cli import main

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if __name__ == "__main__":
    print("[dev] Launching CLI...")
    main()
