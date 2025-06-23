#!/usr/bin/env python3
"""
Development script for GyroSI Baby ML.
"""
import sys
import os
from frontend.gyro_app import main

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if __name__ == "__main__":
    main()
