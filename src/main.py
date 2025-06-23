#!/usr/bin/env python3
"""
Main entry point for GyroSI Baby ML application.
"""
import sys
import os
from frontend.gyro_app import main

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

if __name__ == "__main__":
    main()
