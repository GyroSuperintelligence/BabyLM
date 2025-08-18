# kernel/bootstrap.py
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# Add the kernel directory to the path 
sys.path.insert(0, ROOT)
# Prepend parent of `kernel/`
sys.path.insert(0, os.path.dirname(ROOT))
