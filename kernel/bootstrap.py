# kernel/bootstrap.py
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# Add the kernel directory to the path so our shims can be found
sys.path.insert(0, ROOT)
# Prepend parent of `kernel/` so `gpt_oss` package here wins import order.
sys.path.insert(0, os.path.dirname(ROOT))
