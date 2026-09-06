import os
import sys

# Ensure src/ is on sys.path so 'sparklm' package can be imported during tests.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
