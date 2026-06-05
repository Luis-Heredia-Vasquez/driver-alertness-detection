"""Project-level pytest configuration to make `src` importable.

This file ensures the project root is on `sys.path` so tests can `import src`.
It is lightweight and has no runtime side-effects outside of test sessions.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Ensure the project root is on sys.path so `import src` will work
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
