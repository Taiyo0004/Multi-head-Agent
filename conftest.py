"""Make the package importable without installing it.

Keeps ``pytest`` working from any working directory and on any OS, so
``pytest``, ``python -m pytest`` and an IDE's runner all behave the same
without a ``pip install -e .`` step first.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
