"""Offline training for the shipped checkpoints. Not imported by the entry point."""

import sys
from pathlib import Path

# The package lives under src/; put it on the path so these scripts can run with
# `python -m training.<name>` from the repository root without installing.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
