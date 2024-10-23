"""
This file contains aliases for common paths in the repository.
"""

from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[1].absolute()

# Absolute path to the `src` directory
src = root / 'src'

# Absolute path to the `data` directory
data = root / 'data'
