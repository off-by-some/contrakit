"""
Constants used throughout the Mathematical Theory of Contradiction library.

This module contains default values and constants that are used consistently
across different parts of the library to ensure reproducible results and
maintain numerical stability.
"""

from pathlib import Path

# Repository paths
REPO_ROOT = Path(__file__).parent.parent  # Project root (above contrakit package)
FIGURES_DIR = REPO_ROOT / "figures"

# Ensure figures directory exists
FIGURES_DIR.mkdir(exist_ok=True)

# Default random seed for reproducible results
DEFAULT_SEED = 416

# Numerical stability constants
EPS = 1e-12
NORMALIZATION_TOL = 1e-10
