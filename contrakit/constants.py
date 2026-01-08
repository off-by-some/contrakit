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
EPS_SQRT = 1e-15  # For square root operations (EPS^0.5)
NORMALIZATION_TOL = 1e-10
FRAME_INDEPENDENCE_TOL = 1e-9
ZERO_DETECTION_TOL = 5e-13

# Property testing defaults
DEFAULT_PROPERTY_TEST_TRIALS = 500
DEFAULT_PROPERTY_TEST_TOLERANCE = 1e-10

# Dimension ranges for property testing
MIN_TEST_DIMENSION = 2
MAX_TEST_DIMENSION = 8
MAX_OUTPUT_DIMENSION = 6
MAX_SECOND_DIMENSION = 4

# Logarithmic stability
LOG_STABILITY_EPS = 1e-300

# Probability clamping
PROBABILITY_CLAMP_MIN = 1e-12

# Property testing trial divisions
EXPENSIVE_TEST_TRIAL_DIVISOR = 3

# Stochastic matrix dimensions for data processing tests
MIN_STOCHASTIC_DIM = 2
MAX_STOCHASTIC_DIM = 6
