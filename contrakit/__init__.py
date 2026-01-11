"""
Mathematical Theory of Contradiction - A Python package for exploring contradiction mathematics.

Core Assumptions:
- Finite alphabets: All observable outcomes and context sets must be finite and discrete
- Frame-independence properties: FI set must be nonempty, compact, convex, and product-closed
- Asymptotic regime: Operational theorems require large sample limits for convergence guarantees
- Source-channel separation: Some communication results assume separable source and channel coding

Scope & Limitations:
- Works with discrete probability distributions only (no continuous variables)
- Complexity scales exponentially with outcome space size
- Results are domain-general once the frame-independent baseline is specified
- Finite-blocklength refinements remain an open research direction
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("contrakit")
except importlib.metadata.PackageNotFoundError:
    __version__ = "1.0.1-dev"

# Core classes
from .space import Space
from .context import Context
from .distribution import Distribution
from .behavior.behavior import Behavior

# High-level API
from .observatory import (
    Observatory, LensScope, ConceptHandle, ValueHandle,
    NoConceptsDefinedError, EmptyBehaviorError
)

# Frame independence
from .frame import FIResult, FrameIndependence

# Agreement measures
from .agreement import (
    AgreementMeasure,
    BhattacharyyaCoefficient,
    LinearOverlap,
    HellingerAffinity
)

# Contradiction measures
from .contradiction import (
    ContradictionMeasure,
    LogarithmicContradiction,
    LinearContradiction,
    QuadraticContradiction
)

# Convex optimization models
from .convex_models import (
    # New interface classes
    Solution,
    Context,
    Solver,
    AlphaStar,
    VarianceMinimizer,
    KLDivergenceMinimizer,
    ConditionalSolver,
    extract_lambdas_from_weights
)

# Lens utilities
from .lens import lens_space, as_lens_context

# Everything for convenience
__all__ = [
    # Core
    'Space', 'Context', 'Distribution', 'Behavior',
    # High-level API
    'Observatory', 'LensScope', 'ConceptHandle', 'ValueHandle',
    'NoConceptsDefinedError', 'EmptyBehaviorError',
    # Frame independence
    'FIResult', 'FrameIndependence',
    # Agreement measures
    'AgreementMeasure', 'BhattacharyyaCoefficient', 'LinearOverlap', 'HellingerAffinity',
    # Contradiction measures
    'ContradictionMeasure', 'LogarithmicContradiction', 'LinearContradiction', 'QuadraticContradiction',
    # Convex models - new interface
    'Solution', 'Context', 'Solver', 'AlphaStar', 'VarianceMinimizer',
    'KLDivergenceMinimizer', 'ConditionalSolver', 'extract_lambdas_from_weights',
    # Lens utilities
    'lens_space', 'as_lens_context'
]
