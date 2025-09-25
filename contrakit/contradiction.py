"""
Contradiction Measures for Quantifying Inconsistency

This module provides classes for computing contradiction measures, which quantify
the degree of logical inconsistency in behavioral patterns. Contradiction measures
extend agreement measures by considering entire behavioral patterns rather than
just pairs of distributions.

Key Concepts:
- Contradiction Measure: A function K(P) that quantifies overall inconsistency in a behavior
- Logarithmic Measures: Measure contradiction in bits (base 2) or nats (base e)
- Linear Measures: Simple linear combinations of agreement values
- Quadratic Measures: Quadratic combinations for different sensitivity

These measures provide the final quantitative assessment of how contradictory
a behavioral pattern is, forming the core output of the mathematical theory.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Assuming these are your core classes (imported from your core module)
from .behavior.behavior import Behavior
from .space import Space
from .context import Context
from .distribution import Distribution
from .frame import FrameIndependence
from .agreement import AgreementMeasure, BhattacharyyaCoefficient


class ContradictionMeasure(ABC):
    """
    Abstract base class for measuring contradiction in behavioral patterns.

    Contradiction measures quantify how inconsistent a behavioral pattern is
    by analyzing how well different observational contexts agree with each other.
    They provide a single number that summarizes the overall level of contradiction
    in a complex set of observations.

    Attributes:
        agreement_measure: The underlying agreement measure used for pairwise comparisons

    Subclasses must implement:
    - The contradiction computation (__call__)
    - A descriptive name for the measure

    Example:
        # Create a logarithmic contradiction measure
        measure = LogarithmicContradiction(base=2.0)  # Measures in bits

        # Compute contradiction for a behavior
        contradiction_level = measure(some_behavior)
    """

    def __init__(self, agreement_measure: Optional[AgreementMeasure] = None):
        """
        Initialize the contradiction measure.

        Args:
            agreement_measure: The agreement measure to use for pairwise comparisons.
                              Defaults to BhattacharyyaCoefficient if None provided.
        """
        self.agreement_measure = agreement_measure or BhattacharyyaCoefficient()

    @abstractmethod
    def __call__(self, behavior: 'Behavior') -> float:
        """
        Compute the contradiction measure for a behavioral pattern.

        Args:
            behavior: The behavioral pattern to analyze

        Returns:
            A numerical value quantifying the degree of contradiction
            (typically 0 for no contradiction, higher values for more contradiction)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this contradiction measure."""
        pass


class LogarithmicContradiction(ContradictionMeasure):
    """
    Logarithmic contradiction measure based on information theory.

    This measure computes contradiction as the negative logarithm of the maximum
    agreement coefficient. The base determines the units:
    - Base 2: Measures contradiction in bits (information theory standard)
    - Base e: Measures contradiction in nats (natural logarithm)
    - Other bases: Custom logarithmic units

    The logarithmic form has desirable properties for information-theoretic
    interpretations and provides an intuitive scale where small values indicate
    consistency and larger values indicate greater contradiction.

    Args:
        base: The logarithmic base (2.0 for bits, math.e for nats, etc.)
        agreement_measure: The underlying agreement measure (default: Bhattacharyya)

    Example:
        # Measure contradiction in bits
        measure_bits = LogarithmicContradiction(base=2.0)
        contradiction_bits = measure_bits(some_behavior)

        # Measure contradiction in nats
        measure_nats = LogarithmicContradiction(base=math.e)
        contradiction_nats = measure_nats(some_behavior)
    """
    
    def __init__(self, base: float = 2.0, agreement_measure: Optional[AgreementMeasure] = None):
        super().__init__(agreement_measure)
        self.base = base
    
    def __call__(self, behavior: 'Behavior') -> float:
        alpha_star = behavior.alpha_star
        return -np.log(max(alpha_star, 1e-300)) / np.log(self.base)
    
    @property
    def name(self) -> str:
        return f"K_log{self.base}"


class LinearContradiction(ContradictionMeasure):
    """K(P) = 1 - α*(P)"""
    
    def __call__(self, behavior: 'Behavior') -> float:
        return 1.0 - behavior.alpha_star
    
    @property
    def name(self) -> str:
        return "K_linear"


class QuadraticContradiction(ContradictionMeasure):
    """K(P) = (1 - α*(P))²"""
    
    def __call__(self, behavior: 'Behavior') -> float:
        alpha = behavior.alpha_star
        return (1.0 - alpha) ** 2
    
    @property
    def name(self) -> str:
        return "K_quadratic"
