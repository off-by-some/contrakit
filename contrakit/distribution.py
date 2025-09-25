
"""
Probability Distributions for Observational Contexts

This module defines probability distributions over observational outcomes.
Distributions represent the likelihood of different measurement results
within a specific observational context, forming the basic building blocks
for behavioral patterns.

Key Concepts:
- Outcome: A specific combination of values for observables in a context
- Probability Distribution: Assignment of probabilities to possible outcomes
- Normalization: Probabilities must sum to 1.0
- Support: The set of outcomes with non-zero probability

Distributions are fundamental to defining what we observe in different
measurement contexts and form the basis for analyzing consistency.
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Tuple, Sequence, Set, Union, Optional
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np

# Import constants for numerical stability
from .constants import EPS, NORMALIZATION_TOL


@dataclass(frozen=True)
class Distribution:
    """
    A probability distribution over possible observational outcomes.

    This class represents the probability of observing different combinations
    of values for a set of observables within a measurement context. It ensures
    that probabilities are properly normalized and provides efficient access
    to probability values for specific outcomes.

    Attributes:
        outcomes: Tuple of possible outcome combinations (each is a tuple of values)
        probs: Tuple of probabilities corresponding to each outcome

    Example:
        # Distribution for coin flips
        coin_distribution = Distribution(
            outcomes=(("Heads",), ("Tails",)),
            probs=(0.6, 0.4)  # Slightly biased coin
        )

        # Distribution for weather observations
        weather_distribution = Distribution(
            outcomes=(("Sunny", "Warm"), ("Cloudy", "Cool"), ("Rainy", "Cold")),
            probs=(0.5, 0.3, 0.2)
        )
    """
    outcomes: Tuple[Tuple[Any, ...], ...]
    probs: Tuple[float, ...]
    _index: Dict[Tuple[Any, ...], int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if len(self.outcomes) != len(self.probs):
            raise ValueError("Outcomes and probabilities must have same length")

        total_prob = sum(self.probs)
        if not np.isclose(total_prob, 1.0, atol=NORMALIZATION_TOL):
            raise ValueError(f"Probabilities must sum to 1, got {total_prob}")

        if any(p < -EPS for p in self.probs):
            raise ValueError("Probabilities must be non-negative")

        # Build fast index for O(1) lookups
        object.__setattr__(self, '_index', {outcome: i for i, outcome in enumerate(self.outcomes)})
    
    @classmethod
    def from_dict(cls, pmf: Dict[Tuple[Any, ...], float]) -> Distribution:
        """Create from outcome -> probability mapping."""
        outcomes = tuple(pmf.keys())
        probs = tuple(pmf.values())
        return cls(outcomes, probs)
    
    @classmethod
    def uniform(cls, outcomes: Sequence[Tuple[Any, ...]]) -> Distribution:
        """Uniform distribution over outcomes."""
        n = len(outcomes)
        if n == 0:
            raise ValueError("Cannot create uniform distribution over empty outcomes")
        return cls(tuple(outcomes), tuple(1/n for _ in range(n)))

    @classmethod
    def random(cls, outcomes: Sequence[Tuple[Any, ...]], alpha: float = 1.0, rng: Optional[np.random.Generator] = None) -> Distribution:
        """Random distribution using Dirichlet sampling."""
        if rng is None:
            rng = np.random.default_rng()
        k = len(outcomes)
        if k == 0:
            raise ValueError("Cannot create random distribution over empty outcomes")
        probs = rng.dirichlet(np.full(k, float(alpha)))
        return cls(tuple(outcomes), tuple(map(float, probs)))

    def __getitem__(self, outcome: Tuple[Any, ...]) -> float:
        """Get probability of outcome."""
        idx = self._index.get(outcome)
        return self.probs[idx] if idx is not None else 0.0
    
    def __add__(self, other: Distribution) -> Distribution:
        """Convex combination with equal weights: (p + q)/2"""
        return self.mix(other, 0.5)
    
    def _ordered_union_outcomes(self, other: Distribution) -> List[Tuple[Any, ...]]:
        """Get union of outcomes in stable order."""
        seen = set()
        result = []
        for outcome in self.outcomes + other.outcomes:
            if outcome not in seen:
                seen.add(outcome)
                result.append(outcome)
        return result

    def mix(self, other: Distribution, weight: float) -> Distribution:
        """Convex combination: (1-weight)*self + weight*other"""
        if not (0 <= weight <= 1):
            raise ValueError("Weight must be in [0,1]")

        all_outcomes = self._ordered_union_outcomes(other)
        new_probs = {}

        for outcome in all_outcomes:
            p1 = self[outcome]
            p2 = other[outcome]
            new_probs[outcome] = (1 - weight) * p1 + weight * p2

        # Preserve the deterministic order we just built
        outcomes = tuple(all_outcomes)
        probs = tuple(new_probs[outcome] for outcome in outcomes)
        return Distribution(outcomes, probs)
    
    def to_dict(self) -> Dict[Tuple[Any, ...], float]:
        """Convert to outcome -> probability mapping."""
        return dict(zip(self.outcomes, self.probs))

    def to_array(self) -> np.ndarray:
        """Convert to numpy array of probabilities."""
        return np.array(self.probs)

    def l1_distance(self, other: Distribution) -> float:
        """Compute L1 distance between two distributions."""
        if self.outcomes == other.outcomes:
            # Fast path: same ordering
            return float(np.sum(np.abs(np.array(self.probs) - np.array(other.probs))))
        else:
            # Align outcomes in stable order
            keys = self._ordered_union_outcomes(other)
            p = np.array([self[k] for k in keys])
            q = np.array([other[k] for k in keys])
            return float(np.sum(np.abs(p - q)))
