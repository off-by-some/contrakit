"""
Observational Contexts for Multi-Perspective Measurements

This module defines observational contexts, which represent specific subsets of
observables that are measured together. Contexts are fundamental to the theory
of contradiction because they define which variables are jointly observed,
allowing us to analyze consistency across different measurement perspectives.

Key Concepts:
- Context: A subset of observables measured simultaneously
- Outcome: A specific combination of values for the observables in a context
- Restriction: How a complete assignment maps to a context-specific outcome

Contexts form the building blocks for defining behaviors and analyzing
whether different measurement perspectives are logically consistent.
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Tuple, Sequence, Set, Union, Optional
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import cvxpy as cp
from .agreement import BhattacharyyaCoefficient
from .space import Space

@dataclass(frozen=True, eq=False)
class Context:
    """
    A measurement context defining which observables are measured together.

    A context represents a specific subset of observables from the overall space
    that are measured simultaneously. This allows us to model different experimental
    or observational setups where only certain variables are accessible together.

    Attributes:
        space: The complete observable space this context belongs to
        observables: Tuple of observable names included in this context

    Example:
        # Define a context for measuring temperature and pressure together
        weather_space = Space.create(Temperature=["Hot", "Warm", "Cold"],
                                   Pressure=["High", "Low"],
                                   Humidity=["Wet", "Dry"])

        # Context for temperature-pressure measurements
        temp_pressure_context = Context(weather_space, ("Temperature", "Pressure"))
    """
    space: Space
    observables: Tuple[str, ...]
    _indices: Tuple[int, ...] = field(init=False, repr=False, compare=False)
    
    def __post_init__(self):
        unknown = set(self.observables) - set(self.space.names)
        if unknown:
            raise ValueError(f"Unknown observables in context: {unknown}")
        if len(set(self.observables)) != len(self.observables):
            raise ValueError("Context observables must be unique")

        # Precompute indices for fast restrict_assignment
        indices = tuple(self.space.index_of(name) for name in self.observables)
        object.__setattr__(self, '_indices', indices)

    def __hash__(self) -> int:
        # Space is frozen and eq-able, so it's hashable. This makes contexts
        # from equal-but-distinct Space instances compare the way you want.
        return hash((self.space, self.observables))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Context) and
                self.space == other.space and
                self.observables == other.observables)

    def __repr__(self) -> str:
        """String representation for debugging."""
        obs_str = ', '.join(repr(obs) for obs in self.observables)
        return f"Context(space={len(self.space)} obs, observables=({obs_str}))"

    @classmethod
    def make(cls, space: Space, observables: Union[str, Sequence[str]]) -> Context:
        """Create context from space and observable names."""
        if isinstance(observables, str):
            observables = [observables]
        return cls(space, tuple(observables))
    
    def __len__(self) -> int:
        """Number of observables in context."""
        return len(self.observables)
    
    def __contains__(self, name) -> bool:
        """Check if observable is measured in this context."""
        if hasattr(name, 'name'):
            return name.name in self.observables
        return name in self.observables
    
    def __or__(self, observables: Union[str, Sequence[str]]) -> Context:
        """Extend context: ctx | 'X' or ctx | ['X', 'Y']"""
        if isinstance(observables, str):
            observables = [observables]

        result = list(self.observables)
        for obs in observables:
            if obs not in result:
                result.append(obs)

        return Context.make(self.space, result)

    def __and__(self, other: Context) -> Context:
        """Intersect contexts: ctx1 & ctx2"""
        if self.space != other.space:
            raise ValueError("Contexts must share the same space")

        other_set = set(other.observables)
        result = [obs for obs in self.observables if obs in other_set]

        return Context.make(self.space, result)
    
    def outcomes(self) -> List[Tuple[Any, ...]]:
        """All possible outcomes in this context."""
        return self.space.outcomes_for(self.observables)
    
    def restrict_assignment(self, assignment: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Restrict global assignment to this context."""
        return tuple(assignment[i] for i in self._indices)

