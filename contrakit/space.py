"""
Observable Spaces for the Mathematical Theory of Contradiction

This module provides classes for defining the structure of measurement systems
and observational spaces. A space defines what can be measured (observables)
and what values each measurement can take.

Key Concepts:
- Observable: Something that can be measured (e.g., "Temperature", "Opinion")
- Alphabet: The set of possible values for an observable (e.g., ["Hot", "Warm", "Cold"])
- Assignment: A complete specification of values for all observables
- Context: A subset of observables measured together

These classes form the foundation for defining what observations are possible
and how they relate to each other in the theory of contradiction.
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Tuple, Sequence, Set, Union, Optional, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import cvxpy as cp
from .agreement import BhattacharyyaCoefficient


@lru_cache(maxsize=None)
def _assignments_cache(names: Tuple[str, ...], alph_tuple: Tuple[Tuple[Any, ...], ...]) -> List[Tuple[Any, ...]]:
    """Cache for expensive assignment generation."""
    return list(itertools.product(*alph_tuple))


@lru_cache(maxsize=None)
def _outcomes_cache(obs_tuple: Tuple[str, ...], alph_tuple: Tuple[Tuple[Any, ...], ...]) -> List[Tuple[Any, ...]]:
    """Cache for expensive outcome generation."""
    return list(itertools.product(*alph_tuple))
    


@dataclass(frozen=True)
class Space:
    """
    An observable space defining what can be measured and their possible values.

    A space represents the structure of a measurement system, specifying what
    variables (observables) exist and what values each can take. This forms
    the foundation for defining behaviors and analyzing consistency.

    Attributes:
        names: Tuple of observable names (e.g., "Temperature", "Pressure")
        alphabets: Dictionary mapping each observable name to its possible values

    Example:
        # Define a space for weather observations
        weather_space = Space(
            names=("Temperature", "Humidity", "Precipitation"),
            alphabets={
                "Temperature": ("Hot", "Warm", "Cold"),
                "Humidity": ("High", "Medium", "Low"),
                "Precipitation": ("Rain", "Snow", "None")
            }
        )

        # Or use the convenience method
        weather_space = Space.create(
            Temperature=["Hot", "Warm", "Cold"],
            Humidity=["High", "Medium", "Low"],
            Precipitation=["Rain", "Snow", "None"]
        )
    """
    names: Tuple[str, ...]
    alphabets: Dict[str, Tuple[Any, ...]]
    
    def __post_init__(self):
        if set(self.alphabets.keys()) != set(self.names):
            missing = set(self.names) - set(self.alphabets.keys())
            extra = set(self.alphabets.keys()) - set(self.names)
            if missing:
                raise ValueError(f"Missing alphabets for observables: {missing}")
            if extra:
                raise ValueError(f"Extra alphabets for unknown observables: {extra}")
        
        if any(len(self.alphabets[name]) == 0 for name in self.names):
            empty = [name for name in self.names if len(self.alphabets[name]) == 0]
            raise ValueError(f"Empty alphabets for observables: {empty}")
    
    @classmethod
    def create(cls, **observables: Sequence[Any]) -> Space:
        """Create space from observable_name=alphabet pairs.
        
        Example: Space.create(X=[0,1], Y=['a','b','c'])
        """
        names = tuple(observables.keys())
        alphabets = {k: tuple(v) for k, v in observables.items()}
        return cls(names, alphabets)
    
    @classmethod 
    def binary(cls, *names: str) -> Space:
        """Create space with binary observables."""
        return cls.create(**{name: [0, 1] for name in names})
    
    def __len__(self) -> int:
        """Number of observables."""
        return len(self.names)
    
    def __contains__(self, name) -> bool:
        """Check if observable exists."""
        if hasattr(name, 'name'):
            return name.name in self.names
        return name in self.names
    
    def __getitem__(self, name) -> Tuple[Any, ...]:
        """Get alphabet for observable."""
        if hasattr(name, 'name'):
            return self.alphabets[name.name]
        return self.alphabets[name]

    def __eq__(self, other: object) -> bool:
        """Structural equality: same names and alphabets."""
        return (isinstance(other, Space) and
                self.names == other.names and
                self.alphabets == other.alphabets)

    def __hash__(self) -> int:
        """Hash based on structural content."""
        # Convert alphabets dict to a hashable form
        alpha_items = tuple(sorted(self.alphabets.items()))
        return hash((self.names, alpha_items))

    def __or__(self, observables: Union[str, Sequence[str]]) -> Space:
        """Restrict to subset of observables: space | ['X', 'Y']"""
        if isinstance(observables, str):
            observables = [observables]

        # Preserve original space order, drop duplicates
        wanted = []
        want_set = set(observables)
        for name in self.names:
            if name in want_set and name not in wanted:
                wanted.append(name)

        if set(wanted) != want_set:
            unknown = want_set - set(self.names)
            raise ValueError(f"Unknown observables: {unknown}")

        return Space(tuple(wanted), {name: self.alphabets[name] for name in wanted})
    
    def __matmul__(self, other: Space) -> Space:
        """Tensor product: space1 @ space2"""
        if set(self.names) & set(other.names):
            overlap = set(self.names) & set(other.names)
            raise ValueError(f"Overlapping observables in tensor product: {overlap}")
        
        return Space(
            self.names + other.names,
            {**self.alphabets, **other.alphabets}
        )
    
    def assignments(self) -> Iterator[Tuple[Any, ...]]:
        """All possible global assignments (lazy generator)."""
        alph_values = [self.alphabets[name] for name in self.names]
        return itertools.product(*alph_values)
    
    def assignment_count(self) -> int:
        """Count of all possible assignments without materializing them."""
        return int(np.prod([len(self.alphabets[name]) for name in self.names]))

    def outcomes_for(self, observables: Sequence[str]) -> List[Tuple[Any, ...]]:
        """Outcomes for subset of observables."""
        obs_tuple = tuple(observables)
        alph_tuple = tuple(self.alphabets[name] for name in obs_tuple)
        return _outcomes_cache(obs_tuple, alph_tuple)

    def difference(self, other: "Space") -> Dict[str, Any]:
        """
        Compare this space with another and return the differences.

        Args:
            other: The other Space to compare against

        Returns:
            Dict with keys:
            - 'only_self': set of observables only in this space
            - 'only_other': set of observables only in other space
            - 'alphabet_diffs': dict of observable -> (self_alphabet, other_alphabet) for differing alphabets
        """
        result = {
            'only_self': set(),
            'only_other': set(),
            'alphabet_diffs': {}
        }

        self_names = set(self.names)
        other_names = set(other.names)

        # Find observables unique to each space
        result['only_self'] = self_names - other_names
        result['only_other'] = other_names - self_names

        # Find differing alphabets for common observables
        common_names = self_names & other_names
        for name in common_names:
            self_alpha = tuple(self.alphabets.get(name, []))
            other_alpha = tuple(other.alphabets.get(name, []))
            if self_alpha != other_alpha:
                result['alphabet_diffs'][name] = (self_alpha, other_alpha)

        return result

    def index_of(self, name: str) -> int:
        """Index of observable in names tuple."""
        return self.names.index(name)

    def rename(self, name_map: Dict[str, str]) -> Space:
        """Create space with renamed observables."""
        old_names = list(self.names)
        new_names = [name_map.get(n, n) for n in old_names]
        if len(set(new_names)) != len(new_names):
            raise ValueError("name_map must be a bijection (duplicates found).")
        # Move alphabets under new names
        new_alphabets = {name_map.get(n, n): self.alphabets[n] for n in old_names}
        return Space(tuple(new_names), new_alphabets)
