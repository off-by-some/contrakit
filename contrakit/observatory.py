"""
Observatory API - High-Level Interface for Creating Behaviors

This module provides a fluent, user-friendly API for creating and managing
observational behaviors in the Mathematical Theory of Contradiction. It wraps
the core classes (Space, Context, Distribution, Behavior) with a more intuitive
interface for common use cases.

Key Components:
- Observatory: Main entry point that manages the overall observational system
- ConceptHandle: Typed handle for observable concepts with their value alphabets
- ValueHandle: Typed handle for concept symbols with syntactic sugar
- PerspectiveMap: Manages probability distributions across different measurement contexts
- LensScope: Creates lens-specific contexts for modeling different perspectives

Example Usage:
    # Create observatory with global alphabet
    observatory = Observatory.create(symbols=["Yes", "No"])

    # Define concepts - uses global alphabet by default
    voter = observatory.concept("Voter")  # Uses ["Yes", "No"]
    candidate = observatory.concept("Candidate", symbols=["Qualified", "Unqualified"])
    reviewer = observatory.concept("Reviewer", symbols=["Maybe", *observatory.alphabet])  # ["Maybe", "Yes", "No"]

    # Add distributions
    yes, no = voter.alphabet
    observatory.perspectives[voter] = {yes: 0.6, no: 0.4}

    # Create behavior object
    behavior = observatory.perspectives.to_behavior()

    # Use lenses for different perspectives
    with observatory.lens(reviewer) as lens_r:
        local_concept = lens_r.define("LocalConcept", symbols=["Value1", "Value2"])
        val1, val2 = local_concept.alphabet
        lens_r.perspectives[local_concept] = {val1: 0.7, val2: 0.3}
        behavior_r = lens_r.to_behavior()
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Sequence, Set, Union, Optional, Iterator, Protocol
from dataclasses import dataclass, field
import numpy as np

from .space import Space
from .context import Context
from .distribution import Distribution
from .behavior.behavior import Behavior
from .lens import lens_space, as_lens_context
from .constants import EPS, NORMALIZATION_TOL

class NoConceptsDefinedError(ValueError):
    """
    Raised when attempting to generate a behavior from a lens with no concepts defined.
    """
    pass


class EmptyBehaviorError(ValueError):
    """
    Raised when attempting to generate a behavior with no distributions defined.
    """
    pass


class Observable(Protocol):
    """
    Protocol for objects that can be observed in the framework.
    
    Both ConceptHandle and LensScope implement this protocol, enabling
    uniform treatment of concepts and lenses as observables.
    """
    
    @property
    def name(self) -> str:
        """Name of the observable."""
        ...
    
    @property
    def symbols(self) -> Tuple[Any, ...]:
        """Possible outcomes when observing this."""
        ...
    
    @property
    def alphabet(self) -> Tuple['ValueHandle', ...]:
        """ValueHandle objects for syntactic sugar."""
        ...
    
    @property
    def observatory(self) -> 'Observatory':
        """Reference to the parent Observatory."""
        ...



class ValueHandle:
    """
    Represents a single value in a concept's alphabet with syntactic sugar.

    Provides the & operator for creating tuples of symbols, making it easy to
    specify joint outcomes in distributions.

    Example:
        reviewer_a = observatory.concept("Reviewer_A", symbols=["Yes", "No"])
        yes, no = reviewer_a.alphabet
        joint_outcome = yes & no  # ("Yes", "No")
    """

    def __init__(self, value: Any, concept: 'ConceptHandle'):
        self.value = value
        self.concept = concept

    def __and__(self, other: Union['ValueHandle', Any]) -> Tuple[Any, ...]:
        """Create tuple using & operator."""
        if isinstance(other, ValueHandle):
            return (self.value, other.value)
        else:
            return (self.value, other)

    def __rand__(self, other: Any) -> Tuple[Any, ...]:
        """Handle case when & is used with non-ValueHandle on left."""
        return (other, self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"ValueHandle({self.value!r})"

    def __eq__(self, other):
        if isinstance(other, ValueHandle):
            return self.value == other.value and self.concept == other.concept
        return False

    def __hash__(self):
        return hash((self.value, self.concept))


class ConceptHandle:
    """
    Represents a measurable variable/dimension in the universe.

    Acts as a variable handle for constructing joint outcomes and provides
    access to its alphabet of possible symbols.

    Attributes:
        name: The name of the concept (e.g., "Reviewer_A")
        symbols: Tuple of possible symbols for this concept
        observatory: Reference to the parent Observatory
    """

    def __init__(self, name: str, symbols: Tuple[Any, ...], observatory: 'Observatory'):
        self.name = name
        self.symbols = symbols
        self.observatory = observatory
        self._value_handles = tuple(ValueHandle(v, self) for v in symbols)

    @property
    def alphabet(self) -> Tuple[ValueHandle, ...]:
        """Get tuple of ValueHandle objects for syntactic sugar."""
        return self._value_handles

    def __and__(self, other: Union['ConceptHandle', 'LensScope']) -> Tuple[Union['ConceptHandle', 'LensScope'], ...]:
        """Create joint observable tuple using & operator."""
        return (self, other)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"ConceptHandle({self.name!r})"

    def __eq__(self, other):
        if isinstance(other, ConceptHandle):
            return self.name == other.name and self.observatory is other.observatory
        return False

    def __hash__(self):
        return hash((self.name, id(self.observatory)))

@dataclass
class DistributionWrapper:
    """
    Wrapper for distribution access that provides the .distribution attribute.

    This wrapper allows the API to return an object with a .distribution property
    as specified in the original API design.
    """
    distribution: Distribution


@dataclass
class PerspectiveMap:
    """
    Manages probability distributions across different observational contexts.

    Provides a dictionary-like interface for setting and accessing distributions,
    with automatic validation and the ability to compute marginals from joint distributions.

    Attributes:
        _observatory: Reference to the parent Observatory
        _distributions: Dictionary mapping contexts to distributions
    """

    _observatory: Optional['Observatory'] = None
    _distributions: Dict[Tuple[str, ...], Distribution] = field(default_factory=dict)

    def __post_init__(self):
        # Space will be accessed lazily when needed
        pass

    @property
    def _get_space(self) -> Space:
        """Lazily get the space from the observatory."""
        if self._observatory and self._observatory._space:
            return self._observatory._space
        raise ValueError("No space available - define some concepts first")

    def __getitem__(self, key: Union[str, Observable, Tuple[Union[str, Observable], ...]]) -> DistributionWrapper:
        """Get distribution wrapper for a context."""
        if isinstance(key, (str, ConceptHandle, LensScope)):
            key = (key,)

        # Convert Observable objects to their names
        context_tuple = tuple(k.name if hasattr(k, 'name') else k for k in key)
        if context_tuple not in self._distributions:
            raise KeyError(f"No distribution defined for context {context_tuple}")

        return DistributionWrapper(distribution=self._distributions[context_tuple])

    def __setitem__(self, key: Union[str, Observable, Tuple[Union[str, Observable], ...]], value: Dict[Union[Tuple, str, ValueHandle], float]):
        """Set distribution for a context."""
        if isinstance(key, (str, ConceptHandle, LensScope)):
            key = (key,)

        # Convert Observable objects to their names
        context_tuple = tuple(k.name if hasattr(k, 'name') else k for k in key)
        context = Context(self._get_space, context_tuple)

        # Convert value dict to proper format
        pmf = {}
        for outcome_key, prob in value.items():
            if isinstance(outcome_key, ValueHandle):
                # Handle ValueHandle for marginal distributions
                outcome = (outcome_key.value,)
            elif isinstance(outcome_key, tuple):
                # Handle tuples (either raw symbols or ValueHandle tuples from & operator)
                outcome_parts = []
                for part in outcome_key:
                    if isinstance(part, ValueHandle):
                        outcome_parts.append(part.value)
                    else:
                        outcome_parts.append(part)
                outcome = tuple(outcome_parts)
            elif isinstance(outcome_key, str):
                outcome = (outcome_key,)
            else:
                outcome = (outcome_key,)
            pmf[outcome] = prob

        # Auto-complete marginal distributions if incomplete
        pmf = self._complete_distribution(context, pmf)

        distribution = Distribution.from_dict(pmf)
        self._distributions[context_tuple] = distribution

    def _complete_distribution(self, context: Context, pmf: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """Auto-complete a distribution if it's incomplete for a marginal context."""
        # Only auto-complete for single-observable contexts (marginals)
        if len(context.observables) != 1:
            return pmf

        observable = context.observables[0]
        possible_symbols = list(self._get_space[observable])

        # Check if we have all possible symbols
        current_symbols = set()
        for outcome in pmf.keys():
            if isinstance(outcome, tuple) and len(outcome) == 1:
                current_symbols.add(outcome[0])
            else:
                # Not a proper marginal outcome format, don't auto-complete
                return pmf

        missing_symbols = set(possible_symbols) - current_symbols
        if not missing_symbols:
            return pmf

        # Calculate remaining probability
        current_prob_sum = sum(pmf.values())
        remaining_prob = 1.0 - current_prob_sum

        if remaining_prob <= 0:
            return pmf

        # Distribute remaining probability equally among missing symbols
        prob_per_missing = remaining_prob / len(missing_symbols)

        # Create complete outcomes for missing symbols
        for symbol in missing_symbols:
            pmf[(symbol,)] = prob_per_missing

        return pmf

    def __contains__(self, key: Union[str, Observable, Tuple[Union[str, Observable], ...]]) -> bool:
        """Check if context has a distribution."""
        if isinstance(key, (str, ConceptHandle, LensScope)):
            key = (key,)
        context_tuple = tuple(k.name if hasattr(k, 'name') else k for k in key)
        return context_tuple in self._distributions

    def add_joint(self, *args):
        """
        Add a joint distribution for multiple observables.

        Args:
            *args: Observable names followed by distribution dict as last argument
        """
        if len(args) < 2:
            raise ValueError("add_joint requires at least one observable and a distribution")

        *observables, distribution = args
        context_tuple = tuple(observables)
        self[context_tuple] = distribution

    def validate(self, allow_empty: bool = False) -> bool:
        """
        Validate all distributions for normalization, consistency, and marginal compatibility.

        Args:
            allow_empty: If True, allow empty distribution sets.
                        If False (default), raise error for empty behaviors.

        Returns:
            True if all validations pass

        Raises:
            EmptyBehaviorError: If no distributions are defined and allow_empty=False
            ValueError: If any other validation fails
        """
        if not self._distributions:
            if not allow_empty:
                raise EmptyBehaviorError(
                    "Cannot generate behavior with no distributions defined. "
                    "Set allow_empty=True to create empty behaviors, or define distributions first."
                )

        # Check normalization
        for context_tuple, dist in self._distributions.items():
            total_prob = sum(dist.probs)
            if not np.isclose(total_prob, 1.0, atol=NORMALIZATION_TOL):
                raise ValueError(f"Distribution for context {context_tuple} not normalized: sum={total_prob}")

        # Check marginal consistency (simplified check)
        # This would need more sophisticated logic for full marginal consistency
        # For now, just ensure no negative probabilities
        for context_tuple, dist in self._distributions.items():
            if any(p < -EPS for p in dist.probs):
                raise ValueError(f"Negative probabilities in context {context_tuple}")

        return True

    @property
    def values(self) -> Dict[Tuple[str, ...], Dict[Tuple, float]]:
        """
        Get all distributions with computed marginals.

        Returns:
            Dictionary mapping contexts to their probability distributions
        """
        result = {}
        for context_tuple, dist in self._distributions.items():
            result[context_tuple] = dist.to_dict()
        return result

    def to_behavior(self, allow_empty: bool = False) -> Behavior:
        """
        Convert the perspective map to a Behavior object.

        Args:
            allow_empty: If True, allow creating behavior with no distributions.
                        If False (default), raise error for empty behaviors.

        Returns:
            Behavior object combining all distributions

        Raises:
            EmptyBehaviorError: If no distributions are defined and allow_empty=False
        """
        self.validate(allow_empty=allow_empty)

        # Convert context tuples to Context objects
        context_distributions = {}
        for context_tuple, dist in self._distributions.items():
            context = Context(self._get_space, context_tuple)
            context_distributions[context] = dist

        return Behavior(self._get_space, context_distributions)



class LensPerspectiveProxy:
    """
    Proxy that captures assignments to perspectives within a LensScope.

    Users assign distributions on BASE observables only. We store both:
    - base-space contexts (for Behavior over base space)
    - low-level lens-tagged contexts (for raw Behavior over lens space)
    """

    def __init__(self, scope: 'LensScope') -> None:
        self._scope = scope

    def __setitem__(self, key, value):
        # Normalize context key to tuple of names
        if isinstance(key, (str, ConceptHandle, LensScope)):
            key = (key,)
        ctx_base = tuple(k.name if hasattr(k, 'name') else k for k in key)

        # Validate base space exists and includes all observables
        if self._scope._observatory._space is None:
            raise NoConceptsDefinedError(
                "Cannot set perspectives before defining concepts. "
                "Call observatory.concept() first."
            )
        base_space = self._scope._observatory._space
        
        # Check for missing observables (now includes observable lenses in space)
        missing = [n for n in ctx_base if n not in base_space.names]
        if missing:
            if len(missing) == 1:
                obs_name = missing[0]
                raise ValueError(f"Unknown observable '{obs_name}'.\n"
                               f"Did you mean to call `obs.concept(\"{obs_name}\")` first?\n"
                               f"Or, if this is a lens, pass `symbols=...` when creating it.")
            else:
                raise ValueError(f"Unknown observables: {missing}.\n"
                               f"Did you mean to call `obs.concept(name)` first for each?\n"
                               f"Or, if these are lenses, pass `symbols=...` when creating them.")

        # Normalize PMF outcomes to tuples of symbols
        pmf_base: Dict[Tuple, float] = {}
        for outcome_key, prob in value.items():
            if isinstance(outcome_key, ValueHandle):
                outcome = (outcome_key.value,)
            elif isinstance(outcome_key, tuple):
                vals = []
                for v in outcome_key:
                    vals.append(v.value if isinstance(v, ValueHandle) else v)
                outcome = tuple(vals)
            else:
                outcome = (outcome_key,)
            pmf_base[outcome] = prob

        # Auto-complete marginal distributions for single-observable contexts
        pmf_base = self._scope._autocomplete_base(ctx_base, pmf_base)

        # Store base context
        self._scope._contexts_base[ctx_base] = pmf_base

        # Build low-level (tagged) context and outcomes
        axis_name = self._scope._axis_name
        ctx_ll = as_lens_context(ctx_base, axis_name)
        pmf_ll = {outcome + (0,): p for outcome, p in pmf_base.items()}
        self._scope._contexts_ll[ctx_ll] = pmf_ll

    def __getitem__(self, key):
        # Normalize context key to tuple of names
        if isinstance(key, (str, ConceptHandle, LensScope)):
            key = (key,)
        ctx_base = tuple(k.name if hasattr(k, 'name') else k for k in key)

        # Return stored context if it exists
        if ctx_base in self._scope._contexts_base:
            return self._scope._contexts_base[ctx_base]
        else:
            raise KeyError(f"No perspectives set for context: {ctx_base}")


@dataclass
class LensScope:
    """
    High-level lens scope that hides the lens tag by default.

    - with semantics preserved
    - to_behavior(): Behavior over BASE space (no lens tag)
    - to_behavior_raw(): Behavior over LENS space (explicit tag axis)
    
    Can also be observed as a concept when symbols are provided, enabling meta-lenses.
    """

    _observatory: 'Observatory'
    _name: str
    _symbols: Optional[Tuple[Any, ...]] = field(default=None)
    _axis_name: str = field(init=False)
    _contexts_base: Dict[Tuple[str, ...], Dict[Tuple, float]] = field(default_factory=dict)
    _contexts_ll: Dict[Tuple[str, ...], Dict[Tuple, float]] = field(default_factory=dict)
    _value_handles: Optional[Tuple['ValueHandle', ...]] = field(init=False, default=None)

    def __post_init__(self):
        # Axis name chosen to avoid collisions with user concepts
        self._axis_name = f"__lens__{self._name}"
        
        # If symbols provided, create value handles for observability
        if self._symbols is not None:
            self._value_handles = tuple(ValueHandle(v, self) for v in self._symbols)
        
        # Ensure lens axis exists lazily upon raw behavior creation

    def __enter__(self) -> 'LensScope':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # don't suppress

    def define(self, concept: str, symbols: Optional[Sequence[Any]] = None) -> ConceptHandle:
        return self._observatory.concept(concept, symbols)

    @property
    def perspectives(self) -> LensPerspectiveProxy:
        return LensPerspectiveProxy(self)

    @property
    def observable_name(self) -> str:
        """Get the observable name of this lens."""
        return self._axis_name

    @property
    def name(self) -> str:
        """Get the user-provided name of this lens."""
        return self._name
    
    @property
    def symbols(self) -> Tuple[Any, ...]:
        """Get symbols for observing this lens (Observable protocol)."""
        if self._symbols is None:
            raise ValueError(f"Lens '{self._name}' is not observable - no symbols defined. "
                           "Create with symbols parameter to make it observable.")
        return self._symbols
    
    @property
    def alphabet(self) -> Tuple['ValueHandle', ...]:
        """Get ValueHandle objects for syntactic sugar (Observable protocol)."""
        if self._value_handles is None:
            raise ValueError(f"Lens '{self._name}' is not observable - no symbols defined. "
                           "Create with symbols parameter to make it observable.")
        return self._value_handles
    
    @property
    def observatory(self) -> 'Observatory':
        """Get reference to parent Observatory (Observable protocol)."""
        return self._observatory
    
    def __and__(self, other: Union['LensScope', 'ConceptHandle']) -> Tuple[Union['LensScope', 'ConceptHandle'], ...]:
        """Create joint observable tuple using & operator."""
        return (self, other)
    
    def __str__(self):
        return self._name
    
    def __repr__(self):
        observable_status = "observable" if self._symbols is not None else "non-observable"
        return f"LensScope({self._name!r}, {observable_status})"
    
    def __eq__(self, other):
        if isinstance(other, LensScope):
            return self._name == other._name and self._observatory is other._observatory
        return False
    
    def __hash__(self):
        return hash((self._name, id(self._observatory)))

    def _ensure_lens_axis(self):
        """Ensure the observatory has a lens space including this scope's axis."""
        base_space = self._observatory._space
        if base_space is None:
            raise NoConceptsDefinedError(
                "Cannot create a lens space with no concepts defined. Call define() first."
            )

        # Initialize registry if needed
        if not hasattr(self._observatory, "_lens_axes"):
            self._observatory._lens_axes = set()

        # If axis already present, ensure lens_space is up-to-date and return
        if self._axis_name in getattr(self._observatory, "_lens_axes", set()):
            return

        # Rebuild lens space to include all known axes + this one
        axes = {name: [0] for name in getattr(self._observatory, "_lens_axes", set()) | {self._axis_name}}
        self._observatory._lens_space = lens_space(base_space, **axes)
        self._observatory._lens_axes = set(axes.keys())

    def to_behavior(self, allow_empty: bool = False) -> Behavior:
        # No concepts at all → explicit error regardless of allow_empty
        if self._observatory._space is None:
            raise NoConceptsDefinedError(
                "Cannot generate behavior: no concepts defined. Call define() first."
            )

        if not self._contexts_base:
            if not allow_empty:
                raise EmptyBehaviorError(
                    "Cannot generate behavior: no distributions defined. "
                    "Pass allow_empty=True to create an empty behavior."
                )
            # Empty behavior over base space
            return Behavior.from_contexts(self._observatory._space, {})

        return Behavior.from_contexts(self._observatory._space, dict(self._contexts_base))

    def to_behavior_raw(self, allow_empty: bool = False) -> Behavior:
        # Ensure lens axis is present in lens space
        self._ensure_lens_axis()

        if not self._contexts_ll:
            if not allow_empty:
                raise EmptyBehaviorError(
                    "Cannot generate behavior: no distributions defined. "
                    "Pass allow_empty=True to create an empty behavior."
                )
            return Behavior.from_contexts(self._observatory._lens_space, {})

        return Behavior.from_contexts(self._observatory._lens_space, dict(self._contexts_ll))

    def _autocomplete_base(self, ctx_base: Tuple[str, ...], pmf: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """Auto-complete and normalize distributions for contexts."""
        space = self._observatory._space

        if len(ctx_base) == 1:
            # Single-observable context: auto-complete missing outcomes
            observable = ctx_base[0]
            possible_symbols = list(space[observable])

            # Current observed symbols
            current_symbols = set(outcome[0] for outcome in pmf.keys())
            missing_symbols = [s for s in possible_symbols if s not in current_symbols]
            if not missing_symbols:
                return pmf

            remaining = 1.0 - sum(pmf.values())
            if remaining < -NORMALIZATION_TOL:
                # Let downstream normalization error surface; do not modify
                return pmf
            fill = remaining / len(missing_symbols) if missing_symbols else 0.0
            completed = dict(pmf)
            for s in missing_symbols:
                completed[(s,)] = completed.get((s,), 0.0) + fill
            return completed
        else:
            # Multi-observable context: normalize if needed
            total_prob = sum(pmf.values())
            if abs(total_prob - 1.0) < NORMALIZATION_TOL:
                return pmf
            elif total_prob > 0:
                # Normalize to sum to 1.0
                normalized = {outcome: prob / total_prob for outcome, prob in pmf.items()}
                return normalized
            else:
                # All probabilities are zero - let downstream error surface
                return pmf

    def contexts_low_level(self) -> Dict[Tuple[str, ...], Dict[Tuple, float]]:
        """Expose the low-level lens-tagged context map (advanced use)."""
        return dict(self._contexts_ll)

    def compose(self, other: 'LensScope') -> 'LensComposition':
        """Combine this lens with another to create a multi-perspective analysis."""
        return LensComposition((self, other))

    def intersection(self, other: 'LensScope') -> 'LensComposition':
        """Find consensus/agreement between this lens and another."""
        return LensComposition((self, other), _mode="intersection")

    def difference(self, other: 'LensScope') -> 'LensComposition':
        """Find what this lens contributes beyond the other lens."""
        return LensComposition((self, other), _mode="difference")

    def symmetric_difference(self, other: 'LensScope') -> 'LensComposition':
        """Find conflicts/disagreements between this lens and another."""
        return LensComposition((self, other), _mode="symmetric_difference")

    def __or__(self, other: 'LensScope') -> 'LensComposition':
        """Combine lenses using | operator: lens1 | lens2."""
        return self.compose(other)


@dataclass(frozen=True)
class LensComposition:
    """
    Immutable snapshot of multiple lenses for contradictory analysis.

    Created by combining LensScope objects using compose() or the | operator.
    Represents the state of lenses at the time of composition creation.
    """

    lenses: Tuple[LensScope, ...]
    _mode: str = "union"  # union, intersection, difference, symmetric_difference

    def __or__(self, other: 'LensScope') -> 'LensComposition':
        """Combine this composition with another lens using | operator."""
        return LensComposition(self.lenses + (other,), _mode=self._mode)

    def _map_context_to_lens_name(self, ctx_tuple: Tuple[str, ...]) -> Optional[str]:
        """Map context tuple to lens name using stored lens references."""
        for axis_name in ctx_tuple:
            for lens in self.lenses:
                if axis_name == lens._axis_name:
                    return lens.name
        return None

    @property
    def perspective_contributions(self) -> Dict[str, float]:
        """
        Get contribution of each lens to the contradiction in the composed behavior.

        Returns dict mapping lens names to contribution weights (0.0-1.0).
        Higher values indicate lenses more responsible for the disagreement.

        Note: This creates a new behavior each time - for repeated access,
        store the result of to_behavior() and use its properties directly.
        """
        behavior = self.to_behavior()

        # Map context tuples to lens names and aggregate contributions
        contributions = {}
        for ctx_tuple, weight in behavior.least_favorable_lambda().items():
            lens_name = self._map_context_to_lens_name(ctx_tuple)
            if lens_name:
                contributions[lens_name] = contributions.get(lens_name, 0.0) + weight

        return contributions

    @property
    def witness_distribution(self) -> Dict[str, float]:
        """Alias for perspective_contributions (for advanced users familiar with theory)."""
        return self.perspective_contributions

    def to_behavior(self) -> Behavior:
        """Create a behavior from the combined lenses based on the composition mode."""
        if not self.lenses:
            raise ValueError("No lenses to combine")

        # Ensure lens spaces are created for all lenses
        for lens in self.lenses:
            lens.to_behavior_raw(allow_empty=True)

        # Get the combined lens space from the observatory
        observatory = self.lenses[0]._observatory
        combined_lens_space = observatory._lens_space

        if self._mode == "union":
            # Combine all contexts (current behavior)
            combined_contexts = {}
            for lens in self.lenses:
                combined_contexts.update(lens.contexts_low_level())
            return Behavior.from_contexts(combined_lens_space, combined_contexts)

        elif self._mode == "intersection":
            # Find intersection: contexts that exist in both lenses
            lens_contexts = [lens.contexts_low_level() for lens in self.lenses]
            if len(lens_contexts) == 2:
                # For two lenses, find common context keys
                common_contexts = {}
                ctx_keys_1 = set(lens_contexts[0].keys())
                ctx_keys_2 = set(lens_contexts[1].keys())
                common_keys = ctx_keys_1 & ctx_keys_2

                for key in common_keys:
                    # For intersection, we could take the minimum probability or geometric mean
                    # For now, take the first lens's probabilities for common contexts
                    common_contexts[key] = lens_contexts[0][key]
                return Behavior.from_contexts(combined_lens_space, common_contexts)
            else:
                # For multiple lenses, this would be more complex
                # For now, fall back to union
                combined_contexts = {}
                for lens in self.lenses:
                    combined_contexts.update(lens.contexts_low_level())
                return Behavior.from_contexts(combined_lens_space, combined_contexts)

        elif self._mode == "difference":
            # Find difference: contexts in first lens but not constraining second
            if len(self.lenses) == 2:
                lens1_ctx = self.lenses[0].contexts_low_level()
                lens2_ctx = self.lenses[1].contexts_low_level()

                # Contexts unique to first lens
                unique_contexts = {}
                for ctx_key, ctx_dist in lens1_ctx.items():
                    if ctx_key not in lens2_ctx:
                        unique_contexts[ctx_key] = ctx_dist

                return Behavior.from_contexts(combined_lens_space, unique_contexts)
            else:
                # For multiple lenses, fall back to union
                combined_contexts = {}
                for lens in self.lenses:
                    combined_contexts.update(lens.contexts_low_level())
                return Behavior.from_contexts(combined_lens_space, combined_contexts)

        elif self._mode == "symmetric_difference":
            # Find symmetric difference: contexts in either lens but not both
            if len(self.lenses) == 2:
                lens1_ctx = self.lenses[0].contexts_low_level()
                lens2_ctx = self.lenses[1].contexts_low_level()

                symmetric_diff_contexts = {}

                # Contexts only in first lens
                for ctx_key, ctx_dist in lens1_ctx.items():
                    if ctx_key not in lens2_ctx:
                        symmetric_diff_contexts[ctx_key] = ctx_dist

                # Contexts only in second lens
                for ctx_key, ctx_dist in lens2_ctx.items():
                    if ctx_key not in lens1_ctx:
                        symmetric_diff_contexts[ctx_key] = ctx_dist

                return Behavior.from_contexts(combined_lens_space, symmetric_diff_contexts)
            else:
                # For multiple lenses, this is complex - fall back to union
                combined_contexts = {}
                for lens in self.lenses:
                    combined_contexts.update(lens.contexts_low_level())
                return Behavior.from_contexts(combined_lens_space, combined_contexts)

        else:
            # Unknown mode, fall back to union
            combined_contexts = {}
            for lens in self.lenses:
                combined_contexts.update(lens.contexts_low_level())
            return Behavior.from_contexts(combined_lens_space, combined_contexts)


@dataclass
class Observatory:
    """
    Main entry point for creating observational systems.

    Provides a high-level API for defining observables, their relationships,
    and the probability distributions that characterize different observational
    perspectives.

    Attributes:
        _concepts: Dictionary mapping concept names to their ConceptHandle objects
        _space: The underlying Space object (created when first concept is defined)
        _perspectives: PerspectiveMap for managing distributions
    """

    _concepts: Dict[str, ConceptHandle] = field(default_factory=dict)
    _observable_lenses: Dict[str, LensScope] = field(default_factory=dict)
    _space: Optional[Space] = None
    _perspectives: PerspectiveMap = field(init=False)
    _global_alphabet: Tuple[Any, ...] = field(default_factory=tuple)
    _global_concept: Optional[ConceptHandle] = None
    _generation: int = field(default=0, init=False)

    def __post_init__(self):
        self._perspectives = PerspectiveMap(_observatory=self)

    @classmethod
    def create(cls, symbols: Optional[Sequence[Any]] = None) -> Observatory:
        """
        Create a new Observatory instance.

        Args:
            symbols: Optional global alphabet symbols that can be used by concepts.
                     Can be strings, ValueHandle objects, or other symbols.

        Returns:
            New Observatory instance
        """
        if symbols is not None:
            # Extract symbols from ValueHandle objects if needed
            extracted_symbols = []
            for s in symbols:
                if isinstance(s, ValueHandle):
                    extracted_symbols.append(s.value)
                else:
                    extracted_symbols.append(s)

            instance = cls(_global_alphabet=tuple(extracted_symbols))
            # Create a global concept for the alphabet
            instance._global_concept = ConceptHandle("__global__", instance._global_alphabet, instance)
            return instance
        return cls()

    @property
    def alphabet(self) -> Tuple[ValueHandle, ...]:
        """Global alphabet as ValueHandle objects."""
        if self._global_concept:
            return self._global_concept.alphabet
        return ()

    def define_many(self, specs: Sequence[Union[str, Dict]]) -> Tuple[ConceptHandle, ...]:
        """
        Create multiple concepts with flexible symbol specification.

        Args:
            specs: List where each item is either:
                - str: Concept name (uses global alphabet)
                - dict: {"name": str, "symbols": Optional[Sequence]}

        Returns:
            Tuple of ConceptHandle objects in same order as specs

        Example:
            temp, pressure = obs.define_many([
                "Temperature",  # Uses global alphabet
                {"name": "Pressure", "symbols": ["Low", "High"]}
            ])
        """
        concepts = []
        for spec in specs:
            if isinstance(spec, str):
                concepts.append(self.concept(spec))
            elif isinstance(spec, dict):
                name = spec["name"]
                symbols = spec.get("symbols")
                concepts.append(self.concept(name, symbols))
            else:
                raise ValueError(f"Invalid spec: {spec}. Must be str or dict.")
        return tuple(concepts)

    def concept(self, name: str, symbols: Optional[Sequence[Any]] = None) -> ConceptHandle:
        """
        Define a new observable concept with its possible symbols.

        Args:
            name: Name of the observable concept
            symbols: Sequence of possible symbols for this concept.
                     If None, uses the global alphabet if available.

        Returns:
            A ConceptHandle object representing the defined concept
        """
        if name in self._concepts:
            raise ValueError(f"Concept '{name}' already exists")

        if symbols is None:
            if not self._global_alphabet:
                raise ValueError(f"No symbols provided for concept '{name}' and no global alphabet is set")
            symbols_tuple = self._global_alphabet
        else:
            # Extract symbols from ValueHandle objects if needed
            extracted_symbols = []
            for s in symbols:
                if isinstance(s, ValueHandle):
                    extracted_symbols.append(s.value)
                else:
                    extracted_symbols.append(s)
            symbols_tuple = tuple(extracted_symbols)

        concept_handle = ConceptHandle(name, symbols_tuple, self)
        self._concepts[name] = concept_handle

        # Create or update space
        if self._space is None:
            # First concept
            alphabets = {name: symbols_tuple}
            self._space = Space((name,), alphabets)
        else:
            # Add to existing space
            new_alphabets = {**self._space.alphabets, name: symbols_tuple}
            new_names = tuple(list(self._space.names) + [name])
            self._space = Space(new_names, new_alphabets)

        # Increment generation counter for cache invalidation
        self._generation += 1

        return concept_handle

    @property
    def perspectives(self) -> PerspectiveMap:
        """Get perspectives for managing distributions."""
        if not self._concepts:
            raise ValueError("Perspectives not available - define some concepts first")
        return self._perspectives

    @property
    def alphabet(self) -> Tuple['ValueHandle', ...]:
        """Get the global alphabet as ValueHandle objects."""
        if self._global_concept is None:
            return ()
        return self._global_concept.alphabet

    def lens(self, name: Union[str, ConceptHandle], symbols: Optional[Sequence[Any]] = None) -> 'LensScope':
        """
        Create a lens scope for the given perspective name.

        Args:
            name: Name of the lens or ConceptHandle
            symbols: Optional symbols for making this lens observable by other lenses.
                    If provided, this lens can be observed as a concept in meta-lenses.

        Returns:
            LensScope that can optionally be observed if symbols are provided

        - with semantics preserved
        - Behaviors default to BASE space via to_behavior()
        - Use to_behavior_raw() to obtain LENS-tagged behaviors
        - If symbols provided, lens becomes observable for meta-lens construction
        """
        lens_name = name.name if isinstance(name, ConceptHandle) else str(name)
        
        # Convert symbols to tuple if provided
        symbols_tuple = None
        if symbols is not None:
            # Extract symbols from ValueHandle objects if needed
            extracted_symbols = []
            for s in symbols:
                if isinstance(s, ValueHandle):
                    extracted_symbols.append(s.value)
                else:
                    extracted_symbols.append(s)
            symbols_tuple = tuple(extracted_symbols)
        
        # Create the lens
        lens_scope = LensScope(self, lens_name, symbols_tuple)
        
        # If observable (has symbols), register it and add to space
        if symbols_tuple is not None:
            self._observable_lenses[lens_name] = lens_scope
            
            # Add to space like a concept
            if self._space is None:
                # First observable - create space
                alphabets = {lens_name: symbols_tuple}
                self._space = Space((lens_name,), alphabets)
            else:
                # Add to existing space
                new_alphabets = {**self._space.alphabets, lens_name: symbols_tuple}
                new_names = tuple(list(self._space.names) + [lens_name])
                self._space = Space(new_names, new_alphabets)
            
            # Increment generation counter for cache invalidation
            self._generation += 1
        
        return lens_scope

# Backward-compatibility shim for import only (semantics changed intentionally)
LensBuilder = LensScope
