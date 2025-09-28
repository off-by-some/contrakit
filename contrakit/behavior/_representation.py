"""
Core Behavior Representation and Algebra

This module provides the fundamental representation and algebraic operations
for multi-perspective behavioral patterns. It contains constructors, tensor
operations, and basic transformations, but no analysis or optimization routines.
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Tuple, Sequence, Set, Union, Optional, Mapping, Iterable
from dataclasses import dataclass, field
import numpy as np
from ..space import Space
from ..context import Context
from ..distribution import Distribution
from ..frame import FrameIndependence


# Type aliases for sugar constructors
Outcome = Tuple[Any, ...]
ContextKey = Union[str, Tuple[str, ...]]


@dataclass
class Behavior:
    """
    A multi-perspective behavior representing probability distributions across observational contexts.

    A behavior captures how different measurements or observations relate to each other
    within a shared observable space. This allows us to detect whether the observations
    are consistent with a single underlying reality or whether they contain contradictions.

    Attributes:
        space: The observable space defining what can be measured and their possible values
        distributions: Dictionary mapping observational contexts to their probability distributions

    The behavior class provides methods for:
    - Computing agreement coefficients between different observational contexts
    - Measuring overall contradiction levels
    - Testing frame independence (consistency across all contexts)
    - Analyzing optimal measurement strategies

    Example:
        # Create a simple behavior for coffee preferences
        space = Space.create(Morning_Coffee=["Yes", "No"], Evening_Coffee=["Yes", "No"])

        # Define consistent responses (people who drink coffee in morning also do in evening)
        behavior = Behavior.from_contexts(space, {
            ("Morning_Coffee",): {("Yes",): 0.5, ("No",): 0.5},
            ("Evening_Coffee",): {("Yes",): 0.5, ("No",): 0.5},
            ("Morning_Coffee", "Evening_Coffee"): {("Yes", "Yes"): 0.5, ("No", "No"): 0.5}
        })
    """
    space: Space
    distributions: Dict[Context, Distribution] = field(default_factory=dict)

    @classmethod
    def from_contexts(cls, space: Space, context_dists: Optional[Dict] = None, *,
                      atol: float = 1e-10) -> Behavior:
        """
        Create a behavior from a dictionary of context-specific probability distributions.

        This method allows you to define how different combinations of observables
        (contexts) are distributed, which forms the basis for analyzing consistency
        and contradiction across different observational perspectives.

        Args:
            space: The observable space that defines what variables exist and their possible values
            context_dists: Dictionary mapping observational contexts to their probability distributions.
                          Keys can be tuples for multi-variable contexts or strings for single variables.
                          Values can be either dictionaries (outcome -> probability) or Distribution objects.
            atol: Absolute tolerance for consistency checking (default: 1e-10).

        Returns:
            A new Behavior instance with the specified distributions

        Raises:
            ValueError: If inconsistencies are detected between provided marginals and
                       implied marginals from joint distributions.

        Examples:
            # Single-variable context (what fraction of people prefer coffee?)
            space = Space.create(Coffee_Preference=["Yes", "No"])
            behavior = Behavior.from_contexts(space, {
                "Coffee_Preference": {("Yes",): 0.6, ("No",): 0.4}
            })

            # Multi-variable context (how do morning and evening preferences relate?)
            space = Space.create(Morning=["Yes", "No"], Evening=["Yes", "No"])
            behavior = Behavior.from_contexts(space, {
                ("Morning", "Evening"): {("Yes", "Yes"): 0.5, ("No", "No"): 0.5}
            })
        """
        behavior = cls(space)

        if context_dists is not None:
            for ctx_spec, pmf in context_dists.items():
                if isinstance(ctx_spec, str):
                    ctx_observables = [ctx_spec]
                else:
                    ctx_observables = list(ctx_spec)

                context = Context.make(space, ctx_observables)

                # Handle both dict and Distribution inputs
                if isinstance(pmf, Distribution):
                    # For Distribution objects, convert to dict for processing
                    pmf_dict = pmf.to_dict()
                else:
                    # Assume it's already a dict
                    pmf_dict = pmf

                # Validate pmf against context's outcome space
                valid_outcomes = context.outcomes()
                if not set(pmf_dict).issubset(set(valid_outcomes)):
                    extras = set(pmf_dict) - set(valid_outcomes)
                    raise ValueError(f"Invalid outcomes for context {tuple(ctx_observables)}: {extras}")

                # Fill missing outcomes with 0 probability (deterministic ordering)
                full_pmf = {outcome: pmf_dict.get(outcome, 0.0) for outcome in valid_outcomes}
                behavior[context] = Distribution.from_dict(full_pmf)

        # Always check consistency to catch errors early
        report = behavior.check_consistency(atol=atol)  # from mixin
        if not report["ok"]:
            pretty = "\n".join(
                f"  joint {m['joint']} vs marginal {m['marginal']}: provided={m['provided']} implied={m['implied']}"
                for m in report["mismatches"]
            )
            raise ValueError("Inconsistent contexts detected:\n" + pretty)

        return behavior

    @classmethod
    def frame_independent(cls, space: Space, contexts: Sequence[Sequence[str]],
                         assignment_weights: Optional[np.ndarray] = None) -> Behavior:
        """Create FI behavior from global assignment distribution."""
        assignments = list(space.assignments())  # Convert to list to avoid iterator exhaustion
        n = space.assignment_count()

        if assignment_weights is None:
            assignment_weights = np.ones(n) / n
        else:
            assignment_weights = np.array(assignment_weights)
            assignment_weights = assignment_weights / assignment_weights.sum()

        behavior = cls(space)

        for ctx_obs in contexts:
            context = Context.make(space, ctx_obs)
            pmf = {}

            for outcome in context.outcomes():
                prob = 0.0
                for assignment, weight in zip(assignments, assignment_weights):
                    if context.restrict_assignment(assignment) == outcome:
                        prob += weight
                pmf[outcome] = prob

            behavior[context] = Distribution.from_dict(pmf)

        return behavior

    @classmethod
    def random(cls, space: Space, contexts: Sequence[Sequence[str]],
               alpha: float = 1.0, seed: Optional[int] = None) -> Behavior:
        """Generate random behavior over given contexts."""
        behavior = cls(space)
        rng = np.random.default_rng(seed)

        for ctx_obs in contexts:
            context = Context.make(space, ctx_obs)
            outcomes = context.outcomes()
            behavior[context] = Distribution.random(outcomes, alpha=alpha, rng=rng)

        return behavior

    @classmethod
    def from_mu(cls, space: Space, contexts: Sequence[Sequence[str]],
                mu: np.ndarray, normalize: bool = True) -> Behavior:
        """Create behavior from global assignment distribution.

        This creates a behavior where each context shows the marginal distribution
        derived from a global probability distribution over all possible assignments.

        Args:
            space: The observable space
            contexts: List of context specifications (sequences of observable names)
            mu: Probability distribution over all global assignments.
                Must have length equal to space.num_assignments()
            normalize: Whether to normalize mu to sum to 1

        Returns:
            Behavior with marginal distributions for each context

        Raises:
            ValueError: If mu has wrong length or contexts are invalid

        Example:
            space = Space.create(A=["0","1"], B=["0","1"])
            contexts = [("A",), ("B",), ("A", "B")]
            mu = np.array([0.1, 0.2, 0.3, 0.4])  # P(A=0,B=0), P(A=0,B=1), P(A=1,B=0), P(A=1,B=1)
            behavior = Behavior.from_mu(space, contexts, mu)
        """
        # Get all possible global assignments
        all_assignments = list(space.assignments())
        expected_length = space.assignment_count()

        if len(mu) != expected_length:
            raise ValueError(f"mu must have length {expected_length}, got {len(mu)}")

        # Check nonnegativity
        mu_array = np.asarray(mu, dtype=float)
        if (mu_array < -1e-12).any():
            raise ValueError("mu must be nonnegative")

        # Normalize if requested
        mu = mu_array
        if normalize:
            mu = mu / mu.sum()

        behavior = cls(space)

        # Create name to index mapping for efficient lookup
        name_to_idx = {name: i for i, name in enumerate(space.names)}

        # For each context, compute marginal distribution
        for ctx_obs in contexts:
            context = Context.make(space, ctx_obs)

            # Initialize probability mass function for this context
            pmf = {outcome: 0.0 for outcome in context.outcomes()}

            # Sum over all global assignments weighted by mu
            for global_assignment, weight in zip(all_assignments, mu):
                # Extract the values for this context's observables
                context_outcome = tuple(global_assignment[name_to_idx[name]] for name in ctx_obs)
                pmf[context_outcome] += weight

            behavior[context] = Distribution.from_dict(pmf)

        return behavior

    @classmethod
    def from_counts(
        cls,
        space: Space,
        context_tables: Mapping[ContextKey, Mapping[Outcome, Union[int, float]]],
        *,
        normalize: str = "per_context",   # "per_context" | "global" | "none"
    ) -> "Behavior":
        """
        Build a Behavior from integer/float counts per context.

        Args:
            space: The observable space
            context_tables: Mapping from context keys to count tables
            normalize: Normalization strategy:
                - "per_context": each context table is normalized to a pmf (default)
                - "global": a single global Z is used for all contexts
                - "none": assume the values are already probabilities

        Returns:
            A new Behavior instance with normalized distributions

        Example:
            # Raw counts for Simpson's paradox
            space = Space.create(Treatment=["A","B"], Outcome=[1,0], School=["S1","S2"])
            counts = {
                ("Treatment","Outcome","School"): {
                    ("A",1,"S1"): 9, ("A",0,"S1"): 1,
                    ("B",1,"S1"): 2, ("B",0,"S1"): 8,
                    ("A",1,"S2"): 1, ("A",0,"S2"): 9,
                    ("B",1,"S2"): 8, ("B",0,"S2"): 2,
                }
            }
            behavior = Behavior.from_counts(space, counts, normalize="global")
        """
        dists = {}
        if normalize not in {"per_context", "global", "none"}:
            raise ValueError("normalize must be one of {'per_context','global','none'}")

        Z_global = None
        if normalize == "global":
            Z_global = sum(v for table in context_tables.values() for v in table.values())
            if Z_global <= 0:
                raise ValueError("Global normalization constant must be positive")

        for ctx, table in context_tables.items():
            if normalize == "per_context":
                Z = sum(table.values())
                if Z <= 0:
                    raise ValueError(f"Context {ctx} has nonpositive total")
                probs = {k: v / Z for k, v in table.items()}
            elif normalize == "global":
                probs = {k: v / Z_global for k, v in table.items()}
            else:
                probs = dict(table)

            # Convert context key to Context object
            ctx_observables = cls._normalize_context_key(ctx)
            context = Context.make(space, ctx_observables)
            dists[context] = Distribution.from_dict(probs)

        return cls(space=space, distributions=dists)

    @staticmethod
    def _normalize_context_key(ctx: ContextKey) -> List[str]:
        """Convert context key to list of observable names."""
        if isinstance(ctx, str):
            return [ctx]
        else:
            return list(ctx)

    def __len__(self) -> int:
        """Number of contexts."""
        return len(self.distributions)

    def __contains__(self, context: Context) -> bool:
        """Check if context is measured."""
        return context in self.distributions

    def __getitem__(self, context: Context) -> Distribution:
        """Get distribution for context."""
        return self.distributions[context]

    def __setitem__(self, context: Context, distribution: Distribution):
        """Set distribution for context."""
        if context.space != self.space:
            raise ValueError("Context must belong to the same space")

        # Forbid multiple contexts with identical observable sets
        for existing_ctx in self.distributions.keys():
            if tuple(existing_ctx.observables) == tuple(context.observables):
                raise ValueError(
                    "Duplicate context with the same observables is not allowed: "
                    f"{tuple(context.observables)}. Use Behavior.duplicate_context(...) "
                    "to introduce tagged copies, or add a label observable."
                )

        # Validate that distribution outcomes match context outcomes exactly
        want = tuple(context.outcomes())
        if set(distribution.outcomes) != set(want):
            extra = set(distribution.outcomes) - set(want)
            missing = set(want) - set(distribution.outcomes)
            raise ValueError(f"Outcome mismatch for context {context.observables} (extra={extra}, missing={missing})")

        # Check for duplicate outcomes in distribution
        if len(set(distribution.outcomes)) != len(distribution.outcomes):
            raise ValueError(f"Distribution for context {context.observables} has duplicate outcomes")

        self.distributions[context] = distribution

    def __matmul__(self, other: Behavior) -> Behavior:
        """Tensor product: behavior1 @ behavior2"""
        if set(self.space.names) & set(other.space.names):
            raise ValueError("Behaviors must have disjoint observables for tensor product")

        product_space = self.space @ other.space
        product_behavior = type(self)(product_space, {})

        for ctx1, dist1 in self.distributions.items():
            for ctx2, dist2 in other.distributions.items():
                # Combined context
                product_context = Context.make(product_space,
                                             ctx1.observables + ctx2.observables)

                # Product distribution
                product_pmf = {}
                for outcome1, p1 in zip(dist1.outcomes, dist1.probs):
                    for outcome2, p2 in zip(dist2.outcomes, dist2.probs):
                        combined_outcome = outcome1 + outcome2
                        product_pmf[combined_outcome] = p1 * p2

                product_behavior[product_context] = Distribution.from_dict(product_pmf)

        return product_behavior

    def __add__(self, other: Behavior) -> Behavior:
        """Convex combination with equal weights."""
        return self.mix(other, 0.5)

    def mix(self, other: Behavior, weight: float) -> Behavior:
        """Convex combination: (1-weight)*self + weight*other"""
        if self.space != other.space:
            raise ValueError("Behaviors must have the same space (names + alphabets)")

        if set(self.distributions.keys()) != set(other.distributions.keys()):
            raise ValueError("Behaviors must have the same contexts")

        mixed = type(self)(self.space, {})
        for context in self.distributions:
            mixed[context] = self.distributions[context].mix(other.distributions[context], weight)

        return mixed

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Behavior(space={len(self.space)} obs, {len(self)} contexts)"

    def rename_observables(self, name_map: Dict[str, str]) -> Behavior:
        """Create behavior with renamed observables."""
        new_space = self.space.rename(name_map)
        new_behavior = type(self)(new_space, {})

        for context, distribution in self.distributions.items():
            # Rename context observables
            new_observables = [name_map.get(name, name) for name in context.observables]
            new_context = Context.make(new_space, new_observables)
            new_behavior[new_context] = distribution  # Outcomes unchanged

        return new_behavior

    def permute_outcomes(self, value_maps: Dict[str, Dict[Any, Any]]) -> Behavior:
        """Apply permutations to outcome labels within observables."""
        # Validate that each mapping is actually a bijection
        for name, alpha in self.space.alphabets.items():
            if name in value_maps:
                mapping = value_maps[name]
                if set(mapping.keys()) != set(alpha) or set(mapping.values()) != set(alpha):
                    raise ValueError(f"value_maps[{name}] must be a bijection over {alpha}")

        new_behavior = type(self)(self.space, {})

        for context, distribution in self.distributions.items():
            new_pmf = {}

            # Group outcomes by their mapped values
            for outcome, prob in zip(distribution.outcomes, distribution.probs):
                # Apply permutation to each position
                new_outcome = []
                for i, obs_name in enumerate(context.observables):
                    old_val = outcome[i]
                    if obs_name in value_maps:
                        new_val = value_maps[obs_name][old_val]
                    else:
                        new_val = old_val
                    new_outcome.append(new_val)

                new_outcome = tuple(new_outcome)
                new_pmf[new_outcome] = new_pmf.get(new_outcome, 0.0) + prob

            new_behavior[context] = Distribution.from_dict(new_pmf)

        return new_behavior

    def coarse_grain(self, observable: str, merge_map: Dict[Any, Any]) -> Behavior:
        """Coarse-grain outcomes for a specific observable."""
        # Create new space with coarse-grained alphabet
        old_alphabet = self.space.alphabets[observable]
        if set(merge_map.keys()) != set(old_alphabet):
            raise ValueError(f"merge_map must cover alphabet {old_alphabet}")

        # New alphabet preserves order of first appearance
        new_alphabet = []
        seen = set()
        for old_val in old_alphabet:
            new_val = merge_map[old_val]
            if new_val not in seen:
                seen.add(new_val)
                new_alphabet.append(new_val)

        new_alphabets = dict(self.space.alphabets)
        new_alphabets[observable] = tuple(new_alphabet)
        new_space = Space(self.space.names, new_alphabets)

        new_behavior = type(self)(new_space, {})

        for context, distribution in self.distributions.items():
            if observable not in context.observables:
                # Context doesn't involve this observable - copy unchanged
                new_context = Context.make(new_space, context.observables)
                new_behavior[new_context] = distribution
            else:
                # Apply coarse-graining
                new_context = Context.make(new_space, context.observables)
                new_pmf = {}

                obs_idx = context.observables.index(observable)

                for outcome, prob in zip(distribution.outcomes, distribution.probs):
                    # Map the observable value
                    new_outcome = list(outcome)
                    new_outcome[obs_idx] = merge_map[outcome[obs_idx]]
                    new_outcome = tuple(new_outcome)

                    new_pmf[new_outcome] = new_pmf.get(new_outcome, 0.0) + prob

                new_behavior[new_context] = Distribution.from_dict(new_pmf)

        return new_behavior

    def drop_contexts(self, keep_contexts: Set[Tuple[str, ...]]) -> Behavior:
        """Keep only specified contexts (by observable names)."""
        # Normalize keep_contexts to handle order variations
        normalized_keep = {tuple(sorted(ctx)) for ctx in keep_contexts}

        new_behavior = type(self)(self.space, {})

        for context, distribution in self.distributions.items():
            # Check both original and normalized order
            ctx_normalized = tuple(sorted(context.observables))
            if context.observables in keep_contexts or ctx_normalized in normalized_keep:
                new_behavior[context] = distribution

        if len(new_behavior) == 0:
            raise ValueError("No contexts remain after filtering")

        return new_behavior

    def duplicate_context(self, target_context: Tuple[str, ...],
                         count: int, tag_prefix: str = "Tag") -> Behavior:
        """Duplicate a context using singleton tag observables."""
        if count < 2:
            raise ValueError("count must be >= 2 for duplication")

        # Extend space with singleton tag observables (avoid name collisions)
        tag_names = []
        i = 1
        while len(tag_names) < count:
            name = f"{tag_prefix}{i}"
            if name not in self.space.names:
                tag_names.append(name)
            i += 1

        extended_space = self.space @ Space.create(**{name: [0] for name in tag_names})

        new_behavior = type(self)(extended_space, {})

        for context, distribution in self.distributions.items():
            if context.observables == target_context:
                # Create multiple copies with different tags
                for i, tag_name in enumerate(tag_names):
                    new_observables = list(context.observables) + [tag_name]
                    new_context = Context.make(extended_space, new_observables)

                    # Extend outcomes with tag value
                    new_pmf = {}
                    for outcome, prob in zip(distribution.outcomes, distribution.probs):
                        extended_outcome = outcome + (0,)  # Tag always has value 0
                        new_pmf[extended_outcome] = prob

                    new_behavior[new_context] = Distribution.from_dict(new_pmf)
            else:
                # Keep other contexts unchanged (no tag)
                new_context = Context.make(extended_space, context.observables)
                new_behavior[new_context] = distribution

        return new_behavior

    def product_l1_distance(self, other: Behavior) -> float:
        """Product L1 distance: max over contexts of L1 distances."""
        if self.space != other.space:
            raise ValueError("Behaviors must have same space (names + alphabets)")

        if set(self.distributions.keys()) != set(other.distributions.keys()):
            raise ValueError("Behaviors must have same contexts")

        max_distance = 0.0
        for context in self.distributions:
            dist1 = self.distributions[context]
            dist2 = other.distributions[context]
            distance = dist1.l1_distance(dist2)
            max_distance = max(max_distance, distance)

        return max_distance

