"""
Behavior Class - Core API for Multi-Perspective Behavioral Analysis

This module provides the main Behavior class that combines representation,
algebraic operations, and analysis capabilities for studying contradictions
in multi-perspective observational data.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from ._representation import Behavior as BaseBehavior
from ._analysis import BehaviorAnalysisMixin
from ..agreement import BhattacharyyaCoefficient
from ..frame import FrameIndependence
from ..context import Context
from ..space import Space
from .agreement_api import Agreement


class Behavior(BaseBehavior, BehaviorAnalysisMixin):
    """
    A multi-perspective behavior representing probability distributions across observational contexts.

    A behavior captures how different measurements or observations relate to each other
    within a shared observable space. This allows us to detect whether the observations
    are consistent with a single underlying reality or whether they contain contradictions.

    This class combines:
    - Core representation and algebraic operations (from _representation)
    - Analysis, sampling, and optimization routines (from _analysis)

    Attributes:
        space: The observable space defining what can be measured and their possible values
        distributions: Dictionary mapping observational contexts to their probability distributions

    Key Properties:
    - alpha_star: Optimal agreement coefficient between contexts
    - agreement: Alias for alpha_star
    - K: Contradiction measure in bits (-log₂(alpha_star))
    - contradiction_bits: Alias for K
    - context: List of all measurement contexts

    Key Methods:
    - Constructors: from_contexts(), from_mu(), from_counts(), frame_independent(), random()
    - Algebra: __matmul__ (tensor product), __or__ (union), mix() (convex combinations)
    - Combination: union() (combine contexts from different behaviors)
    - Transformations: rename_observables(), permute_outcomes(), coarse_grain()
    - Analysis: worst_case_weights()
    - Sampling: sample_observations(), count_observations()
    - Utilities: is_frame_independent()

    Example:
        # Create a behavior for coffee preferences
        space = Space.create(Morning_Coffee=["Yes", "No"], Evening_Coffee=["Yes", "No"])

        # Define distributions across contexts
        behavior = Behavior.from_contexts(space, {
            ("Morning_Coffee",): {("Yes",): 0.6, ("No",): 0.4},
            ("Evening_Coffee",): {("Yes",): 0.6, ("No",): 0.4},
            ("Morning_Coffee", "Evening_Coffee"): {("Yes", "Yes"): 0.5, ("No", "No"): 0.5}
        })

        # Check for contradictions
        print(f"Agreement coefficient: {behavior.alpha_star}")
        print(f"Contradiction (bits): {behavior.K}")
    """
    
    def __init__(self, space, distributions=None):
        """Initialize Behavior with both dataclass and mixin setup."""
        # Initialize as dataclass
        super().__init__(space, distributions or {})
        # Initialize mixin attributes
        self._alpha_cache = None

    @property
    def global_agreement(self) -> float:
        """
        The optimal agreement coefficient (α*).

        This value reflects the maximum possible consistency across all
        perspectives, under the most adversarial choice of weights. It is
        the strongest guarantee on how well the observations can agree.

        Returns:
            float in [0,1].

        Example:
            >>> behavior.global_agreement
            0.71

        Interpretation:
            If α* < 1, then no matter how perspectives are weighted,
            a contradiction remains. Agreement measures the best possible
            level of consistency achievable.
        """
        return self.agreement.result

    @property
    def contradiction_bits(self) -> float:
        """
        The contradiction measure K, expressed in bits.

        Contradiction is quantified as the information cost required to
        reconcile inconsistent perspectives. Higher values indicate
        stronger contradictions.

        Returns:
            float ≥ 0 (0 means no contradiction).

        Example:
            >>> behavior.contradiction_bits
            1.0

        Interpretation:
            A contradiction cost of 1 bit means that one additional
            yes/no question would always be needed to resolve the
            inconsistency between perspectives.
        """
        return self.K

    @property
    def context(self) -> List[Context]:
        """
        List all observational contexts in the behavior.

        A context is defined by which observables were measured together.
        Examining contexts reveals what perspectives are being compared.

        Returns:
            list of Context

        Example:
            >>> behavior.context
            [Context(['Morning']), Context(['Evening']), Context(['Morning','Evening'])]

        Interpretation:
            This behavior includes forecasts for morning, evening, and the
            whole day. The contexts define the structure of perspectives
            under analysis.
        """
        return list(self.distributions.keys())

    @property
    def agreement(self) -> Agreement:
        """
        Fluent API for agreement calculations.

        Returns:
            Agreement: A query builder object for fluent agreement calculations.

        Examples:
            # Basic α* (minimax agreement across all contexts)
            >>> alpha = behavior.agreement.result  # float: agreement score
            >>> theta = behavior.agreement.explanation  # np.ndarray: scenario distribution
            >>> scenarios = behavior.agreement.scenarios()  # list of (scenario, prob) pairs

            # Per-context agreement scores
            >>> per_context_scores = behavior.agreement.by_context().context_scores  # dict

            # With explicit trust weights over contexts
            >>> w = {("Witness_A",): 0.5, ("Witness_B",): 0.5}
            >>> weighted_score = behavior.agreement.for_weights(w).result  # float

            # Filter to contexts containing a specific feature
            >>> hair_score = behavior.agreement.for_feature("Hair").result  # float

            # Fixed feature distribution
            >>> fixed_score = behavior.agreement.for_feature("Hair", [0.3, 0.7]).result  # float

            # These can chain
            >>> combo_score = (
            ...     behavior.agreement
            ...     .for_weights(w)
            ...     .for_feature("Hair", [0.5, 0.5])
            ...     .result  # float: bottleneck score under all constraints
            ... )
        """
        return Agreement(self)

    def is_frame_independent(self, tol: float = 1e-9) -> bool:
        """
        Test whether the behavior is frame-independent.

        Frame independence means that all contexts can be explained by a
        single consistent underlying distribution. If not, contradiction
        is present.

        Args:
            tol (float): Numerical tolerance for the check.

        Returns:
            bool

        Example:
            >>> behavior.is_frame_independent()
            False

        Interpretation:
            If the morning forecast says mostly sun, the evening forecast
            also says mostly sun, but the all-day forecast insists on rain,
            then no single underlying model can explain them all. The
            result is False, indicating contradiction.
        """
        return FrameIndependence.check(self, tol).is_fi

    def union(self, other: "Behavior") -> "Behavior":
        """
        Create a new behavior that combines contexts from this behavior and another.

        This is useful for comparing different observational strategies or combining
        multiple perspectives into a single analysis.

        Parameters
        ----------
        other : Behavior
            Another behavior to combine with.

        Returns
        -------
        Behavior
            A new behavior containing all contexts from both input behaviors.
        """
        if self.space != other.space:
            raise ValueError("Behaviors must have the same outcome space for union")

        # Combine contexts from both behaviors
        # Convert Context objects to tuples for from_contexts
        combined_contexts = {}
        for ctx, dist in self.distributions.items():
            combined_contexts[tuple(ctx.observables)] = dist.to_dict()
        for ctx, dist in other.distributions.items():
            combined_contexts[tuple(ctx.observables)] = dist.to_dict()

        # Create combined behavior
        from .behavior import Behavior as _B
        return _B.from_contexts(self.space, combined_contexts)

    def __or__(self, other: "Behavior") -> "Behavior":
        """
        Union operator for behaviors.

        Allows syntax like: (beh1 | beh2).agreement.result
        """
        return self.union(other)

    @property
    def worst_case_weights(self) -> Dict[Tuple[str, ...], float]:
        """Get the least favorable context mixing λ* (CVXPY dual variables)."""
        return self.least_favorable_lambda()

    def agreement_for_observable(self, observable: str, feature_distribution: np.ndarray) -> dict[tuple, float]:
        """
        Per-context agreement when we assume a distribution for one observable.

        Parameters
        ----------
        observable : str
            Observable name, e.g. "Hair".
        feature_distribution : np.ndarray
            1-D probabilities over `observable`'s alphabet order in THIS behavior's space.

        Returns
        -------
        dict[tuple, float]
            Mapping {context_key -> agreement in [0,1]} for contexts that include `observable`.
        """
        q = np.asarray(feature_distribution, dtype=float)
        if q.ndim != 1 or q.size == 0:
            raise ValueError("feature_distribution must be a non-empty 1-D probability vector")

        try:
            feat_idx = self.space.names.index(observable)
        except ValueError:
            raise ValueError(f"observable '{observable}' not found in this behavior's space")

        # normalize and guard
        q = np.maximum(q, 0.0)
        Z = q.sum()
        if Z <= 0:
            raise ValueError("feature_distribution must have positive mass")
        q = q / Z

        feat_alphabet = list(self.space.alphabets[feat_idx])
        sym_to_idx = {s: i for i, s in enumerate(feat_alphabet)}

        results: dict[tuple, float] = {}
        for ctx in self.context:
            if observable not in ctx.observables:
                continue
            # index of `observable` inside this context's observable tuple
            j = ctx.observables.index(observable)
            g = 0.0
            dist = self.distributions[ctx].to_dict()
            for outcome, p in dist.items():
                # outcome is a tuple of symbols aligned to ctx.observables
                s = outcome[j]
                qi = q[sym_to_idx[s]]
                g += (float(p) ** 0.5) * (float(qi) ** 0.5)
            results[tuple(ctx.observables)] = float(g)
        return results
