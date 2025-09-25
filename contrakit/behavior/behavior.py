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
from ..frame import FrameIndependence
from ..context import Context
from ..space import Space


@dataclass
class AgreementResult:
    score: float
    """
    Agreement score α(λ), ranging from 0 to 1.

    This quantifies how well the observed evaluations align, *given your weighting of trust*
    across perspectives (λ). A value of 1.0 indicates perfect coherence; 0.0 indicates total contradiction.
    """

    theta: np.ndarray
    """
    Optimal global distribution θ*(λ), a probability vector.

    This is the single best explanation that tries to match all the observed reviewer behaviors as closely as possible, according to how you weighted each one.

    Intuitively, if agreement_for_weights is the level of consensus from your point of view, then theta describes how that consensus was constructed — according to your perspective.

    Each element corresponds to a complete set of reviewer decisions, and its value reflects how plausible that full scenario is — from your perspective.
    """

    space: Space
    """
    The observable space that defines the assignment order for theta.

    This is needed to map the probability vector elements back to meaningful
    variable assignments for interpretation.
    """

    def scenarios(self) -> List[Tuple[Tuple[Any, ...], float]]:
        """
        Get all global scenarios from theta, matched to the variable assignments in the space.

        Returns:
            List of (assignment, probability) tuples for all possible scenarios.
            Each assignment is a tuple representing a complete scenario across all observables.
            The probabilities sum to 1.0.

        Example:
            >>> result = behavior.agreement_for_weights(weights)
            >>> scenarios = result.scenarios()
            >>> print(f"Total scenarios: {len(scenarios)}")

            # Use to create a behavior from scenarios:
            >>> context = tuple(result.space.names)  # All observables
            >>> context_dist = {scenario: prob for scenario, prob in scenarios}
            >>> consensus_behavior = Behavior.from_contexts(result.space, {context: context_dist})
        """
        assignments = list(self.space.assignments())
        return [(assignments[i], self.theta[i]) for i in range(len(assignments))]

    def __repr__(self) -> str:
        """Enhanced representation showing key information."""
        return f"AgreementResult(score={self.score:.6f}, theta_shape={self.theta.shape})"


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
    - Algebra: __matmul__ (tensor product), mix() (convex combinations)
    - Transformations: rename_observables(), permute_outcomes(), coarse_grain()
    - Analysis: worst_case_weights(), agreement_for_weights()
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
    def agreement(self) -> float:
        """
        The optimal agreement coefficient (α*).

        This value reflects the maximum possible consistency across all
        perspectives, under the most adversarial choice of weights. It is
        the strongest guarantee on how well the observations can agree.

        Returns:
            float in [0,1].

        Example:
            >>> behavior.agreement
            0.71

        Interpretation:
            If α* < 1, then no matter how perspectives are weighted,
            a contradiction remains. Agreement measures the best possible
            level of consistency achievable.
        """
        return self.alpha_star

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

    def agreement_with(self, other: "Behavior", weights: Optional[dict] = None):
        """
        Compare this behavior with another by merging their distributions and computing agreement.

        Args:
            other (Behavior): The other behavior to compare against.
            weights (dict, optional): Optional trust weights for specific contexts in the combined behavior.

        Returns:
            float or AgreementResult:
                - If weights is None: returns the α* agreement score (float).
                - If weights is provided: returns AgreementResult object with more detailed info.
        """
        # Ensure compatible observable spaces
        if self.space != other.space:
            diff = self.space.difference(other.space)

            error_msg = "Cannot compare behaviors with different observable spaces.\n"
            if diff['only_self']:
                error_msg += f"  - This behavior has observables not in other: {sorted(diff['only_self'])}\n"
            if diff['only_other']:
                error_msg += f"  - Other behavior has observables not in this: {sorted(diff['only_other'])}\n"
            if diff['alphabet_diffs']:
                error_msg += "  - Different alphabets for shared observables:\n"
                for name, (self_alpha, other_alpha) in diff['alphabet_diffs'].items():
                    error_msg += f"    {name}: {self_alpha} vs {other_alpha}\n"
            error_msg += "\nTip: Define shared concepts globally in the universe, not in lens scopes."
            raise ValueError(error_msg)

        # Check for overlapping contexts
        overlapping = set(self.distributions) & set(other.distributions)
        if overlapping:
            overlapping_obs = sorted(tuple(ctx.observables) for ctx in overlapping)
            raise ValueError(f"Cannot compare behaviors with overlapping contexts: {overlapping_obs}")

        # Merge distributions. Keys are Context objects already, so
        # construct the combined Behavior directly rather than using
        # from_contexts (which expects string/tuple keys).
        merged = {**self.distributions, **other.distributions}
        combined = Behavior(self.space, merged)

        return (
            combined.agreement_for_weights(weights).score
            if weights else
            combined.agreement
        )



    @property
    def worst_case_weights(self):
        """
        Find the context weighting that exposes contradiction most strongly.

        The model identifies how much weight to place on each perspective
        so that the overall agreement is minimized. This highlights where
        the tension between perspectives is concentrated.

        Returns:
            dict: Mapping from context (tuple of observables) to weight.

        Example:
            >>> from contradiction import Space, Behavior
            >>> space = Space.create(Morning=["Sunny","Rain"], Evening=["Sunny","Rain"])
            >>> behavior = Behavior.from_contexts(space, {
            ...     ("Morning",): {("Sunny",): 0.9, ("Rain",): 0.1},
            ...     ("Evening",): {("Sunny",): 0.9, ("Rain",): 0.1},
            ...     ("Morning","Evening"): {("Rain","Rain"): 1.0}
            ... })
            >>> behavior.worst_case_weights()
            {('Morning',): 0.5, ('Evening',): 0.5, ('Morning','Evening'): 0.0}

        Interpretation:
            Here the morning and evening forecasts are strongly sunny,
            while the daily joint forecast insists on rain. The weighting
            shows that the contradiction is driven by the single-time
            forecasts, not the joint perspective.
        """
        return self.least_favorable_lambda()


    def agreement_for_weights(self, perspective_weights) -> AgreementResult:
        """
        This method asks:
            "How contradictory do the evaluations seem from your point of view?"

        Where perspective_weights is a dictionary mapping each evaluation context
        (i.e. scoring criteria or reviewer) to a trust weight.

        For example, imagine you're conducting a hiring process and reviewing candidate evaluations:
        - Each reviewer scored the candidates based on different criteria.
        - Some focused on technical skills, others on culture fit, others on combinations.

        let's say we know Reviewer C tends to be more lenient than Reviewer A
        and Reviewer B. We might have:

        ```python
        weights = {
            ("Reviewer_A",): 0.6,
            ("Reviewer_B",): 0.3,
            ("Reviewer_C",): 0.1
        }
        ```
        This means you personally think Reviewer_A should have 60% of the final say, while
        Reviewer_B and Reviewer_C should have 30% and 20%, respectively.

        Args:
            perspective_weights (dict):
                A dictionary mapping each evaluation context (i.e. scoring criteria or reviewer)
                to a trust weight. These weights must be non-negative and will be automatically
                normalized to sum to 1.

        Returns:
            AgreementResult:
                - score: Agreement score α(λ): 0 (fully contradictory) to 1 (fully consistent)
                - theta: Optimal global distribution θ*(λ): a probability vector explaining the behavior
                  as best as possible under the given trust weights
                - space: The observable space for interpreting theta

        Example:
            >>> from contrakit import Space, Behavior

            >>> # Each reviewer rated the candidate "Hire" or "No_Hire"
            >>> H, N = "Hire", "No_Hire"
            >>> space = Space.create(**{r: [H, N] for r in ["Candidate", "Reviewer_A", "Reviewer_B", "Reviewer_C"]})
            >>> behavior = Behavior.from_contexts(space, {
            ...     # Individual reviewer tendencies
            ...     ("Reviewer_A",): {(H,): 0.7}, # A hires 70% of the time
            ...     ("Reviewer_B",): {(H,): 0.4}, # B is more skeptical
            ...     ("Reviewer_C",): {(H,): 0.8}, # C is more generous
            ...     # Reviewer A and B:
            ...     #   - Both said "Hire" in 30% of cases.
            ...     #   - A said "Hire" while B said "No Hire" 40% of the time.
            ...     #   - A said "No Hire" while B said "Hire" only 10% of the time.
            ...     #   - Both rejected the candidate in 20% of cases.
            ...     ("Reviewer_A", "Reviewer_B"): {
            ...         (H, H): 0.3, (H, N): 0.4, (N, H): 0.1, (N, N): 0.2
            ...     },
            ...     # Reviewer B and C joint responses
            ...     ("Reviewer_B", "Reviewer_C"): {
            ...         (H, H): 0.35, (H, N): 0.05, (N, H): 0.35, (N, N): 0.25
            ...     },
            ...     # Reviewer A and C joint responses
            ...     ("Reviewer_A", "Reviewer_C"): {
            ...         (H, H): 0.6, (H, N): 0.1, (N, H): 0.1, (N, N): 0.2
            ...     }
            >>> })
            >>> # Shows there's contradiction between the reviewers
            >>> print(f"Initial agreement: {behavior.agreement:.6f}")  # 0.998319
            >>> weights = {("Reviewer_A",): 0.6, ("Reviewer_B",): 0.3, ("Reviewer_C",): 0.1}
            >>> # But given your perspective, the evaluations can be made to agree fully
            >>> result = behavior.agreement_for_weights(weights)
            >>> print(f"Agreement: {result.score:.6f}")  # 1.000000
            >>> scenarios = result.scenarios()
            >>> # Sort and show top scenarios
            >>> sorted_scenarios = sorted(scenarios, key=lambda x: x[1], reverse=True)
            >>> for i, (scenario, prob) in enumerate(sorted_scenarios[:4]):
            ...     print(f"{i+1}. {scenario}: {prob:.1%} probability")
            1. ('No_Hire', 'Hire', 'No_Hire', 'Hire'): 18.3% probability
            2. ('Hire', 'Hire', 'No_Hire', 'Hire'): 18.3% probability
            3. ('No_Hire', 'Hire', 'Hire', 'Hire'): 13.2% probability
            4. ('Hire', 'Hire', 'Hire', 'Hire'): 13.2% probability

        You’re not just averaging scores — you're asking whether all the evaluations could reflect
        a single, coherent underlying picture of the candidates, **given how much you trust each source**.
        Lower values mean even your trusted reviewers are in structural conflict.
        """
        # Normalize weights to sum to 1
        total_weight = sum(perspective_weights.values())
        if total_weight <= 0:
            raise ValueError("Weights must sum to a positive value")
        normalized_weights = {k: v / total_weight for k, v in perspective_weights.items()}

        score, theta = self.alpha_given_lambda(normalized_weights)
        return AgreementResult(score=score, theta=theta, space=self.space)
