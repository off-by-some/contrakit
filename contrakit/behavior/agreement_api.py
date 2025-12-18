# contrakit/behavior/agreement_api.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple, Optional, Union, List, Any, Mapping
import numpy as np

ContextKey = Tuple[str, ...]  # e.g. ("Hair",) or ("Hair","Witness")


@dataclass(frozen=True)
class Agreement:
    """
    Fluent API for agreement calculations against a Behavior.

    Usage:
      ABC.agreement.result                           -> α* (float)
      ABC.agreement.context_scores                   -> {context -> score}
      ABC.agreement.by_context().result              -> bottleneck score (min of context scores)
      ABC.agreement.for_weights(w).result            -> weighted α* (float)
      ABC.agreement.for_feature("Hair").result       -> α* over Hair contexts only (float)
      ABC.agreement.for_feature("Hair", q).result    -> agreement with Hair fixed to q (float)
      ABC.agreement.explanation                      -> θ (np.ndarray or None)
      ABC.agreement.scenarios()                      -> [(scenario, probability), ...]
    """
    _behavior: "Behavior"
    _mode_by_context: bool = False
    _weights: Optional[Dict[ContextKey, float]] = None
    _feature_name: Optional[str] = None
    _feature_dist: Optional[np.ndarray] = None
    _fixed_expl_vec: Optional[np.ndarray] = None
    _keep: Optional[Tuple[ContextKey, ...]] = None
    _drop: Optional[Tuple[ContextKey, ...]] = None

    # ---------- fluent refinements ----------
    def for_weights(self, weights: Dict[ContextKey, float]) -> "Agreement":
        """Specify trust weights across contexts."""
        if not weights:
            raise ValueError("weights must be a non-empty mapping")

        # Validate weights
        if any(w < 0 for w in weights.values()):
            raise ValueError("Weights must be non-negative")

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")

        # Filter weights to only include contexts that will be present after filtering
        filtered_behavior = self._filtered_behavior()
        active_contexts = {tuple(ctx.observables) for ctx in filtered_behavior.context}
        filtered_weights = {k: v for k, v in weights.items() if k in active_contexts}

        if not filtered_weights:
            raise ValueError("None of the provided weights match the active contexts after filtering")

        # Renormalize the filtered weights
        weight_sum = sum(filtered_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in filtered_weights.items()}

        return replace(self, _weights=normalized_weights)

    def for_feature(self, feature: str, feature_distribution: Optional[np.ndarray] = None) -> "Agreement":
        """
        Restrict agreement calculation by fixing one observable's distribution.
        If feature_distribution is None, this just records `feature` so you can attach
        the distribution later (useful in chained calls).
        """
        # Validate that feature exists in the behavior's space
        if feature not in self._behavior.space.names:
            raise ValueError(f"Observable '{feature}' not found in behavior space")

        # Compute contexts that include this feature
        all_contexts = {tuple(ctx.observables) for ctx in self._behavior.context}
        feature_contexts = {ctx_key for ctx_key in all_contexts if feature in ctx_key}

        if not feature_contexts:
            raise ValueError(f"No contexts include feature '{feature}'")

        # Set keep to contexts that include the feature
        keep_set = tuple(feature_contexts)

        if feature_distribution is None:
            # Filter mode: only evaluate contexts containing the feature
            return replace(self, _feature_name=feature, _feature_dist=None, _keep=keep_set)

        q = np.asarray(feature_distribution, float).ravel()
        if q.size == 0 or np.any(q < 0) or q.sum() <= 0:
            raise ValueError("feature_distribution must be a positive 1-D probability vector")

        # Validate distribution length matches the alphabet size
        expected_length = len(self._behavior.space.alphabets[feature])
        if len(q) != expected_length:
            raise ValueError(f"feature_distribution must have length {expected_length} for observable '{feature}'")

        # Fixed feature mode: evaluate with fixed distribution on contexts containing the feature
        return replace(self, _feature_name=feature, _feature_dist=q / q.sum(), _keep=keep_set)

    def with_explanation(self, theta: np.ndarray) -> "Agreement":
        """Score using a fixed global explanation θ (must be in THIS behavior's basis)."""
        vec = np.asarray(theta, float).ravel()
        if vec.size != self._behavior.space.assignment_count():
            raise ValueError(
                f"theta has length {vec.size}, but this space needs {self._behavior.space.assignment_count()}"
            )
        if np.any(vec < 0):
            raise ValueError("theta must have non-negative values")
        s = vec.sum()
        if s <= 0:
            raise ValueError("theta must have positive mass")
        return replace(self, _fixed_expl_vec=vec / s)

    def by_context(self) -> "Agreement":
        """Return per-context agreement scores instead of overall score."""
        return replace(self, _mode_by_context=True)

    def overall(self) -> "Agreement":
        """Return overall score instead of per-context scores."""
        return replace(self, _mode_by_context=False)

    def keep_contexts(self, contexts: Iterable[ContextKey]) -> "Agreement":
        """Evaluate on a subset of contexts."""
        return replace(self, _keep=tuple(contexts), _drop=None)

    def drop_contexts(self, contexts: Iterable[ContextKey]) -> "Agreement":
        """Evaluate while excluding specific contexts."""
        return replace(self, _drop=tuple(contexts), _keep=None)

    # ---------- evaluation ----------
    @property
    def result(self) -> float:
        """
        Always returns a scalar agreement score.
        In by_context mode, returns the minimum score across contexts.
        """
        beh = self._filtered_behavior()

        # In by_context mode, always return min of context_scores
        if self._mode_by_context:
            return min(self.context_scores.values())

        # (A) Feature fixed: portable per-context scorer (no need to pass θ between spaces)
        if self._feature_name and self._feature_dist is not None:
            ctx_scores = self._agreement_given_feature(beh, self._feature_name, self._feature_dist)
            return min(ctx_scores.values()) if ctx_scores else 0.0

        # (B) θ fixed explicitly: score that θ against all contexts
        if self._fixed_expl_vec is not None:
            scores = beh.per_context_scores(mu=self._fixed_expl_vec)
            return min(scores) if scores.size > 0 else 0.0

        # (C) Minimax (α*) under optional weights
        weights = self._weights or self._uniform_weights(beh)
        if self._weights is None:
            # For minimax (no weights specified), use the dedicated minimax solver
            return float(beh.alpha_star)
        else:
            # For weighted agreement, use ConditionalSolver to avoid recursion
            from ..convex_models import ConditionalSolver
            opt_context = beh._to_context()
            solver = ConditionalSolver(opt_context)
            solution = solver.solve(weights)
            return float(solution.objective)

    @property
    def context_scores(self) -> Mapping[ContextKey, float]:
        """
        Per-context agreement scores under the current builder state.
        """
        beh = self._filtered_behavior()

        # (A) Feature fixed: portable per-context scorer (no need to pass θ between spaces)
        if self._feature_name and self._feature_dist is not None:
            return self._agreement_given_feature(beh, self._feature_name, self._feature_dist)

        # (B) θ fixed explicitly: score that θ against all contexts
        if self._fixed_expl_vec is not None:
            scores = beh.per_context_scores(mu=self._fixed_expl_vec)
            keys = [tuple(ctx.observables) for ctx in beh.context]
            return dict(zip(keys, map(float, scores)))

        # (C) Minimax (α*) under optional weights
        weights = self._weights or self._uniform_weights(beh)
        if self._weights is None:
            # For minimax, use the dedicated solver that gives optimal θ
            scores = beh.per_context_scores()  # Uses optimal θ by default
            keys = [tuple(ctx.observables) for ctx in beh.context]
            return dict(zip(keys, map(float, scores)))
        else:
            # For weighted agreement, use ConditionalSolver to get θ directly
            from ..convex_models import ConditionalSolver
            opt_context = beh._to_context()
            solver = ConditionalSolver(opt_context)
            solution = solver.solve(weights)
            theta = solution.weights
            scores = beh.per_context_scores(mu=theta)
            keys = [tuple(ctx.observables) for ctx in beh.context]
            return dict(zip(keys, map(float, scores)))

    @property
    def explanation(self) -> Optional[np.ndarray]:
        """
        The θ distribution used, or None if in fixed-feature mode.
        Returns the scenario probability vector over space.assignments().
        """
        beh = self._filtered_behavior()

        # No explanation in fixed-feature mode (no full θ implied)
        if self._feature_name and self._feature_dist is not None:
            return None

        # Choose θ (fixed if present; else θ*(λ))
        if self._fixed_expl_vec is not None:
            theta = self._fixed_expl_vec
        else:
            weights = self._weights or self._uniform_weights(beh)
            if self._weights is None:
                # For minimax, get θ from the dedicated solver
                _, theta, _, _ = beh._solve_alpha_star_with_mu()
            else:
                # For weighted, use ConditionalSolver to get θ directly
                from ..convex_models import ConditionalSolver
                opt_context = beh._to_context()
                solver = ConditionalSolver(opt_context)
                solution = solver.solve(weights)
                theta = solution.weights

        return np.asarray(theta, float).ravel()

    def scenarios(self) -> List[Tuple[Tuple[Any, ...], float]]:
        """
        Human-readable θ as (scenario, probability) pairs.
        Returns empty list if explanation is None.
        """
        theta = self.explanation
        if theta is None:
            return []

        beh = self._filtered_behavior()
        assignments = list(beh.space.assignments())
        return [(assignments[i], float(theta[i])) for i in range(len(assignments))]

    def feature_distribution(self, name: str) -> np.ndarray:
        """
        Marginal distribution for the given observable.
        If explanation exists, returns marginal computed from θ.
        If in fixed-feature mode and name matches fixed feature, returns the fixed distribution.
        """
        beh = self._filtered_behavior()

        # If in fixed-feature mode and this is the fixed feature, return the fixed distribution
        if self._feature_name and self._feature_dist is not None and name == self._feature_name:
            return self._feature_dist

        # Otherwise, compute from explanation (theta)
        theta = self.explanation
        if theta is None:
            raise ValueError(f"No explanation available for marginal distribution of '{name}'")

        # Find the observable index
        try:
            idx = beh.space.names.index(name)
        except ValueError:
            raise ValueError(f"Observable '{name}' not found in behavior space")

        # Compute marginal
        alphabet = list(beh.space.alphabets[name])
        marginal = np.zeros(len(alphabet), dtype=float)
        assignments = list(beh.space.assignments())

        for i, assignment in enumerate(assignments):
            symbol = assignment[idx]
            symbol_idx = alphabet.index(symbol)
            marginal[symbol_idx] += float(theta[i])

        return marginal

    def for_marginal(self, name: str) -> np.ndarray:
        """Alias for feature_distribution."""
        return self.feature_distribution(name)



    # ---------- helpers ----------
    @staticmethod
    def _uniform_weights(beh: "Behavior") -> Dict[ContextKey, float]:
        keys = [tuple(ctx.observables) for ctx in beh.context]
        if not keys:
            return {}
        w = 1.0 / len(keys)
        return {k: w for k in keys}

    @staticmethod
    def _agreement_given_feature(beh: "Behavior", observable: str, feature_distribution: np.ndarray) -> Dict[ContextKey, float]:
        """
        Per-context agreement when fixing one observable's distribution to `feature_distribution`.
        Uses the Bhattacharyya overlap between each context's P and the induced Q on that coordinate.
        """
        q = np.asarray(feature_distribution, dtype=float).ravel()
        if q.ndim != 1 or q.size == 0 or np.any(q < 0):
            raise ValueError("feature_distribution must be a non-negative 1-D probability vector")
        Z = q.sum()
        if Z <= 0:
            raise ValueError("feature_distribution must have positive mass")
        q = q / Z

        # Locate observable and its alphabet
        try:
            _ = beh.space.names.index(observable)  # sanity check it exists
        except ValueError:
            raise ValueError(f"Observable '{observable}' not found in this behavior's space")

        symbols = list(beh.space.alphabets[observable])
        sym_to_idx = {s: i for i, s in enumerate(symbols)}

        results: Dict[ContextKey, float] = {}
        for ctx in beh.context:
            if observable not in ctx.observables:
                continue
            j = ctx.observables.index(observable)
            g = 0.0
            dist = beh.distributions[ctx].to_dict()
            for outcome, p in dist.items():
                s = outcome[j]
                qi = q[sym_to_idx[s]]
                g += (float(p) ** 0.5) * (float(qi) ** 0.5)
            results[tuple(ctx.observables)] = float(g)
        return results

    def _filtered_behavior(self) -> "Behavior":
        """Apply keep/drop filters by building a sub-behavior when requested."""
        all_contexts = {tuple(ctx.observables) for ctx in self._behavior.context}

        # Start with all contexts
        filtered_contexts = all_contexts

        # Apply keep filter (intersection)
        if self._keep:
            keep_set = set(self._keep)
            filtered_contexts = filtered_contexts & keep_set

        # Apply drop filter (remove)
        if self._drop:
            drop_set = set(self._drop)
            filtered_contexts = filtered_contexts - drop_set

        # If no filtering applied, return original behavior
        if filtered_contexts == all_contexts:
            return self._behavior

        # Build filtered distributions
        dists = {}
        for ctx in self._behavior.context:
            key = tuple(ctx.observables)
            if key in filtered_contexts:
                dists[key] = self._behavior.distributions[ctx].to_dict()

        if not dists:
            raise ValueError("No contexts remain after filtering")

        # Lazy import to avoid cycles
        from .behavior import Behavior as _B
        return _B.from_contexts(self._behavior.space, dists)
