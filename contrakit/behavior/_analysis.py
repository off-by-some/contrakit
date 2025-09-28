"""
Behavior Analysis and Optimization

This module provides the heavy computational routines for behavior analysis,
including sampling utilities, convex optimization programs for agreement
coefficient calculation, and contradiction measurement.
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Tuple, Sequence, Set, Union, Optional, Mapping, Iterable
from dataclasses import dataclass, field
import numpy as np
from ..agreement import BhattacharyyaCoefficient
from ..convex_models import Context as OptimizationContext, AlphaStar, ConditionalSolver, extract_lambdas_from_weights
from ..space import Space
from ..context import Context
from ..distribution import Distribution
from numba import njit


# Numba-accelerated sampling functions
@njit
def numba_sample_outcomes(cdfs, ctx_indices, n_samples, n_outcomes, random_vals):
    """
    Numba-accelerated outcome sampling using inverse CDF.
    """
    outcome_indices = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        ctx_idx = ctx_indices[i]
        r = random_vals[i]
        # Binary search in CDF
        left, right = 0, n_outcomes - 1
        while left <= right:
            mid = (left + right) // 2
            if cdfs[ctx_idx, mid] < r:
                left = mid + 1
            else:
                right = mid - 1
        # Clamp to valid range
        outcome_idx = left
        if outcome_idx >= n_outcomes:
            outcome_idx = n_outcomes - 1
        elif outcome_idx < 0:
            outcome_idx = 0
        outcome_indices[i] = outcome_idx

    return outcome_indices


@njit
def numba_count_samples(ctx_indices, outcome_indices, n_contexts, n_outcomes):
    """
    Numba-accelerated counting using direct array operations.
    """
    outcome_counts = np.zeros((n_contexts, n_outcomes), dtype=np.int32)
    context_counts = np.zeros(n_contexts, dtype=np.int32)

    for i in range(len(ctx_indices)):
        ctx_idx = ctx_indices[i]
        outcome_idx = outcome_indices[i]
        outcome_counts[ctx_idx, outcome_idx] += 1
        context_counts[ctx_idx] += 1

    return outcome_counts, context_counts


class BehaviorAnalysisMixin:
    """
    Mixin class providing analysis and optimization methods for Behavior.

    This mixin adds sampling utilities, agreement coefficient calculations,
    contradiction measurement, and fitting routines to the core Behavior class.
    """

    def sample_observations(self, n_samples: int, context_weights: Optional[np.ndarray] = None,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample observations from this behavior using fast numba-accelerated methods.

        This method efficiently samples large numbers of observations from the behavior's
        distributions across different contexts, which is useful for statistical analysis
        and compression experiments.

        Args:
            n_samples: Number of observations to sample
            context_weights: Weights for context selection [n_contexts].
                           If None, uses uniform weights.
            seed: Random seed for reproducibility

        Returns:
            ctx_indices: Context indices for each sample [n_samples]
            outcome_indices: Outcome indices for each sample [n_samples]

        Example:
            # Sample 1000 observations with uniform context weights
            ctx_indices, outcome_indices = behavior.sample_observations(1000)

            # Sample with custom weights favoring first context
            weights = np.array([0.8, 0.1, 0.1])  # 3 contexts
            ctx_indices, outcome_indices = behavior.sample_observations(1000, weights)
        """
        if not self.distributions:
            raise ValueError("Cannot sample from empty behavior")

        rng = np.random.default_rng(seed)
        contexts = list(self.distributions.keys())
        n_contexts = len(contexts)

        if context_weights is None:
            context_weights = np.ones(n_contexts) / n_contexts
        else:
            context_weights = np.array(context_weights, dtype=float)
            context_weights = context_weights / context_weights.sum()

        # Sample context indices
        ctx_indices = rng.choice(n_contexts, size=n_samples, p=context_weights)

        # Build CDFs for all contexts - handle variable outcome counts
        cdfs_list = []
        max_outcomes = 0

        for context in contexts:
            dist = self.distributions[context]
            outcomes = context.outcomes()
            probs = np.array([dist[outcome] for outcome in outcomes], dtype=np.float32)
            cdf = np.cumsum(probs)
            cdfs_list.append(cdf)
            max_outcomes = max(max_outcomes, len(outcomes))

        # Convert to numpy array with padding
        cdfs = np.zeros((n_contexts, max_outcomes), dtype=np.float32)
        for i, cdf in enumerate(cdfs_list):
            cdfs[i, :len(cdf)] = cdf

        # Sample outcomes using numba-accelerated method
        random_vals = rng.random(size=n_samples, dtype=np.float32)
        outcome_indices = numba_sample_outcomes(cdfs, ctx_indices, n_samples, max_outcomes, random_vals)

        return ctx_indices, outcome_indices

    def count_observations(self, ctx_indices: np.ndarray, outcome_indices: np.ndarray,
                          n_contexts: Optional[int] = None, n_outcomes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Count observations from sampled context and outcome indices.

        This method efficiently counts how many times each (context, outcome) pair was observed,
        which is useful for statistical analysis and compression experiments.

        Args:
            ctx_indices: Context indices for each sample [n_samples]
            outcome_indices: Outcome indices for each sample [n_samples]
            n_contexts: Number of contexts (optional, inferred from max ctx_indices if None)
            n_outcomes: Number of outcomes (optional, inferred from max outcome_indices if None)

        Returns:
            per_ctx_counts: Count matrix [n_contexts, n_outcomes]
            per_ctx_totals: Total counts per context [n_contexts]

        Example:
            # Sample observations
            ctx_indices, outcome_indices = behavior.sample_observations(1000)

            # Count them
            counts, totals = behavior.count_observations(ctx_indices, outcome_indices)

            # counts[i, j] = number of times outcome j was observed in context i
        """
        if n_contexts is None:
            n_contexts = int(np.max(ctx_indices)) + 1
        if n_outcomes is None:
            n_outcomes = int(np.max(outcome_indices)) + 1

        # Use numba-accelerated counting
        per_ctx_counts, per_ctx_totals = numba_count_samples(
            ctx_indices.astype(np.int32),
            outcome_indices.astype(np.int32),
            n_contexts, n_outcomes
        )

        return per_ctx_counts, per_ctx_totals

    def build_constraint_matrices(self):
        """Build constraint matrices for optimization problems."""
        assignments = self.space.assignments()
        n = self.space.assignment_count()

        constraint_rows = []
        rhs_values = []
        context_info = []

        for context, distribution in self.distributions.items():
            for outcome in context.outcomes():
                # Build constraint row
                row = np.zeros(n)
                for j, assignment in enumerate(assignments):
                    if context.restrict_assignment(assignment) == outcome:
                        row[j] = 1.0

                constraint_rows.append(row)
                rhs_values.append(distribution[outcome])
                context_info.append((context, outcome))

        # Normalization constraint
        constraint_rows.append(np.ones(n))
        rhs_values.append(1.0)
        context_info.append((None, "normalization"))

        A_eq = np.vstack(constraint_rows)
        b_eq = np.array(rhs_values)

        return A_eq, b_eq, assignments, context_info

    def _build_incidence_and_roots(self):
        assignments = self.space.assignments()
        n = self.space.assignment_count()
        mats = {}
        roots = {}
        ctxs = list(self.distributions.keys())
        for ctx in ctxs:
            dist = self.distributions[ctx]
            A = np.zeros((len(dist.outcomes), n))
            for j, a in enumerate(assignments):
                out = ctx.restrict_assignment(a)
                oi = dist._index.get(out)
                if oi is not None:
                    A[oi, j] = 1.0
            mats[ctx] = A
            roots[ctx] = np.sqrt(dist.to_array())
        return ctxs, mats, roots, assignments

    def _to_context(self) -> OptimizationContext:
        """Build OptimizationContext from this Behavior (stable ctx order).
        """
        n = self.space.assignment_count()
        name_to_idx = {nm: i for i, nm in enumerate(self.space.names)}

        # stable order by observable names
        ctx_objs_sorted = sorted(self.distributions.keys(), key=lambda c: tuple(c.observables))

        # Detect duplicate observable-sets with differing pmfs
        seen_by_obs = {}
        for ctx_obj in ctx_objs_sorted:
            key = tuple(ctx_obj.observables)
            outs = list(ctx_obj.outcomes())
            p = np.array([self.distributions[ctx_obj][o] for o in outs], float)

            if key not in seen_by_obs:
                seen_by_obs[key] = [(ctx_obj, outs, p)]
            else:
                # check if pmf differs from any already-seen pmf for same observables
                if not any(np.allclose(p, p_prev, atol=1e-12) for _, _, p_prev in seen_by_obs[key]):
                    raise ValueError(
                        "Multiple contexts with the same observable set but different distributions "
                        f"are not representable as-is: {key}. "
                        "Use Behavior.duplicate_context(target_context, count) to tag copies, "
                        "or add an explicit label observable (e.g., Witness ∈ {A,B,C})."
                    )
                seen_by_obs[key].append((ctx_obj, outs, p))

        # Group contexts by identical (observables, probability_distribution) patterns
        # This is a pure optimization - mathematically equivalent but faster
        unique_patterns = {}
        pattern_to_contexts = {}

        for ctx_obj in ctx_objs_sorted:
            key = tuple(ctx_obj.observables)
            outs = list(ctx_obj.outcomes())
            p = np.array([self.distributions[ctx_obj][o] for o in outs], float)
            
            # Create a hashable pattern key from observables and probabilities
            pattern_key = (key, tuple(p))
            
            if pattern_key not in unique_patterns:
                unique_patterns[pattern_key] = (ctx_obj, key, outs, p)
                pattern_to_contexts[pattern_key] = []
            
            pattern_to_contexts[pattern_key].append(ctx_obj)
        
        # Build optimization context using deduplicated patterns
        ctxs = []
        Ms, p_tabs = {}, {}
        
        for pattern_key, (representative_ctx, key, outs, p) in unique_patterns.items():
            ctxs.append(key)
            
            # Build incidence matrix once per unique pattern
            out_to_idx = {o: i for i, o in enumerate(outs)}
            idxs = [name_to_idx[o] for o in key]
            
            M = np.zeros((len(outs), n), float)
            # Use lazy generator - only materializes one assignment at a time
            for j, a in enumerate(self.space.assignments()):
                o = tuple(a[i] for i in idxs)
                M[out_to_idx[o], j] = 1.0
            
            Ms[key] = M
            p_tabs[key] = p
        
        
        return OptimizationContext(contexts=ctxs, matrices=Ms, probabilities=p_tabs, n_assignments=n)

    def _solve_alpha_star_with_mu(self, eps: float = 1e-12):
        # Initialize cache if not exists (for mixin compatibility)
        if not hasattr(self, '_alpha_cache'):
            self._alpha_cache = None

        if self._alpha_cache is not None:
            d = self._alpha_cache
            return d["alpha"], d["mu"], d["scores"], d["contexts"]

        context = self._to_context()
        solver = AlphaStar(context)
        solution = solver.solve()

        alpha_star = solution.objective
        mu_star = solution.weights
        lam_star = solution.lambdas

        # map tuple ctx -> Context object (same sorted order)
        tuple2ctx = {tuple(c.observables): c for c in self.distributions.keys()}
        ctx_objs_sorted = [tuple2ctx[c] for c in context.contexts]

        # per-context scores, same metric used in α*
        scores = np.array([
            float(context.sqrt_prob(c) @ np.sqrt(np.clip(context.matrix(c) @ mu_star, 1e-15, None)))
            for c in context.contexts
        ])

        self._alpha_cache = {
            "alpha": float(alpha_star),
            "mu": mu_star.copy(),
            "scores": scores,
            "contexts": ctx_objs_sorted,
            "lambda": lam_star,  # keys are tuple(ctx); keep if you want to expose it
        }
        return self._alpha_cache["alpha"], self._alpha_cache["mu"], self._alpha_cache["scores"], self._alpha_cache["contexts"]

    def per_context_scores(
        self,
        agreement=None,
        mu: Union[str, np.ndarray] = "optimal",
        eps: float = 1e-12
    ) -> np.ndarray:
        """
        Compute per-context agreement scores F(p_c, q_c) where q_c comes from μ.
        If `agreement` is None, defaults to Bhattacharyya coefficient.
        """
        ctxs, mats, roots, assignments = self._build_incidence_and_roots()
        n = self.space.assignment_count()

        if isinstance(mu, str):
            if mu == "optimal":
                # If using optimal μ and default agreement (Bhattacharyya), reuse cached scores
                if agreement is None:
                    _, mu_opt, scores_opt, _ = self._solve_alpha_star_with_mu(eps=eps)
                    return scores_opt
                # Otherwise compute optimal μ and then compute scores with custom agreement
                _, mu_opt, _, _ = self._solve_alpha_star_with_mu(eps=eps)
                mu = mu_opt
            elif mu == "uniform":
                mu = np.ones(n) / n
            else:
                raise ValueError("mu must be 'optimal', 'uniform', or a length-n array")
        else:
            mu = np.asarray(mu, dtype=float)
            if mu.shape != (n,):
                raise ValueError(f"mu must have shape ({n},)")

        # Default agreement = Bhattacharyya (specialized for α*)
        if agreement is None:
            bc = BhattacharyyaCoefficient()
            def F(p, q):
                return float(bc(p, q)) if callable(bc) else float(bc.score(p, q))
        else:
            def F(p, q):
                return float(agreement(p, q)) if callable(agreement) else float(agreement.score(p, q))

        scores = []
        for ctx in ctxs:
            dist = self.distributions[ctx]
            p = dist.to_array()
            q = mats[ctx] @ mu
            # Numeric guard
            q = np.clip(q, eps, 1.0)
            scores.append(F(p, q))

        return np.array(scores, dtype=float)

    # If you mutate self.distributions, call this to invalidate cache:
    def _invalidate_cache(self):
        self._alpha_cache = None

    def least_favorable_lambda(self):
        """Get the least favorable context mixing λ* (CVXPY dual variables)."""
        if self._alpha_cache is None:
            self._solve_alpha_star_with_mu()
        return self._alpha_cache.get("lambda", {})

    def alpha_given_lambda(self, lam_dict):
        """Compute α(λ) for given λ using CVXPY."""
        context = self._to_context()
        solver = ConditionalSolver(context)
        solution = solver.solve(lam_dict)
        # Return both the agreement score (objective) and the optimal global distribution (theta)
        return float(solution.objective), np.asarray(solution.weights, dtype=float)

    @property
    def alpha_star(self) -> float:
        alpha, *_ = self._solve_alpha_star_with_mu()
        return alpha

    @property
    def K(self) -> float:
        a = float(self.alpha_star)
        a = min(max(a, 1e-300), 1.0)   # clip to [tiny, 1]
        val = -np.log2(a)  # Use the already imported np
        return 0.0 if abs(val) < 5e-13 else float(val)

    def aggregate(self, aggregator, agreement=None, mu="optimal"):
        """Aggregate per-context agreement scores using an aggregator.

        Args:
            aggregator: Aggregator (callable object or function) to combine scores.
                       Can be an Aggregator subclass, numpy function, or any Callable[[np.ndarray], float].
            agreement: Agreement measure (default: Bhattacharyya coefficient).
            mu: Either "optimal" (default), "uniform", or custom array.

        Returns:
            float: Aggregated score.
        """
        scores = self.per_context_scores(agreement=agreement, mu=mu)
        return float(aggregator(scores))

    def check_consistency(self, atol: float = 1e-10) -> Dict[str, Any]:
        """
        Lightweight consistency report:
          - For any joint context, compare its implied marginals to any
            provided marginals over the same observables.
          - Returns a dict with 'ok' flag and mismatches (if any).
        """
        report = {"ok": True, "mismatches": []}

        # Index marginals provided by the user
        provided = {}  # key: tuple(obs) -> (context, pmf-vector in the order of context.outcomes())
        for ctx, dist in self.distributions.items():
            key = tuple(ctx.observables)
            provided[key] = (ctx, np.array([dist[o] for o in ctx.outcomes()], float))

        # For each joint, compute implied marginals and compare
        for ctx, dist in self.distributions.items():
            if len(ctx.observables) <= 1:
                continue  # nothing to marginalize

            # All nonempty proper subsets
            obs = ctx.observables
            outcomes = list(ctx.outcomes())
            probs = np.array([dist[o] for o in outcomes], float)

            for k in range(1, len(obs)):
                # iterate subsets
                # we can use itertools.combinations
                for subset in itertools.combinations(obs, k):
                    sub_ctx = Context.make(self.space, list(subset))
                    sub_outs = list(sub_ctx.outcomes())
                    # implied marginal over subset
                    implied = {o: 0.0 for o in sub_outs}

                    # Sum probs over outcomes of the joint that restrict to the subset outcome
                    idx_map = [obs.index(name) for name in subset]
                    for out, p in zip(outcomes, probs):
                        sub_out = tuple(out[i] for i in idx_map)
                        implied[sub_out] += p

                    implied_vec = np.array([implied[o] for o in sub_outs], float)

                    # If user provided this marginal, compare
                    key = tuple(subset)
                    if key in provided:
                        _, provided_vec = provided[key]
                        if not np.allclose(provided_vec, implied_vec, atol=atol):
                            report["ok"] = False
                            report["mismatches"].append({
                                "joint": tuple(obs),
                                "marginal": key,
                                "provided": provided_vec.tolist(),
                                "implied": implied_vec.tolist(),
                                "atol": atol,
                            })
        return report

    def fit_fi_to_empirical(self, hat_pc, mu_hat):
        """Fit FI model to empirical per-context distributions.

        Args:
            hat_pc: Empirical per-context distributions {context_key: {outcome: prob}}
            mu_hat: Empirical context weights {context_key: weight}

        Returns:
            Tuple of (alpha_hat, Q_star_emp) where Q_star_emp maps context keys to outcome distributions.
        """
        from ..convex_models import OptimizationContext, AlphaStar

        ctx_objs_sorted = sorted(self.distributions.keys(), key=lambda c: tuple(c.observables))
        ctxs = [tuple(c.observables) for c in ctx_objs_sorted]

        Ms, p_tabs = {}, {}
        for ctx_obj in ctx_objs_sorted:
            key = tuple(ctx_obj.observables)
            outs = list(ctx_obj.outcomes())
            out_to_idx = {o: i for i, o in enumerate(outs)}
            idxs = [self.space.names.index(o) for o in ctx_obj.observables]

            M = np.zeros((len(outs), self.space.assignment_count()), float)
            for j, a in enumerate(self.space.assignments()):
                o = tuple(a[i] for i in idxs)
                M[out_to_idx[o], j] = 1.0

            if key in hat_pc:
                p_vec = np.array([hat_pc[key].get(out, 0.0) for out in outs], float)
                p_vec = p_vec / p_vec.sum() if p_vec.sum() > 0 else p_vec
            else:
                p_vec = np.array([self.distributions[ctx_obj][out] for out in outs], float)

            Ms[key] = M
            p_tabs[key] = p_vec

        context = OptimizationContext(contexts=ctxs, matrices=Ms, probabilities=p_tabs, n_assignments=self.space.assignment_count())
        solver = AlphaStar(context)
        solution = solver.solve()
        alpha_hat = solution.objective
        theta_star = solution.weights

        Q_star_emp = {}
        for i, ctx_key in enumerate(ctxs):
            q_vec = Ms[ctx_key] @ theta_star
            q_vec = np.clip(q_vec, 1e-15, 1.0)
            q_vec = q_vec / q_vec.sum() if q_vec.sum() > 0 else q_vec

            ctx_obj = next(c for c in self.distributions.keys() if tuple(c.observables) == ctx_key)
            outcomes = list(ctx_obj.outcomes())
            Q_star_emp[ctx_key] = {outcomes[j]: float(q_vec[j]) for j in range(len(outcomes))}

        return alpha_hat, Q_star_emp
