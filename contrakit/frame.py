"""
Frame Independence Analysis for the Mathematical Theory of Contradiction

This module provides tools for testing whether a behavioral pattern is frame-independent,
meaning all observations can be explained by a single underlying probability distribution
over the complete set of observables.

Key Concepts:
- Frame Independence: All observational contexts are consistent with a single underlying model
- Frame Dependence: Some observations contradict each other, requiring multiple models
- Residual: Quantitative measure of how well observations fit a single model

The frame independence check is fundamental to the theory as it determines whether
a behavioral pattern contains logical contradictions or can be explained consistently.
"""

from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Sequence, Set, Union, Optional
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import cvxpy as cp
from .agreement import BhattacharyyaCoefficient
from .convex_models import Solver

if TYPE_CHECKING:
    from .behavior.behavior import Behavior

@dataclass
class FIResult:
    """
    Result of a frame independence check.

    This class encapsulates the outcome of testing whether a behavioral pattern
    can be explained by a single underlying probability distribution.

    Attributes:
        is_fi: True if the behavior is frame-independent (no contradictions)
        residual: Quantitative measure of inconsistency (values near 0 indicate consistency)
        assignment_weights: Weights for each possible assignment of values to observables
        assignments: List of all possible complete assignments to the observable space
        solver_info: Information about which optimization solver was used

    Example:
        result = FrameIndependence.check(some_behavior)
        if result.is_fi:
            print(f"Behavior is consistent (residual: {result.residual:.6f})")
        else:
            print(f"Behavior contains contradictions (residual: {result.residual:.6f})")
    """
    is_fi: bool
    residual: float
    assignment_weights: np.ndarray
    assignments: List[Tuple[Any, ...]]
    solver_info: str


class FrameIndependence:
    """
    Utilities for testing frame independence of behavioral patterns.

    This class provides methods for determining whether a set of observations
    can be explained by a single underlying probability distribution. Frame
    independence is a key property in the mathematical theory of contradiction,
    indicating whether different observational contexts are logically consistent.

    The main method is `check()`, which performs a comprehensive test of
    frame independence using convex optimization techniques.
    """
    
    @staticmethod
    def build_constraints(behavior: 'Behavior') -> Tuple[np.ndarray, np.ndarray, List[Tuple[Any, ...]]]:
        """
        Build linear constraints for frame independence feasibility testing.

        This method constructs a system of linear equations that must be satisfied
        for a behavior to be frame-independent. The constraints ensure that the
        observed probabilities in different contexts are consistent with some
        underlying probability distribution over the complete observable space.

        Args:
            behavior: The behavioral pattern to test for frame independence

        Returns:
            A tuple containing:
            - Constraint matrix (coefficient matrix for linear constraints)
            - Right-hand side vector (target values for constraints)
            - List of all possible assignments to the observable space

        The constraint system A*x = b represents the requirement that observed
        probabilities must match what would be expected from a single underlying
        probability distribution.
        """
        assignments = behavior.space.assignments()
        n = behavior.space.assignment_count()

        constraint_rows = []
        rhs_values = []

        for context, distribution in behavior.distributions.items():
            for outcome in context.outcomes():
                # Build constraint row: sum over assignments that restrict to this outcome
                row = np.zeros(n)
                for j, assignment in enumerate(assignments):
                    if context.restrict_assignment(assignment) == outcome:
                        row[j] = 1.0
                
                constraint_rows.append(row)
                rhs_values.append(distribution[outcome])
        
        # Normalization constraint
        constraint_rows.append(np.ones(n))
        rhs_values.append(1.0)
        
        A_eq = np.vstack(constraint_rows)
        b_eq = np.array(rhs_values)
        
        return A_eq, b_eq, assignments
    
    @classmethod
    def check(cls, behavior: Behavior, tol: float = 1e-9) -> FIResult:
        A_eq, b_eq, assignments = cls.build_constraints(behavior)
        n = A_eq.shape[1]

        # --- Stage 1: try exact feasibility ---
        x = cp.Variable(n, nonneg=True)
        feas = cp.Problem(cp.Minimize(0), [A_eq @ x == b_eq])
        solver = Solver()
        solver.solve(feas)
        if feas.status in ("optimal", "optimal_inaccurate") and x.value is not None:
            xval = np.asarray(x.value).ravel()
            residual = float(np.linalg.norm(A_eq @ xval - b_eq))
            if residual <= tol:
                return FIResult(True, residual, xval, assignments, "feasible")

        # --- Stage 2: robust NNLS (minimize the worst constraint error) ---
        x2 = cp.Variable(n, nonneg=True)
        r  = cp.Variable(nonneg=True)  # max-abs residual (∞-norm)
        # Minimize ||A x - b||_∞ with nonnegativity (the 1-row normalization is inside A_eq)
        ls = cp.Problem(
            cp.Minimize(r),
            [A_eq @ x2 - b_eq <= r, b_eq - A_eq @ x2 <= r]
        )

        solver.solve(ls)
        if x2.value is not None and r.value is not None:
            xval = np.asarray(x2.value).ravel()
            residual = float(r.value)
            return FIResult(residual <= tol, residual, xval, assignments, "nnls")
        else:
            # final fallback if solver totally failed
            xval = np.zeros(n)
            residual = float(np.linalg.norm(A_eq @ xval - b_eq))
            return FIResult(False, residual, xval, assignments, "fallback:failed")





