"""Agreement coefficient optimization solvers."""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
import cvxpy as cp
import os

# Numerical constants
EPSILON = 1e-15
DEFAULT_TOLERANCE = 1e-6


class Solution(NamedTuple):
    """Optimization solution container."""
    objective: float
    weights: np.ndarray
    lambdas: Dict[Tuple[str, ...], float]
    solver: str
    diagnostics: Dict[str, float]


class Context:
    """Precomputed context data for efficient solving."""

    def __init__(self, contexts: List[Tuple[str, ...]], matrices: Dict,
                 probabilities: Dict, n_assignments: int):
        self.contexts = contexts
        self.matrices = matrices
        self.probabilities = probabilities
        self.n_assignments = n_assignments
        self._sqrt_probs = {c: np.sqrt(np.maximum(p, 0.0))
                           for c, p in probabilities.items()}

    def sqrt_prob(self, context: Tuple[str, ...]) -> np.ndarray:
        return self._sqrt_probs[context]

    def matrix(self, context: Tuple[str, ...]) -> np.ndarray:
        return self.matrices[context]

    def prob(self, context: Tuple[str, ...]) -> np.ndarray:
        return self.probabilities[context]


class Solver:
    """Base optimization solver with environment-aware solver selection."""

    def __init__(self):
        self._preferred_solver = os.getenv("CT_SOLVER", "").upper()

    def solve(self, problem: cp.Problem) -> str:
        """Solve problem using available solvers with fallback."""
        solvers = self._get_solver_order()

        for solver in solvers:
            if solver is None:
                continue

            try:
                if solver == cp.SCS:
                    problem.solve(
                        solver=solver,
                        eps=1e-11,
                        max_iters=200_000,
                        acceleration_lookback=20,
                        acceleration_interval=5,
                        verbose=False
                    )
                else:
                    problem.solve(solver=solver, verbose=False)

                return problem.solver_stats.solver_name if problem.solver_stats else "Unknown"

            except Exception:
                continue

        # Final fallback
        problem.solve(solver=cp.SCS, eps=1e-9, verbose=False)
        return "SCS"

    def _get_solver_order(self) -> List[Optional[cp.Solver]]:
        """Get solver order based on environment preference."""
        mosek = getattr(cp, "MOSEK", None)
        scs = cp.SCS

        if self._preferred_solver == "MOSEK":
            return [mosek, scs]
        elif self._preferred_solver == "SCS":
            return [scs, mosek]
        else:
            return [mosek, scs]


class AlphaStar:
    """Alpha-star coefficient solver."""

    def __init__(self, context: Context):
        self.context = context
        self.solver = Solver()

    def solve(self, method: str = "hypograph") -> Solution:
        """Compute alpha-star using specified method."""
        if method == "hypograph":
            return self._solve_hypograph()
        elif method == "geometric":
            return self._solve_geometric()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_hypograph(self) -> Solution:
        """Solve using hypograph formulation."""
        theta = cp.Variable(self.context.n_assignments, nonneg=True)
        t = cp.Variable()

        constraints = [
            cp.sum(theta) == 1,
            theta >= 0,
            0 <= t,
            t <= 1
        ]

        # Add context constraints
        inequality_constraints = []
        for c in self.context.contexts:
            q = self.context.matrix(c) @ theta
            constraints.append(q >= EPSILON)

            g = self.context.sqrt_prob(c) @ cp.sqrt(q)
            inequality_constraints.append(t - g <= 0)

        problem = cp.Problem(cp.Maximize(t), constraints + inequality_constraints)
        solver_name = self.solver.solve(problem)

        return self._extract_solution(theta, problem, solver_name, inequality_constraints)

    def _solve_geometric(self) -> Solution:
        """Solve using geometric mean cone formulation."""
        theta = cp.Variable(self.context.n_assignments, nonneg=True)
        t = cp.Variable()

        constraints = [
            cp.sum(theta) == 1,
            t >= 0
        ]

        r_vars = {}
        for c in self.context.contexts:
            p = self.context.prob(c)
            M = self.context.matrix(c)
            q = M @ theta
            constraints.append(q >= EPSILON)

            m = p.shape[0]
            r = cp.Variable(m, nonneg=True)
            r_vars[c] = r
            constraints.append(cp.sum(r) >= t)

            for o in range(m):
                constraints.append(r[o] <= cp.geo_mean(cp.hstack([p[o], q[o]])))

        problem = cp.Problem(cp.Maximize(t), constraints)
        solver_name = self.solver.solve(problem)

        return self._extract_solution(theta, problem, solver_name)

    def _extract_solution(self, theta_var: cp.Variable, problem: cp.Problem,
                         solver_name: str, dual_constraints: Optional[List] = None) -> Solution:
        """Extract solution components from solved problem."""
        alpha_star = 0.0
        lam_dict = {}
        diagnostics = {}
        theta_star = np.asarray(theta_var.value).ravel()

        # Compute objective values for all contexts
        g_vals = np.array([
            float(self.context.sqrt_prob(c) @ np.sqrt(
                np.maximum(self.context.matrix(c) @ theta_star, EPSILON)
            ))
            for c in self.context.contexts
        ])

        if g_vals.size == 0:
            # No feasible context → total contradiction
            return Solution(
                objective=alpha_star,
                weights=theta_star,
                lambdas=lam_dict,
                solver=solver_name,
                diagnostics=diagnostics
            )

        alpha_star = float(np.clip(g_vals.min(), 0.0, 1.0))

        # Extract lambda coefficients
        if dual_constraints:
            lam_vec = self._extract_dual_lambdas(dual_constraints, g_vals, alpha_star)
        else:
            lam_vec = self._compute_active_lambdas(g_vals, alpha_star)

        lam_dict = {c: float(lam_vec[i]) for i, c in enumerate(self.context.contexts)}

        # Compute diagnostics
        diagnostics = self._compute_diagnostics(theta_star, lam_dict, alpha_star)

        return Solution(
            objective=alpha_star,
            weights=theta_star,
            lambdas=lam_dict,
            solver=solver_name,
            diagnostics=diagnostics
        )

    def _extract_dual_lambdas(self, dual_constraints: List, g_vals: np.ndarray,
                             alpha_star: float) -> np.ndarray:
        """Extract lambda from dual variables."""
        lam_vec = np.array([con.dual_value for con in dual_constraints], float)
        lam_vec = np.clip(np.nan_to_num(lam_vec, nan=0.0), 0.0, None)

        if lam_vec.sum() > 0:
            lam_vec /= lam_vec.sum()

        # Fallback to active set if dual extraction is unreliable
        tolerance = max(1e-8, 1e-7 * alpha_star)
        active = np.where(np.abs(g_vals - alpha_star) <= tolerance)[0]

        if active.size and active.size != len(self.context.contexts):
            lam_vec = np.zeros_like(lam_vec)
            lam_vec[active] = 1.0 / active.size

        return lam_vec

    def _compute_active_lambdas(self, g_vals: np.ndarray, alpha_star: float) -> np.ndarray:
        """Compute lambda using uniform distribution over active constraints."""
        tolerance = max(1e-8, 1e-7 * alpha_star)
        lam_vec = np.zeros(len(self.context.contexts), float)
        active = np.where(np.abs(g_vals - alpha_star) <= tolerance)[0]

        if active.size:
            lam_vec[active] = 1.0 / active.size

        return lam_vec

    def _compute_diagnostics(self, theta_star: np.ndarray,
                           lam_dict: Dict[Tuple[str, ...], float],
                           alpha_star: float) -> Dict[str, float]:
        """Compute solution quality diagnostics."""
        g_vals = []
        for c in self.context.contexts:
            q = self.context.matrix(c) @ theta_star
            q = np.maximum(q, EPSILON)
            g_vals.append(float(self.context.sqrt_prob(c) @ np.sqrt(q)))

        g_vals = np.asarray(g_vals, float)
        lam_vec = np.array([lam_dict.get(c, 0.0) for c in self.context.contexts], float)

        return {
            "primal_feasibility": float(np.max(np.maximum(0.0, alpha_star - g_vals))),
            "dual_feasibility": float(np.max(np.maximum(0.0, -lam_vec))),
            "complementarity": float(np.max(np.abs(lam_vec * (alpha_star - g_vals)))),
            "objective_gap": float(abs(alpha_star - g_vals.min()))
        }


class VarianceMinimizer:
    """Worst-case importance sampling variance minimizer."""

    def __init__(self, context: Context):
        self.context = context
        self.solver = Solver()

    def solve(self) -> Solution:
        """Minimize worst-case variance."""
        theta = cp.Variable(self.context.n_assignments, nonneg=True)
        t = cp.Variable()

        constraints = [
            cp.sum(theta) == 1,
            theta >= 0,
            t >= 0
        ]

        for c in self.context.contexts:
            q = self.context.matrix(c) @ theta
            constraints.append(q >= EPSILON)

            variance = cp.sum(cp.multiply(self.context.prob(c)**2, cp.inv_pos(q))) - 1
            constraints.append(variance <= t)

        problem = cp.Problem(cp.Minimize(t), constraints)
        solver_name = self.solver.solve(problem)

        theta_star = np.asarray(theta.value).ravel()

        return Solution(
            objective=float(t.value),
            weights=theta_star,
            lambdas={},  # Not applicable for this problem
            solver=solver_name,
            diagnostics={}
        )


class KLDivergenceMinimizer:
    """Worst-case KL divergence minimizer."""

    def __init__(self, context: Context):
        self.context = context
        self.solver = Solver()

    def solve(self) -> Solution:
        """Minimize worst-case KL divergence (in bits)."""
        theta = cp.Variable(self.context.n_assignments, nonneg=True)
        t_nats = cp.Variable()

        constraints = [
            cp.sum(theta) == 1,
            theta >= 0,
            t_nats >= 0
        ]

        for c in self.context.contexts:
            q = self.context.matrix(c) @ theta
            constraints.append(q >= EPSILON)

            kl_term = cp.sum(cp.rel_entr(self.context.prob(c), q))
            constraints.append(kl_term <= t_nats)

        problem = cp.Problem(cp.Minimize(t_nats), constraints)
        solver_name = self.solver.solve(problem)

        # Convert to bits using feasible computation
        theta_star = np.asarray(theta.value).ravel()
        kl_bits_values = []

        for c in self.context.contexts:
            p = np.clip(self.context.prob(c), 1e-300, None)
            q = np.clip(self.context.matrix(c) @ theta_star, 1e-300, None)
            kl_bits = np.sum(p * (np.log(p) - np.log(q))) / np.log(2.0)
            kl_bits_values.append(float(kl_bits))

        return Solution(
            objective=float(max(kl_bits_values)),
            weights=theta_star,
            lambdas={},
            solver=solver_name,
            diagnostics={}
        )


class ConditionalSolver:
    """Solve for optimal alpha given fixed lambda distribution."""

    def __init__(self, context: Context):
        self.context = context
        self.solver = Solver()

    def solve(self, lambda_dict: Dict[Tuple[str, ...], float]) -> Solution:
        """Compute optimal alpha and theta for fixed lambda."""
        theta = cp.Variable(self.context.n_assignments, nonneg=True)

        constraints = [
            cp.sum(theta) == 1,
            theta >= 0
        ]

        lam_vec = np.array([lambda_dict.get(c, 0.0) for c in self.context.contexts], float)
        obj_terms = []

        for weight, context in zip(lam_vec, self.context.contexts):
            if weight == 0:
                continue

            q = self.context.matrix(context) @ theta
            constraints.append(q >= EPSILON)
            obj_terms.append(weight * (self.context.sqrt_prob(context) @ cp.sqrt(q)))

        problem = cp.Problem(cp.Maximize(cp.sum(obj_terms)), constraints)
        solver_name = self.solver.solve(problem)

        return Solution(
            objective=float(problem.value),
            weights=np.asarray(theta.value).ravel(),
            lambdas=lambda_dict,
            solver=solver_name,
            diagnostics={}
        )


def extract_lambdas_from_weights(context: Context, weights: np.ndarray,
                                tolerance: float = DEFAULT_TOLERANCE) -> Dict[Tuple[str, ...], float]:
    """Extract lambda coefficients from optimal weights by identifying active constraints."""
    g_values = {
        c: float(context.sqrt_prob(c) @ np.sqrt(
            np.maximum(context.matrix(c) @ weights, EPSILON)
        ))
        for c in context.contexts
    }

    min_g = min(g_values.values())
    active_contexts = [c for c, v in g_values.items() if v - min_g <= tolerance]
    weight = 1.0 / len(active_contexts) if active_contexts else 0.0

    return {c: (weight if c in active_contexts else 0.0) for c in context.contexts}

