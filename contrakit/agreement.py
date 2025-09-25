import numpy as np
from typing import  Dict
from abc import ABC, abstractmethod
from .constants import DEFAULT_SEED
from numba import njit


"""
Agreement Measures for Quantifying Observational Consistency

This module provides mathematical functions for measuring how well different
probability distributions agree with each other. Agreement measures quantify
the degree of similarity between two probability distributions, which is
essential for detecting contradictions across different observational contexts.

Key Concepts:
- Agreement Measure: A function F(p,q) that quantifies similarity between distributions p and q
- Bhattacharyya Coefficient: A specific agreement measure based on quantum state overlap
- Linear Overlap: Simple dot product of probability vectors
- Hellinger Affinity: Another measure based on the Hellinger distance

These measures form the foundation for computing overall agreement coefficients
and detecting inconsistencies in behavioral data.
"""

class AgreementMeasure(ABC):
    """
    Abstract base class for measuring agreement between probability distributions.

    Agreement measures quantify how similar two probability distributions are.
    They are essential for detecting whether different observational contexts
    are consistent with each other or contain contradictions.

    Subclasses must implement:
    - The agreement computation (__call__)
    - A human-readable name (name property)
    - Property validation (validate_properties)
    """
    
    @abstractmethod
    def __call__(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute agreement F(p,q) between two probability distributions."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this agreement measure."""
        pass
    
    def validate_properties(self, trials: int = 500, tolerance: float = 1e-10) -> Dict[str, bool]:
        """Validate mathematical properties of this agreement measure."""
        return self._test_properties(trials, tolerance)
    
    def _test_properties(self, trials: int, tolerance: float) -> Dict[str, bool]:
        """Default property testing - can be overridden by specific measures."""
        rng = np.random.default_rng(DEFAULT_SEED)
        
        properties = {
            "normalization": True,    # F(p,p) = 1
            "symmetry": True,         # F(p,q) = F(q,p)  
            "bounded": True,          # F(p,q) ∈ [0,1]
            "equality_condition": True # F(p,q) = 1 iff p = q
        }
        
        for _ in range(trials):
            dim = rng.integers(2, 8)
            p = self._random_pmf(dim, rng)
            q = self._random_pmf(dim, rng)
            
            # Test normalization: F(p,p) = 1
            if abs(self(p, p) - 1.0) > tolerance:
                properties["normalization"] = False
            
            # Test symmetry: F(p,q) = F(q,p)
            if abs(self(p, q) - self(q, p)) > tolerance:
                properties["symmetry"] = False
            
            # Test boundedness: F(p,q) ∈ [0,1]
            f_val = self(p, q)
            if f_val < -tolerance or f_val > 1.0 + tolerance:
                properties["bounded"] = False
            
            # Test equality condition
            if np.allclose(p, q, atol=1e-12):
                if abs(self(p, q) - 1.0) > tolerance:
                    properties["equality_condition"] = False
            else:
                if self(p, q) > 1.0 - tolerance:
                    properties["equality_condition"] = False
        
        return properties
    
    @staticmethod
    def _random_pmf(k: int, rng) -> np.ndarray:
        """Generate a random probability mass function."""
        x = rng.random(k)
        return x / x.sum()


class BhattacharyyaCoefficient(AgreementMeasure):
    """Bhattacharyya coefficient: F(p,q) = Σ √(p(o)q(o))"""

    def __call__(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Bhattacharyya coefficient between two probability distributions."""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Use the same numba implementation for consistency
        return self._compute_bhattacharyya_core(
            np.array([1.0]),        # single context weight
            p.reshape(1, -1),       # single context outcome probabilities
            q.reshape(1, -1)        # single context reference probabilities
        )

    @property
    def name(self) -> str:
        return "Bhattacharyya"

    @staticmethod
    @njit
    def _compute_bhattacharyya_core(context_weights, outcome_probs, q_probs):
        """
        Core numba-accelerated Bhattacharyya computation.

        Args:
            context_weights: array of weights for each context
            outcome_probs: 2D array [n_contexts, n_outcomes] of outcome probabilities
            q_probs: 2D array [n_contexts, n_outcomes] of reference probabilities
        """
        T = 0.0
        n_contexts, n_outcomes = outcome_probs.shape

        for c in range(n_contexts):
            bc = 0.0
            for o in range(n_outcomes):
                p = outcome_probs[c, o]
                q = q_probs[c, o]
                if p > 0.0 and q > 0.0:
                    # Clamp to avoid NaN from tiny negative epsilons
                    pq = max(p * q, 1e-12)
                    bc += np.sqrt(pq)
                # Zero contribution when either p or q is zero
            T += context_weights[c] * bc

        return T

    def _compute_bhattacharyya(self, context_weights, outcome_probs, q_probs):
        """Python wrapper for the core numba computation."""
        return self._compute_bhattacharyya_core(context_weights, outcome_probs, q_probs)

    @classmethod
    def from_counts(cls, context_weights, outcome_counts, outcome_totals, q_probs):
        """
        Compute Bhattacharyya coefficient from empirical count data.

        This method handles empirical count data by normalizing counts to probabilities,
        then computing the Bhattacharyya coefficient using the core numba implementation.

        Args:
            context_weights: array of weights for each context
            outcome_counts: 2D array [n_contexts, n_outcomes] of empirical counts
            outcome_totals: array of total counts per context
            q_probs: 2D array [n_contexts, n_outcomes] of reference probabilities

        Returns:
            float: Bhattacharyya coefficient value

        Example:
            >>> import numpy as np
            >>> bc = BhattacharyyaCoefficient()
            >>> counts = np.array([[50, 50], [60, 40]])
            >>> totals = np.array([100, 100])
            >>> weights = np.array([0.5, 0.5])
            >>> q_probs = np.array([[0.5, 0.5], [0.5, 0.5]])
            >>> result = bc.from_counts(weights, counts, totals, q_probs)
        """
        return cls._compute_bhattacharyya_from_counts(context_weights, outcome_counts, outcome_totals, q_probs)

    @staticmethod
    @njit
    def _compute_bhattacharyya_from_counts(context_weights, outcome_counts, outcome_totals, q_probs):
        """
        Core numba implementation for computing Bhattacharyya coefficient from counts.
        """
        # Convert counts to probabilities
        n_contexts, n_outcomes = outcome_counts.shape
        outcome_probs = np.zeros((n_contexts, n_outcomes))

        for c in range(n_contexts):
            if outcome_totals[c] > 0:
                for o in range(n_outcomes):
                    outcome_probs[c, o] = outcome_counts[c, o] / outcome_totals[c]

        # Use the core computation
        T = 0.0
        for c in range(n_contexts):
            bc = 0.0
            for o in range(n_outcomes):
                p = outcome_probs[c, o]
                q = q_probs[c, o]
                if p > 0.0 and q > 0.0:
                    # Clamp to avoid NaN from tiny negative epsilons
                    pq = max(p * q, 1e-12)
                    bc += np.sqrt(pq)
                # Zero contribution when either p or q is zero
            T += context_weights[c] * bc

        return T

    def _test_properties(self, trials: int, tolerance: float) -> Dict[str, bool]:
        """Test extended properties specific to Bhattacharyya coefficient."""
        properties = super()._test_properties(trials, tolerance)
        
        # Add specific tests for DPI and joint concavity
        properties.update({
            "data_processing_monotone": True,
            "joint_concavity": True,
            "product_multiplicative": True
        })
        
        rng = np.random.default_rng(DEFAULT_SEED)
        
        for _ in range(trials // 3):  # Fewer trials for expensive tests
            dim = rng.integers(2, 6)
            p = self._random_pmf(dim, rng)
            q = self._random_pmf(dim, rng)
            
            # Test data-processing monotonicity
            out_dim = rng.integers(2, 6)
            K = self._random_stochastic_matrix(out_dim, dim, rng)
            if self(K @ p, K @ q) + tolerance < self(p, q):
                properties["data_processing_monotone"] = False
            
            # Test joint concavity
            p1, q1 = self._random_pmf(dim, rng), self._random_pmf(dim, rng)
            p2, q2 = self._random_pmf(dim, rng), self._random_pmf(dim, rng)
            lam = rng.random()
            
            lhs = self(lam*p1 + (1-lam)*p2, lam*q1 + (1-lam)*q2)
            rhs = lam*self(p1, q1) + (1-lam)*self(p2, q2)
            if lhs + tolerance < rhs:
                properties["joint_concavity"] = False
            
            # Test product multiplicativity
            dim2 = rng.integers(2, 4)
            r, s = self._random_pmf(dim2, rng), self._random_pmf(dim2, rng)
            
            f_product = self(np.kron(p, r), np.kron(q, s))
            f_factors = self(p, q) * self(r, s)
            if abs(f_product - f_factors) > tolerance:
                properties["product_multiplicative"] = False
        
        return properties
    
    @staticmethod
    def _random_stochastic_matrix(m: int, n: int, rng) -> np.ndarray:
        """Generate a random column-stochastic matrix."""
        M = rng.random((m, n))
        return M / M.sum(axis=0, keepdims=True)


class LinearOverlap(AgreementMeasure):
    """Linear overlap: F(p,q) = Σ p(o)q(o)"""
    
    def __call__(self, p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * q))
    
    @property
    def name(self) -> str:
        return "Linear"


class HellingerAffinity(AgreementMeasure):
    """Hellinger affinity: F(p,q) = Σ √(p(o))√(q(o)) (same as Bhattacharyya)"""
    
    def __call__(self, p: np.ndarray, q: np.ndarray) -> float:
        return BhattacharyyaCoefficient()(p, q)
    
    @property
    def name(self) -> str:
        return "Hellinger"