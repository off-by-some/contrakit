# `convex_models.py`

*Convex optimization solvers for agreement coefficients and contradiction measures.*

---

## Overview

The `convex_models.py` module provides high-level convex optimization programs for computing agreement coefficients and related worst-case criteria over multi-context behaviors. It implements the mathematical core of the contradiction framework, solving optimization problems that quantify how well different observational perspectives can be reconciled.

The module uses CVXPY to solve constrained optimization problems that find optimal global distributions and witness contexts for contradiction analysis.

---

## Why This Module Exists

Multi-perspective behavioral analysis requires solving complex optimization problems to quantify contradiction. Rather than implementing these mathematical programs repeatedly across different use cases, this module provides:

* Standardized solvers for agreement coefficient computation
* Multiple optimization formulations for robustness
* Environment-aware solver selection with fallbacks
* Proper numerical stabilization for reliable results

The module separates the mathematical optimization logic from higher-level APIs, allowing the core algorithms to evolve independently while maintaining consistent interfaces.

---

## Architecture

The module implements a solver-based architecture with clear separation of concerns:

- **Data structures**: `Context` and `Solution` classes for problem representation and results
- **Solver infrastructure**: `Solver` class with environment-aware CVXPY solver selection
- **Optimization programs**: Specialized solver classes for different contradiction measures
- **Numerical stabilization**: Epsilon constraints to prevent domain errors in logarithms and square roots

**Data Flow**:
- Input: Precomputed context data (matrices, probabilities, assignments)
- Processing: CVXPY optimization with solver selection and fallback
- Output: Optimal distributions, witness weights, and diagnostic information

**Dependencies**:
- `cvxpy`: Convex optimization framework
- External solvers: MOSEK (preferred) or SCS (fallback)

---

## Key Classes and Functions

### `class Context`

Precomputed context data container for efficient optimization.

**Attributes**:

* `contexts` *(List[Tuple[str, ...]])* — List of context specifications
* `matrices` *(Dict[Tuple[str,...], np.ndarray])* — Incidence matrices M_c mapping assignments to outcomes
* `probabilities` *(Dict[Tuple[str,...], np.ndarray])* — Probability distributions p_c per context
* `n_assignments` *(int)* — Number of global assignments

### `class Solution(NamedTuple)`

Optimization result container returned by all solvers.

**Attributes**:

* `objective` *(float)* — Scalar objective value (α*, variance, or KL divergence)
* `weights` *(np.ndarray)* — Optimal global distribution θ* over assignments
* `lambdas` *(Dict[Tuple[str, ...], float])* — Context weights λ* (when applicable)
* `solver` *(str)* — CVXPY solver used ("MOSEK", "SCS", etc.)
* `diagnostics` *(Dict[str, float])* — Quality measures and convergence diagnostics

### `class AlphaStar`

Solver for computing the agreement coefficient α* and witness weights.

**Parameters**:

* `context` *(Context)* — Precomputed context data

**Methods**:

* `solve(method="hypograph") -> Solution` — Compute α* using specified formulation

**Example**:

```python
solver = AlphaStar(context)
result = solver.solve()
print(f"Agreement: {result.objective:.6f}")
print(f"Witness weights: {result.lambdas}")
```

### `class VarianceMinimizer`

Solver for minimizing worst-case importance sampling variance.

**Parameters**:

* `context` *(Context)* — Precomputed context data

**Methods**:

* `solve() -> Solution` — Find optimal distribution minimizing max variance

### `class KLDivergenceMinimizer`

Solver for minimizing worst-case KL divergence.

**Parameters**:

* `context` *(Context)* — Precomputed context data

**Methods**:

* `solve() -> Solution` — Find optimal distribution minimizing max KL divergence (in bits)

### `class ConditionalSolver`

Solver for maximizing agreement under fixed context weights.

**Parameters**:

* `context` *(Context)* — Precomputed context data

**Methods**:

* `solve(lambda_dict) -> Solution` — Compute optimal agreement for given λ

### `extract_lambdas_from_weights(context, weights, tolerance=1e-6) -> Dict`

Extract witness weights from optimal distribution by identifying active constraints.

**Parameters**:

* `context` *(Context)* — Context data
* `weights` *(np.ndarray)* — Candidate optimal distribution
* `tolerance` *(float)* — Threshold for active constraint detection

**Returns**:

* *(Dict[Tuple[str, ...], float])* — Uniform distribution over active contexts

---

## Drawbacks or Gotchas

* **Solver dependencies**: Requires CVXPY and either MOSEK or SCS solver
* **Computational complexity**: Optimization scales with number of global assignments
* **Numerical stability**: Epsilon constraints may affect precision near boundaries
* **Solver reliability**: Dual extraction can be noisy on some solvers/platforms
* **Memory usage**: Large context matrices increase memory requirements

---

## Related Modules

* [`behavior.py`](../behavior.md) — Multi-perspective behavioral analysis
* [`space.py`](../space.md) — Observable space definitions
* [`context.py`](../context.md) — Context representation

---

## See Also

* [Mathematical Theory Paper](../../docs/paper/A%20Mathematical%20Theory%20of%20Contradiction.pdf) — Formal foundations of optimization problems
* [Quickstart Examples](../../../examples/) — Usage patterns and examples
* [API Reference](../../docs/api/) — Complete reference documentation

---

```markdown
<!--
This file documents the public API of convex_models.py.
For internal implementation details, see the source code.
Last updated 2025-09.
-->
```
