# convex\_models — Agreement Coefficient Optimization Solvers

High-level convex programs for computing the **agreement coefficient** and related worst-case criteria over multi-context behaviors. This module assumes you’ve precomputed a **Context** object that encodes:

* the set of contexts $\mathcal{C}=\{c\}$,
* per-context probability vectors $p_c$,
* per-context incidence matrices $M_c$ mapping global assignments to context outcomes, and
* the number of global assignments $n$.

Only the **public API** is documented below.

---

## Mathematics (compact, exact)

Let $\theta\in\Delta^{n-1}$ be a global distribution over complete assignments. For each context $c$:

* predicted marginal: $q_c = M_c\,\theta \in \Delta(\mathcal{O}_c)$,
* agreement (Bhattacharyya overlap): $g_c(\theta) \;=\; \sum_{o\in\mathcal{O}_c}\sqrt{p_c(o)\,q_c(o)} \;=\; \langle \sqrt{p_c}, \sqrt{q_c}\rangle$.

### Alpha-star (max–min agreement)

$$
\alpha^\star \;=\; \max_{\theta\in\Delta^{n-1}}\; \min_{c\in\mathcal{C}} \; g_c(\theta).
$$

Equivalent minimax form (Sion’s theorem):

$$
\alpha^\star \;=\; \min_{\lambda\in\Delta(\mathcal{C})}\; \max_{\theta\in\Delta^{n-1}} \sum_{c}\lambda_c\, g_c(\theta).
$$

The **least-favorable $\lambda^\star$** identifies contexts that witness contradiction.

### Worst-case variance (importance sampling)

For each $c$, the (dimensionless) IS variance contribution with proposal $q_c$ is

$$
\mathrm{Var}_c = \sum_o \frac{p_c(o)^2}{q_c(o)} - 1,
$$

and we minimize the worst case $\max_c \mathrm{Var}_c$ over $\theta$.

### Worst-case KL divergence

For each $c$, the KL divergence from $p_c$ to $q_c$ is

$$
D_{\mathrm{KL}}(p_c\|q_c)=\sum_o p_c(o)\log \frac{p_c(o)}{q_c(o)}.
$$

We minimize $\max_c D_{\mathrm{KL}}(p_c\|q_c)$ over $\theta$. Results are reported in **bits**.

**Numerics.** All programs enforce $q_c \ge \varepsilon$ elementwise (with $\varepsilon=10^{-15}$) to stabilize $\sqrt{\cdot}$, $\log$, and $1/x$.

---

## Module constants

* `EPSILON: float = 1e-15`
  Lower bound used in constraints $q_c \ge \text{EPSILON}$.

* `DEFAULT_TOLERANCE: float = 1e-6`
  Tolerance used by helpers to detect active constraints.

---

## Data containers

### `class Solution(NamedTuple)`

Result bundle returned by all solvers.

* `objective: float`
  Scalar objective value (for `AlphaStar`: $\alpha^\star$; for variance/KL: worst-case value).

* `weights: np.ndarray`
  Optimal $\theta^\star$ over global assignments (shape `(n,)`, sums to 1).

* `lambdas: Dict[Tuple[str, ...], float]`
  Least-favorable context weights $\lambda$ (when applicable; empty otherwise). Keys are context tuples.

* `solver: str`
  The CVXPY solver actually used (“MOSEK”, “SCS”, …).

* `diagnostics: Dict[str, float]`
  Optional quality measures (see `AlphaStar` below).

---

### `class Context`

Precomputed, read-only bundle used by solvers.

```python
Context(
    contexts: List[Tuple[str, ...]],
    matrices: Dict[Tuple[str,...], np.ndarray],    # M_c : (m_c × n)
    probabilities: Dict[Tuple[str,...], np.ndarray],# p_c : (m_c,)
    n_assignments: int                              # n
)
```

**Methods (public):**

* `sqrt_prob(context) -> np.ndarray` returns $\sqrt{p_c}$.
* `matrix(context) -> np.ndarray` returns $M_c$.
* `prob(context) -> np.ndarray` returns $p_c$.

Shapes must satisfy `matrix(c) @ theta` → `(m_c,)` and entries of `prob(c)` nonnegative with sum 1.

---

## Solver selection

### `class Solver`

Environment-aware CVXPY launcher.

* Prefers the solver set in `CT_SOLVER` (“MOSEK” or “SCS”).
* Fallback order: MOSEK → SCS (or reversed if preferred is SCS).
* Tuned SCS parameters when used.

You won’t usually use `Solver` directly; all solvers below use it internally.

---

## Agreement (α\*) solver

### `class AlphaStar`

Compute $\alpha^\star$ and a witness $\lambda$.

```python
AlphaStar(context: Context)
```

#### `solve(method: str = "hypograph") -> Solution`

Compute $\alpha^\star$ with one of two equivalent formulations:

* `"hypograph"` (default):

  $$
  \max_{\theta\in\Delta,\, t}\; t
  \quad \text{s.t.}\quad
  t \le g_c(\theta)\ \forall c,\;\; (M_c\theta)\ge \varepsilon.
  $$

  Duals (when reliable) are used to extract $\lambda$.

* `"geometric"` (geometric-mean cone): introduces auxiliaries $r_{c,o}$ with
  $r_{c,o} \le \mathrm{geo\_mean}(p_c(o), q_c(o))$ and $\sum_o r_{c,o} \ge t$.

**Returns** `Solution` with:

* `objective = α*` in $[0,1]$.
* `weights = θ*`.
* `lambdas = λ*` (least-favorable context weights). When duals are noisy, a uniform distribution over the **active** constraints $\{c: g_c(\theta^\star)=\alpha^\star\}$ is returned.
* `diagnostics` containing:

  * `"primal_feasibility"`: $\max_c \max(0, \alpha^\star - g_c)$,
  * `"dual_feasibility"`: $\max_c \max(0, -\lambda_c)$,
  * `"complementarity"`: $\max_c |\lambda_c(\alpha^\star - g_c)|$,
  * `"objective_gap"`: $|\alpha^\star - \min_c g_c|$.

---

## Worst-case variance minimizer

### `class VarianceMinimizer`

Minimize the worst-case IS variance across contexts.

```python
VarianceMinimizer(context: Context)
```

#### `solve() -> Solution`

Program:

$$
\min_{\theta\in\Delta,\, t\ge 0}\; t
\quad \text{s.t.}\quad
\sum_o \frac{p_c(o)^2}{q_c(o)} - 1 \le t,\;\; q_c=M_c\theta \ge \varepsilon,\ \forall c.
$$

**Returns** `Solution` with:

* `objective = \max_c \mathrm{Var}_c` (nonnegative scalar),
* `weights = θ*`,
* `lambdas = {}` (not applicable),
* empty `diagnostics`.

---

## Worst-case KL minimizer

### `class KLDivergenceMinimizer`

Minimize the worst-case $D_{\mathrm{KL}}(p_c\|q_c)$ across contexts.

```python
KLDivergenceMinimizer(context: Context)
```

#### `solve() -> Solution`

CVXPY objective is in **nats**; the returned `objective` is recomputed in **bits** as

$$
\max_c \frac{1}{\log 2}\sum_o p_c(o)\big(\log p_c(o) - \log q_c(o)\big),
\quad q_c = M_c \theta \ge \varepsilon.
$$

**Returns** `Solution` with:

* `objective = \max_c D_{\mathrm{KL}}(p_c\|q_c)` in **bits**,
* `weights = θ*`,
* `lambdas = {}` (not applicable).

---

## Conditional α (fixed λ)

### `class ConditionalSolver`

Maximize agreement for a **fixed** context weighting $\lambda$.

```python
ConditionalSolver(context: Context)
```

#### `solve(lambda_dict: Dict[Tuple[str, ...], float]) -> Solution`

Program:

$$
\max_{\theta\in\Delta}\; \sum_c \lambda_c\, g_c(\theta)
\quad\text{s.t.}\quad q_c=M_c\theta \ge \varepsilon.
$$

**Returns** `Solution` with:

* `objective = \sum_c \lambda_c\, g_c(\theta^\star)` in $[0,1]$,
* `weights = θ*`,
* `lambdas = lambda_dict` (echoed).

---

## Utilities

### `extract_lambdas_from_weights(context: Context, weights: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> Dict[Tuple[str, ...], float]`

Recover a **witness** $\lambda$ from a candidate $\theta$ by detecting **active** contexts at tolerance:

$$
g_c(\theta)=\left\langle \sqrt{p_c}, \sqrt{M_c\theta}\right\rangle,\quad
\text{active}=\arg\min_c g_c(\theta).
$$

Returns a uniform distribution over the active set; zeros elsewhere. Useful when dual variables are unavailable.

---

## Usage example

```python
# Build Context from your modeling layer
C = Context(
    contexts=[("A",), ("B",), ("A","B")],
    matrices={
        ("A",):   M_A,            # shape (m_A, n)
        ("B",):   M_B,            # shape (m_B, n)
        ("A","B"):M_AB,           # shape (m_AB, n)
    },
    probabilities={
        ("A",):    p_A,           # shape (m_A,), sum=1
        ("B",):    p_B,           # shape (m_B,), sum=1
        ("A","B"): p_AB,          # shape (m_AB,), sum=1
    },
    n_assignments=n
)

# Alpha-star
alpha = AlphaStar(C).solve()
alpha.objective       # α* in [0,1]
alpha.weights.shape   # (n,)
alpha.lambdas         # least-favorable λ*

# Fixed-λ agreement
lam = {("A",): 0.5, ("B",): 0.5, ("A","B"): 0.0}
cond = ConditionalSolver(C).solve(lam)

# Robustness alternatives
vmin = VarianceMinimizer(C).solve()         # minimize worst-case variance
kmin = KLDivergenceMinimizer(C).solve()     # minimize worst-case KL (bits)
```

---

## Notes & tips

* **Feasibility guards:** Every solver enforces $q_c \ge \text{EPSILON}$ to avoid domain errors in `sqrt`, `log`, `inv_pos`.
* **Scaling:** Large $n$ (many assignments) will dominate runtime/memory; consider sparsity in $M_c$ if applicable.
* **Solver choice:** Set `CT_SOLVER=MOSEK` for speed/robustness if MOSEK is available; otherwise SCS is used with tuned settings.
* **Witness reliability:** Duals can be noisy on some solvers/problems; `AlphaStar` falls back to an **active-set** $\lambda$. Use `extract_lambdas_from_weights` explicitly when you have a candidate $\theta$.
