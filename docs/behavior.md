# Behavior — A Practical API for Multi-Perspective Behavioral Analysis

The `Behavior` class is the central façade for representing and analyzing **distributions across observational contexts** (a.k.a. “perspectives”). It unifies three layers:

* **Representation & algebra** (composition, renaming, coarse-graining, etc.)
* **Analysis & optimization** (agreement coefficient α\*, contradiction bits K, per-context scores, worst-case weights)
* **Sampling utilities** (fast Numba-accelerated experiment generation and counting)

Use `Behavior` when you need to ask: *“Could these context-wise observations have come from a single coherent underlying model?”* If yes, the behavior is **frame-independent** (FI). If not, we quantify **how contradictory** it is.

---

## TL;DR / Quickstart

```python
from contrakit import Space, Behavior

# 1) Define the observable universe
space = Space.create(Morning=["Sun","Rain"], Evening=["Sun","Rain"])

# 2) Provide context-wise distributions
behavior = Behavior.from_contexts(space, {
    ("Morning",): {("Sun",): 0.8, ("Rain",): 0.2},
    ("Evening",): {("Sun",): 0.8, ("Rain",): 0.2},
    ("Morning","Evening"): {("Sun","Sun"): 0.5, ("Rain","Rain"): 0.5},  # intentionally incompatible
})

# 3) Check agreement & contradiction
print("α* =", behavior.agreement)            # optimal agreement coefficient in [0,1]
print("K bits =", behavior.contradiction_bits)  # = -log2(α*)

# 4) Inspect worst-case perspective weighting (λ*)
print("worst-case weights:", behavior.worst_case_weights)

# 5) Ask from your point of view (custom λ)
weights = {("Morning",): 0.6, ("Evening",): 0.3, ("Morning","Evening"): 0.1}
res = behavior.agreement_for_weights(weights)
print(res)                 # AgreementResult(score=..., theta_shape=(num_global_assignments,))
print(res.scenarios()[:5]) # top global scenarios θ*(λ) with probabilities

# 6) Simulate observations quickly
ctx_idx, out_idx = behavior.sample_observations(50_000, seed=7)
counts, totals = behavior.count_observations(ctx_idx, out_idx)
```

---

## Core Concepts (1 minute mental model)

* **Observables**: named variables with finite alphabets.
* **Contexts**: which observables were measured together (e.g., `("Morning","Evening")`).
* **Behavior**: a dict of per-context PMFs `p_c(o)`.
* **Frame Independence (FI)**: exists a single global law μ over all observables whose marginals match *every* context.
* **Agreement α\***: best-possible Bhattacharyya overlap between your behavior and any FI behavior, **guarded by the worst context**.
* **Contradiction bits K**: `K = -log2(α*)`. Higher = stronger structural clash between perspectives.

---

## Public API

### Construction

```python
Behavior.from_contexts(space, context_dists: dict) -> Behavior
```

Define per-context PMFs explicitly. Validates outcomes exactly against each context’s outcome set; fills missing outcomes with 0 (stable ordering).

```python
Behavior.frame_independent(space, contexts: Sequence[Sequence[str]], assignment_weights: Optional[np.ndarray]=None) -> Behavior
```

Generate an **FI** behavior from a global assignment distribution (uniform by default) and return each context’s marginals.

```python
Behavior.random(space, contexts, alpha: float=1.0, seed: Optional[int]=None) -> Behavior
```

Dirichlet-random PMFs per context.

```python
Behavior.from_mu(space, contexts, mu: np.ndarray, normalize: bool=True) -> Behavior
```

Build per-context marginals from a provided global μ (length must equal `space.assignment_count()`).

```python
Behavior.from_counts(space, context_tables, normalize="per_context"|"global"|"none") -> Behavior
```

Create from raw counts. `normalize="per_context"` (default) turns each table into a PMF; `"global"` uses one Z; `"none"` assumes probabilities.

---

### Properties

```python
behavior.agreement           # α* ∈ [0,1]
behavior.contradiction_bits  # K = -log2(α*) ≥ 0
behavior.context             # list[Context], as stored
```

---

### Frame Independence & Comparison

```python
behavior.is_frame_independent(tol: float = 1e-9) -> bool
```

True iff the behavior sits in FI (equivalently α\* = 1).

```python
behavior.agreement_with(other: Behavior, weights: Optional[dict] = None)
```

Compare by *merging* distributions (no overlapping contexts allowed).

* If `weights is None`: returns α\* for the merged behavior.
* If `weights` given: returns `AgreementResult` under your weighting.

---

### Worst-Case & Custom Weights

```python
behavior.worst_case_weights  # dict[ContextKey, float]
```

The least-favorable λ\* (dual solution): which contexts “witness” the contradiction.

```python
behavior.agreement_for_weights(perspective_weights: dict) -> AgreementResult
```

Your perspective, your λ (auto-normalized). Returns:

* `score`: α(λ)
* `theta`: θ\*(λ), the optimal global law explaining the data under your weights
* `space`: to interpret θ’s assignment order

```python
AgreementResult.scenarios() -> list[ (assignment_tuple, prob) ]
```

Map θ entries back to concrete global assignments.

---

### Per-Context Analysis & Aggregation

```python
behavior.per_context_scores(agreement=None, mu="optimal"|"uniform"|np.ndarray) -> np.ndarray
```

Compute `F(p_c, q_c)` for each context:

* `agreement=None` uses the Bhattacharyya coefficient (same objective as α\*).
* `mu="optimal"` reuses the cached α\* optimizer; `"uniform"` uses 1/n; or pass a custom global μ.

```python
behavior.aggregate(aggregator, agreement=None, mu="optimal") -> float
```

Fold the vector of per-context scores via your aggregator (mean, median, min, custom callable, …).

---

### Sampling & Counting (Numba-accelerated)

```python
behavior.sample_observations(n_samples: int,
                             context_weights: Optional[np.ndarray]=None,
                             seed: Optional[int]=None) -> (ctx_indices, outcome_indices)
```

Efficiently sample large synthetic datasets across contexts using inverse-CDF with a vectorized Numba kernel. `context_weights` defaults to uniform over measured contexts.

```python
behavior.count_observations(ctx_indices: np.ndarray,
                            outcome_indices: np.ndarray,
                            n_contexts: Optional[int]=None,
                            n_outcomes: Optional[int]=None)
   -> (per_ctx_counts: np.ndarray[nC,nO], per_ctx_totals: np.ndarray[nC])
```

Fast Numba tally of (context, outcome) frequencies.

---

### Algebra & Transformations

```python
behavior @ other                       # tensor product (disjoint spaces only)
behavior + other                       # equal convex combination
behavior.mix(other, weight: float)     # convex combine with weight ∈ [0,1]
behavior.rename_observables(name_map: dict) -> Behavior
behavior.permute_outcomes(value_maps: dict[name -> bijection]) -> Behavior
behavior.coarse_grain(observable: str, merge_map: dict) -> Behavior
behavior.drop_contexts(keep_contexts: set[tuple[str,...]]) -> Behavior
behavior.duplicate_context(target_context: tuple[str,...], count: int, tag_prefix: str="Tag") -> Behavior
behavior.product_l1_distance(other) -> float     # max L1 over shared contexts
```

---

## Numerical & Performance Notes

* **Caching**: The α\* solver and its μ\* (θ\*) are cached. **If you mutate distributions**, call:

  ```python
  behavior._invalidate_cache()
  ```

  (Constructors that return a *new* `Behavior` don’t share cache.)
* **Stability**: Probability vectors inside optimizations are clipped to `[1e-15, 1]` where needed and renormalized for safety; Bhattacharyya is robust under this guard.
* **Complexity drivers**:

  * `space.assignment_count()` (size of μ/θ) is the main axis of complexity.
  * Number of contexts and their outcome counts drive constraints.
  * `_to_context()` automatically deduplicates identical (observables, pmf) patterns to speed up solves.
* **Sampling speed**: `sample_observations` and `count_observations` use Numba JIT kernels with binary-search inverse CDF and dense tallying for near-C performance.

---

## Validation & Invariants

* `__setitem__` enforces exact outcome sets per context (no extras/missing; no duplicates).
* All algebraic ops require **compatible spaces/contexts** (clear error messages explain mismatches).
* `from_mu`: checks length and non-negativity of μ; optional normalization.

---

## Recipes

### 1) “Is this behavior coherent?” (FI test)

```python
if behavior.is_frame_independent():
    print("No contradiction (α* = 1, K = 0).")
else:
    print(f"Contradiction detected: α*={behavior.agreement:.6f}, K={behavior.contradiction_bits:.3f} bits")
```

### 2) “Where is the tension coming from?” (λ\* witness)

```python
for ctx, w in behavior.worst_case_weights.items():
    print(tuple(ctx), f"{w:.3f}")
```

### 3) “How do *I* see it?” (custom λ)

```python
weights = {("A",): 0.5, ("B",): 0.25, ("A","B"): 0.25}
res = behavior.agreement_for_weights(weights)
print("α(λ) =", res.score)
for assign, prob in sorted(res.scenarios(), key=lambda t: t[1], reverse=True)[:5]:
    print(assign, f"{prob:.2%}")
```

### 4) “Give me a single consensus behavior from θ\*(λ)”

```python
assignments = list(res.space.assignments())
context = tuple(res.space.names)  # full joint
consensus = Behavior.from_contexts(res.space, {
    context: {assignments[i]: float(res.theta[i]) for i in range(len(assignments))}
})
```

### 5) Large-scale evaluation via Monte Carlo

```python
ctx_idx, out_idx = behavior.sample_observations(2_000_000, seed=123)
counts, totals = behavior.count_observations(ctx_idx, out_idx)
# derive empirical p̂_c and re-fit FI model to empirical tables
alpha_hat, Q_star_emp = behavior.fit_fi_to_empirical(hat_pc={}, mu_hat=None)  # see method docstring
```

---

## Error Messages You Might See (and what they mean)

* **“Context must belong to the same space”**: You’re assigning a `Distribution` built for a different `Space`.
* **“Outcome mismatch (extra/missing)”**: The distribution’s labels don’t exactly match the context’s outcomes.
* **“Behaviors must have the same space/contexts”**: Required by `mix`, `product_l1_distance`, etc.
* **“Cannot compare behaviors with overlapping contexts”**: `agreement_with` merges; duplicates would collide.
* **“mu must have length …”**: Global assignment vector must match `space.assignment_count()`.
* **“Weights must sum to a positive value”**: `agreement_for_weights` will renormalize, but total must be > 0.

---

## Design Rationale (for the curious)

* **Bhattacharyya at the core**: Joint concavity + multiplicativity make it ideal for a max-min “distance-to-FI” measure. It detects perfect matches (1.0) and composes cleanly across independent subsystems.
* **Minimax structure**: The inner product with λ exposes the *most problematic* contexts. You get an interpretable witness (λ\*) and a concrete best-explanation (θ\*).
* **Information-theoretic readout**: `K = -log2(α*)` expresses contradiction as bits of structural tension — *how many binary decisions you’d need, on average, to repair the clash*.

---

## Reference: Key Methods (signatures)

```python
# Constructors
Behavior.from_contexts(space, context_dists: dict) -> Behavior
Behavior.frame_independent(space, contexts, assignment_weights=None) -> Behavior
Behavior.random(space, contexts, alpha=1.0, seed=None) -> Behavior
Behavior.from_mu(space, contexts, mu: np.ndarray, normalize=True) -> Behavior
Behavior.from_counts(space, context_tables, normalize="per_context") -> Behavior

# Analysis
behavior.agreement                      # α*
behavior.contradiction_bits             # K
behavior.is_frame_independent(tol=1e-9) -> bool
behavior.worst_case_weights             # λ*
behavior.agreement_for_weights(weights) -> AgreementResult
behavior.per_context_scores(agreement=None, mu="optimal") -> np.ndarray
behavior.aggregate(aggregator, agreement=None, mu="optimal") -> float
behavior.least_favorable_lambda() -> dict
behavior.alpha_given_lambda(lam_dict) -> float|tuple
behavior.fit_fi_to_empirical(hat_pc, mu_hat) -> (alpha_hat, Q_star_emp)

# Sampling
behavior.sample_observations(n_samples, context_weights=None, seed=None)
behavior.count_observations(ctx_indices, outcome_indices, n_contexts=None, n_outcomes=None)

# Algebra / transforms
behavior @ other
behavior + other
behavior.mix(other, weight)
behavior.rename_observables(name_map)
behavior.permute_outcomes(value_maps)
behavior.coarse_grain(observable, merge_map)
behavior.drop_contexts(keep_contexts)
behavior.duplicate_context(target_context, count, tag_prefix="Tag")
behavior.product_l1_distance(other)
```

---

## Practical Tips

* **After any in-place edit of `behavior.distributions` call** `behavior._invalidate_cache()` **before reading `agreement` again.**
* Prefer *constructing new* behaviors (e.g., `from_contexts`, `mix`) over manual in-place mutation to keep caches coherent.
* Use **small alphabets and fewer observables** where possible; the size of the global assignment space dominates solve time.
* For interpretability, always report both **α\*** and **K**; people find the bit scale intuitive.
