# Distribution — Probability Distributions for Observational Contexts

`Distribution` is the canonical PMF (probability mass function) used across contexts. It is **immutable**, validated, and optimized for constant-time lookups and safe convex operations.

---

## Mathematics (crisp & minimal)

Let a finite outcome alphabet be $\mathcal{O}=\{o_1,\dots,o_k\}$.
A distribution $p$ on $\mathcal{O}$ is a vector $p\in\Delta^{k-1}$ with

$$
\forall i:\; p(o_i)\ge 0,\quad \sum_{i=1}^k p(o_i)=1.
$$

* **Support:** $\mathrm{supp}(p)=\{o\in\mathcal{O}\,:\,p(o)>0\}$.
* **Convex mixture:** For $0\le \lambda\le 1$,

$$
(\,(1-\lambda)p+\lambda q\,)(o)= (1-\lambda)p(o)+\lambda q(o).
$$

* **$\ell_1$ distance:** $\|p-q\|_1 = \sum_{o\in\mathcal{O}} |p(o)-q(o)|$.
  (Half of this is the **total variation distance**: $\mathrm{TV}(p,q)=\tfrac12\|p-q\|_1$.)

---

## Public API

### Class

```python
@dataclass(frozen=True)
class Distribution:
    outcomes: Tuple[Tuple[Any, ...], ...]
    probs:    Tuple[float, ...]
```

An immutable PMF over `outcomes` (each outcome is a tuple matching a context’s observable order).

**Validation (on construction):**

* `len(outcomes) == len(probs)`; else `ValueError`
* `sum(probs) ≈ 1` within `NORMALIZATION_TOL`; else `ValueError`
* All probabilities ≥ `-EPS`; else `ValueError` (guards tiny negatives)

**Indexing & equality:**

* Fast `__getitem__(outcome)` lookup; returns `0.0` if the outcome is absent.

---

### Constructors

#### `Distribution.from_dict(pmf: Dict[Tuple[Any, ...], float]) -> Distribution`

Build from an `outcome -> probability` mapping (order is the dict’s iteration order).

#### `Distribution.uniform(outcomes: Sequence[Tuple[Any, ...]]) -> Distribution`

Uniform PMF over given outcomes.
Errors on empty `outcomes`.

#### `Distribution.random(outcomes: Sequence[Tuple[Any, ...]], alpha: float = 1.0, rng: Optional[np.random.Generator] = None) -> Distribution`

Dirichlet draw over the simplex with parameter `alpha` (symmetric).
Errors on empty `outcomes`. Uses `rng` or a fresh generator.

---

### Accessors

#### `__getitem__(outcome: Tuple[Any, ...]) -> float`

Probability of `outcome` (0.0 if not present).

#### `to_dict() -> Dict[Tuple[Any, ...], float]`

Dictionary view (stable order).

#### `to_array() -> np.ndarray`

Dense probability vector in the stored order.

---

### Composition & Metrics

#### `__add__(other: Distribution) -> Distribution`

Equal convex combination: $\tfrac12 p + \tfrac12 q$.
(Delegates to `mix(other, 0.5)`.)

#### `mix(other: Distribution, weight: float) -> Distribution`

Convex mixture: $(1-\text{weight})\,p + \text{weight}\,q$, `weight ∈ [0,1]`.

* **Outcome alignment:** the union of both supports is taken in a **stable, deterministic order**; missing probabilities are treated as 0.0.
* **Result:** A valid `Distribution` over the union support.

#### `l1_distance(other: Distribution) -> float`

$\ell_1$ distance across the **union** of supports:

* Fast path when orders match; else aligns by outcome keys.
* For total variation: `TV = 0.5 * l1_distance(other)`.

---

## Usage patterns

```python
# Build explicitly
coin = Distribution(
    outcomes=(("Heads",), ("Tails",)),
    probs=(0.6, 0.4)
)

# Or from a dict
weather = Distribution.from_dict({
    ("Sunny","Warm"): 0.5,
    ("Cloudy","Cool"): 0.3,
    ("Rainy","Cold"): 0.2,
})

# Uniform / random
U = Distribution.uniform([("R",), ("G",), ("B",)])
R = Distribution.random([("A",), ("B",), ("C",)], alpha=0.5)

# Query / transform
p_heads = coin[("Heads",)]         # 0.6
as_dict = weather.to_dict()
as_vec  = weather.to_array()

# Mix and compare (supports need not match)
p = coin.mix(U, 0.25)
d = coin.l1_distance(p)            # TV = 0.5 * d
```

---

## Guarantees & error cases

* **Normalization & non-negativity** are enforced at construction.
* **Stable support order** is preserved; mixtures use a deterministic union order.
* **Edge cases**:

  * Empty outcome sets → `ValueError` in `uniform`/`random`.
  * Mixture `weight` outside `[0,1]` → `ValueError`.
  * Sums far from 1 or negative probs beyond tolerance → `ValueError`.

---

## Practical notes

* Treat `outcomes` as the **schema** for the vector in `probs`; downstream contexts rely on consistent ordering.
* When mixing heterogeneous supports, think of the result as the **minimal common refinement** (the union) with zero-filled gaps.
* Use `l1_distance` when you need a **metric**; convert to total variation with `* 0.5` when that interpretation is desired.
