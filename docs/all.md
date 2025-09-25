# Observatory — High-Level API for Building Behaviors

`observatory` is a fluent layer on top of `Space`, `Context`, `Distribution`, and `Behavior`. It lets you define **concepts** (observables), assign **probability distributions** to **perspectives** (contexts), and produce `Behavior` objects—optionally through **lenses** that model viewpoints without changing the base universe.

Below, you’ll find **only the public API**, with concise math where it clarifies semantics.

---

## Mathematical snapshot

* Let $\mathcal{X}=\{X_1,\dots,X_n\}$ be the set of **concepts** with finite alphabets $\mathcal{O}_{X_i}$.
* A **context** $c \subseteq \mathcal{X}$ has outcome alphabet $\mathcal{O}_c:=\prod_{X\in c}\mathcal{O}_X$.
* A **perspective assignment** provides a pmf $p_c \in \Delta(\mathcal{O}_c)$.
* A resulting **behavior** is $P=\{p_c : c\in\mathcal{C}\}$.

**Lens math.** A lens $L$ produces a *raw* (tagged) behavior on the extended space $\mathcal{X}\cup\{L\}$ where $L$ is a dummy axis with alphabet $\{0\}$. For any base context $c$ and outcome $o\in\mathcal{O}_c$,

$$
q_{(c,L)}(o,0)\;=\;p_c(o),\qquad q_{(c,L)}(o,\ell\neq 0)=0.
$$

`to_behavior()` projects back to the base space $\mathcal{X}$; `to_behavior_raw()` returns the tagged version.

---

## Top-level types & exceptions

### `NoConceptsDefinedError` *(ValueError)*

Raised when an operation requires at least one concept (e.g., generating a behavior) but none were defined.

### `EmptyBehaviorError` *(ValueError)*

Raised when attempting to generate a behavior with no distributions and `allow_empty=False`.

---

## Concepts and values

### `class ValueHandle`

A typed handle for a single symbol in a concept’s alphabet. Provides tuple sugar for building outcomes.

* **Attributes**

  * `value: Any` – the raw symbol.
  * `concept: ConceptHandle` – the owner concept.

* **Operators**

  * `a & b -> tuple`
    Build joint outcomes succinctly:

    ```python
    yes, no = reviewer.alphabet
    yes & no      # -> ("Yes", "No")
    "Hot" & yes   # -> ("Hot", "Yes")
    ```

  * `__str__` / `__repr__`
    The string forms of a `ValueHandle` are stable and convenient for debugging and tests:

    * `str(vh)` → the raw value (e.g. `'Yes'`).
    * `repr(vh)` → `ValueHandle('value')` (e.g. `ValueHandle('Yes')`).

---

### `class ConceptHandle`

Represents an observable (variable/dimension).

* **Attributes**

  * `name: str`
  * `symbols: tuple[Any, ...]`

* **Properties**

  * `alphabet -> tuple[ValueHandle, ...]`
    Convenience handles corresponding to `symbols`.

* **Operators**

  * `concept & other -> (concept, other)`
    Sugar for forming context keys.

Note: the `&` sugar is supported for all Observables — not just `ConceptHandle`s. In particular,
observable lenses (see Lenses) also participate in `a & b` to form joint context keys when used
as part of perspective assignments. In short: All Observables (concepts and observable lenses)
support `a & b` to build joint context keys.

---

## Building systems

### `class Observatory`

Main entry for defining concepts and managing perspectives.

* **Construction**

  * `Observatory.create(symbols: Sequence[Any] | None = None) -> Observatory`
    Optional *global alphabet* (`observatory.alphabet`) used as the default for concepts that omit `symbols`.
    
    If a `symbols` sequence is supplied to `Observatory.create(...)`, the ordering (and any
    duplicates) is preserved verbatim in the resulting global alphabet. The sequence may contain
    strings and/or `ValueHandle` instances; when `ValueHandle`s are present their `.value` is used.

* **Properties**

  * `alphabet -> tuple[ValueHandle, ...]`
    The global alphabet as `ValueHandle`s (empty tuple if not set).
  * `perspectives -> PerspectiveMap`
    Dict-like map for assigning distributions (see below).

* **Concepts**

  * `concept(name: str, symbols: Sequence[Any] | None = None) -> ConceptHandle`
    Define a new observable. If `symbols=None`, uses the global alphabet.
    
    Important notes on `symbols`:

    * `symbols` may be a sequence of `str` and/or `ValueHandle` instances (from this or other
      observatories). When `ValueHandle`s are provided, their `.value` attribute is used as the
      alphabet symbol.
    * The ordering of `symbols` is preserved exactly as provided, including duplicates. Tests may
      assert exact ordering and duplicates are kept verbatim.
    * Cross-observatory `ValueHandle`s are allowed; only their `.value` is used when constructing
      the concept alphabet.
  * `define_many(specs: Sequence[str | dict]) -> tuple[ConceptHandle, ...]`
    Batch creation. Each item is either `"Name"` or `{"name": "Name", "symbols": ...}`.

* **Lenses**

  * `lens(name: str | ConceptHandle, symbols: Sequence[Any] | None = None) -> LensScope`
    Create a lens scope. If `symbols` is provided, the lens becomes **observable** (it can itself be used in contexts, and its symbols are exposed as `alphabet`).

---

## Assigning distributions

### `class PerspectiveMap`

Manages pmfs on contexts; accessible via `observatory.perspectives`.

* **Indexing (read)**

  * `__getitem__(key) -> DistributionWrapper`
    `key` can be a name, `ConceptHandle`, `LensScope`, or a tuple thereof. Returns a wrapper with `.distribution: Distribution`.

* **Indexing (write)**

  * `__setitem__(key, value: dict) -> None`
    Set a pmf for a context. Outcome keys may be raw symbols, `ValueHandle`s, or tuples combining them (including `&` sugar).
    **Normalization**: pmfs must sum to 1 (within tolerance).
    **Auto-completion**: for single-observable contexts, missing symbols get remaining mass evenly (if total < 1).

  ```python
  yes, no = voter.alphabet
  pmf = {yes: 0.6, no: 0.4}
  observatory.perspectives[voter] = pmf  # context is ("Voter",)
  ```

* **Other methods**

  * `add_joint(*observables, distribution: dict) -> None`
    Convenience for multi-observable contexts:

    ```python
    A, B = a_concept, b_concept
    perspectives.add_joint(A, B, {("x","y"): 0.5, ("x","z"): 0.5})
    ```
  * `validate(allow_empty: bool = False) -> bool`
    Checks (i) nonnegativity and (ii) per-context normalization $\sum_o p_c(o)=1$.
    *Note*: full global marginal consistency is not enforced here (that’s analyzed via `Behavior`).
  * `values -> dict[tuple[str, ...], dict[tuple, float]]`
    Plain dictionary view of all defined pmfs.
  * `to_behavior(allow_empty: bool = False) -> Behavior`
    Produce a `Behavior` over the **base space** from all defined contexts.
    Raises `EmptyBehaviorError` if none and `allow_empty=False`.

  * `perspectives` availability and snapshot semantics
    * Accessing `observatory.perspectives` requires that at least one concept has been defined
      in the observatory; otherwise a `ValueError("Perspectives not available: no concepts defined")`
      is raised. This helps catch accidental usage before any observables exist.
    * The returned `PerspectiveMap` is *bound* to the observatory's current space. If you define
      new concepts after grabbing a reference to `observatory.perspectives`, you should reacquire
      `observatory.perspectives` before using it; indexing with newly added concepts may raise
      `KeyError` on older snapshots.

---

## Lenses (perspective scoping)

### `class LensScope`

Context manager for building perspectives without polluting the base space. If constructed with `symbols`, the lens is itself observable.

* **Usage pattern**

  ```python
  with observatory.lens("Reviewer") as L:
      skill = L.define("Skill", symbols=["High","Low"])
      high, low = skill.alphabet
      L.perspectives[skill] = {high: 0.7, low: 0.3}
      B = L.to_behavior()       # Behavior on base space
      Braw = L.to_behavior_raw()# Behavior on lens-extended space

* Lens perspective proxy name
  The dict-like proxy used for `L.perspectives` is commonly referred to in tests as
  `LensPerspectiveProxy`. It behaves like a mapping for assigning per-concept pmfs and may be
  inspected in test code; relying on its public mapping-like behavior is recommended rather than
  depending on implementation details.
  ```

* **Properties**

  * `name -> str`
  * `symbols -> tuple[Any, ...]` *(only if provided at creation; otherwise raises)*
  * `alphabet -> tuple[ValueHandle, ...]` *(only if symbols provided)*
  * `observatory -> Observatory`
  * `perspectives -> (dict-like)`
    A proxy that accepts the same pmf format as `PerspectiveMap`, but records both base and lens-tagged contexts under the hood.

* **Defining concepts inside a lens**

  * `define(concept: str, symbols: Sequence[Any] | None = None) -> ConceptHandle`
    Shorthand for `observatory.concept(...)`.

* **Behavior export**

  * `to_behavior(allow_empty: bool = False) -> Behavior`
    Returns a `Behavior` over the **base** space. Mathematically, you get $\{p_c\}$.
  * `to_behavior_raw(allow_empty: bool = False) -> Behavior`
    Returns a `Behavior` over the **lens-extended** space; for each base $p_c$ you get the tagged $q_{(c,L)}$ satisfying $q_{(c,L)}(o,0)=p_c(o)$.

  Errors and empty behavior rules
  * If no concepts have been defined inside the lens at all, `to_behavior(...)` and
    `to_behavior_raw(...)` raise `NoConceptsDefinedError` even when `allow_empty=True`. The
    absence of concepts is considered a separate error class from having concepts with no
    distributions.
  * If concepts exist but no distributions have been set for any of them, then `to_behavior(...)`
    raises `EmptyBehaviorError` when `allow_empty=False`. If `allow_empty=True` the call returns
    an empty `Behavior` (no contexts). Note that `allow_empty=True` does not bypass the "no
    concepts defined" check above.

* **Lens algebra**

  * `compose(other: LensScope) -> LensComposition`
  * `intersection(other: LensScope) -> LensComposition`
  * `difference(other: LensScope) -> LensComposition`
  * `symmetric_difference(other: LensScope) -> LensComposition`
  * `lens1 | lens2 -> LensComposition` *(alias of `compose`)*

* **Advanced**

  * `contexts_low_level() -> dict[tuple[str, ...], dict[tuple, float]]`
    Introspect the **lens-tagged** context map used by `to_behavior_raw()`.

---

## Lens compositions

### `class LensComposition`

Immutable collection of lenses with a chosen composition mode.

* **Operators**

  * `composition | lens -> LensComposition`
    Add another lens (keeps current mode).

* **Derived quantities**

  * `perspective_contributions -> dict[str, float]`
    A lens-level witness built from the least-favorable context weights $\lambda^\star$ of the composed behavior:

    $$
    \text{contrib}(L)\;=\;\sum_{c:\,c\text{ tags }L}\;\lambda^\star_c,
    \qquad \sum_c \lambda^\star_c = 1.
    $$

    Intuition: higher mass → that lens’s contexts more tightly constrain the minimax agreement.
  * `witness_distribution -> dict[str, float]`
    Alias of `perspective_contributions`.

* **Behavior export**

  * `to_behavior() -> Behavior`
    Build a `Behavior` from the composed lenses.
    Modes:

    * **union** *(default)*: include all lens-tagged contexts.
    * **intersection**: include only contexts common to both (for two lenses; otherwise falls back to union).
    * **difference** *(A\B)*: contexts of the first lens not present in the second (two lenses; else union).
    * **symmetric\_difference**: contexts present in exactly one of the two lenses (two lenses; else union).

---

## Low-level distribution access

### `class DistributionWrapper`

A lightweight wrapper providing:

* `distribution: Distribution` – the underlying `Distribution` object for the requested context.

Returned by `PerspectiveMap.__getitem__`.

---

## Protocols

### `class Observable` *(Protocol)*

Satisfied by any object that can be addressed as an observable in this API (e.g., `ConceptHandle`, observable `LensScope`):

* **Properties**
  `name: str`, `symbols: tuple[Any, ...]`, `alphabet: tuple[ValueHandle, ...]`, `observatory: Observatory`

---

## Worked example (end-to-end)

```python
# 1) Create an observatory with a global alphabet
obs = Observatory.create(symbols=["Yes","No"])

# 2) Define concepts (uses global alphabet by default)
voter    = obs.concept("Voter")
candidate= obs.concept("Candidate", symbols=["Qualified","Unqualified"])

# 3) Assign perspectives (contexts and pmfs)
yes, no = voter.alphabet
obs.perspectives[voter] = {yes: 0.6, no: 0.4}

# 4) Build a base behavior
B = obs.perspectives.to_behavior()

# 5) Model a viewpoint with a lens (base and raw forms)
with obs.lens("Reviewer", symbols=["Strict","Lenient"]) as L:
    skill = L.define("Skill", symbols=["High","Low"])
    high, low = skill.alphabet
    L.perspectives[skill] = {high: 0.7, low: 0.3}

    B_base = L.to_behavior()       # behavior over {Voter, Candidate, Skill}
    B_raw  = L.to_behavior_raw()   # behavior over {Voter, Candidate, Skill, __lens__Reviewer}
```

---

## Notes & guarantees

* **Normalization**: Each pmf must satisfy $\sum_o p_c(o)=1$ (within tolerance) and $p_c(o)\ge 0$. Violations raise `ValueError`.
* **Auto-complete**: Single-observable marginals are auto-completed if you provide a partial pmf (remaining mass spread uniformly over missing symbols).
* **Empty behaviors**: `to_behavior(..., allow_empty=False)` raises `EmptyBehaviorError` when nothing is defined.
* **Lens neutrality**: `to_behavior()` never adds lens axes to your base space; `to_behavior_raw()` makes the tagging explicit.

That’s the whole public surface—with math for how the pieces map to behaviors and how lenses lift and project distributions.



# Space — Observable Spaces for the Mathematical Theory of Contradiction

`Space` defines **what can be measured** (observables) and **which values are possible** (alphabets). Everything else in the library—contexts, behaviors, agreement/contradiction—sits on top of this foundation.

Think of a `Space` as the schema of your measurement system:

* **Observables**: named variables (e.g., `"Temperature"`, `"Opinion"`).
* **Alphabets**: finite sets of values for each observable (e.g., `("Hot","Warm","Cold")`).
* **Assignments**: full joint specifications (a value for *every* observable).
* **Contexts**: subsets of observables that are observed together (defined by other modules using this space).

---

## TL;DR / Quickstart

```python
from contrakit import Space

# 1) Define the universe of observables
weather = Space.create(
    Temperature=["Hot", "Warm", "Cold"],
    Humidity=["High", "Medium", "Low"],
    Precipitation=["Rain", "Snow", "None"],
)

# 2) Subspace / restriction to a subset of observables (order preserved)
wx_pair = weather | ["Temperature", "Precipitation"]

# 3) Tensor product to combine disjoint universes
prefs = Space.create(Coffee=["Yes","No"])
combined = weather @ prefs  # requires no overlapping names

# 4) Enumerate global assignments (lazy) and count them
gen = weather.assignments()          # iterator of tuples
n   = weather.assignment_count()     # exact count without materializing

# 5) List outcomes for a context (cached)
wx_outcomes = weather.outcomes_for(["Temperature","Precipitation"])

# 6) Rename observables safely (bijective rename)
renamed = weather.rename({"Precipitation": "PPT"})
```

---

## Core Ideas (30-second model)

* `Space` is **frozen** and **structurally hashed** → safe to use as keys and for caching.
* Names define the **order** of coordinates in assignments; order is stable and matters.
* Alphabets are **finite** and **non-empty** (enforced).
* Product operations (`@`) build larger universes; restriction (`|`) forms subspaces while preserving original name order.

---

## Public API

### Construction

```python
Space(names: Tuple[str, ...], alphabets: Dict[str, Tuple[Any, ...]])
```

Low-level constructor (used by helpers below). Validates:

* Every `name` has an alphabet.
* No extra alphabets.
* All alphabets are non-empty.

```python
Space.create(**observables: Sequence[Any]) -> Space
```

Ergonomic builder:

```python
Space.create(X=[0,1], Y=["a","b","c"])
```

```python
Space.binary(*names: str) -> Space
```

Binary convenience:

```python
Space.binary("A","B","C")  # each with alphabet (0,1)
```

---

### Introspection & Access

```python
len(space) -> int
```

Number of observables.

```python
name in space -> bool
```

Membership by string or object with `.name`.

```python
space[name] -> Tuple[Any, ...]
```

Alphabet lookup (accepts string or object with `.name`).

```python
space.index_of(name: str) -> int
```

Index of an observable inside `space.names` (the coordinate order for assignments).

---

### Enumeration & Counting

```python
space.assignments() -> Iterator[Tuple[Any, ...]]
```

**Lazy** generator of all global assignments in the fixed `space.names` order.

> Use when you need to iterate; avoid materializing for large spaces.

```python
space.assignment_count() -> int
```

Exact count of global assignments (`∏ |alphabet|`) without materializing.

```python
space.outcomes_for(observables: Sequence[str]) -> List[Tuple[Any, ...]]
```

All outcomes for a **context** (subset of observables). **LRU-cached**; safe to call frequently.

---

### Algebra of Spaces

```python
space | ["X","Y"] -> Space
```

**Restriction** (subspace). Preserves the original order of `space.names`; rejects unknown names and deduplicates input.

```python
space @ other -> Space
```

**Tensor product** (disjoint union). Requires **no overlapping names**; raises if overlaps are found.

```python
space.rename(name_map: Dict[str, str]) -> Space
```

Return a new space with renamed observables. The mapping must be a **bijection** on the used names (no collisions after renaming).

---

### Comparison, Equality, Hashing

```python
space == other  # structural equality (names + alphabets)
hash(space)     # content hash (names + alphabets)
```

```python
space.difference(other: Space) -> Dict[str, Any]
```

Structured diff:

* `'only_self'`: names in `space` not in `other`
* `'only_other'`: names in `other` not in `space`
* `'alphabet_diffs'`: `{name: (self_alpha, other_alpha)}` for shared names with different alphabets

---

## Design Guarantees & Invariants

* **Frozen dataclass**: Instances are immutable; methods return **new** `Space` objects.
* **Order is law**: `space.names` fixes the coordinate order of assignments; downstream code (contexts/behaviors) relies on this.
* **Total coverage**: `alphabets` must cover all `names` exactly—no extras, no missing.
* **Non-empty alphabets**: Every observable must have at least one value.

---

## Performance Notes

* `outcomes_for(...)` uses an **LRU cache** keyed by `(obs_tuple, alph_tuple)`—ideal for repeated context queries.
* `assignments()` is deliberately **lazy** (generator); prefer `assignment_count()` when you only need cardinality.
* All structures are tuples internally (names, alphabets) for stable hashing and cache keys.

---

## Common Errors (and How to Fix Them)

* **“Missing alphabets for observables: {...}”**
  Add all required `name: alphabet` entries (or correct typos in `names`).

* **“Extra alphabets for unknown observables: {...}”**
  Remove alphabets not listed in `names` (or fix mismatched keys).

* **“Empty alphabets for observables: \[...]”**
  Ensure every observable has a non-empty alphabet.

* **“Unknown observables: {...}” (from `|`)**
  Restrict only to names present in `space.names`.

* **“Overlapping observables in tensor product: {...}” (from `@`)**
  Rename colliding observables first or ensure the spaces are disjoint.

* **“name\_map must be a bijection (duplicates found).” (from `rename`)**
  Make sure the target names are unique (no two sources map to the same target).

---

## Practical Recipes

### Build a measurement universe, then project contexts

```python
survey = Space.create(
    AgeGroup=["18-24","25-34","35-44","45+"],
    Opinion=["Yes","No","Unsure"],
    Channel=["Email","SMS","Phone"],
)
# Context outcomes (cached):
survey.outcomes_for(["Opinion"])          # 3 outcomes
survey.outcomes_for(["Opinion","Channel"])# 3 × 3 = 9 outcomes
```

### Merge independent domains

```python
device = Space.create(Device=["Mobile","Desktop"])
region = Space.create(Region=["NA","EU","APAC"])
universe = device @ region                 # 2 × 3 = 6 global cells
```

### Safe renaming before alignment

```python
a = Space.create(A=["0","1"], B=["x","y"])
b = Space.create(Aprime=["0","1"], B=["x","y"])
b2 = b.rename({"Aprime":"A"})
a.difference(b2)  # now only alphabet diffs (if any), not name diffs
```

### Compute sizes without iterating

```python
n_states = (Space.binary("V1","V2","V3") @ Space.create(Color=["R","G","B"])).assignment_count()
# = 2*2*2 * 3 = 24
```

---

## Why this design?

* **Immutability by default** ensures reproducibility and safe reuse in caches/solvers.
* **Strict structural equality** avoids subtle mismatches that would corrupt downstream inferences.
* **Explicit algebra** (`|`, `@`, `rename`) keeps transformations declarative and auditable.
* **Caching where it counts**: outcomes for contexts repeat often in optimization loops; global assignment enumeration stays lazy to avoid blow-ups.



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


# Context — Observational Contexts for Multi-Perspective Measurements

`Context` specifies **which observables are measured together**. It’s the unit on which per-context distributions live and the surface where consistency (or contradiction) is tested across perspectives.

---

## Mathematics (precise & minimal)

Given a space $(\mathcal{X}, \{\mathcal{O}_X\}_{X\in\mathcal{X}})$ with finite observables and alphabets:

* A **context** $c \subseteq \mathcal{X}$ has outcome alphabet

  $$
  \mathcal{O}_c \equiv \prod_{X\in c}\mathcal{O}_X.
  $$
* A **global assignment** $s \in \mathcal{O}_{\mathcal{X}} \equiv \prod_{X\in\mathcal{X}}\mathcal{O}_X$ restricts to $c$ via the projection

  $$
  r_c:\mathcal{O}_{\mathcal{X}}\to \mathcal{O}_c,\qquad r_c(s)=s|_c.
  $$
* If $\mu$ is a global law on $\mathcal{O}_{\mathcal{X}}$, the **marginal** on $c$ is

  $$
  p_c(o)=\sum_{s:\, s|_c=o}\mu(s), \quad o\in\mathcal{O}_c.
  $$

These maps are exactly what `Context.outcomes()` and `Context.restrict_assignment(...)` implement.

---

## Public API

### Class

```python
@dataclass(frozen=True, eq=False)
class Context:
    space: Space
    observables: Tuple[str, ...]
```

An immutable specification of a measured subset of `space.names` in a fixed order.
Validation on construction:

* All `observables` exist in `space`.
* No duplicates in `observables`.

**Equality & hashing**

* `c1 == c2` iff both `space` and `observables` match.
* Hashable; can be used as dict keys (e.g., in behaviors).

**Representation**

* `repr(context)` includes space size and ordered observables.

---

### Constructors

#### `Context.make(space: Space, observables: Union[str, Sequence[str]]) -> Context`

Ergonomic builder. Accepts a single name or a sequence. Order is preserved:

```python
ctx = Context.make(space, ["Temperature","Pressure"])
```

---

### Introspection & set-like ops

#### `len(context) -> int`

Number of observables in the context.

#### `name in context -> bool`

Membership test by string or object with `.name`.

#### `context | observables -> Context`

**Extend** the context by (uniquely) adding names while preserving existing order.
`observables` may be a string or a sequence.

```python
ctx = Context.make(space, "A")
ctx2 = ctx | ["B","C"]   # ("A","B","C")
```

#### `context & other -> Context`

**Intersection** of two contexts (same `space` required). Order follows the left operand’s order:

```python
ctx = Context.make(space, ("A","B","C"))
ctx & Context.make(space, ("B","D"))   # -> ("B",)
```

---

### Semantics

#### `Context.outcomes() -> List[Tuple[Any, ...]]`

All possible outcomes $\mathcal{O}_c$ for the context, in deterministic order induced by the space’s alphabets. Backed by an LRU cache under the hood.

#### `Context.restrict_assignment(assignment: Tuple[Any, ...]) -> Tuple[Any, ...]`

Compute the projection $r_c(s)$.
`assignment` must be a global assignment ordered as `space.names`:

```python
s = next(space.assignments())   # e.g., ("Hot","High","Wet")
ctx = Context.make(space, ("Temperature","Pressure"))
ctx.restrict_assignment(s)      # -> ("Hot","High")
```

---

## Usage patterns

```python
from contrakit import Space, Context

# Define a universe
weather = Space.create(
    Temperature=["Hot","Warm","Cold"],
    Pressure=["High","Low"],
    Humidity=["Wet","Dry"],
)

# Make contexts
tp  = Context.make(weather, ("Temperature","Pressure"))
hum = Context.make(weather, "Humidity")

# Enumerate outcomes (cartesian products)
tp_outcomes  = tp.outcomes()    # 3 × 2 = 6 outcomes
hum_outcomes = hum.outcomes()   # 2 outcomes

# Projection (restriction) of global assignments
s  = ("Warm","Low","Wet")       # order follows weather.names
tp_s = tp.restrict_assignment(s)  # ("Warm","Low")

# Set-like composition
t   = Context.make(weather, "Temperature")
tp2 = t | "Pressure"            # ("Temperature","Pressure")
t∩h = t & hum                   # () if disjoint, or common names in left order
```

---

## Guarantees & errors

* **Order matters**: `observables` are ordered; outcomes and restrictions follow this order.
* **Same-space operations**: `context & other` requires identical `space`; otherwise `ValueError`.
* **Uniqueness**: no duplicate names within a context; enforced at construction.
* **Totality**: `restrict_assignment` expects a full global assignment aligned with `space.names`.



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
