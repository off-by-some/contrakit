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
