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
