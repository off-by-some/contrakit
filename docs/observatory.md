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
