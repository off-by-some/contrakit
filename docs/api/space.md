# `space.py`

*Foundation for defining observable spaces and measurement schemas.*

---

## Overview

The `space.py` module defines the `Space` class, which provides the fundamental representation of observable spaces in the contradiction framework. A space defines what can be measured (observables) and what values each measurement can take (alphabets), serving as the mathematical foundation for all higher-level constructs like contexts, behaviors, and contradiction analysis.

This module implements the core data structure that establishes the "universe" of measurements, ensuring consistent coordinate systems and outcome spaces across the entire framework.

---

## Why This Module Exists

Multi-perspective behavioral analysis requires a consistent foundation for defining what can be measured and what values measurements can take. Without a stable coordinate system, contexts and behaviors cannot be reliably constructed or compared. The Space module addresses this by providing:

* **Structural foundation** for all measurement systems in the framework
* **Immutable definitions** that prevent accidental modification of measurement schemas
* **Algebraic operations** for composing and restricting measurement spaces
* **Efficient enumeration** and caching for computational performance

The module was created to establish the mathematical "universe" that makes contradiction analysis possible, ensuring that all higher-level operations work with consistent, well-defined measurement spaces.

---

## Architecture

The module implements an immutable, hashable data structure with comprehensive algebraic operations:

- **Core Structure**: Frozen dataclass with ordered names and alphabet mappings
- **Algebraic Operations**: Restriction (`|`), tensor product (`@`), and renaming operations
- **Enumeration Layer**: Lazy assignment generation with caching for context outcomes
- **Validation Layer**: Runtime checks for non-empty alphabets and structural consistency

**Key Design Decisions**:
- **Immutability**: Frozen instances prevent accidental schema modifications
- **Structural Hashing**: Spaces can be used as dictionary keys and cached safely
- **Order Preservation**: Observable order determines assignment coordinate ordering
- **Lazy Evaluation**: Assignment enumeration avoids materializing large spaces
- **Caching**: Context outcomes cached for repeated access patterns

**Mathematical Foundations**:
A space $\mathcal{X}$ consists of observables $X_1,\dots,X_n$ with finite alphabets $\mathcal{O}_{X_i}$. Global assignments are elements of $\prod_{i=1}^n \mathcal{O}_{X_i}$. Contexts are subsets of observables, with outcome spaces $\prod_{X\in c} \mathcal{O}_X$.

---

## Key Classes and Functions

### `class Space`

Immutable representation of observable spaces.

**Construction**:

* `Space.create(**observables) -> Space` — Create from keyword arguments mapping names to alphabets
* `Space.binary(*names) -> Space` — Create binary observables (alphabet `[0, 1]`)

**Properties**:

* `names: Tuple[str, ...]` — Ordered tuple of observable names
* `alphabets: Dict[str, Tuple[Any, ...]]` — Alphabet for each observable

**Introspection**:

* `len(space) -> int` — Number of observables
* `name in space -> bool` — Check if observable exists
* `space[name] -> Tuple[Any, ...]` — Get alphabet for observable
* `space.index_of(name) -> int` — Get index of observable in coordinate order

**Enumeration**:

* `space.assignments() -> Iterator[Tuple[Any, ...]]` — Lazy iterator over all global assignments
* `space.assignment_count() -> int` — Count of global assignments without materializing
* `space.outcomes_for(observables) -> List[Tuple[Any, ...]]` — All outcomes for a context (cached)

**Algebraic Operations**:

* `space | observables -> Space` — Restriction to subset of observables
* `space @ other -> Space` — Tensor product of disjoint spaces
* `space.rename(mapping) -> Space` — Rename observables (bijection required)

**Comparison**:

* `space == other -> bool` — Structural equality
* `hash(space) -> int` — Content-based hash
* `space.difference(other) -> Dict` — Structured difference report

### Core Workflow

```python
# 1. Create observable space
weather = Space.create(
    Temperature=["Hot", "Warm", "Cold"],
    Humidity=["High", "Medium", "Low"]
)

# 2. Restrict to subset
temp_only = weather | ["Temperature"]

# 3. Combine with other spaces
prefs = Space.create(Drink=["Coffee", "Tea"])
combined = weather @ prefs

# 4. Enumerate outcomes
temp_outcomes = weather.outcomes_for(["Temperature"])  # Cached
all_assignments = list(weather.assignments())  # Lazy
assignment_count = weather.assignment_count()  # Efficient
```

---

## Drawbacks or Gotchas

* **Immutability overhead**: All operations return new instances, not modifications
* **Memory usage**: Tuple storage and caching add overhead for large alphabets
* **Validation strictness**: Construction rejects invalid spaces (empty alphabets, etc.)
* **Order sensitivity**: Observable order affects assignment enumeration and contexts
* **No dynamic modification**: Spaces cannot be extended or modified after creation
* **Cache invalidation**: No automatic cache clearing when spaces are no longer needed

---

## Related Modules

* [`context.py`](../context.md) — Context definitions that reference spaces
* [`behavior.py`](../behavior.md) — Behaviors that use spaces as foundations
* [`observatory.py`](../observatory.md) — High-level API that builds on spaces

---

## See Also

* [Mathematical Theory Paper](../../docs/paper/A%20Mathematical%20Theory%20of%20Contradiction.pdf) — Formal foundations of observable spaces
* [Quickstart Examples](../../../examples/) — Space usage patterns
* [API Reference](../../docs/api/) — Complete reference documentation

---

```markdown
<!--
This file documents the public API of space.py.
For internal implementation details, see the source code.
Last updated 2025-09.
-->
```
