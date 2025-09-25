# `context.py`

*Classes for representing observational contexts and their relationships.*

---

## Overview

The `context.py` module defines the `Context` class, which represents specific subsets of observables that are measured together. Contexts are essential for modeling different experimental or observational setups where only certain variables are accessible simultaneously.

This module provides the mathematical machinery for working with measurement contexts, including set operations, outcome enumeration, and assignment restrictions that are used throughout the contradiction framework.

---

## Why This Module Exists

Multi-perspective measurements often involve different combinations of observables being measured together. For example, in a weather monitoring system, temperature and pressure might be measured at one station while humidity is measured at another. The Context class formalizes this concept, allowing the framework to:

* Represent which observables are jointly accessible
* Compute all possible outcomes for a given context
* Project global assignments down to context-specific observations
* Compose contexts through set operations

The module was separated to provide clean abstractions for context manipulation while keeping the core logic focused on contradiction analysis.

---

## Architecture

The module provides a single main class with supporting utilities:

- **Context class**: Immutable representation of measurement contexts with set-like operations
- **Construction methods**: Factory methods for creating contexts from observable names
- **Mathematical operations**: Outcome enumeration and assignment restriction algorithms
- **Set operations**: Union and intersection operations for context composition

**Dependencies**:
- `space.py`: Provides the observable space that contexts reference

---

## Key Classes and Functions

### `class Context`

Immutable representation of a measurement context.

**Attributes**:

* `space` *(Space)* — The complete observable space this context belongs to
* `observables` *(Tuple[str, ...])* — Ordered tuple of observable names included in this context

### `Context.make(space, observables) -> Context`

Create context from space and observable names.

**Parameters**:

* `space` *(Space)* — Observable space definition
* `observables` *(Union[str, Sequence[str]])* — Observable names (single string or sequence)

**Returns**:

* *(Context)* — New context instance

**Example**:

```python
space = Space.create(A=["0","1"], B=["0","1"])
ctx = Context.make(space, ["A", "B"])
```

### `Context.outcomes() -> List[Tuple[Any, ...]]`

Get all possible outcomes for this context.

**Returns**:

* *(List[Tuple[Any, ...]])* — All possible outcome combinations in deterministic order

**Example**:

```python
ctx = Context.make(space, ["A", "B"])
outcomes = ctx.outcomes()  # [("0","0"), ("0","1"), ("1","0"), ("1","1")]
```

### `Context.restrict_assignment(assignment) -> Tuple[Any, ...]`

Project global assignment to context-specific outcome.

**Parameters**:

* `assignment` *(Tuple[Any, ...])* — Complete assignment ordered as `space.names`

**Returns**:

* *(Tuple[Any, ...])* — Values for observables in this context

**Example**:

```python
assignment = ("0", "1", "0")  # A=0, B=1, C=0
ctx = Context.make(space, ["A", "C"])
result = ctx.restrict_assignment(assignment)  # ("0", "0")
```

### Context Operations

```python
context | observables -> Context    # Extend context with additional observables
context & other -> Context          # Intersect contexts
len(context) -> int                 # Number of observables
name in context -> bool             # Check if observable is included
```

---

## Drawbacks or Gotchas

* **Immutable only**: Contexts cannot be modified after creation
* **Space coupling**: Contexts are tied to a specific space instance
* **Order matters**: Observable order affects outcome enumeration and assignment restriction
* **Memory scaling**: Outcome enumeration creates all combinations for large contexts

---

## Related Modules

* [`space.py`](../space.md) — Observable space definitions
* [`behavior.py`](../behavior.md) — Multi-perspective behavioral analysis

---

## See Also

* [Quickstart Examples](../../../examples/quickstart/context.py) — Practical usage
* [API Reference](../../docs/api/) — Complete reference documentation

---

```markdown
<!--
This file documents the public API of context.py.
For internal implementation details, see the source code.
Last updated 2025-09.
-->
```
