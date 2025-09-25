# `distribution.py`

*Immutable probability distributions for observational contexts.*

---

## Overview

The `distribution.py` module defines the `Distribution` class, which represents probability mass functions (PMFs) used across observational contexts in the contradiction framework. It provides an immutable, validated container optimized for constant-time lookups and safe convex operations.

This module implements the core data structure for representing probability distributions over finite outcome spaces, ensuring numerical stability and consistent behavior across different contexts.

---

## Why This Module Exists

Multi-perspective behavioral analysis requires consistent, reliable handling of probability distributions. Rather than using raw arrays or dictionaries that could become inconsistent or numerically unstable, this module provides:

* **Immutable distributions** that cannot be accidentally modified
* **Validation** to ensure probabilistic constraints are met
* **Efficient operations** for mixing and comparing distributions
* **Safe defaults** for missing outcomes and numerical edge cases

The module was separated to provide a robust foundation for probability handling while keeping the core logic focused on contradiction analysis.

---

## Architecture

The module implements a single immutable data structure with comprehensive operations:

- **Data structure**: Frozen dataclass with tuple-based storage for immutability
- **Validation layer**: Construction-time checks for normalization and non-negativity
- **Operation layer**: Convex combinations, distance metrics, and format conversions
- **Performance optimizations**: Fast lookups and deterministic ordering

**Key Design Decisions**:
- **Immutability**: Prevents accidental modification of probability data
- **Tuple storage**: Ensures hashability and stable ordering
- **Zero defaults**: Missing outcomes return 0.0 probability
- **Union alignment**: Convex mixtures handle heterogeneous supports gracefully

---

## Key Classes and Functions

### `class Distribution`

Immutable probability mass function container.

**Attributes**:

* `outcomes` *(Tuple[Tuple[Any, ...], ...])* — Ordered tuple of possible outcomes
* `probs` *(Tuple[float, ...])* — Corresponding probability values

### `Distribution.from_dict(pmf) -> Distribution`

Create distribution from outcome-to-probability mapping.

**Parameters**:

* `pmf` *(Dict[Tuple[Any, ...], float])* — Outcome to probability mapping

**Returns**:

* *(Distribution)* — Validated distribution instance

**Example**:

```python
dist = Distribution.from_dict({
    ("Heads",): 0.6,
    ("Tails",): 0.4
})
```

### `Distribution.uniform(outcomes) -> Distribution`

Create uniform distribution over given outcomes.

**Parameters**:

* `outcomes` *(Sequence[Tuple[Any, ...]])* — List of possible outcomes

**Returns**:

* *(Distribution)* — Uniform distribution

### `Distribution.random(outcomes, alpha=1.0, rng=None) -> Distribution`

Create random distribution using Dirichlet sampling.

**Parameters**:

* `outcomes` *(Sequence[Tuple[Any, ...]])* — List of possible outcomes
* `alpha` *(float)* — Dirichlet concentration parameter
* `rng` *(Optional[np.random.Generator])* — Random number generator

**Returns**:

* *(Distribution)* — Random distribution

### Distribution Operations

```python
dist[outcome] -> float                    # Get probability (0.0 if missing)
dist.to_dict() -> Dict                    # Convert to dictionary
dist.to_array() -> np.ndarray             # Convert to array
dist.mix(other, weight) -> Distribution   # Convex combination
dist + other -> Distribution              # Equal mixture
dist.l1_distance(other) -> float          # L1 distance
```

---

## Drawbacks or Gotchas

* **Immutability only**: Distributions cannot be modified after creation
* **Memory overhead**: Tuple storage uses more memory than mutable alternatives
* **Validation strictness**: Construction rejects invalid probability values
* **Outcome ordering**: Results depend on outcome order for operations
* **No continuous support**: Only discrete probability distributions

---

## Related Modules

* [`context.py`](../context.md) — Context definitions that use distributions
* [`behavior.py`](../behavior.md) — Behavioral analysis using distributions
* [`space.py`](../space.md) — Observable spaces that define outcome structures

---

## See Also

* [Mathematical Theory Paper](../../docs/paper/A%20Mathematical%20Theory%20of%20Contradiction.pdf) — Probability theory foundations
* [Quickstart Examples](../../../examples/) — Distribution usage patterns
* [API Reference](../../docs/api/) — Complete reference documentation

---

```markdown
<!--
This file documents the public API of distribution.py.
For internal implementation details, see the source code.
Last updated 2025-09.
-->
```
