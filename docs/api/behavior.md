# behavior.py

*Core class for representing and analyzing probability distributions across observational contexts.*

---

## Overview

The `behavior.py` module defines the `Behavior` class, which represents collections of probability distributions measured across different observational contexts. It provides comprehensive methods for analyzing consistency, detecting contradictions, and quantifying the information cost of reconciling incompatible measurement perspectives.

This module implements the central abstraction for multi-perspective behavioral analysis, serving as the primary interface for contradiction measurement in the contrakit framework.

---

## Why This Module Exists

Traditional statistical analysis assumes observations come from a single coherent model. However, real-world measurements often occur across incompatible contexts or perspectives that cannot be simultaneously explained by one underlying reality. This module addresses the need to:

* Quantify when multiple observational perspectives are fundamentally incompatible
* Measure the information cost required to reconcile contradictory observations
* Identify which contexts contribute most to detected contradictions

The module was separated from lower-level components to provide a clean, high-level API for behavioral analysis while maintaining clear separation between data representation and computational algorithms.

---

## Architecture

The module implements a mixin-based architecture that combines data representation and computational analysis:

- **BaseBehavior** (`_representation.py`): Handles data structures, algebraic operations, and behavior construction
- **BehaviorAnalysisMixin** (`_analysis.py`): Provides optimization algorithms, agreement calculation, and sampling methods

**Dependencies**:
- `space.py`: Observable space definitions
- `context.py`: Context representation
- `frame.py`: Frame independence concepts

---

## Key Classes and Functions

### `class Behavior`

Core class for multi-perspective behavioral analysis.

**Attributes**:

* `space` *(Space)* — The observable space defining variables and their value sets
* `distributions` *(Dict[Context, Distribution])* — Per-context probability distributions

**Key Properties**:

* `agreement` *(float)* — Agreement coefficient α* ∈ [0,1]
* `contradiction_bits` *(float)* — Contradiction cost K = -log₂(α*)
* `context` *(List[Context])* — List of measurement contexts
* `worst_case_weights` *(Dict)* — Optimal witness weights λ*

### `Behavior.from_contexts(space, context_dists) -> Behavior`

Create behavior from context-specific probability distributions.

**Parameters**:

* `space` *(Space)* — Observable space definition
* `context_dists` *(Dict)* — Mapping from context keys to probability distributions

**Returns**:

* *(Behavior)* — New behavior instance

**Example**:

```python
space = Space.create(Morning=["Sun","Rain"], Evening=["Sun","Rain"])
behavior = Behavior.from_contexts(space, {
    ("Morning",): {("Sun",): 0.8, ("Rain",): 0.2},
    ("Evening",): {("Sun",): 0.7, ("Rain",): 0.3}
})
```

### `Behavior.agreement -> Agreement`

Fluent API for agreement calculations with various refinements.

**Returns**:

* *(Agreement)* — A query builder for fluent agreement calculations

**Example**:

```python
# Basic agreement (α*)
agreement = behavior.agreement.result  # float: overall agreement score

# Agreement by context
per_context_scores = behavior.agreement.by_context().context_scores  # {context: agreement}
bottleneck_score = behavior.agreement.by_context().result  # float: min of context scores

# Weighted agreement
weights = {("Morning",): 0.6, ("Evening",): 0.4}
weighted = behavior.agreement.for_weights(weights).result  # float: weighted agreement

# Agreement given feature distribution
hair_dist = np.array([0.7, 0.3])  # [blonde, brunette]
feature_based = behavior.agreement.for_feature("Hair", hair_dist).result  # float: agreement with fixed feature

# Get explanations and scenarios
theta = behavior.agreement.explanation  # np.ndarray: scenario distribution
scenarios = behavior.agreement.scenarios()  # list of (scenario, probability) pairs
```


### `Behavior.is_frame_independent(tol=1e-9) -> bool`

Test whether behavior can be explained by a single underlying model.

**Parameters**:

* `tol` *(float)* — Numerical tolerance for agreement check

**Returns**:

* *(bool)* — True if behavior is frame-independent

**Example**:

```python
if behavior.is_frame_independent():
    print("No contradiction detected")
else:
    print(f"Contradiction: {behavior.contradiction_bits:.3f} bits")
```

### `Behavior.sample_observations(n_samples, context_weights=None, seed=None)`

Generate synthetic observation data for testing and validation.

**Parameters**:

* `n_samples` *(int)* — Number of observations to generate
* `context_weights` *(Optional[np.ndarray])* — Custom context sampling weights
* `seed` *(Optional[int])* — Random seed for reproducibility

**Returns**:

* *(Tuple[np.ndarray, np.ndarray])* — Context indices and outcome indices

---

## Drawbacks or Gotchas

* **Finite alphabets required**: Only works with discrete, finite outcome spaces
* **Computational scaling**: Complexity grows exponentially with global assignment space size
* **Memory intensive**: Large numbers of contexts increase memory requirements
* **Cache management**: Manual cache invalidation needed after in-place distribution modifications
* **No continuous support**: Cannot handle continuous random variables or infinite outcome spaces

---

## Related Modules

* [`space.py`](../space.md) — Observable space definitions
* [`context.py`](../context.md) — Context representation
* [`observatory.py`](../observatory.md) — High-level analysis API

---

## See Also

* [Mathematical Theory Paper](../../docs/paper/A%20Mathematical%20Theory%20of%20Contradiction.pdf) — Formal foundations
* [Quickstart Examples](../../../examples/quickstart/behavior.py) — Practical usage
* [API Reference](../../docs/api/) — Complete reference documentation

---

```markdown
<!--
This file documents the public API of behavior.py.
For internal implementation details, see the source code.
Last updated 2025-09.
-->
```
