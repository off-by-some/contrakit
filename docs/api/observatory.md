# `observatory.py`

*High-level fluent API for building multi-perspective behaviors.*

---

## Overview

The `observatory.py` module defines the `Observatory` class, which provides a high-level, user-friendly interface for constructing behaviors in the contradiction framework. It serves as a fluent layer on top of the core components (`Space`, `Context`, `Distribution`, `Behavior`), allowing users to define concepts, assign probability distributions to perspectives, and generate behavior objects through lenses that model different viewpoints.

This module implements the primary user interface for behavioral modeling, making it easy to construct complex multi-perspective scenarios without directly manipulating low-level data structures.

---

## Why This Module Exists

Building behaviors from scratch requires coordinating multiple components: defining observable spaces, creating contexts, constructing probability distributions, and assembling them into behaviors. This process can be error-prone and verbose when done manually. The Observatory module addresses this by providing:

* **Fluent interface** for defining concepts and assigning distributions
* **Automatic validation** of probability constraints and context consistency
* **Lens system** for modeling different perspectives without base space pollution
* **High-level abstractions** that hide low-level implementation details

The module was created to make the contradiction framework accessible to users who want to focus on modeling scenarios rather than implementation mechanics.

---

## Architecture

The module implements a layered architecture with clear separation of concerns:

- **Core Classes**: `Observatory` as the main entry point, with supporting classes for concepts, lenses, and perspectives
- **Lens System**: Context managers that allow scoped perspective modeling without affecting the base space
- **Validation Layer**: Automatic checking of probability normalization and context consistency
- **Composition System**: Operations for combining multiple lenses with set-theoretic semantics

**Key Design Decisions**:
- **Fluent Interface**: Method chaining for natural workflow from concepts to behaviors
- **Lens Scoping**: Clean separation between base space and viewpoint-specific modeling
- **Automatic Validation**: Runtime checks prevent invalid probability assignments
- **Immutable Handles**: Concept and value handles prevent accidental state mutations

**Mathematical Foundations**:
Let $\mathcal{X}=\{X_1,\dots,X_n\}$ be concepts with finite alphabets $\mathcal{O}_{X_i}$. Contexts $c \subseteq \mathcal{X}$ have outcome spaces $\mathcal{O}_c:=\prod_{X\in c}\mathcal{O}_X$. Perspective assignments provide pmfs $p_c \in \Delta(\mathcal{O}_c)$, yielding behaviors $P=\{p_c : c\in\mathcal{C}\}$. Lenses extend this to tagged spaces $\mathcal{X}\cup\{L\}$ with projection back to base space.

---

## Key Classes and Functions

### `class Observatory`

Main entry point for defining concepts and managing perspectives.

**Construction**:

* `Observatory.create(symbols=None) -> Observatory` — Create with optional global alphabet

**Properties**:

* `alphabet -> tuple[ValueHandle, ...]` — Global alphabet as value handles
* `perspectives -> PerspectiveMap` — Distribution assignment interface

**Concept Management**:

* `concept(name, symbols=None) -> ConceptHandle` — Define new observable
* `define_many(specs) -> tuple[ConceptHandle, ...]` — Batch concept creation

**Lens Creation**:

* `lens(name, symbols=None) -> LensScope` — Create lens scope for viewpoint modeling

### `class ConceptHandle`

Represents an observable variable.

**Attributes**:

* `name: str` — Concept name
* `symbols: tuple[Any, ...]` — Alphabet symbols
* `alphabet: tuple[ValueHandle, ...]` — Value handles for symbols

**Operators**:

* `concept & other -> tuple` — Form context keys

### `class ValueHandle`

Typed handle for individual symbols.

**Attributes**:

* `value: Any` — Raw symbol value
* `concept: ConceptHandle` — Owning concept

**Operators**:

* `a & b -> tuple` — Build joint outcomes

### `class PerspectiveMap`

Manages probability distributions for contexts.

**Operations**:

* `__getitem__(key) -> DistributionWrapper` — Access distributions
* `__setitem__(key, value)` — Assign distributions with validation
* `add_joint(*observables, distribution)` — Multi-observable context assignment
* `validate(allow_empty=False) -> bool` — Check normalization
* `to_behavior(allow_empty=False) -> Behavior` — Generate behavior object

### `class LensScope`

Context manager for scoped perspective modeling.

**Properties**:

* `name: str` — Lens name
* `perspectives` — Proxy for assigning distributions
* `observatory: Observatory` — Parent observatory

**Concept Definition**:

* `define(name, symbols=None) -> ConceptHandle` — Define concepts within lens

**Behavior Export**:

* `to_behavior(allow_empty=False) -> Behavior` — Project to base space
* `to_behavior_raw(allow_empty=False) -> Behavior` — Tagged extended space

**Composition**:

* `compose(other) -> LensComposition` — Set union of contexts
* `intersection(other) -> LensComposition` — Set intersection
* `difference(other) -> LensComposition` — Set difference
* `symmetric_difference(other) -> LensComposition` — Symmetric difference

### `class LensComposition`

Immutable collection of lenses with composition semantics.

**Operations**:

* `composition | lens -> LensComposition` — Add lens to composition

**Analysis**:

* `perspective_contributions -> dict[str, float]` — Witness weights by lens
* `witness_distribution -> dict[str, float]` — Alias for contributions

**Export**:

* `to_behavior() -> Behavior` — Generate composed behavior

### Exceptions

* `NoConceptsDefinedError` — Raised when operations require concepts but none exist
* `EmptyBehaviorError` — Raised when generating behavior with no distributions

### Core Workflow

```python
# 1. Create observatory
obs = Observatory.create(symbols=["Yes", "No"])

# 2. Define concepts
voter = obs.concept("Voter")
candidate = obs.concept("Candidate", symbols=["Qualified", "Unqualified"])

# 3. Assign distributions
yes, no = voter.alphabet
obs.perspectives[voter] = {yes: 0.6, no: 0.4}

# 4. Generate behavior
behavior = obs.perspectives.to_behavior()

# 5. Use lenses for viewpoints
with obs.lens("Reviewer") as L:
    skill = L.define("Skill", symbols=["High", "Low"])
    L.perspectives[skill] = {skill.alphabet[0]: 0.7, skill.alphabet[1]: 0.3}
    base_behavior = L.to_behavior()  # Base space
    raw_behavior = L.to_behavior_raw()  # Extended space
```

---

## Drawbacks or Gotchas

* **Concept requirements**: Perspectives unavailable until concepts are defined
* **Snapshot semantics**: PerspectiveMap instances become stale after new concepts added
* **Validation overhead**: Runtime normalization checks add computational cost
* **Lens complexity**: Advanced lens composition may be unintuitive for simple use cases
* **Memory usage**: Handle objects and validation add overhead for large alphabets
* **No direct editing**: Immutable handles prevent in-place modifications

---

## Related Modules

* [`space.py`](../space.md) — Low-level observable space definitions
* [`context.py`](../context.md) — Context representation and operations
* [`distribution.py`](../distribution.md) — Probability distribution containers
* [`behavior.py`](../behavior.md) — Core behavioral analysis

---

## See Also

* [Mathematical Theory Paper](../../docs/paper/A%20Mathematical%20Theory%20of%20Contradiction.pdf) — Formal foundations of multi-perspective modeling
* [Quickstart Examples](../../../examples/quickstart/observatory.py) — Complete usage walkthrough
* [API Reference](../../docs/api/) — Complete reference documentation

---

```markdown
<!--
This file documents the public API of observatory.py.
For internal implementation details, see the source code.
Last updated 2025-09.
-->
```
