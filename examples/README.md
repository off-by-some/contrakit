# Examples: Demonstrating the Mathematical Theory of Contradiction
*Learn contrakit through hands-on applications across classical and quantum domains*

This directory contains practical demonstrations of the contrakit library. The examples show how to model different types of incompatible perspectives and measure their contradiction costs using the framework's core tools.

## 60-Second Quickstart

**Prerequisites**: Python 3.10+ and [Poetry](https://python-poetry.org/docs/#installation) installed

```bash
# From the contrakit repository root
poetry install
poetry run python examples/intuitions/day_or_night.py
```

**Note**: Some examples (statistics and quantum) require additional dependencies:
```bash
poetry install --with examples  # Install matplotlib and qutip for plotting/visualization
```

## Directory Structure
```
examples/
├── quickstart/              # Golden examples of core concepts
│   ├── __init__.py          # Package initialization
│   ├── space.py             # How observable spaces work
│   ├── context.py           # How observational contexts work
│   ├── distribution.py      # How probability distributions work
│   ├── behavior.py          # How multi-perspective behaviors work
│   ├── observatory.py       # How the high-level Observatory API works
│   └── run.py               # Execute all quickstart examples
├── intuitions/              # Building intuition for contradiction concepts
│   ├── __init__.py          # Package initialization
│   ├── day_or_night.py      # Multiple observer perspectives
│   ├── meta_lens.py         # Recursive framework application
│   └── run.py               # Execute all intuition examples
├── statistics/              # Statistical paradoxes and frame integration
│   ├── __init__.py          # Package initialization
│   ├── simpsons_paradox.py  # Frame integration and paradox resolution
│   └── run.py               # Execute all statistics examples
├── quantum/                 # Quantum modeling examples
│   ├── CHSH.py              # Bell inequality violations
│   ├── KCBS.py              # Quantum contextuality
│   ├── magic_squares.py     # Algebraic quantum contradictions
│   ├── run.py               # Execute all quantum examples
│   └── utils.py             # Shared visualization utilities
├── run.py                   # Execute all examples across directories
└── README.md                # This file
```

## What You'll Learn

These examples teach the contrakit library through concrete applications:

- **Quickstart concepts**: Golden examples of core ideas ([`quickstart/`](quickstart/))
  - Observable spaces, contexts, probability distributions, multi-perspective behaviors, and high-level Observatory API
- **Building intuition**: Core concepts through classical examples ([`intuitions/`](intuitions/))
- **Basic contradiction measurement**: How to define spaces, contexts, and behaviors
- **Recursive application**: Using the same framework at different organizational levels
- **Statistical paradoxes**: Frame integration to resolve contradictions ([`statistics/`](statistics/))
- **Quantum modeling**: Applying contrakit to Bell inequalities, contextuality, and logical paradoxes
- **Practical interpretation**: Understanding what K(P), α\*, and λ\* mean in real scenarios

Each example demonstrates specific aspects of the framework while building toward a complete understanding of how contradiction measurement works across domains.

## Core Framework Elements

The examples all use the same mathematical tools:

**K(P)**: Contradiction measure in bits—the information cost of reconciling incompatible perspectives  
**α\***: Best possible overlap with any frame-independent (classical) model  
**λ\***: Witness that identifies which contexts contribute most to contradictions  
**Invariant**: K(P) = 0 for classically explainable behavior, K(P) > 0 for quantum-like behavior

*For mathematical foundations and proofs, see [`../docs/paper.md`](../docs/paper.md)*

## Quickstart Examples

Examples are organized by topic: quickstart concepts, intuition-building, statistical paradoxes, and quantum applications. These examples you how to use the API, without much narrative.

1. **File: [`quickstart/space.py`](quickstart/space.py)** — Golden example showing how observable spaces work

2. **File: [`quickstart/context.py`](quickstart/context.py)** — Golden example showing how observational contexts work

3. **File: [`quickstart/distribution.py`](quickstart/distribution.py)** — Golden example showing how probability distributions work

4. **File: [`quickstart/behavior.py`](quickstart/behavior.py)** — Golden example showing how multi-perspective behaviors work

5. **File: [`quickstart/observatory.py`](quickstart/observatory.py)** — Golden example showing how the high-level Observatory API works


## Intuitive Examples
These examples show you how contrakit can be used for representing different perspectives.

6. **File: [`intuitions/day_or_night.py`](intuitions/day_or_night.py)** — Multiple observer perspectives using different valid measurement methods

   This example shows how to model situations where different observers use valid but incompatible methods to assess the same phenomenon. Two astronomers determine whether it's day or night using different approaches—one measures sky brightness directly, the other relies on building sensors.

   You'll learn how to define observer lenses with different measurement strategies, calculate agreement coefficients between methods, and use lens operations (union, intersection, difference) to analyze relationships. The example also demonstrates measuring how well each approach captures ground truth, providing a foundation for agreement measurement and lens operations when comparing multiple valid approaches to the same problem.

   ```python
   # Basic agreement measurement
   combined_behavior = harris_method.union(ward_method)
   agreement_coefficient = combined_behavior.agreement.result
   print(f"Methods align {agreement_coefficient*100:.1f}% of the time")

   # Or using the convenient | operator:
   agreement_coefficient = (harris_method | ward_method).agreement.result
   ```

7. **File: [`intuitions/meta_lens.py`](intuitions/meta_lens.py)** — Recursive application across organizational hierarchies (reviewers → supervisors → directors)

   This example demonstrates how to apply the contradiction measurement framework recursively across organizational levels. The example models a hiring process with three levels: reviewers evaluating candidates, supervisors evaluating reviewers, and directors evaluating supervisors.

   The key insight is that the framework scales naturally—"new level, same lens." You'll see how to apply the same mathematical structure at different hierarchical levels, measure contradiction costs at each level using identical tools, and understand how disagreement propagates across abstraction layers. The same math that measures disagreement between reviewers also measures disagreement between supervisors about those reviewers.

   ```python
   # Same contradiction measurement at different levels
   reviewer_disagreement = (alice | bob | charlie).to_behavior().contradiction_bits
   supervisor_disagreement = supervisor.to_behavior().contradiction_bits
   ```

8. **File: [`statistics/simpsons_paradox.py`](statistics/simpsons_paradox.py)** — Frame integration to resolve statistical contradictions by adding context variables

   This example shows how to resolve contradictions by adding context variables to the analysis—a technique called frame integration. The example uses Simpson's Paradox: teaching methods show opposite effectiveness patterns at two schools, but identical overall performance when combined.

   You'll learn to model incompatible statistical relationships using lens spaces, detect structural contradictions that traditional analysis misses, and resolve contradictions by including context (school) as an explicit variable. The example demonstrates that when perspectives clash due to hidden context differences, adding context variables can eliminate contradictions entirely.

   *Requires matplotlib for visualization. Install with `poetry install --with examples`*

   ```python
   # Frame integration: add context to resolve contradiction
   resolved_behavior = Behavior.from_counts(
       Space.create(Treatment=[...], Outcome=[...], School=[...]),
       contexts_with_school_info
   )
   print(f"K(P) drops to {resolved_behavior.contradiction_bits:.6f}")
   ```

## Quantum Modeling Examples

The quantum examples show how to model quantum mechanical phenomena using contrakit. They demonstrate different types of quantum behavior and their contradiction costs.

*Requires matplotlib and qutip for visualization and quantum computations. Install with `poetry install --with examples`*

9. **File: [`quantum/CHSH.py`](quantum/CHSH.py)** — Bell inequality violations and quantum correlation analysis

   This example demonstrates how to model quantum correlations that violate classical locality bounds using contrakit. The CHSH scenario involves two parties measuring entangled particles at different angles. Classical physics limits correlation strength to $S \leq 2$, while quantum mechanics can achieve $S = 2\sqrt{2} \approx 2.828$.

   You'll learn to model quantum joint probabilities from measurement angles, create behavior objects from correlation data, and verify the sharp boundary where K(P) = 0 for $S \leq 2$ but K(P) > 0 for $S > 2$. The example shows how to calculate contradiction costs (~0.012 bits at maximum violation†) and generate parameter sweeps to explore the classical-quantum boundary. The key insight is that quantum correlations exceeding classical bounds incur measurable information costs that grow smoothly with violation strength.

10. **File: [`quantum/KCBS.py`](quantum/KCBS.py)** — Quantum contextuality measurement scenarios

   This example provides a model of measurement contextuality using contrakit's framework. The KCBS scenario uses five quantum observables arranged in a pentagon where adjacent pairs can be measured together. Classical physics limits their expectation value sum to 2, while quantum mechanics can achieve $\sqrt{5} \approx 2.236$.

   The example teaches you to model exclusive measurement constraints (adjacent observables cannot both yield +1), create contextual behaviors from expectation values, and verify contextuality bounds while calculating violation costs. A notable finding is that different quantum phenomena (Bell nonlocality vs. contextuality) can have similar contradiction costs (~0.013 bits†, similar to CHSH) despite different physical origins.

11. **File: [`quantum/magic_squares.py`](quantum/magic_squares.py)** — Algebraic quantum contradictions that resolve classical logical impossibilities

   This example shows how to model quantum phenomena that resolve classical logical impossibilities. The Magic Square involves a 3×3 grid with parity constraints: rows multiply to +1, columns 1-2 multiply to +1, column 3 multiplies to -1. This is classically impossible but quantum mechanics achieves it.

   You'll learn to model parity constraints using probability distributions, create state-independent contextual behaviors, and verify the classical bound ($W \leq 4$) versus quantum achievement ($W = 6$). The example demonstrates calculating high contradiction costs (~0.132 bits†, ~10× higher than statistical violations) and analyzing perturbation robustness and noise tolerance. This reveals that algebraic quantum contradictions (logical impossibilities) cost significantly more information than statistical violations, establishing a taxonomy of quantum behavior types.

*† Exact values computed from minimax optimization—see [`../docs/paper.md`](../docs/paper.md) for derivations*

## Running the Examples

Run individual examples to learn specific aspects of the framework:

```bash
# Quickstart - fundamental concepts
poetry run python examples/quickstart/space.py
poetry run python examples/quickstart/context.py
poetry run python examples/quickstart/distribution.py
poetry run python examples/quickstart/behavior.py
poetry run python examples/quickstart/observatory.py

# Basic framework usage (from repo root)
poetry run python examples/intuitions/day_or_night.py
poetry run python examples/intuitions/meta_lens.py

# Statistics examples (require matplotlib)
poetry run python examples/statistics/simpsons_paradox.py

# Quantum applications (require matplotlib + qutip)
poetry run python -m examples.quantum.CHSH
poetry run python -m examples.quantum.KCBS
poetry run python -m examples.quantum.magic_squares
```

Run examples by category:

```bash
# Run all quickstart examples
poetry run python -m examples.quickstart.run

# Run all intuition-building examples
poetry run python -m examples.intuitions.run

# Run all statistical paradox examples (require matplotlib)
poetry run python -m examples.statistics.run

# Run all quantum examples (require matplotlib + qutip)
poetry run python -m examples.quantum.run
```

Run everything at once:

```bash
# Run all examples across all directories
poetry run python -m examples.run
```

**Expected outputs**:
- **Intuition examples** (`day_or_night.py`, `meta_lens.py`): Produces detailed narrative analyses explaining multi-perspective observation, lens operations, and contradiction measurements
- **Statistics examples** (`simpsons_paradox.py`): Generate analytical summaries and save visualizations to [`figures/`](../figures/)
- **Quantum examples** (`quantum/`): Produces detailed mathematical analyses with invariant checks and save visualization plots to [`figures/`](../figures/)
