# Contrakit

<p align="center">
  <img src="https://raw.githubusercontent.com/off-by-some/contrakit/main/docs/images/contrakit-banner.png" height="300" alt="Contrakit banner">
</p>

<p align="center">
  <a href="https://github.com/off-by-some/contrakit"><img src="https://img.shields.io/github/stars/off-by-some/contrakit?style=flat" alt="GitHub Stars"></a>
  <a href="https://github.com/off-by-some/contrakit"><img src="https://img.shields.io/github/forks/off-by-some/contrakit?style=flat" alt="GitHub Forks"></a>
  <a href="https://github.com/off-by-some/contrakit/issues"><img src="https://img.shields.io/github/issues/off-by-some/contrakit" alt="GitHub Issues"></a>
  <a href="https://pypi.org/project/contrakit/"><img src="https://img.shields.io/pypi/v/contrakit?label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/contrakit/"><img src="https://img.shields.io/pypi/pyversions/contrakit" alt="Python"></a>
  <a href="https://github.com/off-by-some/contrakit/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/off-by-some/contrakit/tree/main/docs"><img src="https://img.shields.io/badge/docs-reference-blue.svg" alt="Docs"></a>
</p>

When multiple experts give conflicting advice about the same problem, most systems try to force artificial consensus or pick a single "winner." 

**Contrakit takes a different approach:** it measures exactly how much those perspectives actually contradict—in bits.


## What is Contrakit?

Most tools treat disagreement as error—something to iron out until every model or expert agrees. But not all clashes are noise. Some are structural: valid perspectives that simply refuse to collapse into one account. **Contrakit is the first Python toolkit to measure that irreducible tension**, and to treat it as information—just as Shannon treated randomness. 

Our work has shown it's not only measurable, but it's useful too. 


## But What Does it Do, Practically?

K(P) is a universal yardstick that quantifies structural disagreement — wherever it appears. We've applied contrakit across wildly different fields and found surprisingly consistent behavior:

* In quantum systems, $K(P)$ measures "how quantum" a system is—whether you're looking at Bell inequalities, KCBS polytopes, or magic squares, the measure stays consistent and comparable ([quantum examples](examples/quantum/)). 
* In neural networks, $K(P)$ computed from task structure alone predicts minimum hallucination rates before any training happens ([hallucination experiments](examples/hallucinations/)). 
* In statistical paradoxes like Simpson's, $K(P)$ reveals exactly how much the aggregated view contradicts the stratified view ([statistical examples](examples/statistics/)), even in cases MI returns 0.
* In consensus algorithms, You can use it to measure how much real disagreement/conflict exists, and only spend extra checking effort exactly where the trouble actually is. ([consensus examples](examples/consensus/))



## Quickstart
**Install:**

```bash
pip install contrakit
```

**Quickstart:**

```python
from contrakit import Observatory

# 1) Model perspectives
obs = Observatory.create(symbols=["Yes","No"])
Y = obs.concept("Outcome")
with obs.lens("ExpertA") as A: A.perspectives[Y] = {"Yes": 0.8, "No": 0.2}
with obs.lens("ExpertB") as B: B.perspectives[Y] = {"Yes": 0.3, "No": 0.7}

# 2) Export behavior and quantify reconcilability
behavior = (A | B).to_behavior()  # compose lenses → behavior
print("alpha*:", round(behavior.alpha_star, 3))  # 0.965 (high agreement)
print("K(P):  ", round(behavior.contradiction_bits, 3), "bits")  # 0.051 bits (low cost)

# 3) Where to look next (witness design)
witness = behavior.least_favorable_lambda()
print("lambda*:", witness)  # ~0.5 each expert (balanced conflict)
```

## Why This Matters

When perspectives clash, three quantities emerge. $α^\star$ measures how close they can get to a single account—the best-case agreement coefficient. $K(P)$ measures the cost of forcing consensus—the bits you pay to pretend they agree. $λ^\star$ identifies which contexts drive the conflict—where the tension concentrates.

Just as entropy priced randomness, $K(P)$ prices contradiction. In [quantum contextuality](examples/quantum/), it measures which measurement scenarios create irreducible tension. In [neural network hallucination](examples/hallucinations/), it predicts minimum error rates from task structure before training. In [statistical paradoxes](examples/statistics/), it quantifies how much aggregated and stratified views contradict.

Computational systems have long handled multiple perspectives by forcing consensus or averaging them away. Contrakit measures epistemic tension itself, treating contradiction as structured information rather than noise. When experts or models disagree, each contradiction points toward boundaries of current understanding. 

When perspectives clash, contrakit measures it, $λ^\star$ reveals where to investigate, and the structure of disagreement guides the next reasoning step.

Quantifying epistemic tension reveals not only how well multiple viewpoints can be reconciled, but what each viewpoint is capable of—how far it can stretch, where it breaks, and what it leaves out.


## The K(P) Tax

The measure follows from [six axioms](docs/paper/axioms/axioms.md) about how perspectives should combine. From these, a unique formula emerges: contradiction bits $K(P)$, built from the Bhattacharyya overlap between distributions. The measure behaves consistently across domains—from distributed consensus to ensemble learning to quantum contextuality.

Contradiction imposes an exact cost. Across [compression, communication, and simulation](docs/paper/theorems/2.%20Operational%20Theorems/), disagreement costs $K(P)$ bits per symbol. Engineering tasks that must reconcile contextual data face real performance deficits—compression needs extra bits, communication loses capacity, simulation incurs variance penalties.

| Task | Impact |
|---|---|
| Compression/shared representation | $+K(P)$ extra bits needed |
| Communication with disagreement | $-K(P)$ bits of capacity lost |
| Simulation with conflicting models | $×(2^{2K(P)} - 1)$ variance penalty |


$λ^\star$ targets measurements where contradiction concentrates. Mixing in feasible "compromise" distributions reduces $K(P)$.


## API Reference

* **Core classes:** [`Observatory`](docs/api/observatory.md) for modeling perspectives, [`Behavior`](docs/api/behavior.md) for analyzing distributions, [`Space`](docs/api/space.md) for defining observable systems
* **Key properties:** `contradiction_bits` (the $K(P)$ measure), `alpha_star` (maximum agreement coefficient)
* **Key methods:** `least_favorable_lambda()` (witness weights showing where conflict concentrates), `to_behavior()` (convert lens compositions to analyzable behaviors)
* **Full docs:** [API reference](docs/api/) | [Mathematical theory](docs/paper/)


## Examples

* The [examples/intuitions/](examples/intuitions/) directory contains observer perspective conflicts. 
* [examples/statistics/](examples/statistics/) resolves Simpson's paradox using $K(P)$. 
* [examples/quantum/](examples/quantum/) measures contradiction across Bell, KCBS, and magic square scenarios. 
* [examples/hallucinations/](examples/hallucinations/) demonstrates neural network hallucination prediction from task structure.

```bash
# Epistemic modeling
poetry run python examples/intuitions/day_or_night.py
poetry run python examples/statistics/simpsons_paradox.py

# Quantum contextuality (writes analysis to figures/)
poetry run python -m examples.quantum.run

# Neural network hallucination experiments
poetry run python examples/hallucinations/run.py
```

## Installing from Source


```bash
# Clone the repository
$ git clone https://github.com/off-by-some/contrakit.git && cd contrakit

# Install dependencies
$ poetry install

# Run tests
$ poetry run pytest -q
```


## A Mathematical Theory of Contradiction

Contrakit implements the formal framework from [A Mathematical Theory of Contradiction](https://zenodo.org/records/17203336). The paper presents six axioms about how perspectives should combine, derives the unique measure $K(P)$, and proves its consequences across compression, communication, and simulation. Mathematical details, proofs, and axiom structure are in [docs/paper/](docs/paper/).

## License

Dual-licensed: **MIT** for code (`LICENSE`), **CC BY 4.0** for docs/figures (`LICENSE-CC-BY-4.0`).

## Citation

```bibtex
@software{bridges2025contrakit,
  author = {Bridges, Cassidy},
  title  = {Contrakit: A Python Library for Contradiction},
  year   = {2025},
  url    = {https://github.com/off-by-some/contrakit},
  license= {MIT, CC-BY-4.0}
}
```

