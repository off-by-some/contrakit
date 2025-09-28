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

Most tools treat disagreement as error—something to iron out until every model or expert agrees. But not all clashes are noise. Some are structural: valid perspectives that simply refuse to collapse into one account. **Contrakit is the first Python toolkit to measure that irreducible tension**, and to treat it as information—just as Shannon treated randomness. Our work has shown it's not only measurable, but it's useful too. 

It tells you things like:

1. How close can all perspectives get to a single account? (agreement $α^\star$)
2. How expensive is it to pretend they agree? (contradiction bits $K(P)$)
3. Which contexts drive the conflict? (witness weights $λ^\star$)

Think of it as an information-theoretic microscope for disagreement. Just as entropy priced *randomness*, Contrakit prices *contradiction*—so you can see exactly what it costs to flatten diverse perspectives into one.

## Quickstart
> **⚠️ Under Construction**: This project is currently under active development. Currently i'm in the process of translating all of my Coq formalizations, notebooks, and personal scripts into API functionality and documentation. The core functionality is ready to use, but APIs, documentation, and features will change.


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
Computational systems have long handled multiple perspectives—but only by forcing consensus or averaging them away. What has been missing is a way to measure epistemic tension itself: to treat contradiction not as noise, but as structured information.

Without this, information is lost. Standard models can’t register paradox as paradox; they flatten it. Contrakit flips the script: when experts or models disagree, you don’t lose information—you gain direction. Each contradiction becomes a gradient pointing toward the boundaries of current understanding. You don’t just resolve conflicts—you use them to build better models.

The loop is simple: perspectives clash → Contrakit measures the clash → $λ*$ shows you where to investigate → your next reasoning step is guided by the structure of the disagreement itself.

By quantifying epistemic tension, Contrakit shows not only how well multiple viewpoints can be reconciled, but what each viewpoint is capable of—how far it can stretch, where it breaks, and what it leaves out. In this way, contradiction becomes more than a clash; it becomes the lens that reveals what a “viewpoint” really is, and the information that drives resolution.


## The K(P) Tax
Contrakit’s measure of epistemic tension isn’t ad-hoc. It follows from six simple axioms about how perspectives should combine. From these, a unique formula emerges: contradiction bits K(P), built from the Bhattacharyya overlap between distributions. That’s why the measure behaves consistently across domains—from distributed consensus to ensemble learning to quantum contextuality.

And contradiction isn’t free. The same tension that guides reasoning also imposes an exact tax: across compression, communication, and simulation, disagreement costs K(P) bits per symbol. In practice, this means real performance deficits in any engineering task that must reconcile contextual data—unless you use the signal of contradiction itself to guide resolution.

| Task | Impact |
|---|---|
| Compression/shared representation | $+K(P)$ extra bits needed |
| Communication with disagreement | $-K(P)$ bits of capacity lost |
| Simulation with conflicting models | $×(2^{2K(P)} - 1)$ variance penalty |


We can now use $λ^\star$ to target measurements and understand where this will have the most impact. Reduce $K(P)$ by mixing in feasible "compromise" distributions.


## API Reference

* **Core classes:** [`Observatory`](https://github.com/off-by-some/contrakit/blob/main/docs/api/observatory.md), [`Behavior`](https://github.com/off-by-some/contrakit/blob/main/docs/api/behavior.md), [`Space`](https://github.com/off-by-some/contrakit/blob/main/docs/api/space.md)
* **Key properties:** `contradiction_bits`, `alpha_star` 
* **Key methods:** `least_favorable_lambda()`, `to_behavior()`
* **Full API:** [docs/api/](https://github.com/off-by-some/contrakit/tree/main/docs/api/) | **Theory:** [docs/paper/](https://github.com/off-by-some/contrakit/tree/main/docs/paper/)


## Examples

```bash
# Epistemic modeling examples
poetry run python examples/intuitions/day_or_night.py      # Observer perspective conflicts
poetry run python examples/statistics/simpsons_paradox.py # Statistical paradox resolution

# Quantum contextuality (writes analysis PNGs to figures/)
poetry run python -m examples.quantum.run
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

Contrakit is powered by a formal framework introduced in [A Mathematical Theory of Contradiction](https://zenodo.org/records/17203336). The paper lays out the six axioms, derives the unique measure K(P), and proves its consequences across compression, communication, and simulation. If you'd like to see the mathematics in full details, make suggestions, comments, or contribute check out [docs/paper/](https://github.com/off-by-some/contrakit/tree/main/docs/paper/)

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

