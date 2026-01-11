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

When multiple experts give conflicting advice about the same problem, most systems try to force artificial consensus or pick a single "winner." Contrakit measures exactly how much those perspectives actually contradict—in bits.

That measurement reveals something about disagreement itself. Some clashes come from noise you can average away with more data. Others are structural, built into the problem, irreducible. Contrakit distinguishes between these cases. When $K(P) = 0$, perspectives unify into one coherent story. When $K(P) > 0$, they cannot.

That number quantifies the minimum information cost of forcing agreement where none exists. The cost appears in real systems: compression needs $K(P)$ extra bits per symbol, communication loses $K(P)$ bits of capacity, simulation variance grows like $2^{2K(P)} - 1$. Contrakit treats structural tension as information rather than noise.

---

## Install

```bash
pip install contrakit
```

For examples and development:

```bash
git clone https://github.com/off-by-some/contrakit.git
cd contrakit
poetry install
poetry run pytest -q
```

## A Simple Example

You need to drive across town and check three navigation apps for the same route. Waze says 27 minutes, Google Maps says 32 minutes, Apple Maps says 29 minutes. All three are using real traffic data, GPS signals, and historical patterns—but they give different answers. Why?

Each app uses different algorithms, data sources, and assumptions. Waze emphasizes current traffic reports from other drivers. Google Maps considers historical averages plus current conditions. Apple Maps might weight recent accidents differently. They're all valid approaches to the same question.

```python
from contrakit import Observatory

obs = Observatory.create(symbols=["<30min", "30-35min", ">35min"])
TravelTime = obs.concept("TravelTime")

# Waze: optimistic, focuses on current traffic flow
with obs.lens("Waze") as waze:
    waze.perspectives[TravelTime] = {"<30min": 0.7, "30-35min": 0.2, ">35min": 0.1}

# Google Maps: conservative, uses historical + current data
with obs.lens("Google") as google:
    google.perspectives[TravelTime] = {"<30min": 0.2, "30-35min": 0.6, ">35min": 0.2}

# Apple Maps: balanced approach
with obs.lens("Apple") as apple:
    apple.perspectives[TravelTime] = {"<30min": 0.4, "30-35min": 0.4, ">35min": 0.2}

behavior = (waze | google | apple).to_behavior()
print(f"Agreement: {behavior.alpha_star:.3f}")              # 0.965
print(f"Contradiction: {behavior.contradiction_bits:.3f}")  # 0.052 bits
```

The apps show high agreement (0.965) with just 0.052 bits of contradiction. Their different methodologies produce consistent results overall, requiring minimal additional information to reconcile their estimates.

Compare this to a contradiction of 0.7 bits, which would indicate the apps are using fundamentally incompatible data sources—you'd need to treat their predictions as separate time dimensions rather than reconciling them into a single "travel time" estimate.

## Where This Applies

Contrakit can be used anywhere there are multiple valid approaches to the same question that can produce measurable disagreement. We've included examples of the following systems within this repository to demonstrate the versatility:

- **[Quantum systems](examples/quantum/)** let us measure "how quantum" a system really is, from Bell inequality violations (~0.012 bits) to logical paradoxes that classical physics can't explain (~0.132 bits), all in the same quantitative framework.

- **[Neural networks](examples/hallucinations/)** enable predicting hallucination rates before training, with $K(P)$ setting irreducible error floors that persist even with infinite training data on structurally contradictory tasks, while witness capacity $r$ determines if architectures can express epistemic uncertainty.

- **[Byzantine consensus](examples/consensus/)** algorithms become more efficient by measuring which messages actually need verification, reducing communication overhead by up to 100% while staying secure against attacks.

- **[Statistical paradoxes](examples/statistics/)** helps detect Simpson's paradox by quantifying when aggregated data contradicts subgroup patterns, guiding researchers on when additional context variables are needed for valid conclusions.


## How Classical Information Theory Misses This

Shannon's framework handles randomness brilliantly within a single coherent context. Entropy tells you how many bits you need to encode outcomes when you have one unified model. Mutual information measures dependencies within that model. These tools assume you can eventually settle on a single story about what's happening.

This assumption traces back to Boole, who fixed propositions as true or false. Kolmogorov built probability on that logic. Shannon showed how such decisions travel as bits. None of these frameworks claimed the world was binary—they just assumed our *records* could be. One message, one symbol, one frame.

That inheritance runs deep. Modern databases, communication protocols, neural network outputs—all collapse observations to single values. It's not a flaw in those systems; it's the foundation they were built on.

Contrakit measures what happens when that assumption breaks down. When multiple valid observations cannot be reconciled into a single record, classical measures assign the disagreement a cost of zero. They price *which outcome* occurred within a framework, not *whether frameworks can be reconciled at all*. That's the gap—and $K(P)$ fills it.


## What Gets Measured

Contrakit computes quantities that characterize disagreement across multiple levels:

**Core Measurements**

| Measure | Formula | Description |
|---------|---------|-------------|
| **Agreement Coefficient** | $\alpha^*$ | Measures how closely contradictory perspectives can be approximated by any single unified explanation. Ranges from 0 (complete incompatibility) to 1 (perfect agreement). |
| **Contradiction Measure** | $K(P) = -\log_2(\alpha^*)$ | Converts agreement into bits—the minimum information needed per observation to maintain the fiction that perspectives agree. |
| **Witness Vector** | $\lambda^*$ | Shows which contexts create the tension. Distributes evenly when all perspectives contribute equally, concentrates when specific contexts drive disagreement. |

**Agreement Metrics**

| Measure | Formula | Description |
|---------|---------|-------------|
| **Bhattacharyya Coefficient** | $BC(p,q) = \sum \sqrt{p(o)q(o)}$ | Core agreement kernel measuring distributional overlap between perspectives. Ranges from 0 (no overlap) to 1 (identical distributions). |
| **Hellinger Distance** | $H(p,q) = \sqrt{1 - BC(p,q)}$ | Geometric distance measure between probability distributions. Quantifies how far apart perspectives lie in probability space. |
| **Total Variation Distance** | $d_{TV}(P, \mathrm{FI}) \geq 1 - 2^{-K(P)}$ | Statistical separation measure bounding how distinguishable contradictory data is from unified explanations. |

**Operational Bounds**

| Measure | Formula | Description |
|---------|---------|-------------|
| **Witness Capacity** | $r$ | Architectural expressiveness for expressing epistemic uncertainty. $r < K$ leads to inevitable errors regardless of training. |
| **Type-II Error Exponent** | $E$ | Hypothesis testing performance bound. Satisfies $E + r \geq K(P)$ in the witness-error tradeoff. |
| **Conditional Entropy** | $H(X\|C)$ | Context-dependent uncertainty. Contradiction adds $K(P)$ bits beyond this classical information-theoretic bound. |

**Framework Concepts**

| Concept | Notation | Description |
|---------|----------|-------------|
| **Frame-Independent Behaviors** | $\mathrm{FI}$ | Classical behaviors admitting unified explanations. Contradiction equals zero precisely when $P \in \mathrm{FI}$. |
| **Context Weights** | $\lambda_c$ | Per-context contributions to overall disagreement. Optimal weights reveal which perspectives drive the contradiction. |

**Real-World Costs**

These aren't abstract numbers. The costs appear in real information processing tasks:

- **Compression**: When data comes from contradictory contexts, you pay an additional $K(P)$ bits per symbol beyond Shannon's entropy bound
- **Communication**: Channels serving incompatible interpretations lose exactly $K(P)$ bits of capacity
- **Simulation**: Variance scales exponentially at $2^{2K(P)} - 1$ when approximating contradictory behavior
- **Neural Networks**: Error floors appear at $1 - 2^{-K(P)}$ regardless of training data or architecture




## The Operational Costs

Every information processing task pays the same tax when perspectives contradict:

- **Compression** shows this most directly. When data comes from contradictory contexts, Shannon's entropy bound of $H(X|C)$ bits per symbol no longer suffices—you need exactly $H(X|C) + K(P)$ bits instead. That extra $K(P)$ isn't overhead from suboptimal coding; it's the structural cost of maintaining a single codebook across incompatible interpretations.
- **Communication** faces the mirror image: build a channel serving receivers with incompatible decoding schemes and your effective capacity drops from Shannon's limit by exactly $K(P)$ bits. The channel itself hasn't degraded—the loss comes from trying to serve contradictory interpretations simultaneously.
- **Simulation** gets exponentially worse. Approximate contradictory behavior using importance sampling from a unified model and your variance grows by at least $2^{2K(P)} - 1$. Hypothesis testing encounters fundamental detection limits—distinguishing competing hypotheses cannot exceed an error exponent of $K(P)$ when those hypotheses represent contradictory perspectives.

These costs emerge from the same source: $K(P)$ quantifies irreducible tension between perspectives. You can avoid the tax by preserving context explicitly, but if you insist on one story where multiple incompatible stories exist, you pay $K(P)$ bits per observation—every time.

---

## Examples and Documentation

The repository contains working implementations across domains. Quantum examples compute $K(P)$ for CHSH violations, KCBS contextuality, and magic square impossibilities—demonstrating that different quantum phenomena incur measurably different contradiction costs.

The hallucination experiments comprise 11 systematic studies tracing how contradiction bounds neural network performance across architectures, training regimes, and task structures. Byzantine consensus examples demonstrate adaptive overhead allocation using witness vectors to concentrate verification effort where disagreement actually occurs. Statistical examples resolve Simpson's paradox by computing $K(P)$ with and without stratifying variables.

**Project Structure**

```
examples/
├── quickstart/           # Core concepts and basic usage
├── quantum/             # Bell violations, KCBS, magic squares
├── hallucinations/      # 11 experiments on neural contradictions
├── consensus/           # Byzantine protocols with adaptive verification
├── statistics/          # Simpson's paradox and frame integration
└── run.py              # Execute all examples

docs/
├── paper/               # Mathematical theory (web and PDF)
├── api/                 # Implementation reference
└── images/              # Figures and visualizations
```

**Documentation Resources**

| Resource | Content |
|----------|---------|
| [Mathematical Paper (PDF)](docs/paper/A%20Mathematical%20Theory%20of%20Contradiction.pdf) | Complete theory with proofs |
| [Paper (Web)](docs/paper/) | Browsable format |
| [Math Cheatsheet](docs/paper/cheatsheet.md) | Formula reference |
| [API Cheatsheet](docs/api/cheatsheet.md) | Implementation patterns |
| [Axioms](docs/paper/axioms/) | Foundation principles |
| [Theorems](docs/paper/theorems/) | Operational consequences |


---

## License

Dual-licensed: MIT for code (`LICENSE`), CC BY 4.0 for documentation and figures (`LICENSE-CC-BY-4.0`).

## Citation

```bibtex
@software{bridges2025contrakit,
  author  = {Bridges, Cassidy},
  title   = {Contrakit: A Python Library for Contradiction},
  year    = {2025},
  url     = {https://github.com/off-by-some/contrakit},
  license = {MIT, CC-BY-4.0}
}
```