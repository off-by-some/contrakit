# Contrakit: A Python Library for Contradiction
*Measure the information cost of incompatible perspectives*

Contrakit implements the mathematical theory of contradiction—a framework for quantifying when multiple valid observations cannot be reconciled within a single coherent explanation. It provides tools to measure contradiction costs in bits, identify optimal detection strategies, and analyze the boundaries between classical and quantum-like behavior across domains.

## Quickstart

**Prerequisites**: Python 3.9+ and [Poetry](https://python-poetry.org/docs/#installation)

```bash
# Clone and install
git clone https://github.com/off-by-some/contrakit.git && cd contrakit
poetry install

# Run a basic example
poetry run python examples/day_or_night.py

# Run quantum examples (generates figures/)
poetry run python -m examples.quantum.run
```

Results appear in the [`figures/`](figures/) directory as PNG visualizations.

## What This Measures

The framework provides three core quantities:

- **K(P)**: Contradiction measure in bits—the information cost of forcing incompatible perspectives into one story
- **α\***: Best possible overlap with any classical (frame-independent) model  
- **λ\***: Optimal strategy for detecting contradictions across contexts

**Key insight**: K(P) = 0 for classically explainable behavior, K(P) > 0 for quantum-like contradictions.

## Core Usage Pattern

```python
from contrakit import Space, Behavior

# 1. Define measurement space
space = Space.create(
    A0=[-1, +1], A1=[-1, +1],  # Alice's measurements
    B0=[-1, +1], B1=[-1, +1]   # Bob's measurements
)

# 2. Specify experimental contexts and probabilities
contexts = {
    ("A0", "B0"): {(+1,+1): 0.25, (+1,-1): 0.25, (-1,+1): 0.25, (-1,-1): 0.25},
    ("A0", "B1"): {(+1,+1): 0.427, (+1,-1): 0.073, (-1,+1): 0.073, (-1,-1): 0.427},
    # ... more measurement combinations
}

# 3. Analyze contradiction
behavior = Behavior.from_contexts(space, contexts)
print(f"Contradiction cost: {behavior.contradiction_bits:.3f} bits")
print(f"Classical overlap: {behavior.alpha_star:.3f}")

# 4. Find optimal detection strategy  
witness = behavior.least_favorable_lambda()
print(f"Focus on contexts: {witness}")
```

## Examples and Applications

The [`examples/`](examples/) directory demonstrates the framework across different domains:

**File: [`day_or_night.py`](examples/day_or_night.py)** — Multiple observer perspectives using different valid measurement methods

**File: [`meta_lens.py`](examples/meta_lens.py)** — Recursive application across organizational hierarchies (reviewers → supervisors → directors)

**File: [`simpsons_paradox.py`](examples/simpsons_paradox.py)** — Frame integration to resolve statistical contradictions by adding context variables

**File: [`quantum/CHSH.py`](examples/quantum/CHSH.py)** — Bell inequality violations and quantum correlation analysis

**File: [`quantum/KCBS.py`](examples/quantum/KCBS.py)** — Quantum contextuality measurement scenarios

**File: [`quantum/magic_squares.py`](examples/quantum/magic_squares.py)** — Algebraic quantum contradictions that resolve classical logical impossibilities

Run all examples:
```bash
# Individual examples
poetry run python examples/day_or_night.py
poetry run python examples/simpsons_paradox.py

# All quantum examples with visualizations
poetry run python -m examples.quantum.run
```

## Contradiction Cost Taxonomy

The examples reveal different types of contradictions:

| Type | Cost (bits) | Examples | Properties |
|------|-------------|----------|------------|
| Probabilistic | ~0.012 | CHSH, KCBS | Statistical violations, state-dependent |
| Algebraic | ~0.132 | Magic Square | Logical impossibilities, state-independent |  
| Hierarchical | Variable | Meta-lenses | Scales with organizational complexity |

## Limitations and Caveats

- **Finite domains only**: Framework requires finite outcome alphabets and context sets
- **No noise modeling**: Current implementation assumes perfect measurements
- **Computational scaling**: Large context sets may require approximation methods
- **Deterministic results**: Examples use fixed random seeds for reproducibility

## Troubleshooting

**Poetry not found**: Install Poetry first: `curl -sSL https://install.python-poetry.org | python3 -`

**Module import errors**: Ensure you're in the contrakit directory and run `poetry install`

**Missing figures directory**: The `figures/` directory is created automatically when running quantum examples

## Documentation and Theory

- **Mathematical foundations**: [`docs/paper.md`](docs/paper.md) — Complete theoretical framework
- **Implementation details**: [`docs/all.md`](docs/all.md) — Full API documentation  
- **Research paper**: [`docs/paper/`](docs/paper/) — Formal mathematical treatment

## Contributing

This library implements research from *A Mathematical Theory of Contradiction*. For contributions:

1. Check existing [issues](https://github.com/off-by-some/contrakit/issues) and examples
2. Follow the established patterns in [`examples/`](examples/)
3. Add tests for new functionality in [`tests/`](tests/)
4. Update documentation as needed

## License

This repository is **dual-licensed**:
- **Code**: MIT License — see [`LICENSE`](LICENSE)
- **Documentation, figures, and written materials**: CC BY 4.0 — see [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0)

## Citation

```bibtex
@software{bridges2025contrakit,
  author = {Bridges, Cassidy},
  title = {Contrakit: A Python Library for Contradiction},
  year = {2025},
  url = {https://github.com/off-by-some/contrakit},
  license = {MIT, CC-BY-4.0}
}
```
