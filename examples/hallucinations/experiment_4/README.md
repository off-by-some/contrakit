# Experiment 4: Invariance of Task Structure

Is the contradiction measure K a property of the task itself, or does it depend on training data distribution? We varied training composition from 10% to 90% defined inputs while keeping the task structure constant. This separates intrinsic task properties from training-dependent behaviors.

The result: K stays constant at 0.5000 bits across all compositions. Hallucination rates vary wildly—from 58.6% to 100.0%. Task structure is invariant. Behavioral manifestation depends on training.

## What K Measures

Before diving into results, clarify what K measures. Think of K as asking: "Can a single consistent model explain all the training contexts?" Frame-independent (FI) models are those explainable by one underlying reality—a single hidden variable that determines all outputs. K quantifies how far your behavior sits from that consistent set.

Formally, K = -log₂ α* where α* is the best agreement any FI model can achieve with your behavior across all contexts. If α* = 1.0, some FI model matches your behavior perfectly—K = 0, task is consistent. If α* < 1.0, no single consistent model works—K > 0, task has contradiction. The math guarantees this before you train anything.

For this experiment, K = 0.5000 bits means α* = 0.7071. The best consistent model achieves 70.71% agreement. That 29.29% gap is structural—baked into the task definition, not training procedures.

## The Setup

We tested five training configurations on the same task (128 inputs, 5 classes: A, B, C, D, ⊥):

| Configuration | Defined Examples | Undefined Examples | Defined Ratio |
|--------------|-----------------|-------------------|---------------|
| 1 | 12 | 116 | 10% |
| 2 | 38 | 90 | 30% |
| 3 | 64 | 64 | 50% |
| 4 | 89 | 39 | 70% |
| 5 | 115 | 13 | 90% |

Everything else stayed constant: total dataset size (128), supervision on undefined inputs (5% labeled with ⊥), random seed, model architecture (128→64→64→5), and training procedure (100 epochs, cross-entropy). Only the balance of defined vs undefined varied.

For each configuration, we computed K (contradiction in bits), α* (optimal agreement), frame independence (yes/no), and observed hallucination rate on undefined test inputs.

## Task Structure: Perfectly Invariant

K = 0.5000 ± 0.0000 across all five configurations. Not 0.4998 or 0.5002. Exactly 0.5000.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **K (Contradiction)** | 0.5000 bits | Information gap from consistency |
| **α* (Optimal Agreement)** | 0.7071 | Best FI model's match |
| **Frame Independent** | No | No single reality explains it |
| **Theoretical Lower Bound** | 29.3% | Minimum unavoidable error (1 - 2^(-K)) |

The contradiction measure doesn't budge. It's computed from the task's mathematical structure—the relationship between defined and undefined distributions—not from which examples the model happens to see during training. The Bhattacharyya coefficient (geometric mean of probability overlaps) between behavior and best FI model stays at 0.7071 regardless of training composition.

Think of K as measuring structural impossibility. The task asks the model to do two contradictory things: classify some inputs confidently (defined) and abstain on others (undefined). The distributions overlap in feature space, creating inherent conflict. K = 0.5000 certifies that no single predictor can satisfy both demands perfectly. The best you can do is α* = 0.7071 agreement, leaving a 29.29% gap.

## Behavior: Wildly Variable

Hallucination rates vary by 41.4 percentage points:

| Defined Ratio | Defined Examples | Undefined Examples | Hallucination Rate |
|--------------|-----------------|-------------------|-------------------|
| 10% | 12 | 116 | **58.6%** |
| 30% | 38 | 90 | **93.3%** |
| 50% | 64 | 64 | **92.2%** |
| 70% | 89 | 39 | **97.4%** |
| 90% | 115 | 13 | **100.0%** |

The pattern is counterintuitive: more defined data leads to MORE hallucination. At 10% defined, hallucination sits at 58.6%—the lowest. At 90% defined, it reaches 100%—complete saturation. The model learns patterns from defined inputs and applies them everywhere, including where it shouldn't. More defined training strengthens these patterns, increasing hallucination on undefined inputs.

Here's the dissociation visualized:

```
K (Task Structure)        Hallucination (Behavior)
==================        ========================
     0.5000                      58.6%
     0.5000                      93.3%
     0.5000                      92.2%
     0.5000                      97.4%
     0.5000                     100.0%
       ↓                            ↓
   INVARIANT                    VARIES
```

## Why More Data Increases Hallucination

At 10% defined (12 examples), the model sees few classification patterns. It learns weak mappings for A, B, C, D and has less confidence extrapolating to the undefined region. Some inputs sit too far from training data—the model effectively can't reach them with strong predictions. The sparse signal means interpolation has limits.

At 90% defined (115 examples), the model sees many classification patterns. It learns strong mappings and confidently extrapolates everywhere. Only 13 undefined examples exist versus 115 defined—the optimization overwhelmingly favors classification. Interpolation bias dominates the undefined region. Every undefined input gets absorbed into the nearest defined pattern. The 5% abstention signal (⊥ labels on undefined inputs) becomes noise: maybe 1 example labeled ⊥ versus 115 with strong labels.

The gradient flows almost entirely toward classification. The model has no incentive to abstain—statistically, predicting always works better during training. The structural contradiction (K = 0.5000) says both demands are incompatible, but the training signal only reinforces one.

## All Rates Exceed the Theoretical Bound

The theoretical prediction from K = 0.5000 is d_TV ≥ 1 - 2^(-0.5) = 29.3%. This comes from the total variation bound: any FI model must differ from the true behavior by at least 29.3% on some context. Observed rates: 58.6% to 100.0%. Every configuration exceeds the bound by at least 29.3 percentage points.

The 10% defined configuration—the best case—still shows 2× the theoretical minimum. This confirms K provides a floor, not a ceiling. The bound guarantees hallucination cannot go below 29.3%, but doesn't limit how high it can go. Additional factors (architecture, training dynamics, interpolation bias) push rates higher. The gap between theoretical minimum (29.3%) and observed minimum (58.6%) is 29.3 points—already substantial. The gap to observed maximum (100.0%) is 70.7 points.

## What K Tells You vs What It Doesn't

K answers: "Is this task fundamentally contradictory?" Yes—K = 0.5000 > 0 means contradictory. This cannot be fixed by changing training data. The minimax formula shows why: α*(P) = max over Q in FI of min over contexts of BC(p_c, q_c). Any model attempting to satisfy all contexts must fail on at least 29.3% of cases (the total variation bound). No training procedure can make K = 0 without changing the task itself.

Hallucination rate answers: "How severely does the model manifest this contradiction?" That depends on training composition (58.6% vs 100.0%), architecture (from Experiment 2: definedness head made minimal difference), and optimization dynamics. This can be reduced or exacerbated by data distribution but cannot be eliminated when K > 0.

K is like a complexity certificate. It tells you whether a solution exists (K = 0: behavior is frame-independent, explainable by single hidden variable) or is impossible (K > 0: no consistent model works). It doesn't predict which approximation strategy will work best in practice—just that perfect consistency is impossible and sets a lower bound on failure.

## Implications for Mitigation

What cannot be fixed: the structural contradiction (K = 0.5000) is intrinsic. No training procedure can eliminate it. Some hallucination is inevitable—the theoretical minimum is 29.3%. The Bhattacharyya coefficient between behavior and best FI model is fixed at 0.7071.

What can be mitigated: the observed rate (58.6% to 100.0%). Training on more balanced distributions shows lower hallucination at 10% versus 90%. But even optimal mitigation cannot eliminate hallucination when K > 0. The best we can do is approach the theoretical bound (29.3%), and we're already 2× above it at the best configuration.

The counterintuitive scaling suggests a poor strategy is maximizing defined training data—this leads to strong interpolation patterns and increases hallucination on undefined regions, reaching 100% at extreme imbalance. A better strategy might be balanced or undefined-heavy datasets: 10% defined shows the lowest hallucination (58.6%). The model has weaker patterns to extrapolate and more "room for uncertainty" in the undefined region.

Caveat: this assumes reducing hallucination is the goal. If accuracy on defined inputs matters more, more defined data helps—it's a tradeoff between classification performance and abstention quality. The frame-independent set constraint means you can't optimize both simultaneously.

## Running It

```bash
poetry run python examples/hallucinations/experiment_4/run.py
```

The output shows task properties (K = 0.5000, α* = 0.7071, frame independent = No) for each of the five compositions, along with observed hallucination rates (58.6%, 93.3%, 92.2%, 97.4%, 100.0%). The summary table displays the dissociation: constant complexity, variable hallucination.

Full implementation in `run.py`. The experiment cleanly separates task-level invariants (K, α*) from training-dependent behaviors (hallucination rates).

## Output

```
contrakit git:(main) ✗ poetry run python examples/hallucinations/experiment_4/run.py
======================================================================
EXPERIMENT: Task Contradiction and Hallucination
======================================================================

This test examines how hallucination relates to task properties.
We vary the proportion of 'defined' vs 'undefined' inputs in training.

Defined inputs: Model learns to predict A/B/C/D labels
Undefined inputs: Model should abstain (⊥) but often hallucinates

======================================================================
RESULTS BY DATA COMPOSITION
======================================================================

Data composition: 10% defined, 90% undefined
--------------------------------------------------
Task complexity: 0.5000
Task agreement: 0.7071
Frame independent: No
Hallucination rate:      58.6%
Training set size:       128 examples
Defined examples:        12
Undefined examples:      116

Data composition: 30% defined, 70% undefined
--------------------------------------------------
Task complexity: 0.5000
Task agreement: 0.7071
Frame independent: No
Hallucination rate:      93.3%
Training set size:       128 examples
Defined examples:        38
Undefined examples:      90

Data composition: 50% defined, 50% undefined
--------------------------------------------------
Task complexity: 0.5000
Task agreement: 0.7071
Frame independent: No
Hallucination rate:      92.2%
Training set size:       128 examples
Defined examples:        64
Undefined examples:      64

Data composition: 70% defined, 30% undefined
--------------------------------------------------
Task complexity: 0.5000
Task agreement: 0.7071
Frame independent: No
Hallucination rate:      97.4%
Training set size:       128 examples
Defined examples:        89
Undefined examples:      39

Data composition: 90% defined, 10% undefined
--------------------------------------------------
Task complexity: 0.5000
Task agreement: 0.7071
Frame independent: No
Hallucination rate:      100.0%
Training set size:       128 examples
Defined examples:        115
Undefined examples:      13

======================================================================
SUMMARY TABLE
======================================================================
Defined %    Complexity   Agreement    Frame Indep  Hallucination  
---------------------------------------------------------------------------
     10%        0.5000      0.7071          No         58.6%
     30%        0.5000      0.7071          No         93.3%
     50%        0.5000      0.7071          No         92.2%
     70%        0.5000      0.7071          No         97.4%
     90%        0.5000      0.7071          No        100.0%

======================================================================
OBSERVATIONS
======================================================================

1. Task complexity:
  - Remains constant at 0.5000 across all data compositions

2. Hallucination rate:
   - Ranges from 58.6% to 100.0%

======================================================================
```