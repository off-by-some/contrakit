# Experiment 4: Invariance of Task Structure

We wanted to understand whether the contradiction measure K is a property of the task itself or whether it depends on how you distribute your training data. So we varied training composition from 10% to 90% defined inputs while keeping the task structure constant. This lets us separate intrinsic task properties from behaviors that depend on training.

The result shows a clean split: K stays constant at 0.5000 bits across all compositions. Meanwhile, hallucination rates swing wildly from 58.6% all the way up to 100.0%. Task structure is invariant. How that structure manifests in behavior depends entirely on training.

## What K Actually Measures

Before we dive into results, it helps to clarify what K is asking. You can think of it as measuring whether a single consistent model can explain all your training contexts. Frame-independent models are ones you can explain with a single underlying reality—one hidden variable that determines all the outputs. K quantifies how far your behavior sits from that consistent set.

Formally, K = -log₂ α* where α* is the best agreement any frame-independent model can achieve with your behavior across all contexts. If α* = 1.0, then some frame-independent model matches your behavior perfectly, which means K = 0 and the task is consistent. If α* < 1.0, no single consistent model works, which means K > 0 and the task has contradiction. The math guarantees this before you train anything.

For this experiment, K = 0.5000 bits means α* = 0.7071. The best consistent model can achieve 70.71% agreement with the behavior the task demands. That 29.29% gap is structural—it's baked into the task definition, not into training procedures.

## The Setup

We tested five training configurations on the same task, which uses 128 inputs and 5 classes (A, B, C, D, ⊥). The configurations ranged from 10% defined inputs (12 examples) up to 90% defined inputs (115 examples). Everything else stayed constant: total dataset size of 128, supervision on undefined inputs at 5% labeled with ⊥, same random seed, same model architecture (128→64→64→5), same training procedure with 100 epochs of cross-entropy. Only the balance between defined and undefined examples changed.

For each configuration, we computed K to measure the contradiction in bits, α* for optimal agreement, whether the task is frame-independent, and the observed hallucination rate on undefined test inputs.

## Task Structure Stays Perfectly Constant

K came out to 0.5000 ± 0.0000 across all five configurations. Not approximately 0.5000—exactly 0.5000 every time. The contradiction measure doesn't budge at all. It's computed from the task's mathematical structure, specifically the relationship between defined and undefined distributions, not from which particular examples the model happens to see during training. The Bhattacharyya coefficient (which measures the geometric mean of probability overlaps) between behavior and the best frame-independent model stays at 0.7071 regardless of training composition.

You can think of K as measuring structural impossibility. The task asks the model to do two contradictory things: classify some inputs confidently (the defined ones) and abstain on others (the undefined ones). The distributions overlap in feature space, creating inherent conflict. K = 0.5000 certifies that no single predictor can satisfy both demands perfectly. The best you can do is α* = 0.7071 agreement, leaving a 29.29% gap that no amount of training can close.

## Behavior Varies Wildly

While K stayed constant, hallucination rates varied by 41.4 percentage points:

| Defined Ratio | Defined Examples | Undefined Examples | Hallucination Rate |
|--------------|-----------------|-------------------|-------------------|
| 10% | 12 | 116 | 58.6% |
| 30% | 38 | 90 | 93.3% |
| 50% | 64 | 64 | 92.2% |
| 70% | 89 | 39 | 97.4% |
| 90% | 115 | 13 | 100.0% |

The pattern is counterintuitive. More defined training data leads to more hallucination, not less. At 10% defined, hallucination sits at 58.6%—the lowest we saw. At 90% defined, it reaches 100%—complete saturation. The model learns patterns from defined inputs and then applies them everywhere, including where it shouldn't. More defined training strengthens these patterns, which actually increases hallucination on undefined inputs.

Here's the dissociation laid out clearly:

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

## Why More Data Makes Things Worse

At 10% defined with only 12 examples, the model sees few classification patterns. It learns weak mappings for A, B, C, and D and has less confidence when extrapolating to the undefined region. Some inputs sit too far from the training data—the model effectively can't reach them with strong predictions. The sparse signal means interpolation has natural limits.

At 90% defined with 115 examples, the model sees many classification patterns. It learns strong mappings and confidently extrapolates everywhere. Only 13 undefined examples exist versus 115 defined ones, so the optimization overwhelmingly favors classification. Interpolation bias dominates the undefined region. Every undefined input gets absorbed into the nearest defined pattern. The 5% abstention signal—those ⊥ labels on undefined inputs—becomes noise in comparison: maybe 1 example labeled ⊥ versus 115 with strong labels.

The gradient flows almost entirely toward classification. The model has no statistical incentive to abstain because predicting always works better during training. The structural contradiction (K = 0.5000) says both demands are incompatible, but the training signal only reinforces one of them.

## All Rates Exceed the Theoretical Bound

The theoretical prediction from K = 0.5000 is that total variation d_TV ≥ 1 - 2^(-0.5) = 29.3%. This comes from the bound that says any frame-independent model must differ from the true behavior by at least 29.3% in some context. Our observed rates ranged from 58.6% to 100.0%. Every configuration exceeded the bound by at least 29.3 percentage points.

The 10% defined configuration—our best case—still shows double the theoretical minimum. This confirms that K provides a floor, not a ceiling. The bound guarantees hallucination cannot go below 29.3%, but it doesn't limit how high it can go. Additional factors like architecture, training dynamics, and interpolation bias push rates higher. The gap between the theoretical minimum (29.3%) and our observed minimum (58.6%) is already 29.3 points. The gap to our observed maximum (100.0%) is 70.7 points.

## What K Tells You and What It Doesn't

K answers the question "is this task fundamentally contradictory?" For us, yes—K = 0.5000 > 0 means the task is contradictory. This can't be fixed by changing training data. The minimax formula shows why: any model attempting to satisfy all contexts must fail on at least 29.3% of cases. No training procedure can make K = 0 without changing the task definition itself.

Hallucination rate answers a different question: "how severely does the model manifest this contradiction?" That depends on training composition (which gave us 58.6% versus 100.0%), architecture (from Experiment 2, we saw the definedness head made minimal difference), and optimization dynamics. Training data distribution can reduce or exacerbate the manifestation, but it can't eliminate the underlying problem when K > 0.

K works like a complexity certificate. It tells you whether a solution exists (K = 0 means behavior is frame-independent, explainable by a single hidden variable) or is impossible (K > 0 means no consistent model works). It doesn't predict which approximation strategy will work best in practice—just that perfect consistency is impossible and sets a lower bound on failure.

## What This Means for Mitigation

Some things can't be fixed. The structural contradiction (K = 0.5000) is intrinsic to the task. No training procedure can eliminate it. Some level of hallucination is inevitable—the theoretical minimum is 29.3%. The Bhattacharyya coefficient between behavior and the best frame-independent model is fixed at 0.7071.

Other things can be mitigated. The observed rate varies from 58.6% to 100.0% depending on training composition. Training on more balanced distributions shows lower hallucination at 10% versus 90%. But even optimal mitigation can't eliminate hallucination when K > 0. The best we can do is approach the theoretical bound of 29.3%, and we're already running at double that in our best configuration.

The counterintuitive scaling suggests that maximizing defined training data is actually a poor strategy—it leads to strong interpolation patterns and increases hallucination on undefined regions, reaching 100% at extreme imbalance. A better strategy might involve balanced or even undefined-heavy datasets. The 10% defined configuration showed the lowest hallucination at 58.6%. The model has weaker patterns to extrapolate from and more "room for uncertainty" in the undefined region.

There's a caveat here. This assumes reducing hallucination is your goal. If accuracy on defined inputs matters more, then more defined data helps—it's a tradeoff between classification performance and abstention quality. The frame-independent set constraint means you can't optimize both simultaneously.

## Running It

```bash
poetry run python examples/hallucinations/experiment_4/run.py
```

The script shows task properties (K = 0.5000, α* = 0.7071, frame independent = No) for each of the five compositions, along with observed hallucination rates. The summary table displays the dissociation: constant complexity, variable hallucination.

The full implementation is in `run.py`. The experiment cleanly separates task-level invariants (K, α*) from training-dependent behaviors (hallucination rates).

---

## Example Output

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