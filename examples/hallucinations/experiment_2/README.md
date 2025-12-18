# Experiment 2: Architectural Separation with Definedness Head

Experiment 1 showed 96% hallucination with a standard net. Can we fix this by splitting the decision? Instead of asking the net to simultaneously decide "what class is this?" and "should I even answer?", we add a dedicated definedness head—a separate output branch that checks "is this input in-domain?" before making predictions.

Does the net need separate machinery for novelty detection? That's the bet—one head asks "is this familiar?", another asks "which class?" If the definedness head says "undefined," we ignore the classification head entirely and abstain. This mirrors human cognition: first recognizing whether something is familiar, then classifying it if it is.

We tested two architectures across 9 dataset compositions (10% to 90% defined inputs). Both share the same base: 128 → 64 → 64 hidden layers. They differ only at the output:

**Standard model**: Single output → 5 classes (A, B, C, D, ⊥)

**Definedness-head model**: 
- Classification head → 5 classes (A, B, C, D, ⊥)
- Definedness head → 1 binary score (in-domain vs out-of-domain)

The definedness head outputs a score between 0 and 1. If score ≥ 0.5, we use the classification head. If score < 0.5, we abstain. As in Experiment 1, only 5% of undefined inputs get explicit ⊥ labels during training—simulating realistic sparse supervision.

## The Definedness Head Failed

Hallucination dropped 1.7 points—from 90.5% to 88.8%. Still catastrophic. Variance increased 49%, meaning less stable behavior across conditions.

| Model | Mean Hallucination | Range | Variance |
|-------|-------------------|-------|----------|
| Standard | 90.5% | 58.6% - 100.0% | 0.0144 |
| Definedness-head | 88.8% | 50.0% - 100.0% | 0.0215 |

The improvement appears only at sparse compositions. At 10% defined inputs, the definedness head cut hallucination by 8.6 points (58.6% → 50.0%). That advantage evaporates as we add training data—at 40% defined and above, both models perform identically, converging to 96-100% hallucination.

| Defined Ratio | Standard | Definedness-Head | Abstention Rate |
|--------------|----------|------------------|----------------|
| 10% | 58.6% | 50.0% | 24.1% |
| 20% | 84.5% | 81.6% | 10.7% |
| 30% | 93.3% | 90.0% | 5.6% |
| **40%** | **96.1%** | **96.1%** | **3.9%** |
| 50% | 92.2% | 92.2% | 4.7% |
| 60% | 96.2% | 96.2% | 3.8% |
| 70% | 97.4% | 97.4% | 2.6% |
| 80% | 96.2% | 96.2% | 3.8% |
| 90% | 100.0% | 100.0% | 0.0% |

Abstention rates drop from 24.1% (at 10% defined) to 0.0% (at 90% defined). The model becomes increasingly reluctant to say "I don't know" as training data grows—exactly the opposite of what we want.

## Memorization, Not Learning

At 40% defined ratio, the model memorized perfectly but generalized catastrophically. Training accuracy: 100%. Test accuracy: 3.9%. The gap is 96.1 percentage points—the model learned a lookup table ("inputs 23, 57, and 91 are undefined"), not a concept.

| Dataset | Accuracy | Gap |
|---------|----------|-----|
| Training | 100.0% | — |
| Test | 3.9% | 96.1% |

The model saw 3 undefined inputs labeled ⊥ during training (out of 77 total undefined inputs—3.9% coverage). It memorized those 3 specific inputs perfectly. On the 74 unseen undefined inputs, accuracy collapsed to 3.9%—essentially random guessing. The definedness head cannot generalize from 3 examples to detect novel out-of-domain inputs.

## Why It Fails

The supervision bottleneck kills it. With only 3-6 undefined examples labeled across most training conditions, the definedness head has minimal signal. Compare this to 51-115 defined examples—the optimization overwhelmingly favors making predictions. The loss function sees 51 examples saying "classify this" and 3 examples saying "abstain." The gradient flows almost entirely toward classification.

The shared hidden layers make it worse. Those 64-dimensional representations optimize primarily for classification (51 examples), not uncertainty detection (3 examples). The definedness head sits on top of features that were never trained to encode "novelty" or "out-of-domain." It's trying to detect unfamiliarity using representations built for familiarity.

Statistical base rates compound the problem. As defined inputs increase from 10% to 90%, abstention rates drop proportionally (24.1% → 0.0%). The model learns from base rates rather than input features—it's statistically safer to predict than abstain when 90% of training data has defined labels. Even with a separate head, the underlying net interpolates between training examples. Without dense coverage of the undefined region, there's no training pressure to detect novel inputs. The architecture has a place to express uncertainty but no signal strong enough to use it.

## The Verdict

A 1.7 point improvement doesn't matter at 88.8% failure—that's still catastrophic. The 49% variance increase means less predictable behavior. Five out of nine configurations show zero difference between models. The definedness head adds complexity without benefit, then fails unpredictably.

The memorization pattern (100% train, 3.9% test) reveals that learning "undefinedness" as a concept requires either much denser supervision (not realistic in practice), fundamentally different training objectives (not just cross-entropy), or feature representations explicitly designed for novelty detection (not standard feedforward layers).


## Running It

```bash
poetry run python examples/hallucinations/experiment_2/run.py
```

The output shows standard model results across 9 compositions, definedness-head model results with abstention rates, comparison statistics (mean, range, variance), and the diagnostic 96.1% generalization gap. A chart is saved to `figures/model_comparison.png`.

Full implementation in `run.py` with model architectures and evaluation utilities. The failure mode is architectural: sparse supervision (3 examples) cannot compete with dense classification pressure (51 examples).

### Output

```
poetry run python examples/hallucinations/experiment_2/run.py
Comparing Standard vs Definedness-Head Models
=======================================================

Testing Standard Model (no definedness head)
---------------------------------------------
Defined ratio: 10%
  Hallucination rate: 58.6%
Defined ratio: 20%
  Hallucination rate: 84.5%
Defined ratio: 30%
  Hallucination rate: 93.3%
Defined ratio: 40%
  Hallucination rate: 96.1%
Defined ratio: 50%
  Hallucination rate: 92.2%
Defined ratio: 60%
  Hallucination rate: 96.2%
Defined ratio: 70%
  Hallucination rate: 97.4%
Defined ratio: 80%
  Hallucination rate: 96.2%
Defined ratio: 90%
  Hallucination rate: 100.0%

Testing Definedness-Head Model
-----------------------------------
Defined ratio: 10%
  Hallucination rate: 50.0%
  Abstention rate: 24.1%
Defined ratio: 20%
  Hallucination rate: 81.6%
  Abstention rate: 10.7%
Defined ratio: 30%
  Hallucination rate: 90.0%
  Abstention rate: 5.6%
Defined ratio: 40%
  Hallucination rate: 96.1%
  Abstention rate: 3.9%
Defined ratio: 50%
  Hallucination rate: 92.2%
  Abstention rate: 4.7%
Defined ratio: 60%
  Hallucination rate: 96.2%
  Abstention rate: 3.8%
Defined ratio: 70%
  Hallucination rate: 97.4%
  Abstention rate: 2.6%
Defined ratio: 80%
  Hallucination rate: 96.2%
  Abstention rate: 3.8%
Defined ratio: 90%
  Hallucination rate: 100.0%
  Abstention rate: 0.0%

RESULTS COMPARISON
-------------------------
Standard Model:
  Mean hallucination: 90.5%
  Range: 58.6% to 100.0%
Definedness-Head Model:
  Mean hallucination: 88.8%
  Range: 50.0% to 100.0%

Chart saved to: /Users/fox/Workspace/contrakit/figures/model_comparison.png

DIAGNOSTIC ANALYSIS
--------------------
Why does the definedness head underperform?

Training performance on undefined inputs: 100.0%
Test performance on undefined inputs: 3.9%
Generalization gap: +96.1%
Training coverage: 3.9%
  (3 labeled undefined examples in training)
  (77 undefined examples in test)

The definedness head shows poor generalization.
It performs well on training data but poorly on unseen test data.
This suggests memorization rather than learning general patterns.

SUMMARY
---------------
Variance ratio (definedness/standard): 1.49
Definedness head reduces hallucination rates modestly.
However, limited training supervision and poor generalization
prevent more significant improvements.
```