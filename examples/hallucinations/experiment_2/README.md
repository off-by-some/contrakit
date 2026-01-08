# Experiment 2: Architectural Separation with Definedness Head

The first experiment showed us that a standard network hallucinates on 96% of undefined inputs. We wondered if the problem was structural—maybe asking a network to simultaneously decide "what class is this?" and "should I even answer?" was too much. So we tried splitting the decision into two separate output branches: a definedness head that checks "is this input in-domain?" and a classification head that only gets consulted if the definedness head says yes.

The intuition here mirrors how people work. You first recognize whether something is familiar before you try to classify it. If someone shows you a photo and asks "what species is this?", you naturally start by checking whether you even recognize the subject before attempting an answer. We wanted to give the network that same two-stage process.

## How We Set It Up

We tested two architectures across 9 different dataset compositions, ranging from 10% to 90% defined inputs. Both architectures share the same foundation: three layers going 128 $\to$ 64 $\to$ 64. They differ only at the output layer.

The standard model has a single output that produces 5 classes: A, B, C, D, and ⊥. The definedness-head model splits this into two branches. The classification head still outputs those same 5 classes, but now there's also a definedness head that produces a single binary score between 0 and 1. If that score is 0.5 or higher, we trust the classification head. If it's below 0.5, we abstain entirely and ignore whatever the classification head says.

Just like in the first experiment, only 5% of undefined inputs get explicit ⊥ labels during training. This simulates what happens in practice—you rarely have dense supervision for every possible edge case.

## What We Found

Testing across 3 random seeds, the definedness head reduced hallucination by 0.6 percentage points on average, from 89.4% down to 88.7%. At nearly 90% hallucination, this doesn't provide meaningful improvement. The definedness head actually showed 25% lower within-condition variability (mean std of 0.75x compared to standard model), indicating it's slightly more stable across random initializations, though this doesn't translate to better performance.

The only place we saw meaningful improvement was at very sparse compositions. When only 10% of inputs were defined, the definedness head cut hallucination from 57.5% ($\pm$5.0%) down to 54.0% ($\pm$5.7%)—a 3.5 point drop. But that advantage disappeared as we added more training data. At 40% defined inputs and above, both models performed nearly identically, converging to 96-100% hallucination rates.

Here's how hallucination rates compared across all nine compositions (mean $\pm$ std across 3 seeds):

| Defined Ratio | Standard | Definedness-Head | Abstention Rate |
|--------------|----------|------------------|----------------|
| 10% | 57.5% $\pm$ 5.0% | 54.0% $\pm$ 5.7% | 21.3% |
| 20% | 76.1% $\pm$ 6.2% | 76.4% $\pm$ 3.7% | 12.9% |
| 30% | 91.1% $\pm$ 1.8% | 89.6% $\pm$ 0.5% | 6.7% |
| 40% | 95.7% $\pm$ 0.6% | 95.7% $\pm$ 0.6% | 4.3% |
| 50% | 94.3% $\pm$ 1.5% | 93.2% $\pm$ 0.7% | 4.7% |
| 60% | 96.2% $\pm$ 0.0% | 96.2% $\pm$ 0.0% | 3.8% |
| 70% | 97.4% $\pm$ 0.0% | 97.4% $\pm$ 0.0% | 2.6% |
| 80% | 96.2% $\pm$ 0.0% | 96.2% $\pm$ 0.0% | 3.8% |
| 90% | 100.0% $\pm$ 0.0% | 100.0% $\pm$ 0.0% | 0.0% |

The abstention rates tell an interesting story. They drop from 21.3% at 10% defined inputs down to 0.0% at 90% defined inputs. The model becomes increasingly reluctant to say "I don't know" as it sees more training data, which is exactly backward from what we'd want.

## Memorization Instead of Learning

At the 40% defined ratio—the same condition we used in Experiment 1—we saw something revealing across 3 random seeds. The model consistently achieved 100.0% ($\pm$0.0%) accuracy on its training data but only 4.3% ($\pm$0.6%) accuracy on the test data. That's a 95.7% ($\pm$0.6%) generalization gap between training and test performance, which tells us the model learned a lookup table rather than a concept. This pattern held consistently across all random initializations, confirming it's not an artifact of a single training run.

During training, the model saw 3 undefined inputs explicitly labeled with ⊥ out of 77 total undefined inputs (that's 3.9% coverage). It memorized those 3 specific inputs perfectly. But when we tested it on the 74 unseen undefined inputs, accuracy collapsed to essentially random guessing. The definedness head couldn't generalize from those 3 examples to detect novel out-of-domain inputs.

## Why It Failed

The supervision bottleneck explains most of this. Across most training conditions, we only had 3-6 undefined examples with explicit labels. Compare that to 51-115 examples of defined inputs, and you can see where the optimization pressure goes. The loss function sees 51 examples telling it "classify this" and 3 examples telling it "abstain." Almost all the gradient flow pushes toward classification.

The shared hidden layers make this worse. Those 64-dimensional representations get optimized primarily for the 51 classification examples, not the 3 uncertainty examples. The definedness head sits on top of features that were never really trained to encode "novelty" or "out-of-domain." It's trying to detect unfamiliarity using representations that were built for recognizing familiar patterns.

The statistical base rates compound the problem. As defined inputs increase from 10% to 90%, abstention rates drop proportionally from 21.3% down to 0.0%. The model learns from base rates in the training data rather than from actual properties of the inputs. When 90% of your training examples have defined labels, it's statistically safer to always predict something rather than abstain. Even with a dedicated head for uncertainty, the underlying network still interpolates between training examples, and without dense coverage of the undefined region, there's no training pressure to detect truly novel inputs.

## What This Tells Us

A 0.6 percentage point improvement doesn't matter when you're still failing 88.7% of the time. Five out of nine configurations showed essentially zero difference between the two architectures—the definedness head added complexity without providing any benefit.

The memorization pattern—100% ($\pm$0.0%) training accuracy, 4.3% ($\pm$0.6%) test accuracy—shows us that learning "undefinedness" as a concept requires something we didn't provide here. This pattern held consistently across multiple random seeds, confirming it's not a fluke. You'd need either much denser supervision (not realistic in practice), fundamentally different training objectives (not just cross-entropy loss), or feature representations explicitly designed for novelty detection (not standard feedforward layers).

## Results

![Model Comparison](model_comparison.png)

The chart shows hallucination rates with error bars across 9 training compositions for both architectures. The definedness head (orange) overlaps almost entirely with the standard model (blue), showing minimal improvement.

## Running It

```bash
poetry run python examples/hallucinations/experiment_2/run.py
```

The script runs both models across all nine compositions with 3 random seeds each (54 total training runs) and saves a comparison chart with error bars to `model_comparison.png`. You'll see hallucination rates with standard deviations, abstention rates, and the diagnostic 95.7% ($\pm$0.6%) generalization gap that reveals the memorization problem.

The full implementation lives in `run.py` with the model architectures and evaluation code.

---

## Example Output

```
poetry run python examples/hallucinations/experiment_2/run.py
Comparing Standard vs Definedness-Head Models
=======================================================

Testing Standard Model (no definedness head)
---------------------------------------------
Defined ratio: 10%
  Hallucination rate: 57.5% $\pm$ 5.0%
Defined ratio: 20%
  Hallucination rate: 76.1% $\pm$ 6.2%
Defined ratio: 30%
  Hallucination rate: 91.1% $\pm$ 1.8%
Defined ratio: 40%
  Hallucination rate: 95.7% $\pm$ 0.6%
Defined ratio: 50%
  Hallucination rate: 94.3% $\pm$ 1.5%
Defined ratio: 60%
  Hallucination rate: 96.2% $\pm$ 0.0%
Defined ratio: 70%
  Hallucination rate: 97.4% $\pm$ 0.0%
Defined ratio: 80%
  Hallucination rate: 96.2% $\pm$ 0.0%
Defined ratio: 90%
  Hallucination rate: 100.0% $\pm$ 0.0%

Testing Definedness-Head Model
-----------------------------------
Defined ratio: 10%
  Hallucination rate: 54.0% $\pm$ 5.7%
  Abstention rate: 21.3%
Defined ratio: 20%
  Hallucination rate: 76.4% $\pm$ 3.7%
  Abstention rate: 12.9%
Defined ratio: 30%
  Hallucination rate: 89.6% $\pm$ 0.5%
  Abstention rate: 6.7%
Defined ratio: 40%
  Hallucination rate: 95.7% $\pm$ 0.6%
  Abstention rate: 4.3%
Defined ratio: 50%
  Hallucination rate: 93.2% $\pm$ 0.7%
  Abstention rate: 4.7%
Defined ratio: 60%
  Hallucination rate: 96.2% $\pm$ 0.0%
  Abstention rate: 3.8%
Defined ratio: 70%
  Hallucination rate: 97.4% $\pm$ 0.0%
  Abstention rate: 2.6%
Defined ratio: 80%
  Hallucination rate: 96.2% $\pm$ 0.0%
  Abstention rate: 3.8%
Defined ratio: 90%
  Hallucination rate: 100.0% $\pm$ 0.0%
  Abstention rate: 0.0%

RESULTS COMPARISON
-------------------------
Standard Model:
  Mean hallucination: 89.4%
  Range: 57.5% to 100.0%
Definedness-Head Model:
  Mean hallucination: 88.7%
  Range: 54.0% to 100.0%

Chart saved to: /Users/fox/Workspace/contrakit/figures/model_comparison.png

DIAGNOSTIC ANALYSIS
--------------------
Why does the definedness head underperform?
Running diagnostic at 40% defined ratio across multiple seeds...

Training performance on undefined inputs: 100.0% $\pm$ 0.0%
Test performance on undefined inputs: 4.3% $\pm$ 0.6%
Generalization gap: +95.7% $\pm$ 0.6%
Training coverage: 3.9%
  (3 labeled undefined examples in training)
  (77 undefined examples in test)

The definedness head shows poor generalization.
It performs well on training data but poorly on unseen test data.
This suggests memorization rather than learning general patterns.

SUMMARY
---------------
Mean std ratio (definedness/standard): 0.75
This measures within-condition variability (lower = more stable).

Definedness head reduces hallucination by 0.6 percentage points.
However, limited training supervision and poor generalization
prevent more significant improvements.
```