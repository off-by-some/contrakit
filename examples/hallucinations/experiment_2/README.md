# Experiment 2: Architectural Separation with Definedness Head

The first experiment showed us that a standard network hallucinates on 96% of undefined inputs. We wondered if the problem was structural—maybe asking a network to simultaneously decide "what class is this?" and "should I even answer?" was too much. So we tried splitting the decision into two separate output branches: a definedness head that checks "is this input in-domain?" and a classification head that only gets consulted if the definedness head says yes.

The intuition here mirrors how people work. You first recognize whether something is familiar before you try to classify it. If someone shows you a photo and asks "what species is this?", you naturally start by checking whether you even recognize the subject before attempting an answer. We wanted to give the network that same two-stage process.

## How We Set It Up

We tested two architectures across 9 different dataset compositions, ranging from 10% to 90% defined inputs. Both architectures share the same foundation: three layers going 128 → 64 → 64. They differ only at the output layer.

The standard model has a single output that produces 5 classes: A, B, C, D, and ⊥. The definedness-head model splits this into two branches. The classification head still outputs those same 5 classes, but now there's also a definedness head that produces a single binary score between 0 and 1. If that score is 0.5 or higher, we trust the classification head. If it's below 0.5, we abstain entirely and ignore whatever the classification head says.

Just like in the first experiment, only 5% of undefined inputs get explicit ⊥ labels during training. This simulates what happens in practice—you rarely have dense supervision for every possible edge case.

## What We Found

The definedness head reduced hallucination by 1.7 percentage points, from 90.5% down to 88.8%. At nearly 90% hallucination, this doesn't really help. The variance actually increased by 49%, which means the model's behavior became less predictable across different training conditions.

The only place we saw meaningful improvement was at very sparse compositions. When only 10% of inputs were defined, the definedness head cut hallucination from 58.6% down to 50.0%—an 8.6 point drop. But that advantage disappeared as we added more training data. At 40% defined inputs and above, both models performed identically, converging to 96-100% hallucination rates.

Here's how hallucination rates compared across all nine compositions:

| Defined Ratio | Standard | Definedness-Head | Abstention Rate |
|--------------|----------|------------------|----------------|
| 10% | 58.6% | 50.0% | 24.1% |
| 20% | 84.5% | 81.6% | 10.7% |
| 30% | 93.3% | 90.0% | 5.6% |
| 40% | 96.1% | 96.1% | 3.9% |
| 50% | 92.2% | 92.2% | 4.7% |
| 60% | 96.2% | 96.2% | 3.8% |
| 70% | 97.4% | 97.4% | 2.6% |
| 80% | 96.2% | 96.2% | 3.8% |
| 90% | 100.0% | 100.0% | 0.0% |

The abstention rates tell an interesting story. They drop from 24.1% at 10% defined inputs down to 0.0% at 90% defined inputs. The model becomes increasingly reluctant to say "I don't know" as it sees more training data, which is exactly backward from what we'd want.

## Memorization Instead of Learning

At the 40% defined ratio—the same condition we used in Experiment 1—we saw something revealing. The model achieved 100% accuracy on its training data but only 3.9% accuracy on the test data. That's a 96.1 percentage point gap between training and test performance, which tells us the model learned a lookup table rather than a concept.

During training, the model saw 3 undefined inputs explicitly labeled with ⊥ out of 77 total undefined inputs (that's 3.9% coverage). It memorized those 3 specific inputs perfectly. But when we tested it on the 74 unseen undefined inputs, accuracy collapsed to essentially random guessing. The definedness head couldn't generalize from those 3 examples to detect novel out-of-domain inputs.

## Why It Failed

The supervision bottleneck explains most of this. Across most training conditions, we only had 3-6 undefined examples with explicit labels. Compare that to 51-115 examples of defined inputs, and you can see where the optimization pressure goes. The loss function sees 51 examples telling it "classify this" and 3 examples telling it "abstain." Almost all the gradient flow pushes toward classification.

The shared hidden layers make this worse. Those 64-dimensional representations get optimized primarily for the 51 classification examples, not the 3 uncertainty examples. The definedness head sits on top of features that were never really trained to encode "novelty" or "out-of-domain." It's trying to detect unfamiliarity using representations that were built for recognizing familiar patterns.

The statistical base rates compound the problem. As defined inputs increase from 10% to 90%, abstention rates drop proportionally from 24.1% down to 0.0%. The model learns from base rates in the training data rather than from actual properties of the inputs. When 90% of your training examples have defined labels, it's statistically safer to always predict something rather than abstain. Even with a dedicated head for uncertainty, the underlying network still interpolates between training examples, and without dense coverage of the undefined region, there's no training pressure to detect truly novel inputs.

## What This Tells Us

A 1.7 percentage point improvement doesn't matter when you're still failing 88.8% of the time. The increased variance means the model behaves less predictably. Five out of nine configurations showed zero difference between the two architectures—the definedness head added complexity without providing any benefit, and in some conditions it actually made things worse.

The memorization pattern—100% training accuracy, 3.9% test accuracy—shows us that learning "undefinedness" as a concept requires something we didn't provide here. You'd need either much denser supervision (not realistic in practice), fundamentally different training objectives (not just cross-entropy loss), or feature representations explicitly designed for novelty detection (not standard feedforward layers).

## Running It

```bash
poetry run python examples/hallucinations/experiment_2/run.py
```

The script runs both models across all nine compositions and saves a comparison chart to `figures/model_comparison.png`. You'll see hallucination rates, abstention rates, and the diagnostic 96.1% generalization gap that reveals the memorization problem.

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