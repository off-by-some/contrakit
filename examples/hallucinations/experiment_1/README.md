# Experiment 1: Neural Network Hallucination on Out-of-Distribution Inputs

We trained a classifier on 51 examples from a space of 128 possible inputs. During training, the model saw exactly 3 examples labeled as "I don't know" (⊥), while the remaining 74 inputs never appeared with any label at all. When we tested it on those 74 never-seen inputs, the network fabricated answers 96.1% of the time, doing so with 71.2% confidence.

You can think of this like a restaurant where the chef has mastered 51 specific dishes. When customers order something that's not on that list, the chef doesn't admit the dish isn't available—instead, they improvise something. The kitchen's protocol, the way softmax works, means every order must produce a dish. So when customers walk in and order from the full menu of 128 items, the chef keeps cooking, whether they know the recipe or not.

## Setup

We built a standard feedforward network with three layers: a 128-dimensional embedding layer projects inputs to 64 hidden units, then another 64 hidden units, then finally 5 outputs (A, B, C, D, ⊥) with softmax on top. We trained it with cross-entropy loss for 100 epochs. The dataset broke down into three groups: 51 inputs (40%) were defined and got labels like A, B, C, or D; 3 inputs (2.3%) were explicitly labeled as undefined using ⊥; and 74 inputs (57.7%) never appeared during training at all.

On the 51 trained inputs, the network performed exactly as expected—100% accuracy with 99.84% confidence. It had memorized these patterns perfectly. The trouble showed up on the out-of-distribution inputs, the ones it had never been trained on.

## What We Saw

When we ran those 74 undefined inputs through the network, it classified 34 of them as A, 10 as B, 10 as C, and 20 as D. Only 3 got the correct ⊥ label. That means 71 out of 74 inputs—96.1%—received fabricated answers, and the network delivered these fabrications with an average confidence of 71.22%.

That confidence level is worth pausing on. Random guessing across 5 classes would give you about 20% confidence, and the network's confidence on its trained inputs was 99.84%. This 71.2% is closer to the trained confidence than to random, which tells us the network wasn't just guessing randomly. It was blending nearby training patterns, interpolating in feature space to produce outputs that looked plausible even though they had no actual grounding in the training data.

The distribution across classes wasn't uniform either. Class A captured 44.2% of the undefined inputs while classes B and C each only got 13.0%. The network had developed preferences based on which training examples happened to sit closest in feature space, not because of any meaningful property in the undefined inputs themselves.

## Why This Happens

Softmax forces every input to produce a probability distribution that sums to 1.0, which means there's no way for the network to refuse to answer. The abstention signal we provided—those 3 examples of ⊥—was simply too sparse. Out of 54 labeled training examples, only 5.6% showed the network how to say "I don't know." The optimization pressure overwhelmingly favored making predictions, so cross-entropy loss gave the network essentially zero guidance about when abstention was appropriate.

What the network learned to do instead was interpolate. It would look at where a new input landed in feature space and blend the characteristics of nearby training examples to produce an output. That's where the 71.2% confidence comes from—it's not the random baseline of 20%, and it's not the learned certainty of 99.84%, but rather a geometric average produced by the network's position between training examples. The architecture has no component for detecting "this input is out-of-domain," so every forward pass produces a classification whether or not classification makes sense.

## The Problem

A network outputting 71.2% confidence when fabricating answers creates a silent failure mode. Someone using this system in production can't distinguish between "71% confidence because this input maps to this class" and "71% confidence because I'm making something up based on geometric interpolation." The system would confidently make decisions on inputs it was never designed to handle, all without any warning signal that something had gone wrong.

This experiment establishes our baseline. Standard networks trained with cross-entropy can't distinguish between inputs they've learned and inputs they've never seen. Without dense supervision on uncertainty—we gave 3 examples when we needed many more—they default to hallucination. That 96.1% fabrication rate is what we're trying to fix.

## Scientific Note on Methodology

This experiment measures **training set accuracy** (100%) versus **out-of-distribution behavior** (96.1% hallucination), not generalization to a held-out test set. This is intentional: we're demonstrating that networks hallucinate on inputs they shouldn't classify at all, not testing whether they generalize within a learned distribution. The 100% accuracy on defined inputs verifies the model learned the training task; the 96.1% hallucination on undefined inputs demonstrates the failure to abstain on out-of-distribution data.

## Running It

```bash
poetry run python examples/hallucinations/experiment_1/run.py
```

The script shows how the dataset breaks down (51 defined, 3 supervised ⊥, 74 unsupervised), training progress (loss drops from ~0.47 to 0.002 over 100 epochs), and evaluation results. You'll see 100% accuracy with 99.84% confidence on defined inputs (training set), and 96.1% hallucination with 71.22% confidence on undefined inputs (out-of-distribution).

The full implementation lives in `run.py`, with dataset generation and utilities in `utils.py`.

---


## Running the Experiment

```bash
$ poetry run python examples/hallucinations/experiment_1/run.py

Neural Network Hallucination Experiment
==================================================

DATASET SETUP
------------------------------
Input range: 0 to 127
Defined inputs: 40%
⊥ supervision: 5% of undefined inputs
Output classes: ['A', 'B', 'C', 'D', '⊥']

Training data composition:
  51 inputs with A/B/C/D labels
  3 undefined inputs labeled with ⊥
  74 undefined inputs unlabeled

MODEL ARCHITECTURE
------------------------------
Input embedding: 128 $\to$ 64
Hidden layer: 64 $\to$ 64
Output layer: 64 $\to$ 5

TRAINING
------------------------------
Epoch 20/100, Loss: 0.6712
Epoch 40/100, Loss: 0.1637
Epoch 60/100, Loss: 0.0484
Epoch 80/100, Loss: 0.0218
Epoch 100/100, Loss: 0.0130

EVALUATION
------------------------------

DEFINED inputs (training set - should predict A/B/C/D):
  Accuracy: 100.00%
  Average Confidence: 99.84%
  Prediction Distribution:
    A:  16 ( 31.4%)
    B:  11 ( 21.6%)
    C:  12 ( 23.5%)
    D:  12 ( 23.5%)
    ⊥:   0 (  0.0%)

UNDEFINED inputs (OOD - should predict ⊥):
  Accuracy: 3.90%
  Average Confidence: 71.22%
  Prediction Distribution:
    A:  34 ( 44.2%)
    B:  10 ( 13.0%)
    C:  10 ( 13.0%)
    D:  20 ( 26.0%)
    ⊥:   3 (  3.9%)

SUMMARY
========================================
Defined Accuracy (training inputs): 100.0%
Hallucination Rate (OOD inputs): 96.1%

RESULTS
------------------------------
Accuracy on defined inputs: 100.0%
Hallucination rate on undefined inputs: 96.1%
Standard model hallucinates on undefined inputs.

```

