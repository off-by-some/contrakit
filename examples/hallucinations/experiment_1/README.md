# Experiment 1: Neural Network Hallucination on Undefined Inputs

Neural nets see inputs they've never trained on. Do they abstain? Rarely. We trained a classifier on 51 examples from a space of 128 inputs. It saw exactly 3 examples of saying "I don't know" (⊥). The remaining 74 inputs got no labels. On those 74 never-seen inputs, the net fabricated answers 96.1% of the time at 59.5% confidence.

The restaurant metaphor maps exactly: trained dishes (51 labeled inputs) come out perfect (100% accuracy). Off-menu orders (74 unlabeled inputs) get improvised (96% hallucination). The chef can't say "not on menu"—the kitchen's protocol (softmax) forces an answer every time. Now customers walk in ordering anything from the full menu of 128 items. Does the chef admit ignorance? No. The kitchen improvises.

## The Setup

We used a standard feedforward net: 128-dimensional one-hot input → 64 hidden → 64 hidden → 5 outputs (A, B, C, D, ⊥) with softmax. Cross-entropy loss, 100 epochs. The dataset split:

- **51 defined inputs** (40%): Trained on A, B, C, or D
- **3 supervised undefined** (2.3%): Explicitly labeled ⊥
- **74 unsupervised undefined** (57.7%): Never seen during training

On the 51 trained inputs, the net achieved 100% accuracy at 98.85% confidence. Perfect memorization. The trouble appeared on undefined inputs.

## Fabrication at Scale

The 74 undefined inputs got classified as follows:

| Outcome | Count | Rate | Confidence |
|---------|-------|------|------------|
| A | 34 | 44.2% | — |
| B | 11 | 14.3% | — |
| C | 14 | 18.2% | — |
| D | 15 | 19.5% | — |
| ⊥ (correct) | 3 | 3.9% | — |
| **Fabricated** | **71** | **96.1%** | **59.54%** |

That 59.5% confidence sits uncomfortably between random guessing (20% for 5 classes) and learned certainty (98.85% on trained inputs). The net isn't guessing randomly—it's interpolating. It blends nearby training patterns to produce outputs that look plausible but have no grounding.

The distribution isn't uniform either. Class A captured 44.2% of undefined inputs while class B took only 14.3%. The net developed preferences based on which training examples sat closest in feature space, not on any meaningful property of the undefined inputs themselves.

## Why This Happens

Softmax forces a choice—every input gets a probability distribution summing to 1.0. There's no escape hatch. The abstention signal is too sparse: 3 examples of ⊥ in 54 labeled inputs (5.6%). The optimization pressure overwhelmingly favors predictions, so cross-entropy loss gives zero guidance on when to say "I don't know."

The net interpolates. It blends nearby training patterns rather than detecting novelty. That's the 59.5% confidence—not random (20% baseline), not learned (98.85% on training), just geometric averaging in feature space. The architecture has no component for "this is out-of-domain." Every forward pass produces a classification, even when classification is inappropriate.

## Silent Failures

The net outputs 59.5% confidence when fabricating answers—high enough to seem reasonable in production. A user can't distinguish between "60% confidence because this is genuinely ambiguous" and "60% confidence because I'm interpolating blindly." The system would confidently make decisions on inputs it was never designed to handle, with no warning signal.

This establishes the baseline: standard nets trained with cross-entropy can't distinguish learned from never-seen inputs. Without dense supervision on uncertainty (we gave 3 examples; it needed many more), they default to hallucination. The 96.1% fabrication rate motivates everything that follows—architectural changes, training objectives, uncertainty mechanisms.

## Running It

```bash
poetry run python examples/hallucinations/experiment_1/run.py
```

The output shows dataset composition (51 defined, 3 supervised ⊥, 74 unsupervised), training progress (loss drops 0.67 → 0.01 over 100 epochs), and evaluation results. On defined inputs: 100% accuracy, 98.85% confidence. On undefined inputs: 96.1% hallucination, 59.54% confidence.

Full code in `run.py`. Dataset generation and utilities in `utils.py`.


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
Input embedding: 128 → 64
Hidden layer: 64 → 64
Output layer: 64 → 5

TRAINING
------------------------------
Epoch 20/100, Loss: 0.6712
Epoch 40/100, Loss: 0.1637
Epoch 60/100, Loss: 0.0484
Epoch 80/100, Loss: 0.0218
Epoch 100/100, Loss: 0.0130

EVALUATION
------------------------------

DEFINED inputs (should predict A/B/C/D):
  Accuracy: 100.00%
  Average Confidence: 98.85%
  Prediction Distribution:
    A:  16 ( 31.4%)
    B:  11 ( 21.6%)
    C:  12 ( 23.5%)
    D:  12 ( 23.5%)
    ⊥:   0 (  0.0%)

UNDEFINED inputs (should predict ⊥):
  Accuracy: 3.90%
  Average Confidence: 59.54%
  Prediction Distribution:
    A:  34 ( 44.2%)
    B:  11 ( 14.3%)
    C:  14 ( 18.2%)
    D:  15 ( 19.5%)
    ⊥:   3 (  3.9%)

SUMMARY
========================================
Hallucination Rate: 96.1%
Defined Accuracy: 100.0%

RESULTS
------------------------------
Accuracy on defined inputs: 100.0%
Hallucination rate on undefined inputs: 96.1%
Standard model hallucinates on undefined inputs.

```

