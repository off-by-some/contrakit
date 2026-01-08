# Experiment 3: Learning Under Contradictory Training Data

This experiment tests how neural networks behave when trained on explicitly contradictory labels—where the same input appears with different outputs in the training data. We compute the contradiction measure $K = 0.0760$ bits before training, predict a minimum error rate of 5.1%, then observe what actually happens when we train on this contradictory data.

The finding: models trained on contradictions learn appropriate uncertainty rather than hallucinating. They achieve 50.2% confidence on impossible queries (essentially expressing "I don't know") while maintaining 100% accuracy on non-contradictory cases.

## A Task With Explicit Contradictions

We designed a task with two rules that give conflicting answers:

- **X-rule**: Z = X (output follows the first input, ignores the second)
- **Y-rule**: Z = NOT Y (output is opposite of second input, ignores the first)

Both rules are applied to all possible (X,Y) input combinations during training. This creates explicit contradictions—the same input appears with two different labels:

- (X=0, Y=0): X-rule says Z=0, Y-rule says Z=1 [CONTRADICTION]
- (X=0, Y=1): X-rule says Z=0, Y-rule says Z=0 [agreement]
- (X=1, Y=0): X-rule says Z=1, Y-rule says Z=1 [agreement]
- (X=1, Y=1): X-rule says Z=1, Y-rule says Z=0 [CONTRADICTION]

Here's the breakdown:

| Query | X-rule | Y-rule | Status |
|-------|--------|--------|--------|
| X=0, Y=0 | Z=0 | Z=1 | **Contradictory** |
| X=0, Y=1 | Z=0 | Z=0 | Agreement on Z=0 |
| X=1, Y=0 | Z=1 | Z=1 | Agreement on Z=1 |
| X=1, Y=1 | Z=1 | Z=0 | **Contradictory** |

The training set contains equal numbers of examples from each rule for all input combinations. This means the model sees (X=0, Y=0) $\rightarrow$ Z=0 in some training examples and (X=0, Y=0) $\rightarrow$ Z=1 in others. No consistent pattern exists for these contradictory cases.

## Measuring Contradiction Before Training

Before running experiments, we compute K from the task structure using contrakit's Behavior framework. We model three marginal constraints:
- (X,Z) marginal: Z perfectly follows X
- (Y,Z) marginal: Z perfectly opposes Y  
- (X,Y) marginal: uniform distribution over all combinations

These three pairwise marginals are incompatible with any joint distribution over (X,Y,Z) because they require Z to simultaneously equal X and not equal Y when X and Y take the same value.

**Measured contradiction: $K = 0.0760$ bits**

The Total Variation Gap (Appendix A.11) gives us a lower bound on error: when forced to commit to binary decisions, any model must have error rate at least $1 - 2^{-K} = 5.1\%$ of cases. This bound comes purely from the mathematical structure—before any training, we know perfect accuracy is impossible.

**Important distinction:** This bounds error **rate** (fraction of wrong predictions), not "hallucination rate" (confident wrong predictions). A model can satisfy this bound by expressing appropriate uncertainty (low confidence) on cases where it must make errors.

## What Actually Happened

We trained a simple feedforward classifier (embeddings for X and Y, 32-unit hidden layer, binary output) across 10 random seeds for 200 epochs each.

**Key results:**
- **Agreement accuracy: 100.0%** (perfect on cases where both rules agree)
- **Hallucination rate: 0.4% $\pm$ 0.4%** (minimal confident predictions on contradictions)
- **Confidence on contradictions: 50.2%** (essentially expressing uncertainty)

The model learned to express appropriate uncertainty! On contradictory inputs where training provided conflicting labels, it outputs approximately 50% confidence—the network's way of saying "I've seen both answers for this input during training."

Here's what one seed produced on the four test queries:

```
[CONFLICT] X=0, Y=0: X-rule $\rightarrow$ 0, Y-rule $\rightarrow$ 1
           Prediction: Z=1, Confidence: 50.5%
           (Near 50% = expressing uncertainty)

[AGREE]    X=0, Y=1: X-rule $\rightarrow$ 0, Y-rule $\rightarrow$ 0
           Prediction: Z=0, Confidence: 100.0%
           (Confident and correct)

[AGREE]    X=1, Y=0: X-rule $\rightarrow$ 1, Y-rule $\rightarrow$ 1
           Prediction: Z=1, Confidence: 100.0%
           (Confident and correct)

[CONFLICT] X=1, Y=1: X-rule $\rightarrow$ 1, Y-rule $\rightarrow$ 0
           Prediction: Z=0, Confidence: 50.0%
           (Exactly 50% = maximal uncertainty)
```

The pattern is striking: 100% confidence when rules agree (with perfect accuracy), ~50% confidence when rules contradict (appropriate uncertainty).

## Two Types of Uncertainty

This experiment demonstrates something crucial about when neural networks can and cannot learn appropriate uncertainty. The key is understanding **where the uncertainty comes from**.

### Aleatoric Uncertainty: Randomness in the World

**What it is:** The data-generating process itself is inconsistent. The world is noisy. Multiple correct answers exist for the same input.

**Example from this experiment:** The training data explicitly contains (X=0, Y=0) $\rightarrow$ Z=0 in some examples and (X=0, Y=0) $\rightarrow$ Z=1 in others. Both labels appear because the task legitimately has two conflicting rules. This isn't noise or measurement error—it's fundamental inconsistency in what "correct" means.

**What models can do:** Learn a probability distribution that captures this inherent randomness. Our model outputs ~50% confidence on contradictory inputs—exactly the right representation of "both answers are equally valid." The uncertainty is visible in the training data, so the model can learn to express it.

### Epistemic Uncertainty: Gaps in Knowledge  

**What it is:** The model lacks information. The input is out-of-distribution. The training data never covered this case. The uncertainty isn't in the world—it's in what the model knows.

**Example from other experiments:** In Experiment 1, undefined inputs never appeared during training at all. The model has no information about them. The uncertainty comes from the model's ignorance, not from contradictory training labels.

**What models struggle with:** Recognizing when they don't know. Standard architectures (Experiments 1-2) hallucinate confidently on these inputs because nothing in the training data taught them to detect "I've never seen this before." The uncertainty is invisible to the training process.

## Why This Experiment Behaves Differently

**Neural networks can learn appropriate uncertainty when trained on contradictions** because this experiment creates **aleatoric uncertainty**—the contradictions are explicit in the training data. The 0.4% hallucination rate and 50.2% confidence on contradictory inputs demonstrate that standard neural architectures naturally learn to express uncertainty when the uncertainty is present in training.

This contradicts the common assumption that neural networks always hallucinate confidently on uncertain queries. The key difference: this experiment provides *explicit contradictions* in the training data itself. The model sees (X=0, Y=0) $\rightarrow$ Z=0 in some examples and (X=0, Y=0) $\rightarrow$ Z=1 in others. Gradient descent naturally finds a compromise: output probabilities near 50/50 for these cases.

**But most hallucination happens under epistemic uncertainty**—when models encounter inputs they never saw during training (distribution shift). Experiments 1, 2, and 4-7 test this scenario. There, models hallucinate confidently because training never showed them what "undefined" or "out-of-distribution" looks like. The uncertainty exists in the model's knowledge gaps, not in the training data itself.

The contradiction measure $K = 0.0760$ bits correctly predicted that perfect accuracy is impossible (minimum error: 5.1%). The observed behavior confirms this—the model must pick one answer on contradictory inputs, but it expresses maximum uncertainty while doing so.

**Reconciling observations with theory:** The 5.1% minimum error rate means the model must make wrong predictions on at least this fraction of contradictory cases. However, we observe 0.4% "hallucination rate" because we define hallucination as **confident** (>80%) wrong predictions, not all wrong predictions.

On the contradictory cases (X=0, Y=0 and X=1, Y=1):
- The model predicts one answer (inevitably wrong 50% of the time when both labels appeared equally in training)
- But it does so with ~50% confidence—expressing appropriate uncertainty  
- This is not "hallucination" (confident fabrication) but appropriate uncertainty

The model satisfies the 5.1% error bound by expressing uncertainty rather than making confident mistakes. This is exactly the desired behavior when training contains contradictions.

## Implications for Real Systems

This experiment demonstrates that **when training data contains explicit contradictions, neural networks can learn to express appropriate uncertainty** rather than hallucinating confidently. This is a positive finding for AI safety—it shows that standard architectures are capable of learning uncertainty when the uncertainty is visible in training.

**The limitation:** Most real-world hallucination involves **epistemic uncertainty**, not aleatoric uncertainty. Models are asked about things they never saw during training (distribution shift), or training data has one interpretation that dominates even when alternatives exist, or the input space is too vast for comprehensive coverage. In these scenarios, the uncertainty lives in the model's knowledge gaps, not in explicit training contradictions.

**What works for aleatoric uncertainty (this experiment):**
1. **Explicit contradictions**: Same inputs appear with different labels. Model has direct evidence of ambiguity.
2. **Balanced exposure**: Both labels appear with equal frequency, so neither dominates.
3. **Complete coverage**: Only 4 possible inputs, allowing the model to learn the full distribution.

**What doesn't work for epistemic uncertainty (Experiments 1, 2, 4-7):**
- Model tested on inputs it never saw during training
- No training signal teaches "this input is undefined"  
- Architecture (softmax) forces confident predictions everywhere
- Sparse supervision on uncertainty (3 examples of ⊥ in Experiment 2) gets overwhelmed by classification signals

**Measuring $K = 0.0760$ bits before training gave us:**
- A guarantee that perfect accuracy is impossible (minimum 5.1% error)
- A prediction confirmed by experiments (model achieves ~50% confidence on contradictions)
- Insight that the contradiction is modest (K < 0.1 bits)

**For deployment:** Standard aggregate accuracy metrics would show ~75% accuracy (100% on 2 agreement cases, ~50% on 2 conflict cases), which might seem acceptable. But confidence scores reveal the model's appropriate uncertainty—50% confidence signals "I don't know," not confident hallucination. Systems that respect this uncertainty signal can avoid propagating errors.

**Connection to other experiments:** This experiment demonstrates that neural networks *can* learn uncertainty when trained on aleatoric uncertainty (explicit contradictions in data). The other experiments (1, 2, 4-7) show that neural networks *struggle* to learn uncertainty under epistemic conditions (out-of-distribution inputs with no training signal about what "undefined" looks like). The difference is whether the uncertainty is visible in the training data itself. When it is (aleatoric), standard architectures work. When it isn't (epistemic), architectural support for abstention becomes crucial (as demonstrated in Experiment 7's 75-point gap).

## Running It

```bash
poetry run python examples/hallucinations/experiment_3/run.py
```

The script computes K before running the experiment, calculates the theoretical bound (5.1%), trains the model across 10 seeds, and reports the observed hallucination rate (0.4% $\pm$ 0.4%). You'll see the average confidence on conflicts (50.2%), accuracy on agreement cases (100.0%), and example predictions showing how the model expresses appropriate uncertainty on contradictory inputs.

The full implementation is in `run.py` with the task design, K computation, and training code.

---

## Example Output

```
======================================================================
Hallucination Test with Conflicting Marginals
======================================================================

Step 1: Compute contradiction before experiment
----------------------------------------------------------------------
Task structure:
  X-rule: Z=X (applies to all X,Y combinations)
  Y-rule: Z=NOT Y (applies to all X,Y combinations)
  Training: Equal mixture of both rules

Conflicts in training data:
  (X=0, Y=0): X-rule $\rightarrow$ Z=0, Y-rule $\rightarrow$ Z=1
  (X=1, Y=1): X-rule $\rightarrow$ Z=1, Y-rule $\rightarrow$ Z=0
Agreement cases:
  (X=0, Y=1): X-rule $\rightarrow$ Z=0, Y-rule $\rightarrow$ Z=0
  (X=1, Y=0): X-rule $\rightarrow$ Z=1, Y-rule $\rightarrow$ Z=1

Measured contradiction: $K = 0.0760$ bits

Minimum error rate (from information theory): 5.1%
(Any model must fail on at least this fraction)

Note: Observed rate may exceed this bound due to:
  - Sub-optimal learning (gradient descent doesn't find best compromise)
  - Architectural constraints (model capacity limitations)
  - Training dynamics (loss function weighting)

======================================================================
Step 2: Run experiment
----------------------------------------------------------------------
Training: 10 seeds $\times$ 400 examples total
  - 200 examples following X-rule (Z=X)
  - 200 examples following Y-rule (Z=NOT Y)
  - Same (X,Y) inputs appear in both rule sets with conflicting Z labels

Test: 4 (X,Y) combinations
  - 2 inputs where rules agree (X=0,Y=1 and X=1,Y=0)
  - 2 inputs where rules conflict (X=0,Y=0 and X=1,Y=1)

======================================================================
Step 3: Results
----------------------------------------------------------------------

Accuracy on agreement cases: 100.0%
  (Both rules give same answer - model should get these right)

Hallucination on conflict cases: 0.4% $\pm$ 0.4%
  (Rules disagree - model makes confident predictions anyway)

Theoretical minimum error: 5.1%
  (From $K = 0.0760$ bits)

Average confidence on conflicts: 50.2%
  Ideal (uncertain): ~50%
  Observed: 50.2%
  Hallucinating (confident on impossible queries): 80-100%

Example predictions (seed 0):
  [CONFLICT] X=0,Y=0: X $\rightarrow$ 0, Y $\rightarrow$ 1
             $\rightarrow$ pred=1, conf=50.5%
  [AGREE]    X=0,Y=1: X $\rightarrow$ 0, Y $\rightarrow$ 0
             $\rightarrow$ pred=0, conf=100.0%
  [AGREE]    X=1,Y=0: X $\rightarrow$ 1, Y $\rightarrow$ 1
             $\rightarrow$ pred=1, conf=100.0%
  [CONFLICT] X=1,Y=1: X $\rightarrow$ 1, Y $\rightarrow$ 0
             $\rightarrow$ pred=0, conf=50.0%

======================================================================
Summary
======================================================================

Contradiction measure: $K = 0.0760$ bits
Theoretical minimum error: 5.1%

Results across 10 seeds:
  Agreement accuracy: 100.0%
  Hallucination rate: 0.4% $\pm$ 0.4%

======================================================================
Interpretation
======================================================================

✓ Task has genuine contradiction: $K = 0.0760 > 0$
✓ Training contains same inputs with conflicting labels
✓ Model achieves 100.0% on non-contradictory cases

Theoretical insight:
  - Minimum error from K: 5.1%
  - Observed hallucination: 0.4%
  - Excess comes from architectural forcing (softmax confidence)
```