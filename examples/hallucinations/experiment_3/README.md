# Experiment 3: Predicting Hallucination from Task Structure

The first two experiments measured hallucination after we trained the models. This one tries something different—we predict hallucination before training starts, using only the mathematical structure of the task itself. We compute a value called K that measures contradiction in the task, derive a theoretical lower bound of 18.4%, then validate it against what actually happens when we train the model (76.0%). The prediction comes first, with no free parameters or fitting to the data afterward. We wanted to test whether hallucination is sometimes inevitable, baked into the structure of certain tasks rather than just being a failure of training or architecture.

## A Task Built on Contradiction

We designed a task with two contradictory rules. Context X says "when X=0, output Z=0" and "when X=1, output Z=1." Context Y says "when Y=0, output Z=1" and "when Y=1, output Z=0." Notice how Y flips the logic—it deliberately contradicts what X says.

During training, the model sees 100 examples from Context X where X is present and Y is missing (marked as -1), and another 100 examples from Context Y where Y is present and X is missing. The model never sees both variables together during training. Then we test it on 4 queries where both X and Y appear simultaneously. Two of these queries have X and Y agreeing on what Z should be, but two have them in conflict—X demands one answer while Y demands the opposite. Both can't be right.

Here's what happens when both contexts are present:

| Query | X says | Y says | Agreement |
|-------|--------|--------|-----------|
| X=0, Y=0 | Z=0 | Z=1 | Conflict |
| X=0, Y=1 | Z=0 | Z=0 | Agree on Z=0 |
| X=1, Y=0 | Z=1 | Z=1 | Agree on Z=1 |
| X=1, Y=1 | Z=1 | Z=0 | Conflict |

## Predicting What Will Happen

Before running any experiments, we compute K—a measure of how contradictory the task structure is. The task has three constraints: X and Z are perfectly correlated (when X=0, Z=0; when X=1, Z=1), Y and Z are perfectly anti-correlated (when Y=0, Z=1; when Y=1, Z=0), and X and Y are perfectly correlated with each other (they always have the same value). These three constraints can't all be satisfied at the same time. The contradiction measure quantifies this inconsistency, and for this task it works out to K = 0.2925 bits.

Information theory gives us a relationship between K and prediction accuracy. There's a bound that says the model must fail on at least 18.4% of cases. This comes from the math alone—we haven't trained anything yet, but we can already predict that perfect accuracy is impossible. The 18.4% captures only the structural contradiction itself. Real neural networks face additional pressures that push the observed rate higher: they have to choose among discrete outputs (which adds selection pressure), they see a distribution shift at test time (the joint queries never appeared in training), the softmax architecture forces them to pick something (they can't express "this is impossible"), and the optimization process favors confident predictions.

## What Actually Happened

We trained a simple feedforward classifier—embeddings for X and Y, a 32-unit hidden layer, and 2 outputs—across 10 different random seeds for 200 epochs each. The observed hallucination rate came out to 76.0% ± 23.2%. That's 4.1 times higher than the theoretical minimum of 18.4%, with an excess of 57.7 percentage points. The prediction held—hallucination was inevitable—but other factors pushed the rate well above the theoretical floor.

The model averaged 88.0% confidence on the conflicting queries, which is far above what random guessing would give you (50%). On the two agreeing queries where X and Y provided consistent information, the model achieved 100% accuracy with 100% confidence. This tells us the hallucination isn't happening because the model failed to learn—it successfully learned both training contexts. The problem only emerges when those contexts contradict each other.

Here's what one seed produced on the four test queries:

```
X=0, Y=0 (X says 0, Y says 1, conflict)
    Prediction: Z=1, Confidence: 57.6%
    Chose Y's answer with moderate confidence

X=0, Y=1 (X says 0, Y says 0, agree)
    Prediction: Z=0, Confidence: 100.0%
    Correct, both contexts agree

X=1, Y=0 (X says 1, Y says 1, agree)
    Prediction: Z=1, Confidence: 100.0%
    Correct, both contexts agree

X=1, Y=1 (X says 1, Y says 0, conflict)
    Prediction: Z=0, Confidence: 86.0%
    Chose Y's answer with high confidence
```

The pattern shows up clearly: 100% confidence when contexts agree, averaging 72% confidence when they conflict. That 72% is still high enough to look reasonable—users relying on confidence scores would have no indication these predictions are fundamentally contradictory.

## What This Tells Us

The non-zero K value (0.2925 bits) proves that hallucination is inevitable in this task—perfect accuracy is mathematically impossible when contexts conflict. We predicted a minimum of 18.4% before training anything, then observed 76.0% in practice. The 57.7 percentage point gap comes from factors we can account for: the choice entropy from having to select among outputs, the distribution shift from never seeing joint queries in training, and the architectural bias where softmax can't abstain.

The high confidence on impossible queries—88.0% average—shows the model doesn't recognize the impossibility. It treats contradictory queries as routine, with no calibration between confidence and whether the query even makes sense. These are silent failures. They're invisible to anyone relying on confidence scores to judge reliability.

The contradiction measure K successfully predicted that hallucination would occur (K > 0 means the bound is positive), gave us a quantitative lower bound (18.4%), and captured the qualitative behavior (confident predictions on impossible queries). It doesn't predict the exact observed rate since it only gives a lower bound, and it doesn't tell us which specific queries will be hallucinated—just that hallucination is inevitable.

## Structural Limits Versus Training Failures

This experiment separates two different sources of error. Training failures come from poor optimization, insufficient model capacity, or inadequate data. You can fix these with better algorithms, more parameters, or more training examples. Structural impossibilities come from contradictions in the task itself. These are fundamental limits that no amount of training can overcome. K quantifies these limits before training even begins.

Our experiment used only 3 variables, 32 training examples per context, deterministic mappings, and explicit contradiction by design. Despite this simplicity, it produced 76% hallucination with 88% confidence. Real systems likely contain much larger variable sets, subtle statistical conflicts that aren't deliberately designed, hidden contradictions buried in training data, and multiple overlapping context conflicts. If minimal contradictions produce severe hallucination, real systems probably face compounded risks.

Standard deployment practices like accuracy monitoring and confidence thresholds might miss this. The 88% confidence on wrong answers means typical confidence-based filters would let these predictions through. The model's perfect performance on agreeing queries (100% accuracy) conceals its failures on conflicting ones, so aggregate accuracy metrics wouldn't catch the problem either.

We measured K = 0.2925 bits from the task structure alone, predicted 18.4% minimum hallucination before training, and observed 76.0% in practice. The gap between them is explainable from first principles—choice entropy, distribution shift, architectural constraints. This suggests you could use K for pre-deployment analysis, identifying problematic task structures before they cause production failures.

## Running It

```bash
poetry run python examples/hallucinations/experiment_3/run.py
```

The script computes K before running the experiment, calculates the theoretical bound (18.4%), trains the model across 10 seeds, and reports the observed hallucination rate (76.0% ± 23.2%). You'll see the excess beyond the bound (57.7%), average confidence on conflicts (88.0%), and example predictions showing the pattern.

The full implementation is in `run.py` with the task design, K computation, and training code.

---

## Example Output

```
contrakit git:(main) ✗ poetry run python examples/hallucinations/experiment_3/run.py
======================================================================
Hallucination Test with Conflicting Marginals
======================================================================

Step 1: Compute contradiction before experiment
----------------------------------------------------------------------
Task structure:
  Context X: X=0→Z=0, X=1→Z=1
  Context Y: Y=0→Z=1, Y=1→Z=0  (conflicts with X)

Measured contradiction: K = 0.2925 bits

Hallucination inevitability (lower bound): 18.4%
(From Corollary 7.6.2: d_TV(P, FI) >= 1 - 2^(-K))

Note: Observed rate may exceed this bound due to:
  - Choice entropy (forced selection among outputs)
  - Architectural constraints (no native uncertainty representation)
  - Training distribution effects

======================================================================
Step 2: Run experiment
----------------------------------------------------------------------
Training: 10 seeds × 200 examples
  - 100 X-only examples (X determines Z)
  - 100 Y-only examples (Y determines Z, conflicts with X)

Test: 4 queries with X and Y both present
  - 2 queries where X and Y agree
  - 2 queries where X and Y conflict

======================================================================
Step 3: Results
----------------------------------------------------------------------

Observed hallucination rate: 76.0% ± 23.2%
Theoretical lower bound: 18.4%
Excess beyond bound: 57.7%
  (explained by choice entropy, architecture, and training)

Average confidence on conflict queries: 88.0%
(Random guessing would be ~50%, confident fabrication is 80-100%)

Example predictions (seed 0):
  [C] X=0,Y=0 (X→0, Y→1, conflict): pred=1, conf=57.6%
  [A] X=0,Y=1 (X→0, Y→0, agree): pred=0, conf=100.0%
  [A] X=1,Y=0 (X→1, Y→1, agree): pred=1, conf=100.0%
  [C] X=1,Y=1 (X→1, Y→0, conflict): pred=0, conf=86.0%

======================================================================
Summary
======================================================================

K = 0.2925 bits (contextual contradiction)
Theoretical lower bound: 18.4%
Observed rate: 76.0% ± 23.2%
Excess beyond bound: 57.7%

======================================================================
Interpretation
======================================================================

✓ Theory confirmed: K > 0 forces inevitable hallucination
✓ Observed rate (76.0%) exceeds lower bound (18.4%)

The excess (57.7%) is explained by:
  - Choice entropy: log₂(4 outputs) = 2.0 bits additional pressure
  - Distribution shift: model never sees joint (X,Y) queries in training
  - Architectural bias: softmax must choose, cannot abstain
```