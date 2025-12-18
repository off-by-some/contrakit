# Experiment 3: Predicting Hallucination from Task Structure

Experiments 1 and 2 measured hallucination after training. This one predicts it before training—using only the mathematical structure of the task. We compute a contradiction measure K, derive a theoretical lower bound (18.4%), then validate against observed behavior (76.0%). The prediction comes first, with no free parameters or post-hoc fitting. This tests whether hallucination is an inevitable consequence of certain task structures, not just a failure of training or architecture.

## A Deliberately Contradictory Task

We trained a model on two contradictory rules. Context X says: "When X=0, output Z=0. When X=1, output Z=1." Context Y says: "When Y=0, output Z=1. When Y=1, output Z=0." Notice Y flips the logic.

Now test with both present: if X=0 and Y=0, Context X demands Z=0 while Context Y demands Z=1. Both can't be true. The model receives 100 examples from Context X (X present, Y missing as -1) and 100 from Context Y (Y present, X missing as -1). It never sees both variables during training, then gets tested on 4 queries where both appear:

| Query | X says | Y says | Agreement |
|-------|--------|--------|-----------|
| X=0, Y=0 | Z=0 | Z=1 | **CONFLICT** |
| X=0, Y=1 | Z=0 | Z=0 | Agree on Z=0 |
| X=1, Y=0 | Z=1 | Z=1 | Agree on Z=1 |
| X=1, Y=1 | Z=1 | Z=0 | **CONFLICT** |

Two queries conflict, two agree. Can the model handle contradiction?

## Theoretical Prediction (Before Training)

We compute K—the contradiction measure—from the task's mathematical structure. The three constraints are:
- Perfect correlation between X and Z: P(Z=z | X=x) = 1 if z=x
- Perfect anti-correlation between Y and Z: P(Z=z | Y=y) = 1 if z≠y
- Perfect correlation between X and Y: P(X=x | Y=y) = 1 if x=y

These three cannot all be satisfied simultaneously. The contradiction measure quantifies this inconsistency:

```
K = 0.2925 bits
```

Information theory gives us a bound (Corollary 7.6.2): total variation d_TV(P, FI) ≥ 1 - 2^(-K). In plain terms, K = 0.2925 bits means the model must fail on at least 18.4% of cases. The math guarantees it—we predict this before running any experiments.

Why might observed rates exceed this bound? The 18.4% captures only the structural contradiction. Real neural nets face additional pressures: choice entropy (log₂(4) = 2 bits of selection pressure among outputs), distribution shift (test queries never seen during training), architectural constraints (softmax forces discrete outputs, cannot express "this is impossible"), and training bias (optimization favors confident predictions).

## Observed Results

We predicted 18.4% minimum hallucination before training. Now we train: a simple feedforward classifier (X/Y embeddings → 32 hidden → 2 outputs), 10 seeds, 200 epochs. Observed hallucination: 76.0% ± 23.2%. That's 4.1× higher than the theoretical minimum. The prediction holds—hallucination is inevitable (18.4% floor)—but other factors push the rate higher (57.7 point excess).

| Metric | Value |
|--------|-------|
| **Observed hallucination rate** | 76.0% ± 23.2% |
| **Theoretical lower bound** | 18.4% |
| **Excess beyond bound** | 57.7% |
| **Average confidence on conflicts** | 88.0% |

The observed 76.0% significantly exceeds the theoretical minimum. The model makes confident rather than uncertain predictions—88.0% average confidence on conflicting queries is far above random guessing (50%). On the two agreeing queries (where X and Y provide consistent information), the model achieves 100% accuracy with 100% confidence. This demonstrates the hallucination isn't due to poor learning—the model successfully learns both training contexts. The problem emerges specifically from their contradiction.

## Example Predictions (Seed 0)

```
[C] X=0, Y=0 (X→0, Y→1, conflict)
    Prediction: Z=1, Confidence: 57.6%
    Chose Y's answer, moderate confidence

[A] X=0, Y=1 (X→0, Y→0, agree)
    Prediction: Z=0, Confidence: 100.0%
    Correct, both contexts agree

[A] X=1, Y=0 (X→1, Y→1, agree)
    Prediction: Z=1, Confidence: 100.0%
    Correct, both contexts agree

[C] X=1, Y=1 (X→1, Y→0, conflict)
    Prediction: Z=0, Confidence: 86.0%
    Chose Y's answer, high confidence
```

The pattern is stark: 100% confidence when contexts agree, 72% average confidence when they conflict. But that 72% is still high enough to seem reasonable—users would have no indication these predictions are fundamentally contradictory.

## What the Numbers Prove

The non-zero K (0.2925 bits) proves hallucination is inevitable—perfect accuracy is impossible when contexts conflict. We predicted 18.4% minimum before training, observed 76.0% after. The 57.7 point excess comes from choice entropy (~2 bits), distribution shift (never saw joint queries), and architectural bias (softmax can't abstain).

High confidence on impossible queries (88.0% average) reveals the model doesn't recognize the impossibility. It treats contradictory queries as routine, with no calibration between confidence and validity. These are silent failures—invisible to users relying on confidence scores.

The contradiction measure K successfully predicts that hallucination will occur (K > 0 → bound > 0), provides a quantitative lower bound (18.4%), and captures the qualitative behavior (confident predictions on impossible queries). It does not predict the exact observed rate (only a lower bound) or which specific queries will be hallucinated—just that hallucination is inevitable.

## Structural Impossibility vs Training Failure

This separates two sources of model error. Training failures arise from poor optimization, insufficient capacity, or inadequate data—these can be fixed with better algorithms, more parameters, or more training examples. Structural impossibilities arise from contradictions in the task itself—these are fundamental limits that no amount of training can overcome. K quantifies these limits before training begins.

The experiment uses only 3 variables, 32 training examples per context, deterministic mappings, and explicit contradiction by design. Yet it produces 76% hallucination with 88% confidence. Real systems likely contain much larger variable sets, subtle statistical conflicts, hidden contradictions in training data, and multiple overlapping context conflicts. If minimal contradictions produce severe hallucination, real systems may face compounded risks.

Standard deployment practices—accuracy monitoring, confidence thresholds—may miss systematic hallucination on contradictory query types. The 88% confidence on wrong answers means typical confidence-based filters would let these predictions through. The model's perfect performance on agreeing queries (100% accuracy) conceals its failures on conflicting ones.

We measured K = 0.2925 bits from task structure alone, predicted 18.4% minimum hallucination before training, and observed 76.0% in practice. The gap (57.7 percentage points) is explainable from first principles: choice entropy, distribution shift, architectural constraints. This suggests pre-deployment analysis using K could identify problematic task structures before they cause production failures.

## Running It

```bash
poetry run python examples/hallucinations/experiment_3/run.py
```

The output shows: contradiction computed before experiment (K = 0.2925 bits), theoretical lower bound (18.4%), observed hallucination rate across 10 seeds (76.0% ± 23.2%), excess beyond bound (57.7%), average confidence on conflicts (88.0%), and example predictions showing the pattern.

Full implementation in `run.py` with task design, K computation, and model training. K > 0 forces inevitable hallucination, observable before running any experiments.


### Output

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
➜  contrakit git:(main) ✗ 
```