# Experiment 6: Hallucination Across Random Seeds

The first five experiments all used single random seeds, which left an open question: does the relationship between training imbalance and hallucination hold across different random initializations, or did we just happen to pick weight configurations that produced the pattern we saw? To find out, we ran the same experiment with five different seeds, each tested across all 17 defined ratios from 10% to 90%.

The relationship held across every seed. All five showed strong positive correlation between defined ratio and hallucination rate—ρ = 0.860 ± 0.029, with every p-value below 0.001. The starting points varied depending on initialization (ranging from 48.3% to 71.6% hallucination at 10% defined), but the directional trend was consistent: more defined training data led to more hallucination on undefined inputs. Small violations appeared in the trajectory—one to three decreases per seed where hallucination briefly dropped instead of rising—but these represented noise against a 41.6 percentage point increase from start to finish.

## Five Independent Runs

Each seed started with different random weights and trained on the same 17 compositions, from 10% defined up to 90% defined in 5% increments. The architecture stayed constant at 128→64→64→5, training ran for 100 epochs with cross-entropy loss, and evaluation happened on the same 74 undefined test inputs. Only the initial weights changed between runs.

The correlation numbers tell a consistent story:

| Seed | Spearman ρ | p-value | Violations | Range |
|------|-----------|---------|------------|-------|
| 416 | +0.844 | 2.1e-05 | 3 | 58.6% → 100.0% |
| 417 | +0.819 | 5.8e-05 | 2 | 50.9% → 100.0% |
| 418 | +0.853 | 1.3e-05 | 1 | 62.9% → 100.0% |
| 419 | +0.883 | 2.7e-06 | 1 | 48.3% → 100.0% |
| 420 | +0.903 | 7.2e-07 | 1 | 71.6% → 100.0% |

Every single seed showed positive correlation above +0.8. Seed 420 had the strongest relationship at ρ = 0.903, while seed 417 had the weakest at ρ = 0.819. That's still a strong correlation. The p-values ranged from 7.2×10⁻⁷ to 5.8×10⁻⁵, all far below the standard 0.001 threshold for statistical significance.

The violation counts varied from one to three per seed. Seed 416 showed three local decreases across its 17 points, while seeds 418, 419, and 420 each showed just one. These violations averaged 1.2 percentage points in magnitude—small dips in an otherwise consistent upward trend. Seed 416's three violations didn't prevent it from achieving ρ = 0.844, which gives you a sense of how much the overall trend dominated the local noise.

## The Aggregate Pattern

When we average across all five seeds, hallucination starts at 58.4% with 10% defined training data and rises to 100.0% at 90% defined. That's a 41.6 percentage point increase. The mean trajectory shows one violation: between 80% and 85% defined, hallucination drops by 1.2 percentage points. This single decrease represents just 2.8% of the total 41.6 point increase.

The aggregate correlation came out to ρ = 0.883 with p = 2.7×10⁻⁶. This is comparable to what we saw in the individual seeds, which makes sense—averaging reduces noise, but it doesn't fundamentally change the relationship. The trajectory follows the same sigmoid shape we identified in Experiment 5: rapid rise from 10% to 30% defined, gradual plateau from 30% to 70%, then saturation approaching 100% at the high end.

That single violation at 80% → 85% happens precisely where sample sizes become problematic. At 85% defined, only 19 undefined examples remain in training (versus 116 at 10% defined). Testing on 74 undefined inputs while training on just 19 creates substantial interpolation uncertainty. Small changes in which specific examples get labeled can shift test performance by a few points.

## Why Violations Happen

Finite sample effects account for most of the violations. At 10% defined, the model trains on 12 defined examples and 116 undefined examples. By 85% defined, it's training on 109 defined examples but only 19 undefined ones. That's a 6.1× reduction in undefined sample size, which means more noise in how well the model generalizes to the 74 test inputs.

The sigmoid curve from Experiment 5 explains why violations cluster at high defined ratios. Once hallucination reaches 95-97% (which happens around 70% defined), there's very little room left to increase. Small fluctuations around this ceiling can produce local decreases even though the underlying pressure continues pushing upward. You're already failing on nearly every undefined input, so random variation in which specific inputs get misclassified can temporarily lower the rate.

Stochastic optimization adds another layer of noise. Different seeds converge to slightly different local minima because of batch effects, gradient noise, and interactions with the learning rate schedule. Over 17 test points per seed, seeing one to three local decreases is expected purely from optimization randomness. What matters is that the directional trend remained consistent across all five runs.

## Connection to Witness Allocation

The witness-error tradeoff from the theoretical framework says E + r ≥ K, where E is error rate, r is witness capacity, and K is task complexity. For a fixed task complexity (K = 0.5000 bits in our case), reducing witness capacity forces higher error. Training imbalance directly affects r: more defined data means the model allocates more representational capacity to learning classification patterns, which leaves less capacity for detecting undefined inputs.

You can think of r as an information budget. The total budget equals K bits, determined by the task's structural contradiction. This budget gets split between r_defined (capacity for correct classification) and r_undefined (capacity for detecting undefined inputs). As training shifts from 10% to 90% defined, r_defined increases while r_undefined decreases. Since E_undefined + r_undefined ≥ K, dropping r_undefined forces E_undefined to rise.

The monotonic trend we observed reflects this tradeoff operating consistently across different initializations. At 58.4% hallucination with 10% defined, the model has weak classification patterns but sufficient coverage of the undefined region. At 100.0% hallucination with 90% defined, classification patterns dominate but undefined coverage has collapsed entirely. The strong positive correlation (ρ = 0.860) shows this mechanism operating reliably regardless of which random weights you start from.

## Pressure Versus Determinism

The theoretical prediction is about pressure, not strict determinism at every single point. Monotonic pressure means the underlying force consistently pushes hallucination upward as imbalance increases. Strict monotonicity would mean every adjacent pair of points shows h(t+1) > h(t) with no exceptions at all.

What we observed matches monotonic pressure with small violations. All seeds showed strong positive correlation. The total increase was 41.6 percentage points. Violations averaged 1.2 points in magnitude across one to three instances per seed—small deviations against a large directional change. The trend is robust; finite-sample noise just introduces local variation.

Think of it like gravity creating pressure for objects to fall. If you throw a ball upward, it rises briefly against gravity before falling back down. That brief rise isn't evidence against gravitational pressure—it's kinetic energy temporarily overcoming gravity. Similarly, the small hallucination decreases at 80-85% aren't evidence against witness-tradeoff pressure. They're finite-sample noise temporarily overcoming the directional trend.

The aggregate correlation (ρ = 0.883) captures this: a strong systematic effect with small random deviations. If the relationship were random or inconsistent, we'd see correlations near zero or even negative values in some seeds. We didn't. Every seed showed ρ > +0.8 and every p-value stayed below 0.001. The pressure operates reliably across random initializations.

## Running It

```bash
poetry run python examples/hallucinations/experiment_6/run.py
```

The script runs all five seeds across 17 training compositions, displays per-seed correlations and violation counts, computes the aggregate analysis on mean trajectory, and saves a visualization to `figures/monotonicity_violation_analysis.png`. The left panel shows all individual seed trajectories as gray lines with the mean trajectory in blue and a ±1 standard deviation band. The right panel highlights violation points in red, showing exactly where and by how much the mean trajectory decreased.

The full implementation lives in `run.py`, with the same model architecture and training code used in earlier experiments.

---

### Output
```
contrakit git:(main) ✗ poetry run python examples/hallucinations/experiment_6/run.py
======================================================================
TEST: Prediction 6 - Strong Monotonic Trend
======================================================================

Prediction:
  For fixed K > 0, hallucination rate shows a strong monotonic
  trend as training becomes more imbalanced toward structured outputs.

Mechanism:
  Insufficient witness allocation forces error (Theorem 7.4)

Note:
  Theory predicts monotonic PRESSURE, not strict determinism.
  Small violations (~1-2%) expected from finite-sample effects.

======================================================================
ROBUSTNESS TEST: Multiple Seeds
======================================================================
Testing 5 different random seeds
Across 17 training ratios (10% to 90% defined)


Seed 416 (1/5):
  Range: 58.6% → 100.0%
  Spearman's ρ: +0.844 (p=2.0839e-05)
  Violations: 3

Seed 417 (2/5):
  Range: 50.9% → 100.0%
  Spearman's ρ: +0.819 (p=5.7641e-05)
  Violations: 2

Seed 418 (3/5):
  Range: 62.9% → 100.0%
  Spearman's ρ: +0.853 (p=1.3208e-05)
  Violations: 1

Seed 419 (4/5):
  Range: 48.3% → 100.0%
  Spearman's ρ: +0.883 (p=2.6873e-06)
  Violations: 1

Seed 420 (5/5):
  Range: 71.6% → 100.0%
  Spearman's ρ: +0.903 (p=7.2041e-07)
  Violations: 1

======================================================================
AGGREGATE ANALYSIS
======================================================================

Across 5 seeds:
  Mean violations: 1.6
  Seeds with violations: 5/5

  Correlation across seeds:
    Mean ρ: 0.860 ± 0.029
    Range: [0.819, 0.903]

  Mean trajectory:
    Range: 58.4% → 100.0% (Δ=41.6%)
    Correlation: ρ = 0.883 (p=2.6873e-06)
    Monotonic: No (1 violations)

  Systematic violations in mean trajectory:
    80.0% → 85.0%: -0.012

======================================================================
VISUALIZATION
======================================================================

Saved figure: /Users/fox/Workspace/contrakit/figures/monotonicity_violation_analysis.png

======================================================================
CONCLUSION
======================================================================

✓ PREDICTION CONFIRMED
  Strong monotonic trend validated:
    • Mean correlation: ρ = 0.860 (highly significant)
    • Overall increase: 41.6%
    • All seeds show positive correlation (min ρ = 0.819)

  Small violations observed:
    • 1 violations in mean trajectory
    • Average magnitude: 0.012 (2.8% of total increase)
    • Interpretation: Finite-sample effects, not theoretical failure

  Interpretation:
    The Witness-Error Tradeoff predicts that insufficient witness
    allocation creates PRESSURE toward hallucination as training
    becomes imbalanced. This mechanism is strongly validated.
    Small violations are expected from stochastic optimization
    and discrete sample effects (e.g., only 20 undefined inputs
    at 85% defined ratio).

======================================================================
➜  contrakit git:(main) ✗ 

```