# Experiment 6: Hallucination Across Random Seeds

The first five experiments all used single random seeds, which left an open question: does the relationship between training imbalance and hallucination hold across different random initializations, or did we just happen to pick weight configurations that produced the pattern we saw? To find out, we ran the same experiment with five different seeds, each tested across all 17 defined ratios from 10% to 90%.

The relationship held across every seed. All five showed strong positive correlation between defined ratio and hallucination rate—$\rho = 0.860 \pm 0.029$, with every p-value below 0.001. The starting points varied depending on initialization (ranging from 48.3% to 71.6% hallucination at 10% defined), but the directional trend was consistent: more defined training data led to more hallucination on undefined inputs. Small violations appeared in the trajectory—one to three decreases per seed where hallucination briefly dropped instead of rising—but these represented noise against a 41.6 percentage point increase from start to finish.

## Five Independent Runs

Each seed started with different random weights and trained on the same 17 compositions, from 10% defined up to 90% defined in 5% increments. The architecture stayed constant at $128 \to 64 \to 64 \to 5$, training ran for 100 epochs with cross-entropy loss, and evaluation happened on the same 74 undefined test inputs. Only the initial weights changed between runs.

The correlation numbers tell a consistent story:

| Seed | Spearman $\rho$ | p-value | Violations | Range |
|------|-----------|---------|------------|-------|
| 416 | +0.844 | 2.1e-05 | 3 | 58.6% $\to$ 100.0% |
| 417 | +0.819 | 5.8e-05 | 2 | 50.9% $\to$ 100.0% |
| 418 | +0.853 | 1.3e-05 | 1 | 62.9% $\to$ 100.0% |
| 419 | +0.883 | 2.7e-06 | 1 | 48.3% $\to$ 100.0% |
| 420 | +0.903 | 7.2e-07 | 1 | 71.6% $\to$ 100.0% |

Every single seed showed positive correlation above +0.8. Seed 420 had the strongest relationship at $\rho = 0.903$, while seed 417 had the weakest at $\rho = 0.819$. That's still a strong correlation. The p-values ranged from $7.2 \times 10^{-7}$ to $5.8 \times 10^{-5}$, all far below the standard 0.001 threshold for statistical significance.

The violation counts varied from one to three per seed. Seed 416 showed three local decreases across its 17 points, while seeds 418, 419, and 420 each showed just one. These violations averaged 1.2 percentage points in magnitude—small dips in an otherwise consistent upward trend. Seed 416's three violations didn't prevent it from achieving $\rho = 0.844$, which gives you a sense of how much the overall trend dominated the local noise.

## The Aggregate Pattern

When we average across all five seeds, hallucination starts at 58.4% with 10% defined training data and rises to 100.0% at 90% defined. That's a 41.6 percentage point increase. The mean trajectory shows one violation: between 80% and 85% defined, hallucination drops by 1.2 percentage points. This single decrease represents just 2.8% of the total 41.6 point increase.

The aggregate correlation came out to $\rho = 0.883$ (95% CI: [0.570, 0.990]) with $p = 2.7 \times 10^{-6}$. This is comparable to what we saw in the individual seeds, which makes sense—averaging reduces noise, but it doesn't fundamentally change the relationship. The trajectory follows the same sigmoid shape we identified in Experiment 5: rapid rise from 10% to 30% defined, gradual plateau from 30% to 70%, then saturation approaching 100% at the high end.

The effect size is large (Cohen's d = 4.04), indicating the relationship is not only statistically significant but also practically meaningful. With only 1 observed violation versus 6.8 $\pm$ 1.2 expected under random ordering (p < 0.0001), we have strong evidence that violations are significantly fewer than would occur by chance, confirming the monotonic pressure predicted by theory.

That single violation at 80% → 85% happens precisely where sample sizes become problematic. At 85% defined, only 19 undefined examples remain in training (versus 116 at 10% defined). Testing on 74 undefined inputs while training on just 19 creates substantial interpolation uncertainty. Small changes in which specific examples get labeled can shift test performance by a few points.

## Why Violations Happen

Finite sample effects account for the observed violations. At 10% defined, the model trains on 12 defined examples and 116 undefined examples. By 85% defined, it's training on 109 defined examples but only 19 undefined ones. That's a $6.1 \times$ reduction in undefined sample size, which means more noise in how well the model generalizes to the 74 test inputs.

Statistical testing confirms violations are consistent with finite-sample noise rather than absence of monotonic pressure. Under the null hypothesis that hallucination rates are randomly ordered (no systematic relationship), we'd expect 6.8 $\pm$ 1.2 violations. We observed only 1 violation, which is significantly fewer (p < 0.0001). This demonstrates that the underlying monotonic pressure is strong enough to suppress most violations despite finite-sample variability.

The sigmoid curve from Experiment 5 explains why violations cluster at high defined ratios. Once hallucination reaches 95-97% (which happens around 70% defined), there's very little room left to increase. Small fluctuations around this ceiling can produce local decreases even though the underlying pressure continues pushing upward. You're already failing on nearly every undefined input, so random variation in which specific inputs get misclassified can temporarily lower the rate.

Stochastic optimization adds another layer of noise. Different seeds converge to slightly different local minima because of batch effects, gradient noise, and interactions with the learning rate schedule. Over 17 test points per seed, seeing one to three local decreases is expected purely from optimization randomness. What matters is that the directional trend remained consistent across all five runs—confirmed by all seeds showing $\rho$ > +0.8 and the large effect size (d = 4.04).

## Connection to Theory

The conservation law from the paper (Theorem 7.4) states **$E + r \geq K$**, where $E$ is error **exponent** (bits), $r$ is witness rate (bits/symbol), and $K$ is task contradiction (bits). This is an information-theoretic result about hypothesis testing—how fast error probability decays.

For neural networks making predictions (not hypothesis testing), this implies training imbalance affects hallucination through witness capacity allocation. More defined training data means the model allocates more representational capacity to learning classification patterns, which leaves less capacity for detecting undefined inputs. This manifests as increased hallucination on undefined cases.

The monotonic trend we observed reflects this mechanism operating consistently across different initializations. At 58.4% hallucination with 10% defined, the model has weak classification patterns but sufficient coverage of the undefined region. At 100.0% hallucination with 90% defined, classification patterns dominate but undefined coverage has collapsed entirely. The strong positive correlation ($\rho = 0.860$) shows this pressure operating reliably regardless of which random weights you start from.

**Note:** We're observing monotonic pressure in hallucination **rates**, which is an empirical consequence of the conservation law, not the law itself. See Experiment 9 for detailed discussion of how the conservation law $E + r \geq K$ (using error exponent) implies phase transitions in error rates for neural networks.

## Pressure Versus Determinism

The theoretical prediction is about pressure, not strict determinism at every single point. Monotonic pressure means the underlying force consistently pushes hallucination upward as imbalance increases. Strict monotonicity would mean every adjacent pair of points shows h(t+1) > h(t) with no exceptions at all.

What we observed matches monotonic pressure with small violations. All seeds showed strong positive correlation. The total increase was 41.6 percentage points. Violations averaged 1.2 points in magnitude across one to three instances per seed—small deviations against a large directional change. The trend is robust; finite-sample noise just introduces local variation.

The statistical evidence strongly supports this interpretation. The effect size (Cohen's d = 4.04) is very large, indicating the monotonic trend dominates over noise. The permutation test shows violations occur significantly less often than under random ordering (1 observed vs 6.8 expected, p < 0.0001), confirming that the underlying pressure suppresses most violations. The 95% confidence interval for correlation [0.570, 0.990] is wide due to small sample size (n=17) but excludes zero and negative values, supporting a positive relationship.

Think of it like gravity creating pressure for objects to fall. If you throw a ball upward, it rises briefly against gravity before falling back down. That brief rise isn't evidence against gravitational pressure—it's kinetic energy temporarily overcoming gravity. Similarly, the small hallucination decreases at 80-85% aren't evidence against witness-tradeoff pressure. They're finite-sample noise temporarily overcoming the directional trend.

The aggregate correlation ($\rho = 0.883$) captures this: a strong systematic effect with small random deviations. If the relationship were random or inconsistent, we'd see correlations near zero or even negative values in some seeds. We didn't. Every seed showed $\rho > +0.8$ and every p-value stayed below 0.001. The pressure operates reliably across random initializations.

## Results

![Monotonicity Analysis](image.png)

The visualization shows all five seed trajectories in gray with the mean trajectory in blue. Violation points are highlighted, showing the relationship between training composition and hallucination is strongly monotonic with only minor deviations.

## Running It

```bash
poetry run python examples/hallucinations/experiment_6/run.py
```

The script runs all five seeds across 17 training compositions, displays per-seed correlations and violation counts, computes the aggregate analysis on mean trajectory, and saves a visualization to `image.png`. The left panel shows all individual seed trajectories as gray lines with the mean trajectory in blue and a $\pm$1 standard deviation band. The right panel highlights violation points in red, showing exactly where and by how much the mean trajectory decreased.

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
  Range: 58.6% $\to$ 100.0%
  Spearman's $\rho$: +0.844 (p=2.0839e-05)
  Violations: 3

Seed 417 (2/5):
  Range: 50.9% $\to$ 100.0%
  Spearman's $\rho$: +0.819 (p=5.7641e-05)
  Violations: 2

Seed 418 (3/5):
  Range: 62.9% $\to$ 100.0%
  Spearman's $\rho$: +0.853 (p=1.3208e-05)
  Violations: 1

Seed 419 (4/5):
  Range: 48.3% $\to$ 100.0%
  Spearman's $\rho$: +0.883 (p=2.6873e-06)
  Violations: 1

Seed 420 (5/5):
  Range: 71.6% $\to$ 100.0%
  Spearman's $\rho$: +0.903 (p=7.2041e-07)
  Violations: 1

======================================================================
AGGREGATE ANALYSIS
======================================================================

Across 5 seeds:
  Mean violations: 1.6
  Seeds with violations: 5/5

  Correlation across seeds:
    Mean $\rho$: 0.860 $\pm$ 0.029
    Range: [0.819, 0.903]

  Mean trajectory:
    Range: 58.4% → 100.0% ($\Delta=41.6\%$)
    Correlation: $\rho$ = 0.883 (p=2.6873e-06)
    Monotonic: No (1 violations)

  Systematic violations in mean trajectory:
    80.0% → 85.0%: -0.012

======================================================================
STATISTICAL RIGOR
======================================================================

Effect Size (Cohen's d): 4.04
  Interpretation: Large effect ($d \geq 0.8$)

95% Confidence Interval for $\rho$: [0.570, 0.990]

Null Hypothesis Test:
  H0: Violation count is consistent with random ordering
  Observed violations: 1
  Expected under H0: 6.8 $\pm$ 1.2
  p-value: 0.0000
  Result: Significantly FEWER violations than random (p < 0.05)
          Strong evidence for monotonic pressure

======================================================================
VISUALIZATION
======================================================================

Saved figure: /Users/fox/Workspace/contrakit/figures/monotonicity_violation_analysis.png

======================================================================
CONCLUSION
======================================================================

✓ PREDICTION CONFIRMED
  Monotonic pressure validated (4/4 criteria met):
    • All seeds positive correlation: ✓
    • Statistical significance (p < 0.01): ✓
    • Medium/large effect size ($d \geq 0.5$): ✓
    • Fewer violations than random: ✓

  Summary statistics:
    • Mean correlation: $\rho$ = 0.860, 95% CI [0.570, 0.990]
    • Effect size: d = 4.04
    • Overall increase: 41.6%
    • p-value: 2.6873e-06

  Violations analysis:
    • 1 violations in mean trajectory
    • Average magnitude: 0.012 (2.8% of total increase)
    • Expected under random: 6.8 $\pm$ 1.2
    • Interpretation: Consistent with finite-sample noise, not theoretical failure

  Interpretation:
    The Witness-Error Tradeoff predicts monotonic PRESSURE (directional
    force), not strict determinism. The strong positive correlation,
    large effect size, and significantly fewer violations than random
    chance provide strong evidence for this mechanism. Small violations
    are statistically expected from finite samples (e.g., only ~19
    undefined training examples at 85% defined ratio).

======================================================================
➜  contrakit git:(main) ✗ 

```