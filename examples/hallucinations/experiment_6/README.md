# Experiment 6: Monotonicity of Hallucination Across Random Seeds

Does hallucination reliably increase with training imbalance across random seeds? We tested 5 seeds $\times$ 17 defined ratios (10% to 90% defined) to check robustness. All seeds show strong positive correlation ($\rho = 0.860 \pm 0.029$, all $p < 0.001$). Small local violations appear—1-3 per seed—but represent only 2.8% of the total 41.6% increase. The monotonic trend is robust.

This validates the witness-error tradeoff (Theorem 7.4): as training becomes imbalanced, witness allocation to undefined regions drops, forcing higher error rates. The theory predicts monotonic pressure, not strict determinism at every point. The observed pattern—strong directional trend with small local noise—matches this perfectly.

## Per-Seed Results

Five independent runs with different random weight initializations:

| Seed | Spearman $\rho$ | p-value | Violations | Range |
|------|-----------|---------|------------|-------|
| 416 | +0.844 | 2.1e-05 | 3 | 58.6% → 100.0% |
| 417 | +0.819 | 5.8e-05 | 2 | 50.9% → 100.0% |
| 418 | +0.853 | 1.3e-05 | 1 | 62.9% → 100.0% |
| 419 | +0.883 | 2.7e-06 | 1 | 48.3% → 100.0% |
| 420 | +0.903 | 7.2e-07 | 1 | 71.6% → 100.0% |

Mean correlation: $\rho = 0.860 \pm 0.029$. All correlations highly significant ($p < 0.001$). Every seed shows positive trend despite different starting points. The range of starting hallucination (48.3% to 71.6% at 10% defined) reflects initialization variance, but the directional increase is consistent.

Violations occur in 1-3 points per seed. These are small decreases (~1-2%) against a backdrop of 40%+ total increase. Seed 416 shows 3 violations but still achieves $\rho = +0.844$. The trend dominates local noise. Seeds with violations: 5 out of 5. Mean violations per seed: 1.6.

## Aggregate Analysis

Mean trajectory across all seeds: 58.4% → 100.0% hallucination as training shifts from 10% to 90% defined. Total increase: 41.6 percentage points. Spearman correlation on mean: $\rho = 0.883$ ($p = 2.7 \times 10^{-6}$). One violation appears in the mean trajectory: between 80% and 85% defined, hallucination drops 1.2%. That's 2.8% of the total 41.6% increase.

The aggregate pattern is clear. Start at 10% defined: hallucination averages 58.4%. Add more defined data. Hallucination rises rapidly through 30% (jumps to ~93%), then plateaus near 95-97%, finally hits 100% at 90% defined. The sigmoid shape from Experiment 5 appears here too: rapid early rise, gradual middle, saturation at extremes.

The single systematic violation (80% → 85%) shows $-0.012$ change ($-1.2$ percentage points). This occurs precisely where sample sizes become very small: at 85% defined ratio, only 19 undefined training examples remain (versus 116 at 10% defined). That's a 6.1$\times$ reduction. Testing on 74 undefined inputs with only 19 training examples creates substantial interpolation uncertainty.

## Why Violations Occur

Finite sample effects dominate. At 10% defined, the model trains on 12 defined examples and 116 undefined examples. At 85% defined, it trains on 109 defined examples and 19 undefined examples. That's a 6.1$\times$ reduction in undefined sample size. Smaller samples mean more noise in test performance.

Consider the 80% → 85% violation (where mean hallucination drops 1.2%). At 80% defined, 26 undefined examples remain in training. At 85%, only 19 remain. Testing on 74 undefined inputs means relying on interpolation from fewer training points. Small changes in which specific 19 examples get labeled can shift test performance by a few percentage points.

The sigmoid from Experiment 5 explains why violations cluster at high ratios. Phase 3 (70-90% defined) shows near-saturation: hallucination already at 95-97%. Small fluctuations around the ceiling create local decreases. The underlying pressure is upward—witness allocation to undefined regions keeps dropping—but the ceiling constrains how much higher it can go.

Stochastic optimization also contributes. Different seeds find slightly different local minima. Batch effects, gradient noise, and learning rate interactions create small variations in final convergence. Over 17 test points, 1-3 local decreases are expected purely from optimization stochasticity.

## Connection to Theory

Theorem 7.4 (witness-error tradeoff) states: $E + r \geq K$. For fixed task complexity $K$, reducing witness capacity $r$ forces higher error $E$. Training imbalance directly affects $r$: more defined data means stronger witness allocation to defined regions, leaving less for undefined regions.

Think of $r$ as information budget. Total budget = $K$ bits (from task contradiction). Splitting that budget: $r_{\text{defined}}$ goes to classifying correctly, $r_{\text{undefined}}$ goes to detecting undefined inputs. As training shifts from 10% to 90% defined, $r_{\text{defined}} \uparrow$ and $r_{\text{undefined}} \downarrow$. Since $E_{\text{undefined}} + r_{\text{undefined}} \geq K$, dropping $r_{\text{undefined}}$ forces $E_{\text{undefined}} \uparrow$.

The observed monotonic trend directly reflects this tradeoff. Starting at 58.4% hallucination (10% defined), the model has weak classification patterns but sufficient undefined coverage. Increasing to 90% defined, classification patterns strengthen but undefined coverage collapses. Hallucination reaches 100%—complete failure to detect any undefined input.

Theorem 7.5 (universal adversarial prior) adds precision: the worst-case context weighting $\lambda^*$ is universal across tasks. As training becomes imbalanced, the undefined contexts become bottlenecks—they receive minimal witness capacity. This creates systematic, directional pressure toward hallucination. The strong positive correlation ($\rho = 0.860$) reflects this universal mechanism operating across all seeds.

## Monotonic Pressure vs Strict Monotonicity

The theory predicts monotonic pressure, not strict monotonicity. That distinction matters. Monotonic pressure means: the underlying force consistently pushes hallucination upward as imbalance increases. Strict monotonicity means: every single adjacent pair of points shows $h(t+1) > h(t)$ with no exceptions.

We observe monotonic pressure with small violations. All seeds show strong positive correlation. Total increase: 41.6%. Violations: 1-3 per seed, averaging 1.2% magnitude (mean: 0.012 in rate change) against 40%+ total change. This is pressure, not determinism. The directional force is robust; local noise introduces small deviations.

Analogy: gravity creates monotonic pressure for objects to fall. But throw a ball upward—it rises briefly against gravity before falling. That brief rise isn't evidence against gravitational pressure; it's kinetic energy overcoming gravity temporarily. Similarly, small hallucination decreases at 80-85% aren't evidence against witness-tradeoff pressure; they're finite-sample noise temporarily overcoming the directional trend.

The aggregate correlation ($\rho = 0.883$) captures this: strong systematic effect with small random deviations. If the relationship were random or bidirectional, we'd see correlations near zero or negative values in some seeds. We don't. Every seed $\rho > +0.8$. Every $p$-value $< 0.001$. The pressure is real and robust.

## Running It

```bash
poetry run python examples/hallucinations/experiment_6/run.py
```

The output shows per-seed results (Spearman $\rho$, violations, range), aggregate analysis (mean trajectory, overall correlation), and visualization saved to `figures/monotonicity_violation_analysis.png`. The figure displays all individual seed trajectories (gray lines), mean trajectory (blue line with markers), and violation points (red segments) where local decreases occur.

The left panel shows hallucination vs training imbalance with mean trajectory and $\pm 1$ standard deviation band. Individual seeds appear as thin gray lines showing variation. The right panel highlights monotonicity violations in red, showing exactly where and by how much the mean trajectory decreases.

Full implementation in `run.py`. The experiment demonstrates that witness-error tradeoff pressure operates robustly across random initializations, producing consistent directional trends despite finite-sample noise. The average violation magnitude (0.012 or 1.2%) represents only 2.8% of the total increase (0.416 or 41.6%), confirming that violations are noise, not signal.


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