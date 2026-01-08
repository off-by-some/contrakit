# Experiment 5: Non-Linearity of Hallucination Scaling

Experiment 4 showed us that hallucination rates vary from 58.6% to 100.0% as training composition changes, but we only tested five discrete points. We wanted to understand the shape of that relationship. So we collected data across 17 different training compositions, ranging from 10% to 90% defined in 5% increments. For robustness, we trained 3 models per composition with different random seeds (51 total training runs). We then fit four mathematical functions to the data: linear, exponential, power law, and sigmoid, using proper model selection criteria (AIC, BIC, and cross-validation).

Sigmoid won decisively across all three model selection criteria. It explained 94.4% of the variance ($R^2$ = 0.9443, 95% CI: [0.7747, 0.9769]) compared to only 67.2% for linear. The evidence for non-linearity is strong: $\Delta BIC$ = 27.3, well above the threshold of 10 for strong statistical evidence. Cross-validation $R^2$ of 0.8655 confirms the model generalizes well. The relationship is non-linear with three distinct phases: a rapid rise from 10-30% defined, a gradual plateau from 30-70%, and near-saturation from 70-90%. Small shifts in training composition have large effects early on, then diminishing effects later.

## Collecting the Data

We trained neural networks on 17 dataset compositions, varying defined inputs from 10% to 90% in 5% increments. For each composition, we trained 3 separate models with different random seeds to ensure robustness (51 total training runs). Everything else stayed constant: same architecture ($128 \to 64 \to 64 \to 5$), same training procedure with 100 epochs of cross-entropy, same evaluation on a separate undefined test set. We measured hallucination rate as the percentage of undefined test inputs where the model predicts A, B, C, or D instead of ⊥. For each composition, we report the mean hallucination rate across the 3 seeds along with the standard deviation.

Patterns emerged immediately in the data. Large increases happen early—going from 10% to 30% defined causes approximately a +23 percentage point jump. Small fluctuations happen later—from 50% to 85% the rate varies by only a few percentage points around 95%. Complete saturation hits at 90% where hallucination reaches 100%. The variability across seeds (measured by standard deviation) is generally small, ranging from 0.0% to 12.5%, with the largest uncertainty at 10% defined where the model is most sensitive to initialization.

Remember from Experiment 4 that $K = 0.5000$ stays constant across all these compositions. The task's contradiction measure, which quantifies how far the behavior sits from any frame-independent (consistent) model, doesn't change at all. What changes is how neural networks manifest that structural impossibility during training.

## What the Curves Tell Us

We fit four functions to the relationship between defined ratio and hallucination rate, using proper statistical methodology to avoid overfitting. Model selection used three independent criteria: AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), and leave-one-out cross-validation. All three criteria agreed on sigmoid as the best model, providing strong consensus.

Sigmoid came out clearly on top. It achieved 59% lower error than linear (RMSE of 0.0199 versus 0.0482) and explained 94.4% of the variance with 95% confidence interval [0.7747, 0.9769], compared to 67.2% for linear. The cross-validation $R^2$ of 0.8655 confirms the model generalizes well to held-out data, not just fitting the training points. The evidence for non-linearity is statistically strong: $\Delta BIC$ = 27.3 (values above 10 indicate strong evidence), meaning the sigmoid is decisively better even after accounting for its additional complexity.

Exponential converged to nearly identical performance as linear, which rules out simple exponential growth. The relationship involves both acceleration early and saturation late, which are characteristics of a sigmoid. Power law performed better than linear or exponential ($R^2$ = 0.8486), capturing some of the non-linearity, but cross-validation revealed it doesn't generalize as well as sigmoid. Sigmoid captures everything: the steep initial rise, the gradual flattening, and the approach to 100%.

## Three Phases

The fitted sigmoid reveals distinct phases in how hallucination develops:

In Phase 1, from 10-30% defined, hallucination jumps from approximately 65.8% to 88.9%—a gain of roughly 23 percentage points. The steepest slope occurs around 15-20% defined. A 5% shift in training composition causes 5-15 point changes in hallucination rate in this region. The model quickly learns strong classification patterns and moves rapidly away from the theoretical minimum of 29.3% that comes from $K = 0.5000$.

Phase 2, from 30-70% defined, shows a gradual plateau. Hallucination increases from 88.9% to 97.4%, approximately 8.5 points over a 40% range. The increases diminish—each 5% shift causes only 1-3 point changes. The system has already saturated most undefined inputs. Further defined data produces minimal additional hallucination. We're already far above the total variation bound of 29.3%.

Phase 3, from 70-90% defined, approaches near-saturation. Hallucination increases from 97.4% to 100.0%, just 2.6 points. Changes are negligible until the final jump at 90%. The model is already hallucinating on nearly all undefined inputs. Complete saturation at 100% hits at extreme imbalance, where every single undefined input gets classified.

The early stages show substantially larger effects per percentage point of defined data compared to later stages. The relationship is deeply non-linear, as confirmed by the statistical analysis showing $\Delta BIC$ = 27.3 in favor of the sigmoid over linear models.

## Why This Shape Emerges

The three phases reflect how neural networks interact with the structural contradiction ($K = 0.5000$). In Phase 1, the rapid rise happens because the model starts near the theoretical minimum of 29.3%. Small amounts of defined data create classification patterns that generalize aggressively. The softmax output forces decisions everywhere, and the undefined region starts getting absorbed into defined patterns. The Bhattacharyya coefficient between learned distributions and the optimal frame-independent model drops quickly as interpolation dominates.

In Phase 2, the plateau happens because most undefined inputs are already hallucinating at 93% or higher. Adding more defined examples strengthens existing patterns but can't push much higher—there's a ceiling near 95-97%. The model has learned to classify confidently. The remaining 5-7% of undefined inputs that resist classification sit far from all training patterns and persist until extreme imbalance.

Phase 3 saturation happens at 90% defined (115 examples versus 13 undefined) when even outlier undefined inputs get overwhelmed. The optimization landscape is so dominated by classification that abstention becomes impossible. The model reaches 100%—complete failure to detect undefined inputs. The frame-independent constraint ($K = 0.5000$ says no consistent model works) manifests as a total inability to abstain.

## What We Can Predict

With the fitted sigmoid, we can interpolate to untested compositions within the 10-90% range tested. The curve shape reveals diminishing returns from increasing defined data. Going from 10% to 30% defined produces the largest increases. Going from 30% to 70% defined produces much smaller changes. Going from 70% to 90% defined shows minimal change until complete saturation at 90%.

After roughly 30% defined, changes in training composition have diminishing impact. The first 20% range produces the largest effects. The last 60% range produces smaller changes. This asymmetry suggests the mechanisms driving early hallucination differ from those maintaining high hallucination at extreme imbalance.

The theoretical bound of 29.3% from $K = 0.5000$ sits far below even our best observed point of approximately 65.8%. The sigmoid shows the model consistently operates at 2 to 3.4 times the theoretical minimum. The gap between what's mathematically unavoidable (29.3%) and what actually happens (65.8% to 100%) captures training dynamics, interpolation bias, and architectural constraints beyond the structural contradiction.

## No Simple Fix

The sigmoid shape shows there's no training composition that dramatically reduces hallucination. Even at the best point with 10% defined, hallucination is still approximately 65.8%—more than double the theoretical minimum. By 30% defined, it has already reached approximately 88.9%. The system quickly saturates near maximum hallucination and stays there.

The curve isn't symmetric. Rapid rise dominates early (10% to 30%), slow saturation dominates late (30% to 90%). The inflection point, where the curve changes from accelerating to decelerating, occurs around 15-20% defined. Before that point, every percentage point of defined data causes large hallucination increases. After that point, the rate of increase slows dramatically.

This connects back to the minimax formulation where $\alpha^*(P)$ equals the maximum over $Q$ in the frame-independent set of the minimum over contexts of the Bhattacharyya coefficient between $p_c$ and $q_c$. The observed hallucination reflects how far the learned model sits from the optimal frame-independent model, which achieves $\alpha^* = 0.7071$. Training composition affects this gap indirectly through optimization dynamics, but the structural floor of $K = 0.5000$ never changes.

## Comparison to Earlier Work

Experiment 4 tested 5 discrete points (10%, 30%, 50%, 70%, 90%) and observed qualitatively that K stays constant while hallucination varies. This experiment tests 17 points with 3 random seeds each (51 total training runs) and quantifies the exact shape: sigmoid with $R^2$ = 0.9443 [0.7747, 0.9769]. The dense sampling reveals the three-phase structure that wasn't visible with only 5 measurements. Statistical evidence for non-linearity is strong ($\Delta BIC$ = 27.3), and cross-validation confirms the model generalizes well (CV $R^2$ = 0.8655).

The counterintuitive finding from Experiment 4—that more defined data leads to more hallucination—is now precisely quantified with proper statistical methodology. The sigmoid shows exactly how this relationship changes: rapid increases in Phase 1, then saturation in Phases 2-3. The remaining ~5.6% unexplained variance likely comes from random training variation, stochastic optimization effects, and test set sampling variation—sources of noise that cannot be eliminated even with multiple random seeds.

Both experiments confirm the dissociation: $K = 0.5000$ represents invariant task structure while hallucination ranging from 58.6% to 100% represents variable training behavior. The sigmoid quantifies how that variable behavior depends on training composition.

## Running It

```bash
poetry run python examples/hallucinations/experiment_5/run.py
```

The script trains 51 models (17 compositions $\times$ 3 seeds), displays mean hallucination rates with standard deviations for each composition, fits four functional forms with proper model selection criteria (AIC, BIC, cross-validation), and identifies sigmoid as the best fit. A visualization gets saved to `figures/hallucination_curve_fitting.png` showing the sigmoid curve overlaid on observed data points with error bars, plus residual analysis.

The full implementation is in `run.py`. The experiment quantifies the non-linear relationship between training composition and hallucination, revealing three distinct phases and demonstrating that small early shifts have outsized effects.

---

### Example Output

```
contrakit git:(main) ✗ poetry run python examples/hallucinations/experiment_5/run.py
======================================================================
TEST: Prediction 7 - Non-linear Hallucination Curve
======================================================================

Prediction:
  Relationship between training imbalance and hallucination
  should be non-linear (exponential or sigmoidal curve).

Mechanism:
  Compounding of local K values through learned priors

======================================================================
DATA COLLECTION
======================================================================

Running 17 experiments with 3 seeds each for robustness...

Defined ratio: 10.0%... [Training output for 3 seeds...]
Hallucination: 65.8% $\pm$ 12.5%
Defined ratio: 15.0%... [Training output for 3 seeds...]
Hallucination: 80.1% $\pm$ 3.8%
[... additional ratios ...]
Defined ratio: 90.0%... [Training output for 3 seeds...]
Hallucination: 100.0% $\pm$ 0.0%

======================================================================
CURVE FITTING ANALYSIS (with model selection criteria)
======================================================================

Fit quality for each functional form:
Model           Params   RMSE         $R^2$                   CV $R^2$        AIC          BIC         
----------------------------------------------------------------------------------------------------
linear          2        0.0482       0.6716 [-0.0194, 0.7978] 0.5257       -99.08       -97.42      
exponential     3        0.0482       0.6715 [-0.0194, 0.7978] 0.5257       -97.08       -94.58      
sigmoid         3        0.0199       0.9443 [0.7747, 0.9769] 0.8655       -127.25      -124.75     
power_law       2        0.0327       0.8486 [0.4264, 0.9057] 0.7588       -112.25      -110.59     

Best model by AIC (penalizes complexity): sigmoid
Best model by BIC (penalizes complexity more): sigmoid
Best model by cross-validation: sigmoid

✓ All criteria agree: SIGMOID is the best model

======================================================================
NON-LINEARITY ASSESSMENT
======================================================================

Linear model:
  $R^2$ = 0.6716 [-0.0194, 0.7978]
  CV $R^2$ = 0.5257
  AIC = -99.08
  BIC = -97.42

Best non-linear model (sigmoid):
  $R^2$ = 0.9443 [0.7747, 0.9769]
  CV $R^2$ = 0.8655
  AIC = -127.25
  BIC = -124.75

Model comparison:
  $\Delta AIC = 28.17$ (>10 = strong evidence for non-linear)
  $\Delta BIC$ = 27.33 (>10 = strong evidence for non-linear)

✓ NON-LINEAR: SIGMOID strongly preferred
  Strong statistical evidence for non-linear structure

======================================================================
VISUALIZATION
======================================================================

Visualization saved to: /Users/fox/Workspace/contrakit/figures/hallucination_curve_fitting.png

======================================================================
CONCLUSION
======================================================================

✓ PREDICTION CONFIRMED
  Best fit: SIGMOID
  $R^2$ = 0.9443 [95% CI: 0.7747-0.9769]
  CV $R^2$ = 0.8655
  The relationship shows statistically significant non-linear structure
  This supports the compounding mechanism hypothesis

======================================================================
➜  contrakit git:(main) ✗ 
```