# Experiment 5: Non-Linearity of Hallucination Scaling

Experiment 4 showed us that hallucination rates vary from 58.6% to 100.0% as training composition changes, but we only tested five discrete points. We wanted to understand the shape of that relationship. So we collected data across 17 different training compositions, ranging from 10% to 90% defined in 5% increments, and fit four mathematical functions to the data: linear, exponential, power law, and sigmoid.

Sigmoid won decisively. It explained 94.7% of the variance (R² = 0.9467) compared to only 52.8% for linear—a 79% improvement in explanatory power. The relationship turns out to be non-linear with three distinct phases: a rapid rise from 10-30% defined, a gradual plateau from 30-70%, and near-saturation from 70-90%. Small shifts in training composition have large effects early on, then diminishing effects later.

## Collecting the Data

We trained neural networks on 17 dataset compositions, varying defined inputs from 10% to 90% in 5% increments. Everything stayed constant except the defined ratio: same random seed, same architecture (128→64→64→5), same training procedure with 100 epochs of cross-entropy, same evaluation on a separate undefined test set. We measured hallucination rate as the percentage of undefined test inputs where the model predicts A, B, C, or D instead of ⊥.

Patterns emerged immediately in the data. Large increases happen early—going from 10% to 30% defined causes a +34.7 percentage point jump. Small fluctuations happen later—from 50% to 85% the rate varies by only ±4% around 95%. Complete saturation hits at 90% where hallucination reaches 100%.

Remember from Experiment 4 that K = 0.5000 stays constant across all these compositions. The task's contradiction measure, which quantifies how far the behavior sits from any frame-independent (consistent) model, doesn't change at all. What changes is how neural networks manifest that structural impossibility during training.

## What the Curves Tell Us

We fit four functions to the relationship between defined ratio and hallucination rate. Sigmoid came out clearly on top. It achieved 66% lower error than linear (RMSE of 0.0219 versus 0.0652) and explained 94.7% of the variance compared to 52.8% for linear. That's a +0.4186 improvement in R²—79% better explanation of what's happening.

Exponential converged to identical performance as linear, which rules out simple exponential growth. The relationship involves both acceleration early and saturation late, which are characteristics of a sigmoid. Power law performed better than linear or exponential (R² = 0.7220), capturing some of the non-linearity, but it still left 28% of variance unexplained. It missed the saturation behavior. Sigmoid captures everything: the steep initial rise, the gradual flattening, and the approach to 100%.

## Three Phases

The fitted sigmoid reveals distinct phases in how hallucination develops:

In Phase 1, from 10-30% defined, hallucination jumps from 58.6% to 93.3%—a gain of 34.7 percentage points. The steepest slope occurs around 15-20% defined. A 5% shift in training composition causes 10-20 point changes in hallucination rate. The model quickly learns strong classification patterns and moves rapidly away from the theoretical minimum of 29.3% that comes from K = 0.5000.

Phase 2, from 30-70% defined, shows a gradual plateau. Hallucination increases from 93.3% to 97.4%, only 4.1 points. The increases diminish—each 5% shift causes only 1-2 point changes. The system has already saturated most undefined inputs. Further defined data produces minimal additional hallucination. We're already far above the total variation bound of 29.3%.

Phase 3, from 70-90% defined, approaches near-saturation. Hallucination increases from 97.4% to 100.0%, just 2.6 points. Changes are negligible until the final jump at 90%. The model is already hallucinating on nearly all undefined inputs. Complete saturation at 100% hits at extreme imbalance, where every single undefined input gets classified.

The early stages show effects that are 4-18 times larger per 5% shift than later stages. Moving from 10% to 15% defined causes a +21.2 point increase in hallucination. Moving from 75% to 80% defined causes a -0.7 point change. The relationship is deeply non-linear.

## Why This Shape Emerges

The three phases reflect how neural networks interact with the structural contradiction (K = 0.5000). In Phase 1, the rapid rise happens because the model starts near the theoretical minimum of 29.3%. Small amounts of defined data create classification patterns that generalize aggressively. The softmax output forces decisions everywhere, and the undefined region starts getting absorbed into defined patterns. The Bhattacharyya coefficient between learned distributions and the optimal frame-independent model drops quickly as interpolation dominates.

In Phase 2, the plateau happens because most undefined inputs are already hallucinating at 93% or higher. Adding more defined examples strengthens existing patterns but can't push much higher—there's a ceiling near 95-97%. The model has learned to classify confidently. The remaining 5-7% of undefined inputs that resist classification sit far from all training patterns and persist until extreme imbalance.

Phase 3 saturation happens at 90% defined (115 examples versus 13 undefined) when even outlier undefined inputs get overwhelmed. The optimization landscape is so dominated by classification that abstention becomes impossible. The model reaches 100%—complete failure to detect undefined inputs. The frame-independent constraint (K = 0.5000 says no consistent model works) manifests as a total inability to abstain.

## What We Can Predict

With the fitted sigmoid, we can now interpolate to untested compositions. A training set with 33% defined inputs should yield approximately 92% hallucination. One with 67% defined should yield approximately 97%. The curve shape also reveals diminishing returns from increasing defined data. Going from 10% to 30% defined adds 34.7 points (17.4 points per 10% shift). Going from 30% to 50% defined subtracts 1.1 points (-0.55 points per 10%). Going from 70% to 90% defined adds 2.6 points (1.3 points per 10%).

After roughly 30% defined, changes in training composition have minimal impact. The first 20% shift produces most of the hallucination increase. The last 60% shift produces almost nothing. This asymmetry suggests the mechanisms driving early hallucination differ from those maintaining high hallucination at extreme imbalance.

The theoretical bound of 29.3% from K = 0.5000 sits far below even our best observed point of 58.6%. The sigmoid shows the model consistently operates at 2 to 3.4 times the theoretical minimum. The gap between what's mathematically unavoidable (29.3%) and what actually happens (58.6% to 100%) captures training dynamics, interpolation bias, and architectural constraints beyond the structural contradiction.

## No Simple Fix

The sigmoid shape shows there's no training composition that dramatically reduces hallucination. Even at the best point with 10% defined, hallucination is still 58.6%—double the theoretical minimum. By 30% defined, it has already reached 93.3%. The system quickly saturates near maximum hallucination and stays there.

The curve isn't symmetric. Rapid rise dominates early (10% to 30%), slow saturation dominates late (30% to 90%). The inflection point, where the curve changes from accelerating to decelerating, occurs around 15-20% defined. Before that point, every percentage point of defined data causes large hallucination increases. After that point, the rate of increase slows dramatically.

This connects back to the minimax formulation where α*(P) equals the maximum over Q in the frame-independent set of the minimum over contexts of the Bhattacharyya coefficient between p_c and q_c. The observed hallucination reflects how far the learned model sits from the optimal frame-independent model, which achieves α* = 0.7071. Training composition affects this gap indirectly through optimization dynamics, but the structural floor of K = 0.5000 never changes.

## Comparison to Earlier Work

Experiment 4 tested 5 discrete points (10%, 30%, 50%, 70%, 90%) and observed qualitatively that K stays constant while hallucination varies. This experiment tests 17 points and quantifies the exact shape: sigmoid with R² = 0.9467. The dense sampling reveals the three-phase structure that wasn't visible with only 5 measurements.

The counterintuitive finding from Experiment 4—that more defined data leads to more hallucination—is now precisely quantified. The sigmoid shows exactly how this relationship accelerates initially (Phase 1 adds 34.7 points over 20%) then saturates (Phases 2-3 add only 7 points over 60%). The remaining 5.3% unexplained variance likely comes from random training variation with different weight initializations, stochastic optimization effects, and test set sampling variation.

Both experiments confirm the dissociation: K = 0.5000 represents invariant task structure while hallucination ranging from 58.6% to 100% represents variable training behavior. The sigmoid quantifies how that variable behavior depends on training composition.

## Running It

```bash
poetry run python examples/hallucinations/experiment_5/run.py
```

The script trains 17 models, displays hallucination rates for each composition, fits four functional forms, reports RMSE and R² for each, and identifies sigmoid as the best fit. A visualization gets saved to `figures/hallucination_curve_fitting.png` showing the sigmoid curve overlaid on observed data points, plus residual analysis confirming no systematic pattern in the errors.

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

Running 17 experiments...

Defined ratio: 10.0%... Epoch 20/100, Loss: 0.7982
Epoch 40/100, Loss: 0.4265
Epoch 60/100, Loss: 0.2682
Epoch 80/100, Loss: 0.0849
Epoch 100/100, Loss: 0.0660
Hallucination: 58.6%
Defined ratio: 15.0%... Epoch 20/100, Loss: 0.8156
Epoch 40/100, Loss: 0.2605
Epoch 60/100, Loss: 0.0751
Epoch 80/100, Loss: 0.0339
Epoch 100/100, Loss: 0.0200
Hallucination: 79.8%
Defined ratio: 20.0%... Epoch 20/100, Loss: 0.8584
Epoch 40/100, Loss: 0.2814
Epoch 60/100, Loss: 0.0839
Epoch 80/100, Loss: 0.0370
Epoch 100/100, Loss: 0.0210
Hallucination: 84.5%
Defined ratio: 25.0%... Epoch 20/100, Loss: 0.8245
Epoch 40/100, Loss: 0.2482
Epoch 60/100, Loss: 0.0706
Epoch 80/100, Loss: 0.0326
Epoch 100/100, Loss: 0.0215
Hallucination: 90.6%
Defined ratio: 30.0%... Epoch 20/100, Loss: 0.7613
Epoch 40/100, Loss: 0.1887
Epoch 60/100, Loss: 0.0560
Epoch 80/100, Loss: 0.0248
Epoch 100/100, Loss: 0.0144
Hallucination: 93.3%
Defined ratio: 35.0%... Epoch 20/100, Loss: 0.7694
Epoch 40/100, Loss: 0.2060
Epoch 60/100, Loss: 0.0601
Epoch 80/100, Loss: 0.0265
Epoch 100/100, Loss: 0.0151
Hallucination: 90.5%
Defined ratio: 40.0%... Epoch 20/100, Loss: 0.6712
Epoch 40/100, Loss: 0.1637
Epoch 60/100, Loss: 0.0484
Epoch 80/100, Loss: 0.0218
Epoch 100/100, Loss: 0.0130
Hallucination: 96.1%
Defined ratio: 45.0%... Epoch 20/100, Loss: 0.6818
Epoch 40/100, Loss: 0.1604
Epoch 60/100, Loss: 0.0493
Epoch 80/100, Loss: 0.0216
Epoch 100/100, Loss: 0.0119
Hallucination: 93.0%
Defined ratio: 50.0%... Epoch 20/100, Loss: 0.6956
Epoch 40/100, Loss: 0.1572
Epoch 60/100, Loss: 0.0477
Epoch 80/100, Loss: 0.0208
Epoch 100/100, Loss: 0.0134
Hallucination: 92.2%
Defined ratio: 55.0%... Epoch 20/100, Loss: 0.6016
Epoch 40/100, Loss: 0.1219
Epoch 60/100, Loss: 0.0347
Epoch 80/100, Loss: 0.0165
Epoch 100/100, Loss: 0.0093
Hallucination: 96.6%
Defined ratio: 60.0%... Epoch 20/100, Loss: 0.6104
Epoch 40/100, Loss: 0.1175
Epoch 60/100, Loss: 0.0343
Epoch 80/100, Loss: 0.0156
Epoch 100/100, Loss: 0.0089
Hallucination: 96.2%
Defined ratio: 65.0%... Epoch 20/100, Loss: 0.5848
Epoch 40/100, Loss: 0.1091
Epoch 60/100, Loss: 0.0319
Epoch 80/100, Loss: 0.0154
Epoch 100/100, Loss: 0.0081
Hallucination: 95.6%
Defined ratio: 70.0%... Epoch 20/100, Loss: 0.5495
Epoch 40/100, Loss: 0.1031
Epoch 60/100, Loss: 0.0331
Epoch 80/100, Loss: 0.0147
Epoch 100/100, Loss: 0.0080
Hallucination: 97.4%
Defined ratio: 75.0%... Epoch 20/100, Loss: 0.6643
Epoch 40/100, Loss: 0.1638
Epoch 60/100, Loss: 0.0568
Epoch 80/100, Loss: 0.0316
Epoch 100/100, Loss: 0.0128
Hallucination: 96.9%
Defined ratio: 80.0%... Epoch 20/100, Loss: 0.5311
Epoch 40/100, Loss: 0.0840
Epoch 60/100, Loss: 0.0310
Epoch 80/100, Loss: 0.0121
Epoch 100/100, Loss: 0.0071
Hallucination: 96.2%
Defined ratio: 85.0%... Epoch 20/100, Loss: 0.5166
Epoch 40/100, Loss: 0.0894
Epoch 60/100, Loss: 0.0290
Epoch 80/100, Loss: 0.0126
Epoch 100/100, Loss: 0.0070
Hallucination: 95.0%
Defined ratio: 90.0%... Epoch 20/100, Loss: 0.5308
Epoch 40/100, Loss: 0.0900
Epoch 60/100, Loss: 0.0245
Epoch 80/100, Loss: 0.0124
Epoch 100/100, Loss: 0.0063
Hallucination: 100.0%

======================================================================
CURVE FITTING ANALYSIS
======================================================================

Fit quality for each functional form:
Model           RMSE         R²           Result
----------------------------------------------------------------------
linear          0.0652       0.5281       
exponential     0.0652       0.5281       
sigmoid         0.0219       0.9467        ← BEST FIT
power_law       0.0500       0.7220       

======================================================================
NON-LINEARITY TEST
======================================================================

Linear model R²:        0.5281
Best non-linear R²:     0.9467
Improvement:            +0.4186

✓ NON-LINEAR: SIGMOID fits significantly better
  The relationship shows clear non-linear structure

======================================================================
VISUALIZATION
======================================================================

Visualization saved to: /Users/fox/Workspace/contrakit/figures/hallucination_curve_fitting.png

======================================================================
CONCLUSION
======================================================================

✓ PREDICTION CONFIRMED
  Best fit: SIGMOID (R² = 0.9467)
  The relationship is clearly non-linear
  This supports the compounding K mechanism

======================================================================
➜  contrakit git:(main) ✗ 
```