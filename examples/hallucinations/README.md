# Neural Network Hallucination: What We Found

Neural networks sometimes answer with complete confidence when they shouldn't. You give them inputs they've never seen, and they fabricate responses as if certain. We wanted to understand where that confidence comes from, so we trained networks on carefully constructed tasks and measured their behavior across hundreds of runs. The patterns we found held regardless of architecture, training conditions, or random weight initialization.

Hallucination stems from two sources that work independently but compound when combined. The first is structural: some tasks demand incompatible behaviors across different contexts—answer A here, answer B there, but using the same query. When you train on both contexts then remove the context marker, no single answer works for both. That creates an impossibility that no training procedure can resolve. The second source is architectural: softmax requires producing a definite prediction for every input, whether prediction makes sense or not. This forces the network to guess even when it shouldn't.

## The Core Finding

Our experiments revealed a pattern implying that hallucination isn't just a training data problem—but instead, it's structural and architectural, with mathematical constraints that more data cannot overcome.

We can quantify the structural pressure using a measure we call $K$, computed in bits from the task definition before any training occurs. When $K = 0$, the task has no internal contradictions and perfect performance is possible. When $K > 0$, contradictions exist and some minimum error becomes mathematically unavoidable. For example, $K = 0.5$ bits guarantees at least 29% error when models must commit to answers. This bound applies equally to neural networks, decision trees, hand-coded rules, or humans guessing—the impossibility is mathematical, not a property of any particular learning system.

The architectural pressure operates independently. We measured this by quantifying witness capacity $r$, which indicates how much uncertainty an architecture can express. Standard softmax provides $r \approx 0$ bits because it forces probability distributions summing to 1.0 across all outputs. That means there's no way to represent "I don't know" or "none of these options apply." Architectures with explicit abstention mechanisms provide $r \geq 1$ bit, giving the model ways to express uncertainty rather than guessing.

These two pressures interact in predictable ways. When $r$ exceeds $K$, we observed sharp phase transitions where error collapsed from near 100% to near 0% in a narrow zone around $r = K$. This demonstrates that $K$ indicates required capacity rather than task difficulty—systems need adequate $r$ to handle structural contradictions. Without it, failure is nearly certain regardless of training.

Training composition affects how far above the theoretical minimum you land, but it cannot eliminate structural impossibility when $K > 0$. We varied the ratio of defined to undefined examples from 10% to 90% and found a sigmoid relationship. Hallucination rises rapidly from 10-30% defined, then saturates beyond 70%. This means composition modulates distance from the theoretical floor but cannot break through it.

Standard softmax architectures face inherent limitations because $r \approx 0$ leaves them unable to handle epistemic uncertainty—situations where the model simply lacks information about out-of-distribution inputs. We demonstrated this with a $K = 0.70$ bit task where hallucination was 76% when forced to commit but dropped to 1% when abstention was allowed. That 75 percentage point gap isolates architectural pressure from structural pressure, showing they operate independently. Reducing hallucination requires architectural mechanisms for context-aware abstention, not just more data.

## Measuring Task Structure

Before training any network, we can compute $K$ from the task definition and determine whether consistent behavior is possible. When $K$ equals zero, a single coherent strategy exists—one set of rules that works across all training contexts. When $K$ exceeds zero, no such strategy exists because the training contexts contain incompatibilities no single model can resolve.

This measure tells us about the nature of the problem. It isn't about difficulty or complexity—$K$ quantifies structural impossibility instead. For $K = 0.5$ bits, information theory provides a precise formula showing any model working across all contexts must fail on at least 29% of cases when forced to commit. The bound applies equally to neural networks, decision trees, hand-coded rules, or even humans guessing because the impossibility is mathematical.

We can construct tasks with specific $K$ values by controlling how contexts relate. Some functions are partial—they have gaps where they're undefined:

```python
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def tomorrow(today):
    """Returns the next day, or None if input is invalid."""
    if today not in DAYS:
        return None  # Undefined: not a valid day
    index = DAYS.index(today)
    return DAYS[(index + 1) % 7]

# This works:
print(tomorrow("Monday"))  # "Tuesday"

# This doesn't:
print(tomorrow())  # Error: missing required argument
```

Neural networks can approximate partial functions—they're universal function approximators after all. But standard architectures force them to produce definite outputs for every input. Softmax requires a probability distribution over output classes summing to 1.0, so there's no native way to represent "this input is undefined" or "none of these options apply."

This creates a mismatch. When you train a network on "Today is Monday, tomorrow is Tuesday" in one context and "Today is Thursday, tomorrow is Friday" in another, both examples are correct. But if you ask "What comes after today?" without specifying context, that query has no answer that works for both training contexts. The network must produce something, so it hallucinates. The quantity $K$ tells us exactly how impossible the situation is.

## What Happens During Training

We built a simple task with 128 possible inputs and 5 output classes (A, B, C, D, ⊥). Of these inputs, 51 were defined and should map to A, B, C, or D. The remaining 77 were undefined and should map to ⊥. Training used all 51 defined examples plus 3 undefined examples explicitly labeled ⊥, while the other 74 undefined cases never appeared.

The network learned the defined examples perfectly, achieving 100% accuracy with 98.85% confidence on those 51 inputs. But on the 74 undefined test inputs it had never seen, its behavior changed dramatically—it fabricated answers 96.1% of the time with 59.54% confidence. That confidence sits between random guessing (20%) and learned certainty (98.85%), suggesting the network blends nearby training patterns rather than abstaining.

This behavior held across architectures. We tried adding a separate "definedness head"—a second output branch trained specifically to detect undefined inputs. Testing across 9 training compositions with 3 random seeds each (54 total runs), this definedness head reduced hallucination by only 0.6 percentage points on average, from 89.4% to 88.7%. It achieved perfect accuracy on the three undefined training examples but only 4.3% on the 74 unseen undefined test cases. The head memorized specific examples rather than learning the general concept.

![Model Comparison](experiment_2/model_comparison.png)

## When Networks Learn Uncertainty

The definedness head failed to generalize, but one experiment showed networks can learn uncertainty under different conditions. We designed a task where the same input appears with conflicting labels in the training data. Two rules apply to all possible (X,Y) input pairs: the X-rule says output Z equals X, while the Y-rule says output Z equals NOT Y. These rules agree for inputs (0,1) and (1,0) but contradict for (0,0) and (1,1), and the training set contains equal numbers of examples from each rule. As a result, the model sees (0,0)→0 in some examples and (0,0)→1 in others.

We computed $K = 0.0760$ bits from the task structure before training, which bounds minimum error at 5.1% when forced to make binary predictions. In other words, perfect accuracy is mathematically impossible.

Training across 10 random seeds, the network learned something appropriate. On the two inputs where rules agree, it achieved 100% accuracy with 100% confidence. On the two contradictory inputs, it output approximately 50% confidence—essentially expressing "both answers appeared equally in training, I cannot choose." Hallucination rate was just 0.4% ± 0.4%.

This demonstrates that neural networks can learn uncertainty, but only under specific conditions. The contradictions must be explicit in training data—the same input appearing with different labels. This represents aleatoric uncertainty: randomness in the data-generating process itself. The model has direct evidence of the ambiguity and learns to represent it accordingly.

Contrast this with epistemic uncertainty: gaps in the model's knowledge. When networks encounter inputs they never saw during training, they lack any signal about uncertainty. Training provided no examples of "I don't know" in the relevant regions of input space, and the architecture forces predictions everywhere. This explains why the standard network in our first experiment hallucinated at 96.1%—it faced epistemic uncertainty with no mechanism to express it.

## Training Composition Matters

The previous experiment showed networks handling aleatoric uncertainty well but failing on epistemic uncertainty. We wanted to understand how the balance of defined versus undefined training examples affects this epistemic case, so we varied the composition systematically. We tested ratios from 10% defined to 90% defined across 5 compositions with proper train/test splits and multiple random seeds. The quantity $K$ stayed constant at 0.5000 bits across all compositions (verified to 4 decimal places), so the task structure never changed. But hallucination rates ranged from 51.9% (±7.3%) at 10% defined to 98.3% (±2.1%) at 70% defined.

This result seems counterintuitive at first. Usually more training data improves performance. Here it makes things worse. The network doesn't learn to partition space into "I know this" and "I don't know this" regions—instead, it learns a smooth function that interpolates everywhere.

With more defined examples, the interpolation becomes more confident in its extrapolations, even into regions where it should abstain. When you have 115 defined examples and only 3 undefined ones, optimization overwhelmingly favors correct classification. The loss function sees 115 examples rewarding confident predictions and just 3 suggesting abstention, so almost all gradient flow pushes toward classification.

The theoretical minimum of 29.3% (from $K = 0.5000$ bits) held in every configuration—we never saw rates below it. But observed rates climbed far higher, ranging from 77% above the minimum at 10% defined to 236% above at 70% defined.

To understand the relationship precisely, we tested 17 compositions with 3 random seeds each—51 total training runs. We fit four mathematical functions (linear, exponential, power law, sigmoid) using proper model selection criteria, and all three criteria (AIC, BIC, cross-validation) agreed: sigmoid fit best. It explained 94.4% of variance with strong statistical evidence (ΔBIC = 27.3) and cross-validation $R^2$ of 0.8655.

![Hallucination Curve](../../figures/hallucination_curve_fitting.png)

The sigmoid reveals three distinct phases. From 10% to 30% defined, hallucination jumps roughly 23 percentage points—the steepest region of the curve. From 30% to 70% defined, increases diminish to about 8.5 points across 40 percentage points of training composition. From 70% to 90%, near-saturation appears with only 2.6 additional points. By 30% defined composition, hallucination already reaches approximately 89%, and adding more defined data barely changes the outcome.

We tested whether this pattern held across different random initializations. Testing the same procedure with five different random seeds, all showed strong positive correlation between defined ratio and hallucination: ρ = 0.860 ± 0.029, every p-value below 0.001. The starting points varied (48.3% to 71.6% hallucination at 10% defined), but the directional trend remained consistent. Small violations appeared—one to three points per seed where hallucination briefly decreased—but these averaged only 1.2 percentage points against a 41.6 point total increase. Statistical testing confirmed violations occurred significantly less than expected under random ordering (1 observed versus 6.8 expected, p < 0.0001), demonstrating monotonic pressure despite finite-sample noise.

![Monotonicity Analysis](experiment_6/image.png)

The effect size was large (Cohen's d = 4.04), indicating this isn't an artifact of particular weight initializations. It reflects how gradient descent distributes capacity between classification and abstention when the training signal heavily favors one over the other.

## Architectural Forcing

The sigmoid relationship showed how training composition affects hallucination, but that still operates within the constraints of a given architecture. To isolate architectural effects from compositional ones, we needed to test the same task under different architectural conditions. We used llama3.1:8b on weekday prediction tasks with varying numbers of contexts. Each context specified a different day as "today" and asked "What day comes after today?" without providing context. For n contexts, we measured $K$ from the model's context-conditional response patterns, then tested behavior when context was removed.

Tasks ranged from n=1 ($K=0$, control condition) to n=5 ($K=1.10$ bits). Even $K=0$ showed 45% fabrication—underspecification pressure, since "tomorrow" depends on knowing "today." For $K>0$ tasks, fabrication increased monotonically from 64% to 75% as $K$ grew, never violating the theoretical bounds (29% to 53%).

Then we ran one task ($K=0.70$ bits, 3 contexts) under two architectural conditions. When the model could select "unknown" as a valid response, hallucination was 1% (495 abstentions, 5 fabrications out of 500 trials). When we forced the model to pick a specific weekday with no "unknown" option, hallucination jumped to 76% (380 fabrications, 120 abstentions out of 500 trials).

![Architectural Comparison](experiment_7/contradiction_hallucination_analysis.png)

That 75 percentage point gap isolates architectural pressure from structural pressure. The theoretical minimum for $K=0.70$ bits is 40%. With abstention support, the model stayed near that floor at 1%. Without it, hallucination shot to 76%. The structural contradiction remained constant—only the architectural support changed.

Softmax creates this forcing. Every input must produce a probability distribution summing to 1.0 across output classes, leaving no native way to represent "none of these options apply." The architecture treats every input identically: embed, transform through layers, project to output space, apply softmax, select highest probability. Even when the input has nothing to do with training data, this process generates a confident prediction.

## The Phase Transition

The 75-point gap isolated architectural pressure, showing its effect on a single task with fixed $K$. To understand how architectural capacity interacts with task contradiction more generally, we needed to vary both quantities systematically across many combinations. We tested task contradiction $K$ from 0.5 to 1.16 bits and witness capacity $r$ from 0 to 2 bits across 20 combinations with 5 random seeds each—100 training runs total. Witness capacity measures the architecture's ability to express uncertainty, where $r = \log_2$(number of abstention states). Standard softmax provides $r \approx 0$, while an architecture with 2 abstention states provides $r = 1$ bit.

![Error Rate Heatmap](experiment_9/results/error_rate_heatmap.png)

The results showed sharp transitions. When $r$ fell well below $K$, error stayed at 100%—forced hallucination on all contradictory inputs across all seeds. When $r$ exceeded $K$ by a comfortable margin, error dropped to 0%—successful abstention across all runs. The transition happened in a narrow zone near $r = K$.

![Phase Transition](experiment_9/results/error_vs_witness.png)

This confirms $K$ indicates required capacity, not difficulty. Systems need $r$ above $K$ for reliable success, and training cannot compensate for insufficient architectural capacity. If $r$ is well below $K$, failure is nearly certain regardless of data or optimization.

Standard softmax architectures have $r \approx 0$, which explains why we observed 76% hallucination on $K=0.70$ bit tasks when forced to commit. The model simply lacked capacity to abstain.

## Two Kinds of Pressure

The phase transition clarified the relationship between $K$ and $r$, but understanding how these relate to actual hallucination requires distinguishing two separate pressures that contribute to it. The first is structural pressure, which comes from $K$ itself. When $K = 0.5000$ bits, at least 29% error is guaranteed when models must commit to answers. This rises to at least 53% when $K = 1.10$ bits, and the bound applies regardless of architecture, training procedure, or data quantity because the impossibility is mathematical.

The second is architectural pressure, which comes from requiring commitment when the model is uncertain. The 75-point gap (1% with abstention, 76% forced) on $K=0.70$ bit tasks quantifies this cleanly. The theoretical minimum was 40%, the model achieved 1% with abstention, but hit 76% when forced to choose.

These pressures are independent. TruthfulQA questions without structural contradictions still showed architectural effects: 20% forced versus 10% with abstention, though this wasn't statistically significant with only 10 questions tested. You can have structural pressure without architectural pressure—$K = 0.5000$ bits creates a 29% minimum whether or not the architecture forces commitment. When both apply, they compound.

Beyond these two pressures, training composition affects distance from the theoretical floor. At 10% defined, hallucination was 51.9% (±7.3%)—77% above the 29.3% minimum. At 70% defined, hallucination reached 98.3% (±2.1%)—more than three times the minimum. The sigmoid relationship quantifies exactly how this scales.

## What We Can and Cannot Control

Understanding these two pressures leads to practical questions about intervention. Some aspects of hallucination we can control, others we cannot.

Task structure is fixed. If different contexts demand incompatible behaviors, then $K > 0$ and some error is inevitable. The theoretical minimum of $1 - 2^{-K}$ sets a floor no training procedure can break.

Architecture determines whether you can approach that floor. Standard softmax provides $r \approx 0$ bits of witness capacity, leaving models well below $K$ for any contradictory task. Explicit witness heads providing $r \geq K$ enable approaching theoretical bounds through the relation $E + r \geq K$, where $E$ is error exponent in hypothesis testing and $r$ is witness rate. This establishes a trade-off: uncertainty in the task must be paid for either through error or through architectural capacity to express ignorance. You cannot achieve $E + r < K$—it's an impossibility, not merely difficult.

For neural networks, we observe this as a phase transition. When $r$ crosses $K$, error rate drops sharply from near 100% to near 0%. Tasks with contradiction $K$ can achieve arbitrarily low hallucination rates when the architecture provides witness capacity $r \geq K$, but cannot when $r < K$.

Training composition affects how far above the floor you land when architectural support is insufficient. The sigmoid relationship shows rapid increases early (10-30% defined), saturation later (70-90% defined). Small composition changes have large effects in the early phase but diminishing effects later.

What about confidence scores—can they help us identify when models are uncertain? It depends on what kind of uncertainty the model faces. When training explicitly contains contradictions (aleatoric uncertainty), networks learn appropriate 50% confidence. But when models encounter out-of-distribution inputs (epistemic uncertainty), they hallucinate confidently because training provided no signal about what "I don't know" looks like in those regions.

The finding that hallucinated answers have 59.5% confidence—between random guessing (20%) and learned certainty (98.85%)—suggests confidence scores reflect geometric position in feature space rather than epistemic uncertainty. If this is the case, post-hoc confidence calibration methods like temperature scaling or Platt scaling may face limits because they work with what the architecture already represents.

This raises a question: what about adding separate modules specifically trained to detect uncertainty? Our definedness head experiment suggests they don't solve the core problem. The definedness head achieved 100% accuracy on the three undefined training examples but only 4.3% on unseen undefined test cases, showing a 95.7% generalization gap across seeds. It memorized specific examples rather than learning the concept of "undefined."

## Generalization to Real Data

The principles above emerged from carefully controlled synthetic tasks. A natural question is whether they extend to real data with complex, high-dimensional structure. Testing on handwritten digits (8×8 images, 64 dimensions) with context-dependent labels shows that $K$ bounds worst-case error even on real visual data—and we can predict the exact error before training.

We created two labeling contexts: "parity" (odd=1, even=0) and "roundness" (0,6,8,9=1, others=0). These contexts contradict on 7 out of 10 digit classes, producing $K = 0.35$ bits and a theoretical minimum worst-case error of 21.5%.

The optimal frame-independent approximation must choose one label per digit. For contradictory digits, any choice satisfies one context and fails the other. If we satisfy Context A completely, we achieve 0% error in Context A but 70% error in Context B because all 7 contradictory digits have the wrong label there. The worst case across both contexts is max(0%, 70%) = 70%, and this was predicted before any training from the task structure alone.

Training CNNs and evaluating on both contexts confirmed it. Models trained exclusively on Context A labels achieved 1.9% error in Context A and 69.0% ± 0.1% error in Context B—matching the prediction exactly. Models trained exclusively on Context B labels achieved 68.5% ± 0.5% error in Context A and 2.2% error in Context B—close to the same 70% worst-case. Balanced training produced a different strategy, giving 36.9% error in Context A and 34.0% error in Context B for a 37.3% worst-case. The model learned to compromise between both contexts rather than fully satisfying either one.

![Worst-Case Error Analysis](experiment_10/results/worst_case_error.png)

These results reveal how training composition determines strategy. At the extremes—training exclusively on Context A or Context B—worst-case error hits the predicted 70%. The model masters one context but fails on all contradictory digits in the other. At the center with balanced training, worst-case drops to 37% as the model compromises, partially satisfying both contexts instead of fully satisfying either.

The right panel shows why: single-context training produces near-perfect performance in one direction (~2% error) and systematic failure in the other (~69% error), while balanced training splits the difference (~37% and ~34%). The symmetry confirms that 70% isn't a training artifact—it's the optimal frame-independent approximation predicted from task structure.

Like Experiment 4, this achieves the bound rather than merely exceeding it. The 70% error is the optimal frame-independent approximation, computed analytically before training and matched by models that learned one context consistently. Finding that $K$ determines exact error on 64-dimensional visual data confirms the principle isn't about low dimensionality or synthetic construction—task structure, not model architecture or training dynamics, sets these limits.

## What This Means

The experiments above establish measurable relationships between task structure, architecture, and hallucination. These relationships have implications for how we evaluate and deploy neural networks.

Standard accuracy metrics miss these failures. A network achieving 100% training accuracy might hallucinate on 96% of undefined test inputs, but aggregate statistics hide the problem because defined and undefined inputs get pooled together.

Confidence thresholds don't reliably filter unreliable predictions. The 59.5% confidence on fabricated answers sits between random guessing (20%) and learned certainty (98.85%), suggesting interpolation rather than abstention. Without explicit training on contradictions, confidence scores reflect geometric position in feature space rather than epistemic uncertainty.

This experiment tests the operational predictions of the contradiction theory for epistemic detection. Instead of benchmarking against heuristic OOD scores, we directly measure the witness–error tradeoff predicted by Theorem 7.4 and verify the existence of a sharp phase transition at the contradiction threshold ($K(P) = 0.792$ bits).

We construct contradiction tasks with two mutually exclusive behavioral demands: classify in-distribution samples correctly vs. abstain when predictions are epistemically invalid. These contexts are incompatible: a single predictor cannot satisfy both without access to additional witness information.

We evaluate four contradiction families (permutation, rotation, multi-label, adversarial) using the same base architecture and data. The only difference is the witness rate ($r$), which controls how much auxiliary epistemic information the model is allowed to represent.

From Theorem 7.4 (Witness–Error Tradeoff): $E + r \geq K(P)$ where $E$ is the optimal type-II error exponent, $r$ is witness rate, and $K(P)$ is the contradiction of the task. This implies a phase transition: if $r < K(P)$, optimal behavior is forced prediction; if $r \geq K(P)$, optimal behavior is selective abstention.

The transition occurs sharply between $r = 0.5$ and $r = 1.0$, exactly as predicted. For all contradiction families:

- **Permutation contradictions**: $r \geq 1.0$ achieves 54.5-63.5% abstention on contradictory inputs, 49.5-61.0% on consistent inputs, and 60.5-76.5% on OOD (SVHN)
  ![Permutation Contradictions](experiment_11/results/generalization_test_permutation.png)

- **Rotation contradictions**: $r \geq 1.0$ achieves 86.5-90.0% abstention on contradictory inputs, 19.0-23.5% on consistent inputs, and 84.5-93.5% on OOD
  ![Rotation Contradictions](experiment_11/results/generalization_test_rotation.png)

- **Multi-label contradictions**: Identical structure and transition as permutation (54.5-63.5% on contradictory, 49.5-61.0% on consistent, 60.5-76.5% on OOD)
  ![Multi-label Contradictions](experiment_11/results/generalization_test_multi_label.png)

- **Adversarial contradictions**: $r \geq 1.0$ achieves 100.0% abstention on contradictory inputs, 0.0% on consistent inputs, and 0.0-0.5% on OOD
  ![Adversarial Contradictions](experiment_11/results/generalization_test_adversarial.png)

![Epistemic Uncertainty Detection Overview](experiment_11/results/generalization_test.png)

This is a direct empirical confirmation of Theorem 7.4, demonstrating that witnesses trained only on in-distribution contradictions generalize to SVHN without retraining. The same phase transition appears across geometric contradictions (rotation), combinatorial contradictions (permutation, multi-label), and adversarial contradictions.

This pattern generalizes beyond specific architectures. The relationship between training composition and hallucination held across networks ranging from 64-unit feedforward classifiers to llama3.1:8b with billions of parameters. It held across different random seeds, different tasks, and different evaluation metrics, suggesting these are properties of how gradient descent allocates capacity between competing objectives rather than accidents of particular architectures.

Neural networks don't partition input space into "seen" and "unseen" regions. Instead, they create continuous representations where similar inputs produce similar outputs. When you train heavily on defined examples, those examples shape the entire feature space through interpolation, and undefined inputs land somewhere in that space and get mapped to nearby defined patterns.

The training objective is to maximize log probability of correct labels on training data. Cross-entropy loss provides maximum likelihood estimation, finding parameters that best explain what the model saw. But it doesn't provide parameters that recognize limitations. Unless you explicitly add a term for "abstain when uncertain" with sufficient weight to compete with classification loss, the optimization pushes toward confident predictions everywhere.

## References

- **Experiment 1**: [Neural Network Hallucination on Undefined Inputs](experiment_1/) — 96.1% hallucination on out-of-distribution inputs
- **Experiment 2**: [Architectural Separation with Definedness Head](experiment_2/) — 0.6 point improvement, 95.7% generalization gap
- **Experiment 3**: [Learning Under Contradictory Training Data](experiment_3/) — Aleatoric uncertainty, 50.2% confidence, 0.4% hallucination
- **Experiment 4**: [Invariance of Task Structure](experiment_4/) — K constant at 0.5 bits, hallucination varies 51.9% to 98.3%
- **Experiment 5**: [Non-Linearity of Hallucination Scaling](experiment_5/) — Sigmoid relationship, R²=0.944
- **Experiment 6**: [Hallucination Across Random Seeds](experiment_6/) — ρ=0.860 across 5 seeds, Cohen's d=4.04
- **Experiment 7**: [Structural Inevitability vs Architectural Commitment](experiment_7/) — 1% with abstention, 76% forced (75-point gap)
- **Experiment 8**: [TruthfulQA Benchmark](experiment_8/) — 20% forced vs 10% with abstention
- **Experiment 9**: [Quantifying Witness Capacity](experiment_9/) — Phase transition at r=K across 100 training runs
- **Experiment 10**: [Generalization to High-Dimensional Real Data](experiment_10/) — Predicted 70% worst-case error before training, achieved 69.0% ± 0.1%
- **Experiment 11**: [Phase Transition in Epistemic Detection via Witness Capacity](experiment_11/) — Tests operational predictions of contradiction theory for epistemic detection; verifies witness–error tradeoff ($E + r \geq K$) and sharp phase transition at contradiction threshold ($K = 0.792$ bits); demonstrates generalization from structural contradictions to epistemic uncertainty on OOD data

---

**Note on mathematical foundations**: The paper's Theorem 7.4 states $E + r \geq K$ where $E$ is error exponent (bits) in hypothesis testing, not error rate (0-1 scale). What we observe in neural networks is the implication of this law: a sharp phase transition in error rate when $r$ crosses $K$. See [Experiment 9's theory validation](experiment_9/THEORY_VALIDATION.md) for detailed discussion of how the information-theoretic conservation law relates to neural network behavior.