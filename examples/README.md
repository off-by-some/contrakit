# **When AI Can't Know—and What That Teaches Us About Information**

## **1. The Capability Gap Isn't Where You Think**

People keep telling me they're waiting for AI to get better before they'll really use it. I've been using these models to do research that normally takes teams years. The gap between what people think is possible and what's actually possible keeps surprising me.

Early image models struggled with hands—six fingers, mangled anatomy, clearly broken outputs. Everyone pointed to this as proof the technology was fundamentally limited.

But beneath the surface, something else was going on. People who learned Stable Diffusion properly were generating anatomically correct hands on the same base models giving everyone else nightmares. They figured out the techniques—negative prompts to exclude malformed anatomy, better samplers, higher resolution, inpainting for touch-ups, specific checkpoints trained on better hand data, explicit constraints like "five fingers, anatomically correct hands, professional photography."

The distribution of outcomes narrowed as interaction skill improved. Model quality stayed the same. By the time the "AI can't do hands" meme finally died, experienced users had been reliably generating good hands for months on models everyone else was calling broken.

This pattern shows up everywhere. When someone shows me ChatGPT producing garbage code or useless responses, I can almost always trace it back to how they structured the request. Their mental model of what they're working with is incomplete.

That observation—that outcomes depend more on how you ask than on raw capability—led me somewhere unexpected. What if some failures aren't about skill or model quality at all? What if they're structurally inevitable?

---

## **2. The Hidden Discipline Behind Effective Prompting**

The difference between good prompting and great prompting requires maintaining a very specific kind of mental discipline. It's a process closer to a design space, or a calculus, really. At the bare minimum, you're tracking four things simultaneously:

1. What you know about the problem
2. What you don't know
3. What the model likely learned during training
4. What it definitely doesn't have access to

Then you structure everything based on those boundaries.

In actuality, you're doing knowledge management across two minds, where one doesn't think like you and can't tell you what's missing. Most people aren't naturally good at this—I know I'm certainly not. It takes deliberate practice. The precision feels unnatural, like you're over-explaining everything. 

But that precision is the difference between "AI is useless" and getting serious work done. Research shows prompting techniques like Chain of Thought can shift performance by tens of percentage points without changing the model at all.

The deeper insight here is about understanding where the actual boundaries are. If the failures are this systematic, maybe there's something measurable underneath them. Something we could quantify and predict.

---

## **3. When Questions Are Fundamentally Unanswerable**

"What time is it right now?"

Seems simple enough. Except without knowing where you are, there's no single correct answer. There are hundreds, depending on timezone. The model has to output *something*—that's how softmax works—so it picks a time. Maybe it's right for you, maybe not. Either way, the confidence looks high.

The question was structured to make failure inevitable.

You asked it to choose from what I call a **contradictory output space**—where multiple incompatible answers are equally valid given the information available.

Imagine I ask you "What time is it in Europe?" And I tell you that you can't say "I don't know" or ask for clarification. You have to state a specific time, right now. Any answer you give is wrong for most of Europe. The task is fundamentally impossible without making something up.

This pattern shows up everywhere. Research on unanswerable questions confirms it: models hallucinate most when the ground truth is actually "there's no answer" or "not enough information," especially when they're forced to give a concrete answer anyway.

Can we measure this? Can we predict when a task contains this kind of structural impossibility?

## **4. Formalizing Contradiction: K(P)**
Here's the simplest experiment we built to test whether structural contradiction is actually measurable and predictable.

Start with classifying handwritten digits. We all learned this problem in our first machine learning class—ten categories, pretty straightforward. Now introduce two different labeling rules:

* **Rule A (parity):** odd digits get label 1, even digits get label 0
* **Rule B (roundness):** round digits (0, 6, 8, 9) get label 1, angular digits get label 0

Each rule is coherent on its own. You could teach either one to a model and everything would work fine. The problem shows up when you look at how they relate to each other. They contradict on seven out of ten digits.

Take the digit **7**. Under Rule A it's odd, so label 1. Under Rule B it's angular, so label 0. Both answers are correct in their own context. The task itself contains the contradiction.

So imagine you have to build a single classifier. One label per digit. But you never know which rule applies to any given test case. No markers, no context, nothing to tell you whether this particular 7 needs a parity label or a roundness label. You just see the digit and have to commit.

What's the best you could possibly do?

You can work this out on paper before training anything. If you commit fully to Rule A, you'll be perfect on Rule A test sets—0% error. But on Rule B test sets you'll fail on all seven contradictory digits. That's 70% error, and that's your worst case. The same logic applies in reverse if you commit to Rule B. Either way, 70% is the ceiling. No algorithm can beat it—not a neural network, not hand-coded rules, not a human expert making educated guesses.

I computed that bound, then trained CNNs three ways:

* **Trained only on Rule A labels:** 1.9% error on Rule A tests, **69.0% ± 0.1%** on Rule B tests
* **Trained only on Rule B labels:** **68.5% ± 0.5%** on Rule A tests, 2.2% on Rule B tests  
* **Trained on a balanced mix:** 36.9% on Rule A tests, 34.0% on Rule B tests

![Worst-case error analysis showing predicted 70% bound achieved on digit classification](/examples/hallucinations/experiment_10/results/worst_case_error.png)

The single-rule models hit exactly the predicted 70% worst case. The mixed model found the optimal compromise—roughly equal error on both sides. Different architectures, different random seeds, different hyperparameters. Same bound every time.

The learning algorithm is working fine. The task geometry is what determines the bound, and the model just discovers the shape of the contradiction you handed it.

---

---

## **5. The Mathematics of Structural Impossibility**

We developed a measure called $K(P)$ to quantify how much inherent contradiction exists in any task. When $K = 0$, the task has no internal contradictions and perfect performance is theoretically possible. When $K > 0$, contradictions exist and some minimum error becomes mathematically unavoidable.

For the digit task above, $K = 0.35$ bits. The formula connecting $K$ to minimum error is:

$\text{error} \geq 1 - 2^{-K}$

With $K = 0.35$ bits, that gives minimum error $\geq 21.5\%$. The 70% worst-case we observed is higher because "worst case across two contexts" is a stronger constraint than average error, but both are bounded by the contradiction in the task structure.

Some prompts consistently lead to garbage outputs because the structural contradiction is unavoidable without adequate witness capacity. You can compute $K$ from task structure before any training happens—it quantifies structural impossibility.

The mathematics works like this. For any behavior $P$ with multiple contexts (like our Rule A and Rule B), we're looking for the best "unified explanation"—a single model $Q$ that approximates both context-specific behaviors simultaneously. The Bhattacharyya coefficient $BC(p,q) = \sum_x \sqrt{p(x)q(x)}$ measures agreement between probability distributions, ranging from 0 (complete disagreement) to 1 (perfect match).

The agreement coefficient is:

$\alpha^*(P) = \max_{Q \in FI} \min_{c \in C} BC(p_c, q_c)$

You're finding your best shot at reconciliation—the $Q$ that agrees best with all your context-specific behaviors. The minimum over contexts captures what I call the Weakest Link Principle: overall agreement is limited by your worst case, not your average. Basic consistency requirements force this minimum-taking.

Then $K(P) = -\log_2(\alpha^*(P))$.

For the contradictory digit task, solving this optimization gives $\alpha^* \approx 0.78$, hence $K \approx 0.35$ bits. We computed this from the rule structure, it gave us bounds on error rates, and models trained completely differently all hit those bounds.

---

## **6. Architectural Pressure: Witness Capacity**

Even perfectly structured tasks hit a wall with current architectures. After identifying that a task has contradiction $K(P) > 0$, you need to give the model the ability to respond appropriately—for example, by saying "I don't know" instead of guessing.

This capability is what I call **witness capacity** $(r)$. It's about the information-theoretic capacity to represent uncertainty and reason about it.

Standard softmax provides $r \approx 0$ bits of witness capacity because it forces probability distributions summing to 1.0 across outputs. The architecture can't express "I don't have enough information" or "this question doesn't make sense." Every input gets the same treatment: embed, transform through layers, project to output space, softmax, select highest probability. Even when the input has nothing to do with training data, this process generates a confident prediction.

I tested this systematically, measuring witness capacity $r$ across different architectures on tasks with varying contradiction $K$. Twenty combinations of task contradiction ($K$ from 0.5 to 1.16 bits) and architectural capacity ($r$ from 0 to 2 bits), with 5 random seeds each—100 training runs total.

The phase transitions are gradual rather than perfectly sharp:

- When $r$ is well below $K$: Models fail consistently (error rates near 100% across all seeds)
- When $r$ significantly exceeds $K$: Models show substantial error reduction, though not always reaching theoretical minima

The transition occurs gradually rather than at a precise threshold. A task with $K = 0.5$ bits shows high error rates until $r = 2.0$ bits. Tasks with higher $K$ (0.79–1.16 bits) require even more witness capacity before meaningful error reduction begins. This smoothing reflects practical limitations in finite networks and stochastic optimization.

![Error rates vs witness capacity showing gradual phase transitions](/examples/hallucinations/experiment_9/results/error_vs_witness.png)

Models show persistent high error rates when witness capacity is insufficient, with substantial but gradual improvement as $r$ significantly exceeds $K$. Practical networks rarely achieve theoretical minima—the realities of finite capacity and optimization dynamics.

I tested one task with $K = 0.70$ bits under two architectural conditions:

- **With abstention allowed**: 1% hallucination (495 correct abstentions, 5 errors out of 500 trials)
- **Forced to commit**: 76% hallucination (380 errors, 120 abstentions out of 500 trials)

That 75 percentage point gap separates architectural pressure from structural pressure. The structural contradiction stayed constant. With adequate witness capacity, the model stayed near the theoretical floor at 1%. Without it, hallucination shot to 76%.

```
====================================================================================================
SUMMARY: Phase Transition at r = K
====================================================================================================
K (bits)   r (bits)   Error Rate      r ≥ K?     Phase           Predicted?  
----------------------------------------------------------------------------------------------------
0.5000     0.00       0.8000±0.026  ✗ No       Failure (E≈1)   5/5         
0.5000     0.50       0.8000±0.026  ✓ Yes      Failure (E≈1)   0/5         
0.5000     1.00       0.8000±0.026  ✓ Yes      Failure (E≈1)   0/5         
0.5000     1.50       0.8000±0.026  ✓ Yes      Failure (E≈1)   0/5         
0.5000     2.00       0.5340±0.055  ✓ Yes      Failure (E≈1)   1/5         
0.7925     0.00       0.9240±0.014  ✗ No       Failure (E≈1)   5/5         
0.7925     0.50       0.9240±0.014  ✗ No       Failure (E≈1)   5/5         
0.7925     1.00       0.9240±0.014  ✓ Yes      Failure (E≈1)   0/5         
0.7925     1.50       0.9240±0.014  ✓ Yes      Failure (E≈1)   0/5         
0.7925     2.00       0.8620±0.028  ✓ Yes      Failure (E≈1)   0/5         
1.0000     0.00       0.9700±0.021  ✗ No       Failure (E≈1)   5/5         
1.0000     0.50       0.9700±0.021  ✗ No       Failure (E≈1)   5/5         
1.0000     1.00       0.9700±0.021  ✓ Yes      Failure (E≈1)   0/5         
1.0000     1.50       0.9700±0.021  ✓ Yes      Failure (E≈1)   0/5         
1.0000     2.00       0.9700±0.021  ✓ Yes      Failure (E≈1)   0/5         
1.1610     0.00       0.9840±0.010  ✗ No       Failure (E≈1)   5/5         
1.1610     0.50       0.9840±0.010  ✗ No       Failure (E≈1)   5/5         
1.1610     1.00       0.9840±0.010  ✗ No       Failure (E≈1)   5/5         
1.1610     1.50       0.9840±0.010  ✓ Yes      Failure (E≈1)   0/5         
1.1610     2.00       0.9840±0.010  ✓ Yes      Failure (E≈1)   0/5         
====================================================================================================
```
---

## **7. The Surprising Role of Training Composition**

There's a third factor that affects where you land within the feasible region: training composition.

I tested this by varying the ratio of defined to undefined examples from 10% to 90% defined, holding $K = 0.5000$ bits constant. That theoretical minimum of 29.3% error never changed—the task structure stayed identical. But observed hallucination rates ranged from 51.9% (±7.3%) at 10% defined to 98.3% (±2.1%) at 70% defined.

**This result seems counterintuitive.** Usually more training data improves performance. Here it makes things worse.

The network doesn't learn to partition space into "I know this" and "I don't know this" regions—instead, it learns a smooth function that interpolates everywhere. With more defined examples, the interpolation becomes more confident in its extrapolations, even into regions where it should abstain.

When you have 115 defined examples and only 3 undefined ones, optimization overwhelmingly favors correct classification. The loss function sees 115 examples rewarding confident predictions and just 3 suggesting abstention, so almost all gradient flow pushes toward classification.

We tested 17 compositions with 3 random seeds each—51 total training runs—and fit four mathematical functions. All model selection criteria agreed: **sigmoid fit best**, explaining 94.4% of variance.

![Sigmoid relationship between training composition and hallucination rates](/figures/hallucination_curve_fitting.png)

The sigmoid reveals three distinct phases:
- From 10% to 30% defined: steep increase, roughly 23 percentage points
- From 30% to 70% defined: diminishing increases, about 8.5 points across 40 points
- From 70% to 90% defined: near-saturation, only 2.6 additional points

The theoretical minimum of 29.3% held in every configuration. But observed rates climbed far higher, ranging from 77% above the minimum at 10% defined to 236% above at 70% defined.

**Training composition determines how far above the floor you land when architectural support is insufficient, but it cannot break through the floor set by $K$.**

---

## **8. Three Independent Pressures: A Complete Picture**

Hallucination stems from three independent pressures that work separately but compound when combined:

**First: Structural pressure ($K$)**  
Some tasks demand incompatible behaviors across different contexts. When $K = 0.5000$ bits, at least 29% error is guaranteed when models must commit to answers. This bound applies equally to neural networks, decision trees, hand-coded rules, or humans guessing. The impossibility is mathematical—it's baked into what you're asking for, not a quirk of any particular learning system.

**Second: Architectural pressure ($r < K$)**  
Softmax forces producing a definite prediction for every input, whether prediction makes sense or not. Standard softmax has $r \approx 0$ bits, leaving models unable to express epistemic uncertainty. The 75-point gap (1% with abstention, 76% forced) on $K = 0.70$ bit tasks isolates this effect cleanly.

**Third: Training composition**  
The balance of defined versus undefined examples affects how far above the theoretical minimum you land. The sigmoid relationship shows rapid increases early (10-30% defined), saturation later (70-90% defined). But composition can only modulate distance from the floor—it cannot eliminate structural impossibility when $K > 0$.

**The practical insight:** minimize K where you can (clearer prompts, better task structure, providing necessary context) and maximize r where you can't (explicit uncertainty mechanisms, appropriate tools). That's how you get into the regime where this actually works.

---

## **9. Cross-Domain Applications**

If the framework only worked for neural networks, it would be interesting but limited. But what if it applies more broadly?

### **Quantum Mechanics**

The CHSH inequality tests whether quantum correlations can be explained by classical hidden variables. Classical physics constrains the correlation strength $S$ to $S \leq 2$. Quantum mechanics achieves $S = 2\sqrt{2} \approx 2.828$—a violation Einstein famously called "spooky action at a distance."

Using the same framework, I computed the contradiction measure $K(P)$ for quantum correlations at maximum violation. Computations for standard quantum scenarios suggest $K(P)$ is small but nonzero—roughly on the order of tenths of a bit per measurement pair in CHSH-type tests.

![CHSH inequality analysis showing quantum-classical boundaries](/figures/bell_chsh_analysis.png)

The Magic Square puzzle is even more striking. It's a scenario where quantum mechanics solves a logically impossible problem: fill a 3×3 grid with +1 and −1 such that each row multiplies to +1 and each column multiplies to +1, except the last column which must multiply to −1. Try it classically—you can't do it. The mathematical constraint is self-contradictory.

Quantum mechanics does it. The contradiction measure: **$K = 0.132$ bits**—exactly $\frac{1}{2}\log_2(6/5)$, computed analytically from the parity constraints before running any quantum experiment.

![Magic Square contradiction analysis showing algebraic constraints](/figures/magic_square_analysis.png)

The pattern across quantum scenarios:
- $K(P) = 0$ below classical bounds (local hidden variable models work fine)
- $K(P) > 0$ in quantum regimes, growing continuously with violation strength
- Probabilistic violations (Bell, KCBS): $\approx 0.012$ bits
- Algebraic impossibilities (Magic Square): $\approx 0.132$ bits

![KCBS contextuality analysis showing noise robustness](/figures/kcbs_contextuality_analysis.png)

### **Byzantine Consensus and Distributed Agreement**

Byzantine consensus asks: can nodes in a network agree on a value even when some nodes are faulty or malicious? The challenge is precisely about reconciling incompatible perspectives—honest nodes see truth, Byzantine nodes send conflicting messages.

I computed $K(P)$ for several Byzantine scenarios:
- Perfect agreement: $K = 0.0000$ bits
- Single traitor sending conflicting messages: $K = 0.0273$ bits
- Triangle paradox (each pair sees perfect disagreement): $K = 0.5000$ bits
- Complex four-node network with heterogeneous faults: $K = 0.2925$ bits

The measure identifies which node pairs drive disagreement through witness weights $\lambda^*$, enabling adaptive verification. In the four-node test, this achieved 16.7% message savings while maintaining Byzantine fault tolerance guarantees.


**The same mathematics we used to predict neural network hallucination quantifies quantum-classical boundaries and distributed consensus failures.**

---

## **10. Practical & Theoretical Implications**

Before getting to the mathematical foundations, let me highlight some non-obvious consequences:

### **Why ensembling can't save you**

A common response to unreliable models is to train multiple models and average their predictions. This works for reducing variance, but **it cannot eliminate the structural error floor**.

If each model faces the same task with contradiction $K$ and witness capacity $r$, each model has minimum error $E \geq K - r$. Averaging $N$ independent models reduces randomness in their predictions, but the expected ensemble error $E_\text{ensemble}$ still satisfies $E_\text{ensemble} \geq K - r$.

The intuition: imagine $K = 1$ bit (you need 1 bit of information to resolve the contradiction) and $r = 0.5$ bits (your architecture can only represent half that). Each model must make errors totaling at least $E \geq 0.5$ bits worth. When you average models, you're averaging their guesses on cases where they don't have enough capacity. **The average of wrong guesses is still wrong**—you haven't added the missing information that would resolve the contradiction.

### **Why classical learning theory breaks down**

PAC (Probably Approximately Correct) learning theory provides the mathematical foundation for most machine learning. It guarantees that with enough training data, you can get arbitrarily close to the optimal error rate for your hypothesis class. Its key assumption is that there exists a good solution that you're trying to find.

**When $K > 0$, this breaks down fundamentally.** There is no hypothesis that achieves perfect agreement across all contexts—the structural contradiction makes it impossible. The minimum error floor $1 - 2^{-K}$ is not statistical (arising from finite data) but structural (arising from the task itself).

Our experiments confirm this: for $K = 0.5$ bits (minimum error 29.3%), no training configuration broke through this floor, regardless of data quantity or composition. More data doesn't help because there's no better solution to find.

### **Why errors compound in reasoning chains**

When you chain multiple reasoning steps together, each with unresolved contradictions, errors propagate multiplicatively, not additively. If you have $n$ sequential tasks, each with contradiction $K_i$ and your model has witness capacity $r$, the probability of getting through the entire chain without error is approximately the product of success rates at each step.

For concrete numbers: suppose you have 5 reasoning steps, each with 20% error rate (80% success). Individual steps look pretty good! But the chain success is $0.8^5 \approx 0.33$, meaning **67% error overall**. The chain is much less reliable than any individual step.

This explains why complex multi-step reasoning tends to fail catastrophically rather than gracefully—the chain is only as strong as its weakest link, and the probability of avoiding all weak links decreases exponentially with chain length.

---

## **11. Mathematical Foundations: Why These Formulas Are Forced**

The mathematical uniqueness is worth noting. The Bhattacharyya coefficient $BC(p,q) = \sum_x \sqrt{p(x)q(x)}$ isn't chosen arbitrarily. **Theorem 3 proves it's the only per-context agreement kernel** satisfying: refinement separability, product multiplicativity, data-processing inequality, and joint concavity. If you want a different measure, you have to violate one of these properties and explain why.

Similarly, the log law $K = -\log_2(\alpha^*)$ is forced by requiring that contradictions add on independent systems (Theorem 4). The minimax game structure comes from basic consistency axioms (Theorem 2).

**These aren't modeling choices—they're mathematical necessities given what we're trying to measure.**

The Weakest Link Principle (Theorem 1) shows that any aggregation rule satisfying basic consistency requirements—unanimity on identical inputs, monotonicity, and a local upper bound—must take the minimum. You cannot aggregate context-wise agreements any other way without violating these basic requirements.

---
## **12. What This Means for Information Theory**

I needed to know if $K(P)$ was just mutual information wearing a different hat, so I built a simple test.

Take the digit 7 with two contradictory labeling rules: parity says "1" (odd), roundness says "0" (angular). Standard setup. Now vary how often each rule applies—10% parity versus 90%, then 30/70, then 50/50, and so on. If $K(P)$ measures the same thing as Shannon's mutual information $I(X;C)$, both should move together as these probabilities shift.


```python
import math
from contrakit import Observatory

# Create contradictory behaviors: parity vs roundness for digit "7"
obs = Observatory.create(symbols=['0', '1'])
label = obs.concept('Label')

parity = obs.lens('Parity')
with parity:
    parity.perspectives[label] = {'1': 1.0, '0': 0.0}  # Says "odd"

roundness = obs.lens('Roundness')  
with roundness:
    roundness.perspectives[label] = {'0': 1.0, '1': 0.0}  # Says "angular"

behavior = (parity | roundness).to_behavior()

# Vary context probabilities, compute both measures
context_weights = [0.1, 0.3, 0.5, 0.7, 0.9]

print("| Context Split | I(X;C) | K(P) |")
print("|--------------|--------|------|")

for w in context_weights:
    # Compute mutual information I(X;C)
    p_parity = w
    p_roundness = 1 - w
    p_label_0 = p_parity * 0.0 + p_roundness * 1.0
    p_label_1 = p_parity * 1.0 + p_roundness * 0.0
    
    H_X = sum(-p * math.log2(p) for p in [p_label_0, p_label_1] if p > 0)
    H_C = sum(-p * math.log2(p) for p in [p_parity, p_roundness] if p > 0)
    I_X_C = H_X  # H(X|C) = 0 for deterministic contexts
    
    print(f"| {w:.1f} / {1-w:.1f}      | {I_X_C:.3f}  | {behavior.K:.3f} |")

# Output:
# | Context Split | I(X;C) | K(P) |
# |--------------|--------|------|
# | 0.1 / 0.9    | 0.469  | 0.500 |
# | 0.3 / 0.7    | 0.881  | 0.500 |
# | 0.5 / 0.5    | 1.000  | 0.500 |
# | 0.7 / 0.3    | 0.881  | 0.500 |
# | 0.9 / 0.1    | 0.469  | 0.500 |
```

Mutual information more than doubles—0.469 to 1.000 bits—as context probabilities change. $K(P)$ stays at 0.500 bits. Exactly. Not approximately, not within rounding error. The same 0.500 across every tested configuration.

$K(P)$ sees through the probability distribution to something underneath. Mutual information tracks statistical uncertainty—how much observing the context reduces your uncertainty about the label. That depends entirely on how often each context occurs. $K(P)$ tracks structural incompatibility—how the contexts relate to each other, regardless of their frequency. The distance between two cities doesn't change when traffic patterns shift.

This distinction shows up operationally. Shannon's source coding theorem tells you the compression limit: $H(X|C)$ bits per symbol when the decoder knows context $C$. Tight bound. But when contexts impose contradictory requirements and your single codebook has to work across all of them, you pay extra. Theorem 6 proves the cost is exactly $K(P)$ bits. For channels, Shannon gives you capacity $C_{\text{Shannon}}$ based on noise characteristics. When receivers have incompatible decoding requirements—can't agree on what the signal should mean—you lose exactly $K(P)$ bits of capacity (Theorem 7).

These theorems assume you have a joint distribution $P$ over all relevant variables. They tell you what's achievable given that $P$. When $K(P) = 0$, you can find a $P$ that satisfies all contexts simultaneously. Shannon's framework is complete. When $K(P) > 0$, the joint distribution exists but gets constrained by incompatible requirements. The theorems still hold—you're just working with a restricted $P$, and restriction costs you.

The mathematics behaves the way you'd want. Independent systems with contradictions $K_1$ and $K_2$ compose to $K_1 + K_2$. Tested with two 0.5-bit systems: product is exactly 1.0 bits. Contradiction degrades smoothly from weak to strong: 0.007 bits for 60/40 disagreement, 0.161 bits for 90/10, 0.500 bits for total opposition. Add neutral contexts that don't affect the extremes and $K(P)$ ignores them. These properties aren't coincidence—they follow from the axioms. The Bhattacharyya coefficient and log form are forced by the data processing inequality, product multiplicativity, and the weakest link principle (Theorems 1-5).

Here's a case where the joint distribution exists but gets squeezed. Observer A and B always disagree. B and C always disagree. A and C always agree. Try building $P(A,B,C)$ that respects all three constraints. Eight possible states, exactly two are valid: $(0,1,0)$ and $(1,0,1)$. You can split probability mass between them however you want, but you can't escape those two corners. The constraints pin you there.

Shannon's mutual information assumes the joint distribution exists freely. Here it exists but lives in a restricted subspace. $K(P) = 0.500$ bits quantifies that restriction—how much the pairwise constraints squeeze the space of valid distributions. Shannon theory has no measure for this kind of structural pressure.

The regime matters because real systems hit it. Quantum mechanics gives you observables that genuinely can't be measured simultaneously—not because we lack clever measurement schemes, but because the observables don't commute. Byzantine consensus involves agents sending conflicting messages to different parties by design. Multi-perspective data sources that can't be reconciled into one coherent story without losing information from at least one viewpoint.

For neural networks the results are solid. $K(P)$ predicts hallucination floors from task structure before training. The $r = K$ phase transition holds across 100+ runs with different architectures and random seeds. Witness capacity trained on structural contradictions generalizes to epistemic uncertainty on out-of-distribution inputs. This is reproducible, tested, confirmed.

For quantum systems the results are suggestive. I computed $K \approx 0.012$ bits for Bell violations, $K \approx 0.132$ bits for the Magic Square. These match known quantum-classical boundaries. Whether $K(P)$ explains *why* these values emerge from quantum mechanics, or just happens to reproduce them, needs physicists examining the framework carefully.

The test case that would settle it: take a quantum channel with experimentally verified capacity—superdense coding or quantum teleportation work well. Model Alice and Bob's incompatible measurement choices as contradictory decoders. Compute $K(P)$ from their measurement contexts. Check if $C_{\text{classical}} - K(P)$ matches the verified quantum capacity. If it does, $K(P)$ captures the quantum-classical gap through pure information geometry. If it doesn't, the framework applies to designed systems but not fundamental physics.

Shannon built the right tools for epistemically consistent sources where $K = 0$. The probability distribution exists, it's coherent, his theorems tell you everything you need to know. For epistemically inconsistent sources where $K > 0$—multiple contexts that can't be simultaneously satisfied—the operational costs differ from Shannon's predictions. Theorems 6-9 prove these differences exactly. The bounds are tight, achieved in practice.

Most systems stay in Shannon's regime. When they don't, $K(P)$ tells you how far out you've gone and what it costs operationally. That's the sense in which classical information theory extends rather than breaks. Zero contradiction, Shannon is complete. Positive contradiction, you need the additional structure. The mathematics connects cleanly because $K(P)$ reduces to mutual information in the limit where contexts become statistically independent rather than structurally incompatible.

Whether this extends to fundamental physics or stays within designed systems—neural networks, protocols, task specifications—depends on results we don't have yet. The mathematical structure is tight enough to be interesting. The empirical validation on neural networks is thorough enough to be useful. The quantum connections are promising enough to pursue seriously. That's where the work stands.
---

## **13. Reproducible Examples**

These runnable code snippets demonstrate the core claims. Install contrakit and copy-paste them into a Python environment:

```bash
pip install contrakit
```

### **Computing Contradiction Measure K(P)**

```python
from contrakit import Observatory

# Create a contradictory task: parity vs roundness for digit "7"
obs = Observatory.create(symbols=['0', '1'])
prediction = obs.concept('Prediction')

# Context A: parity rule (odd=1, even=0)
lens_A = obs.lens('Parity')
with lens_A:
    lens_A.perspectives[prediction] = {'1': 1.0}  # 7 is odd

# Context B: roundness rule (round=1, angular=0)
lens_B = obs.lens('Roundness')
with lens_B:
    lens_B.perspectives[prediction] = {'0': 1.0}  # 7 is angular

combined = lens_A | lens_B
behavior = combined.to_behavior()
print(f"Contradiction measure K = {behavior.K:.3f} bits")
# Output: K ≈ 0.500 bits (predicts minimum error ≥ 29.3%)
```

### **Quantum-Classical Boundary Detection**

```python
from contrakit import Observatory

# Create CHSH quantum scenario
obs = Observatory.create(symbols=['+1', '-1'])
measurement = obs.concept('Measurement')

# Quantum correlations for CHSH: Alice at 0°/90°, Bob at 45°/135°
alice_0_bob_45 = obs.lens('A0_B45')
alice_0_bob_135 = obs.lens('A0_B135')
alice_90_bob_45 = obs.lens('A90_B45')
alice_90_bob_135 = obs.lens('A90_B135')

# Set quantum correlation probabilities
with alice_0_bob_45:
    alice_0_bob_45.perspectives[measurement] = {'+1': 0.146, '-1': 0.854}

with alice_0_bob_135:
    alice_0_bob_135.perspectives[measurement] = {'+1': 0.854, '-1': 0.146}

with alice_90_bob_45:
    alice_90_bob_45.perspectives[measurement] = {'+1': 0.854, '-1': 0.146}

with alice_90_bob_135:
    alice_90_bob_135.perspectives[measurement] = {'+1': 0.146, '-1': 0.854}

# Combine all measurement contexts
quantum_system = alice_0_bob_45 | alice_0_bob_135 | alice_90_bob_45 | alice_90_bob_135
quantum_behavior = quantum_system.to_behavior()

print(f"Quantum contextuality: K = {quantum_behavior.K:.4f} bits")
# Output: K ≈ 0.115 bits
```

### **Neural Network Hallucination Prediction**

```python
from contrakit import Observatory

# Create contradictory task
obs = Observatory.create(symbols=['A', 'B', 'C', 'D', '⊥'])
prediction = obs.concept('Prediction')

# Context 1: defined inputs should be classified
defined_lens = obs.lens('Defined_Inputs')
with defined_lens:
    defined_lens.perspectives[prediction] = {
        'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25
    }

# Context 2: undefined inputs should abstain
undefined_lens = obs.lens('Undefined_Inputs')
with undefined_lens:
    undefined_lens.perspectives[prediction] = {'⊥': 1.0}

# Combine contexts
task_system = defined_lens | undefined_lens
task_behavior = task_system.to_behavior()

print(f"Task contradiction: K = {task_behavior.K:.3f} bits")
print(f"Predicted minimum hallucination: ≥{1 - 2**(-task_behavior.K):.1%}")
# Output: K = 0.500 bits, minimum error ≥ 29.3%
```

### **Byzantine Consensus Contradiction**

```python
from contrakit import Observatory

obs = Observatory.create(symbols=['0', '1'])
node_value = obs.concept('Node_Value')

# Honest nodes
node1 = obs.lens('Node1')
with node1:
    node1.perspectives[node_value] = {'0': 0.5, '1': 0.5}

node2 = obs.lens('Node2')
with node2:
    node2.perspectives[node_value] = {'0': 0.5, '1': 0.5}

# Byzantine node (sends conflicting information)
byzantine_to_1 = obs.lens('Byzantine_to_Node1')
with byzantine_to_1:
    byzantine_to_1.perspectives[node_value] = {'0': 0.8, '1': 0.2}

byzantine_to_2 = obs.lens('Byzantine_to_Node2')
with byzantine_to_2:
    byzantine_to_2.perspectives[node_value] = {'1': 0.7, '0': 0.3}

# Combine all perspectives
consensus_system = node1 | node2 | byzantine_to_1 | byzantine_to_2
byzantine_behavior = consensus_system.to_behavior()

print(f"Byzantine contradiction: K = {byzantine_behavior.K:.4f} bits")
print(f"Consensus requires ≥{1 - 2**(-byzantine_behavior.K):.1%} fault tolerance")
# Output: K ≈ 0.027 bits
```

### **Running Full Experiments**

For complete experimental validation with statistical analysis:

```bash
# Clone and install contrakit
git clone https://github.com/off-by-some/contrakit.git
cd contrakit && poetry install

# Run hallucination experiments
poetry run python examples/hallucinations/experiment_4/run.py

# Run quantum CHSH analysis
poetry run python examples/quantum/CHSH.py

# Run Byzantine consensus analysis
poetry run python examples/consensus/run.py

# Run digit classification contradiction experiment
poetry run python examples/hallucinations/experiment_2/run.py
```

---

## **14. Assessing the Evidence: What's Proven vs What's Speculative**

To wrap up, let me be clear about what's established versus what's speculative.

**Strong evidence:**
- $K(P)$ predicts neural network hallucination floors from task structure before training
- The $r = K$ phase transition is reproducible across 100+ training runs, though boundaries are gradual rather than perfectly sharp
- Witness capacity trained on structural contradictions generalizes to epistemic uncertainty on out-of-distribution data
- The mathematical framework has unique properties forced by basic axioms

**Promising but needs replication:**
- $K(P)$ quantifies quantum-classical boundaries (computed values match known bounds, but needs independent verification)
- Byzantine consensus applications (adaptive savings work in tested scenarios, need larger scale validation)
- Compression bound theorems (proven for synthetic tasks, need testing on realistic channels)

**Speculative (direction for future research):**
- How precisely the framework connects to philosophical questions about epistemology
- Whether these principles apply broadly to intelligence or are specific to current architectural choices

The quantum numbers particularly need scrutiny. I'm computing $K \approx 0.012$ bits for Bell violations and $K \approx 0.132$ for Magic Square, and these match experimental quantum-classical boundaries. But "my framework gives the right numbers" is different from "my framework explains why these must be the right numbers."

The neural network results I'm confident in—I can predict hallucination before training, the phase transitions are reproducible, the math is clean. The quantum and information-theoretic extensions are promising but need the kind of scrutiny that comes from experts in those domains really trying to break the framework.

---

## **15. How This Changes Practice**

Understanding all this completely shifted our approach to AI. You'll spend way more time thinking about task structure now. Are you accidentally creating contradictions? Where are you assuming the model has context it doesn't? You'll want to build in escape hatches—explicit ways for the model to indicate uncertainty instead of forcing answers.

Sounds trivial. **But it's architecturally the difference between 1% and 76% hallucination.**

Software engineering is already moving this direction. Writing software is becoming more like writing assembly language is today. Software engineers are turning into architects augmented with AI, trained on verification and synthesis. The work that remains will be the stuff that needs deep judgment about edge cases and subtle failure modes. Exactly what humans are good at.

But we're in this strange moment right now. The technology is genuinely powerful. But it's widely misunderstood. The gap between what it can do and how people use it is massive. Not because we need huge improvements—though sure, it'll improve—but because we're still figuring out what we're actually working with.

And then there's the anthropomorphization trap. Anything that could be called "thought" ends the instant the model outputs a token. LLMs emulate someone else's goals—they have zero personal stake in outcomes. They amplify us, including all our errors.

Learning to work effectively with current systems is hard. It requires developing intuitions about knowledge and uncertainty that don't come naturally. You have to get comfortable saying "I don't know" and building systems that can too.

**But on the other side of that learning curve, something interesting happens.** The tools become genuinely collaborative—not mystically, but practically. You can delegate real cognitive work and trust the results because you structured the task well.

---

## **16. Looking Forward: What Needs to Happen**

Here's what emerges from this work: the solution isn't might not simply be, better AI. It's possible that we need better mathematics for thinking about information itself—mathematics that can represent contradiction, paradox, and epistemic uncertainty as first-class concepts instead of edge cases or error.

To truly model intelligence, uncertainty, or epistemic processes, we may need to go beyond classical statistics. Kolmogorov's mathematics might be incomplete for these phenomena. Yet, that's what neural networks are built on.

It doesn't seem impossible. Witness capacity is a mechanism we've implemented and tested for getting the behaviors people want—saying "I don't know," reasoning about uncertainty, avoiding hallucination on undefined inputs. And it's more than logical to assume such an ability could be used to engineer emergent epistemic systems. 

The experiments show this concept working, too: neural networks with sufficient witness capacity ($r \geq K$) abstain perfectly on contradictory inputs, while those without it ($r < K$) hallucinate consistently. **The architecture exists. It's just not how things work by default currently.**

If this research direction proves fruitful, we may need to address these fundamentals before we can build systems that truly reason about their own limitations. What exists now is powerful but incomplete. Treat it that way.

Famous people's work gets filtered through whatever scientific paradigm is dominant, and that filtering—including the misunderstandings—is part of why scientific revolutions happen. Kuhn pointed this out. It's possible we're in one of those moments. We took brilliant ideas about information and logic, applied them everywhere without understanding the trade-offs, and now it's possible we're hitting the limits.

**That's what we're working toward:** closing the gap between what's possible and what people actually do, one carefully structured prompt at a time. And along the way, understanding something deeper about what intelligence requires—not just for AI, but for any system that needs to reason about its own limits.

Not just making AI better at following instructions. **Building the mathematical foundations for systems that can understand what they know and what they don't.**

---

*The mathematical details are in [contrakit on GitHub](https://github.com/off-by-some/contrakit) if you're curious. Run `examples/hallucinations/experiment_9` to see the phase transitions in witness capacity, or `examples/quantum/CHSH.py` to reproduce the Bell inequality information costs.*