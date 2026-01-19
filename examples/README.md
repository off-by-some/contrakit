# **When AI Can't Know—and What That Teaches Us About Information**

## **1. The Capability Gap Isn't Where You Think**

People keep telling me they're waiting for AI to get better before they'll really use it. Meanwhile, I've been using it to do research that normally takes teams years, and I'm constantly shocked by what's already possible.

The gap isn't where people think it is.

Early image models struggled with hands—six fingers, mangled anatomy, clearly broken outputs. Everyone pointed to this as proof that "AI isn't ready yet."

But something interesting was happening beneath the surface. People who learned Stable Diffusion properly were generating anatomically correct hands on the same base models that were giving everyone else nightmares. They figured out the techniques—negative prompts to exclude malformed anatomy, better samplers, higher resolution, inpainting for touch-ups, specific checkpoints trained on better hand data, and explicit constraints like "five fingers, anatomically correct hands, professional photography."

The distribution of outcomes narrowed as interaction skill improved, not as models got better. By the time the "AI can't do hands" meme finally died, experienced users had been reliably generating good hands for months on models everyone else was calling broken.

This pattern isn't specific to images. When someone shows me ChatGPT producing garbage code or useless responses, I can almost always trace it to how they structured the request. The model isn't failing—the user's mental model of what they're working with is incomplete.

**This observation—that outcomes depend more on how you ask than on model capability—led me to a deeper question: what if some failures aren't about skill or model quality at all? What if they're structurally inevitable?**

---

## **2. The Hidden Discipline Behind Effective Prompting**

Good prompting isn't about magic phrases or prompt templates. It requires maintaining a very specific kind of mental discipline.

You're tracking four things simultaneously:
1. What you know about the problem
2. What you don't know
3. What the model likely learned during training
4. What it definitely doesn't have access to

Then you structure everything based on those boundaries.

You're doing knowledge management across two minds, where one doesn't think like you and can't tell you what it's missing.

Most people aren't naturally good at this—I'm certainly not. It takes deliberate practice. The precision feels unnatural, like you're over-explaining everything. But that precision is the difference between "AI is useless" and getting serious work done. Research shows prompting techniques like Chain of Thought can shift performance by tens of percentage points without changing the model at all.

But it goes deeper than techniques. It depends on understanding what you're trying to accomplish and **where the actual boundaries are**.

Which raises a question: if the failures are this systematic, is there something measurable underneath them? Something we could quantify and predict?

---

## **3. When Questions Are Fundamentally Unanswerable**

"What time is it right now?"

Seems simple, right? Except without knowing where you are, there's no single correct answer. There are hundreds, depending on timezone. The model has to output *something*—that's how softmax works—so it picks a time. Maybe it's right for you, maybe not. Either way, the confidence looks high.

**The model didn't fail. The question was structured to make failure inevitable.**

You asked it to choose from what I call a **contradictory output space**—where multiple incompatible answers are equally valid given the information available.

Imagine I ask you "What time is it in Europe?" And I tell you that you can't say "I don't know" or ask for clarification. You have to state a specific time, right now. Any answer you give is wrong for most of Europe. The task is fundamentally impossible without making something up.

This pattern shows up everywhere. Research on unanswerable questions confirms it: models hallucinate most when the ground truth is actually "there's no answer" or "not enough information," especially when they're forced to give a concrete answer anyway.

**This led me to wonder: can we measure this? Can we predict when a task contains this kind of structural impossibility?**

---

## **4. Formalizing Contradiction: K(P)**

Here's what we constructed to test whether structural contradiction is measurable and predictable.

Take a simple task: classifying handwritten digits. But now imagine two different labeling rules that are both correct in their own contexts:

- **Rule A (parity)**: odd digits get label 1, even digits get label 0
- **Rule B (roundness)**: round digits (0,6,8,9) get label 1, angular digits get label 0

These rules contradict on 7 out of 10 digit classes. The digit "7" should be labeled 1 under Rule A (it's odd) but 0 under Rule B (it's angular). Both labelings are correct in their respective contexts. The contradiction is structural—it exists in what we're asking for, not in the data or the model.

The question is: if you're forced to pick one label per digit and can't tell which rule applies to a given test case, what's the best you can possibly do?

The answer is mathematical. If you satisfy Rule A completely, you get 0% error on Rule A test sets but 70% error on Rule B test sets (all 7 contradictory digits are labeled wrong for that context). The worst case across both contexts is 70%. This is the optimal frame-independent strategy—the best single model can do when contexts are indistinguishable.

**I computed this before training any neural network.** Then I trained CNNs three different ways:

- Exclusively on Rule A labels: 1.9% error on Rule A tests, **69.0% ± 0.1%** on Rule B tests  
- Exclusively on Rule B labels: **68.5% ± 0.5%** error on Rule A tests, 2.2% on Rule B tests  
- Balanced mix of both: 36.9% error on Rule A tests, 34.0% on Rule B tests (compromise strategy)

The single-rule models achieved exactly the predicted 70% worst case. The balanced model found the optimal compromise when forced to handle both. The predictions held across multiple random seeds and training configurations.

![Worst-case error analysis showing predicted 70% bound achieved on digit classification](/examples/hallucinations/experiment_10/results/worst_case_error.png)

**This isn't about neural networks being inadequate learners. It's about the task containing a structural impossibility.** No learning algorithm—neural networks, decision trees, hand-coded rules, or humans guessing—can do better than 70% worst-case error when you remove the context markers and force a single model to handle both rules.

The important thing: **we predicted this failure before training any network, purely from analyzing how the two rules contradict each other.**

---

## **5. The Mathematics of Structural Impossibility**

To make this general, we developed a measure called $K(P)$ that quantifies how much inherent contradiction exists in any task. When $K = 0$, the task has no internal contradictions and perfect performance is theoretically possible. When $K > 0$, contradictions exist and some minimum error becomes mathematically unavoidable.

For the digit task above, $K = 0.35$ bits. The formula connecting $K$ to minimum error is:

$$\text{error} \geq 1 - 2^{-K}$$

With $K = 0.35$ bits, that gives minimum error $\geq 21.5\%$. The 70% worst-case we observed is higher because "worst case across two contexts" is a stronger constraint than average error, but both are bounded by the contradiction in the task structure.

**This explains why some prompts consistently lead to garbage outputs: the structural contradiction is unavoidable without adequate witness capacity.**

**K measures structural impossibility, not difficulty.** You can compute it from task structure before any training happens.

Here's how it works mathematically. For any behavior $P$ with multiple contexts (like our Rule A and Rule B), we're looking for the best "unified explanation"—a single model $Q$ that approximates both context-specific behaviors simultaneously. The Bhattacharyya coefficient $BC(p,q) = \sum_x \sqrt{p(x)q(x)}$ measures agreement between probability distributions, ranging from 0 (complete disagreement) to 1 (perfect match).

The agreement coefficient is:

$$\alpha^*(P) = \max_{Q \in FI} \min_{c \in C} BC(p_c, q_c)$$

The "max over models" finds your best shot at reconciliation—the $Q$ that agrees best with all your context-specific behaviors. The "min over contexts" captures the Weakest Link Principle: overall agreement is limited by your worst case, not your average. This minimum-taking is mathematically forced by basic consistency requirements.

Then $K(P) = -\log_2(\alpha^*(P))$.

For the contradictory digit task, solving this optimization gives $\alpha^* \approx 0.78$, hence $K \approx 0.35$ bits. We computed this from the rule structure, it gave us bounds on error rates, and models trained completely differently all hit those bounds.

---

## **6. Architectural Pressure: Witness Capacity**

Even perfectly structured tasks hit a wall with current architectures. After identifying that a task has contradiction $K(P) > 0$, you need to give the model the ability to respond appropriately—for example, by saying "I don't know" instead of guessing.

I call this capability **witness capacity** $(r)$. It's about the information-theoretic capacity to represent uncertainty and reason about it.

Standard softmax provides $r \approx 0$ bits of witness capacity because it forces probability distributions summing to 1.0 across outputs. There's no built-in way to express "I don't have enough information" or "this question doesn't make sense." The architecture treats every input identically: embed, transform through layers, project to output space, softmax, select highest probability. Even when the input has nothing to do with training data, this process generates a confident prediction.

I tested this systematically, measuring witness capacity $r$ across different architectures on tasks with varying contradiction $K$. The setup: 20 combinations of task contradiction ($K$ from 0.5 to 1.16 bits) and architectural capacity ($r$ from 0 to 2 bits), with 5 random seeds each—100 training runs total.

**The results showed sharp phase transitions:**

- When $r < K$: Models failed consistently (100% error on undefined inputs across all seeds)
- When $r \geq K$: Models succeeded consistently (0% error when trained properly)

The transition happened within a narrow band around $r = K$. For a task with $K = 0.7925$ bits, models with $r = 0.5$ bits (insufficient capacity) hallucinated on all undefined inputs. Models with $r = 1.0$ bits (sufficient capacity) abstained perfectly.

![Error rates vs witness capacity showing sharp phase transitions](/examples/hallucinations/experiment_9/results/error_vs_witness.png)

**That's a 100 percentage point difference from architectural capacity alone.** The model literally cannot express uncertainty when $r < K$, so it must guess. When $r \geq K$, it can abstain and avoid hallucination entirely.

To make this concrete, I tested one task with $K = 0.70$ bits under two architectural conditions:

- **With abstention allowed**: 1% hallucination (495 correct abstentions, 5 errors out of 500 trials)
- **Forced to commit**: 76% hallucination (380 errors, 120 abstentions out of 500 trials)

That **75 percentage point gap** isolates architectural pressure from structural pressure. The structural contradiction remained constant. With adequate witness capacity, the model stayed near the theoretical floor at 1%. Without it, hallucination shot to 76%.

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

Using the same framework, I computed the contradiction measure $K(P)$ for quantum correlations at maximum violation. Computations for standard quantum scenarios suggest $K(P)$ is small but nonzero—roughly on the order of hundredths of a bit per measurement pair in CHSH-type tests.

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

This eventually forced me to confront a more basic question: what kinds of information structures can our current mathematics actually represent?

Shannon's framework is extraordinarily successful at what it was designed to do. Entropy tells you how many bits are required to encode a source. Mutual information measures statistical dependence. Channel capacity characterizes the maximum reliable transmission rate.

**But there is a structural assumption baked into this framework:** that information can be modeled by a single globally consistent probability distribution. One coherent model. One joint story about how the data is generated.

This assumption works perfectly when the underlying information source is globally consistent. **The problem is that not all information sources are.**

In many real systems, the same data admits multiple incompatible descriptions depending on context, perspective, or measurement. There is no single joint distribution that simultaneously satisfies all constraints. In these cases, the information source itself is epistemically inconsistent.

The mathematics in contrakit formalizes that distinction. It introduces a structural invariant, $K(P)$, that measures how much global inconsistency is present in an information source. When $K(P) = 0$, the source admits a single coherent probabilistic model and classical information theory applies exactly. When $K(P) > 0$, no such model exists.

**In that regime, classical information theory is no longer complete.**

The operational consequences are precise and unavoidable. Theorems 6–9 prove that epistemic inconsistency imposes an exact and irreducible information-theoretic cost:

**Compression**: For a source $X$ with context $C$, classical theory predicts an optimal rate of $H(X|C)$ bits. When the source contains contradiction $K(P)$, the true optimal rate is $H(X|C) + K(P)$. That extra $K(P)$ isn't inefficiency—it's the fundamental cost of forcing a single coherent codebook onto an epistemically inconsistent system.

**Communication**: A channel serving contradictory decoders loses exactly $K(P)$ bits of capacity.

**Hypothesis testing**: Distinguishing inconsistent models from classical ones requires error exponent at least $K(P)$.

**This is not a modeling artifact. It is a structural property of the information source itself.**

$K(P)$ is measurable. When computed for quantum systems, it reproduces known classical-quantum boundaries with precision. Below $K(P) = 0$, classical models suffice. Above $K(P) = 0$, no classical probabilistic model can represent the system without additional side information.

In this sense, **classical information theory is not wrong—it is incomplete.** It fully characterizes epistemically consistent sources (where $K = 0$). It does not fully characterize epistemically inconsistent ones (where $K > 0$).

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
- The $r = K$ phase transition is reproducible across 100+ training runs with sharp boundaries
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