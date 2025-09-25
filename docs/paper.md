## 1.2 Our Contribution: Axiomatizing the Impossible

We develop a reconciliation calculus for contexts (frames) and show that there is an essentially unique (under our axioms) scalar capturing the informational cost of enforcing one story across incompatible contexts.

1. **Axiomatic characterization (inevitability)**. 
We prove that, under axioms A0–A5, the essentially unique contradiction measure is 
    
    $$
    K(P)=-\log_2 \alpha^\star(P)
    $$
    
    where 
    
    $$
    \alpha^\star(P)=\max_{Q\in \mathrm{FI}}\ \min_{c}\ \mathrm{BC}\!\big(p_c, q_c\big)
    $$
    
    Here $\mathrm{FI}$ is the convex set of frame-independent behaviors (the "unified story" polytope for finite alphabets). In quantum settings, $\mathrm{FI}$ coincides with non-contextual models, yielding a principled violation strength. (Theorems 2–4; aggregator lemma; Theorem 1)
    
2. **Agreement-kernel uniqueness.** 
Assuming refinement separability, product multiplicativity, and a data-processing inequality, we show the per-context agreement is uniquely the Bhattacharyya affinity. (Theorem 3)
3. **Well-posedness & calibrated zero.** 
For finite alphabets with $\mathrm{FI}$ nonempty/compact/convex/product-closed, the program for $\alpha^\star(P)$ attains an optimum with $\alpha^\star(P)\in[0,1]$; thus $K(P)\geq 0$ and $K(P)=0$ iff $P\in\mathrm{FI}$. This establishes an absolute zero and a stable scale. (Proposition family)
4. **Resource laws.** 
We prove additivity $K(P\otimes R)=K(P)+K(R)$ and monotonicity under free operations (post-processing, outcome-independent context mixing, convex mixing, adding $\mathrm{FI}$ ancillas). (Theorem 5 + corollaries)
5. **Operational triad from one overlap.** 
The same $\alpha^\star$ yields, under standard conditions (finite alphabets, i.i.d. sampling): (i) discrimination error exponents for testing real vs. simulated behavior, (ii) simulation overheads—the "contradiction tax"—to imitate multi-context data, and (iii) prediction lower bounds (irreducible regret when restricted to an $\mathrm{FI}$ model). (Theorems 6–8)
6. **Computability & estimation.** 
We provide a practical minimax/convex program (with column-generation option) for $\alpha^\star$, plus a consistent plug-in estimator for $K$ from empirical frequencies with bootstrap CIs; A reference implementation (*contrakit*) accompanies the paper.
7. **Specialization to quantum contextuality**. 
With $\mathrm{FI} =$  the non-contextual set, $K$ is a contextuality monotone (zero iff non-contextual; monotone under free ops; additive). (Theorem 9)

When one story won’t fit, we measure the seam.

## 1.3 Structure and Scope

The paper moves from motivation → mechanism → consequences.

While **§§1–2** motivate the need for a single yardstick and ground it in a concrete device; **§§3–5** build the calculus; **§6** gives task-level consequences; **§7** makes it implementable; **§§8–10** place, bound, and extend the results; while the appendices supply proofs and worked cases.

**Overview:**

- Motivation by example (**§2**). The Lenticular Coin is a minimal classical device exhibiting odd-cycle incompatibility; it previews how $K(P)$ registers contradiction in bits.
- Framework → axioms → results (**§§3–5**). **§3** formalizes observables, contexts (frames), behaviors, the baseline $\mathrm{FI}$, Bhattacharyya overlap, and the minimax program (with standing assumptions). **§4** states and motivates axioms A0–A5. **§5** presents the main theorems, including the fundamental formula $K(P)=-\log_2 \alpha^{\star}(P)$ and additivity.
- From quantity to consequences (**§6**) and practice (**§7**). **§6** develops operational theorems in discrimination (error exponents), simulation (overheads; the contradiction tax), and prediction (regret). **§7** provides a practical minimax/convex program for $\alpha^{\star}$, a plug-in estimator for $K$ with bootstrap intervals, and a brief note on the reference implementation (*contrakit*).
- Context and boundaries (**§§8–10**; **App**.). **§8** positions the work relative to contextuality, Dempster–Shafer, social choice, and information distances. **§9** states limitations and scope. **§10** sketches near-term extensions. 

**App. A** contains full proofs and technical lemmas; **App. B** holds worked examples, and **App. C** offers case studies for review.

**How to read:**

- For guarantees, skim **§2**, then read **§§3–5** for the formal core and **§6** for operational meaning; see **App. A** for proofs.
- For implementation, jump from **§2** to **§§6–7** (algorithms, estimation), backfilling definitions from **§3** as needed; see **App. B** for worked cases.

**Scope:**

Throughout we assume finite alphabets and the usual compatibility/no-disturbance setting. The baseline $\mathrm{FI}$ is nonempty, compact, convex, and product-closed—hence a polytope for finite alphabets. Results are domain-general in the following sense: once $\mathrm{FI}$ is specified for a given domain, the same scalar $K(P)$ applies without modification, with a calibrated zero ($K=0$ iff $P\in \mathrm{FI}$) and shared units across cases.

---

# 2. Building Natural Intuition: The Lenticular Coin

> *"What we observe is not nature itself but nature exposed to our method of questioning."

— Werner Heisenberg, Physics and Philosophy: The Revolution in Modern Science*
> 

Here, "contradiction" doesn't refer to the classical logical impossibility ($A$ and $¬A$), but rather to *epistemic incompatibility*: when two equally valid observations cannot be reconciled within a single reference frame ($A@X ∧ B@Y$ where $X$ and $Y$ represent incompatible contexts). This is similar to special relativity, where two observers can measure different times for the same event—and both are correct—because the reference frame fundamentally matters. We can consider special relativity: two clocks read different times for the same event and both are right—*because frame does the real work*. 

Lenticular images make this tactile. Tilt a postcard: **from one angle you see one picture; from another, a different one. The substrate doesn’t change—your perspective does.** 

If we apply that to a fair coin, then like Shannon’s coin, it has two sides and we flip it at random. Unlike Shannon’s coin however, each face is printed lenticularly so that what you see, depends on the viewing angle. We put the coin on the table with each face lenticularly printed, so the message you see depends on where you stand. We flip the coin. Since one person stands to the left, and the other to the right, when the coin lands, the left observer sees YES and the right observer sees NO; on the next flip those roles swap. 

When they compare notes, they’ll always disagree:

| Coin Side | LEFT Observer Sees | RIGHT Observer Sees |
| --- | --- | --- |
| HEADS | YES | NO |
| TAILS | NO | YES |

This is intuitive, that isn’t a mistake or noise; it’s baked into the viewing geometry. What happened depends on where you looked.

Formally we’d say: 

Let $S\in{\text{HEADS},\text{TAILS}}$ be the face up, $P$ the viewpoint (e.g., $\text{LEFT}$ or $\text{RIGHT}$), and let $O(S,P)\in{\text{YES},\text{NO}}$ be the visible message. 

By design,

$$
O(S,P)=
\begin{cases}
\text{YES}, & (S,P)\in\{(\text{HEADS},\text{LEFT}),(\text{TAILS},\text{RIGHT})\},\\
\text{NO},  & (S,P)\in\{(\text{HEADS},\text{RIGHT}),(\text{TAILS},\text{LEFT})\}.
\end{cases}
$$

We commence each trial as follows: flip the coin (fair, $1/2$–$1/2$), both observers record what they see, then compare notes. They always disagree. From either seat alone, the sequence looks like a fair binary source. Jointly, the outcomes are perfectly anti-correlated. While it remains true that what happens depended on where you were, this version still admits a single global description once we include $P$ in the state: the device implements a fixed rule ("$\text{LEFT}$ shows the opposite of $\text{RIGHT}$, with flip swapping roles"). 

Thus, this is anti-correlation, not an irreconcilable contradiction.

---

## 2.1 Model Identification ≠ Perspectival Information

Learning the device's rule is genuine information; after it's known, the per-flip fact of "we disagree" carries no further surprise—it is exactly what the rule predicts. Before you discover the rule, several live hypotheses compete (e.g., "always-same," "always-opposite," "independent"). Observing outcomes drains that model uncertainty. That is not the irreducible information we speak on here.

**Formally**: if $M$ denotes which rule is true and $D_{1:k}$ the first $k$ observations, the information gained about the rule is the drop in uncertainty: 

$$
I(M; D_{1:k}) \;=\; H(M)\;-\;H\!\left(M\mid D_{1:k}\right).
$$

With a uniform prior over the three hypotheses, two consecutive "opposite" outcomes yield the posterior $(0.8,\,0.2,\,0)$ (in the order "always-opposite," "independent," "always-same"), cutting entropy from $\log_2 3 \approx 1.585$ bits to about $0.722$ bits. You are learning—but thereafter each new row shaves off less and less.

Intuitively: the surprise lives in discovering the rule. Once your posterior has essentially collapsed, "we disagree—again" is confirmation, not news. Each flip still tells you which joint outcome happened—that's one bit about the event—but it no longer tells you anything fresh about the governing rule. So the first Lenticular Coin sits at the *model-identification layer*: you infer the rule that governs the observations. 

That is standard Shannon/Bayesian territory—useful, but not yet our target notion. It shows that perspective changes what you see, not what is true: there is a single global rule, simply viewed from different seats. 

Once viewpoint is modeled within the state, **one law explains everything**.

1. $\text{LEFT}$ and $\text{RIGHT}$ always disagree;
2. $\text{HEADS}$ → $\text{LEFT}$ says $\text{YES}$ ($\text{RIGHT}$ says $\text{NO}$),
3. $\text{TAILS}$ → $\text{RIGHT}$ says $\text{YES}$ ($\text{LEFT}$ says $\text{NO}$).

The law is more than a lookup table; it is the rule everyone follows when turning what they see into a report. Given a state $S$ and a seat $P$, the law fixes which word must be written down. In information-theoretic terms, it is the channel $p(o\mid s,p)$; in plain terms, it is the shared reporting language that makes my "$\text{YES}$" mean the same thing as your "$\text{YES}$".

This matters because, once the law is fixed, **records should cohere**: different seats can yield different entries, but all entries are expected to fit under the same rule. We will use this distinction shortly.

To continue, we use a mundane feature of lenticular media: the transition band. It introduces a lawful "both" outcome—legitimate ambiguity—where "what happened" begins to blur. This is where a frame-independent summary begins to fail unless the context label is carried along; the reports remain consistent, but the summary without frames does not. 

This pressure toward contradiction will become explicit in §2.3.

---

## 2.2 The Lenticular Coin: the Natural “Both” Band

The first coin taught us a rule: $\text{LEFT}$ and $\text{RIGHT}$ must disagree. This constituted genuine learning—a discovery that reduces informational uncertainty as you understand how the device operates. After the rule is known, each flip merely confirms expectation. 

To show the persisting structure we care about when we say “frame”, we only need to acknowledge a mundane physical fact about lenticular media: there is a transition band where both layers are simultaneously visible. That band is not an error; it is part of the object. Place the coin as before, but mark three viewing positions: $\text{LEFT}$, $\text{MIDDLE}$, and $\text{RIGHT}$. 

Each face is printed lenticularly so being positioned at $\text{LEFT}$ cleanly shows $\text{YES}$, $\text{RIGHT}$ cleanly shows $\text{NO}$, and being at $\text{MIDDLE}$ shows natural transition band where both overlays are visibly present. When the coin flips from $\text{HEADS}$ to $\text{TAILS}$, the clean views swap ($\text{YES}$↔$\text{NO}$), yet the $\text{MIDDLE}$ never changes, always showing $\text{BOTH}$.

**Formally:**

For face $S\in\{\text{HEADS},\text{TAILS}\}$ and position $P\in\{\text{LEFT},\text{MIDDLE},\text{RIGHT}\}$, the observation $O$ satisfies

$$
O(S,P)=
\begin{cases}
\text{BOTH}, & P=\text{MIDDLE},\\
\text{YES},  & (S,P)\in\{(\text{HEADS},\text{LEFT}),(\text{TAILS},\text{RIGHT})\},\\
\text{NO},   & (S,P)\in\{(\text{HEADS},\text{RIGHT}),(\text{TAILS},\text{LEFT})\}.
\end{cases}

$$

Nothing metaphysical is hiding here; this is just a postcard effect, elevated to a protocol.

However, two things now become unavoidable:

1. **Ambiguity is intrinsic**. a competent observer at $\text{MIDDLE}$ can truthfully report $\text{BOTH}$; that outcome is lawful, not noise.
2. **Perspective becomes a per-trial budget**. reports are reproducible only if the viewing frame travels with the message. "I saw $\text{YES}$" is underspecified; "I saw $\text{YES}$ from $\text{LEFT}$" is reconstructible.

Put differently, **with three seats the law is now *context-indexed***. For a fixed seat $P$:

- $P=\text{LEFT}$: $\text{YES}$ on $\text{HEADS}$, $\text{NO}$ on $\text{TAILS}$.
- $P=\text{RIGHT}$: $\text{NO}$ on $\text{HEADS}$, $\text{YES}$ on $\text{TAILS}$ (the inverse of $\text{LEFT}$).
- $P=\text{MIDDLE}$: $\text{BOTH}$ on both flips (constant).

As a consequence, you cannot tell the full story unless **you model** $P$. There is a small but steady information loss—about  $\frac{2}{3}$ of a bit per record (App B.1)—if you drop the frame. It’d be no different than asking ‘did they break the law?’ without saying where it happened.

Run the experiment for many flips and this structure shows up in plain statistics: $\text{LEFT}$ and $\text{RIGHT}$ disagree predictably; the $\text{MIDDLE}$ registers a stable experience of $\text{BOTH}$ events; and the frame labels are continually required to reconcile otherwise incompatible yet honest reports. The disagreement is no longer just "they always oppose" (a rule you learn once).

The extra content is small, but it never goes away. It is not the one-off surprise of model identification; it is a steady coordination cost—bits you must carry every time if downstream agreement is the goal. 

This is to build an intuition on perspective: This is to build an intuition on perspective: the frame itself is information, and while not entirely new, it’s modeled far less often than it should be. Shannon’s model doesn’t forbid modeling frames; it simply doesn’t quantify incompatibility across contexts. This is not contradiction yet: the reports are consistent—but we needed to show this distinction. 

We show this to distinctly separate information loss from dropping frames (priced by $H(P\mid O)$, here $\tfrac{2}{3}$ bit/record) from structural contradiction across frames (priced by $K(P)$), so readers won't conflate "forgot the label" with "no single story fits."

---

## **2.3 The Lenticular Coin’s Irreducible Contradiction**

Having built intuition around perspective and missing information, we finally now arrive to the paper’s purpose: a type of contradiction that persists even when context is fully preserved and the setup is completely transparent. This time we disallow ambiguity.

This time we disallow ambiguity.

> Axiom (Single-Valued Reports).
> 
> 1. Each observer must report a single word: YES or NO. 
> 2. No BOTH entries allowed.

Consider the same lenticular coin, now mounted on a motorized turntable that rotates in precise increments. Three observers—Nancy, Dylan, Tyler—sit at fixed viewing angles along the viewing axis. The lenticular surface shows $\text{YES}$ at $0^\circ$, $\text{NO}$ at $180^\circ$, and $\text{BOTH}$ at the transition $90^\circ$ (with half-width $w_{\text{both}}$). Fix three platform orientations that graze but avoid the transition band:

$$
\psi_1=0^\circ,\qquad \psi_2=90^\circ-\varepsilon,\qquad \psi_3=180^\circ-\varepsilon,\quad \\ \text{with }\varepsilon > w_{\text{both}}+\delta.
$$

At each orientation $\psi_k$, the platform stops; exactly two observers look while the third looks away (no omniscient snapshot). 

![image.png](attachment:957f05af-b878-46e7-ad8d-8769853d20df:image.png)

The three rounds:

Round	             Orientation $\psi$	 Observers	      Reports

| 1 | $0^\circ$ | Nancy, Tyler | $\text{YES}$,$\text{NO}$ |
| --- | --- | --- | --- |
| 2 | $90^\circ-\varepsilon$ | Tyler, Dylan | $\text{NO}$,$\text{YES}$ |
| 3 | $180^\circ-\varepsilon$ | Dylan, Nancy | $\text{NO}$,$\text{YES}$ |

Every local report is valid for lenticular viewing. As **§§2.1–2.2** established, once the codebook is fixed, the local laws $(S,P)\!\mapsto\!O$ render each context coherent; the only failure we saw came from dropping $P$, not from the law—and that was fixed by modeling $P$.

But here, a different question creates a different failure mode: can we assign each person a single, round-independent label ($\text{YES}$/$\text{NO}$) that matches all three pairwise observations? 

The rounds impose:

$$
\text{Nancy} \neq \text{Tyler},\quad
\text{Tyler} \neq \text{Dylan},\quad
\text{Dylan} \neq \text{Nancy}.
$$

To make it tactile, imagine the conversation between Tyler, Dylan, and Nancy:

| **Round** | **Nancy** | **Tyler** | **Dylan** |
| --- | --- | --- | --- |
| Round 1 —$\psi_1=0^\circ$ | "From here I see YES." | "Strange, because I see NO." | "I didn't look this round." |
| Round 2 —$\psi_2=90^\circ-\varepsilon$ | "I wasn't looking this time." | "Again I see NO." | "Well I see YES." |
| Round 3 —$\psi_3=180^\circ-\varepsilon$ | "From my seat I see YES." | "I sat out this one." | "Now I see NO." |

When asked to collectively describe how the coin operated, they would be unable to reach agreement, despite each telling the truth. The series of observations created a situation where no single, coherent description of the coin could accommodate all their valid experiences.

In short: the local laws are all obeyed, yet there is no single global law—no fixed YES/NO per person—that makes all three pairs correct at once. This is the classic odd-cycle impossibility: three pairwise "not equal" constraints cannot be satisfied by any global assignment. Each observation is right in its context, yet no frame-independent set of labels can satisfy all three at once. 

The incompatibility isn't noise; it's geometric—arising from how valid views interlock. Put differently: even an omniscient observer must choose a frame.

You can know everything—but not from one place.

This differs from the missing-context case in §2.2: carrying the frame there resolved ambiguity. Here, even with perfect context preservation, no frame-independent summary exists. The three questions, asked in sequence, admit no coherent single-story answer. The turntable is simple, not exotic; one could build this in a classroom.

Information-theoretically, "no global law" means there is no joint $Q\in\mathrm{FI}$ whose every context marginal matches $P$. Classical information theory can represent each context separately once $\psi$ is included among the variables; what it was *never designed* to represent under a single $Q$ is precisely the odd-cycle pattern. At best, two of the three pairwise constraints can be satisfied, so the irreducible disagreement rate is 1/3 per cycle.

Consequently, any frame-independent approximation must be wrong in at least one context. Numbers computed under a unified $Q$—entropy, code length, likelihood, mutual information, test-error exponents—are systematically misspecified. The gap is quantifiable: coding incurs $D(P\|Q^{\star})$ extra bits per sample for $Q^{\star}=\arg\min_{Q\in\mathrm{FI}} D(P\|Q)$; testing exponents are capped by $\alpha^{\star}(P)<1$ (equivalently, $K=-\log_2 \alpha^{\star}(P)$).

Quantitatively, the best frame-independent agreement is

$$
\alpha^{\star}=\sqrt{\tfrac{2}{3}},
$$

so the contradiction bit count is

$$
K(P)=-\log_2 \alpha^{\star}(P)=\tfrac{1}{2}\log_2\!\frac{3}{2}\approx 0.2925\ \text{bits per flip}.
$$

This is the **contradiction bit**: the per-observation cost of compressing all perspectives into one coherent view. The optimal dual weights are uniform, $\lambda^{\star}=(1/3,1/3,1/3)$, and the optimal $Q^{\star}$ saturates $\mathrm{BC}(p_c,q^{\star}_c)=\sqrt{2/3}$ in all three contexts.

---

## 2.4 The Key Insight: Revisiting the Axiom

> Axiom (Single-Valued Reports):
> 
> 1. Each observer must report a single word: YES or NO. 
> 2. No BOTH entries allowed 

It is fair to ask whether this axiom creates the contradiction. If we permit BOTH, the clash indeed disappears. That is the point. The contradiction does not come from the coin; it comes from our reporting architecture—from forcing plural observations into single-valued records. The world is pluralistic, yet our summaries of the world, are not.

This stance is inherited, it was never inevitable. Consider how fundamental this constraint is in the systems we rely on. Boole fixed propositions as true and false. Kolmogorov placed probability on that logic, and Shannon showed how such decisions travel as bits. None of these frameworks declared the world binary, they were just well-suited for the task at hand. 

If anything, they merely declare our records *can be* binary. Modern databases and protocols naturally followed suit: one message, one symbol, one frame. It wasn’t until recently that plurism emerged as an engineering problem.

However, none of these systems claimed the world was binary. What they did claim, and what we've inherited, is that our **records** *can be* binary. We've adopted this convention not because it's natural, but because it's standard. It's the foundation of virtually every digital system, database, and channel of formal communication: observations must collapse to a single symbol. One report, one truth, one frame.

Classical measures do an *excellent* job within a context: they price which outcome occurred. What they do not register, under a frame-independent summary, is a different kind of information—that several individually valid contexts can be valid, but cannot be made to agree at once. That is not “more surprise about outcomes”; it is a statement about feasibility across contexts.

Thus, this imposed axiom is no more a contrivance than the digital computer itself. The contradiction bit $K(P)$ then measures the structural cost of insisting on one story when no such story exists. The observed clash is not noise, deception, or paradox in nature; it’s simply the price of flattening—of collapsing perspectival structure to a single label.

Namely, there are two kinds of information are in play:

- **Statistical surprise**: Which outcome happened?—handled inside each context by Shannon’s framework.
- **Structural surprise**: Can these valid observations coexist in any global description?—assigned zero by a single-story summary, and restored by K(P).

Shannon priced randomness within a frame. When frames themselves clash, there is an additional, irreducible cost. Measuring that cost is necessary for any account of reasoning across perspectives.

Succinctly put:

> *To model intelligence is to model perspective. To model perspective is to model contradiction. And to model contradiction is to step beyond the frame and build the next layer.*
> 

---

# 3. Mathematical Foundations — The Geometry of Irreconcilability

Having established the foundations, the lenticular coin showed something we can now make precise: **contradiction has structure**. When Nancy, Dylan, and Tyler couldn't agree despite all being correct, they encountered an irreducible incompatibility built into their situation's geometry.

**Contradiction has geometry just as information has entropy.** While Shannon showed us that uncertainty follows precise mathematical laws, we'll show that clashes between irreconcilable perspectives follow equally discoverable patterns with measurable costs.

## 3.1 The Lenticular Coin, Reframed.

Let's be precise about what we measured when our three friends observed the lenticular coin:

- **Observables**: Things we can measure (like "what word does Nancy see?")
- **Contexts**: Different ways to probe the system (like "which pair of observers look simultaneously?")
- **Outcomes**: What actually happens when we look (like "Nancy sees YES, Tyler sees NO")
- **Behaviors**: The complete statistical pattern across all possible contexts

A **behavior** functions like a comprehensive experimental logbook. For every way you might probe the system, it records the probability distribution over what you'll observe. In our rotating coin experiment, this logbook includes entries like: "When Nancy and Tyler both look simultaneously, there's a 50% chance Nancy sees YES while Tyler sees NO, a 50% chance Nancy sees NO while Tyler sees YES, and 0% chance they agree."

The mathematical formalization captures this intuitive picture directly. We have observables $\mathcal{X} = \{X_1, X_2, \ldots, X_n\}$, where each can display various outcomes. A **context** $c$ is simply a subset of observables we examine together. A **behavior** $P$ assigns to each context $c$ a probability distribution $p_c$ over the outcomes we might see (App. A.1.1–A.1.4).

This isn't exotic machinery—just systematic bookkeeping for multi-perspective experiments.

### The Frame-Independent Baseline: When Perspectives Align

Some behaviors exhibit no contradictions at all. Consider a simple coin that always shows the same face to every observer—heads to Nancy, heads to Dylan, heads to Tyler. Here, all perspectives align perfectly. Even though different people look at different times or from different angles, their reports weave into one coherent explanation: "The coin landed heads."

This represents **frame-independent** behavior: one underlying reality explains everything different observers see. Disagreements arise only because observers examine different aspects of the same coherent system, not because the system itself contains contradictions.

But our lenticular coin behaves differently. Nancy, Dylan, and Tyler see genuinely incompatible things that resist integration into any single coherent explanation. This represents **frame-dependent** behavior—the signature of irreducible contradiction.

Mathematically, a behavior is **frame-independent** when there exists a single "master explanation" that simultaneously accounts for what every context would observe. More precisely, there must be a probability distribution $\mu$ over complete global states such that each context's observations are simply different slices of these states.

A **complete global state** is a full specification of every observable simultaneously—like saying "Nancy sees YES, Dylan sees BOTH, Tyler sees NO" all at once. If such states exist and one distribution $\mu$ over them reproduces all our contextual observations, then we have our baseline.

Three equivalent ways to understand this baseline:

1. Unified explanation: All observations integrate into one coherent account
2. Hidden variable model: A single random "state of affairs" underlies what each context reveals
3. Geometric picture: The baseline is the convex hull of "deterministic global assignments"—scenarios where every observable has a definite value

The **frame-independent set** (App. A.1.6) contains all behaviors admitting such unified explanations. These form our "no-contradiction" baseline. Crucially, FI has excellent mathematical properties in our finite setting: it's nonempty, convex, compact, and closed. This gives us a solid foundation for measuring distances from it.

Two cases will matter later. If there exists $Q\in\mathrm{FI}$ with $Q=P$, then $\alpha^\star(P)=1$ and $K(P)=0$ (no contradiction). If no such $Q$ exists, then $\alpha^\star(P)<1$ and $K(P)>0$ quantifies the minimal deviation any unified account must incur to explain all contexts at once

## 3.2 Measuring the Distance to Agreement

To quantify contradiction, we need a notion of "distance" between our observed behavior and the closest frame-independent explanation. When comparing probability distributions across multiple contexts, the Bhattacharyya coefficient provides exactly what we need. 

For probability distributions $p$ and $q$ over the same outcomes:

$$
\text{BC}(p,q) = \sum_{\text{outcomes}} \sqrt{p(\text{outcome}) \cdot q(\text{outcome})}
$$

This measures "probability overlap" (App. A.2.1). When $p$ and $q$ are identical, $\text{BC}(p,q) = 1$ (perfect overlap). When they assign probability to completely disjoint supports, $\text{BC}(p,q) = 0$ (no overlap). Between these extremes, the coefficient tracks how much the distributions have in common.

Three properties make this measure particularly suitable:

1. **Perfect agreement detection**: $\text{BC}(p,q) = 1$ if and only if $p = q$ (see App. A.2.2.2)
2. **Mathematical tractability**: It's concave and well-behaved for optimization (see App. A.2.2.3)
3. **Compositional structure**: For independent systems, $\text{BC}(p_1 \otimes p_2, q_1 \otimes q_2) = \text{BC}(p_1,q_1) \cdot \text{BC}(p_2,q_2)$ (see App. A.2.2.4)

This third property proves essential—contradiction costs multiply across independent subsystems, just like probabilities do.

## 3.3 The Core Measurement: Maximum Achievable Agreement

Given an observed behavior $P$, we now address the central question: across all possible frame-independent behaviors, what's the maximum agreement we can achieve with our observations?

A subtle but crucial choice emerges. We could measure agreement context-by-context, then average. But which contexts deserve more weight? The natural answer: let the worst-case contexts determine the overall assessment. If even one context shows poor agreement with our proposed frame-independent explanation, that explanation fails to capture the true structure.

This leads to our **agreement measure** (App. A.3.1):

$$
\alpha^\star(P) = \max_{Q \in \text{FI}} \min_{\text{contexts } c} \text{BC}(p_c, q_c)
$$

The formula reads: "Among all frame-independent behaviors $Q$, find the one that maximizes the worst-case agreement with our observed behavior $P$ across all contexts."

The max-min structure captures something essential about contradiction. Perspective clash concerns **universal reconcilability**. A truly frame-independent explanation must account for *every* context satisfactorily. One persistently problematic context breaks the entire unified narrative.

This optimization problem has well-behaved mathematical structure. By Sion's minimax theorem, we can equivalently write:

$$
\alpha^\star(P) = \min_{\lambda\in\Delta(\mathcal{C})} \max_{Q \in \text{FI}} \sum_{\text{contexts } c} \lambda_c \cdot \text{BC}(p_c, q_c)
$$

where $\lambda$ is a probability distribution over contexts. The optimal weighting $\lambda^\star$ reveals which contexts create the worst contradictions—they receive the highest weights in the sum. (see App. A.3)

### The Contradiction Bit

We can now define our central measure:

$$
K(P) = -\log_2 \alpha^\star(P)
$$

This is the **contradiction bit count**—the information-theoretic cost of the perspective clash. (App. A.3.1) 

When $\alpha^\star(P) = 1$ (perfect frame-independence), we get $K(P) = 0$ contradiction bits. As $\alpha^\star(P)$ decreases toward zero, $K(P)$ grows, measuring how much information is lost when we insist on a single unified explanation.

**Key properties:**

- $K(P) = 0$ if and only if $P$ is frame-independent (App. A.4.3)
- $K(P) \leq \frac{1}{2}\log_2(\max_{c\in\mathcal{C}}|\mathcal{O}_c|)$ (contradiction is bounded) (App. A.4)
- For our lenticular coin: $K(P) = \frac{1}{2}\log_2(3/2) \approx 0.29$ bits per observation (App. B.2)

## 3.4 The Game-Theoretic Structure

The minimax formulation (App. A.3.2) reveals the deeper structure of contradiction 

The optimization:

$$
\alpha^\star(P) = \min_{\lambda\in\Delta(\mathcal{C})} \max_{Q \in \text{FI}} \sum_c \lambda_c \cdot \text{BC}(p_c, q_c)
$$

has the structure of a two-player game (App. A.3.2):

- **The Adversary**: Chooses context weights $\lambda$ to focus on the most problematic contradictions
- **The Explainer**: Selects a frame-independent behavior $Q$ to maximize agreement despite the adversarial focus

The optimal strategies $(\lambda^\star, Q^\star)$ reveal the deepest structural features (Theorem A.5.1):

- $\lambda^\star$ identifies which contexts resist reconciliation
- $Q^\star$ provides the best possible unified approximation
- The game value $\alpha^\star(P)$ measures the ultimate limits of perspective reconciliation

**For our lenticular coin**, the solution exhibits perfect symmetry:

- $\lambda^\star = (1/3, 1/3, 1/3)$—all contexts are equally problematic
- $Q^\star$ has, in each context, $(q_{00},q_{01},q_{10},q_{11})=(1/6,1/3,1/3,1/6)$ (see App. B.2)
- $\alpha^\star(P) = \sqrt{2/3}$—the limit of achievable agreement

This isn't coincidental—it reflects the democratic nature of odd-cycle contradictions. No single context is more problematic than others; the contradiction distributes evenly across all perspectives.

---

## 3.5 Recursive Structure and Hierarchical Application

The formalism extends to hierarchies without modification. An *observer's* assessment can be treated as an additional **observable** with its own labeled outcomes, and the resulting system is analyzed with no changes.

**Promoting assessments to observables**
Let $\mathcal{X}$ be the current set of observables with behavior $P=\{p_c\}_{c\in\mathcal{C}}$. Suppose an observer $\ell$ assigns, for some object $Y\in\mathcal{X}$, a distribution over labels $\Sigma_\ell(Y)$ (e.g., *Reliable/Unreliable*). We **promote** this assessment to a new observable $X_\ell^Y$ with outcome alphabet $\Sigma_\ell(Y)$, and we extend the context family by allowing $X_\ell^Y$ to co-occur with the relevant observables it summarizes. The augmented behavior is $P' = P \uplus \{p^{(\ell,Y)}\}$. No definitions change: contexts, behaviors, $\mathrm{FI}$, $\alpha^\star$, and $K=-\log_2\alpha^\star$ are applied exactly as before to $P'$.

**Level stability**
Two closure facts ensure stability under this promotion:

1. **Relabeling and product closure (App. A.1.8) of $\mathrm{FI}$:** Adding $X_\ell^Y$ amounts to adjoining an observable with fixed marginals (determined by $\ell$). Because $\mathrm{FI}$ is convex and closed under products with fixed factors, the maximizer $Q^\star\in \mathrm{FI}$ in (3.1) extends to the enlarged space without changing form.
2. **Max–min preservation (App. A.2.2.4, App. A.3.2):** The objective in (3.1) for $P'$ is still a worst-case Bhattacharyya overlap over contexts. New contexts contribute multiplicative factors already fixed by $\ell$; taking the minimum over contexts preserves the max–min structure.

Consequently, the same computation that measures peer-level clash among base observables also measures meta-level clash among observers' assessments. No new machinery is introduced—only additional rows in the same logbook.

**Ceiling for cross-level contradiction (App. A.4.x)**
Cross-level disagreement cannot exceed the uncertainty already present in the upstream assessment. If an upstream observable $X_\ell^Y$ has marginal $p\in[0,1]$ on a label (e.g., "Reliable"), then for any downstream assessment opposing it,$K_{\text{cross}} \;\le\; H_2(p)\;=\; -\,p\log_2 p \;-\; (1-p)\log_2(1-p).$Intuition: disagreement bits cannot exceed the bits of slack available at the level being contradicted. A $50/50$ assessment admits up to $1$ bit of clash; a $95/5$ assessment caps near $0.286$ bits—the binary entropy of a $95/5$ split. This matches the gradients reported in the examples.

**Consequences**

1. **Recursive measurability.** Repeating the promotion step (reviewers → supervisor → director, …) yields a hierarchy in which $K$ quantifies tension both within a level and across levels, with identical definitions.
2. **Budget of disagreement.** As upstream assessments sharpen (lower entropy), the maximum attainable cross-level contradiction decreases; degenerate assessments (probability $0$ or $1$) admit none.

*Proof sketch (bounds).* For the ceiling: oppose a fixed marginal $p$ by maximizing the decrease in Bhattacharyya overlap; Jensen concavity of $\sqrt{\cdot}$ inside $\mathrm{BC}$ upper-bounds the adversary's gain by the upstream entropy $H_2(p)$. For level stability: embed $Q^\star\in\mathrm{FI}$ into the enlarged space with the fixed factor induced by $X_\ell^Y$; multiplicativity of $\mathrm{BC}$ over independent components and the outer $\min_c$ preserve the max–min form and value up to the added fixed factors.

> Plain-language takeaway. Watching the watchers doesn't need new machinery. You promote opinions to outcomes and run the same distance-to-agreement calculation. And the fiercest possible clash across levels is limited by how unsure the lower level already was.
> 

## 3.6 Beyond Entropy — A Theory of Contradiction

We’ve now arrived at the core deliverable of this framework: a general-purpose method for measuring contradiction as a first-class information-theoretic quantity.

1. **Recognition**: $K(P) = 0$ precisely characterizes frame-independent behaviors
2. **Quantification**: $K(P)$ measures the information-theoretic cost of perspective clash
3. **Optimization**: The minimax structure identifies worst-case contexts and optimal approximations
4. **Bounds**: Contradiction is mathematically well-behaved and bounded
5. **Universality**: The framework applies to any multi-context system, regardless of domain

This framework aligns formally with the language of contextuality in quantum foundations—most notably the sheaf-theoretic formulation introduced by Abramsky and Brandenburger. **But unlike prior approaches rooted in quantum mechanics, we derive this structure independently, from first principles within information theory.** No quantum assumptions are required.

The geometry we uncover does not belong exclusively to quantum physics—though the resemblance is striking. While quantum contextuality was not the starting point of our inquiry, it emerged as a natural consequence. What matters more is that this same geometry **recurs across domains** whenever multiple perspectives must be reconciled under global constraints: in distributed systems, organizational paradoxes, statistical puzzles, and beyond.

This suggests that contradiction is not a quantum anomaly—but instead exists as a **universal structural phenomenon** in information itself. In this view, the contradiction bit becomes a natural companion to Shannon’s entropy: where entropy quantifies randomness within a single frame, contradiction quantifies incompatibility across frames. Together, they form a multi-dimensional theory of information—one capable of describing not just uncertainty, but also irreconcilability.

In the next section, we’ll show how the characteristics within our lenticular coin naturally lead to this solution—not as an invention, but as an inevitability.

---

## 4. The Axioms

We’ll now establish that any reasonable contradiction measure must satisfy six elementary properties, which together uniquely determine the form $K(P) = -\log_2 \alpha^\star(P)$.

---

### Axiom A0: Label Invariance

> *Contradiction lives in the structure of perspectives—not within perspectives themselves.*
> 

**Formally**:

$K$ is invariant under outcome and context relabelings (permutations).

---

### Axiom A1: Reduction

> *We cannot measure a contradiction if no contradiction exists.*
> 

**Formally**:

$K(P) = 0$ if and only if $P \in \mathrm{FI}$.

---

### Axiom A2: Continuity

> *Small disagreements deserve small measures.*
> 

**Formally**:

$K$ is continuous in the product $L_1$ metric:

$$
d(P,P') = \max_{c \in \mathcal{C}} \bigl\|p(\cdot|c) - p'(\cdot|c)\bigr\|_1
$$

---

### Axiom A3: Free Operations

> *Structure lost may conceal contradiction — but never invent it.*
> 

**Formally**:

$K$ is monotone under stochastic post-processing of outcomes, stochastic merging of contexts via public lotteries independent of outcomes and hidden variables, convex combinations $\lambda P + (1-\lambda)P'$, and tensoring with FI ancillas.

---

### Axiom A4: Grouping

> *Contradiction is a matter of substance, not repetition.*
> 

**Formally**:

$K$ depends only on refined statistics when contexts are split via public lotteries, independent of outcomes and hidden variables. In particular, duplicating or removing identical rows leaves $K$ unchanged.

---

### Axiom A5: Independent Composition

> *Contradictions compound; they do not cancel.*
> 

**Formally**:

For operationally independent behaviors on disjoint observable sets:

$$
K(P \otimes R) = K(P) + K(R)
$$

This requires that FI be closed under products: for any $Q_A \in \mathrm{FI}_A$ and $Q_B \in \mathrm{FI}_B$, we have $Q_A \otimes Q_B \in \mathrm{FI}_{A \sqcup B}$.

---

---

## 4.1 From Lenticular Insight to Axiomatic Foundation

These six axioms we have established are not contrivances — they emerge naturally from the lessons learned from our lenticular coin. Each insight about how perspectives interact translates directly into a mathematical requirement that any reasonable contradiction measure must satisfy.

Let's revisit the key properties we observed:

1. Multiple valid perspectives coexist 
2. Each perspective maintains local consistency
3. Ambiguity inheres in the system itself
4. Universal agreement may prove unattainable
5. Context forms an essential part of any message
6. Contradictions follow discoverable patterns
7. Information coordination carries inherent costs

We’ll now show how together these axioms represent our intuitive understanding of the lenticular coin, translating them into precise mathematical constraints. 

**Multiple Valid Perspectives → Label Invariance (A0)**

- When Nancy sees “YES,” Dylan sees “NO,” and Pat sees “BOTH,” the contradiction doesn’t arise from the particular symbols they observe. The structural impossibility of reconciling their reports persists whether we write (YES, NO, BOTH) or (1, 0, ½) or use any other labeling scheme.
- Axiom A0 captures this: contradiction lives in the *relationships* between perspectives, not in the arbitrary labels we assign to outcomes or contexts.

**Local Consistency → Reduction (A1)**

- Each observer’s individual reports make perfect sense within their own context—Nancy consistently sees “YES” from the left, Dylan consistently sees “NO” from the right. The contradiction emerges only when we attempt to synthesize these locally-coherent perspectives into a single global account.
- Axiom A1 formalizes this: if such a synthesis succeeds (if $P \in \mathrm{FI}$), then no contradiction exists to measure ($K(P) = 0$).

**Inherent Ambiguity → Continuity (A2)**

- As Tyler moves from Nancy’s position toward Dylan’s, the coin’s appearance shifts gradually from “YES” through ambiguous states to “NO.” The contradiction doesn’t jump discontinuously—it evolves smoothly with the changing perspectives. Axiom A2 ensures our measure respects this natural continuity, changing gradually as the underlying perspective structure shifts.

**Unattainable Universal Agreement → Free Operations (A3)**

- No amount of averaging Nancy’s and Dylan’s reports, no coarse-graining of their observations, no random mixing of their contexts can eliminate the fundamental fact that they see different things from their respective positions. The contradiction is structural, not superficial—it cannot be dissolved through information-processing shortcuts.
- Axiom A3 formalizes this irreducibility: operations that merely blur or combine existing information cannot create false reconciliation.

**Context as Essential → Grouping (A4)**

- Whether Nancy states her observation once or repeats it ten times doesn’t change the nature of her disagreement with Dylan. The contradiction isn’t about volume or frequency—it’s about the existence of genuinely distinct contextual perspectives.
- Axiom A4 captures this by making contradiction depend only on which distinct contexts exist, not on how often each appears or how we partition contexts through public randomization.

**Discoverable Patterns → Independent Composition (A5)**

- If Nancy and Dylan disagree about both the coin’s political message *and* its artistic style, these represent two separate contradictions that compound systematically. The total coordination cost must account for both incompatibilities.
- Axiom A5 ensures that contradictions in independent domains add rather than interfere: $K(P \otimes R) = K(P) + K(R)$.

These axioms ensure that any contradiction measure captures what we learned: that some perspective-dependent behaviors carry irreducible coordination costs—costs that cannot be eliminated by relabeling, averaging, or wishful thinking, but can be quantified and predicted.

The resulting unique form $K(P) = -\log_2 α^*(P)$ thus inherits all the conceptual richness of our original insight while providing the mathematical precision needed for information theory. The “contradiction bit” emerges not as an abstract mathematical construct, but as the natural quantification of something we discovered empirically: the fundamental cost of reconciling genuinely incompatible perspectives.

---

## 4.2 Boundaries of the Theory

It’s important to note that this framework applies wherever observations can be modeled as distributions across defined contexts. This includes not only physical experiments but, in principle, also domains like epistemology or cultural conflict.  **The challenge in those areas is not theoretical incompatibility but practical specification: unless contexts and independence can be formalized, $K(P)$ remains undefined rather than violated.

**Core Applicability Conditions**

- **Multiple Sources, Context-Dependent Reports**: Information originates from agents situated in distinct contexts. Each report is tied to the observer’s position, perspective, or frame. Unlike Shannon’s single objective source, contradiction theory begins with many sources that cannot always be reconciled.
- **Finite Contexts and Outcomes**: Reports are modeled as draws from a finite set of outcomes (YES/NO, 0/1, etc.) conditioned on a finite set of contexts. This restriction ensures that contradiction can be precisely defined and measured.
- **Structural Conflict, Not Noise**: Uncertainty here is not statistical randomness alone but the structural incompatibility of perspectives. Even when every report is noiseless and perfectly reliable, contradiction can remain.
- **Observers as Integral, Not Interchangeable**: The identity and position of observers are not incidental. Contradiction arises precisely because different observers, even when equally reliable, cannot always be merged into a single coherent account.
- **No Guaranteed Global Reconstruction**: The goal is not to recover a single objective story, since such a story may not exist. Instead, the goal is to measure the irreducible cost of forcing incompatible perspectives into one frame.
- **Semantics as Central**: Whereas Shannon set meaning aside, contradiction theory takes it as unavoidable. Contradiction is a property of how reports from different frames do or do not cohere, and thus cannot be defined without reference to the semantic relation between contexts.

## 

## Scope

Scope of mathematics:

- We will work in finite discrete settings for clarity; extensions to continuous domains follow naturally.
- Subsequent work applies the framework to real datasets, quantifies contextuality strength across domains, and explores computable proxies for cross-frame description length.

---

# Appendix A — Full proofs and Technical Lemmas

This appendix provides proofs and technical details for the mathematical framework introduced in Section 3. We maintain the intuitive explanations from the main text while supplying complete formal justification for all claims.

Standing Assumptions (finite case). Throughout Appendix A we assume finite outcome alphabets and that $\mathrm{FI}$ (the frame-independent set) is nonempty, convex, compact, and closed under products. These conditions are satisfied in our finite setting by construction (A.1)

## A.1 Formal Setup and Definitions

### A.1.1 Basic Structures

**Definition A.1.1 (Observable System).** Let $\mathcal{X} = \{X_1, \ldots, X_n\}$ be a finite set of observables. For each $x \in \mathcal{X}$, fix a finite nonempty outcome set $\mathcal{O}_x$. A **context** is a subset $c \subseteq \mathcal{X}$. The outcome alphabet for context $c$ is $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$.

**Definition A.1.2 (Behavior).** Given a finite nonempty family $\mathcal{C} \subseteq 2^{\mathcal{X}}$ of contexts, a **behavior** $P$ is a family of probability distributions

$$
P = \{p_c \in \Delta(\mathcal{O}_c) : c \in \mathcal{C}\}
$$

where $\Delta(\mathcal{O}_c)$ denotes the probability simplex over $\mathcal{O}_c$.

**Remark (No nondisturbance required).** We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in A.1.4. When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.

**Definition A.1.3 (Deterministic Global Assignment).** Let $\mathcal{O}_{\mathcal{X}} := \prod_{x \in \mathcal{X}} \mathcal{O}_x$. A **deterministic global assignment** is an element $s \in \mathcal{O}_{\mathcal{X}}$. It induces a deterministic behavior $q_s$ by restriction:

$$
q_s(o \mid c) = \begin{cases} 1 & \text{if } o = s|_c \\ 0 & \text{otherwise} \end{cases}
$$

for each context $c \in \mathcal{C}$ and outcome $o \in \mathcal{O}_c$.

**Definition A.1.4 (Frame-Independent Set).** The **frame-independent set** is

$$
\text{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\} \subseteq \prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)
$$

**Proposition A.1.5 (Alternative Characterization of FI).** $Q \in \text{FI}$ if and only if there exists a **global law** $\mu \in \Delta(\mathcal{O}_{\mathcal{X}})$ such that

$$
q_c(o) = \sum_{s \in \mathcal{O}_{\mathcal{X}} : s|_c = o} \mu(s) \quad \forall c \in \mathcal{C}, o \in \mathcal{O}_c
$$

**Proof.** The forward direction is immediate from the definition of convex hull. For the reverse direction, given $\mu$, define $Q$ by the displayed formula. Then $Q$ is a convex combination of the deterministic behaviors $\{q_s\}$ with weights $\{\mu(s)\}$, hence $Q \in \text{FI}$. □

### A.1.2 Basic Properties of FI

**Proposition A.1.6 (Topological Properties).** The frame-independent set FI is nonempty, convex, and compact.

**Proof.**

- **Nonempty**: Contains all deterministic behaviors $q_s$.
- **Convex**: By definition as a convex hull.
- **Compact**: FI is a finite convex hull in the finite-dimensional space $\prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)$, hence a polytope, hence compact. □

**Definition A.1.7 (Context simplex).** $\Delta(\mathcal{C}) := \{\lambda \in \mathbb{R}^{\mathcal{C}} : \lambda_c \geq 0, \sum_{c \in \mathcal{C}} \lambda_c = 1\}$.

**Proposition A.1.8 (Product Structure).** Let $P$ be a behavior on $(\mathcal{X}, \mathcal{C})$ and $R$ be a behavior on $(\mathcal{Y}, \mathcal{D})$ with $\mathcal{X} \cap \mathcal{Y} = \emptyset$ (we implicitly relabel so disjointness holds). For distributions $p \in \Delta(\mathcal{O}_c)$ and $r \in \Delta(\mathcal{O}_d)$ on disjoint coordinates, $p \otimes r \in \Delta(\mathcal{O}_c \times \mathcal{O}_d)$ is $(p \otimes r)(o_c, o_d) = p(o_c)r(o_d)$.

Define the product behavior $P \otimes R$ on $(\mathcal{X} \sqcup \mathcal{Y}, \mathcal{C} \otimes \mathcal{D})$ where $\mathcal{C} \otimes \mathcal{D} := \{c \cup d : c \in \mathcal{C}, d \in \mathcal{D}\}$ by

$$
(p \otimes r)(o_c, o_d \mid c \cup d) = p(o_c \mid c) \cdot r(o_d \mid d)
$$

Then:

1. If $Q \in \text{FI}_{\mathcal{X},\mathcal{C}}$ and $S \in \text{FI}_{\mathcal{Y},\mathcal{D}}$, then $Q \otimes S \in \text{FI}_{\mathcal{X} \sqcup \mathcal{Y}, \mathcal{C} \otimes \mathcal{D}}$.
2. For deterministic assignments, $q_s \otimes q_t = q_{s \sqcup t}$.

**Proof.**

1. If $Q$ arises from global law $\mu$ and $S$ arises from global law $\nu$, then $Q \otimes S$ arises from the product global law $\mu \otimes \nu$ on $\mathcal{O}_{\mathcal{X} \sqcup \mathcal{Y}}$. From $q_s \otimes q_t = q_{s \sqcup t}$, it follows that
    
    $$
    (\sum_s \mu_s q_s)\otimes(\sum_t \nu_t q_t)=\sum_{s,t}\mu_s\nu_t\,(q_s\otimes q_t)=\sum_{s,t}\mu_s\nu_t\,q_{s\sqcup t}\in \mathrm{conv}\{q_{s\sqcup t}\}.
    $$
    
2. Direct verification from definitions: $q_s \otimes q_t = q_{s \sqcup t}$ because $\delta_{s|_c} \otimes \delta_{t|_d} = \delta_{(s \sqcup t)|_{c \cup d}}$. □

**Definition A.2.1 (Bhattacharyya Coefficient).** For probability distributions $p, q \in \Delta(\mathcal{O})$ on a finite alphabet $\mathcal{O}$:

$$
\text{BC}(p, q) := \sum_{o \in \mathcal{O}} \sqrt{p(o) q(o)}
$$

**Lemma A.2.2 (Bhattacharyya Properties).** For distributions $p, q \in \Delta(\mathcal{O})$:

1. **Range**: $0 \leq \text{BC}(p, q) \leq 1$
2. **Perfect agreement**: $\text{BC}(p, q) = 1 \Leftrightarrow p = q$
3. **Joint concavity**: Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$. Therefore $(x,y)\mapsto\sqrt{xy}$ is jointly concave on $\mathbb{R}_{\geq 0}^2$ (extend by continuity on the boundary). Summing over coordinates preserves concavity, so $\text{BC}$ is jointly concave on $\Delta(\mathcal{O})\times\Delta(\mathcal{O})$.
4. **Product structure**: $\text{BC}(p \otimes r, q \otimes s) = \text{BC}(p, q) \cdot \text{BC}(r, s)$

**Proof.**

(1) *Range.* Nonnegativity is obvious. For the upper bound, by Cauchy-Schwarz:

$$
\text{BC}(p, q) = \sum_o \sqrt{p(o) q(o)} \leq \sqrt{\sum_o p(o)} \sqrt{\sum_o q(o)} = 1
$$

(2) *Perfect agreement.* The Cauchy-Schwarz equality condition gives $\text{BC}(p, q) = 1$ iff $\sqrt{p(o)}$ and $\sqrt{q(o)}$ are proportional, i.e., $\frac{\sqrt{p(o)}}{\sqrt{q(o)}}$ is constant over $\{o : p(o) q(o) > 0\}$. Since both are probability distributions, this constant must be 1, giving $p = q$.

(3) *Joint concavity.* Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$; extend to $\mathbb{R}_{\geq 0}^2$ by continuity. Summing over coordinates preserves concavity.

(4) *Product structure.* Expand the tensor product and factor the sum. □

## A.3 The Agreement Measure and Minimax Theorem

**Definition A.3.1 (Agreement and Contradiction).** For a behavior $P$:

$$
\alpha^\star(P) := \max_{Q \in \text{FI}} \min_{c \in \mathcal{C}} \text{BC}(p_c, q_c)
$$

$$
K(P) := -\log_2 \alpha^\star(P)
$$

**Theorem A.3.2 (Minimax Equality).** Define the payoff function

$$
f(\lambda, Q) := \sum_{c \in \mathcal{C}} \lambda_c \text{BC}(p_c, q_c)
$$

for $\lambda \in \Delta(\mathcal{C})$ and $Q \in \text{FI}$. Then:

$$
\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \text{FI}} f(\lambda, Q)
$$

Maximizers/minimizers exist by compactness and continuity of $f$.

**Proof.** We apply Sion's minimax theorem (M. Sion, *Pacific J. Math.* **8** (1958), 171–176). We need to verify:

1. $\Delta(\mathcal{C})$ and FI are nonempty, convex, and compact ✓
2. $f(\lambda, \cdot)$ is concave on FI for each $\lambda$ ✓
3. $f(\cdot, Q)$ is convex (actually linear) on $\Delta(\mathcal{C})$ for each $Q$ ✓

**Details:**

- Compactness of $\Delta(\mathcal{C})$: Standard simplex.
- Compactness of FI: Proposition A.1.6.
- Concavity in $Q$: Since $Q \mapsto (q_c)_{c \in \mathcal{C}}$ is affine and each $\text{BC}(p_c, \cdot)$ is concave (Lemma A.2.2.3), the nonnegative linear combination $\sum_c \lambda_c \text{BC}(p_c, q_c)$ is concave in $Q$.
- Linearity in $\lambda$: Obvious from the definition.

By Sion's theorem, $\min_\lambda \max_Q f(\lambda, Q) = \max_Q \min_\lambda f(\lambda, Q)$.

It remains to show this common value equals $\alpha^\star(P)$. For any $Q \in \text{FI}$, let $a_c := \text{BC}(p_c, q_c)$. Then:

$$
\min_{\lambda \in \Delta(\mathcal{C})} \sum_{c} \lambda_c a_c = \min_{c \in \mathcal{C}} a_c
$$

with the minimum achieved by $\lambda$ supported on $\arg\min_c a_c$.

Therefore:

$$
\max_{Q \in \text{FI}} \min_{\lambda \in \Delta(\mathcal{C})} f(\lambda, Q) = \max_{Q \in \text{FI}} \min_{c \in \mathcal{C}} \text{BC}(p_c, q_c) = \alpha^\star(P)
$$

and hence the common value equals $\alpha^\star(P)$. This completes the proof. □

## A.4 Bounds and Characterizations

**Lemma A.4.1 (Uniform Law Lower Bound).** For any behavior $P$:

$$
\alpha^\star(P) \geq \min_{c \in \mathcal{C}} \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

**Proof.** Let $\mu$ be the uniform (counting-measure) distribution on $\mathcal{O}_{\mathcal{X}}$ (so each global state is equally likely). This induces $Q^{\text{unif}} \in \text{FI}$ with uniform **context marginals**: $q_c^{\text{unif}}(o) = \frac{1}{|\mathcal{O}_c|}$ for all $c \in \mathcal{C}$, $o \in \mathcal{O}_c$.

For any context $c$:

$$
\text{BC}(p_c, q_c^{\text{unif}}) = \sum_{o \in \mathcal{O}_c} \sqrt{p_c(o) \cdot \frac{1}{|\mathcal{O}_c|}} = \frac{1}{\sqrt{|\mathcal{O}_c|}} \sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)}
$$

The function $\sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)}$ is concave on the simplex $\Delta(\mathcal{O}_c)$, so its minimum is attained at a vertex (a point mass), where the sum equals 1. Therefore:

$$
\sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)} \geq 1
$$

This minimum is achieved when $p_c$ is a point mass. Therefore:

$$
\text{BC}(p_c, q_c^{\text{unif}}) \geq \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

Since $\alpha^\star(P) \geq \min_{c} \text{BC}(p_c, q_c^{\text{unif}})$, the result follows. □

**Corollary A.4.2 (Bounds on K).** For any behavior $P$:

$$
0 \leq K(P) \leq \frac{1}{2} \log_2 \left(\max_{c \in \mathcal{C}} |\mathcal{O}_c|\right)
$$

**Proof.** The lower bound follows from $\alpha^\star(P) \leq 1$. The upper bound follows from Lemma A.4.1 and the fact that $-\log_2(x^{-1/2}) = \frac{1}{2}\log_2(x)$. □

**Theorem A.4.3 (Characterization of Frame-Independence).** For any behavior $P$:

$$
\alpha^\star(P) = 1 \Leftrightarrow P \in \text{FI} \Leftrightarrow K(P) = 0
$$

**Proof.**

($\Rightarrow$) If $\alpha^\star(P) = 1$, then there exists $Q \in \text{FI}$ such that $\min_c \text{BC}(p_c, q_c) = 1$. This implies $\text{BC}(p_c, q_c) = 1$ for all $c \in \mathcal{C}$. By Lemma A.2.2, this gives $p_c = q_c$ for all $c$, hence $P = Q \in \text{FI}$.

($\Leftarrow$) If $P \in \text{FI}$, take $Q = P$ in the definition of $\alpha^\star(P)$. Then $\min_c \text{BC}(p_c, q_c) = \min_c \text{BC}(p_c, p_c) = 1$.

The equivalence with $K(P) = 0$ follows from the definition $K(P) = -\log_2 \alpha^\star(P)$. □

**Remark (No nondisturbance required).** We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in A.1.4. When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.

## A.5 Duality Structure and Optimal Strategies

**Theorem A.5.1 (Minimax Duality).** Let $(\lambda^\star, Q^\star)$ be optimal strategies for the minimax problem in Theorem A.3.2. Then:

1. $f(\lambda^\star, Q^\star) = \alpha^\star(P)$
2. $\text{supp}(\lambda^\star) \subseteq \{c \in \mathcal{C} : \text{BC}(p_c, q_c^\star) = \alpha^\star(P)\}$
3. If $\lambda^\star_c > 0$, then $\text{BC}(p_c, q_c^\star) = \alpha^\star(P)$

**Proof.** Existence of optimal strategies follows from compactness and continuity.

1. This is immediate from the minimax equality.
2. & 3. For fixed $Q^\star$, the inner optimization $\min_{\lambda \in \Delta(\mathcal{C})} \sum_c \lambda_c a_c$ with $a_c := \text{BC}(p_c, q_c^\star)$ has value $\min_c a_c$ and optimal solutions supported on $\arg\min_c a_c$. Since $(\lambda^\star, Q^\star)$ is optimal for the full problem, $\lambda^\star$ must be optimal for this inner problem, giving the result. □

## References for Appendix A

- Sion, M. "On general minimax theorems."*Pacific Journal of Mathematics* 8.1 (1958): 171-176.
- Bhattacharyya, A. "On a measure of divergence between two statistical populations defined by their probability distributions."*Bulletin of the Calcutta Mathematical Society* 35 (1943): 99-109.

---

# Appendix B — Worked Examples

## **B.1 Worked Example: The Irreducible Perspective: Carrying the Frame.**

Let's consider an example. Let's say there are three friends—*Nancy*, *Dylan*, *Tyler*—watch the coin from three seats: $\text{LEFT}$, $\text{MIDDLE}$, and $\text{RIGHT}$. After each flip they copy only the three words they saw, comma-separated, into a shared notebook. Each knew where they sat, and the order the games went, so they didn't think to write the positions.

Notebook

| **Flip** | **Observer Reports** |
| --- | --- |
| 1 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |
| 2 | $\text{NO}$,$\text{BOTH}$,$\text{YES}$ |
| 3 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |
| 4 | $\text{NO}$,$\text{BOTH}$,$\text{YES}$ |
| 5 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |

Now, using only this notebook—can you tell who sat where? You can't. (Unless you're grading your own work, in which case: 10/10, excellent deduction).

It's because "$\text{YES}$, $\text{BOTH}$, $\text{NO}$" fits two different worlds on every line:

- On $\text{HEADS}$ the left seat is $\text{YES}$, the middle is $\text{BOTH}$, the right is $\text{NO}$;
- On $\text{TAILS}$ the left is $\text{NO}$, the middle is $\text{BOTH}$, the right is $\text{YES}$.

Strip names and seats, and those worlds collapse to the same record. You are exactly one yes/no short: "who wrote $\text{YES}$—left or right?" Formally, each report depends on two coordinates: the coin's face $S\in\{\text{HEADS},\text{TAILS}\}$ and the viewpoint $P\in\{L,M,R\}$. 

The observation $O\in\{\text{YES},\text{NO},\text{BOTH}\}$ is

$$
O(S,P)= \begin{cases} \text{BOTH}, & P=M,\\ \text{YES}, & (S,P)\in\{(\text{HEADS},L),(\text{TAILS},R)\},\\ \text{NO}, & (S,P)\in\{(\text{HEADS},R),(\text{TAILS},L)\}. \end{cases}
$$

The smoking gun appears when we hide $P$: distinct worlds land on the same word. 

- $(\text{HEADS},L)$ and $(\text{TAILS},R)$ both read $\text{YES}$;
- $(\text{HEADS},R)$ and $(\text{TAILS},L)$ both read $\text{NO}$.

There is no frame-independent reconstruction from $O$ alone. Every crisp report owes one more yes/no if you want the scene to be recoverable. Carry $P$, and the correlations are captured perfectly; drop it, and two different worlds become indistinguishable on the page.

Now quantify the coordination cost of omitting $P$. Suppose seats are used equally and the coin is fair, so $\text{YES}$, $\text{BOTH}$, $\text{NO}$ each appear about one-third of the time in the notebook.

- Seeing $\text{BOTH}$ pins $P=M$: no uncertainty remains (0 missing bits).
- Seeing $\text{YES}$ or $\text{NO}$ tells you only "not the middle," leaving a binary uncertainty—left or right (1 missing bit).

Averaging,

$$
\text{Missing}=\tfrac13\cdot 0+\tfrac13\cdot 1+\tfrac13\cdot 1=\tfrac23\ \text{bits per line}.
$$

In Shannon's notation, with $H(P)=\log_2 3$,

$$
I(O;P)=H(P)-H(P\mid O)=\log_2 3-\tfrac23\approx 0.918\ \text{bits},\qquad \boxed{H(P\mid O)=\tfrac23\ \text{bits per line}}.
$$

Plainly: unless you carry the frame, you are on average two-thirds of a bit short of knowing where each observer sat. Thus, the cost is small, but persistent. 

It is not the one-off surprise of model identification; it is a steady coordination cost dictated by the object's geometry. If you also record the coin face $S$ alongside $O$, then together $(S,O)$ identify the seat exactly, driving this residual to zero—but at the price of carrying additional information with every observation. The lesson is not that perspective is noise to be averaged out, but that it is structure to be carried. 

Keep the frame and the story is coherent. Drop it, and worlds become epistemically indistinct.

## B.2 The Lenticular Coin / Odd–Cycle Example

**Example B.2.1 (Odd-Cycle Contradiction).**

We model Nancy (N), Dylan (D), and Tyler (T) as three **binary** observables $X_N,X_D,X_T\in\{0,1\}$ encoding $\mathrm{YES}=1$, $\mathrm{NO}=0$. The three **contexts** are the pairs

$$
c_1=\{N,T\},\qquad c_2=\{T,D\},\qquad c_3=\{D,N\}
$$

The observed behavior $P=\{p_{c_i}\}_{i=1}^3$ is the **"perfect disagreement"** behavior on each edge:

$$
p_{c}(1,0)=p_{c}(0,1)=\tfrac12,\qquad p_{c}(0,0)=p_{c}(1,1)=0\quad\text{for }c\in\{c_1,c_2,c_3\}
$$

Equivalently: each visible pair always disagrees, but the direction of disagreement is uniformly random.

We compute

$$
\alpha^\star(P)\;=\;\max_{Q\in\mathrm{FI}}\ \min_{c\in\{c_1,c_2,c_3\}} \mathrm{BC}(p_c,q_c)
$$

and show $\alpha^\star(P)=\sqrt{\tfrac23}$, hence $K(P)=\tfrac12\log_2\!\frac32$.

### B.2.1 Universal Upper Bound: $\alpha^\star\le \sqrt{2/3}$

**Lemma B.2.2 (Upper Bound).** Let $Q\in\mathrm{FI}$ arise from a global law $\mu$ on $\{0,1\}^3$. For a context $c=\{i,j\}$, write the induced pair distribution

$$
q_c(00),\ q_c(01),\ q_c(10),\ q_c(11)\quad\text{and}\quad
D_{ij}:=q_c(01)+q_c(10)=\Pr_{\mu}[X_i\ne X_j]
$$

For our $p_c$ (uniform over the off-diagonals), the Bhattacharyya coefficient is

$$
\mathrm{BC}(p_c,q_c)
=\sqrt{\tfrac12}\big(\sqrt{q_c(01)}+\sqrt{q_c(10)}\big)
\le \sqrt{D_{ij}}
$$

with equality iff $q_c(01)=q_c(10)=D_{ij}/2$ (by concavity of $\sqrt{\cdot}$ at fixed sum).

Thus

$$
\min_{c}\ \mathrm{BC}(p_c,q_c)\ \le\ \min_{c}\ \sqrt{D_c}
$$

The triple $(D_{NT},D_{TD},D_{DN})$ must be feasible as edge-disagreement probabilities of a joint $\mu$ on $\{0,1\}^3$. For three bits, every deterministic assignment has either 0 disagreements (all equal) or exactly 2 (one bit flips against the other two). Hence any convex combination obeys the **cut-polytope constraint**

$$
D_{NT}+D_{TD}+D_{DN}\ \le\ 2
$$

Consequently at least one edge has $D_c\le 2/3$, so

$$
\min_c \sqrt{D_c}\ \le\ \sqrt{2/3}
$$

Taking the maximum over $Q\in\mathrm{FI}$ yields the universal upper bound

$$
\alpha^\star(P)\ \le\ \sqrt{2/3}
$$

### B.2.2 Achievability: An Explicit Optimal $\mu^\star$

**Proposition B.2.3 (Achievability).** Let $\mu^\star$ be the **uniform** distribution over the six nonconstant bitstrings:

$$
\mu^\star=\text{Unif}\big(\{100,010,001,011,101,110\}\big)
$$

(Equivalently: put zero mass on $000$ and $111$, equal mass on Hamming-weight $1$ and $2$ states.)

A direct check shows that for any edge $c\in\{c_1,c_2,c_3\}$,

$$
q_c^\star(01)=q_c^\star(10)=\tfrac13,\qquad q_c^\star(00)=q_c^\star(11)=\tfrac16
$$

so $D_c^\star=q_c^\star(01)+q_c^\star(10)=\tfrac23$ and the off-diagonals are **balanced**, hence the BC upper bound is tight:

$$
\mathrm{BC}(p_c,q_c^\star)
=\sqrt{\tfrac12}\big(\sqrt{\tfrac13}+\sqrt{\tfrac13}\big)
=\sqrt{\tfrac23}\quad\text{for each }c
$$

Therefore

$$
\min_{c}\mathrm{BC}(p_c,q_c^\star)=\sqrt{\tfrac23}
$$

which matches the upper bound. We conclude

$$
\boxed{\ \alpha^\star(P)=\sqrt{\tfrac23}\ ,\qquad
K(P)=-\log_2\alpha^\star(P)=\tfrac12\log_2\!\frac32\ }
$$

**Corollary B.2.4 (Optimal Witness).** By symmetry, any optimal contradiction witness $\lambda^\star$ can be taken **uniform** on the three contexts. Moreover, the equalization $\mathrm{BC}(p_c,q_c^\star)=\alpha^\star$ on all edges shows $\lambda^\star$ may place positive mass on each context (cf. the support condition in the minimax duality).




## 1.2 Our Contribution: Axiomatizing the Impossible

We develop a reconciliation calculus for contexts (frames) and show that there is an essentially unique (under our axioms) scalar capturing the informational cost of enforcing one story across incompatible contexts.

1. **Axiomatic characterization (inevitability)**. 
We prove that, under axioms A0–A5, the essentially unique contradiction measure is 
    
    $$
    K(P)=-\log_2 \alpha^\star(P)
    $$
    
    where 
    
    $$
    \alpha^\star(P)=\max_{Q\in \mathrm{FI}}\ \min_{c}\ \mathrm{BC}\!\big(p_c, q_c\big)
    $$
    
    Here $\mathrm{FI}$ is the convex set of frame-independent behaviors (the "unified story" polytope for finite alphabets). In quantum settings, $\mathrm{FI}$ coincides with non-contextual models, yielding a principled violation strength. (Theorems 2–4; aggregator lemma; Theorem 1)
    
2. **Agreement-kernel uniqueness.** 
Assuming refinement separability, product multiplicativity, and a data-processing inequality, we show the per-context agreement is uniquely the Bhattacharyya affinity. (Theorem 3)
3. **Well-posedness & calibrated zero.** 
For finite alphabets with $\mathrm{FI}$ nonempty/compact/convex/product-closed, the program for $\alpha^\star(P)$ attains an optimum with $\alpha^\star(P)\in[0,1]$; thus $K(P)\geq 0$ and $K(P)=0$ iff $P\in\mathrm{FI}$. This establishes an absolute zero and a stable scale. (Proposition family)
4. **Resource laws.** 
We prove additivity $K(P\otimes R)=K(P)+K(R)$ and monotonicity under free operations (post-processing, outcome-independent context mixing, convex mixing, adding $\mathrm{FI}$ ancillas). (Theorem 5 + corollaries)
5. **Operational triad from one overlap.** 
The same $\alpha^\star$ yields, under standard conditions (finite alphabets, i.i.d. sampling): (i) discrimination error exponents for testing real vs. simulated behavior, (ii) simulation overheads—the "contradiction tax"—to imitate multi-context data, and (iii) prediction lower bounds (irreducible regret when restricted to an $\mathrm{FI}$ model). (Theorems 6–8)
6. **Computability & estimation.** 
We provide a practical minimax/convex program (with column-generation option) for $\alpha^\star$, plus a consistent plug-in estimator for $K$ from empirical frequencies with bootstrap CIs; A reference implementation (*contrakit*) accompanies the paper.
7. **Specialization to quantum contextuality**. 
With $\mathrm{FI} =$  the non-contextual set, $K$ is a contextuality monotone (zero iff non-contextual; monotone under free ops; additive). (Theorem 9)

When one story won’t fit, we measure the seam.

## 1.3 Structure and Scope

The paper moves from motivation → mechanism → consequences.

While **§§1–2** motivate the need for a single yardstick and ground it in a concrete device; **§§3–5** build the calculus; **§6** gives task-level consequences; **§7** makes it implementable; **§§8–10** place, bound, and extend the results; while the appendices supply proofs and worked cases.

**Overview:**

- Motivation by example (**§2**). The Lenticular Coin is a minimal classical device exhibiting odd-cycle incompatibility; it previews how $K(P)$ registers contradiction in bits.
- Framework → axioms → results (**§§3–5**). **§3** formalizes observables, contexts (frames), behaviors, the baseline $\mathrm{FI}$, Bhattacharyya overlap, and the minimax program (with standing assumptions). **§4** states and motivates axioms A0–A5. **§5** presents the main theorems, including the fundamental formula $K(P)=-\log_2 \alpha^{\star}(P)$ and additivity.
- From quantity to consequences (**§6**) and practice (**§7**). **§6** develops operational theorems in discrimination (error exponents), simulation (overheads; the contradiction tax), and prediction (regret). **§7** provides a practical minimax/convex program for $\alpha^{\star}$, a plug-in estimator for $K$ with bootstrap intervals, and a brief note on the reference implementation (*contrakit*).
- Context and boundaries (**§§8–10**; **App**.). **§8** positions the work relative to contextuality, Dempster–Shafer, social choice, and information distances. **§9** states limitations and scope. **§10** sketches near-term extensions. 

**App. A** contains full proofs and technical lemmas; **App. B** holds worked examples, and **App. C** offers case studies for review.

**How to read:**

- For guarantees, skim **§2**, then read **§§3–5** for the formal core and **§6** for operational meaning; see **App. A** for proofs.
- For implementation, jump from **§2** to **§§6–7** (algorithms, estimation), backfilling definitions from **§3** as needed; see **App. B** for worked cases.

**Scope:**

Throughout we assume finite alphabets and the usual compatibility/no-disturbance setting. The baseline $\mathrm{FI}$ is nonempty, compact, convex, and product-closed—hence a polytope for finite alphabets. Results are domain-general in the following sense: once $\mathrm{FI}$ is specified for a given domain, the same scalar $K(P)$ applies without modification, with a calibrated zero ($K=0$ iff $P\in \mathrm{FI}$) and shared units across cases.

---

# 2. Building Natural Intuition: The Lenticular Coin

> *"What we observe is not nature itself but nature exposed to our method of questioning."

— Werner Heisenberg, Physics and Philosophy: The Revolution in Modern Science*
> 

Here, "contradiction" doesn't refer to the classical logical impossibility ($A$ and $¬A$), but rather to *epistemic incompatibility*: when two equally valid observations cannot be reconciled within a single reference frame ($A@X ∧ B@Y$ where $X$ and $Y$ represent incompatible contexts). This is similar to special relativity, where two observers can measure different times for the same event—and both are correct—because the reference frame fundamentally matters. We can consider special relativity: two clocks read different times for the same event and both are right—*because frame does the real work*. 

Lenticular images make this tactile. Tilt a postcard: **from one angle you see one picture; from another, a different one. The substrate doesn’t change—your perspective does.** 

If we apply that to a fair coin, then like Shannon’s coin, it has two sides and we flip it at random. Unlike Shannon’s coin however, each face is printed lenticularly so that what you see, depends on the viewing angle. We put the coin on the table with each face lenticularly printed, so the message you see depends on where you stand. We flip the coin. Since one person stands to the left, and the other to the right, when the coin lands, the left observer sees YES and the right observer sees NO; on the next flip those roles swap. 

When they compare notes, they’ll always disagree:

| Coin Side | LEFT Observer Sees | RIGHT Observer Sees |
| --- | --- | --- |
| HEADS | YES | NO |
| TAILS | NO | YES |

This is intuitive, that isn’t a mistake or noise; it’s baked into the viewing geometry. What happened depends on where you looked.

Formally we’d say: 

Let $S\in{\text{HEADS},\text{TAILS}}$ be the face up, $P$ the viewpoint (e.g., $\text{LEFT}$ or $\text{RIGHT}$), and let $O(S,P)\in{\text{YES},\text{NO}}$ be the visible message. 

By design,

$$
O(S,P)=
\begin{cases}
\text{YES}, & (S,P)\in\{(\text{HEADS},\text{LEFT}),(\text{TAILS},\text{RIGHT})\},\\
\text{NO},  & (S,P)\in\{(\text{HEADS},\text{RIGHT}),(\text{TAILS},\text{LEFT})\}.
\end{cases}
$$

We commence each trial as follows: flip the coin (fair, $1/2$–$1/2$), both observers record what they see, then compare notes. They always disagree. From either seat alone, the sequence looks like a fair binary source. Jointly, the outcomes are perfectly anti-correlated. While it remains true that what happens depended on where you were, this version still admits a single global description once we include $P$ in the state: the device implements a fixed rule ("$\text{LEFT}$ shows the opposite of $\text{RIGHT}$, with flip swapping roles"). 

Thus, this is anti-correlation, not an irreconcilable contradiction.

---

## 2.1 Model Identification ≠ Perspectival Information

Learning the device's rule is genuine information; after it's known, the per-flip fact of "we disagree" carries no further surprise—it is exactly what the rule predicts. Before you discover the rule, several live hypotheses compete (e.g., "always-same," "always-opposite," "independent"). Observing outcomes drains that model uncertainty. That is not the irreducible information we speak on here.

**Formally**: if $M$ denotes which rule is true and $D_{1:k}$ the first $k$ observations, the information gained about the rule is the drop in uncertainty: 

$$
I(M; D_{1:k}) \;=\; H(M)\;-\;H\!\left(M\mid D_{1:k}\right).
$$

With a uniform prior over the three hypotheses, two consecutive "opposite" outcomes yield the posterior $(0.8,\,0.2,\,0)$ (in the order "always-opposite," "independent," "always-same"), cutting entropy from $\log_2 3 \approx 1.585$ bits to about $0.722$ bits. You are learning—but thereafter each new row shaves off less and less.

Intuitively: the surprise lives in discovering the rule. Once your posterior has essentially collapsed, "we disagree—again" is confirmation, not news. Each flip still tells you which joint outcome happened—that's one bit about the event—but it no longer tells you anything fresh about the governing rule. So the first Lenticular Coin sits at the *model-identification layer*: you infer the rule that governs the observations. 

That is standard Shannon/Bayesian territory—useful, but not yet our target notion. It shows that perspective changes what you see, not what is true: there is a single global rule, simply viewed from different seats. 

Once viewpoint is modeled within the state, **one law explains everything**.

1. $\text{LEFT}$ and $\text{RIGHT}$ always disagree;
2. $\text{HEADS}$ → $\text{LEFT}$ says $\text{YES}$ ($\text{RIGHT}$ says $\text{NO}$),
3. $\text{TAILS}$ → $\text{RIGHT}$ says $\text{YES}$ ($\text{LEFT}$ says $\text{NO}$).

The law is more than a lookup table; it is the rule everyone follows when turning what they see into a report. Given a state $S$ and a seat $P$, the law fixes which word must be written down. In information-theoretic terms, it is the channel $p(o\mid s,p)$; in plain terms, it is the shared reporting language that makes my "$\text{YES}$" mean the same thing as your "$\text{YES}$".

This matters because, once the law is fixed, **records should cohere**: different seats can yield different entries, but all entries are expected to fit under the same rule. We will use this distinction shortly.

To continue, we use a mundane feature of lenticular media: the transition band. It introduces a lawful "both" outcome—legitimate ambiguity—where "what happened" begins to blur. This is where a frame-independent summary begins to fail unless the context label is carried along; the reports remain consistent, but the summary without frames does not. 

This pressure toward contradiction will become explicit in §2.3.

---

## 2.2 The Lenticular Coin: the Natural “Both” Band

The first coin taught us a rule: $\text{LEFT}$ and $\text{RIGHT}$ must disagree. This constituted genuine learning—a discovery that reduces informational uncertainty as you understand how the device operates. After the rule is known, each flip merely confirms expectation. 

To show the persisting structure we care about when we say “frame”, we only need to acknowledge a mundane physical fact about lenticular media: there is a transition band where both layers are simultaneously visible. That band is not an error; it is part of the object. Place the coin as before, but mark three viewing positions: $\text{LEFT}$, $\text{MIDDLE}$, and $\text{RIGHT}$. 

Each face is printed lenticularly so being positioned at $\text{LEFT}$ cleanly shows $\text{YES}$, $\text{RIGHT}$ cleanly shows $\text{NO}$, and being at $\text{MIDDLE}$ shows natural transition band where both overlays are visibly present. When the coin flips from $\text{HEADS}$ to $\text{TAILS}$, the clean views swap ($\text{YES}$↔$\text{NO}$), yet the $\text{MIDDLE}$ never changes, always showing $\text{BOTH}$.

**Formally:**

For face $S\in\{\text{HEADS},\text{TAILS}\}$ and position $P\in\{\text{LEFT},\text{MIDDLE},\text{RIGHT}\}$, the observation $O$ satisfies

$$
O(S,P)=
\begin{cases}
\text{BOTH}, & P=\text{MIDDLE},\\
\text{YES},  & (S,P)\in\{(\text{HEADS},\text{LEFT}),(\text{TAILS},\text{RIGHT})\},\\
\text{NO},   & (S,P)\in\{(\text{HEADS},\text{RIGHT}),(\text{TAILS},\text{LEFT})\}.
\end{cases}

$$

Nothing metaphysical is hiding here; this is just a postcard effect, elevated to a protocol.

However, two things now become unavoidable:

1. **Ambiguity is intrinsic**. a competent observer at $\text{MIDDLE}$ can truthfully report $\text{BOTH}$; that outcome is lawful, not noise.
2. **Perspective becomes a per-trial budget**. reports are reproducible only if the viewing frame travels with the message. "I saw $\text{YES}$" is underspecified; "I saw $\text{YES}$ from $\text{LEFT}$" is reconstructible.

Put differently, **with three seats the law is now *context-indexed***. For a fixed seat $P$:

- $P=\text{LEFT}$: $\text{YES}$ on $\text{HEADS}$, $\text{NO}$ on $\text{TAILS}$.
- $P=\text{RIGHT}$: $\text{NO}$ on $\text{HEADS}$, $\text{YES}$ on $\text{TAILS}$ (the inverse of $\text{LEFT}$).
- $P=\text{MIDDLE}$: $\text{BOTH}$ on both flips (constant).

As a consequence, you cannot tell the full story unless **you model** $P$. There is a small but steady information loss—about  $\frac{2}{3}$ of a bit per record (App B.1)—if you drop the frame. It’d be no different than asking ‘did they break the law?’ without saying where it happened.

Run the experiment for many flips and this structure shows up in plain statistics: $\text{LEFT}$ and $\text{RIGHT}$ disagree predictably; the $\text{MIDDLE}$ registers a stable experience of $\text{BOTH}$ events; and the frame labels are continually required to reconcile otherwise incompatible yet honest reports. The disagreement is no longer just "they always oppose" (a rule you learn once).

The extra content is small, but it never goes away. It is not the one-off surprise of model identification; it is a steady coordination cost—bits you must carry every time if downstream agreement is the goal. 

This is to build an intuition on perspective: This is to build an intuition on perspective: the frame itself is information, and while not entirely new, it’s modeled far less often than it should be. Shannon’s model doesn’t forbid modeling frames; it simply doesn’t quantify incompatibility across contexts. This is not contradiction yet: the reports are consistent—but we needed to show this distinction. 

We show this to distinctly separate information loss from dropping frames (priced by $H(P\mid O)$, here $\tfrac{2}{3}$ bit/record) from structural contradiction across frames (priced by $K(P)$), so readers won't conflate "forgot the label" with "no single story fits."

---

## **2.3 The Lenticular Coin’s Irreducible Contradiction**

Having built intuition around perspective and missing information, we finally now arrive to the paper’s purpose: a type of contradiction that persists even when context is fully preserved and the setup is completely transparent. This time we disallow ambiguity.

This time we disallow ambiguity.

> Axiom (Single-Valued Reports).
> 
> 1. Each observer must report a single word: YES or NO. 
> 2. No BOTH entries allowed.

Consider the same lenticular coin, now mounted on a motorized turntable that rotates in precise increments. Three observers—Nancy, Dylan, Tyler—sit at fixed viewing angles along the viewing axis. The lenticular surface shows $\text{YES}$ at $0^\circ$, $\text{NO}$ at $180^\circ$, and $\text{BOTH}$ at the transition $90^\circ$ (with half-width $w_{\text{both}}$). Fix three platform orientations that graze but avoid the transition band:

$$
\psi_1=0^\circ,\qquad \psi_2=90^\circ-\varepsilon,\qquad \psi_3=180^\circ-\varepsilon,\quad \\ \text{with }\varepsilon > w_{\text{both}}+\delta.
$$

At each orientation $\psi_k$, the platform stops; exactly two observers look while the third looks away (no omniscient snapshot). 

![image.png](attachment:957f05af-b878-46e7-ad8d-8769853d20df:image.png)

The three rounds:

Round	             Orientation $\psi$	 Observers	      Reports

| 1 | $0^\circ$ | Nancy, Tyler | $\text{YES}$,$\text{NO}$ |
| --- | --- | --- | --- |
| 2 | $90^\circ-\varepsilon$ | Tyler, Dylan | $\text{NO}$,$\text{YES}$ |
| 3 | $180^\circ-\varepsilon$ | Dylan, Nancy | $\text{NO}$,$\text{YES}$ |

Every local report is valid for lenticular viewing. As **§§2.1–2.2** established, once the codebook is fixed, the local laws $(S,P)\!\mapsto\!O$ render each context coherent; the only failure we saw came from dropping $P$, not from the law—and that was fixed by modeling $P$.

But here, a different question creates a different failure mode: can we assign each person a single, round-independent label ($\text{YES}$/$\text{NO}$) that matches all three pairwise observations? 

The rounds impose:

$$
\text{Nancy} \neq \text{Tyler},\quad
\text{Tyler} \neq \text{Dylan},\quad
\text{Dylan} \neq \text{Nancy}.
$$

To make it tactile, imagine the conversation between Tyler, Dylan, and Nancy:

| **Round** | **Nancy** | **Tyler** | **Dylan** |
| --- | --- | --- | --- |
| Round 1 —$\psi_1=0^\circ$ | "From here I see YES." | "Strange, because I see NO." | "I didn't look this round." |
| Round 2 —$\psi_2=90^\circ-\varepsilon$ | "I wasn't looking this time." | "Again I see NO." | "Well I see YES." |
| Round 3 —$\psi_3=180^\circ-\varepsilon$ | "From my seat I see YES." | "I sat out this one." | "Now I see NO." |

When asked to collectively describe how the coin operated, they would be unable to reach agreement, despite each telling the truth. The series of observations created a situation where no single, coherent description of the coin could accommodate all their valid experiences.

In short: the local laws are all obeyed, yet there is no single global law—no fixed YES/NO per person—that makes all three pairs correct at once. This is the classic odd-cycle impossibility: three pairwise "not equal" constraints cannot be satisfied by any global assignment. Each observation is right in its context, yet no frame-independent set of labels can satisfy all three at once. 

The incompatibility isn't noise; it's geometric—arising from how valid views interlock. Put differently: even an omniscient observer must choose a frame.

You can know everything—but not from one place.

This differs from the missing-context case in §2.2: carrying the frame there resolved ambiguity. Here, even with perfect context preservation, no frame-independent summary exists. The three questions, asked in sequence, admit no coherent single-story answer. The turntable is simple, not exotic; one could build this in a classroom.

Information-theoretically, "no global law" means there is no joint $Q\in\mathrm{FI}$ whose every context marginal matches $P$. Classical information theory can represent each context separately once $\psi$ is included among the variables; what it was *never designed* to represent under a single $Q$ is precisely the odd-cycle pattern. At best, two of the three pairwise constraints can be satisfied, so the irreducible disagreement rate is 1/3 per cycle.

Consequently, any frame-independent approximation must be wrong in at least one context. Numbers computed under a unified $Q$—entropy, code length, likelihood, mutual information, test-error exponents—are systematically misspecified. The gap is quantifiable: coding incurs $D(P\|Q^{\star})$ extra bits per sample for $Q^{\star}=\arg\min_{Q\in\mathrm{FI}} D(P\|Q)$; testing exponents are capped by $\alpha^{\star}(P)<1$ (equivalently, $K=-\log_2 \alpha^{\star}(P)$).

Quantitatively, the best frame-independent agreement is

$$
\alpha^{\star}=\sqrt{\tfrac{2}{3}},
$$

so the contradiction bit count is

$$
K(P)=-\log_2 \alpha^{\star}(P)=\tfrac{1}{2}\log_2\!\frac{3}{2}\approx 0.2925\ \text{bits per flip}.
$$

This is the **contradiction bit**: the per-observation cost of compressing all perspectives into one coherent view. The optimal dual weights are uniform, $\lambda^{\star}=(1/3,1/3,1/3)$, and the optimal $Q^{\star}$ saturates $\mathrm{BC}(p_c,q^{\star}_c)=\sqrt{2/3}$ in all three contexts.

---

## 2.4 The Key Insight: Revisiting the Axiom

> Axiom (Single-Valued Reports):
> 
> 1. Each observer must report a single word: YES or NO. 
> 2. No BOTH entries allowed 

It is fair to ask whether this axiom creates the contradiction. If we permit BOTH, the clash indeed disappears. That is the point. The contradiction does not come from the coin; it comes from our reporting architecture—from forcing plural observations into single-valued records. The world is pluralistic, yet our summaries of the world, are not.

This stance is inherited, it was never inevitable. Consider how fundamental this constraint is in the systems we rely on. Boole fixed propositions as true and false. Kolmogorov placed probability on that logic, and Shannon showed how such decisions travel as bits. None of these frameworks declared the world binary, they were just well-suited for the task at hand. 

If anything, they merely declare our records *can be* binary. Modern databases and protocols naturally followed suit: one message, one symbol, one frame. It wasn’t until recently that plurism emerged as an engineering problem.

However, none of these systems claimed the world was binary. What they did claim, and what we've inherited, is that our **records** *can be* binary. We've adopted this convention not because it's natural, but because it's standard. It's the foundation of virtually every digital system, database, and channel of formal communication: observations must collapse to a single symbol. One report, one truth, one frame.

Classical measures do an *excellent* job within a context: they price which outcome occurred. What they do not register, under a frame-independent summary, is a different kind of information—that several individually valid contexts can be valid, but cannot be made to agree at once. That is not “more surprise about outcomes”; it is a statement about feasibility across contexts.

Thus, this imposed axiom is no more a contrivance than the digital computer itself. The contradiction bit $K(P)$ then measures the structural cost of insisting on one story when no such story exists. The observed clash is not noise, deception, or paradox in nature; it’s simply the price of flattening—of collapsing perspectival structure to a single label.

Namely, there are two kinds of information are in play:

- **Statistical surprise**: Which outcome happened?—handled inside each context by Shannon’s framework.
- **Structural surprise**: Can these valid observations coexist in any global description?—assigned zero by a single-story summary, and restored by K(P).

Shannon priced randomness within a frame. When frames themselves clash, there is an additional, irreducible cost. Measuring that cost is necessary for any account of reasoning across perspectives.

Succinctly put:

> *To model intelligence is to model perspective. To model perspective is to model contradiction. And to model contradiction is to step beyond the frame and build the next layer.*
> 

---

# 3. Mathematical Foundations — The Geometry of Irreconcilability

Having established the foundations, the lenticular coin showed something we can now make precise: **contradiction has structure**. When Nancy, Dylan, and Tyler couldn't agree despite all being correct, they encountered an irreducible incompatibility built into their situation's geometry.

**Contradiction has geometry just as information has entropy.** While Shannon showed us that uncertainty follows precise mathematical laws, we'll show that clashes between irreconcilable perspectives follow equally discoverable patterns with measurable costs.

## 3.1 The Lenticular Coin, Reframed.

Let's be precise about what we measured when our three friends observed the lenticular coin:

- **Observables**: Things we can measure (like "what word does Nancy see?")
- **Contexts**: Different ways to probe the system (like "which pair of observers look simultaneously?")
- **Outcomes**: What actually happens when we look (like "Nancy sees YES, Tyler sees NO")
- **Behaviors**: The complete statistical pattern across all possible contexts

A **behavior** functions like a comprehensive experimental logbook. For every way you might probe the system, it records the probability distribution over what you'll observe. In our rotating coin experiment, this logbook includes entries like: "When Nancy and Tyler both look simultaneously, there's a 50% chance Nancy sees YES while Tyler sees NO, a 50% chance Nancy sees NO while Tyler sees YES, and 0% chance they agree."

The mathematical formalization captures this intuitive picture directly. We have observables $\mathcal{X} = \{X_1, X_2, \ldots, X_n\}$, where each can display various outcomes. A **context** $c$ is simply a subset of observables we examine together. A **behavior** $P$ assigns to each context $c$ a probability distribution $p_c$ over the outcomes we might see (App. A.1.1–A.1.4).

This isn't exotic machinery—just systematic bookkeeping for multi-perspective experiments.

### The Frame-Independent Baseline: When Perspectives Align

Some behaviors exhibit no contradictions at all. Consider a simple coin that always shows the same face to every observer—heads to Nancy, heads to Dylan, heads to Tyler. Here, all perspectives align perfectly. Even though different people look at different times or from different angles, their reports weave into one coherent explanation: "The coin landed heads."

This represents **frame-independent** behavior: one underlying reality explains everything different observers see. Disagreements arise only because observers examine different aspects of the same coherent system, not because the system itself contains contradictions.

But our lenticular coin behaves differently. Nancy, Dylan, and Tyler see genuinely incompatible things that resist integration into any single coherent explanation. This represents **frame-dependent** behavior—the signature of irreducible contradiction.

Mathematically, a behavior is **frame-independent** when there exists a single "master explanation" that simultaneously accounts for what every context would observe. More precisely, there must be a probability distribution $\mu$ over complete global states such that each context's observations are simply different slices of these states.

A **complete global state** is a full specification of every observable simultaneously—like saying "Nancy sees YES, Dylan sees BOTH, Tyler sees NO" all at once. If such states exist and one distribution $\mu$ over them reproduces all our contextual observations, then we have our baseline.

Three equivalent ways to understand this baseline:

1. Unified explanation: All observations integrate into one coherent account
2. Hidden variable model: A single random "state of affairs" underlies what each context reveals
3. Geometric picture: The baseline is the convex hull of "deterministic global assignments"—scenarios where every observable has a definite value

The **frame-independent set** (App. A.1.6) contains all behaviors admitting such unified explanations. These form our "no-contradiction" baseline. Crucially, FI has excellent mathematical properties in our finite setting: it's nonempty, convex, compact, and closed. This gives us a solid foundation for measuring distances from it.

Two cases will matter later. If there exists $Q\in\mathrm{FI}$ with $Q=P$, then $\alpha^\star(P)=1$ and $K(P)=0$ (no contradiction). If no such $Q$ exists, then $\alpha^\star(P)<1$ and $K(P)>0$ quantifies the minimal deviation any unified account must incur to explain all contexts at once

## 3.2 Measuring the Distance to Agreement

To quantify contradiction, we need a notion of "distance" between our observed behavior and the closest frame-independent explanation. When comparing probability distributions across multiple contexts, the Bhattacharyya coefficient provides exactly what we need. 

For probability distributions $p$ and $q$ over the same outcomes:

$$
\text{BC}(p,q) = \sum_{\text{outcomes}} \sqrt{p(\text{outcome}) \cdot q(\text{outcome})}
$$

This measures "probability overlap" (App. A.2.1). When $p$ and $q$ are identical, $\text{BC}(p,q) = 1$ (perfect overlap). When they assign probability to completely disjoint supports, $\text{BC}(p,q) = 0$ (no overlap). Between these extremes, the coefficient tracks how much the distributions have in common.

Three properties make this measure particularly suitable:

1. **Perfect agreement detection**: $\text{BC}(p,q) = 1$ if and only if $p = q$ (see App. A.2.2.2)
2. **Mathematical tractability**: It's concave and well-behaved for optimization (see App. A.2.2.3)
3. **Compositional structure**: For independent systems, $\text{BC}(p_1 \otimes p_2, q_1 \otimes q_2) = \text{BC}(p_1,q_1) \cdot \text{BC}(p_2,q_2)$ (see App. A.2.2.4)

This third property proves essential—contradiction costs multiply across independent subsystems, just like probabilities do.

## 3.3 The Core Measurement: Maximum Achievable Agreement

Given an observed behavior $P$, we now address the central question: across all possible frame-independent behaviors, what's the maximum agreement we can achieve with our observations?

A subtle but crucial choice emerges. We could measure agreement context-by-context, then average. But which contexts deserve more weight? The natural answer: let the worst-case contexts determine the overall assessment. If even one context shows poor agreement with our proposed frame-independent explanation, that explanation fails to capture the true structure.

This leads to our **agreement measure** (App. A.3.1):

$$
\alpha^\star(P) = \max_{Q \in \text{FI}} \min_{\text{contexts } c} \text{BC}(p_c, q_c)
$$

The formula reads: "Among all frame-independent behaviors $Q$, find the one that maximizes the worst-case agreement with our observed behavior $P$ across all contexts."

The max-min structure captures something essential about contradiction. Perspective clash concerns **universal reconcilability**. A truly frame-independent explanation must account for *every* context satisfactorily. One persistently problematic context breaks the entire unified narrative.

This optimization problem has well-behaved mathematical structure. By Sion's minimax theorem, we can equivalently write:

$$
\alpha^\star(P) = \min_{\lambda\in\Delta(\mathcal{C})} \max_{Q \in \text{FI}} \sum_{\text{contexts } c} \lambda_c \cdot \text{BC}(p_c, q_c)
$$

where $\lambda$ is a probability distribution over contexts. The optimal weighting $\lambda^\star$ reveals which contexts create the worst contradictions—they receive the highest weights in the sum. (see App. A.3)

### The Contradiction Bit

We can now define our central measure:

$$
K(P) = -\log_2 \alpha^\star(P)
$$

This is the **contradiction bit count**—the information-theoretic cost of the perspective clash. (App. A.3.1) 

When $\alpha^\star(P) = 1$ (perfect frame-independence), we get $K(P) = 0$ contradiction bits. As $\alpha^\star(P)$ decreases toward zero, $K(P)$ grows, measuring how much information is lost when we insist on a single unified explanation.

**Key properties:**

- $K(P) = 0$ if and only if $P$ is frame-independent (App. A.4.3)
- $K(P) \leq \frac{1}{2}\log_2(\max_{c\in\mathcal{C}}|\mathcal{O}_c|)$ (contradiction is bounded) (App. A.4)
- For our lenticular coin: $K(P) = \frac{1}{2}\log_2(3/2) \approx 0.29$ bits per observation (App. B.2)

## 3.4 The Game-Theoretic Structure

The minimax formulation (App. A.3.2) reveals the deeper structure of contradiction 

The optimization:

$$
\alpha^\star(P) = \min_{\lambda\in\Delta(\mathcal{C})} \max_{Q \in \text{FI}} \sum_c \lambda_c \cdot \text{BC}(p_c, q_c)
$$

has the structure of a two-player game (App. A.3.2):

- **The Adversary**: Chooses context weights $\lambda$ to focus on the most problematic contradictions
- **The Explainer**: Selects a frame-independent behavior $Q$ to maximize agreement despite the adversarial focus

The optimal strategies $(\lambda^\star, Q^\star)$ reveal the deepest structural features (Theorem A.5.1):

- $\lambda^\star$ identifies which contexts resist reconciliation
- $Q^\star$ provides the best possible unified approximation
- The game value $\alpha^\star(P)$ measures the ultimate limits of perspective reconciliation

**For our lenticular coin**, the solution exhibits perfect symmetry:

- $\lambda^\star = (1/3, 1/3, 1/3)$—all contexts are equally problematic
- $Q^\star$ has, in each context, $(q_{00},q_{01},q_{10},q_{11})=(1/6,1/3,1/3,1/6)$ (see App. B.2)
- $\alpha^\star(P) = \sqrt{2/3}$—the limit of achievable agreement

This isn't coincidental—it reflects the democratic nature of odd-cycle contradictions. No single context is more problematic than others; the contradiction distributes evenly across all perspectives.

---

## 3.5 Recursive Structure and Hierarchical Application

The formalism extends to hierarchies without modification. An *observer's* assessment can be treated as an additional **observable** with its own labeled outcomes, and the resulting system is analyzed with no changes.

**Promoting assessments to observables**
Let $\mathcal{X}$ be the current set of observables with behavior $P=\{p_c\}_{c\in\mathcal{C}}$. Suppose an observer $\ell$ assigns, for some object $Y\in\mathcal{X}$, a distribution over labels $\Sigma_\ell(Y)$ (e.g., *Reliable/Unreliable*). We **promote** this assessment to a new observable $X_\ell^Y$ with outcome alphabet $\Sigma_\ell(Y)$, and we extend the context family by allowing $X_\ell^Y$ to co-occur with the relevant observables it summarizes. The augmented behavior is $P' = P \uplus \{p^{(\ell,Y)}\}$. No definitions change: contexts, behaviors, $\mathrm{FI}$, $\alpha^\star$, and $K=-\log_2\alpha^\star$ are applied exactly as before to $P'$.

**Level stability**
Two closure facts ensure stability under this promotion:

1. **Relabeling and product closure (App. A.1.8) of $\mathrm{FI}$:** Adding $X_\ell^Y$ amounts to adjoining an observable with fixed marginals (determined by $\ell$). Because $\mathrm{FI}$ is convex and closed under products with fixed factors, the maximizer $Q^\star\in \mathrm{FI}$ in (3.1) extends to the enlarged space without changing form.
2. **Max–min preservation (App. A.2.2.4, App. A.3.2):** The objective in (3.1) for $P'$ is still a worst-case Bhattacharyya overlap over contexts. New contexts contribute multiplicative factors already fixed by $\ell$; taking the minimum over contexts preserves the max–min structure.

Consequently, the same computation that measures peer-level clash among base observables also measures meta-level clash among observers' assessments. No new machinery is introduced—only additional rows in the same logbook.

**Ceiling for cross-level contradiction (App. A.4.x)**
Cross-level disagreement cannot exceed the uncertainty already present in the upstream assessment. If an upstream observable $X_\ell^Y$ has marginal $p\in[0,1]$ on a label (e.g., "Reliable"), then for any downstream assessment opposing it,$K_{\text{cross}} \;\le\; H_2(p)\;=\; -\,p\log_2 p \;-\; (1-p)\log_2(1-p).$Intuition: disagreement bits cannot exceed the bits of slack available at the level being contradicted. A $50/50$ assessment admits up to $1$ bit of clash; a $95/5$ assessment caps near $0.286$ bits—the binary entropy of a $95/5$ split. This matches the gradients reported in the examples.

**Consequences**

1. **Recursive measurability.** Repeating the promotion step (reviewers → supervisor → director, …) yields a hierarchy in which $K$ quantifies tension both within a level and across levels, with identical definitions.
2. **Budget of disagreement.** As upstream assessments sharpen (lower entropy), the maximum attainable cross-level contradiction decreases; degenerate assessments (probability $0$ or $1$) admit none.

*Proof sketch (bounds).* For the ceiling: oppose a fixed marginal $p$ by maximizing the decrease in Bhattacharyya overlap; Jensen concavity of $\sqrt{\cdot}$ inside $\mathrm{BC}$ upper-bounds the adversary's gain by the upstream entropy $H_2(p)$. For level stability: embed $Q^\star\in\mathrm{FI}$ into the enlarged space with the fixed factor induced by $X_\ell^Y$; multiplicativity of $\mathrm{BC}$ over independent components and the outer $\min_c$ preserve the max–min form and value up to the added fixed factors.

> Plain-language takeaway. Watching the watchers doesn't need new machinery. You promote opinions to outcomes and run the same distance-to-agreement calculation. And the fiercest possible clash across levels is limited by how unsure the lower level already was.
> 

## 3.6 Beyond Entropy — A Theory of Contradiction

We’ve now arrived at the core deliverable of this framework: a general-purpose method for measuring contradiction as a first-class information-theoretic quantity.

1. **Recognition**: $K(P) = 0$ precisely characterizes frame-independent behaviors
2. **Quantification**: $K(P)$ measures the information-theoretic cost of perspective clash
3. **Optimization**: The minimax structure identifies worst-case contexts and optimal approximations
4. **Bounds**: Contradiction is mathematically well-behaved and bounded
5. **Universality**: The framework applies to any multi-context system, regardless of domain

This framework aligns formally with the language of contextuality in quantum foundations—most notably the sheaf-theoretic formulation introduced by Abramsky and Brandenburger. **But unlike prior approaches rooted in quantum mechanics, we derive this structure independently, from first principles within information theory.** No quantum assumptions are required.

The geometry we uncover does not belong exclusively to quantum physics—though the resemblance is striking. While quantum contextuality was not the starting point of our inquiry, it emerged as a natural consequence. What matters more is that this same geometry **recurs across domains** whenever multiple perspectives must be reconciled under global constraints: in distributed systems, organizational paradoxes, statistical puzzles, and beyond.

This suggests that contradiction is not a quantum anomaly—but instead exists as a **universal structural phenomenon** in information itself. In this view, the contradiction bit becomes a natural companion to Shannon’s entropy: where entropy quantifies randomness within a single frame, contradiction quantifies incompatibility across frames. Together, they form a multi-dimensional theory of information—one capable of describing not just uncertainty, but also irreconcilability.

In the next section, we’ll show how the characteristics within our lenticular coin naturally lead to this solution—not as an invention, but as an inevitability.

---

## 4. The Axioms

We’ll now establish that any reasonable contradiction measure must satisfy six elementary properties, which together uniquely determine the form $K(P) = -\log_2 \alpha^\star(P)$.

---

### Axiom A0: Label Invariance

> *Contradiction lives in the structure of perspectives—not within perspectives themselves.*
> 

**Formally**:

$K$ is invariant under outcome and context relabelings (permutations).

---

### Axiom A1: Reduction

> *We cannot measure a contradiction if no contradiction exists.*
> 

**Formally**:

$K(P) = 0$ if and only if $P \in \mathrm{FI}$.

---

### Axiom A2: Continuity

> *Small disagreements deserve small measures.*
> 

**Formally**:

$K$ is continuous in the product $L_1$ metric:

$$
d(P,P') = \max_{c \in \mathcal{C}} \bigl\|p(\cdot|c) - p'(\cdot|c)\bigr\|_1
$$

---

### Axiom A3: Free Operations

> *Structure lost may conceal contradiction — but never invent it.*
> 

**Formally**:

$K$ is monotone under stochastic post-processing of outcomes, stochastic merging of contexts via public lotteries independent of outcomes and hidden variables, convex combinations $\lambda P + (1-\lambda)P'$, and tensoring with FI ancillas.

---

### Axiom A4: Grouping

> *Contradiction is a matter of substance, not repetition.*
> 

**Formally**:

$K$ depends only on refined statistics when contexts are split via public lotteries, independent of outcomes and hidden variables. In particular, duplicating or removing identical rows leaves $K$ unchanged.

---

### Axiom A5: Independent Composition

> *Contradictions compound; they do not cancel.*
> 

**Formally**:

For operationally independent behaviors on disjoint observable sets:

$$
K(P \otimes R) = K(P) + K(R)
$$

This requires that FI be closed under products: for any $Q_A \in \mathrm{FI}_A$ and $Q_B \in \mathrm{FI}_B$, we have $Q_A \otimes Q_B \in \mathrm{FI}_{A \sqcup B}$.

---

---

## 4.1 From Lenticular Insight to Axiomatic Foundation

These six axioms we have established are not contrivances — they emerge naturally from the lessons learned from our lenticular coin. Each insight about how perspectives interact translates directly into a mathematical requirement that any reasonable contradiction measure must satisfy.

Let's revisit the key properties we observed:

1. Multiple valid perspectives coexist 
2. Each perspective maintains local consistency
3. Ambiguity inheres in the system itself
4. Universal agreement may prove unattainable
5. Context forms an essential part of any message
6. Contradictions follow discoverable patterns
7. Information coordination carries inherent costs

We’ll now show how together these axioms represent our intuitive understanding of the lenticular coin, translating them into precise mathematical constraints. 

**Multiple Valid Perspectives → Label Invariance (A0)**

- When Nancy sees “YES,” Dylan sees “NO,” and Pat sees “BOTH,” the contradiction doesn’t arise from the particular symbols they observe. The structural impossibility of reconciling their reports persists whether we write (YES, NO, BOTH) or (1, 0, ½) or use any other labeling scheme.
- Axiom A0 captures this: contradiction lives in the *relationships* between perspectives, not in the arbitrary labels we assign to outcomes or contexts.

**Local Consistency → Reduction (A1)**

- Each observer’s individual reports make perfect sense within their own context—Nancy consistently sees “YES” from the left, Dylan consistently sees “NO” from the right. The contradiction emerges only when we attempt to synthesize these locally-coherent perspectives into a single global account.
- Axiom A1 formalizes this: if such a synthesis succeeds (if $P \in \mathrm{FI}$), then no contradiction exists to measure ($K(P) = 0$).

**Inherent Ambiguity → Continuity (A2)**

- As Tyler moves from Nancy’s position toward Dylan’s, the coin’s appearance shifts gradually from “YES” through ambiguous states to “NO.” The contradiction doesn’t jump discontinuously—it evolves smoothly with the changing perspectives. Axiom A2 ensures our measure respects this natural continuity, changing gradually as the underlying perspective structure shifts.

**Unattainable Universal Agreement → Free Operations (A3)**

- No amount of averaging Nancy’s and Dylan’s reports, no coarse-graining of their observations, no random mixing of their contexts can eliminate the fundamental fact that they see different things from their respective positions. The contradiction is structural, not superficial—it cannot be dissolved through information-processing shortcuts.
- Axiom A3 formalizes this irreducibility: operations that merely blur or combine existing information cannot create false reconciliation.

**Context as Essential → Grouping (A4)**

- Whether Nancy states her observation once or repeats it ten times doesn’t change the nature of her disagreement with Dylan. The contradiction isn’t about volume or frequency—it’s about the existence of genuinely distinct contextual perspectives.
- Axiom A4 captures this by making contradiction depend only on which distinct contexts exist, not on how often each appears or how we partition contexts through public randomization.

**Discoverable Patterns → Independent Composition (A5)**

- If Nancy and Dylan disagree about both the coin’s political message *and* its artistic style, these represent two separate contradictions that compound systematically. The total coordination cost must account for both incompatibilities.
- Axiom A5 ensures that contradictions in independent domains add rather than interfere: $K(P \otimes R) = K(P) + K(R)$.

These axioms ensure that any contradiction measure captures what we learned: that some perspective-dependent behaviors carry irreducible coordination costs—costs that cannot be eliminated by relabeling, averaging, or wishful thinking, but can be quantified and predicted.

The resulting unique form $K(P) = -\log_2 α^*(P)$ thus inherits all the conceptual richness of our original insight while providing the mathematical precision needed for information theory. The “contradiction bit” emerges not as an abstract mathematical construct, but as the natural quantification of something we discovered empirically: the fundamental cost of reconciling genuinely incompatible perspectives.

---

## 4.2 Boundaries of the Theory

It’s important to note that this framework applies wherever observations can be modeled as distributions across defined contexts. This includes not only physical experiments but, in principle, also domains like epistemology or cultural conflict.  **The challenge in those areas is not theoretical incompatibility but practical specification: unless contexts and independence can be formalized, $K(P)$ remains undefined rather than violated.

**Core Applicability Conditions**

- **Multiple Sources, Context-Dependent Reports**: Information originates from agents situated in distinct contexts. Each report is tied to the observer’s position, perspective, or frame. Unlike Shannon’s single objective source, contradiction theory begins with many sources that cannot always be reconciled.
- **Finite Contexts and Outcomes**: Reports are modeled as draws from a finite set of outcomes (YES/NO, 0/1, etc.) conditioned on a finite set of contexts. This restriction ensures that contradiction can be precisely defined and measured.
- **Structural Conflict, Not Noise**: Uncertainty here is not statistical randomness alone but the structural incompatibility of perspectives. Even when every report is noiseless and perfectly reliable, contradiction can remain.
- **Observers as Integral, Not Interchangeable**: The identity and position of observers are not incidental. Contradiction arises precisely because different observers, even when equally reliable, cannot always be merged into a single coherent account.
- **No Guaranteed Global Reconstruction**: The goal is not to recover a single objective story, since such a story may not exist. Instead, the goal is to measure the irreducible cost of forcing incompatible perspectives into one frame.
- **Semantics as Central**: Whereas Shannon set meaning aside, contradiction theory takes it as unavoidable. Contradiction is a property of how reports from different frames do or do not cohere, and thus cannot be defined without reference to the semantic relation between contexts.

## 

## Scope

Scope of mathematics:

- We will work in finite discrete settings for clarity; extensions to continuous domains follow naturally.
- Subsequent work applies the framework to real datasets, quantifies contextuality strength across domains, and explores computable proxies for cross-frame description length.

---

# Appendix A — Full proofs and Technical Lemmas

This appendix provides proofs and technical details for the mathematical framework introduced in Section 3. We maintain the intuitive explanations from the main text while supplying complete formal justification for all claims.

Standing Assumptions (finite case). Throughout Appendix A we assume finite outcome alphabets and that $\mathrm{FI}$ (the frame-independent set) is nonempty, convex, compact, and closed under products. These conditions are satisfied in our finite setting by construction (A.1)

## A.1 Formal Setup and Definitions

### A.1.1 Basic Structures

**Definition A.1.1 (Observable System).** Let $\mathcal{X} = \{X_1, \ldots, X_n\}$ be a finite set of observables. For each $x \in \mathcal{X}$, fix a finite nonempty outcome set $\mathcal{O}_x$. A **context** is a subset $c \subseteq \mathcal{X}$. The outcome alphabet for context $c$ is $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$.

**Definition A.1.2 (Behavior).** Given a finite nonempty family $\mathcal{C} \subseteq 2^{\mathcal{X}}$ of contexts, a **behavior** $P$ is a family of probability distributions

$$
P = \{p_c \in \Delta(\mathcal{O}_c) : c \in \mathcal{C}\}
$$

where $\Delta(\mathcal{O}_c)$ denotes the probability simplex over $\mathcal{O}_c$.

**Remark (No nondisturbance required).** We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in A.1.4. When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.

**Definition A.1.3 (Deterministic Global Assignment).** Let $\mathcal{O}_{\mathcal{X}} := \prod_{x \in \mathcal{X}} \mathcal{O}_x$. A **deterministic global assignment** is an element $s \in \mathcal{O}_{\mathcal{X}}$. It induces a deterministic behavior $q_s$ by restriction:

$$
q_s(o \mid c) = \begin{cases} 1 & \text{if } o = s|_c \\ 0 & \text{otherwise} \end{cases}
$$

for each context $c \in \mathcal{C}$ and outcome $o \in \mathcal{O}_c$.

**Definition A.1.4 (Frame-Independent Set).** The **frame-independent set** is

$$
\text{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\} \subseteq \prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)
$$

**Proposition A.1.5 (Alternative Characterization of FI).** $Q \in \text{FI}$ if and only if there exists a **global law** $\mu \in \Delta(\mathcal{O}_{\mathcal{X}})$ such that

$$
q_c(o) = \sum_{s \in \mathcal{O}_{\mathcal{X}} : s|_c = o} \mu(s) \quad \forall c \in \mathcal{C}, o \in \mathcal{O}_c
$$

**Proof.** The forward direction is immediate from the definition of convex hull. For the reverse direction, given $\mu$, define $Q$ by the displayed formula. Then $Q$ is a convex combination of the deterministic behaviors $\{q_s\}$ with weights $\{\mu(s)\}$, hence $Q \in \text{FI}$. □

### A.1.2 Basic Properties of FI

**Proposition A.1.6 (Topological Properties).** The frame-independent set FI is nonempty, convex, and compact.

**Proof.**

- **Nonempty**: Contains all deterministic behaviors $q_s$.
- **Convex**: By definition as a convex hull.
- **Compact**: FI is a finite convex hull in the finite-dimensional space $\prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)$, hence a polytope, hence compact. □

**Definition A.1.7 (Context simplex).** $\Delta(\mathcal{C}) := \{\lambda \in \mathbb{R}^{\mathcal{C}} : \lambda_c \geq 0, \sum_{c \in \mathcal{C}} \lambda_c = 1\}$.

**Proposition A.1.8 (Product Structure).** Let $P$ be a behavior on $(\mathcal{X}, \mathcal{C})$ and $R$ be a behavior on $(\mathcal{Y}, \mathcal{D})$ with $\mathcal{X} \cap \mathcal{Y} = \emptyset$ (we implicitly relabel so disjointness holds). For distributions $p \in \Delta(\mathcal{O}_c)$ and $r \in \Delta(\mathcal{O}_d)$ on disjoint coordinates, $p \otimes r \in \Delta(\mathcal{O}_c \times \mathcal{O}_d)$ is $(p \otimes r)(o_c, o_d) = p(o_c)r(o_d)$.

Define the product behavior $P \otimes R$ on $(\mathcal{X} \sqcup \mathcal{Y}, \mathcal{C} \otimes \mathcal{D})$ where $\mathcal{C} \otimes \mathcal{D} := \{c \cup d : c \in \mathcal{C}, d \in \mathcal{D}\}$ by

$$
(p \otimes r)(o_c, o_d \mid c \cup d) = p(o_c \mid c) \cdot r(o_d \mid d)
$$

Then:

1. If $Q \in \text{FI}_{\mathcal{X},\mathcal{C}}$ and $S \in \text{FI}_{\mathcal{Y},\mathcal{D}}$, then $Q \otimes S \in \text{FI}_{\mathcal{X} \sqcup \mathcal{Y}, \mathcal{C} \otimes \mathcal{D}}$.
2. For deterministic assignments, $q_s \otimes q_t = q_{s \sqcup t}$.

**Proof.**

1. If $Q$ arises from global law $\mu$ and $S$ arises from global law $\nu$, then $Q \otimes S$ arises from the product global law $\mu \otimes \nu$ on $\mathcal{O}_{\mathcal{X} \sqcup \mathcal{Y}}$. From $q_s \otimes q_t = q_{s \sqcup t}$, it follows that
    
    $$
    (\sum_s \mu_s q_s)\otimes(\sum_t \nu_t q_t)=\sum_{s,t}\mu_s\nu_t\,(q_s\otimes q_t)=\sum_{s,t}\mu_s\nu_t\,q_{s\sqcup t}\in \mathrm{conv}\{q_{s\sqcup t}\}.
    $$
    
2. Direct verification from definitions: $q_s \otimes q_t = q_{s \sqcup t}$ because $\delta_{s|_c} \otimes \delta_{t|_d} = \delta_{(s \sqcup t)|_{c \cup d}}$. □

**Definition A.2.1 (Bhattacharyya Coefficient).** For probability distributions $p, q \in \Delta(\mathcal{O})$ on a finite alphabet $\mathcal{O}$:

$$
\text{BC}(p, q) := \sum_{o \in \mathcal{O}} \sqrt{p(o) q(o)}
$$

**Lemma A.2.2 (Bhattacharyya Properties).** For distributions $p, q \in \Delta(\mathcal{O})$:

1. **Range**: $0 \leq \text{BC}(p, q) \leq 1$
2. **Perfect agreement**: $\text{BC}(p, q) = 1 \Leftrightarrow p = q$
3. **Joint concavity**: Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$. Therefore $(x,y)\mapsto\sqrt{xy}$ is jointly concave on $\mathbb{R}_{\geq 0}^2$ (extend by continuity on the boundary). Summing over coordinates preserves concavity, so $\text{BC}$ is jointly concave on $\Delta(\mathcal{O})\times\Delta(\mathcal{O})$.
4. **Product structure**: $\text{BC}(p \otimes r, q \otimes s) = \text{BC}(p, q) \cdot \text{BC}(r, s)$

**Proof.**

(1) *Range.* Nonnegativity is obvious. For the upper bound, by Cauchy-Schwarz:

$$
\text{BC}(p, q) = \sum_o \sqrt{p(o) q(o)} \leq \sqrt{\sum_o p(o)} \sqrt{\sum_o q(o)} = 1
$$

(2) *Perfect agreement.* The Cauchy-Schwarz equality condition gives $\text{BC}(p, q) = 1$ iff $\sqrt{p(o)}$ and $\sqrt{q(o)}$ are proportional, i.e., $\frac{\sqrt{p(o)}}{\sqrt{q(o)}}$ is constant over $\{o : p(o) q(o) > 0\}$. Since both are probability distributions, this constant must be 1, giving $p = q$.

(3) *Joint concavity.* Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$; extend to $\mathbb{R}_{\geq 0}^2$ by continuity. Summing over coordinates preserves concavity.

(4) *Product structure.* Expand the tensor product and factor the sum. □

## A.3 The Agreement Measure and Minimax Theorem

**Definition A.3.1 (Agreement and Contradiction).** For a behavior $P$:

$$
\alpha^\star(P) := \max_{Q \in \text{FI}} \min_{c \in \mathcal{C}} \text{BC}(p_c, q_c)
$$

$$
K(P) := -\log_2 \alpha^\star(P)
$$

**Theorem A.3.2 (Minimax Equality).** Define the payoff function

$$
f(\lambda, Q) := \sum_{c \in \mathcal{C}} \lambda_c \text{BC}(p_c, q_c)
$$

for $\lambda \in \Delta(\mathcal{C})$ and $Q \in \text{FI}$. Then:

$$
\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \text{FI}} f(\lambda, Q)
$$

Maximizers/minimizers exist by compactness and continuity of $f$.

**Proof.** We apply Sion's minimax theorem (M. Sion, *Pacific J. Math.* **8** (1958), 171–176). We need to verify:

1. $\Delta(\mathcal{C})$ and FI are nonempty, convex, and compact ✓
2. $f(\lambda, \cdot)$ is concave on FI for each $\lambda$ ✓
3. $f(\cdot, Q)$ is convex (actually linear) on $\Delta(\mathcal{C})$ for each $Q$ ✓

**Details:**

- Compactness of $\Delta(\mathcal{C})$: Standard simplex.
- Compactness of FI: Proposition A.1.6.
- Concavity in $Q$: Since $Q \mapsto (q_c)_{c \in \mathcal{C}}$ is affine and each $\text{BC}(p_c, \cdot)$ is concave (Lemma A.2.2.3), the nonnegative linear combination $\sum_c \lambda_c \text{BC}(p_c, q_c)$ is concave in $Q$.
- Linearity in $\lambda$: Obvious from the definition.

By Sion's theorem, $\min_\lambda \max_Q f(\lambda, Q) = \max_Q \min_\lambda f(\lambda, Q)$.

It remains to show this common value equals $\alpha^\star(P)$. For any $Q \in \text{FI}$, let $a_c := \text{BC}(p_c, q_c)$. Then:

$$
\min_{\lambda \in \Delta(\mathcal{C})} \sum_{c} \lambda_c a_c = \min_{c \in \mathcal{C}} a_c
$$

with the minimum achieved by $\lambda$ supported on $\arg\min_c a_c$.

Therefore:

$$
\max_{Q \in \text{FI}} \min_{\lambda \in \Delta(\mathcal{C})} f(\lambda, Q) = \max_{Q \in \text{FI}} \min_{c \in \mathcal{C}} \text{BC}(p_c, q_c) = \alpha^\star(P)
$$

and hence the common value equals $\alpha^\star(P)$. This completes the proof. □

## A.4 Bounds and Characterizations

**Lemma A.4.1 (Uniform Law Lower Bound).** For any behavior $P$:

$$
\alpha^\star(P) \geq \min_{c \in \mathcal{C}} \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

**Proof.** Let $\mu$ be the uniform (counting-measure) distribution on $\mathcal{O}_{\mathcal{X}}$ (so each global state is equally likely). This induces $Q^{\text{unif}} \in \text{FI}$ with uniform **context marginals**: $q_c^{\text{unif}}(o) = \frac{1}{|\mathcal{O}_c|}$ for all $c \in \mathcal{C}$, $o \in \mathcal{O}_c$.

For any context $c$:

$$
\text{BC}(p_c, q_c^{\text{unif}}) = \sum_{o \in \mathcal{O}_c} \sqrt{p_c(o) \cdot \frac{1}{|\mathcal{O}_c|}} = \frac{1}{\sqrt{|\mathcal{O}_c|}} \sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)}
$$

The function $\sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)}$ is concave on the simplex $\Delta(\mathcal{O}_c)$, so its minimum is attained at a vertex (a point mass), where the sum equals 1. Therefore:

$$
\sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)} \geq 1
$$

This minimum is achieved when $p_c$ is a point mass. Therefore:

$$
\text{BC}(p_c, q_c^{\text{unif}}) \geq \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

Since $\alpha^\star(P) \geq \min_{c} \text{BC}(p_c, q_c^{\text{unif}})$, the result follows. □

**Corollary A.4.2 (Bounds on K).** For any behavior $P$:

$$
0 \leq K(P) \leq \frac{1}{2} \log_2 \left(\max_{c \in \mathcal{C}} |\mathcal{O}_c|\right)
$$

**Proof.** The lower bound follows from $\alpha^\star(P) \leq 1$. The upper bound follows from Lemma A.4.1 and the fact that $-\log_2(x^{-1/2}) = \frac{1}{2}\log_2(x)$. □

**Theorem A.4.3 (Characterization of Frame-Independence).** For any behavior $P$:

$$
\alpha^\star(P) = 1 \Leftrightarrow P \in \text{FI} \Leftrightarrow K(P) = 0
$$

**Proof.**

($\Rightarrow$) If $\alpha^\star(P) = 1$, then there exists $Q \in \text{FI}$ such that $\min_c \text{BC}(p_c, q_c) = 1$. This implies $\text{BC}(p_c, q_c) = 1$ for all $c \in \mathcal{C}$. By Lemma A.2.2, this gives $p_c = q_c$ for all $c$, hence $P = Q \in \text{FI}$.

($\Leftarrow$) If $P \in \text{FI}$, take $Q = P$ in the definition of $\alpha^\star(P)$. Then $\min_c \text{BC}(p_c, q_c) = \min_c \text{BC}(p_c, p_c) = 1$.

The equivalence with $K(P) = 0$ follows from the definition $K(P) = -\log_2 \alpha^\star(P)$. □

**Remark (No nondisturbance required).** We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in A.1.4. When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.

## A.5 Duality Structure and Optimal Strategies

**Theorem A.5.1 (Minimax Duality).** Let $(\lambda^\star, Q^\star)$ be optimal strategies for the minimax problem in Theorem A.3.2. Then:

1. $f(\lambda^\star, Q^\star) = \alpha^\star(P)$
2. $\text{supp}(\lambda^\star) \subseteq \{c \in \mathcal{C} : \text{BC}(p_c, q_c^\star) = \alpha^\star(P)\}$
3. If $\lambda^\star_c > 0$, then $\text{BC}(p_c, q_c^\star) = \alpha^\star(P)$

**Proof.** Existence of optimal strategies follows from compactness and continuity.

1. This is immediate from the minimax equality.
2. & 3. For fixed $Q^\star$, the inner optimization $\min_{\lambda \in \Delta(\mathcal{C})} \sum_c \lambda_c a_c$ with $a_c := \text{BC}(p_c, q_c^\star)$ has value $\min_c a_c$ and optimal solutions supported on $\arg\min_c a_c$. Since $(\lambda^\star, Q^\star)$ is optimal for the full problem, $\lambda^\star$ must be optimal for this inner problem, giving the result. □

## References for Appendix A

- Sion, M. "On general minimax theorems."*Pacific Journal of Mathematics* 8.1 (1958): 171-176.
- Bhattacharyya, A. "On a measure of divergence between two statistical populations defined by their probability distributions."*Bulletin of the Calcutta Mathematical Society* 35 (1943): 99-109.

---

# Appendix B — Worked Examples

## **B.1 Worked Example: The Irreducible Perspective: Carrying the Frame.**

Let's consider an example. Let's say there are three friends—*Nancy*, *Dylan*, *Tyler*—watch the coin from three seats: $\text{LEFT}$, $\text{MIDDLE}$, and $\text{RIGHT}$. After each flip they copy only the three words they saw, comma-separated, into a shared notebook. Each knew where they sat, and the order the games went, so they didn't think to write the positions.

Notebook

| **Flip** | **Observer Reports** |
| --- | --- |
| 1 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |
| 2 | $\text{NO}$,$\text{BOTH}$,$\text{YES}$ |
| 3 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |
| 4 | $\text{NO}$,$\text{BOTH}$,$\text{YES}$ |
| 5 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |

Now, using only this notebook—can you tell who sat where? You can't. (Unless you're grading your own work, in which case: 10/10, excellent deduction).

It's because "$\text{YES}$, $\text{BOTH}$, $\text{NO}$" fits two different worlds on every line:

- On $\text{HEADS}$ the left seat is $\text{YES}$, the middle is $\text{BOTH}$, the right is $\text{NO}$;
- On $\text{TAILS}$ the left is $\text{NO}$, the middle is $\text{BOTH}$, the right is $\text{YES}$.

Strip names and seats, and those worlds collapse to the same record. You are exactly one yes/no short: "who wrote $\text{YES}$—left or right?" Formally, each report depends on two coordinates: the coin's face $S\in\{\text{HEADS},\text{TAILS}\}$ and the viewpoint $P\in\{L,M,R\}$. 

The observation $O\in\{\text{YES},\text{NO},\text{BOTH}\}$ is

$$
O(S,P)= \begin{cases} \text{BOTH}, & P=M,\\ \text{YES}, & (S,P)\in\{(\text{HEADS},L),(\text{TAILS},R)\},\\ \text{NO}, & (S,P)\in\{(\text{HEADS},R),(\text{TAILS},L)\}. \end{cases}
$$

The smoking gun appears when we hide $P$: distinct worlds land on the same word. 

- $(\text{HEADS},L)$ and $(\text{TAILS},R)$ both read $\text{YES}$;
- $(\text{HEADS},R)$ and $(\text{TAILS},L)$ both read $\text{NO}$.

There is no frame-independent reconstruction from $O$ alone. Every crisp report owes one more yes/no if you want the scene to be recoverable. Carry $P$, and the correlations are captured perfectly; drop it, and two different worlds become indistinguishable on the page.

Now quantify the coordination cost of omitting $P$. Suppose seats are used equally and the coin is fair, so $\text{YES}$, $\text{BOTH}$, $\text{NO}$ each appear about one-third of the time in the notebook.

- Seeing $\text{BOTH}$ pins $P=M$: no uncertainty remains (0 missing bits).
- Seeing $\text{YES}$ or $\text{NO}$ tells you only "not the middle," leaving a binary uncertainty—left or right (1 missing bit).

Averaging,

$$
\text{Missing}=\tfrac13\cdot 0+\tfrac13\cdot 1+\tfrac13\cdot 1=\tfrac23\ \text{bits per line}.
$$

In Shannon's notation, with $H(P)=\log_2 3$,

$$
I(O;P)=H(P)-H(P\mid O)=\log_2 3-\tfrac23\approx 0.918\ \text{bits},\qquad \boxed{H(P\mid O)=\tfrac23\ \text{bits per line}}.
$$

Plainly: unless you carry the frame, you are on average two-thirds of a bit short of knowing where each observer sat. Thus, the cost is small, but persistent. 

It is not the one-off surprise of model identification; it is a steady coordination cost dictated by the object's geometry. If you also record the coin face $S$ alongside $O$, then together $(S,O)$ identify the seat exactly, driving this residual to zero—but at the price of carrying additional information with every observation. The lesson is not that perspective is noise to be averaged out, but that it is structure to be carried. 

Keep the frame and the story is coherent. Drop it, and worlds become epistemically indistinct.

## B.2 The Lenticular Coin / Odd–Cycle Example

**Example B.2.1 (Odd-Cycle Contradiction).**

We model Nancy (N), Dylan (D), and Tyler (T) as three **binary** observables $X_N,X_D,X_T\in\{0,1\}$ encoding $\mathrm{YES}=1$, $\mathrm{NO}=0$. The three **contexts** are the pairs

$$
c_1=\{N,T\},\qquad c_2=\{T,D\},\qquad c_3=\{D,N\}
$$

The observed behavior $P=\{p_{c_i}\}_{i=1}^3$ is the **"perfect disagreement"** behavior on each edge:

$$
p_{c}(1,0)=p_{c}(0,1)=\tfrac12,\qquad p_{c}(0,0)=p_{c}(1,1)=0\quad\text{for }c\in\{c_1,c_2,c_3\}
$$

Equivalently: each visible pair always disagrees, but the direction of disagreement is uniformly random.

We compute

$$
\alpha^\star(P)\;=\;\max_{Q\in\mathrm{FI}}\ \min_{c\in\{c_1,c_2,c_3\}} \mathrm{BC}(p_c,q_c)
$$

and show $\alpha^\star(P)=\sqrt{\tfrac23}$, hence $K(P)=\tfrac12\log_2\!\frac32$.

### B.2.1 Universal Upper Bound: $\alpha^\star\le \sqrt{2/3}$

**Lemma B.2.2 (Upper Bound).** Let $Q\in\mathrm{FI}$ arise from a global law $\mu$ on $\{0,1\}^3$. For a context $c=\{i,j\}$, write the induced pair distribution

$$
q_c(00),\ q_c(01),\ q_c(10),\ q_c(11)\quad\text{and}\quad
D_{ij}:=q_c(01)+q_c(10)=\Pr_{\mu}[X_i\ne X_j]
$$

For our $p_c$ (uniform over the off-diagonals), the Bhattacharyya coefficient is

$$
\mathrm{BC}(p_c,q_c)
=\sqrt{\tfrac12}\big(\sqrt{q_c(01)}+\sqrt{q_c(10)}\big)
\le \sqrt{D_{ij}}
$$

with equality iff $q_c(01)=q_c(10)=D_{ij}/2$ (by concavity of $\sqrt{\cdot}$ at fixed sum).

Thus

$$
\min_{c}\ \mathrm{BC}(p_c,q_c)\ \le\ \min_{c}\ \sqrt{D_c}
$$

The triple $(D_{NT},D_{TD},D_{DN})$ must be feasible as edge-disagreement probabilities of a joint $\mu$ on $\{0,1\}^3$. For three bits, every deterministic assignment has either 0 disagreements (all equal) or exactly 2 (one bit flips against the other two). Hence any convex combination obeys the **cut-polytope constraint**

$$
D_{NT}+D_{TD}+D_{DN}\ \le\ 2
$$

Consequently at least one edge has $D_c\le 2/3$, so

$$
\min_c \sqrt{D_c}\ \le\ \sqrt{2/3}
$$

Taking the maximum over $Q\in\mathrm{FI}$ yields the universal upper bound

$$
\alpha^\star(P)\ \le\ \sqrt{2/3}
$$

### B.2.2 Achievability: An Explicit Optimal $\mu^\star$

**Proposition B.2.3 (Achievability).** Let $\mu^\star$ be the **uniform** distribution over the six nonconstant bitstrings:

$$
\mu^\star=\text{Unif}\big(\{100,010,001,011,101,110\}\big)
$$

(Equivalently: put zero mass on $000$ and $111$, equal mass on Hamming-weight $1$ and $2$ states.)

A direct check shows that for any edge $c\in\{c_1,c_2,c_3\}$,

$$
q_c^\star(01)=q_c^\star(10)=\tfrac13,\qquad q_c^\star(00)=q_c^\star(11)=\tfrac16
$$

so $D_c^\star=q_c^\star(01)+q_c^\star(10)=\tfrac23$ and the off-diagonals are **balanced**, hence the BC upper bound is tight:

$$
\mathrm{BC}(p_c,q_c^\star)
=\sqrt{\tfrac12}\big(\sqrt{\tfrac13}+\sqrt{\tfrac13}\big)
=\sqrt{\tfrac23}\quad\text{for each }c
$$

Therefore

$$
\min_{c}\mathrm{BC}(p_c,q_c^\star)=\sqrt{\tfrac23}
$$

which matches the upper bound. We conclude

$$
\boxed{\ \alpha^\star(P)=\sqrt{\tfrac23}\ ,\qquad
K(P)=-\log_2\alpha^\star(P)=\tfrac12\log_2\!\frac32\ }
$$

**Corollary B.2.4 (Optimal Witness).** By symmetry, any optimal contradiction witness $\lambda^\star$ can be taken **uniform** on the three contexts. Moreover, the equalization $\mathrm{BC}(p_c,q_c^\star)=\alpha^\star$ on all edges shows $\lambda^\star$ may place positive mass on each context (cf. the support condition in the minimax duality).


## 6. Operational Interpretations

A key strength of contradiction bits is that the same quantity governs fundamental limits in multiple practical tasks. This operational convergence provides strong evidence that we've identified genuine structure rather than an arbitrary mathematical construction.

### 6.1 Detection Power: Distinguishing Real from Fake

1. **Setup**: 
You observe samples from a system and must determine whether they come from the actual multi-perspective behavior $P$ or from someone trying to fake it using any single-story explanation $Q \in \mathrm{FI}$. Nature chooses a distribution $\lambda \in \Delta(\mathcal C)$ over contexts, where $\Delta(\mathcal C)=\{\lambda\geq 0:\sum_{c\in\mathcal C}\lambda_c=1\}$.
2. **Result**: 
The worst-case per-sample error exponent you can guarantee—no matter which context distribution nature uses—is at least $K(P)$. In more favorable testing conditions, the achievable exponent can only be larger.

3. **Formal statement**: 
For any fixed context distribution $\lambda$:
    
    $$
    \alpha_\lambda(P)\;:=\;\max_{Q\in \mathrm{FI}} \sum_{c\in\mathcal C} \lambda_c\,\mathrm{BC}\!\bigl(p(\cdot|c),\,q(\cdot|c)\bigr), \qquad E_{\mathrm{BH}}(\lambda)\;:=\;-\log_2 \alpha_\lambda(P)
    $$
    
    so the Bhattacharyya bound satisfies $E_{\mathrm{opt}}(\lambda) \ge E_{\mathrm{BH}}(\lambda)$.
    
    Taking nature's least-favorable mix gives:
    
    $$
    K(P)\;=\;\min_{\lambda\in\Delta(\mathcal C)} E_{\mathrm{BH}}(\lambda) \quad\Longrightarrow\quad \inf_{\lambda} E_{\mathrm{opt}}(\lambda)\;\ge\;K(P)
    $$
    
    When the problem is balanced—i.e., under the least-favorable $\lambda^\star$ the Chernoff optimizer occurs at $s=\tfrac12$—this becomes an equality: $E_{\mathrm{opt}}(\lambda^\star)=K(P)$. If $K(P)=0$, some single story perfectly mimics the real behavior. If $K(P) > 0$, the incompatibility yields an exponential detection advantage at a guaranteed rate of at least $K(P)$ bits per sample.
    
    (Here $\mathrm{BC}(p,q)=\sum_o \sqrt{p(o)\,q(o)}$ is the Bhattacharyya coefficient.)
    
    In generic (non-balanced) instances the Chernoff optimizer does not occur at s=\tfrac12, so the achievable detection exponent strictly exceeds the Bhattacharyya-based floor K(P); equality occurs precisely when the Hellinger-weighted log-likelihood ratio has zero mean under the least-favorable mix.
    

---

### 6.2 Simulation Variance: The Cost of Faking Unity

1. **Setup**:
You want to simulate multi-perspective behavior $P$ using a single global story $Q \in \mathrm{FI}$ through importance sampling with weights $w = \frac{p}{q}$.
2. **Result**:
No matter how you choose $Q$, the **worst-case per-context variance** satisfies $\displaystyle \inf_{Q\in FI}\max_c \operatorname{Var}_{Q_c}[w_c]\;\ge\;2^{2K(P)}-1.$
3. **Proof**:
For a fixed context $c$, with importance weights $w_c=p_c/q_c$,

$$
\mathbb{E}_{Q_c}[w_c]=1,\qquad
\mathbb{E}_{Q_c}[w_c^2]=e^{D_2(p_c\|q_c)}\ge e^{D_{1/2}(p_c\|q_c)}=\mathrm{BC}(p_c,q_c)^{-2},
$$

1. so $\operatorname{Var}_{Q_c}[w_c]\ge \mathrm{BC}(p_c,q_c)^{-2}-1$.

$$
\max_c \operatorname{Var}_{Q_c}[w_c]\ge \min_c \mathrm{BC}(p_c,q_c)^{-2}-1
$$

1. Therefore, for any $Q\in\mathrm{FI}$,

$$
\inf_{Q\in\mathrm{FI}}\max_c\operatorname{Var}_{Q_c}[w_c]\ge \alpha^*(P)^{-2}-1 = 2^{2K(P)}-1
$$

1. Replacing $\min_c \mathrm{BC}(p_c,q_c)$ by $\min_{\mu}\sum_c\mu_c\,\mathrm{BC}(p_c,q_c)$ leaves the bound unchanged by the minimax equivalence (Thm 2). Hence the variance floor is governed by the same $\alpha^\star$ used elsewhere.
2. **Interpretation**:
Each additional contradiction bit multiplies 1+ the best-case variance lower bound by 4 (equivalently, the factor $2^{2K}$ quadruples). Thus the variance lower bound transforms as $B \mapsto 4B+3$.

---

### 6.3 Predictive Regret: The Price of Forced Consensus

1. **Setup**:Try to predict outcomes across all contexts using a single conditional model that must come from some unified story $Q \in \mathrm{FI}$.
2. **Result**:Under logarithmic loss (optimal for probability predictions), any such predictor suffers worst-case log-loss excess of at least $2K(P)$ bits per round. The lower bound depends on the same $\alpha^\star$ via Hellinger/Chernoff geometry; hence it is $2K(P)$ bits per round.
3. **Interpretation**: This represents irreducible overhead—no matter how you tune your single-story predictor, it must pay at least this much extra in the hardest contexts compared to context-specific optimal predictors.

---

### 6.4 Asymptotic Equipartition Property for Contradictory Sources

1. **Setup:** Consider a sequence of independent draws $(X_i,C_i)$ from distribution $p(x|c)\lambda(c)$. We observe only the outcomes $X^n$, while contexts may be hidden, known at the decoder, or explicitly transmitted.
2. **Theorem (Meta-AEP):** There exists a sequence of **witness sequences** $W_n\in\{0,1\}^{m_n}$ with rate $m_n/n \to K(P)$ such that a meta-typical set $\mathcal{T}_\varepsilon^n \subseteq \mathcal{X}^n \times \{0,1\}^{m_n}$ *satisfies $P(\mathcal{T}_\varepsilon^n) \geq 1-\varepsilon$* and
    
    $$
    \frac{1}{n}\log_2 |\mathcal{T}_\varepsilon^n|
    =\begin{cases}
    H(X)+K(P), & \textbf{latent contexts}\\[2pt]
    H(X|C)+K(P), & \textbf{known contexts at decoder}\\[2pt]
    H(C)+H(X|C)+K(P), & \textbf{headered (contexts transmitted)}
    \end{cases}
    $$
    
3. **Remarks:** When typicality is defined solely over $X^n$ (without witnesses), the standard Shannon exponents apply: $H(X)$ for latent contexts or $H(X|C)$ for known contexts. The contradiction measure $K(P)$ manifests as a supplementary information rate of $m_n/n \to K(P)$ witness bits, yielding effective compression rates of $H(\cdot)+K(P)$. This witness rate is precisely $K(P)=-\log_2\alpha^*$*, where $\alpha^*$* is computed via the minimax formulation (Theorem 2). When $K(P)=0$ (frame-independent case), our result reduces exactly to Shannon's classic AEP.
4. **Operational Interpretation:** The total coding cost decomposes into two fundamental components:
**Statistical cost:** $H(\cdot)$ — information required to compress symbols under the given context regime.
**Contradiction tax:** $K(P)$ — irreducible meta-information needed to certify or resolve perspectival incompatibility.
**Protocol design:** Optimal encoding requires allocating $nK(P)$ witness bits beyond the Shannon information budget, with context handling determining whether the base rate is $H(X)$ or $H(X|C)$.