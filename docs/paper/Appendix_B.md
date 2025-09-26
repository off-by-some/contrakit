# Appendix B — Worked Examples
This appendix provides proofs and technical details for the mathematical framework introduced in *The Mathematical Theory of Contradiction*.

## B.1 Worked Example: The Irreducible Perspective: Carrying the Frame.

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

#### B.2.1 Universal Upper Bound: $\alpha^\star\le \sqrt{2/3}$

#### Lemma B.2.2 (Upper Bound).

Let $Q\in\mathrm{FI}$ arise from a global law $\mu$ on $\{0,1\}^3$. For a context $c=\{i,j\}$, write the induced pair distribution

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

#### B.2.2 Achievability: An Explicit Optimal $\mu^\star$

#### Proposition B.2.3 (Achievability).

Let $\mu^\star$ be the **uniform** distribution over the six nonconstant bitstrings:

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

#### Corollary B.2.4 (Optimal Witness).

By symmetry, any optimal contradiction witness $\lambda^\star$ can be taken **uniform** on the three contexts. Moreover, the equalization $\mathrm{BC}(p_c,q_c^\star)=\alpha^\star$ on all edges shows $\lambda^\star$ may place positive mass on each context (cf. the support condition in the minimax duality).

## B.3 Axiom Ablation Analysis

The axiomatization developed above is not merely a convenient characterization—it reveals the fundamental structure of contradiction as a resource.

In this resource theory, frame-independent behaviors $\mathrm{FI}$ constitute the free objects, the free operations consist of 

1. outcome post-processing by arbitrary stochastic kernels $\Lambda_c$ applied within each context $c$,
2. public lotteries over contexts (independent of outcomes and hidden variables), and the contradiction measure $K$ serves as a faithful, convex, additive monotone: $K(P) \geq 0$ with equality precisely on $\mathrm{FI}$, $K$ is non-increasing under free operations, and $K$ is additive for independent systems (noting that for $|\mathcal{C}|=1$, all axioms collapse correctly and $K \equiv 0$).

To establish the robustness of this framework, we examine both weakenings and violations of each axiom. This analysis serves two purposes: it demonstrates that our axioms capture the minimal necessary structure, and it illuminates how each constraint contributes to the theory's coherence.

#### B.3.1 Axiom Weakening

Several axioms admit natural weakenings that preserve the main theoretical results while relaxing technical requirements:

**Continuity Relaxation (A2).** 

Full continuity in the product $\|\cdot\|_1$ metric can be weakened to lower semicontinuity of $K$ (equivalently, upper semicontinuity of $\alpha^\star$). This suffices because our proofs require continuity only to ensure the existence of optima and to prevent discontinuous jumps at the $\mathrm{FI}$ boundary. Lower semicontinuity guarantees both properties directly.

**Grouping Decomposition (A4).** 

The grouping axiom equivalently follows from two more primitive requirements: *replication invariance* (duplicating any context row leaves $K$ unchanged) and *context anonymity* (permutation invariance over contexts). These conditions together imply invariance under public lotteries over contexts that are independent of outcomes and hidden variables—the lottery-splitting invariance employed in our main results.

**Compositional Robustness (A5).** 

When $\mathrm{FI}_{A\times B}$ fails to be strictly product-closed, exact additivity still holds for the product-closed hull $\overline{\mathrm{FI}_A\otimes \mathrm{FI}_B}$—the smallest convex, closed set containing all products:

$$
Q_A\in \mathrm{FI}_A,\quad Q_B\in \mathrm{FI}B
\;\Longrightarrow\;
Q_A\otimes Q_B \in \mathrm{FI}{A\times B}.
$$

More generally, we obtain subadditivity $K(P\otimes R) \leq K(P) + K(R)$ with equality precisely when the optimal dual weights factorize across $A$ and $B$ (i.e., $\lambda^\star{A\times B} = \lambda^\star_A \otimes \lambda^\star_B$)—a condition that can be verified independently.

We now demonstrate that each axiom is essential by constructing explicit counterexamples that satisfy all remaining axioms while violating the target property. 

#### B.3.2 Label Invariance (A0).

*Counterexample:* Define $\tilde K(P) := \|p(\cdot\!\mid c_0) - p(\cdot\!\mid c_1)\|_1$ for fixed context labels $c_0, c_1$.
*Failure mode:* Context relabeling changes $\tilde K$ despite identical behavioral content, violating the principle that physical properties should depend only on operational statistics.

#### B.3.3 FI-Characterization (A1).

*Counterexample:* Define $\tilde K(P) := \min_c \|p(\cdot\!\mid c) - u\|_1$ where $u$ is the uniform distribution.
*Failure mode:* For $P \in \mathrm{FI}$ with non-uniform rows, $\tilde K(P) > 0$, incorrectly assigning nonzero contradiction to some frame-independent behaviors.

#### B.3.4 Continuity (A2).

*Counterexample:* Define $\tilde K(P) := \mathbf{1}\{\alpha^\star(P) < 1\}$.
*Failure mode:* Consider a sequence $P_n \to P$ with $\alpha^\star(P_n) \uparrow 1$. Then $\tilde K(P_n) = 1$ for all $n$, but $\tilde K(P) = 0$, creating a discontinuous jump at the $\mathrm{FI}$ boundary.

#### B.3.5 Data-Processing Monotonicity (A3).

*Counterexample:* Replace Bhattacharyya coefficient with linear overlap:

$$
\tilde\alpha(P) := \max_{Q\in\mathrm{FI}} \min_c \langle p_c, q_c \rangle, \quad \tilde K := -\log \tilde\alpha
$$

*Failure mode:* Linear overlap fails as a data-processing monotone. For example, take $p=q=(1,0)$ with linear overlap $\langle p,q\rangle=1$. Coarse-graining with 

$$
\Lambda=\begin{pmatrix}
\tfrac12 & \tfrac12\\[2pt]
\tfrac12 & \tfrac12
\end{pmatrix} 
$$

gives $\Lambda p=\Lambda q=(1/2,1/2)$ and $\langle \Lambda p,\Lambda q\rangle=1/2<1$, so similarity decreases under processing, allowing coarse-graining to artificially increase contradiction.

#### B.3.6 Context Grouping (A4).

*Counterexample:* Define

$$
\tilde\alpha(P) := \max_{Q\in\mathrm{FI}} \frac{1}{|\mathcal{C}|} \sum_c \mathrm{BC}(p_c, q_c), \quad \tilde K := -\log \tilde\alpha

$$

*Failure mode:* When one context acts as a bottleneck (smallest Bhattacharyya coefficient), duplicating that context row decreases the arithmetic mean, causing $\tilde K$ to increase despite unchanged behavioral content.

#### B.3.7 Additivity (A5).

*Counterexample:* Retain Bhattacharyya coefficient but use non-logarithmic aggregation $h(x) = 1-x$:

$$
\tilde K(P) = 1 - \alpha^\star(P)
$$

*Failure mode:* For independent systems $P = R$ with $\alpha^\star(P) = 1/2$, we get $\tilde K(P \otimes P) = 1 - 1/4 = 3/4$ while $\tilde K(P) + \tilde K(P) = 1$, violating additivity (indeed, this aggregator yields strict subadditivity in general). 

<aside>

Note that additivity on independent products forces $h(xy) = h(x) + h(y)$ (the functional equation for logarithms). 

With domain $h:(0,1] \to [0,\infty)$, normalization $h(1)=0$, monotonicity ($x_1 \leq x_2 \Rightarrow h(x_1) \geq h(x_2)$), and continuity, the Cauchy functional equation uniquely determines $h(x) = -k\log x$. 

Choosing bits as our unit fixes $k=1/\ln 2$, yielding $h(x) = -\log_2 x$. (Note that as $\alpha^\star \downarrow 0$, $K \to \infty$, properly capturing "complete contradiction" with infinite information cost.)

</aside>

#### B.3.8 Ablation Summary

Thus, additivity on independent systems forces a logarithmic form. Fixing base-2 units (bits) then yields the unique measure consistent with A0–A5:

$$
K(P) \;=\; -\log_2 \alpha^\star(P).
$$

Equivalently, every halving of $\alpha^\star$ increases $K$ by one bit. Any other admissible measure is merely a unit change—a positive rescaling or a different logarithm base (i.e., a constant multiple of $K$). This completes the violation analysis and confirms that our axioms capture the minimal structure necessary for a coherent theory of contradiction as a resource.

---

## B.4 Consensus as Decoding: $K(P)$ Tax

This section shows that when distributed systems must reach consensus despite frame-dependent disagreements, any protocol forcing a single committed decision pays an unavoidable overhead of exactly $K(P)$ bits per decision beyond the Shannon baseline. This is a direct consequence of the operational theorems in §6.

These are information-rate lower bounds, not round-complexity bounds. They coexist with FLP/Byzantine results: even with perfect channels and replicas, if local perspectives are structurally incompatible**,** a $K(P)$-bit tax must be paid to force consensus.

---

### B.4.1 Setup

Consider a distributed consensus protocol where replicas must agree on proposals. Each replica evaluates proposals using its own local validity predicate—rules derived from the replica's particular message history and context. The question is: what's the fundamental cost of forcing agreement when these local contexts lead to systematically different judgments?

This isn't about noisy communication or Byzantine faults. Even with perfect replicas and reliable channels, structural disagreements emerge when local contexts are legitimately incompatible.

#### Assumptions

Standing assumptions from §2 apply (finite alphabets; $\mathrm{FI}$ convex/compact; product-closure when used).

- **Finite outcomes:** All observables have finite alphabets
- **Frame-independent baseline:** The set FI of frame-independent behaviors is nonempty, convex, and compact
- **Asymptotic regime:** Either i.i.d. decisions or stationary/ergodic sequences where asymptotic equipartition applies
- **Local context availability:** Decoders may use their own local context $C$ during decoding
- **FI product-closure:** $\mathrm{FI}$ is closed under products on disjoint observables.

---

### B.4.2 Mathematical Model

#### Contexts and Behaviors

- **Local validity:** For proposal π, replica i evaluates predicate $V_i(\pi)$ based on its message history, yielding $X_i^{\pi} \in \{YES, NO\}$
- **Contexts:** A context c represents a subset of replicas observed under particular scheduling conditions (e.g., pair {i,j} with their joint reports)
- **System behavior:** The system induces $P = \{p_c\}_{c\in\mathcal{C}}$, a family of outcome distributions, one per context
- **Contradiction measure:** With Bhattacharyya overlap $\mathrm{BC}(p,q)=\sum_o \sqrt{p(o)q(o)}$, we define:

$$
\alpha^\star(P)=\max_{Q\in\mathrm{FI}}\min_{c\in\mathcal{C}}\mathrm{BC}(p_c,q_c),\qquad
K(P)=-\log_2 \alpha^\star(P)
$$

By our axioms, $K(P)$ uniquely measures contradiction: $K(P) = 0$ if and only if all contexts can be reconciled by a single frame-independent account.

---

### B.4.3 Why Odd Cycles Create Contradiction

Suppose an adversary schedules messages so three correct replicas A, B, C produce pairwise disagreements in pairwise contexts:

- Context {A,B}: $p_{AB}(YN)=p_{AB}(NY)=\tfrac12$, others 0
- Context {B,C}: $p_{BC}(YN)=p_{BC}(NY)=\tfrac12$, others 0
- Context {C,A}: $p_{CA}(YN)=p_{CA}(NY)=\tfrac12$, others 0

This creates an "odd cycle" of constraints: $X_A \neq X_B$, $X_B \neq X_C$, $X_C \neq X_A$. Precisely what we saw within our Lenticular Coin. No global assignment can satisfy all three simultaneously.

**Lemma B.4.1:**

If each pairwise context (A,B), (B,C), (C,A) assigns **zero** probability to equal outcomes (i.e., supports only $X_A \neq X_B$, $X_B \neq X_C$, $X_C \neq X_A$), then $P\notin \mathrm{FI}$ and $K(P)>0$.

For this symmetric anti-correlation pattern, $K(P) = \tfrac12\log_2(3/2) \approx 0.29$ bits per decision.

**Proof sketch:**

The optimal frame-independent approximation assigns outcomes uniformly over the six "not-all-equal" patterns of $(X_A, X_B, X_C)$. Each pairwise marginal then has Bhattacharyya overlap $\sqrt{2/3}$ with the observed anti-correlation, giving $\alpha^\star(P) = \sqrt{2/3}$ and $K(P) = \tfrac12\log_2(3/2)$.

---

### B.4.4 Consensus as Common Decoding

A consensus decision is fundamentally a **common representation problem**: we need a finite string Z that every correct replica, using its own local context, can decode to recover the same decision sequence $X^n$.

This is precisely the setup analyzed in our operational theorems. Different replicas effectively use different "codebooks" based on their local contexts, yet they must all decode to the same outcome.

---

### B.4.5 The Fundamental Lower Bound

**Theorem B.4.2 (The K(P) Tax):**

Any consensus scheme outputting a single representation $Z$ decodable by every context requires communication rate: (Theorems 11–12)

$$
\tfrac1n\mathbb{E}[\ell(Z)] \ge
\begin{cases}
H(X\mid C)+K(P)-o(1), & \text{decoders know } C^n\\
H(X)+K(P)-o(1), & \text{decoders don't}
\end{cases}
$$

where $H(X|C)$ is the standard entropy given context information, and $o(1) \to 0$ as block length $n \to \infty$.

**What this means:**

Beyond the usual Shannon entropy cost $H(X|C)$, there's an additional $K(P)$ bits per decision needed to force consensus when contexts naturally disagree.

This overhead is:

- **Unavoidable:** No protocol can do better
- **Tight:** The bound is achievable to within $o(1)$
- **Mechanism-independent:** Whether you use extra metadata, additional rounds, committee proofs, or side channels

**Corollary B.4.3 (Channel Capacity):**

Under source–channel separation, a channel with Shannon capacity $C_{\mathrm{Shannon}}$ can only carry usable consensus payload at rate:

$$
C_{\mathrm{common}} = C_{\mathrm{Shannon}} - K(P)
$$

**Corollary B.4.4 (Witness-Error Tradeoff):**

If r is the witness rate (extra bits beyond Shannon baseline) and E is the level-$\eta$ type-II exponent for detecting frame contradictions, then: (Theorem 7.4)

$$
E + r \geq K(P)
$$

You can trade witness bits for statistical detectability, but their sum is bounded by $K(P)$.

### B.4.6 Constructive Witnesses

The bound is achievable: there exist witness strings $W_n$ of rate $K(P)+o(1)$ such that $\mathrm{TV}((X^n,W_n),\tilde{Q}_n)\to 0$. (Theorem 10)

**Implementation flexibility:**

The same information budget can be realized through:

- Extra metadata fields
- Additional communication rounds
- Cryptographic proofs
- Side channel coordination
- Schema negotiation

The lower bound constrains the total information cost, not the specific mechanism.