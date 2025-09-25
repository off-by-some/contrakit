# Appendix A — Full proofs and Technical Lemmas

This appendix provides proofs and technical details for the mathematical framework introduced in *The Mathematical Theory of Contradiction*. 

**Standing Assumptions (finite case).** 

Throughout Appendix A we assume finite outcome alphabets and that $\mathrm{FI}$ (the frame-independent set) is nonempty, convex, compact, and closed under products. These conditions are satisfied in our finite setting by construction (A.1)

## A.1 Formal Setup and Definitions

## A.1.1 Basic Structures

### **Definition A.1.1 (Observable System).**

Let $\mathcal{X} = \{X_1, \ldots, X_n\}$ be a finite set of observables. For each $x \in \mathcal{X}$, fix a finite nonempty outcome set $\mathcal{O}_x$. A **context** is a subset $c \subseteq \mathcal{X}$. The outcome alphabet for context $c$ is $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$.

### **Definition A.1.2 (Behavior).**

Given a finite nonempty family $\mathcal{C} \subseteq 2^{\mathcal{X}}$ of contexts, a **behavior** $P$ is a family of probability distributions

$$
P = \{p_c \in \Delta(\mathcal{O}_c) : c \in \mathcal{C}\}
$$

where $\Delta(\mathcal{O}_c)$ denotes the probability simplex over $\mathcal{O}_c$.

**Remark (No nondisturbance required).** We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in A.1.4. When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.

### **Definition A.1.3 (Deterministic Global Assignment).**

Let $\mathcal{O}_{\mathcal{X}} := \prod_{x \in \mathcal{X}} \mathcal{O}_x$. A **deterministic global assignment** is an element $s \in \mathcal{O}_{\mathcal{X}}$. It induces a deterministic behavior $q_s$ by restriction:

$$
q_s(o \mid c) = \begin{cases} 1 & \text{if } o = s|_c \\ 0 & \text{otherwise} \end{cases}
$$

for each context $c \in \mathcal{C}$ and outcome $o \in \mathcal{O}_c$.

### **Definition A.1.4 (Frame-Independent Set).**

The **frame-independent set** is

$$
\text{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\} \subseteq \prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)
$$

### **Proposition A.1.5 (Alternative Characterization of FI).**

$Q \in \text{FI}$ if and only if there exists a **global law** $\mu \in \Delta(\mathcal{O}_{\mathcal{X}})$ such that

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

### **Definition A.1.7 (Context simplex).**

$$
\Delta(\mathcal{C}) := \{\lambda \in \mathbb{R}^{\mathcal{C}} : \lambda_c \geq 0, \sum_{c \in \mathcal{C}} \lambda_c = 1\}
$$

### **Proposition A.1.8 (Product Structure).**

Let $P$ be a behavior on $(\mathcal{X}, \mathcal{C})$ and $R$ be a behavior on $(\mathcal{Y}, \mathcal{D})$ with $\mathcal{X} \cap \mathcal{Y} = \emptyset$ (we implicitly relabel so disjointness holds). For distributions $p \in \Delta(\mathcal{O}_c)$ and $r \in \Delta(\mathcal{O}_d)$ on disjoint coordinates, $p \otimes r \in \Delta(\mathcal{O}_c \times \mathcal{O}_d)$ is $(p \otimes r)(o_c, o_d) = p(o_c)r(o_d)$.

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

### **Definition A.2.1 (Bhattacharyya Coefficient).**

For probability distributions $p, q \in \Delta(\mathcal{O})$ on a finite alphabet $\mathcal{O}$:

$$
\text{BC}(p, q) := \sum_{o \in \mathcal{O}} \sqrt{p(o) q(o)}
$$

### **Lemma A.2.2 (Bhattacharyya Properties).**

For distributions $p, q \in \Delta(\mathcal{O})$:

1. **Range**: $0 \leq \text{BC}(p, q) \leq 1$
2. **Perfect agreement**: $\text{BC}(p, q) = 1 \Leftrightarrow p = q$
3. **Joint concavity**: Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$. Therefore $(x,y)\mapsto\sqrt{xy}$ is jointly concave on $\mathbb{R}_{\geq 0}^2$ (extend by continuity on the boundary). Summing over coordinates preserves concavity, so $\text{BC}$ is jointly concave on $\Delta(\mathcal{O})\times\Delta(\mathcal{O})$.
4. **Product structure**: $\text{BC}(p \otimes r, q \otimes s) = \text{BC}(p, q) \cdot \text{BC}(r, s)$

**Proof.**

1. **Range.** 
Nonnegativity is obvious. For the upper bound, by Cauchy-Schwarz:
    
    $$
    \text{BC}(p, q) = \sum_o \sqrt{p(o) q(o)} \leq \sqrt{\sum_o p(o)} \sqrt{\sum_o q(o)} = 1
    $$
    
2. **Perfect agreement.** 
The Cauchy-Schwarz equality condition gives $\text{BC}(p, q) = 1$ iff $\sqrt{p(o)}$ and $\sqrt{q(o)}$ are proportional, i.e., $\frac{\sqrt{p(o)}}{\sqrt{q(o)}}$ is constant over $\{o : p(o) q(o) > 0\}$. Since both are probability distributions, this constant must be 1, giving $p = q$.
3. **Joint concavity.** 
Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$; extend to $\mathbb{R}_{\geq 0}^2$ by continuity. Summing over coordinates preserves concavity.
4. **Product structure.** 
Expand the tensor product and factor the sum. □

## A.3 The Agreement Measure and Minimax Theorem

### **Definition A.3.1 (Agreement and Contradiction).**

For a behavior $P$:

$$
\alpha^\star(P) := \max_{Q \in \text{FI}} \min_{c \in \mathcal{C}} \text{BC}(p_c, q_c)
$$

$$
K(P) := -\log_2 \alpha^\star(P)
$$

### **Theorem A.3.2 (Minimax Equality).**

Define the payoff function

$$
f(\lambda, Q) := \sum_{c \in \mathcal{C}} \lambda_c \text{BC}(p_c, q_c)
$$

for $\lambda \in \Delta(\mathcal{C})$ and $Q \in \text{FI}$. Then:

$$
\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \text{FI}} f(\lambda, Q)
$$

Maximizers/minimizers exist by compactness and continuity of $f$.

**Proof.** 

We apply Sion's minimax theorem (M. Sion, *Pacific J. Math.* **8** (1958), 171–176). We need to verify:

1. $\Delta(\mathcal{C})$ and FI are nonempty, convex, and compact ✓
2. $f(\lambda, \cdot)$ is concave on FI for each $\lambda$ ✓
3. $f(\cdot, Q)$ is convex (actually linear) on $\Delta(\mathcal{C})$ for each $Q$ ✓

**Details:**

- Compactness of $\Delta(\mathcal{C})$: Standard simplex.
- Compactness of FI: Proposition A.1.6.
- Concavity in $Q$: Since $Q \mapsto (q_c)_{c \in \mathcal{C}}$ is affine and each $\text{BC}(p_c, \cdot)$ is concave (Lemma A.2.2.3), the nonnegative linear combination $\sum_c \lambda_c \text{BC}(p_c, q_c)$ is concave in $Q$.
- Linearity in $\lambda$: Obvious from the definition.

By Sion's theorem, $\min_\lambda \max_Q f(\lambda, Q) = \max_Q \min_\lambda f(\lambda, Q)$. It remains to show this common value equals $\alpha^\star(P)$. 

For any $Q \in \text{FI}$, let $a_c := \text{BC}(p_c, q_c)$. Then:

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

### **Lemma A.4.1 (Uniform Law Lower Bound).**

For any behavior $P$:

$$
\alpha^\star(P) \geq \min_{c \in \mathcal{C}} \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

**Proof.** 

Let $\mu$ be the uniform (counting-measure) distribution on $\mathcal{O}_{\mathcal{X}}$ (so each global state is equally likely). This induces $Q^{\text{unif}} \in \text{FI}$ with uniform **context marginals**: $q_c^{\text{unif}}(o) = \frac{1}{|\mathcal{O}_c|}$ for all $c \in \mathcal{C}$, $o \in \mathcal{O}_c$.

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

### **Corollary A.4.2 (Bounds on K).**

For any behavior $P$:

$$
0 \leq K(P) \leq \frac{1}{2} \log_2 \left(\max_{c \in \mathcal{C}} |\mathcal{O}_c|\right)
$$

**Proof.** 

The lower bound follows from $\alpha^\star(P) \leq 1$. The upper bound follows from Lemma A.4.1 and the fact that $-\log_2(x^{-1/2}) = \frac{1}{2}\log_2(x)$. □

### **Theorem A.4.3 (Characterization of Frame-Independence).**

For any behavior $P$:

$$
\alpha^\star(P) = 1 \Leftrightarrow P \in \text{FI} \Leftrightarrow K(P) = 0
$$

**Proof.**

($\Rightarrow$) If $\alpha^\star(P) = 1$, then there exists $Q \in \text{FI}$ such that $\min_c \text{BC}(p_c, q_c) = 1$. This implies $\text{BC}(p_c, q_c) = 1$ for all $c \in \mathcal{C}$. By Lemma A.2.2, this gives $p_c = q_c$ for all $c$, hence $P = Q \in \text{FI}$.

($\Leftarrow$) If $P \in \text{FI}$, take $Q = P$ in the definition of $\alpha^\star(P)$. Then $\min_c \text{BC}(p_c, q_c) = \min_c \text{BC}(p_c, p_c) = 1$.

The equivalence with $K(P) = 0$ follows from the definition $K(P) = -\log_2 \alpha^\star(P)$. □

**Remark (No nondisturbance required).** 

We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in A.1.4. When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.

## A.5 Duality Structure and Optimal Strategies

**Theorem A.5.1 (Minimax Duality).** Let $(\lambda^\star, Q^\star)$ be optimal strategies for the minimax problem in Theorem A.3.2. Then:

1. $f(\lambda^\star, Q^\star) = \alpha^\star(P)$
2. $\text{supp}(\lambda^\star) \subseteq \{c \in \mathcal{C} : \text{BC}(p_c, q_c^\star) = \alpha^\star(P)\}$
3. If $\lambda^\star_c > 0$, then $\text{BC}(p_c, q_c^\star) = \alpha^\star(P)$

**Proof.** Existence of optimal strategies follows from compactness and continuity.

1. This is immediate from the minimax equality.
2. & 3. For fixed $Q^\star$, the inner optimization $\min_{\lambda \in \Delta(\mathcal{C})} \sum_c \lambda_c a_c$ with $a_c := \text{BC}(p_c, q_c^\star)$ has value $\min_c a_c$ and optimal solutions supported on $\arg\min_c a_c$. Since $(\lambda^\star, Q^\star)$ is optimal for the full problem, $\lambda^\star$ must be optimal for this inner problem, giving the result. □

---

## A.6  Theorem 1: The Weakest Link Principle

**Statement.**

Any unanimity-respecting (idempotent), monotone aggregator with a weakest-link cap equals the minimum.

**Assumptions.**

- $\mathcal{C}$ finite, nonempty; $A:[0,1]^{\mathcal{C}} \to [0,1]$.
- $A$ satisfies:
    1. **Monotonicity:** $x\le y \Rightarrow A(x)\le A(y)$.
    2. **Idempotence (unanimity):** $A(t,\dots,t)=t$ for all $t\in[0,1]$.
    3. **Local upper bound (weakest-link cap):** $A(x)\le x_i$ for all $i\in\mathcal{C}$.

**Claim.**
For all $x\in[0,1]^{\mathcal{C}}$, $A(x)=\min_{i\in\mathcal{C}}x_i$.

$$
A(x) \;=\; \min_{i \in \mathcal{C}} x_i \quad \text{for all } x.

$$

**Proof.**

1. Let $m=\min_{i\in\mathcal{C}} x_i$ (exists since $\mathcal{C}$ is finite and nonempty).
2. (i)+(ii): $(m,\ldots,m)\le x \Rightarrow A(x)\ge A(m,\ldots,m)=m$.
3. (iii): $A(x)\le x_i$ for all $i \Rightarrow A(x)\le m$. Hence $A(x)=m$. □

**Diagnostics / Consequences.**

1. **Bottleneck truth.** Overall agreement is set by the worst context. High scores elsewhere cannot compensate. This is the precise asymmetry: a single low coordinate rules the aggregate.
2. **Game alignment.** In §3.4 the payoff is $\max_Q \min_c \mathrm{BC}(p_c,q_c)$. The "min" over contexts is not a choice—it is forced by weakest-link aggregation under A0–A4. You do not average tests; you survive the hardest one.
3. **Diagnostics.** The contexts that attain the minimum are exactly those that receive positive weight in $\lambda^{\star}$ (the active constraints). They identify *where* reconciliation fails.
4. **Stability under bookkeeping.** Duplicating or splitting non-bottleneck contexts leaves the minimum unchanged (A4), preventing frequency from inflating consensus.
5. **Decision consequence.** For any threshold $\tau$, if some context has $BC(p_c,q_c)<\tau$, no strength elsewhere can raise the aggregate above $\tau$. Hence $K=-\log_2\alpha^{\star}$ inherits a strict "weakest-case" guarantee.

A quick contrast: with $x=(0.90,0.60,0.95)$, the minimum is $0.60$; an average would report $0.816$, overstating consensus and violating grouping/monotonicity under duplication of the weak row.

---

## A.7 Theorem 2: Contradiction as a Game (Representation)

**Statement.**
Any contradiction measure $K$ satisfying A0–A4 admits an adversarial (min–max) representation:

$$
K(P)\;=\;h\!\left(\max_{Q\in\mathrm{FI}}\ \min_{c\in\mathcal C} F\big(p_c,q_c\big)\right),
$$

for some per-context agreement functional $F: \Delta(\mathcal{O})\times\Delta(\mathcal{O})\rightarrow[0,1]$ (for each finite alphabet $\mathcal{O}$) and a strictly decreasing, continuous scalar map $h:(0,1]\to\mathbb{R}_{\ge0}$.

Equivalently,

$$
K(P)\;=\;h\!\left(\min_{\lambda\in\Delta(\mathcal C)}\ \max_{Q\in\mathrm{FI}}\ \sum_{c}\lambda_c\,F\big(p_c,q_c\big)\right).
$$

**Assumptions**

- Finite alphabets; standing §3 conditions: $\mathrm{FI}$ nonempty, convex, compact (product-closure not needed here).
- Axioms A0–A4 (Label invariance, Reduction, Continuity, Free-ops monotonicity, Grouping).
- $F$ is an agreement score on finite simplices with:
(i) normalization $F(p,p)=1$; (ii) symmetry; (iii) continuity;
(iv) **data-processing monotonicity** (DPI): $F(\Phi p,\Phi q)\ge F(p,q)$ 
       for any stochastic map $\Phi$; 
(v) **joint concavity** in $(p,q)$;
(vi) calibration $F(p,q)=1\iff p=q$.

**Claim.**
Under the assumptions, there exist $F$ and $h$ with the listed properties such that the two displays above hold, and the inner optima are attained.

**Proof.**

1. **Compare only to frame-independent surrogates (A1, A3).**
Free-ops monotonicity makes $K$ nonincreasing under post-processing and context lotteries. Hence any meaningful "distance to consensus" must be computed against $\mathrm{FI}$; comparing to non-FI $Q$'s would violate A3 under coarse-grainings. This yields the **outer $\max_{Q\in\mathrm{FI}}$**.
2. **Reduce to per-context scores (A4).**
Grouping insensitivity removes dependence on sampling multiplicities. Thus $K$ can only depend on the *set* of context-wise agreements between $P$ and a candidate $Q$, i.e., the vector $\{F(p_c,q_c)\}_c$.
3. **Force weakest-link aggregation (A0–A4 + Thm. A.6.1).**
Label invariance and continuity constrain any aggregator on $[0,1]^\mathcal C$ to be unanimity-respecting and monotone; the weakest-link cap follows from A3 (each coordinate upper-bounds any admissible aggregate). By Theorem A.6.1, the only such aggregator is the **minimum**. Hence the inner aggregator is **$\min_{c}$**.
4. **Monotone rescaling freedom.**
Calibration and continuity (A1–A2) permit a strictly decreasing, continuous reparameterization $h$ of the agreement scale $\alpha(P):=\max_{Q\in\mathrm{FI}}\min_c F(p_c,q_c) \in (0,1]$. By compactness of $\mathrm{FI}$ and continuity of $F$, the maximum is attained; $\alpha(P)\in(0,1]$, and $\alpha(P)=1$ iff $P\in\mathrm{FI}$ (by normalization and calibration of $F$). (If one uses a kernel $F$ that can attain $0$, replace $(0,1]$ by $[0,1]$; for $F=\mathrm{BC}$, Lemma A.4.1 ensures $\alpha(P)>0$.) Thus $K(P)=h(\alpha(P))$.
5. **Dual form and attainment (Sion).**
For fixed $Q$, with $a_c(Q):=F(p_c,q_c)$, $\min_{\lambda\in\Delta(\mathcal C)}\sum_c \lambda_c a_c(Q)=\min_{c} a_c(Q)$. 
Since $Q\mapsto \sum_c \lambda_c F(p_c,q_c)$ is concave on compact convex $\mathrm{FI}$ (joint concavity of $F$) and linear in $\lambda$, Sion's minimax theorem gives:
$\max_{Q\in\mathrm{FI}}\min_{c}F(p_c,q_c)\;=\;\min_{\lambda\in\Delta(\mathcal C)}\max_{Q\in\mathrm{FI}}\sum_c \lambda_c F(p_c,q_c),$
with optima attained by compactness/continuity. 
Composing with $h$ proves both displays. □

**Diagnostics / Consequences.**

1. **Worst-case testing.** A single hard context governs $\alpha(P)$ (hence $K$).
2. **Active constraints.** Minimizers $c$ attaining $\min_c F(p_c,q_c^\star)$ coincide with the support of any optimal $\lambda^\star$ in the dual (cf. A.5.1).
3. **Stability.** Grouping invariance means duplicating or splitting non-active contexts leaves $\alpha(P)$ unchanged.
4. **Bhattacharyya specialization.** Any $F$ obeying DPI + joint concavity fits the scheme; for $F=\mathrm{BC}$, $\max_{Q\in\mathrm{FI}}\min_{c}\mathrm{BC}(p_c,q_c)=\min_{\lambda\in\Delta(\mathcal C)}\max_{Q\in\mathrm{FI}}\sum_c \lambda_c \,\mathrm{BC}(p_c,q_c),$

with optima attained and $\lambda^\star$ interpretable as adversarial context weights (A.5.1).

**Sharpness.**
Drop A4 and averaging over contexts can be forced (violates weakest-link, App. B.3.6). Drop DPI and post-processing can manufacture contradiction (App. B.3.5). Either failure breaks the game form.

**Cross-refs.**

- Kernel uniqueness $\Rightarrow F=\mathrm{BC}$: Theorem A.6.3.
- Log law (pinning $h$ to $-\log$): Theorem A.6.4.
- Additivity on products: Theorem A.6.5.

## A.8 Theorem 3: Uniqueness of the Agreement Kernel (Bhattacharyya)

**Statement.**

Under refinement separability, product multiplicativity, DPI, joint concavity, and basic regularity, the unique per-context agreement kernel is the Bhattacharyya affinity:

$$
F(p,q) = \sum_o \sqrt{p(o)\,q(o)}.
$$

**Assumptions.**

$F$ maps pairs of distributions (for each finite alphabet $\mathcal O$) to $[0,1]$ and satisfies:

1. **Normalization & calibration:** $F(p,p)=1$ for all $p$; and $F(p,q)=1 \iff p=q$.
2. **Symmetry:** $F(p,q)=F(q,p)$.
3. **Continuity.**
4. **Refinement separability (label-invariant additivity across refinements):** If an outcome is refined into finitely many suboutcomes and $p,q$ are refined accordingly, then total agreement is the (label-invariant) sum of suboutcome agreements; iterating/refining in any order yields the same value.
5. **Product multiplicativity:** $F(p\otimes r, q\otimes s)=F(p,q) F(r,s)$.
6. **Data-processing inequality (DPI):** $F(\Lambda p,\Lambda q)\geq F(p,q)$ for any stochastic map $\Lambda$.
7. **Joint concavity** in $(p,q)$.

*(Existence: $\mathrm{BC}$ satisfies (1)–(7); see Lemma A.2.2.)*

**Proof.**

We proceed in three steps.

1. **Step 1 (Refinement separability $\Rightarrow$ coordinatewise sum form).**
    
    Refinement separability and label invariance imply there exists a continuous, symmetric bivariate function $g:[0,1]^2\to[0,1]$ with $g(0,0)=0$ such that for every finite alphabet $\mathcal O$ and $p,q\in\Delta(\mathcal O)$,
    
    $$
    F(p,q)\;=\;\sum_{o\in\mathcal O} g\big(p(o),\,q(o)\big).
    \tag{6.3.1}
    $$
    
    *Justification.*
    
    By refinement separability, splitting a unit mass into atoms and iterating refinements yields an additive, label-invariant decomposition; hence (6.3.1) holds for a unique $g$ with $g(0,0)=0$.
    
    Now impose diagonal normalization. 
    Define $\phi(x):=g(x,x)$. For any $p\in\Delta(\mathcal O)$,
    
    $$
    1\;=\;F(p,p)\;=\;\sum_{o} g\big(p(o),p(o)\big)\;=\;\sum_{o}\phi\big(p(o)\big).
    \tag{6.3.2}
    $$
    
    Taking $\mathcal O=\{1,2,3\}$ with probabilities $(x,y,1-x-y)$ and using (6.3.2) twice (once for $(x,y,1-x-y)$, once for $(x+y,1-x-y)$), we obtain for all $x,y\ge 0$ with $x+y\le 1$:
    $\phi(x)+\phi(y)+\phi(1-x-y)=1=\phi(x+y)+\phi(1-(x+y))$, hence $\phi(x+y)=\phi(x)+\phi(y)$.
    
    Thus $\phi$ is additive on $[0,1]$; by continuity and $\phi(1)=1$, we get
    
    $$
    \phi(x)\;=\;x\qquad\text{for all }x\in[0,1].
    \tag{6.3.3}
    $$
    
    Hence $g(x,x)=x$.
    

1. **Step 2 (Product multiplicativity $\Rightarrow$ geometric mean on each coordinate).**
    
    Consider distributions supported on a **single** atom: for $x,y,u,v\in[0,1]$, let
    
    $$
    p=(x,1-x),\quad q=(y,1-y),\quad r=(u,1-u),\quad s=(v,1-v).
    $$
    
    Then (6.3.1) and product multiplicativity (Assumption 5) give
    
    $$
    \begin{equation}\tag{6.3.4}
    \begin{aligned}
    & g(xu,yv) + g\big(x(1-u),y(1-v)\big) \\
    &\quad {}+ g\big((1-x)u,(1-y)v\big) \\
    &\quad {}+ g\big((1-x)(1-u),(1-y)(1-v)\big) \\
    &= \big[g(x,y)+g(1-x,1-y)\big] \\
    &\quad {}\times \big[g(u,v)+g(1-u,1-v)\big].
    \end{aligned}
    \end{equation}
    
    $$
    
    Since (6.3.4) holds for all $x,y,u,v\in[0,1]$, varying one variable at a time and using (6.3.1) (refinement additivity) forces each term to factorize; in particular $g(xu,yv)=g(x,y)\,g(u,v)$.
    
    By symmetry of $g$ and (6.3.4),
    
    $$
    g(x,y)^2\;=\;g(x,y)\,g(y,x)\;=\;g(xy,yx)\;=\;g(xy,xy)\stackrel{(6.3.3)}{=}\,xy.
    $$
    
    Since $F\in[0,1]$, we take the nonnegative root:
    
    $$
    g(x,y)\;=\;\sqrt{xy}\qquad\text{for all }x,y\in[0,1].
    \tag{6.3.5}
    $$
    
2. **Step 3 (Conclusion and uniqueness).**
    
    Plugging (6.3.5) into (6.3.1) yields, for every finite alphabet,
    
    $$
    F(p,q)\;=\;\sum_{o}\sqrt{p(o)\,q(o)}\;=\;\mathrm{BC}(p,q).
    $$
    
    By Lemma A.2.2, $\mathrm{BC}$ satisfies normalization, symmetry, continuity, DPI, joint concavity, and product multiplicativity. Hence $\mathrm{BC}$ meets all assumptions.
    
    For **uniqueness**, Steps 1–2 show any $F$ obeying assumptions (1)–(5) must equal the right-hand side of (6.3.5) on each coordinate; summing gives $\mathrm{BC}$. Thus there is no other admissible kernel. Assumptions (6)–(7) are then automatically satisfied by $\mathrm{BC}$ (Lemma A.2.2), and they rule out putative alternatives even if Step 1 were weakened.
    
    This completes the proof. □
    

---

**Diagnostics.**

The formula $F(p,q)=\langle \sqrt{p},\sqrt{q} \rangle$ identifies the **Hellinger embedding**; multiplicativity becomes $\langle \sqrt{p\otimes r},\sqrt{q\otimes s}\rangle = \langle \sqrt{p},\sqrt{q} \rangle \langle \sqrt{r},\sqrt{s} \rangle$; DPI and concavity follow from Jensen/Cauchy–Schwarz (Lemma A.2.2).

**Sharpness.**

Dropping *any* of refinement separability, product multiplicativity, or DPI admits non-$\mathrm{BC}$ kernels (cf. App. B.3.5–B.3.7).

**Cross-refs.**

Representation (A.6.2); log law and additivity on products use $\mathrm{BC}$ (A.6.4–A.6.5).

---

## A.9 Theorem 4: Fundamental Formula (Log Law)

**Statement.**

Under A0–A5, the contradiction measure is uniquely (up to units)

$$
K(P)\;=\;-\log_2 \alpha^\star(P),\qquad 
\alpha^\star(P)\;=\;\max_{Q\in\mathrm{FI}}\ \min_{c\in\mathcal C}\mathrm{BC}(p_c,q_c).
$$

**Assumptions.**

Finite alphabets; $\mathrm{FI}$ convex/compact and product-closed (Prop. A.1.6, A.1.8). Axioms A0–A5 (Label invariance, Reduction/Calibration, Continuity, Free-ops monotonicity/DPI, Grouping, Independent composition). Kernel uniqueness $F=\mathrm{BC}$ (Thm. A.6.3).

**Proof.**

Let

$$
\alpha^\star(P)\;:=\;\max_{Q\in\mathrm{FI}}\ \min_{c\in\mathcal C}\mathrm{BC}(p_c,q_c)\ \in (0,1].
$$

By compactness of $\mathrm{FI}$ and continuity of $\mathrm{BC}$ (Lemma A.2.2), the maximum is attained. Lemma A.4.1 ensures $\alpha^\star(P)>0$, so the logarithm below is finite.

By the representation theorem (A.6.2) specialized to $F=\mathrm{BC}$ (A.6.3), any admissible contradiction measure obeying A0–A4 must be of the form

$$
K(P)\;=\;h\!\big(\alpha^\star(P)\big)
$$

for some strictly decreasing, continuous $h:(0,1]\to \mathbb{R}_{\ge0}$ with $h(1)=0$.

A5 (independent composition) states $K(P\otimes R)=K(P)+K(R)$. Using Prop. A.1.8 (product structure of FI) and Lemma A.2.2.4 (product structure of $\mathrm{BC}$), Theorem A.6.5 (proved below) yields the **product law**

$$
\alpha^\star(P\otimes R)\;=\;\alpha^\star(P)\,\alpha^\star(R).
\tag{6.4.1}
$$

Therefore,

$$
h\!\big(\alpha^\star(P)\alpha^\star(R)\big)\;=\;h\!\big(\alpha^\star(P\otimes R)\big)\;=\;K(P\otimes R)\;=\;K(P)+K(R)\;=\;h\!\big(\alpha^\star(P)\big)+h\!\big(\alpha^\star(R)\big).
$$

Thus $h$ satisfies Cauchy's multiplicative equation on $(0,1]$:

$$
h(xy)=h(x)+h(y).
$$

Together with continuity and $h(1)=0$, the unique solutions are $h(x)=-k\log x$ with a constant $k>0$. Choosing **bits** as units fixes $k=1/\ln 2$, i.e.

$$
h(x)\;=\;-\log_2 x.
$$

Hence $K(P)=-\log_2 \alpha^\star(P)$, and writing $\alpha^\star:=\alpha$ gives the stated formula.

*Uniqueness (up to units).*

If $\tilde K$ also satisfies A0–A5, then $\tilde K=h\circ\alpha^\star$ for some $h$ as above; A5 enforces $h(x)=-k\log x$. Different choices of the constant $k$ correspond exactly to a change of units (e.g., nats vs. bits). □

**Diagnostics.**

- Additivity on independent products: $K(P\otimes R)=K(P)+K(R)$ follows from (6.4.1) and the log law.
- Calibration: $K(P)=0 \iff \alpha^\star(P)=1 \iff P\in\mathrm{FI}$ (Thm. A.4.3).
- Scale: Each halving of $\alpha^\star$ increases $K$ by one bit.

**Cross-refs.**

Product law for $\alpha^\star$ (A.6.5); bounds (A.4.1–A.4.2); representation (A.6.2); kernel uniqueness $F=\mathrm{BC}$ (A.6.3).

---

## A.10 Theorem 5: Additivity on Independent Products

**Statement.**

For independent systems on disjoint observables with product-closed $\mathrm{FI}$,

$$
\alpha^\star(P\otimes R)=\alpha^\star(P)\,\alpha^\star(R)
\quad\Longrightarrow\quad
K(P\otimes R)=K(P)+K(R).
$$

**Assumptions.**

Finite alphabets; $\mathrm{FI}$ convex/compact and product-closed (Prop. A.1.6, A.1.8); $F=\mathrm{BC}$ (Def. A.2.1; Lemma A.2.2).

**Proof.**

Let $P$ be a behavior on $(\mathcal{X},\mathcal{C})$ and $R$ on $(\mathcal{Y},\mathcal{D})$ with $\mathcal{X}\cap\mathcal{Y}=\emptyset$. Write

$$
\alpha^\star(P)=\max_{Q_A\in\mathrm{FI}_{\mathcal{X},\mathcal{C}}}\ \min_{c\in\mathcal{C}}\mathrm{BC}(p_c,q_c),
\qquad
\alpha^\star(R)=\max_{Q_B\in\mathrm{FI}_{\mathcal{Y},\mathcal{D}}}\ \min_{d\in\mathcal{D}}\mathrm{BC}(r_d,s_d).
$$

**Lower bound ($\geq$).**

Let $Q_A^\star,Q_B^\star$ be maximizers for $P$ and $R$. By product closure (Prop. A.1.8), $Q_{AB}:=Q_A^\star\otimes Q_B^\star\in\mathrm{FI}_{\mathcal{X}\sqcup\mathcal{Y},\ \mathcal{C}\otimes\mathcal{D}}$, and for any $(c,d)$,

$\mathrm{BC}\!\big(p_c\otimes r_d,\ q_c^\star\otimes s_d^\star\big)
=\mathrm{BC}(p_c,q_c^\star)\,\mathrm{BC}(r_d,s_d^\star)$ **(Lemma A.2.2, product rule)**.

Therefore

$$
\min_{(c,d)} \mathrm{BC}\!\big(p_c\otimes r_d,\ q_c^\star\otimes s_d^\star\big)
=\Big(\min_c \mathrm{BC}(p_c,q_c^\star)\Big)\Big(\min_d \mathrm{BC}(r_d,s_d^\star)\Big)
= \alpha^\star(P)\,\alpha^\star(R).
$$

Maximizing over $Q_{AB}$ yields $\alpha^\star(P\otimes R)\geq \alpha^\star(P)\alpha^\star(R)$.

**Upper bound ($\leq$).**

Use the dual form (Thm. A.3.2):

$$
\alpha^\star(P\otimes R)
=\min_{\mu\in\Delta(\mathcal{C}\otimes\mathcal{D})}
\ \max_{Q_{AB}\in\mathrm{FI}_{\mathcal{X}\sqcup\mathcal{Y}}}
\ \sum_{c,d}\mu_{c,d}\,\mathrm{BC}\!\big(p_c\otimes r_d,\ q_{c\cup d}\big).
$$

1. **Restriction of the minimizer.** Restrict the minimization to product weights $\mu=\mu_A\otimes \mu_B$ with $\mu_A\in\Delta(\mathcal{C}),\ \mu_B\in\Delta(\mathcal{D})$. Since we minimize over a smaller set, the value can only **increase**, hence this yields an **upper bound**:
    
    $$
    \alpha^\star(P\otimes R)
    \ \leq\
    \min_{\mu_A,\mu_B}\ \max_{Q_{AB}\in\mathrm{FI}}\ \sum_{c,d}\mu_A(c)\mu_B(d)\,\mathrm{BC}\!\big(p_c\otimes r_d,\ q_{c\cup d}\big).
    $$
    
2. **Restriction of the maximizer.** The objective is concave in $Q_{AB}$ (Lemma A.2.2, joint concavity; linear summation preserves concavity). Therefore the maximum over the convex compact set $\mathrm{FI}$ is attained at an extreme point (a convex combination of deterministic assignments), which factorizes across disjoint subsystems: $q_{c\cup d}=q_c\otimes s_d$. Thus we may restrict to product FI models $Q_{AB}=Q_A\otimes Q_B$ **without loss**, and certainly without violating the upper bound:
    
    $$
    \alpha^\star(P\otimes R)
    \ \leq\
    \min_{\mu_A,\mu_B}\ \max_{Q_A,Q_B}\ \sum_{c,d}\mu_A(c)\mu_B(d)\,\mathrm{BC}(p_c,q_c)\,\mathrm{BC}(r_d,s_d).
    $$
    
3. **Factorization.** Using multiplicativity of $\mathrm{BC}$ and Fubini,
    
    $$
    \sum_{c,d}\mu_A(c)\mu_B(d)\,\mathrm{BC}(p_c,q_c)\,\mathrm{BC}(r_d,s_d)
    =\Big(\sum_c \mu_A(c)\mathrm{BC}(p_c,q_c)\Big)\Big(\sum_d \mu_B(d)\mathrm{BC}(r_d,s_d)\Big).
    $$
    
    Thus
    
    $$
    \alpha^\star(P\otimes R)
    \ \leq\
    \min_{\mu_A}\max_{Q_A}\sum_c \mu_A(c)\mathrm{BC}(p_c,q_c)\ \times\
    \min_{\mu_B}\max_{Q_B}\sum_d \mu_B(d)\mathrm{BC}(r_d,s_d)
    = \alpha^\star(P)\,\alpha^\star(R).
    $$
    
    Together these give equality, $\alpha^\star(P\otimes R)=\alpha^\star(P)\alpha^\star(R)$. The additivity of $K$ follows from the log law (A.6.4): $K=-\log_2\alpha^\star$ gives
    
    $$
    K(P\otimes R)= -\log_2\big(\alpha^\star(P)\alpha^\star(R)\big)=K(P)+K(R).
    $$
    
    □
    

**Diagnostics.**

Independence composes multiplicatively at the agreement level and additively in contradiction bits; the only structural inputs are FI product structure (Prop. A.1.8), concavity and multiplicativity of $\mathrm{BC}$ (Lemma A.2.2), and the minimax dual (Thm. A.3.2).

**Cross-refs.**

$\mathrm{BC}$ multiplicativity (Lemma A.2.2); FI product structure (Prop. A.1.8); log law (A.6.4).

## A.11 Proposition: Total-Variation Gap (TVG)

**Statement.**

For any behavior $P$,

$$
d_{\mathrm{TV}}(P,\mathrm{FI}) \ :=\ \inf_{Q\in\mathrm{FI}}\ \max_{c\in\mathcal C}\ \mathrm{TV}(p_c,q_c)
\ \ \ge\ 1-\alpha^\star(P)\ =\ 1-2^{-K(P)}.
$$

**Assumptions.**

Finite alphabets; $F=\mathrm{BC}$ (Def. A.2.1); $\mathrm{FI}$ convex and compact (Prop. A.1.6).

**Proof.**

For any distributions $p,q$, one has the inequality $\mathrm{BC}(p,q)\le 1-\mathrm{TV}(p,q)$ (standard Pinsker-type bound).

Thus for each context $c$ and any $Q\in\mathrm{FI}$,

$$
\mathrm{BC}(p_c,q_c)\ \le\ 1-\mathrm{TV}(p_c,q_c).
$$

Taking the minimum over $c$ and then the maximum over $Q$,

$$
\alpha^\star(P)\ =\ \max_{Q\in\mathrm{FI}}\ \min_c \mathrm{BC}(p_c,q_c)\ \le\ 1-\inf_{Q\in\mathrm{FI}}\ \max_c \mathrm{TV}(p_c,q_c)\ =\ 1-d_{\mathrm{TV}}(P,\mathrm{FI}).
$$

Rearranging yields

$$
d_{\mathrm{TV}}(P,\mathrm{FI}) \ \ge\ 1-\alpha^\star(P).
$$

Apply the log law $K(P)=-\log_2\alpha^\star(P)$ (Thm. A.6.4) to obtain the equivalent form $d_{\mathrm{TV}}(P,\mathrm{FI}) \ge 1-2^{-K(P)}$. □

**Diagnostics.**

The bound shows that contradiction cannot be hidden in total variation: any $\mathrm{FI}$ simulator within uniform TV $\le \varepsilon$ across contexts must have $\varepsilon\ge 1-2^{-K(P)}$. Contradiction bits therefore lower-bound *observable* statistical discrepancy.

**Cross-refs.**

Bhattacharyya bound (Lemma A.2.2); log law (Thm. A.6.4); definition of $d_{\mathrm{TV}}$ (Prop. 6.L).

## A.12 Proposition: Smoothing Bound & Tightness

**Statement.**

For any behavior $P$, any $R\in\mathrm{FI}$, and any $t\in[0,1]$,

$$
K\!\big((1-t)P+tR\big)\ \le\ -\log_2\!\Big((1-t)\,2^{-K(P)}+t\Big)\ \le\ (1-t)\,K(P).
$$

This upper bound is *tight* whenever $R=Q^\star$ is an optimal FI simulator for $P$.

Moreover, to guarantee $K((1-t)P+tR)\le\kappa$ for some target $\kappa\ge0$, it suffices that

$$
t\ \ge\ \frac{1-2^{-\kappa}}{\,1-2^{-K(P)}\,}.
$$

**Assumptions.**

Finite alphabets; $\mathrm{FI}$ convex/compact (Prop. A.1.6); $\mathrm{BC}$ jointly concave in its first argument (Lemma A.2.2).

**Proof.**

Start from the dual minimax form (Thm. A.3.2):

$$
\alpha^\star(P)=\min_{\lambda\in\Delta(\mathcal C)} \max_{Q\in\mathrm{FI}}\ \sum_c \lambda_c\,\mathrm{BC}(p_c,q_c).
$$

For each $c$, concavity of $\mathrm{BC}$ in its first argument gives

$$
\mathrm{BC}\big((1-t)p_c+t r_c,\ q_c\big)\ \ge\ (1-t)\,\mathrm{BC}(p_c,q_c)+t\,\mathrm{BC}(r_c,q_c).
$$

Summing with weights $\lambda_c$ and maximizing over $Q$, then minimizing over $\lambda$, yields

$$
\alpha^\star\!\big((1-t)P+tR\big)\ \ge\ (1-t)\,\alpha^\star(P)+t\,\max_{Q\in\mathrm{FI}}\sum_c\lambda_c \mathrm{BC}(r_c,q_c).
$$

Taking $Q=R$ shows the last max is $\ge1$, hence

$$
\alpha^\star((1-t)P+tR)\ \ge\ (1-t)\,\alpha^\star(P)+t.
$$

Applying $K=-\log_2\alpha^\star$ (convex, decreasing) gives the bound

$$
K((1-t)P+tR)\ \le\ -\log_2\!\big((1-t)\,2^{-K(P)}+t\big).
$$

Convexity of $-\log$ also gives the linear relaxation $\le (1-t),K(P)$.

Tightness follows if $R=Q^\star$ is an FI optimizer for $P$: then concavity is met with equality.

The rearranged inequality

$$
t\ \ge\ \frac{1-2^{-\kappa}}{1-2^{-K(P)}}
$$

gives the minimal fraction $t$ of FI mixing needed to drive contradiction below $\kappa$. □

**Diagnostics.**

The result quantifies "FI smoothing": adding any amount of consistent noise lowers $K$ at least as fast as the displayed curve. The linear relaxation is coarse but easy to compute; the log form is exact when mixing along an optimizer.

**Cross-refs.**

Dual minimax (Thm. A.3.2); concavity of $\mathrm{BC}$ (Lemma A.2.2); log law (A.6.4).