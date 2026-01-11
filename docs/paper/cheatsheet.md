# A Mathematical Theory of Contradiction - Complete Cheatsheet

**Author:** Cassidy Bridges

---

## Table of Contents

- [**🎯 Quick Start**](#-quick-start) - Essential formulas at a glance
- [**🏗️ Core Foundations**](#-core-foundations) - Basic definitions and measurement
- [**⚖️ Axioms**](#-axioms) - Fundamental properties
- [**📐 Theorems**](#-theorems) - Representation and uniqueness results
- [**🔧 Operational Applications**](#-operational-applications) - Information-theoretic consequences
- [**🧮 Practical Tools**](#-practical-tools) - Computation and estimation
- [**📚 Reference**](#-reference) - Examples, bounds, and technical details

---

## 🎯 Quick Start

### Essential Definition
**Contradiction Measure:**

$$
K(P) = -\log_2 \alpha^\star(P) \quad \text{where} \quad \alpha^\star(P) = \max_{Q \in \mathrm{FI}} \min_c \mathrm{BC}(p_c, q_c)
$$

**What does this mean?** K(P) measures how many extra bits you need when different observers see the same data differently but you want one unified explanation. It's like a "compatibility tax" - zero if everyone agrees, positive if they fundamentally disagree about what happened. The number tells you exactly how much information overhead you pay for reconciliation.

### Key Properties
- $K(P) = 0$ iff $P$ is frame-independent (no contradiction)
- $K(P \otimes R) = K(P) + K(R)$ (additive for independent systems)
- Operational tax: $K(P)$ bits per observation in compression, communication, testing

**What does this mean?** These properties tell you when contradiction matters: contradiction is zero only when all perspectives can be unified, and the costs add up for independent problems. Every time you use compression, send messages, or test hypotheses with contradictory data, you pay exactly K(P) extra bits per observation.

### Most Important Theorems
- **Theorem 4:** $K(P) = -\log_2 \alpha^\star(P)$ (fundamental formula)
- **Theorem 6:** Compression rates increase by $K(P)$ bits
- **Theorem 13:** Channel capacity decreases by $K(P)$ bits
- **Theorem 7.4:** Witness-error tradeoff: $E + r \geq K(P)$

---

## 🏗️ Core Foundations

### System Model
- **Observable System:** $\mathcal{X} = \{X_1, \ldots, X_n\}$ observables, each with outcome set $\mathcal{O}_x$
- **Context:** $c \subseteq \mathcal{X}$, outcome space $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$
- **Behavior:** $P = \{p_c : c \in \mathcal{C}\}$ probability distributions over $\mathcal{O}_c$

**What does this mean?** Think of a system where different people look at the same thing from different angles (contexts). Each context sees a different subset of measurements. A behavior P records what every possible context would observe. This models situations where the same underlying reality looks different depending on your viewpoint or measurement setup.

### Frame-Independent Baseline
**Definition:** The set of behaviors admitting unified explanations

$$
\mathrm{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\}
$$
*Where $q_s$ are deterministic behaviors (discrete case); extends to joint density representations (continuous case)*

**What does this mean?** FI is the "easy case" - behaviors where there's one true underlying reality that explains everything all observers see. If your system is in FI, there's no fundamental contradiction between different viewpoints. Any behavior outside FI has genuine incompatibility that can't be explained away.

### Agreement Measure
**Bhattacharyya Coefficient:** Quantifies distributional overlap

$$
\mathrm{BC}(p,q) = \sum_o \sqrt{p(o) q(o)} \quad (\text{discrete}) \quad \text{or} \quad \int \sqrt{p(x) q(x)} \, dx \quad (\text{continuous})
$$

**Key Properties:**
- Range: $[0,1]$ (0 = no overlap, 1 = identical)
- Perfect agreement: $\mathrm{BC}(p,q) = 1 \iff p = q$
- Jointly concave and multiplicative: $\mathrm{BC}(p \otimes r, q \otimes s) = \mathrm{BC}(p,q) \cdot \mathrm{BC}(r,s)$

**What does this mean?** BC measures how similar two probability distributions are. It's like a sophisticated version of correlation that works for any distributions. When BC = 1, the distributions are identical; when BC = 0, they have completely different outcomes. We use this to measure how well different contexts agree with each other.

---

### Contradiction Measurement

**Core Definition:**
$$
\alpha^\star(P) = \max_{Q \in \mathrm{FI}} \min_{c \in \mathcal{C}} \mathrm{BC}(p_c, q_c), \quad K(P) = -\log_2 \alpha^\star(P)
$$

**What does this mean?** α* finds the best possible unified explanation (from FI) and measures how well it matches your actual behavior. The worst-matching context determines the score. K(P) converts this into bits - it's the log of how much worse your system is than the best unified explanation.

**Alternative Characterization:**
$$
\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \mathrm{FI}} \sum_c \lambda_c \cdot \mathrm{BC}(p_c, q_c)
$$

**What does this mean?** This is the same calculation but weighted by context importance. λ tells you which contexts are the "weakest link" - the ones that cause the most contradiction. The adversarial perspective: nature picks context weights to maximize the contradiction you see.

**Fundamental Properties:**
$$
K(P) = 0 \iff P \in \mathrm{FI} \iff \alpha^\star(P) = 1, \quad 0 \leq K(P) \leq \frac{1}{2} \log_2 (\max_c |\mathcal{O}_c|)
$$

**What does this mean?** These bounds tell you what K(P) values are possible. Zero only when there's no contradiction. The upper bound depends on how many outcomes each context can see - more possibilities allow more contradiction.

---

## ⚖️ Axioms

**Six fundamental properties uniquely determine the contradiction measure:**

| Axiom | Property | Intuition | What does this mean? |
|-------|----------|-----------|---------------------|
| **A0** | Label Invariance | Contradiction is structural, not notational | Relabeling outcomes or contexts shouldn't change the contradiction level. The incompatibility is about the pattern of disagreement, not what you call the labels. |
| **A1** | Reduction | Zero iff frame-independent (no contradiction) | No contradiction means you can unify all perspectives. This gives the scale a natural zero point. |
| **A2** | Continuity | Small changes → small contradiction changes | Tiny tweaks to probabilities shouldn't cause huge jumps in contradiction measurement. |
| **A3** | Free Operations | Monotone under legitimate transformations | Adding noise, averaging perspectives, or combining systems shouldn't create contradiction where none existed. |
| **A4** | Grouping | Depends only on refined statistics | How you group observations shouldn't change the fundamental contradiction level. |
| **A5** | Independent Composition | Additive for disjoint systems | Contradictions from separate, independent problems should just add up. |

**A3: Free Operations** - $K$ monotone under:
- Stochastic post-processing within contexts
- Convex mixtures: $K((1-t)P + tQ) \leq \max(K(P), K(Q))$
- Public lotteries over contexts
- Tensoring with FI ancillas: $K(P \otimes R) \leq K(P)$ for $R \in \mathrm{FI}$

**A5: Independent Composition** - For disjoint observables:

$$
K(P \otimes R) = K(P) + K(R)
$$

---

## 📐 Theorems

### Representation Theory

**Theorem 1** *(Weakest Link Principle)*:
Any unanimity-respecting, monotone aggregator with weakest-link property equals the minimum:
$$
\mathrm{A}(x) = \min_i x_i \quad \text{for all} \quad x \in [0,1]^{\mathcal{C}}
$$

**What does this mean?** Among reasonable ways to combine multiple opinions, the only fair one is to take the worst-case opinion. This justifies why contradiction is measured by the worst-agreeing context.

**Theorem 2** *(Contradiction as Minimax Game)*:
Any contradiction measure satisfying A0-A4 admits minimax representation:
$$
K(P) = h\left(\max_Q \min_c F(p_c, q_c)\right) = h\left(\min_\lambda \max_Q \sum_c \lambda_c F(p_c, q_c)\right)
$$
for some strictly decreasing continuous $h$ and agreement kernel $F$

**What does this mean?** Contradiction can always be framed as a two-player game: one player tries to find the best unified explanation, the other tries to find the worst disagreement. This shows contradiction is a fundamental adversarial phenomenon.

### Uniqueness & Fundamental Results

**Theorem 3** *(Unique Agreement Kernel)*:
Under refinement separability, product multiplicativity, DPI, joint concavity, and regularity:
$$
F(p,q) = \sum_o \sqrt{p(o) q(o)} = \mathrm{BC}(p,q)
$$

**What does this mean?** There's only one reasonable way to measure agreement between distributions. The Bhattacharyya coefficient is uniquely determined by natural mathematical properties.

**Theorem 4** *(Fundamental Formula - Log Law)*:
Under axioms A0-A5, the contradiction measure takes the form:
$$
K(P) = -\log_2 \alpha^\star(P)
$$

**What does this mean?** The contradiction measure must be logarithmic in the agreement level. This makes contradiction additive (like information) and gives it the right units (bits).

**Theorem 5** *(Additivity on Products)*:
For independent systems on disjoint observables with FI product-closed:
$$
\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R), \quad K(P \otimes R) = K(P) + K(R)
$$

**What does this mean?** Contradictions from separate, independent systems multiply for agreement but add for bits. This is why contradiction behaves like information - it composes additively across independent components.

---

## 🔧 Operational Applications

### Asymptotic Equipartition & Compression

**Theorem 6** *(AEP with Contradiction Tax)*:
$$
\lim_{n\to\infty} \frac{1}{n} \log_2 |\mathcal{S}_n| \geq H(X|C) + K(P)
$$

**What does this mean?** When you have many observations from contradictory data, the number of typical patterns grows exponentially with rate H(X|C) + K(P). The K(P) term is the extra "complexity tax" you pay because different contexts disagree.

**Theorem 7** *(Compression, Known Contexts)*:
$$
\lim_{n\to\infty} \frac{1}{n} \mathbb{E}[\ell_n^\star] = H(X|C) + K(P)
$$

**What does this mean?** The best compression rate for contradictory data is H(X|C) + K(P) bits per observation when you know which context generated each observation. You need K(P) extra bits to handle the context disagreements.

**Theorem 8** *(Compression, Latent Contexts)*:
$$
\lim_{n\to\infty} \frac{1}{n} \mathbb{E}[\ell_n^\star] = H(X) + K(P)
$$

**What does this mean?** When contexts are unknown, compression costs H(X) + K(P) bits. You need the full entropy H(X) plus K(P) to handle both the uncertainty and the contradiction across unknown contexts.

**Corollary 6.1** *(Meta-AEP Regimes)*:
With witnesses $W_n$ of rate $K(P)$, meta-typical sets satisfy:
$$
\frac{1}{n}\log_2|\mathcal T_\varepsilon^n| = \begin{cases}
H(X)+K(P) & \text{(latent contexts)} \\
H(X|C)+K(P) & \text{(known contexts)} \\
H(C)+H(X|C)+K(P) & \text{(contexts in header)}
\end{cases}
$$

**What does this mean?** By including witness information at rate K(P), you can achieve different compression regimes. The witness essentially "explains" the contradiction, letting you compress as if contexts were unified.

### Hypothesis Testing & Simulation

**Theorem 9** *(Testing Frame-Independence)*:
Optimal type-II error exponent satisfies $E \geq K(P)$

**What does this mean?** When testing if data is frame-independent (unified) versus contradictory, you can never reject the unified hypothesis faster than rate K(P). Contradiction fundamentally limits how sharply you can distinguish unified from contradictory explanations.

**Theorem 10** *(Witnessing for Approximation)*:
Rate $K(P)+o(1)$ achieves vanishing total variation: $\mathrm{TV}((X^n, W_n), \tilde{Q}_n) \to 0$

**What does this mean?** You can simulate contradictory data using a unified model plus K(P) bits of witness information per observation. This shows contradiction can be "witnessed" - made compatible by adding the right side information.

### Communication & Coding

**Theorem 11** *(Common Message Problem)*:
Rate $\geq H(X|C) + K(P)$ for messages decodable by all contexts

**Theorem 12** *(Common Representation Cost)*:
Rate $\geq H(X|C) + K(P)$ (known contexts) or $H(X) + K(P)$ (latent contexts)

**Theorem 13** *(Channel Capacity with Common Decoding)*:
$$
C_{\text{common}} = C_{\text{Shannon}} - K(P)
$$

**What does this mean?** When all receivers must decode the same message despite having contradictory contexts, the effective channel capacity drops by K(P) bits. Different "interpretations" of the received signal reduce communication efficiency.

**Theorem 14** *(Rate-Distortion with Common Reconstruction)*:
$$
R(D) = R_{\text{Shannon}}(D) + K(P)
$$

**What does this mean?** Lossy compression with a single reconstruction for all contexts costs K(P) extra bits. You can't losslessly compress contradictory data to a single representation without paying the contradiction tax.

### Geometric Structure

**Theorem 15** *(Contradiction Geometry)*:
(a) Hellinger metric: $J(A,B) = \max_c \arccos(\mathrm{BC}(p_c^A, p_c^B))$
(b) Subadditivity: $J(P \otimes R) \leq J(P) + J(R)$
(c) Log-additivity: $K(P \otimes R) = K(P) + K(R)$

---

## 🔍 Operational Interpretations

### Testing & Discrimination

**Proposition 7.1** *(Testing Real vs Frame-Independent)*:
$$
\inf_\lambda E_{\text{opt}}(\lambda) \geq \min_\lambda E_{\mathrm{BH}}(\lambda) = K(P)
$$

**What does this mean?** The best you can do when testing if real data matches a unified model is limited by K(P). If the data has contradiction K(P), you'll always have at least that much uncertainty in distinguishing it from unified data.

**Theorem 7.4** *(Witness-Error Tradeoff)*:
For witness rate $r$, type-II exponent $E$:
$$
E + r \geq K(P), \quad E^*(r) = K(P) - r \quad \text{for} \quad r \in [0, K(P)]
$$

### Prediction & Simulation

**Proposition 7.2** *(Importance Sampling Penalty)*:

$$
\inf_{Q \in \mathrm{FI}} \max_c \mathrm{Var}_{Q_c}[w_c] \geq 2^{2K(P)} - 1
$$

**Proposition 7.3** *(Single-Predictor Regret)*:

$$
\inf_{Q \in \mathrm{FI}} \max_c \mathbb{E}_{p_c}[\log_2(p_c(X)/q_c(X))] \geq 2K(P) \text{ bits/round}
$$

### Universal Structure

**Theorem 7.5** *(Universal Adversarial Structure)*:
Optimal $\lambda^\star$ simultaneously optimal for testing, simulation, and coding

### Geometric Properties

**Proposition 7.6** *(Hellinger Sphere Structure)*:

$$
\alpha^\star(P) = 1 - D_H^2(P, \mathrm{FI}) \quad \text{where} \quad D_H^2(P, \mathrm{FI}) = \min_Q \max_c H^2(p_c, q_c)
$$

**Corollary 7.6.1** *(Level Set Geometry)*:
Level sets $\{P: K(P) = \kappa\} = $ outer Hellinger Chebyshev spheres of radius $\sqrt{1 - 2^{-\kappa}}$ around FI

**Corollary 7.6.2** *(TV Gap)*:
$$
d_{\mathrm{TV}}(P, \mathrm{FI}) \geq 1 - 2^{-K(P)}
$$

### Smoothing & Interpolation

**Proposition 7.7** *(Smoothing Bound)*:

$$
K((1-t)P + tR) \leq -\log_2((1-t)2^{-K(P)} + t) \leq (1-t)K(P)
$$

**Corollary 7.7.1** *(Minimal Smoothing)*:
To reduce contradiction to $\leq \kappa$:
$$
t \geq \frac{1 - 2^{-\kappa}}{1 - 2^{-K(P)}}
$$

### Computational Properties

**Proposition 7.8** *(Convex Program for K)*:

$$
D_H^2(P, \mathrm{FI}) = \min_\mu \max_c H^2(p_c, q_c(\mu)), \quad K(P) = -\log_2(1 - D_H^2(P, \mathrm{FI}))
$$

**Theorem 7.9** *(Equalizer Principle + Sparse Optimizers)*:
Active contexts satisfy $\mathrm{BC}(p_c, q_c^\star) = \alpha^\star(P)$, with optimal $Q^\star$ supported on at most $1 + \sum_c (|\mathcal{O}_c| - 1)$ deterministic assignments

---

## 🧮 Practical Tools

### Computational Methods

**Convex Program Formulation:**

$$
\min_\mu \max_c H^2(p_c, q_c(\mu)) \quad \text{where} \quad q_c(\mu) = \sum_s \mu(s) \cdot \delta_{s|_c}
$$

**What does this mean?** To compute K(P), solve this optimization problem. It finds the best unified explanation (μ) that minimizes the worst-case disagreement with your actual data. Modern convex optimization solvers can handle this efficiently.

**Minimax Program:**
$$
\max_Q \min_c \mathrm{BC}(p_c, q_c) \quad \text{solved via Sion's theorem duality}
$$

**What does this mean?** This is an equivalent formulation where you find the unified explanation that maximizes the minimum agreement across contexts. Sion's theorem guarantees this minimax equals the original max-min.

### Statistical Estimation

**Plug-in Estimator:**
$\hat{K}$ from empirical frequencies with bootstrap confidence intervals

**What does this mean?** From data samples, estimate the context distributions, then plug them into the K(P) formula. Bootstrap gives you confidence intervals so you know how reliable your estimate is.

**Regularized Estimation:**
Add pseudocounts $\epsilon > 0$ for consistency when zeros occur:

$$
\tilde{p}_o = \frac{n_o + \epsilon}{n + k\epsilon} \quad \text{where } k \text{ is number of outcomes}
$$

**What does this mean?** When you have small datasets, some outcomes might never appear. Adding a tiny pseudocount (like 0.01) to all outcomes prevents impossible zero probabilities and makes the estimator statistically well-behaved.

---

## 📚 Reference

### Technical Lemmas

**Lemma A.4.1** *(Uniform Law Lower Bound)*:
$$
\alpha^\star(P) \geq \min_c \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

**Theorem A.5.1** *(Minimax Duality)*:
$\lambda^\star$ locates tension; $Q^\star$ maximizes agreement; active contexts saturate bound

**Odd-Cycle Bounds:**

$$
\alpha^\star(P) \leq \sqrt{2/3}, \quad \mu^\star = \text{Unif}(\{100,010,001,011,101,110\})
$$

**Lemma B.4.1** *(Odd Cycles Create Contradiction)*:
Pairwise anti-correlations imply $K(P) > 0$

### Worked Examples

**Lenticular Coin** *(Minimal Contradiction Device)*:
Three contexts with perfect disagreement, yielding:

$$
\alpha^\star = \sqrt{2/3}, \quad K = \frac{1}{2} \log_2(3/2) \approx 0.29 \text{ bits per observation}
$$

**What does this mean?** This is the simplest possible contradictory system - three observers looking at a lenticular image that shows different answers from different angles. No matter how you try to unify their reports, you always pay 0.29 bits per observation in overhead.

**Gaussian Contradiction** *(Continuous Extension)*:
For N(0,σ₁²) vs N(0,σ₂²) measurement conflicts:

$$
\alpha^\star(P) = \sqrt{\frac{2 \sqrt{\sigma_1 \sigma_2}}{\sigma_1 + \sigma_2}}, \quad K(P) = -\log_2 \sqrt{\frac{2 \sqrt{\sigma_1 \sigma_2}}{\sigma_1 + \sigma_2}}
$$

**Example:** σ₁=1, σ₂=4 → α* ≈ 0.894, K ≈ 0.161 bits

**What does this mean?** When two measurement devices have different precision (σ₁ ≠ σ₂), their readings can't be perfectly reconciled. The contradiction depends on how different the precisions are - more similar precisions mean less contradiction.

### Key Identities

**Product Law:**

$$
\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R), \quad K(P \otimes R) = K(P) + K(R)
$$

**Hellinger Geometry:**

$$
H(p,q) = \sqrt{1 - \mathrm{BC}(p,q)}, \quad D_H^2(P, \mathrm{FI}) = \min_Q \max_c H^2(p_c, q_c), \quad \alpha^\star(P) = 1 - D_H^2(P, \mathrm{FI})
$$

**Information Bounds:**

$$
d_{\mathrm{TV}}(P, \mathrm{FI}) \geq 1 - 2^{-K(P)}, \quad E + r \geq K(P)
$$

**Smoothing Properties:**

$$
K((1-t)P + tR) \leq -\log_2((1-t)2^{-K(P)} + t), \quad t \geq \frac{1 - 2^{-\kappa}}{1 - 2^{-K(P)}}
$$

**Stability:** High stability - reducing K(P) > 0 to near-zero requires t ≈ 100% FI mixture

---