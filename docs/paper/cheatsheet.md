# A Mathematical Theory of Contradiction - Complete Cheatsheet

**Author:** Cassidy Bridges  
**Core Result:** $K(P) = -\log_2 \alpha^\star(P)$ where $\alpha^\star(P) = \max_{Q \in \mathrm{FI}} \min_c \mathrm{BC}(p_c, q_c)$

---

## 1. Core Definitions

### Basic Structures
- **Observable System:** $\mathcal{X} = \{X_1, \ldots, X_n\}$ observables, each with outcome set $\mathcal{O}_x$ (finite discrete or continuous)
- **Context:** $c \subseteq \mathcal{X}$, outcome space $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$
- **Behavior:** $P = \{p_c : c \in \mathcal{C}\}$ probability distributions over $\mathcal{O}_c$ (discrete or continuous)

### Frame-Independent Set
$\mathrm{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\}$ where $q_s$ are deterministic behaviors (discrete case); extends to distributions admitting joint density representations (continuous case)

### Bhattacharyya Coefficient
$\mathrm{BC}(p,q) = \sum_o \sqrt{p(o) q(o)}$ (discrete) or $\int \sqrt{p(x) q(x)} \, dx$ (continuous)
- Range: $[0,1]$
- Perfect agreement: $\mathrm{BC}(p,q) = 1 \iff p = q$
- Jointly concave, multiplicative: $\mathrm{BC}(p \otimes r, q \otimes s) = \mathrm{BC}(p,q) \cdot \mathrm{BC}(r,s)$

---

## 2. Core Measurement

### Contradiction Measure
$\alpha^\star(P) = \max_{Q \in \mathrm{FI}} \min_{c \in \mathcal{C}} \mathrm{BC}(p_c, q_c)$

$K(P) = -\log_2 \alpha^\star(P)$

### Minimax Representation
$\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \mathrm{FI}} \sum_c \lambda_c \cdot \mathrm{BC}(p_c, q_c)$

### Characterization
$K(P) = 0 \iff P \in \mathrm{FI} \iff \alpha^\star(P) = 1$

### Bounds
$0 \leq K(P) \leq \frac{1}{2} \log_2 (\max_c |\mathcal{O}_c|)$ (discrete case); continuous case bounds under investigation

---

## 3. Axioms (A0-A5)

**A0: Label Invariance** - $K$ invariant under outcome/context permutations

**A1: Reduction** - $K(P) = 0 \iff P \in \mathrm{FI}$

**A2: Continuity** - $K$ continuous in product $L_1$ metric

**A3: Free Operations** - $K$ monotone under:
- Stochastic post-processing within contexts
- Convex mixtures
- Public lotteries over contexts
- Tensoring with FI ancillas

**A4: Grouping** - $K$ depends only on refined statistics; insensitive to context replication/merging via public lotteries

**A5: Independent Composition** - $K(P \otimes R) = K(P) + K(R)$

---

## 4. Representation & Uniqueness Theorems

### Theorem 1: Weakest Link Principle
Any unanimity-respecting, monotone aggregator with weakest-link cap equals the minimum:
$\mathrm{A}(x) = \min_i x_i$ for all $x \in [0,1]^{\mathcal{C}}$

### Theorem 2: Contradiction as Game
Any $K$ satisfying A0-A4 admits minimax form:
$K(P) = h(\max_Q \min_c F(p_c, q_c)) = h(\min_\lambda \max_Q \sum_c \lambda_c F(p_c, q_c))$
for some strictly decreasing continuous $h$ and agreement kernel $F$

### Theorem 3: Uniqueness of Agreement Kernel
Under refinement separability, product multiplicativity, DPI, joint concavity, and regularity:
$F(p,q) = \sum_o \sqrt{p(o) q(o)} = \mathrm{BC}(p,q)$

### Theorem 4: Fundamental Formula (Log Law)
Under A0-A5: $K(P) = -\log_2 \alpha^\star(P)$

### Theorem 5: Additivity on Products
For independent systems on disjoint observables with FI product-closed:
$\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R)$
$K(P \otimes R) = K(P) + K(R)$

---

## 5. Operational Theorems

### Theorem 6: AEP with Contradiction Tax
$\lim_{n\to\infty} \frac{1}{n} \log_2 |\mathcal{S}_n| \geq H(X|C) + K(P)$ for high-probability sets

**Corollary 6.1** *(Meta-AEP with Three Regimes)*
With witnesses $W_n\in\{0,1\}^{m_n}$ and $m_n/n\to K(P)$, meta-typical sets $\mathcal T_\varepsilon^n$ with $P(\mathcal T_\varepsilon^n)\ge 1-\varepsilon$ and
$\frac{1}{n}\log_2|\mathcal T_\varepsilon^n| = \begin{cases} H(X)+K(P), & \text{latent contexts},\\ H(X\mid C)+K(P), & \text{known contexts at decoder},\\ H(C)+H(X\mid C)+K(P), & \text{contexts in message header.} \end{cases}$

### Theorem 7: Optimal Compression (Known Contexts)
$\lim_{n\to\infty} \frac{1}{n} \mathbb{E}[\ell_n^\star] = H(X|C) + K(P)$

### Theorem 8: Optimal Compression (Latent Contexts)
$\lim_{n\to\infty} \frac{1}{n} \mathbb{E}[\ell_n^\star] = H(X) + K(P)$

### Theorem 9: Testing Frame-Independence
Optimal type-II exponent: $E \geq K(P)$

### Theorem 10: Witnessing for TV-Approximation
Rate $K(P)+o(1)$ achieves vanishing TV: $\mathrm{TV}((X^n, W_n), \tilde{Q}_n) \to 0$

### Theorem 11: Common Message Problem
Rate $\geq H(X|C) + K(P)$ for messages decodable by all contexts

### Theorem 12: Common Representation Cost
Rate $\geq H(X|C) + K(P)$ (known contexts) or $H(X) + K(P)$ (latent)

### Theorem 13: Channel Capacity with Common Decoding
$C_{\text{common}} = C_{\text{Shannon}} - K(P)$

### Theorem 14: Rate-Distortion with Common Reconstruction
$R(D) = R_{\text{Shannon}}(D) + K(P)$

### Theorem 15: Contradiction Geometry
(a) Hellinger metric: $J(A,B) = \max_c \arccos(\mathrm{BC}(p_c^A, p_c^B))$
(b) Subadditivity: $J(P \otimes R) \leq J(P) + J(R)$
(c) Log-additivity: $K(P \otimes R) = K(P) + K(R)$

---

## 6. Operational Interpretations

### Proposition 7.1: Testing Real vs Frame-Independent
$\inf_\lambda E_{\text{opt}}(\lambda) \geq \min_\lambda E_{\mathrm{BH}}(\lambda) = K(P)$

### Proposition 7.2: Importance Sampling Penalty
$\inf_{Q \in \mathrm{FI}} \max_c \mathrm{Var}_{Q_c}[w_c] \geq 2^{2K(P)} - 1$

### Proposition 7.3: Single-Predictor Regret
$\inf_{Q \in \mathrm{FI}} \max_c \mathbb{E}_{p_c}[\log_2(p_c(X)/q_c(X))] \geq 2K(P)$ bits/round

### Theorem 7.4: Witness-Error Tradeoff
For witness rate $r$, type-II exponent $E$: $E + r \geq K(P)$
Achievable: $E^*(r) = K(P) - r$ for $r \in [0, K(P)]$

**Corollary 7.4.1** *(Linear Tradeoff Curve)*
Optimal tradeoff: $E^*(r) = K(P) - r$ for $r \in [0, K(P)]$; $E^*(r) = 0$ for $r \geq K(P)$

### Theorem 7.5: Universal Adversarial Structure
Optimal $\lambda^\star$ simultaneously optimal for testing, simulation, and coding

### Proposition 7.6: Hellinger Sphere Structure
$\alpha^\star(P) = 1 - D_H^2(P, \mathrm{FI})$ where $D_H^2(P, \mathrm{FI}) = \min_Q \max_c H^2(p_c, q_c)$

**Corollary 7.6.1** *(Level Set Geometry)*
Level sets $\{P: K(P) = \kappa\} = $ outer Hellinger Chebyshev spheres of radius $\sqrt{1 - 2^{-\kappa}}$ around FI

### Corollary 7.6.2: TV Gap
$d_{\mathrm{TV}}(P, \mathrm{FI}) \geq 1 - 2^{-K(P)}$

### Proposition 7.7: Smoothing Bound
$K((1-t)P + tR) \leq -\log_2((1-t)2^{-K(P)} + t) \leq (1-t)K(P)$
Tight when $R = Q^\star$

**Corollary 7.7.1** *(Minimal Smoothing)*
To ensure $K((1-t)P + tR) \leq \kappa$: $t \geq \frac{1 - 2^{-\kappa}}{1 - 2^{-K(P)}}$

### Proposition 7.8: Convex Program for K
$D_H^2(P, \mathrm{FI}) = \min_\mu \max_c H^2(p_c, q_c(\mu))$
$K(P) = -\log_2(1 - D_H^2(P, \mathrm{FI}))$

### Theorem 7.9: Equalizer Principle + Sparse Optimizers
Active contexts: $\mathrm{BC}(p_c, q_c^\star) = \alpha^\star(P)$
Optimal $Q^\star$ supported on at most $1 + \sum_c (|\mathcal{O}_c| - 1)$ deterministic assignments

---

## 7. Technical Lemmas

### Lemma A.2.2: Bhattacharyya Properties
Range $[0,1]$; $\mathrm{BC}(p,q)=1 \iff p=q$; jointly concave; multiplicative

### Lemma A.4.1: Uniform Law Lower Bound
$\alpha^\star(P) \geq \min_c \frac{1}{\sqrt{|\mathcal{O}_c|}}$

### Theorem A.5.1: Minimax Duality
$\lambda^\star$ locates tension; $Q^\star$ maximizes agreement; active contexts saturate bound

### Lemma B.2.2: Upper Bound for Odd-Cycle
$\alpha^\star(P) \leq \sqrt{2/3}$

### Proposition B.2.3: Achievability for Odd-Cycle
$\mu^\star = \text{Unif}(\{100,010,001,011,101,110\})$ achieves $\alpha^\star(P) = \sqrt{2/3}$

### Corollary B.2.4: Optimal Witness for Odd-Cycle
Uniform $\lambda^\star = (1/3,1/3,1/3)$; $K(P) = \frac{1}{2} \log_2(3/2) \approx 0.2925$ bits

### Lemma B.4.1: Odd Cycles Create Contradiction
Pairwise anti-correlations imply $K(P) > 0$

### Theorem B.4.2: The K(P) Tax for Consensus
Consensus requires rate $\geq H(X|C) + K(P)$

**Corollary B.4.3** *(Channel Capacity)*
Under source-channel separation: $C_{\text{common}} = C_{\text{Shannon}} - K(P)$

**Corollary B.4.4** *(Witness-Error Tradeoff)*
For witness rate $r$ and type-II exponent $E$: $E + r \geq K(P)$

---

## 8. Worked Examples

### Lenticular Coin (Odd-Cycle)
Three contexts with perfect disagreement:
- $p_c(1,0) = p_c(0,1) = 1/2$ for each pair context
- $\alpha^\star = \sqrt{2/3}$, $K = \frac{1}{2} \log_2(3/2) \approx 0.29$ bits per observation

### Gaussian Contradiction (Continuous Extension)
For Gaussian measurement precision conflicts N(0,σ₁²) vs N(0,σ₂²):
- Optimal FI approximation: $Q^\star = \mathcal{N}(0, \sqrt{\sigma_1 \sigma_2}^2)$
- $\alpha^\star(P) = \sqrt{\frac{2 \sqrt{\sigma_1 \sigma_2}}{\sigma_1 + \sigma_2}}$
- $K(P) = -\log_2 \sqrt{\frac{2 \sqrt{\sigma_1 \sigma_2}}{\sigma_1 + \sigma_2}}$
- Example: σ₁=1, σ₂=4 → α* ≈ 0.894, K ≈ 0.161 bits
- Extends theory to continuous observables with verified computational results

### Frame-Carrying Information Loss
Three-view setup with ambiguous middle position:
- $H(P \mid O) = 2/3$ bits missing when context dropped
- Structural cost of flattening perspectives

### Consensus Protocol Tax
Distributed systems with incompatible validity predicates pay $K(P)$ bits per decision beyond Shannon baseline

---

## 9. Key Inequalities & Identities

### Product Law
$\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R)$
$K(P \otimes R) = K(P) + K(R)$

### Hellinger Distance
$H(p,q) = \sqrt{1 - \mathrm{BC}(p,q)}$
$D_H^2(P, \mathrm{FI}) = \min_Q \max_c H^2(p_c, q_c)$
$\alpha^\star(P) = 1 - D_H^2(P, \mathrm{FI})$

### TV Bound
$d_{\mathrm{TV}}(P, \mathrm{FI}) \geq 1 - 2^{-K(P)}$

### Smoothing
$K((1-t)P + tR) \leq -\log_2((1-t)2^{-K(P)} + t)$
Minimal mixing: $t \geq \frac{1 - 2^{-\kappa}}{1 - 2^{-K(P)}}$ for $K \leq \kappa$

### Stability Property
Contradiction measures are highly stable: reducing $K(P) > 0$ to near-zero requires $t \approx 100\%$ FI mixture (e.g., lenticular coin needs $t \geq 99.6\%$ to reduce from 0.29 to 0.001 bits)

### Witness Tradeoff
$E + r \geq K(P)$ with equality achievable

---

## 10. Computational Tools

### Convex Program
$\min_\mu \max_c H^2(p_c, q_c(\mu))$ where $q_c(\mu) = \sum_s \mu(s) \cdot \delta_{s|_c}$

### Bootstrap Estimation
Plug-in estimator $\hat{K}$ from empirical frequencies with confidence intervals

### Regularized Estimation
Add pseudocounts $\epsilon > 0$ to all outcome counts for statistical consistency when true distributions contain zeros: $\tilde{p}_o = (n_o + \epsilon)/(n + k\epsilon)$ where $k$ is number of outcomes

### Minimax Program
$\max_Q \min_c \mathrm{BC}(p_c, q_c)$ solved via Sion's theorem duality