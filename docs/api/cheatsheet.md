
# Mathematical Theory of Contradiction - Cheatsheet

## 1. Core Structures

### Observable System
- **Observables**: Finite set $\mathcal{X} = \{X_1, \ldots, X_n\}$
- **Outcomes**: Each observable $x$ has finite outcome set $\mathcal{O}_x$ (nonempty)
- **Context**: Subset $c \subseteq \mathcal{X}$ of observables measured together
- **Context outcomes**: $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$

```python
from contrakit import Space

space = Space.create(
    Satisfaction=["Happy", "Unhappy"],
    Recommend=["Yes", "No"]
)
print(f"Observables: {space.names}")
print(f"Satisfaction alphabet: {space.alphabets['Satisfaction']}")
```

### Behavior
A behavior $P$ assigns probability distributions to contexts:
$P = \{p_c \in \Delta(\mathcal{O}_c) : c \in \mathcal{C}\}$

**Key property**: No consistency required across overlapping contexts.

```python
from contrakit import Behavior
behavior = Behavior.from_contexts(space, {
    ("Satisfaction",): {("Happy",): 0.7, ("Unhappy",): 0.3},
    ("Recommend",): {("Yes",): 0.6, ("No",): 0.4},
    ("Satisfaction", "Recommend"): {
        ("Happy", "Yes"): 0.3, ("Happy", "No"): 0.4,
        ("Unhappy", "Yes"): 0.3, ("Unhappy", "No"): 0.0
    }
})
```

### Frame-Independent Set (FI)
All behaviors explainable by hidden variables (deterministic global assignments):
$\text{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\}$

**Alternative characterization**: $Q \in \text{FI}$ iff exists global law $\mu \in \Delta(\mathcal{O}_{\mathcal{X}})$ where:
$q_c(o) = \sum_{s : s|_c = o} \mu(s)$

**Properties**: Nonempty, convex, compact

```python
# Check if behavior is frame-independent
is_fi = behavior.is_frame_independent()
print(f"Frame-independent: {is_fi}")  # False (contradictory)

# Create FI behavior from global assignment probabilities
import numpy as np
contexts = [["Satisfaction"], ["Recommend"], ["Satisfaction", "Recommend"]]
mu = np.array([0.5, 0.0, 0.5, 0.0])  # Probability over 4 global states
fi_behavior = Behavior.from_mu(space, contexts, mu)
print(f"FI: {fi_behavior.is_frame_independent()}")  # True
```

### Context Simplex
Probability distributions over contexts:
$\Delta(\mathcal{C}) := \{\lambda \in \mathbb{R}^{\mathcal{C}} : \lambda_c \geq 0, \sum_c \lambda_c = 1\}$

```python
# Custom context weights for agreement calculation
context_weights = {
    ("Satisfaction",): 0.4,
    ("Recommend",): 0.3,
    ("Satisfaction", "Recommend"): 0.3
}
weighted_agreement = behavior.agreement.for_weights(context_weights).result
print(f"Weighted agreement: {weighted_agreement:.4f}")
```

### Observatory API (High-Level Interface)
For easier behavior construction:

```python
from contrakit import Observatory

obs = Observatory.create(symbols=["Happy", "Unhappy", "Yes", "No"])
satisfaction = obs.concept("Satisfaction", symbols=["Happy", "Unhappy"])
recommend = obs.concept("Recommend", symbols=["Yes", "No"])

# Set marginals
obs.perspectives[satisfaction] = {"Happy": 0.7, "Unhappy": 0.3}
obs.perspectives[recommend] = {"Yes": 0.6, "No": 0.4}

# Set joint using & syntax
happy, unhappy = satisfaction.alphabet
yes, no = recommend.alphabet
obs.perspectives[satisfaction, recommend] = {
    happy & yes: 0.3, happy & no: 0.4,
    unhappy & yes: 0.3, unhappy & no: 0.0
}

# Convert to behavior
behavior = obs.perspectives.to_behavior()
```

---

## 2. Agreement Kernel

### Bhattacharyya Coefficient
For distributions $p, q \in \Delta(\mathcal{O})$:
$\text{BC}(p, q) := \sum_{o \in \mathcal{O}} \sqrt{p(o) q(o)}$

**Properties**:
- **Range**: $0 \leq \text{BC}(p, q) \leq 1$
- **Perfect agreement**: $\text{BC}(p, q) = 1 \Leftrightarrow p = q$
- **Joint concavity**: Concave in $(p, q)$
- **Product structure**: $\text{BC}(p \otimes r, q \otimes s) = \text{BC}(p, q) \cdot \text{BC}(r, s)$

```python
from contrakit.agreement import BhattacharyyaCoefficient
import numpy as np

bc = BhattacharyyaCoefficient()
p = np.array([0.7, 0.3])
q = np.array([0.6, 0.4])
agreement = bc(p, q)
print(f"BC = {agreement:.4f}")  # 0.9949
```

---

## 3. Contradiction Measures

### Optimal Agreement Coefficient
$\alpha^\star(P) := \max_{Q \in \text{FI}} \min_{c \in \mathcal{C}} \text{BC}(p_c, q_c)$

**Interpretation**: Best frame-independent explanation's worst-case agreement.

```python
alpha_star = behavior.alpha_star
print(f"α* = {alpha_star:.4f}")  # e.g., 0.9883
```

### Contradiction Measure (in bits)
$K(P) := -\log_2 \alpha^\star(P)$

**Interpretation**: Information required to reconcile behavior with FI models.

```python
K = behavior.K
print(f"K = {K:.4f} bits")  # e.g., 0.0170 bits
```

---

## 4. Minimax Duality

### Payoff Function
$$f(\lambda, Q) := \sum_{c \in \mathcal{C}} \lambda_c \text{BC}(p_c, q_c)$$

### Minimax Equality (Sion's Theorem)
$$\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \text{FI}} f(\lambda, Q) = \max_{Q \in \text{FI}} \min_{\lambda \in \Delta(\mathcal{C})} f(\lambda, Q)$$

**Interpretations**:
- **Primal** (left): Best FI model's worst-case agreement
- **Dual** (right): Worst-case context weighting's best agreement

### Optimal Strategies
For optimal $(\lambda^\star, Q^\star)$:
1. **Value equality**: $f(\lambda^\star, Q^\star) = \alpha^\star(P)$
2. **Support property**: $\text{supp}(\lambda^\star) \subseteq \{c : \text{BC}(p_c, q_c^\star) = \alpha^\star(P)\}$
3. **Equality condition**: $\lambda^\star_c > 0 \Rightarrow \text{BC}(p_c, q_c^\star) = \alpha^\star(P)$

**Diagnostic**: Contexts with positive $\lambda^\star$ weight are the active constraints (bottlenecks).

```python
# Find worst-case context weights (adversarial weights)
worst_weights = behavior.worst_case_weights
for context, weight in worst_weights.items():
    print(f"{context}: {weight:.4f}")
# Example output:
# ('Recommend',): 0.5000
# ('Satisfaction', 'Recommend'): 0.5000

# Verify: agreement under adversarial weights equals α*
adversarial_agreement = behavior.agreement.for_weights(worst_weights).result
print(f"Adversarial α = {adversarial_agreement:.4f}")
print(f"α* = {behavior.alpha_star:.4f}")
print(f"Match: {np.isclose(adversarial_agreement, behavior.alpha_star)}")
```

---

## 5. Bounds

### Uniform Law Lower Bound
$$\alpha^\star(P) \geq \min_{c \in \mathcal{C}} \frac{1}{\sqrt{|\mathcal{O}_c|}}$$

### Contradiction Bounds
$$0 \leq K(P) \leq \frac{1}{2} \log_2 \left(\max_{c \in \mathcal{C}} |\mathcal{O}_c|\right)$$

### Frame-Independence Characterization
$\alpha^\star(P) = 1 \Leftrightarrow P \in \text{FI} \Leftrightarrow K(P) = 0$

```python
# All three conditions are equivalent
is_fi = behavior.is_frame_independent()
alpha_is_one = abs(behavior.alpha_star - 1.0) < 1e-6
k_is_zero = abs(behavior.K) < 1e-6
assert is_fi == alpha_is_one == k_is_zero
```

---

## 6. Product Structure

### Tensor Product of Behaviors
For $P$ on $(\mathcal{X}, \mathcal{C})$ and $R$ on $(\mathcal{Y}, \mathcal{D})$ with $\mathcal{X} \cap \mathcal{Y} = \emptyset$:

**Product distributions**:
$$(p \otimes r)(o_c, o_d) = p(o_c) \cdot r(o_d)$$

**Product behavior**:
$$(P \otimes R)(o_c, o_d \mid c \cup d) = p_c(o_c) \cdot r_d(o_d)$$

**Properties**:
- **FI preservation**: $Q \in \text{FI}_\mathcal{X}, S \in \text{FI}_\mathcal{Y} \Rightarrow Q \otimes S \in \text{FI}_{\mathcal{X} \sqcup \mathcal{Y}}$
- **Deterministic combination**: $q_s \otimes q_t = q_{s \sqcup t}$

### Additivity Laws
$\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R)$
$K(P \otimes R) = K(P) + K(R)$

**Interpretation**: Independent contradictions compose additively in bits.

```python
# Create two independent behaviors
space1 = Space.create(Q1=["Yes", "No"])
space2 = Space.create(Q2=["Agree", "Disagree"])
behavior1 = Behavior.from_contexts(space1, {("Q1",): {("Yes",): 0.8, ("No",): 0.2}})
behavior2 = Behavior.from_contexts(space2, {("Q2",): {("Agree",): 0.7, ("Disagree",): 0.3}})

# Tensor product (@ operator)
combined = behavior1 @ behavior2

# Verify additivity
print(f"K(P) = {behavior1.K:.4f}")
print(f"K(R) = {behavior2.K:.4f}")
print(f"K(P⊗R) = {combined.K:.4f}")
# K(P⊗R) = K(P) + K(R)
```

---

## 7. Additional Bounds

### Total Variation Gap
$$d_{\text{TV}}(P, \text{FI}) := \inf_{Q \in \text{FI}} \max_{c \in \mathcal{C}} \text{TV}(p_c, q_c) \geq 1 - \alpha^\star(P) = 1 - 2^{-K(P)}$$

**Interpretation**: Contradiction lower-bounds observable statistical discrepancy. Any FI simulator must differ from $P$ by at least $1 - 2^{-K(P)}$ in total variation on some context.

### Smoothing BoundP
For any $R \in \text{FI}$ and $t \in [0,1]$:
$K((1-t)P + tR) \leq -\log_2((1-t) \cdot 2^{-K(P)} + t) \leq (1-t) K(P)$

**Tight when**: $R = Q^\star$ (optimal FI simulator for $P$)

**To achieve target** $K \leq \kappa$:
$t \geq \frac{1 - 2^{-\kappa}}{1 - 2^{-K(P)}}$

**Interpretation**: Quantifies how much FI noise must be mixed in to reduce contradiction below a threshold.

```python
# Calculate mixing fraction needed to achieve target contradiction
K_current = behavior.K
kappa_target = 0.01  # Target: 0.01 bits

if K_current < 1e-10:
    print(f"Behavior is already frame-independent, no mixing needed")
else:
    t_needed = (1 - 2**(-kappa_target)) / (1 - 2**(-K_current))
    print(f"Need to mix {t_needed:.2%} FI behavior to reach {kappa_target} bits")
```

---

## 8. Operational Theorems: The $K(P)$ Tax

**Core principle**: Contradiction adds exactly $K(P)$ bits per symbol to all information-theoretic tasks.

### 8.1 Asymptotic Equipartition & Compression

**Theorem 6** *(AEP with Contradiction Tax)*
- Typical sets for $(X^n, W_n)$ have size $2^{n(H(X|C) + K(P))}$ with witnesses $W_n$ of rate $K(P)$
- Without adequate witnesses: $\liminf_{n \to \infty} \frac{1}{n} \log_2 |\mathcal{S}_n| \ge H(X|C) + K(P)$

**Theorem 7** *(Compression, Known Contexts)*
- With $C^n$ at decoder: $\lim_{n \to \infty} \frac{1}{n} \mathbb{E}[\ell_n^*] = H(X|C) + K(P)$

**Theorem 8** *(Compression, Latent Contexts)*
- Without $C^n$: $\lim_{n \to \infty} \frac{1}{n} \mathbb{E}[\ell_n^*] = H(X) + K(P)$

### 8.2 Hypothesis Testing & Simulation

**Theorem 9** *(Testing Frame-Independence)*
- Testing $\mathcal{H}_0: Q \in \text{FI}$ vs $\mathcal{H}_1: P$: optimal type-II error exponent $\ge K(P)$
- Achieved via Chernoff bound at $s = 1/2$ (Bhattacharyya)

**Theorem 10** *(Witnessing for TV-Approximation)*
- Witnesses $W_n$ with rate $K(P) + o(1)$ achieve $\text{TV}((X^n, W_n), \tilde{Q}_n) \to 0$ for FI laws $\tilde{Q}_n$
- No rate $< K(P)$ achieves vanishing TV

**Proposition 7.1** *(Testing Real vs Frame-Independent)*
- For contexts drawn from $\lambda \in \Delta(\mathcal{C})$: $E_{\text{opt}}(\lambda) \ge E_{\text{BH}}(\lambda) = -\log_2 \alpha_\lambda(P)$
- Least-favorable: $\inf_\lambda E_{\text{opt}}(\lambda) = K(P)$

### 8.3 Multi-Decoder Communication

**Theorem 11** *(Common Message Problem)*
- Single message decodable by all contexts: rate $= H(X|C) + K(P)$

**Theorem 12** *(Common Representation Cost)*
- Representation $Z = Z(X^n)$ enabling all contexts to decode:
  - Known contexts: $\frac{1}{n} I(X^n; Z) \ge H(X|C) + K(P) - o(1)$
  - Latent contexts: $\frac{1}{n} I(X^n; Z) \ge H(X) + K(P) - o(1)$

### 8.4 Noisy Channels & Rate-Distortion

**Theorem 13** *(Channel Capacity with Common Decoding)*
- Over DMC with Shannon capacity $C_{\text{Shannon}}$: payload rate $R_{\text{payload}} = C_{\text{Shannon}} - K(P)$

**Theorem 14** *(Rate-Distortion with Common Reconstruction)*
- Under common-reconstruction requirement: $R(D) = R_{\text{Shannon}}(D) + K(P)$

### 8.5 Prediction & Simulation Costs

**Proposition 7.2** *(Importance Sampling Penalty)*
- Simulating $P$ using $Q \in \text{FI}$: $\inf_{Q \in \text{FI}} \max_c \text{Var}_{Q_c}[w_c] \ge 2^{2K(P)} - 1$

**Proposition 7.3** *(Single-Predictor Penalty)*
- Using one predictor $Q \in \text{FI}$ across all contexts (log-loss): $\inf_{Q \in \text{FI}} \max_c \mathbb{E}_{p_c}[\log_2 \frac{p_c(X)}{q_c(X)}] \ge 2K(P)$ bits/round

### 8.6 Tradeoffs & Geometry

**Theorem 7.4** *(Witness-Error Tradeoff)*
- Witness rate $r$ and type-II exponent $E$: $E + r \ge K(P)$
- Optimal tradeoff: $E^*(r) = K(P) - r$ for $r \in [0, K(P)]$ (linear)

**Theorem 7.5** *(Universal Adversarial Prior)*
- Optimal context weights $\lambda^\star$ simultaneously optimize:
  1. Hypothesis testing lower bounds
  2. Witness design (soft-covering)
  3. Multi-decoder coding surcharge
  4. Rate-distortion common-reconstruction surcharge

**Theorem 15** *(Contradiction Geometry)*
- **Hellinger metric**: $J(A,C) \le J(A,B) + J(B,C)$ where $J(A,B) = \max_c \arccos(\text{BC}(p^A_c, p^B_c))$
- **Subadditivity**: $J(P \otimes R) \le J(P) + J(R)$ (angles subadditive, bits additive)
- **Pairwise bound**: $K_{\text{pair}}(A,C) \le -\log_2 \cos(J(A,B) + J(B,C))$

**Proposition 7.6** *(Chebyshev Radius Identity)*
- With Hellinger distance $H(p,q) = \sqrt{1 - \text{BC}(p,q)}$: $\alpha^\star(P) = 1 - D_H^2(P, \text{FI})$
- Level sets: $\{P: K(P) = \kappa\}$ are Hellinger spheres of radius $\sqrt{1 - 2^{-\kappa}}$ around FI

### 8.7 Computational Structure

**Proposition 7.8** *(Convex Program for K)*
- $D_H^2(P,\text{FI}) = \min_{\mu \in \Delta(\mathcal{O}_{\mathcal{X}})} \max_{c \in \mathcal{C}} H^2(p_c, q_c(\mu))$
- $K(P) = -\log_2(1 - D_H^2(P,\text{FI}))$

**Theorem 7.9** *(Equalizer Principle + Sparse Optimizers)*
- **Equalizer**: Active contexts $c$ (with $\lambda_c^\star > 0$) satisfy $\text{BC}(p_c, q_c^\star) = \alpha^\star(P)$
- **Sparsity**: $Q^\star$ can arise from global law $\mu^\star$ supported on $\le 1 + \sum_{c \in \mathcal{C}}(|\mathcal{O}_c| - 1)$ deterministic assignments

---

## 9. Axiomatic Foundation

### Core Axioms
- **A0 (Label invariance)**: Permutation-invariant across outcome relabelings
- **A1 (Calibration)**: $K(P) = 0 \Leftrightarrow P \in \text{FI}$
- **A2 (Continuity)**: Continuous in behavior parameters
- **A3 (Free-ops monotonicity)**: Non-increasing under data processing (DPI)
- **A4 (Grouping)**: Invariant under context duplication/splitting
- **A5 (Independent composition)**: $K(P \otimes R) = K(P) + K(R)$

### Weakest Link Principle
Any aggregator satisfying:
1. Monotonicity: $x \leq y \Rightarrow A(x) \leq A(y)$
2. Unanimity: $A(t, \ldots, t) = t$
3. Local upper bound: $A(x) \leq x_i$ for all $i$

Must equal the minimum: $A(x) = \min_i x_i$

**Consequence**: Agreement is determined by worst-case context (bottleneck principle).

---

## 10. Implementation Guide

### Computing $\alpha^\star$ (Primal)
1. Enumerate or sample global laws $\mu \in \Delta(\mathcal{O}_{\mathcal{X}})$
2. For each $\mu$, compute induced $Q$ and evaluate $\min_c \text{BC}(p_c, q_c)$
3. Maximize over all $\mu$ (or $Q \in \text{FI}$)

```python
# Direct computation (library handles optimization)
alpha_star = behavior.alpha_star
K = behavior.K
```

### Computing $\alpha^\star$ (Dual)
1. Optimize over context weights $\lambda \in \Delta(\mathcal{C})$
2. For each $\lambda$, maximize $\sum_c \lambda_c \text{BC}(p_c, q_c)$ over $Q \in \text{FI}$
3. Minimize over $\lambda$

```python
# Get optimal adversarial weights (dual solution)
worst_weights = behavior.worst_case_weights
```

### Identifying Bottleneck Contexts
Contexts with $\lambda^\star_c > 0$ are the active constraints where:
$\text{BC}(p_c, q_c^\star) = \alpha^\star(P)$

These are the contexts preventing reconciliation with FI.

```python
# Find which contexts are constraining
bottlenecks = {ctx: w for ctx, w in worst_weights.items() if w > 1e-6}
print("Bottleneck contexts:", bottlenecks)

# Check individual context agreements
for ctx in behavior.context:
    ctx_key = tuple(ctx.observables)
    ctx_agreement = behavior.agreement.for_weights({ctx_key: 1.0}).result
    print(f"{ctx_key}: {ctx_agreement:.4f}")
```

---

## 11. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Bhattacharyya | $\text{BC}(p, q) = \sum_o \sqrt{p(o) q(o)}$ |
| Optimal Agreement | $\alpha^\star(P) = \max_{Q \in \text{FI}} \min_c \text{BC}(p_c, q_c)$ |
| Contradiction | $K(P) = -\log_2 \alpha^\star(P)$ |
| Minimax | $\alpha^\star = \min_\lambda \max_Q \sum_c \lambda_c \text{BC}(p_c, q_c)$ |
| Product | $\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R)$ |
| Additivity | $K(P \otimes R) = K(P) + K(R)$ |
| Lower Bound | $\alpha^\star(P) \geq \min_c |\mathcal{O}_c|^{-1/2}$ |
| Upper Bound | $K(P) \leq \frac{1}{2} \log_2(\max_c |\mathcal{O}_c|)$ |
| TV Gap | $d_{\text{TV}}(P, \text{FI}) \geq 1 - 2^{-K(P)}$ |

---

## 12. Conceptual Flowchart

```
Behavior P
    ↓
Compute α*(P) = max min BC(p_c, q_c)
              Q∈FI  c
    ↓
K(P) = -log₂ α*(P)
    ↓
Interpret:
- K = 0 ⟺ P ∈ FI (consistent)
- K > 0 ⟺ P ∉ FI (contradictory)
- Bottlenecks: contexts with λ*_c > 0
```

### Complete Example Workflow

```python
from contrakit import Space, Behavior
import numpy as np

# 1. Define observable system
space = Space.create(Q1=["A", "B"], Q2=["X", "Y"])

# 2. Create behavior with potential contradiction
behavior = Behavior.from_contexts(space, {
    ("Q1",): {("A",): 0.6, ("B",): 0.4},
    ("Q2",): {("X",): 0.5, ("Y",): 0.5},
    ("Q1", "Q2"): {
        ("A", "X"): 0.1, ("A", "Y"): 0.5,  # Low correlation
        ("B", "X"): 0.4, ("B", "Y"): 0.0   # Strong correlation
    }
})

# 3. Check contradiction
print(f"K = {behavior.K:.4f} bits")
print(f"α* = {behavior.alpha_star:.4f}")
print(f"Frame-independent: {behavior.is_frame_independent()}")

# 4. Find bottlenecks
worst = behavior.worst_case_weights
print(f"Bottleneck contexts:")
for ctx, weight in worst.items():
    if weight > 1e-6:
        print(f"  {ctx}: λ* = {weight:.4f}")

# 5. Compare with FI behavior
fi_behavior = Behavior.from_mu(
    space,
    [["Q1"], ["Q2"], ["Q1", "Q2"]],
    np.array([0.6, 0.2, 0.2, 0.0])
)
print(f"FI behavior K = {fi_behavior.K:.4f} bits")
```