# Agreement Measure and Minimax Theorem

The agreement measure quantifies how much a behavior can be reconciled with frame-independent explanations. It measures the maximum consistency achievable across all measurement contexts, representing the "best possible agreement" under optimal circumstances.

In short—this captures the reconciliation potential.

## Agreement and Contradiction Measures

For a behavior $P$, the optimal agreement coefficient is:

$$
\alpha^\star(P) := \max_{Q \in \text{FI}} \min_{c \in \mathcal{C}} \text{BC}(p_c, q_c)
$$

This finds the frame-independent behavior $Q$ that maximizes the minimum agreement across all contexts.

And we show this to be the key optimization.

```python
# Create a behavior with contradictions using Observatory API
from contrakit import Observatory

observatory = Observatory.create(symbols=["Good", "Bad"])

# Define perspectives that will create tension
satisfaction = observatory.concept("Satisfaction")
recommendation = observatory.concept("Recommendation")

# Set individual perspectives
observatory.perspectives[satisfaction] = {"Good": 0.8, "Bad": 0.2}
observatory.perspectives[recommendation] = {"Good": 0.7, "Bad": 0.3}

# Create joint perspective that shows low correlation (creates contradiction)
observatory.perspectives[satisfaction, recommendation] = {
    ("Good", "Good"): 0.2,      # Low positive correlation
    ("Good", "Bad"): 0.6,       # High negative correlation
    ("Bad", "Good"): 0.2,       # Low negative correlation
    ("Bad", "Bad"): 0.0         # No bad-bad correlation
}

behavior = observatory.perspectives.to_behavior()

# Compute optimal agreement
alpha_star = behavior.alpha_star
print(f"Optimal agreement coefficient: {alpha_star:.4f}")
# Output:
# Optimal agreement coefficient: 0.9883
```

The contradiction measure quantifies disagreement in bits:

$$
K(P) := -\log_2 \alpha^\star(P)
$$

Put differently—this transforms agreement into contradiction.

```python
# Compute contradiction in bits
contradiction_bits = behavior.K
print(f"Contradiction measure: {contradiction_bits:.4f} bits")
# Output:
# Contradiction measure: 0.0170 bits
```

## Minimax Theorem

The agreement measure has a dual characterization using minimax theory. Define the payoff function:

$$
f(\lambda, Q) := \sum_{c \in \mathcal{C}} \lambda_c \text{BC}(p_c, q_c)
$$

for context weights $\lambda \in \Delta(\mathcal{C})$ and frame-independent behaviors $Q \in \text{FI}$.

**Theorem**: The optimal agreement equals:

$$
\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \text{FI}} f(\lambda, Q)
$$

```python
# Demonstrate the minimax relationship
# The worst-case context weights (that maximize contradiction)
worst_weights = behavior.worst_case_weights
print("Context weights that maximize contradiction:")
for context, weight in worst_weights.items():
    print(f"  {context}: {weight:.4f}")

# Compute agreement under these adversarial weights
adversarial_agreement = behavior.agreement.for_weights(worst_weights).result
print(f"Agreement under worst-case weights: {adversarial_agreement:.4f}")
# Output:
# Context weights that maximize contradiction:
#   ('Recommendation',): 0.5000
#   ('Satisfaction', 'Recommendation'): 0.5000
```

## Proof

The proof applies **Sion's minimax theorem** to establish the equality between the primal and dual formulations.

**Requirements for Sion's theorem:**
1. $\Delta(\mathcal{C})$ and FI are nonempty, convex, and compact ✓
2. $f(\lambda, \cdot)$ is concave on FI for each $\lambda$ ✓  
3. $f(\cdot, Q)$ is convex (actually linear) on $\Delta(\mathcal{C})$ for each $Q$ ✓

**Key details:**
- **Compactness**: $\Delta(\mathcal{C})$ is the standard simplex; FI is compact by construction
- **Concavity in Q**: The Bhattacharyya coefficient is jointly concave, and linear combinations preserve concavity
- **Linearity in λ**: The payoff function is linear in the context weights

By Sion's minimax theorem: $\min_\lambda \max_Q f(\lambda, Q) = \max_Q \min_\lambda f(\lambda, Q)$

The right-hand side equals $\alpha^\star(P)$, and the minimax equality follows.

**Interpretation**: The agreement measure represents the best guarantee of consistency under the most adversarial choice of how to weight different measurement contexts.
