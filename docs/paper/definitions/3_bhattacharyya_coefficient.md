# Bhattacharyya Coefficient

The Bhattacharyya coefficient measures agreement between probability distributions. It quantifies how similar two probability distributions are by taking the sum of square roots of their product probabilities.

This is like measuring the "overlap" between distributions. Put differently, it's a geometric mean. Consider this carefully.

## Definition

For probability distributions $p, q \in \Delta(\mathcal{O})$ on a finite alphabet $\mathcal{O}$—and for our purposes here:

$$
\text{BC}(p, q) := \sum_{o \in \mathcal{O}} \sqrt{p(o) q(o)}
$$

In short—this captures the geometric mean. And it's not a coincidence. Indeed, we see this connection clearly.

```python
from contrakit.agreement import BhattacharyyaCoefficient
import numpy as np

# Create the Bhattacharyya coefficient measure
bc = BhattacharyyaCoefficient()

# Example: Compare two survey response distributions
p = np.array([0.7, 0.3])  # 70% "Yes", 30% "No"
q = np.array([0.6, 0.4])  # 60% "Yes", 40% "No"

agreement_score = bc(p, q)
print(f"Bhattacharyya coefficient: {agreement_score:.4f}")
# Output: 0.9945 (high agreement between similar distributions)

# Perfect agreement case
r = np.array([0.7, 0.3])  # Same as p
perfect_score = bc(p, r)
print(f"Perfect agreement: {perfect_score:.4f}")
# Output: 1.0000 (identical distributions)
```

## Key Properties

1. **Range**: $0 \leq \text{BC}(p, q) \leq 1$

```python
# Demonstrate range with opposite distributions
opposite_p = np.array([1.0, 0.0])  # All "Yes"
opposite_q = np.array([0.0, 1.0])  # All "No"
min_score = bc(opposite_p, opposite_q)
print(f"Minimum agreement (opposite): {min_score:.4f}")
# Output: 0.0000 (no overlap between distributions)
```

2. **Perfect agreement**: $\text{BC}(p, q) = 1 \Leftrightarrow p = q$

```python
# Show that BC=1 only when distributions are identical
same_dist = bc(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
different_dist = bc(np.array([0.5, 0.5]), np.array([0.6, 0.4]))
print(f"Same distributions: {same_dist:.4f}")
print(f"Different distributions: {different_dist:.4f}")
# Output: 1.0000 vs 0.9949
```

3. **Joint concavity**: BC is jointly concave in its arguments

4. **Product structure**: $\text{BC}(p \otimes r, q \otimes s) = \text{BC}(p, q) \cdot \text{BC}(r, s)$

```python
# Demonstrate product structure with tensor products
from contrakit import Space, Behavior

# Create independent survey systems
space1 = Space.create(Q1=["Yes", "No"])
space2 = Space.create(Q2=["Agree", "Disagree"])

# Create behaviors with known agreements
behavior1 = Behavior.from_contexts(space1, {
    ("Q1",): {("Yes",): 0.8, ("No",): 0.2}
})
behavior2 = Behavior.from_contexts(space2, {
    ("Q2",): {("Agree",): 0.9, ("Disagree",): 0.1}
})

# Compare individual agreement vs combined
individual_bc = bc(np.array([0.8, 0.2]), np.array([0.9, 0.1]))
print(f"Individual agreement: {individual_bc:.4f}")

# Combine behaviors and compare overall agreement
combined = behavior1 @ behavior2
# The product structure ensures combined agreement = product of individual agreements
```

## Proof

1. **Range**: Nonnegativity is obvious. For the upper bound, by Cauchy-Schwarz:
   $$
   \text{BC}(p, q) = \sum_o \sqrt{p(o) q(o)} \leq \sqrt{\sum_o p(o)} \sqrt{\sum_o q(o)} = 1
   $$

2. **Perfect agreement**: The Cauchy-Schwarz equality condition gives $\text{BC}(p, q) = 1$ iff $\sqrt{p(o)}$ and $\sqrt{q(o)}$ are proportional, i.e., $\frac{\sqrt{p(o)}}{\sqrt{q(o)}}$ is constant over $\{o : p(o) q(o) > 0\}$. Since both are probability distributions, this constant must be 1, giving $p = q$.

3. **Joint concavity**: Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$; extend to $\mathbb{R}_{\geq 0}^2$ by continuity. Summing over coordinates preserves concavity.

4. **Product structure**: Expand the tensor product and factor the sum.
