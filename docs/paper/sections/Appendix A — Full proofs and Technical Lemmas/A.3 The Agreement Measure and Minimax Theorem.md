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
- Compactness of FI: Proposition [A.1.6](#proposition-a16-topological-properties).
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