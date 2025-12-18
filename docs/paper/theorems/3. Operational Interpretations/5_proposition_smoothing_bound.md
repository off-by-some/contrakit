### Smoothing and Interpolation

**Proposition 7.7** *(Smoothing Bound)*

For any behavior $P$, any $R \in \mathrm{FI}$, and $t \in [0,1]$. We show this smoothing property:

$$
K((1-t)P + tR) \leq -\log_2((1-t)2^{-K(P)} + t) \leq (1-t)K(P)
$$

This bound is tight when $R = Q^*$ is an optimal FI simulator for $P$.

**Proof Strategy:**

Using the dual form $\alpha^\star(P) = \min_\lambda \max_Q \sum_c \lambda_c \mathrm{BC}(p_c, q_c)$ and concavity of $\mathrm{BC}$ in its first argument:
$\alpha^\star((1-t)P + tR) \,\ge\, (1-t)\alpha^\star(P) + t$
Applying $K = -\log_2 \alpha^\star$ gives the bound; tightness holds when $R = Q^*$ (concavity met with equality). *(See App. A.12 for details.)* And this establishes the interpolation property.

**Corollary 7.7.1** *(Minimal Smoothing)*

To ensure $K((1-t)P + tR) \leq \kappa$, it suffices that:

$$
t \,\ge\, \frac{1 - 2^{-\kappa}}{1 - 2^{-K(P)}}
$$

**Proof Strategy:** Rearrangement of the bound with $\alpha^\star = 2^{-K}$ *(App. A.12)*.

**Interpretation:**

Mixing any amount $t$ of frame-independent "noise" with $P$ reduces contradiction at least as fast as the bound predicts. This gives a constructive way to reduce contradiction costs through deliberate randomization. What becomes unavoidable: some randomization is always required.

---

## 