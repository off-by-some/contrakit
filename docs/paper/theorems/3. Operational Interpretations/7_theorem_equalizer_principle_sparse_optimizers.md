# Theorem 7.9: Equalizer Principle + Sparse Optimizers

**Theorem 7.9** *(Equalizer Principle + Sparse Optimizers)*

There exist optimal strategies $(\lambda^\star,Q^\star)$ with:

1. **Equalizer:** For all *active* contexts $c$ (those with $\lambda_c^\star>0$),$\mathrm{BC}(p_c,q_c^\star)\ =\ \alpha^\star(P).$
2. **Sparse global law (Carathéodory bound):** $Q^\star$ can be chosen to arise from a global law $\mu^\star$ supported on at most $1+\sum_{c\in\mathcal C}\big(|\mathcal O_c|-1\big)$ deterministic assignments.

**Proof Strategy:**
(1) Active-set equalization is the KKT/duality condition from the min–max in Theorem 2. (2) FI lives in an affine space of dimension $\sum_c(|\mathcal O_c|-1)$; by Carathéodory, any $Q^\star\in\mathrm{FI}$ is a convex combination of at most $d+1$ extreme points.

**Interpretation:**
At the optimum, all binding contexts tie exactly at $\alpha^\star$. There's always an optimal global explanation using only polynomially many deterministic worlds (in the ambient dimension)