## 7.3 Geometric and Analytic Tools

### Hellinger Sphere Structure

**Proposition 7.6** *(Chebyshev Radius Identity)*

Let $H(p,q) := \sqrt{1 - \mathrm{BC}(p,q)}$ be Hellinger distance. Consider this geometric reformulation:

$$
D_H^2(P, \mathrm{FI}) := \min_{Q \in \mathrm{FI}} \max_{c \in \mathcal{C}} H^2(p_c, q_c)
$$

Then:

$$
\alpha^\star(P) = 1 - D_H^2(P, \mathrm{FI}), \quad K(P) = -\log_2(1 - D_H^2(P, \mathrm{FI}))
$$

**Corollary 7.6.1** *(Level Set Geometry)*

The level sets ${P: K(P) = \kappa}$ are exactly the outer Hellinger Chebyshev spheres of radius $\sqrt{1 - 2^{-\kappa}}$ around $\mathrm{FI}$. And this reveals the geometric structure of contradiction.