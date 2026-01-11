# Proposition 7.8: Convex Program for K

**Proposition 7.8** *(Convex Program for K)*

Let $q_c(\mu)$ be the context marginal induced by a global law $\mu\in\Delta(\mathcal O_{\mathcal X})$. Then

$$
D_H^2(P,\mathrm{FI})\;=\;\min_{\mu\in\Delta(\mathcal O_{\mathcal X})}\ \max_{c\in\mathcal C}\ H^2\big(p_c,\ q_c(\mu)\big),
$$

and $K(P)=-\log_2\big(1-D_H^2(P,\mathrm{FI})\big)$.

**Proof Strategy:**

$H^2(p,\cdot)=1-\sum_o \sqrt{p(o)\,\cdot}$ is convex; $\mu\mapsto q_c(\mu)$ is affine; $\max_c$ preserves convexity. Combine with Proposition 6.A.