# **Theorem 15** *(Contradiction Geometry)* 
(see App. A.2.2, A.10; FI product closure A.1.8)

**(a) Pairwise Hellinger Metric:**

For $J(A,B) := \max_c \arccos(\mathrm{BC}(p^A_c, p^B_c))$—

$$
J(A,C) \le J(A,B) + J(B,C)
$$

Put differently—this establishes the geometric structure.

**(b) Subadditivity under products:**

For $J(P):=\arccos\alpha^\star(P)$, 
angles are subadditive: $J(P\otimes R)=\arccos(\alpha^\star(P)\alpha^\star(R))\le J(P)+J(R)$.

**(c) Log-additivity:**

In bits $K(P)=-\log_2\alpha^\star(P)$, for independent systems on disjoint observables (with FI product-closure) one has $K(P\otimes R)=K(P)+K(R)$.
Moreover, for pairwise models:
$$
K_{\text{pair}}(A,C) \le -\log_2 \cos(J(A,B) + J(B,C))
$$

**Interpretation:** 

On each simplex, the Hellinger angle $\arccos \mathrm{BC}$ is a metric; taking $\max_c$ preserves the triangle inequality. Product multiplicativity means bits add; angles are subadditive via $\arccos(xy) \le \arccos x + \arccos y$; additivity is exact in the log domain.