**Theorem 7.5** *(Universal Adversarial Prior)* 
(App. A.5.1)

Any optimal context weights $\lambda^\star$ in the minimax representation. We show this universal optimality:

$$
\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \mathrm{FI}} \sum_c \lambda_c \mathrm{BC}(p_c, q_c)
$$

are **simultaneously optimal adversaries** for:

1. Hypothesis testing lower bounds
2. Witness design (soft-covering)
3. Multi-decoder coding surcharge
4. Rate-distortion common-reconstruction surcharge

Put differently: one adversarial structure governs all operational limits.

**Proof Strategy:**

All four operational problems reduce to the same minimax in Theorem 2 (App. A.3.2), then inherit the same $\lambda^*$. And this establishes the universal adversarial structure.