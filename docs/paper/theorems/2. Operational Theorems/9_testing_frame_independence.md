## 6.3 Hypothesis Testing Against Frame-Independence

**Theorem 9** *(Testing Frame-Independence)*

For testing $\mathcal{H}_0: Q \in \mathrm{FI}$ vs $\mathcal{H}_1: P$, the optimal level-$\eta$ type-II error exponent (bits/sample) satisfies—

$$
-\frac{1}{n} \log_2 \inf_{\text{level-}\eta} \sup_{Q \in \mathrm{FI}} \Pr_Q[\text{accept } \mathcal{H}_1] \ge K(P)
$$

Put differently—this gives the fundamental testing bound. And we establish this through exponent analysis.

**Proof Strategy:**

The Chernoff bound at $s = 1/2$ gives exactly the Bhattacharyya coefficient $\alpha^\star(P)$, yielding exponent $K(P) = -\log_2 \alpha^\star(P)$.

Equality holds when the Chernoff optimizer is at $s=\tfrac12$.

