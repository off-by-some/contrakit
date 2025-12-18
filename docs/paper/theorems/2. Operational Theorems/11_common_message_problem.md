# **Theorem 11** *(Common Message Problem)*

The natural baseline for a multi-decoder system is the common-message architecture—a single representation delivered to all decoders. In the Gray–Wyner framework, the shared description travels along the “common branch” accessed by every receiver. Our result reveals: when perspectives diverge, this branch must expand by exactly $+K(P)$ bits—regardless of how the private parts are structured (Gray & Wyner, 1974).



A single compressed message that every context can decode with vanishing error requires rate—

$$
\lim_{n \to \infty} \frac{1}{n} \mathbb{E}[\ell_n^*] = H(X|C) + K(P)
$$

Put differently—this establishes the communication cost of consensus.

(see App. A.9)