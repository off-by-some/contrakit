## 6.5 Multi-Decoder Communication

The natural baseline for a multi-decoder system is the common-message architecture—a single representation delivered to all decoders. In the Gray–Wyner framework, the shared description travels along the “common branch” accessed by every receiver. Our result reveals: when perspectives diverge, this branch must expand by exactly $+K(P)$ bits—regardless of how the private parts are structured (Gray & Wyner, 1974).

**Theorem 12** *(Common Representation Cost)*
If representation $Z = Z(X^n)$ enables every context decoder to recover $X^n$ with vanishing error—

- Known contexts: $\frac{1}{n} I(X^n; Z) \ge H(X|C) + K(P) - o(1)$
- Latent contexts: $\frac{1}{n} I(X^n; Z) \ge H(X) + K(P) - o(1)$

Put differently—this gives the fundamental representation bounds.

**Proof Strategy:** 
Source-coding lower bounds give $I(X^n; Z) \ge \mathbb{E}[\ell_n] - o(n)$; apply Theorem 11 (known contexts) or Theorem 8 (latent contexts).