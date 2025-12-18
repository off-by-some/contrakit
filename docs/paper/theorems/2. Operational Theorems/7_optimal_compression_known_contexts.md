# **Theorem 7** *(Optimal Compression, Known Contexts)*

With $C^n$ available to the decoder—

$$
\lim_{n \to \infty} \frac{1}{n} \mathbb{E}[\ell_n^*] = H(X|C) + K(P)
$$

with a strong converse. Put differently—this establishes the fundamental compression bound.

**Proof Strategy:** 

The converses follow from Theorem 9: any compression rate below these thresholds would require simulating $P$ with a witness rate $<K(P)$, which would imply a hypothesis test exceeding the type-II exponent bound in Theorem 9 — which should be impossible.



---