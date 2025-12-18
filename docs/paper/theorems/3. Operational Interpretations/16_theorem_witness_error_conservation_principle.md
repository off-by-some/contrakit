# The Witness-Error Conservation Principle

**Theorem 7.4** *(Witness-Error Tradeoff)* (App. A.3.2, A.9)

Let a scheme use witness rate $r$ bits/symbol and achieve type-II error exponent $E$ for testing $\mathrm{FI}$ vs $P$. Then. We show this fundamental conservation principle:

$$
E + r \,\ge\, K(P)
$$

Moreover, there exist schemes achieving $E + r = K(P) \pm o(1)$. Put differently: the bound is tight.

**Corollary 7.4.1** *(Linear Tradeoff Curve)*

The optimal tradeoff is exactly linear: $E^*(r) = K(P) - r$ for $r \in [0, K(P)]$. For $r \,\ge\, K(P)$, $E^*(r) = 0$ (clipped at zero).

**Proof Strategy:**

- *Converse:* With $nr$ bits of witness, there are $\leq 2^{nr}$ witness values; union bound with the Bhattacharyya (Rényi-1/2) floor $K(P)$ gives an exponent shortfall of at most $r$.
- *Achievability:* Split resource: spend $nr$ bits on a witness (reducing the contradiction by $r$ via the product law/additivity), then test the residual with a Bhattacharyya-optimal statistic. The exponents add (App. A.9, Log Law). And this establishes the precise trade-off.

**Interpretation:**

This is a conservation law—every bit not spent on coordination must reappear as lost statistical power. There is no "free lunch" in multi-context inference. Consider this as the fundamental trade-off principle.

**Consequences:**

1. The tradeoff curve $E^*(r) = K(P) - r$ is exactly linear for $r \in [0, K(P)]$.
2. There is no "free lunch": every bit not spent on witnesses must reappear as lost testing power.

---