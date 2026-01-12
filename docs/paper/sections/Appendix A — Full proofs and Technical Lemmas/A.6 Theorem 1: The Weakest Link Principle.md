## A.6  Theorem 1: The Weakest Link Principle

**Statement.**

Any unanimity-respecting, monotone aggregator on $[0,1]^{\mathcal C}$ that never exceeds any coordinate equals the minimum.

**Assumptions.**

- $\mathcal{C}$ finite, nonempty; $A:[0,1]^{\mathcal{C}} \to [0,1]$.
- $A$ satisfies:
  1. **Monotonicity:** $x\le y \Rightarrow A(x)\le A(y)$.
  2. **Idempotence (unanimity):** $A(t,\dots,t)=t$ for all $t\in[0,1]$.
  3. **Local upper bound (weakest-link cap):** $A(x)\le x_i$ for all $i\in\mathcal{C}$.

**Claim.**

For all $x\in[0,1]^{\mathcal{C}}$, $A(x)=\min_{i\in\mathcal{C}}x_i$.

$$
A(x) \;=\; \min_{i \in \mathcal{C}} x_i \quad \text{for all } x.
$$

**Proof.**

1. Let $m=\min_{i\in\mathcal{C}} x_i$ (exists since $\mathcal{C}$ is finite and nonempty).
2. (i)+(ii): $(m,\ldots,m)\le x \Rightarrow A(x)\ge A(m,\ldots,m)=m$.
3. (iii): $A(x)\le x_i$ for all $i \Rightarrow A(x)\le m$. Hence $A(x)=m$. □

**Diagnostics / Consequences.**

1. **Bottleneck truth.** Overall agreement is set by the worst context. High scores elsewhere cannot compensate. This is the precise asymmetry: a single low coordinate rules the aggregate.
2. **Game alignment.** In [§3.4](#34-the-game-theoretic-structure) the payoff is $\max_Q \min_c \mathrm{BC}(p_c,q_c)$. The "min" over contexts is not a choice—it is forced by weakest-link aggregation under A0–A4. You do not average tests; you survive the hardest one.
3. **Diagnostics.** The contexts that attain the minimum are exactly those that receive positive weight in $\lambda^{\star}$ (the active constraints). They identify *where* reconciliation fails.
4. **Stability under bookkeeping.** Duplicating or splitting non-bottleneck contexts leaves the minimum unchanged (A4), preventing frequency from inflating consensus.
5. **Decision consequence.** For any threshold $\tau$, if some context has $BC(p_c,q_c)<\tau$, no strength elsewhere can raise the aggregate above $\tau$. Hence $K=-\log_2\alpha^{\star}$ inherits a strict "weakest-case" guarantee.

A quick contrast: with $x=(0.90,0.60,0.95)$, the minimum is $0.60$; an average would report $0.816$, overstating consensus and violating grouping/monotonicity under duplication of the weak row.