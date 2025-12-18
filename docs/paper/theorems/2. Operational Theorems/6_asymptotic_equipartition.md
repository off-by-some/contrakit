## 6.1 Asymptotic Equipartition with Tax

**Theorem 6** *(AEP with Contradiction Tax)*

For a framed i.i.d. source, there exist witnesses $W_n$ of rate $K(P)$ such that high-probability sets for $(X^n, W_n)$ have exponent $H(X|C) + K(P)$.

Conversely, if the witness rate is $< K(P)$, then any sets $\mathcal{S}_n$ with $\Pr[(X^n, W_n) \in \mathcal{S}_n] \to 1$ must satisfy—

$$
\liminf_{n \to \infty} \frac{1}{n} \log_2 |\mathcal{S}_n| \ge H(X|C) + K(P)
$$

Put differently—this establishes the fundamental bound.

**Proof Strategy:**

- *Achievability:* Cover $X^n$ given $C^n$ by a conditional typical set of size $\approx 2^{nH(X|C)}$; then **soft-cover** the cross-context mismatch by appending a **witness** of length $\approx nK(P)$. The construction is a direct **resolvability** argument in the sense of **Han & Verdú (1993)**—a random codebook of FI laws at rate $K(P)+\varepsilon$ drives the Bhattacharyya overlap to its minimax target via Rényi-1/2 soft covering, yielding $\mathrm{TV}$-closeness to an FI surrogate.
- *Converse:* If witness rate is $< K(P)$, one gets a level-$\eta$ test $\mathrm{FI}$ vs. $P$ whose type-II exponent would exceed $K(P)$; Chernoff at $s = 1/2$ (Bhattacharyya) forbids this.

We show this through careful construction—and we establish it rigorously. Consider this result carefully.

**Corollary 6.1** *(Meta-AEP with Three Regimes)*

With witnesses $W_n\in\{0,1\}^{m_n}$ and $m_n/n\to K(P)$, there exist meta-typical sets $\mathcal T_\varepsilon^n$ with $P(\mathcal T_\varepsilon^n)\ge 1-\varepsilon$ and

$$
\frac{1}{n}\log_2|\mathcal T_\varepsilon^n|=
\begin{cases}
H(X)+K(P), & \text{latent contexts},\\
H(X\mid C)+K(P), & \text{known contexts at decoder},\\
H(C)+H(X\mid C)+K(P), & \text{contexts in message header.}
\end{cases}
$$

