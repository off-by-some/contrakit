# **Theorem 10** *(Witnessing for TV-Approximation)*

There exist witnesses $W_n$ with rate $K(P)+o(1)$ and FI laws $\tilde{Q}_n$ such that $\mathrm{TV}((X^n, W_n), \tilde{Q}_n) \to 0$. No rate $< K(P)$ achieves vanishing $\mathrm{TV}$. Put differently—this establishes simulation bounds. And we do this through resolvability arguments.

**Proof Strategy:**

- *Achievability:* View the task as **distributed channel synthesis**—witnesses act as the **common randomness** that coordinates contexts. A resolvability-style construction (Han & Verdú, 1993) specialized to the distributed setting of Cuff (2013) draws a $2^{n(K(P)+\varepsilon)}$ FI codebook and selects an index passing a Bhattacharyya test; multiplicativity then gives $\mathrm{TV}\to 0$.

- *Converse:* A simulator with witness rate $< K(P)$ would contradict the exponent bound in Theorem 9. We establish this through contradiction.

  

---

## A.11 Proposition: Total-Variation Gap (TVG)

**Statement.**

For any behavior $P$,

$$
d_{\mathrm{TV}}(P,\mathrm{FI}) \ :=\ \inf_{Q\in\mathrm{FI}}\ \max_{c\in\mathcal C}\ \mathrm{TV}(p_c,q_c)
\ \ \ge\ 1-\alpha^\star(P)\ =\ 1-2^{-K(P)}.
$$

**Assumptions.**

Finite alphabets; $F=\mathrm{BC}$ (Def. A.2.1); $\mathrm{FI}$ convex and compact (Prop. A.1.6).

**Proof.**

For any distributions $p,q$, one has the inequality $\mathrm{BC}(p,q)\le 1-\mathrm{TV}(p,q)$ (standard Pinsker-type bound).

Thus for each context $c$ and any $Q\in\mathrm{FI}$,

$$
\mathrm{BC}(p_c,q_c)\ \le\ 1-\mathrm{TV}(p_c,q_c).
$$

Taking the minimum over $c$ and then the maximum over $Q$,

$$
\alpha^\star(P)\ =\ \max_{Q\in\mathrm{FI}}\ \min_c \mathrm{BC}(p_c,q_c)\ \le\ 1-\inf_{Q\in\mathrm{FI}}\ \max_c \mathrm{TV}(p_c,q_c)\ =\ 1-d_{\mathrm{TV}}(P,\mathrm{FI}).
$$

Rearranging yields

$$
d_{\mathrm{TV}}(P,\mathrm{FI}) \ \ge\ 1-\alpha^\star(P).
$$

Apply the log law $K(P)=-\log_2\alpha^\star(P)$ (Thm. A.6.4) to obtain the equivalent form $d_{\mathrm{TV}}(P,\mathrm{FI}) \ge 1-2^{-K(P)}$. □

**Diagnostics.**

The bound shows that contradiction cannot be hidden in total variation: any $\mathrm{FI}$ simulator within uniform TV $\le \varepsilon$ across contexts must have $\varepsilon\ge 1-2^{-K(P)}$. Contradiction bits therefore lower-bound *observable* statistical discrepancy.

**Cross-refs.**

Bhattacharyya bound (Lemma A.2.2); log law (Thm. A.6.4); definition of $d_{\mathrm{TV}}$ (Prop. 6.L).
