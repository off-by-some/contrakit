## A.11 Proposition: Total-Variation Gap (TVG)

**Statement.**

For any behavior $P$,

$$
d_{\mathrm{TV}}(P,\mathrm{FI}) \ :=\ \inf_{Q\in\mathrm{FI}}\ \max_{c\in\mathcal C}\ \mathrm{TV}(p_c,q_c)
\ \ \ge\ 1-\alpha^\star(P)\ =\ 1-2^{-K(P)}.
$$

**Assumptions.**

Finite alphabets; $F=\mathrm{BC}$ (Def. [A.2.1](#definition-a21-bhattacharyya-coefficient)); $\mathrm{FI}$ convex and compact (Prop. [A.1.6](#proposition-a16-topological-properties)).

**Proof.**

For any distributions $p,q$, one has the inequality $\mathrm{TV}(p,q)\ge 1-\mathrm{BC}(p,q)$ (standard Pinsker-type bound).

Thus for each context $c$ and any $Q\in\mathrm{FI}$,

$$
\mathrm{TV}(p_c,q_c)\ \ge\ 1-\mathrm{BC}(p_c,q_c).
$$

Taking the maximum over $c$ and then the infimum over $Q$,

$$
d_{\mathrm{TV}}(P,\mathrm{FI})\ =\ \inf_{Q\in\mathrm{FI}}\ \max_c \mathrm{TV}(p_c,q_c)\ \ge\ \inf_{Q\in\mathrm{FI}}\ \max_c (1-\mathrm{BC}(p_c,q_c))\ =\ 1-\sup_{Q\in\mathrm{FI}}\ \min_c \mathrm{BC}(p_c,q_c)\ =\ 1-\alpha^\star(P).
$$

Rearranging yields

$$
d_{\mathrm{TV}}(P,\mathrm{FI}) \ \ge\ 1-\alpha^\star(P).
$$

Apply the log law $K(P)=-\log_2\alpha^\star(P)$ (see A.9) to obtain the equivalent form $d_{\mathrm{TV}}(P,\mathrm{FI}) \ge 1-2^{-K(P)}$. □

**Diagnostics.**

The bound shows that contradiction cannot be hidden in total variation: any $\mathrm{FI}$ simulator within uniform TV $\le \varepsilon$ across contexts must have $\varepsilon\ge 1-2^{-K(P)}$. Contradiction bits therefore lower-bound *observable* statistical discrepancy.

**Cross-refs.**

Bhattacharyya bound (Lemma [A.2.2](#lemma-a22-bhattacharyya-properties)); log law (see A.9); definition of $d_{\mathrm{TV}}$ (see §6.7).