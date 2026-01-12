## A.12 Proposition: Smoothing Bound & Tightness

**Statement.**

For any behavior $P$, any $R\in\mathrm{FI}$, and any $t\in[0,1]$,

$$
K\!\big((1-t)P+tR\big)\ \le\ -\log_2\!\Big((1-t)\,2^{-K(P)}+t\Big)\ \le\ (1-t)\,K(P).
$$

This upper bound is *tight* whenever $R=Q^\star$ is an optimal FI simulator for $P$.

Moreover, to guarantee $K((1-t)P+tR)\le\kappa$ for some target $\kappa\ge0$, it suffices that

$$
t\ \ge\ \frac{1-2^{-\kappa}}{\,1-2^{-K(P)}\,}.
$$

**Assumptions.**

Finite alphabets; $\mathrm{FI}$ convex/compact (Prop. [A.1.6](#proposition-a16-topological-properties)); $\mathrm{BC}$ jointly concave in its first argument (Lemma [A.2.2](#lemma-a22-bhattacharyya-properties)).

**Proof.**

Start from the dual minimax form (Thm. [A.3.2](#theorem-a32-minimax-equality)):

$$
\alpha^\star(P)=\min_{\lambda\in\Delta(\mathcal C)} \max_{Q\in\mathrm{FI}}\ \sum_c \lambda_c\,\mathrm{BC}(p_c,q_c).
$$

For each $c$, concavity of $\mathrm{BC}$ in its first argument gives

$$
\mathrm{BC}\big((1-t)p_c+t r_c,\ q_c\big)\ \ge\ (1-t)\,\mathrm{BC}(p_c,q_c)+t\,\mathrm{BC}(r_c,q_c).
$$

Summing with weights $\lambda_c$ and maximizing over $Q$, then minimizing over $\lambda$, yields

$$
\alpha^\star\!\big((1-t)P+tR\big)\ \ge\ (1-t)\,\alpha^\star(P)+t\,\max_{Q\in\mathrm{FI}}\sum_c\lambda_c \mathrm{BC}(r_c,q_c).
$$

Taking $Q=R$ shows the last max is $\ge1$, hence

$$
\alpha^\star((1-t)P+tR)\ \ge\ (1-t)\,\alpha^\star(P)+t.
$$

Applying $K=-\log_2\alpha^\star$ (convex, decreasing) gives the bound

$$
K((1-t)P+tR)\ \le\ -\log_2\!\big((1-t)\,2^{-K(P)}+t\big).
$$

Convexity of $-\log$ also gives the linear relaxation $\le (1-t),K(P)$.

Tightness follows if $R=Q^\star$ is an FI optimizer for $P$: then concavity is met with equality.

The rearranged inequality

$$
t\ \ge\ \frac{1-2^{-\kappa}}{1-2^{-K(P)}}
$$

gives the minimal fraction $t$ of FI mixing needed to drive contradiction below $\kappa$. □

**Diagnostics.**

The result quantifies "FI smoothing": adding any amount of consistent noise lowers $K$ at least as fast as the displayed curve. The linear relaxation is coarse but easy to compute; the log form is exact when mixing along an optimizer.

**Cross-refs.**

Dual minimax (Thm. [A.3.2](#theorem-a32-minimax-equality)); concavity of $\mathrm{BC}$ (Lemma [A.2.2](#lemma-a22-bhattacharyya-properties)); log law (see A.9).