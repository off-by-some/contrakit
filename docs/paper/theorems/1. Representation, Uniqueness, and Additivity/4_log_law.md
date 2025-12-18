## A.9 Theorem 4: Fundamental Formula (Log Law)

**Statement.**

Under A0–A5, the contradiction measure is uniquely (up to units)—

$$
K(P)\;=\;-\log_2 \alpha^\star(P),\qquad 
\alpha^\star(P)\;=\;\max_{Q\in\mathrm{FI}}\ \min_{c\in\mathcal C}\mathrm{BC}(p_c,q_c).
$$

Put differently—this establishes the fundamental formula. We derive this from the axioms. And we do this systematically.

**Assumptions.**

Finite alphabets; $\mathrm{FI}$ convex/compact and product-closed (Prop. A.1.6, A.1.8). Axioms A0–A5 (Label invariance, Reduction/Calibration, Continuity, Free-ops monotonicity/DPI, Grouping, Independent composition). Kernel uniqueness $F=\mathrm{BC}$ (Thm. A.6.3).

**Proof.**

Let—

$$
\alpha^\star(P)\;:=\;\max_{Q\in\mathrm{FI}}\ \min_{c\in\mathcal C}\mathrm{BC}(p_c,q_c)\ \in (0,1].
$$

By compactness of $\mathrm{FI}$ and continuity of $\mathrm{BC}$ (Lemma A.2.2), the maximum is attained. Lemma A.4.1 ensures $\alpha^\star(P)>0$, so the logarithm below is finite.

By the representation theorem (A.6.2) specialized to $F=\mathrm{BC}$ (A.6.3), any admissible contradiction measure obeying A0–A4 must be of the form

$$
K(P)\;=\;h\!\big(\alpha^\star(P)\big)
$$

for some strictly decreasing, continuous $h:(0,1]\to \mathbb{R}_{\ge0}$ with $h(1)=0$.

A5 (independent composition) states $K(P\otimes R)=K(P)+K(R)$. Using Prop. A.1.8 (product structure of FI) and Lemma A.2.2.4 (product structure of $\mathrm{BC}$), Theorem A.6.5 (proved below) yields the **product law**

$$
\alpha^\star(P\otimes R)\;=\;\alpha^\star(P)\,\alpha^\star(R).
\tag{6.4.1}
$$

Therefore,

$$
h\!\big(\alpha^\star(P)\alpha^\star(R)\big)\;=\;h\!\big(\alpha^\star(P\otimes R)\big)\;=\;K(P\otimes R)\;=\;K(P)+K(R)\;=\;h\!\big(\alpha^\star(P)\big)+h\!\big(\alpha^\star(R)\big).
$$

Thus $h$ satisfies Cauchy's multiplicative equation on $(0,1]$:

$$
h(xy)=h(x)+h(y).
$$

Together with continuity and $h(1)=0$, the unique solutions are $h(x)=-k\log x$ with a constant $k>0$. Choosing **bits** as units fixes $k=1/\ln 2$, i.e.

$$
h(x)\;=\;-\log_2 x.
$$

Hence $K(P)=-\log_2 \alpha^\star(P)$, and writing $\alpha^\star:=\alpha$ gives the stated formula.

*Uniqueness (up to units).*

If $\tilde K$ also satisfies A0–A5, then $\tilde K=h\circ\alpha^\star$ for some $h$ as above; A5 enforces $h(x)=-k\log x$. Different choices of the constant $k$ correspond exactly to a change of units (e.g., nats vs. bits). □

**Diagnostics.**

- Additivity on independent products: $K(P\otimes R)=K(P)+K(R)$ follows from (6.4.1) and the log law.
- Calibration: $K(P)=0 \iff \alpha^\star(P)=1 \iff P\in\mathrm{FI}$ (Thm. A.4.3).
- Scale: Each halving of $\alpha^\star$ increases $K$ by one bit.

**Cross-refs.**

Product law for $\alpha^\star$ (A.6.5); bounds (A.4.1–A.4.2); representation (A.6.2); kernel uniqueness $F=\mathrm{BC}$ (A.6.3).

---
