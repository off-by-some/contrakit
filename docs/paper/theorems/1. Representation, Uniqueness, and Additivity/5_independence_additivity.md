# A.10 Theorem 5: Additivity on Independent Products

**Statement.**

For independent systems on disjoint observables with product-closed $\mathrm{FI}$—

$$
\alpha^\star(P\otimes R)=\alpha^\star(P)\,\alpha^\star(R)
\quad\Longrightarrow\quad
K(P\otimes R)=K(P)+K(R).
$$

In short—this establishes additivity.

**Assumptions.**

Finite alphabets; $\mathrm{FI}$ convex/compact and product-closed (Prop. A.1.6, A.1.8); $F=\mathrm{BC}$ (Def. A.2.1; Lemma A.2.2).

**Proof.**

Let $P$ be a behavior on $(\mathcal{X},\mathcal{C})$ and $R$ on $(\mathcal{Y},\mathcal{D})$ with $\mathcal{X}\cap\mathcal{Y}=\emptyset$. Write—

$$
\alpha^\star(P)=\max_{Q_A\in\mathrm{FI}_{\mathcal{X},\mathcal{C}}}\ \min_{c\in\mathcal{C}}\mathrm{BC}(p_c,q_c),
\qquad
\alpha^\star(R)=\max_{Q_B\in\mathrm{FI}_{\mathcal{Y},\mathcal{D}}}\ \min_{d\in\mathcal{D}}\mathrm{BC}(r_d,s_d).
$$

We now establish the bounds systematically—and we do this through careful analysis. Consider the implications thoroughly.

**Lower bound ($\geq$).**

Let $Q_A^\star,Q_B^\star$ be maximizers for $P$ and $R$. By product closure (Prop. A.1.8), $Q_{AB}:=Q_A^\star\otimes Q_B^\star\in\mathrm{FI}_{\mathcal{X}\sqcup\mathcal{Y},\ \mathcal{C}\otimes\mathcal{D}}$, and for any $(c,d)$,

$\mathrm{BC}\!\big(p_c\otimes r_d,\ q_c^\star\otimes s_d^\star\big)
=\mathrm{BC}(p_c,q_c^\star)\,\mathrm{BC}(r_d,s_d^\star)$ **(Lemma A.2.2, product rule)**.

Therefore

$$
\min_{(c,d)} \mathrm{BC}\!\big(p_c\otimes r_d,\ q_c^\star\otimes s_d^\star\big)
=\Big(\min_c \mathrm{BC}(p_c,q_c^\star)\Big)\Big(\min_d \mathrm{BC}(r_d,s_d^\star)\Big)
= \alpha^\star(P)\,\alpha^\star(R).
$$

Maximizing over $Q_{AB}$ yields $\alpha^\star(P\otimes R)\geq \alpha^\star(P)\alpha^\star(R)$.

**Upper bound ($\leq$).**

Use the dual form (Thm. A.3.2):

$$
\alpha^\star(P\otimes R)
=\min_{\mu\in\Delta(\mathcal{C}\otimes\mathcal{D})}
\ \max_{Q_{AB}\in\mathrm{FI}_{\mathcal{X}\sqcup\mathcal{Y}}}
\ \sum_{c,d}\mu_{c,d}\,\mathrm{BC}\!\big(p_c\otimes r_d,\ q_{c\cup d}\big).
$$

1. **Restriction of the minimizer.** Restrict the minimization to product weights $\mu=\mu_A\otimes \mu_B$ with $\mu_A\in\Delta(\mathcal{C}),\ \mu_B\in\Delta(\mathcal{D})$. Since we minimize over a smaller set, the value can only **increase**, hence this yields an **upper bound**:
    
    $$
    \alpha^\star(P\otimes R)
    \ \leq\
    \min_{\mu_A,\mu_B}\ \max_{Q_{AB}\in\mathrm{FI}}\ \sum_{c,d}\mu_A(c)\mu_B(d)\,\mathrm{BC}\!\big(p_c\otimes r_d,\ q_{c\cup d}\big).
    $$
    
2. **Restriction of the maximizer.** The objective is concave in $Q_{AB}$ (Lemma A.2.2, joint concavity; linear summation preserves concavity). Therefore the maximum over the convex compact set $\mathrm{FI}$ is attained at an extreme point (a convex combination of deterministic assignments), which factorizes across disjoint subsystems: $q_{c\cup d}=q_c\otimes s_d$. Thus we may restrict to product FI models $Q_{AB}=Q_A\otimes Q_B$ **without loss**, and certainly without violating the upper bound:
    
    $$
    \alpha^\star(P\otimes R)
    \ \leq\
    \min_{\mu_A,\mu_B}\ \max_{Q_A,Q_B}\ \sum_{c,d}\mu_A(c)\mu_B(d)\,\mathrm{BC}(p_c,q_c)\,\mathrm{BC}(r_d,s_d).
    $$
    
3. **Factorization.** Using multiplicativity of $\mathrm{BC}$ and Fubini,
    
    $$
    \sum_{c,d}\mu_A(c)\mu_B(d)\,\mathrm{BC}(p_c,q_c)\,\mathrm{BC}(r_d,s_d)
    =\Big(\sum_c \mu_A(c)\mathrm{BC}(p_c,q_c)\Big)\Big(\sum_d \mu_B(d)\mathrm{BC}(r_d,s_d)\Big).
    $$
    
    Thus
    
    $$
    \alpha^\star(P\otimes R)
    \ \leq\
    \min_{\mu_A}\max_{Q_A}\sum_c \mu_A(c)\mathrm{BC}(p_c,q_c)\ \times\
    \min_{\mu_B}\max_{Q_B}\sum_d \mu_B(d)\mathrm{BC}(r_d,s_d)
    = \alpha^\star(P)\,\alpha^\star(R).
    $$
    
    Together these give equality, $\alpha^\star(P\otimes R)=\alpha^\star(P)\alpha^\star(R)$. The additivity of $K$ follows from the log law (A.6.4): $K=-\log_2\alpha^\star$ gives
    
    $$
    K(P\otimes R)= -\log_2\big(\alpha^\star(P)\alpha^\star(R)\big)=K(P)+K(R).
    $$
    
    □
    

**Diagnostics.**

Independence composes multiplicatively at the agreement level and additively in contradiction bits; the only structural inputs are FI product structure (Prop. A.1.8), concavity and multiplicativity of $\mathrm{BC}$ (Lemma A.2.2), and the minimax dual (Thm. A.3.2).

**Cross-refs.**

$\mathrm{BC}$ multiplicativity (Lemma A.2.2); FI product structure (Prop. A.1.8); log law (A.6.4).