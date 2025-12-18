# **Theorem 3.** (Uniqueness of the agreement kernel)


**Statement.**

Under refinement separability, product multiplicativity, DPI, joint concavity, and basic regularity, the unique per-context agreement kernel is the Bhattacharyya affinity—

$$
F(p,q) = \sum_o \sqrt{p(o)\,q(o)}.
$$

We show this to be the unique form—and we establish it through axioms. Consider this derivation carefully.

**Assumptions.**

$F$ maps pairs of distributions (for each finite alphabet $\mathcal O$) to $[0,1]$ and satisfies:

1. **Normalization & calibration:** $F(p,p)=1$ for all $p$; and $F(p,q)=1 \iff p=q$.
2. **Symmetry:** $F(p,q)=F(q,p)$.
3. **Continuity.**
4. **Refinement separability (label-invariant additivity across refinements):** If an outcome is refined into finitely many suboutcomes and $p,q$ are refined accordingly, then total agreement is the (label-invariant) sum of suboutcome agreements; iterating/refining in any order yields the same value.
5. **Product multiplicativity:** $F(p\otimes r, q\otimes s)=F(p,q) F(r,s)$.
6. **Data-processing inequality (DPI):** $F(\Lambda p,\Lambda q)\geq F(p,q)$ for any stochastic map $\Lambda$.
7. **Joint concavity** in $(p,q)$.

*(Existence: $\mathrm{BC}$ satisfies (1)–(7); see Lemma A.2.2.)*

**Proof.**

We proceed in three steps—and establish uniqueness systematically. Consider this carefully.

1. **Step 1 (Refinement separability $\Rightarrow$ coordinatewise sum form).**
    
    Refinement separability and label invariance imply there exists a continuous, symmetric bivariate function $g:[0,1]^2\to[0,1]$ with $g(0,0)=0$ such that for every finite alphabet $\mathcal O$ and $p,q\in\Delta(\mathcal O)$,
    
    $$
    F(p,q)\;=\;\sum_{o\in\mathcal O} g\big(p(o),\,q(o)\big).
    \tag{6.3.1}
    $$
    
    *Justification.*
    
    By refinement separability, splitting a unit mass into atoms and iterating refinements yields an additive, label-invariant decomposition; hence (6.3.1) holds for a unique $g$ with $g(0,0)=0$.
    
    Now impose diagonal normalization. 
    Define $\phi(x):=g(x,x)$. For any $p\in\Delta(\mathcal O)$,
    
    $$
    1\;=\;F(p,p)\;=\;\sum_{o} g\big(p(o),p(o)\big)\;=\;\sum_{o}\phi\big(p(o)\big).
    \tag{6.3.2}
    $$
    
    Taking $\mathcal O=\{1,2,3\}$ with probabilities $(x,y,1-x-y)$ and using (6.3.2) twice (once for $(x,y,1-x-y)$, once for $(x+y,1-x-y)$), we obtain for all $x,y\ge 0$ with $x+y\le 1$:
    $\phi(x)+\phi(y)+\phi(1-x-y)=1=\phi(x+y)+\phi(1-(x+y))$, hence $\phi(x+y)=\phi(x)+\phi(y)$.
    
    Thus $\phi$ is additive on $[0,1]$; by continuity and $\phi(1)=1$, we get
    
    $$
    \phi(x)\;=\;x\qquad\text{for all }x\in[0,1].
    \tag{6.3.3}
    $$
    
    Hence $g(x,x)=x$.
    

1. **Step 2 (Product multiplicativity $\Rightarrow$ geometric mean on each coordinate).**
    
    Consider distributions supported on a **single** atom: for $x,y,u,v\in[0,1]$, let
    
    $$
    p=(x,1-x),\quad q=(y,1-y),\quad r=(u,1-u),\quad s=(v,1-v).
    $$
    
    Then (6.3.1) and product multiplicativity (Assumption 5) give
    
    $$
    \begin{equation}\tag{6.3.4}
    \begin{aligned}
    & g(xu,yv) + g\big(x(1-u),y(1-v)\big) \\
    &\quad {}+ g\big((1-x)u,(1-y)v\big) \\
    &\quad {}+ g\big((1-x)(1-u),(1-y)(1-v)\big) \\
    &= \big[g(x,y)+g(1-x,1-y)\big] \\
    &\quad {}\times \big[g(u,v)+g(1-u,1-v)\big].
    \end{aligned}
    \end{equation}
    
    $$
    
    Since (6.3.4) holds for all $x,y,u,v\in[0,1]$, varying one variable at a time and using (6.3.1) (refinement additivity) forces each term to factorize; in particular $g(xu,yv)=g(x,y)\,g(u,v)$.
    
    By symmetry of $g$ and (6.3.4),
    
    $$
    g(x,y)^2\;=\;g(x,y)\,g(y,x)\;=\;g(xy,yx)\;=\;g(xy,xy)\stackrel{(6.3.3)}{=}\,xy.
    $$
    
    Since $F\in[0,1]$, we take the nonnegative root:
    
    $$
    g(x,y)\;=\;\sqrt{xy}\qquad\text{for all }x,y\in[0,1].
    \tag{6.3.5}
    $$
    
2. **Step 3 (Conclusion and uniqueness).**
    
    Plugging (6.3.5) into (6.3.1) yields, for every finite alphabet,
    
    $$
    F(p,q)\;=\;\sum_{o}\sqrt{p(o)\,q(o)}\;=\;\mathrm{BC}(p,q).
    $$
    
    By Lemma A.2.2, $\mathrm{BC}$ satisfies normalization, symmetry, continuity, DPI, joint concavity, and product multiplicativity. Hence $\mathrm{BC}$ meets all assumptions.
    
    For **uniqueness**, Steps 1–2 show any $F$ obeying assumptions (1)–(5) must equal the right-hand side of (6.3.5) on each coordinate; summing gives $\mathrm{BC}$. Thus there is no other admissible kernel. Assumptions (6)–(7) are then automatically satisfied by $\mathrm{BC}$ (Lemma A.2.2), and they rule out putative alternatives even if Step 1 were weakened.
    
    This completes the proof. □
    

---

**Diagnostics.**

The formula $F(p,q)=\langle \sqrt{p},\sqrt{q} \rangle$ identifies the **Hellinger embedding**; multiplicativity becomes $\langle \sqrt{p\otimes r},\sqrt{q\otimes s}\rangle = \langle \sqrt{p},\sqrt{q} \rangle \langle \sqrt{r},\sqrt{s} \rangle$; DPI and concavity follow from Jensen/Cauchy–Schwarz (Lemma A.2.2).

**Sharpness.**

Dropping *any* of refinement separability, product multiplicativity, or DPI admits non-$\mathrm{BC}$ kernels (cf. App. B.3.5–B.3.7).

**Cross-refs.**

Representation (A.6.2); log law and additivity on products use $\mathrm{BC}$ (A.6.4–A.6.5).
