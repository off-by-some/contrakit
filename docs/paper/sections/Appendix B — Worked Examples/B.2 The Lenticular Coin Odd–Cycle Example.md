## B.2 The Lenticular Coin / Odd–Cycle Example

**Example B.2.1 (Odd-Cycle Contradiction).**

We model Nancy (N), Dylan (D), and Tyler (T) as three **binary** observables $X_N,X_D,X_T\in\{0,1\}$ encoding $\mathrm{YES}=1$, $\mathrm{NO}=0$. The three **contexts** are the pairs

$$
c_1=\{N,T\},\qquad c_2=\{T,D\},\qquad c_3=\{D,N\}
$$

The observed behavior $P=\{p_{c_i}\}_{i=1}^3$ is the **"perfect disagreement"** behavior on each edge:

$$
p_{c}(1,0)=p_{c}(0,1)=\tfrac12,\qquad p_{c}(0,0)=p_{c}(1,1)=0\quad\text{for }c\in\{c_1,c_2,c_3\}
$$

Equivalently: each visible pair always disagrees, but the direction of disagreement is uniformly random.

We compute

$$
\alpha^\star(P)\;=\;\max_{Q\in\mathrm{FI}}\ \min_{c\in\{c_1,c_2,c_3\}} \mathrm{BC}(p_c,q_c)
$$

and show $\alpha^\star(P)=\sqrt{\tfrac23}$, hence $K(P)=\tfrac12\log_2\!\frac32$.

#### B.2.1 Universal Upper Bound: $\alpha^\star\le \sqrt{2/3}$

#### Lemma B.2.2 (Upper Bound).

Let $Q\in\mathrm{FI}$ arise from a global law $\mu$ on $\{0,1\}^3$. For a context $c=\{i,j\}$, write the induced pair distribution

$$
q_c(00),\ q_c(01),\ q_c(10),\ q_c(11)\quad\text{and}\quad
D_{ij}:=q_c(01)+q_c(10)=\Pr_{\mu}[X_i\ne X_j]
$$

For our $p_c$ (uniform over the off-diagonals), the Bhattacharyya coefficient is

$$
\mathrm{BC}(p_c,q_c)
=\sqrt{\tfrac12}\big(\sqrt{q_c(01)}+\sqrt{q_c(10)}\big)
\le \sqrt{D_{ij}}
$$

with equality iff $q_c(01)=q_c(10)=D_{ij}/2$ (by concavity of $\sqrt{\cdot}$ at fixed sum).

Thus

$$
\min_{c}\ \mathrm{BC}(p_c,q_c)\ \le\ \min_{c}\ \sqrt{D_c}
$$

The triple $(D_{NT},D_{TD},D_{DN})$ must be feasible as edge-disagreement probabilities of a joint $\mu$ on $\{0,1\}^3$. For three bits, every deterministic assignment has either 0 disagreements (all equal) or exactly 2 (one bit flips against the other two). Hence any convex combination obeys the **cut-polytope constraint**

$$
D_{NT}+D_{TD}+D_{DN}\ \le\ 2
$$

Consequently at least one edge has $D_c\le 2/3$, so

$$
\min_c \sqrt{D_c}\ \le\ \sqrt{2/3}
$$

Taking the maximum over $Q\in\mathrm{FI}$ yields the universal upper bound

$$
\alpha^\star(P)\ \le\ \sqrt{2/3}
$$

#### B.2.2 Achievability: An Explicit Optimal $\mu^\star$

#### Proposition B.2.3 (Achievability).

Let $\mu^\star$ be the **uniform** distribution over the six nonconstant bitstrings:

$$
\mu^\star=\text{Unif}\big(\{100,010,001,011,101,110\}\big)
$$

(Equivalently: put zero mass on $000$ and $111$, equal mass on Hamming-weight $1$ and $2$ states.)

A direct check shows that for any edge $c\in\{c_1,c_2,c_3\}$,

$$
q_c^\star(01)=q_c^\star(10)=\tfrac13,\qquad q_c^\star(00)=q_c^\star(11)=\tfrac16
$$

so $D_c^\star=q_c^\star(01)+q_c^\star(10)=\tfrac23$ and the off-diagonals are **balanced**, hence the BC upper bound is tight:

$$
\mathrm{BC}(p_c,q_c^\star)
=\sqrt{\tfrac12}\big(\sqrt{\tfrac13}+\sqrt{\tfrac13}\big)
=\sqrt{\tfrac23}\quad\text{for each }c
$$

Therefore

$$
\min_{c}\mathrm{BC}(p_c,q_c^\star)=\sqrt{\tfrac23}
$$

which matches the upper bound. We conclude

$$
\boxed{\ \alpha^\star(P)=\sqrt{\tfrac23}\ ,\qquad
K(P)=-\log_2\alpha^\star(P)=\tfrac12\log_2\!\frac32\ }
$$

#### Corollary B.2.4 (Optimal Witness).

By symmetry, any optimal contradiction witness $\lambda^\star$ can be taken **uniform** on the three contexts. Moreover, the equalization $\mathrm{BC}(p_c,q_c^\star)=\alpha^\star$ on all edges shows $\lambda^\star$ may place positive mass on each context (cf. the support condition in the minimax duality).