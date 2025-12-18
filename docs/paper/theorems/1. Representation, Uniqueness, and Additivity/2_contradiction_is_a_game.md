# **Theorem 2.** (Contradiction is a Game.)

**Statement.**

Any contradiction measure $K$ obeying A0–A4 admits a minimax form—

$$
K(P)=h\!\left(\max_{Q\in \mathrm{FI}}\min_{c\in\mathcal C} F(p_c,q_c)\right) = h\!\left(\min_{\lambda\in\Delta(\mathcal C)}\max_{Q\in\mathrm{FI}}\sum_c \lambda_c F(p_c,q_c)\right),
$$

for some per-context agreement kernel $F$ with normalization, symmetry, continuity, DPI, joint concavity, and calibration, and some strictly decreasing continuous $h$.

Optima are attained. We establish this through careful construction. And we do this rigorously. 


Any contradiction measure $K$ satisfying A0–A4 admits an adversarial (min–max) representation:

$$
K(P)\;=\;h\!\left(\max_{Q\in\mathrm{FI}}\ \min_{c\in\mathcal C} F\big(p_c,q_c\big)\right),
$$

for some per-context agreement functional $F: \Delta(\mathcal{O})\times\Delta(\mathcal{O})\rightarrow[0,1]$ (for each finite alphabet $\mathcal{O}$) and a strictly decreasing, continuous scalar map $h:(0,1]\to\mathbb{R}_{\ge0}$.

Equivalently—

$$
K(P)\;=\;h\!\left(\min_{\lambda\in\Delta(\mathcal C)}\ \max_{Q\in\mathrm{FI}}\ \sum_{c}\lambda_c\,F\big(p_c,q_c\big)\right).
$$

Put differently—this captures the adversarial nature.

**Assumptions**

- Finite alphabets; standing §3 conditions: $\mathrm{FI}$ nonempty, convex, compact (product-closure not needed here).
- Axioms A0–A4 (Label invariance, Reduction, Continuity, Free-ops monotonicity, Grouping).
- $F$ is an agreement score on finite simplices with:
(i) normalization $F(p,p)=1$; (ii) symmetry; (iii) continuity;
(iv) **data-processing monotonicity** (DPI): $F(\Phi p,\Phi q)\ge F(p,q)$ 
       for any stochastic map $\Phi$; 
(v) **joint concavity** in $(p,q)$;
(vi) calibration $F(p,q)=1\iff p=q$.

**Claim.**
Under the assumptions, there exist $F$ and $h$ with the listed properties such that the two displays above hold, and the inner optima are attained.

**Proof.**

1. **Compare only to frame-independent surrogates (A1, A3).**
Free-ops monotonicity makes $K$ nonincreasing under post-processing and context lotteries. Hence any meaningful "distance to consensus" must be computed against $\mathrm{FI}$; comparing to non-FI $Q$'s would violate A3 under coarse-grainings. This yields the **outer $\max_{Q\in\mathrm{FI}}$**.
2. **Reduce to per-context scores (A4).**
Grouping insensitivity removes dependence on sampling multiplicities. Thus $K$ can only depend on the *set* of context-wise agreements between $P$ and a candidate $Q$, i.e., the vector $\{F(p_c,q_c)\}_c$.
3. **Force weakest-link aggregation (A0–A4 + Thm. A.6.1).**
Label invariance and continuity constrain any aggregator on $[0,1]^\mathcal C$ to be unanimity-respecting and monotone; the weakest-link cap follows from A3 (each coordinate upper-bounds any admissible aggregate). By Theorem A.6.1, the only such aggregator is the **minimum**. Hence the inner aggregator is **$\min_{c}$**.
4. **Monotone rescaling freedom.**
Calibration and continuity (A1–A2) permit a strictly decreasing, continuous reparameterization $h$ of the agreement scale $\alpha(P):=\max_{Q\in\mathrm{FI}}\min_c F(p_c,q_c) \in (0,1]$. By compactness of $\mathrm{FI}$ and continuity of $F$, the maximum is attained; $\alpha(P)\in(0,1]$, and $\alpha(P)=1$ iff $P\in\mathrm{FI}$ (by normalization and calibration of $F$). (If one uses a kernel $F$ that can attain $0$, replace $(0,1]$ by $[0,1]$; for $F=\mathrm{BC}$, Lemma A.4.1 ensures $\alpha(P)>0$.) Thus $K(P)=h(\alpha(P))$.
5. **Dual form and attainment (Sion).**
For fixed $Q$, with $a_c(Q):=F(p_c,q_c)$, $\min_{\lambda\in\Delta(\mathcal C)}\sum_c \lambda_c a_c(Q)=\min_{c} a_c(Q)$. 
Since $Q\mapsto \sum_c \lambda_c F(p_c,q_c)$ is concave on compact convex $\mathrm{FI}$ (joint concavity of $F$) and linear in $\lambda$, Sion's minimax theorem gives:
$\max_{Q\in\mathrm{FI}}\min_{c}F(p_c,q_c)\;=\;\min_{\lambda\in\Delta(\mathcal C)}\max_{Q\in\mathrm{FI}}\sum_c \lambda_c F(p_c,q_c),$
with optima attained by compactness/continuity. 
Composing with $h$ proves both displays. □

**Diagnostics / Consequences.**

1. **Worst-case testing.** A single hard context governs $\alpha(P)$ (hence $K$).
2. **Active constraints.** Minimizers $c$ attaining $\min_c F(p_c,q_c^\star)$ coincide with the support of any optimal $\lambda^\star$ in the dual (cf. A.5.1).
3. **Stability.** Grouping invariance means duplicating or splitting non-active contexts leaves $\alpha(P)$ unchanged.
4. **Bhattacharyya specialization.** Any $F$ obeying DPI + joint concavity fits the scheme; for $F=\mathrm{BC}$, $\max_{Q\in\mathrm{FI}}\min_{c}\mathrm{BC}(p_c,q_c)=\min_{\lambda\in\Delta(\mathcal C)}\max_{Q\in\mathrm{FI}}\sum_c \lambda_c \,\mathrm{BC}(p_c,q_c),$

with optima attained and $\lambda^\star$ interpretable as adversarial context weights (A.5.1).

**Sharpness.**
Drop A4 and averaging over contexts can be forced (violates weakest-link, App. B.3.6). Drop DPI and post-processing can manufacture contradiction (App. B.3.5). Either failure breaks the game form.

**Cross-refs.**

- Kernel uniqueness $\Rightarrow F=\mathrm{BC}$: Theorem A.6.3.
- Log law (pinning $h$ to $-\log$): Theorem A.6.4.
- Additivity on products: Theorem A.6.5.
