
## B.3 Axiom Ablation Analysis

The axiomatization developed above is not merely a convenient characterization—it reveals the fundamental structure of contradiction as a resource.

Put differently—we show this to be essential.

In this resource theory, frame-independent behaviors $\mathrm{FI}$ constitute the free objects, the free operations consist of 

1. outcome post-processing by arbitrary stochastic kernels $\Lambda_c$ applied within each context $c$,
2. public lotteries over contexts (independent of outcomes and hidden variables), and the contradiction measure $K$ serves as a faithful, convex, additive monotone: $K(P) \geq 0$ with equality precisely on $\mathrm{FI}$, $K$ is non-increasing under free operations, and $K$ is additive for independent systems (noting that for $|\mathcal{C}|=1$, all axioms collapse correctly and $K \equiv 0$).

To establish the robustness of this framework, we examine both weakenings and violations of each axiom. This analysis serves two purposes—it demonstrates that our axioms capture the minimal necessary structure, and it illuminates how each constraint contributes to the theory's coherence.

Formally, we proceed as follows.

#### B.3.1 Axiom Weakening

Several axioms admit natural weakenings that preserve the main theoretical results while relaxing technical requirements:

**Continuity Relaxation (A2).** 

Full continuity in the product $\|\cdot\|_1$ metric can be weakened to lower semicontinuity of $K$ (equivalently, upper semicontinuity of $\alpha^\star$). This suffices because our proofs require continuity only to ensure the existence of optima and to prevent discontinuous jumps at the $\mathrm{FI}$ boundary. Lower semicontinuity guarantees both properties directly.

**Grouping Decomposition (A4).** 

The grouping axiom equivalently follows from two more primitive requirements: *replication invariance* (duplicating any context row leaves $K$ unchanged) and *context anonymity* (permutation invariance over contexts). These conditions together imply invariance under public lotteries over contexts that are independent of outcomes and hidden variables—the lottery-splitting invariance employed in our main results.

**Compositional Robustness (A5).** 

When $\mathrm{FI}_{A\times B}$ fails to be strictly product-closed, exact additivity still holds for the product-closed hull $\overline{\mathrm{FI}_A\otimes \mathrm{FI}_B}$—the smallest convex, closed set containing all products:

$$
Q_A\in \mathrm{FI}_A,\quad Q_B\in \mathrm{FI}B
\;\Longrightarrow\;
Q_A\otimes Q_B \in \mathrm{FI}{A\times B}.
$$

More generally, we obtain subadditivity $K(P\otimes R) \leq K(P) + K(R)$ with equality precisely when the optimal dual weights factorize across $A$ and $B$ (i.e., $\lambda^\star{A\times B} = \lambda^\star_A \otimes \lambda^\star_B$)—a condition that can be verified independently.

We now demonstrate that each axiom is essential by constructing explicit counterexamples that satisfy all remaining axioms while violating the target property.

It is fair to ask whether these weakenings suffice—but the counterexamples show otherwise. 

#### B.3.2 Label Invariance (A0).

*Counterexample:* Define $\tilde K(P) := \|p(\cdot\!\mid c_0) - p(\cdot\!\mid c_1)\|_1$ for fixed context labels $c_0, c_1$.
*Failure mode:* Context relabeling changes $\tilde K$ despite identical behavioral content, violating the principle that physical properties should depend only on operational statistics.

#### B.3.3 FI-Characterization (A1).

*Counterexample:* Define $\tilde K(P) := \min_c \|p(\cdot\!\mid c) - u\|_1$ where $u$ is the uniform distribution.
*Failure mode:* For $P \in \mathrm{FI}$ with non-uniform rows, $\tilde K(P) > 0$, incorrectly assigning nonzero contradiction to some frame-independent behaviors.

#### B.3.4 Continuity (A2).

*Counterexample:* Define $\tilde K(P) := \mathbf{1}\{\alpha^\star(P) < 1\}$.
*Failure mode:* Consider a sequence $P_n \to P$ with $\alpha^\star(P_n) \uparrow 1$. Then $\tilde K(P_n) = 1$ for all $n$, but $\tilde K(P) = 0$, creating a discontinuous jump at the $\mathrm{FI}$ boundary.

#### B.3.5 Data-Processing Monotonicity (A3).

*Counterexample:* Replace Bhattacharyya coefficient with linear overlap:

$$
\tilde\alpha(P) := \max_{Q\in\mathrm{FI}} \min_c \langle p_c, q_c \rangle, \quad \tilde K := -\log \tilde\alpha
$$

*Failure mode:* Linear overlap fails as a data-processing monotone. For example, take $p=q=(1,0)$ with linear overlap $\langle p,q\rangle=1$. Coarse-graining with 

$$
\Lambda=\begin{pmatrix}
\tfrac12 & \tfrac12\\[2pt]
\tfrac12 & \tfrac12
\end{pmatrix} 
$$

gives $\Lambda p=\Lambda q=(1/2,1/2)$ and $\langle \Lambda p,\Lambda q\rangle=1/2<1$, so similarity decreases under processing, allowing coarse-graining to artificially increase contradiction.

#### B.3.6 Context Grouping (A4).

*Counterexample:* Define

$$
\tilde\alpha(P) := \max_{Q\in\mathrm{FI}} \frac{1}{|\mathcal{C}|} \sum_c \mathrm{BC}(p_c, q_c), \quad \tilde K := -\log \tilde\alpha

$$

*Failure mode:* When one context acts as a bottleneck (smallest Bhattacharyya coefficient), duplicating that context row decreases the arithmetic mean, causing $\tilde K$ to increase despite unchanged behavioral content.

#### B.3.7 Additivity (A5).

*Counterexample:* Retain Bhattacharyya coefficient but use non-logarithmic aggregation $h(x) = 1-x$:

$$
\tilde K(P) = 1 - \alpha^\star(P)
$$

*Failure mode:* For independent systems $P = R$ with $\alpha^\star(P) = 1/2$, we get $\tilde K(P \otimes P) = 1 - 1/4 = 3/4$ while $\tilde K(P) + \tilde K(P) = 1$, violating additivity (indeed, this aggregator yields strict subadditivity in general). 

<aside>

Note that additivity on independent products forces $h(xy) = h(x) + h(y)$ (the functional equation for logarithms). 

With domain $h:(0,1] \to [0,\infty)$, normalization $h(1)=0$, monotonicity ($x_1 \leq x_2 \Rightarrow h(x_1) \geq h(x_2)$), and continuity, the Cauchy functional equation uniquely determines $h(x) = -k\log x$. 

Choosing bits as our unit fixes $k=1/\ln 2$, yielding $h(x) = -\log_2 x$. (Note that as $\alpha^\star \downarrow 0$, $K \to \infty$, properly capturing "complete contradiction" with infinite information cost.)

</aside>

#### B.3.8 Ablation Summary

Thus, additivity on independent systems forces a logarithmic form. Fixing base-2 units (bits) then yields the unique measure consistent with A0–A5:

$$
K(P) \;=\; -\log_2 \alpha^\star(P).
$$

Equivalently, every halving of $\alpha^\star$ increases $K$ by one bit. Any other admissible measure is merely a unit change—a positive rescaling or a different logarithm base (i.e., a constant multiple of $K$). This completes the violation analysis and confirms that our axioms capture the minimal structure necessary for a coherent theory of contradiction as a resource.

---
