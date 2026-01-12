## A.4 Bounds and Characterizations

### **Lemma A.4.1 (Uniform Law Lower Bound).**

For any behavior $P$:

$$
\alpha^\star(P) \geq \min_{c \in \mathcal{C}} \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

**Proof.**

Let $\mu$ be the uniform (counting-measure) distribution on $\mathcal{O}_{\mathcal{X}}$ (so each global state is equally likely). This induces $Q^{\text{unif}} \in \text{FI}$ with uniform **context marginals**: $q_c^{\text{unif}}(o) = \frac{1}{|\mathcal{O}_c|}$ for all $c \in \mathcal{C}$, $o \in \mathcal{O}_c$.

For any context $c$:

$$
\text{BC}(p_c, q_c^{\text{unif}}) = \sum_{o \in \mathcal{O}_c} \sqrt{p_c(o) \cdot \frac{1}{|\mathcal{O}_c|}} = \frac{1}{\sqrt{|\mathcal{O}_c|}} \sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)}
$$

The function $\sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)}$ is concave on the simplex $\Delta(\mathcal{O}_c)$, so its minimum is attained at a vertex (a point mass), where the sum equals 1. Therefore:

$$
\sum_{o \in \mathcal{O}_c} \sqrt{p_c(o)} \geq 1
$$

This minimum is achieved when $p_c$ is a point mass. Therefore:

$$
\text{BC}(p_c, q_c^{\text{unif}}) \geq \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

Since $\alpha^\star(P) \geq \min_{c} \text{BC}(p_c, q_c^{\text{unif}})$, the result follows. □

### **Corollary A.4.2 (Bounds on K).**

For any behavior $P$:

$$
0 \leq K(P) \leq \frac{1}{2} \log_2 \left(\max_{c \in \mathcal{C}} |\mathcal{O}_c|\right)
$$

**Proof.**

The lower bound follows from $\alpha^\star(P) \leq 1$. The upper bound follows from Lemma [A.4.1](#lemma-a41-uniform-law-lower-bound) and the fact that $-\log_2(x^{-1/2}) = \frac{1}{2}\log_2(x)$. □

### **Theorem A.4.3 (Characterization of Frame-Independence).**

For any behavior $P$:

$$
\alpha^\star(P) = 1 \Leftrightarrow P \in \text{FI} \Leftrightarrow K(P) = 0
$$

**Proof.**

($\Rightarrow$) If $\alpha^\star(P) = 1$, then there exists $Q \in \text{FI}$ such that $\min_c \text{BC}(p_c, q_c) = 1$. This implies $\text{BC}(p_c, q_c) = 1$ for all $c \in \mathcal{C}$. By Lemma [A.2.2](#lemma-a22-bhattacharyya-properties), this gives $p_c = q_c$ for all $c$, hence $P = Q \in \text{FI}$.

($\Leftarrow$) If $P \in \text{FI}$, take $Q = P$ in the definition of $\alpha^\star(P)$. Then $\min_c \text{BC}(p_c, q_c) = \min_c \text{BC}(p_c, p_c) = 1$.

The equivalence with $K(P) = 0$ follows from the definition $K(P) = -\log_2 \alpha^\star(P)$. □

**Remark (No nondisturbance required).**

We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in [A.1.4](#definition-a14-frame-independent-set). When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.