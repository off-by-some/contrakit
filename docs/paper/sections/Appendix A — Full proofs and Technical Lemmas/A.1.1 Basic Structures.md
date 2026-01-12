## A.1.1 Basic Structures

### **Definition A.1.1 (Observable System).**

Let $\mathcal{X} = \{X_1, \ldots, X_n\}$ be a finite set of observables. For each $x \in \mathcal{X}$, fix a finite nonempty outcome set $\mathcal{O}_x$. A **context** is a subset $c \subseteq \mathcal{X}$. The outcome alphabet for context $c$ is $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$.

### **Definition A.1.2 (Behavior).**

Given a finite nonempty family $\mathcal{C} \subseteq 2^{\mathcal{X}}$ of contexts, a **behavior** $P$ is a family of probability distributions

$$
P = \{p_c \in \Delta(\mathcal{O}_c) : c \in \mathcal{C}\}
$$

where $\Delta(\mathcal{O}_c)$ denotes the probability simplex over $\mathcal{O}_c$.

**Remark (No nondisturbance required).** We do not assume shared-marginal (nondisturbance) consistency for $P$ across overlapping contexts. The baseline set $\mathrm{FI}$ is always defined as in [A.1.4](#definition-a14-frame-independent-set). When empirical $P$ does satisfy nondisturbance, $K$ coincides with a contextuality monotone for the non-contextual set.

### **Definition A.1.3 (Deterministic Global Assignment).**

Let $\mathcal{O}_{\mathcal{X}} := \prod_{x \in \mathcal{X}} \mathcal{O}_x$. A **deterministic global assignment** is an element $s \in \mathcal{O}_{\mathcal{X}}$. It induces a deterministic behavior $q_s$ by restriction:

$$
q_s(o \mid c) = \begin{cases} 1 & \text{if } o = s|_c \\ 0 & \text{otherwise} \end{cases}
$$

for each context $c \in \mathcal{C}$ and outcome $o \in \mathcal{O}_c$.

### **Definition A.1.4 (Frame-Independent Set).**

The **frame-independent set** is

$$
\text{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\} \subseteq \prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)
$$

### **Proposition A.1.5 (Alternative Characterization of FI).**

$Q \in \text{FI}$ if and only if there exists a **global law** $\mu \in \Delta(\mathcal{O}_{\mathcal{X}})$ such that

$$
q_c(o) = \sum_{s \in \mathcal{O}_{\mathcal{X}} : s|_c = o} \mu(s) \quad \forall c \in \mathcal{C}, o \in \mathcal{O}_c
$$

**Proof.** The forward direction is immediate from the definition of convex hull. For the reverse direction, given $\mu$, define $Q$ by the displayed formula. Then $Q$ is a convex combination of the deterministic behaviors $\{q_s\}$ with weights $\{\mu(s)\}$, hence $Q \in \text{FI}$. □

### A.1.2 Basic Properties of FI

### **Proposition A.1.6 (Topological Properties).**

The frame-independent set FI is nonempty, convex, and compact.

**Proof.**

- **Nonempty**: Contains all deterministic behaviors $q_s$.
- **Convex**: By definition as a convex hull.
- **Compact**: FI is a finite convex hull in the finite-dimensional space $\prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)$, hence a polytope, hence compact. □

### **Definition A.1.7 (Context simplex).**

$$
\Delta(\mathcal{C}) := \{\lambda \in \mathbb{R}^{\mathcal{C}} : \lambda_c \geq 0, \sum_{c \in \mathcal{C}} \lambda_c = 1\}
$$

### **Proposition A.1.8 (Product Structure).**

Let $P$ be a behavior on $(\mathcal{X}, \mathcal{C})$ and $R$ be a behavior on $(\mathcal{Y}, \mathcal{D})$ with $\mathcal{X} \cap \mathcal{Y} = \emptyset$ (we implicitly relabel so disjointness holds). For distributions $p \in \Delta(\mathcal{O}_c)$ and $r \in \Delta(\mathcal{O}_d)$ on disjoint coordinates, $p \otimes r \in \Delta(\mathcal{O}_c \times \mathcal{O}_d)$ is $(p \otimes r)(o_c, o_d) = p(o_c)r(o_d)$.

Define the product behavior $P \otimes R$ on $(\mathcal{X} \sqcup \mathcal{Y}, \mathcal{C} \otimes \mathcal{D})$ where $\mathcal{C} \otimes \mathcal{D} := \{c \cup d : c \in \mathcal{C}, d \in \mathcal{D}\}$ by

$$
(p \otimes r)(o_c, o_d \mid c \cup d) = p(o_c \mid c) \cdot r(o_d \mid d)
$$

Then:

1. If $Q \in \text{FI}_{\mathcal{X},\mathcal{C}}$ and $S \in \text{FI}_{\mathcal{Y},\mathcal{D}}$, then $Q \otimes S \in \text{FI}_{\mathcal{X} \sqcup \mathcal{Y}, \mathcal{C} \otimes \mathcal{D}}$.
2. For deterministic assignments, $q_s \otimes q_t = q_{s \sqcup t}$.

**Proof.**

1. If $Q$ arises from global law $\mu$ and $S$ arises from global law $\nu$, then $Q \otimes S$ arises from the product global law $\mu \otimes \nu$ on $\mathcal{O}_{\mathcal{X} \sqcup \mathcal{Y}}$. From $q_s \otimes q_t = q_{s \sqcup t}$, it follows that

    $$
    (\sum_s \mu_s q_s)\otimes(\sum_t \nu_t q_t)=\sum_{s,t}\mu_s\nu_t\,(q_s\otimes q_t)=\sum_{s,t}\mu_s\nu_t\,q_{s\sqcup t}\in \mathrm{conv}\{q_{s\sqcup t}\}.
    $$

2. Direct verification from definitions: $q_s \otimes q_t = q_{s \sqcup t}$ because $\delta_{s|_c} \otimes \delta_{t|_d} = \delta_{(s \sqcup t)|_{c \cup d}}$. □

### **Definition A.2.1 (Bhattacharyya Coefficient).**

For probability distributions $p, q \in \Delta(\mathcal{O})$ on a finite alphabet $\mathcal{O}$:

$$
\text{BC}(p, q) := \sum_{o \in \mathcal{O}} \sqrt{p(o) q(o)}
$$

### **Lemma A.2.2 (Bhattacharyya Properties).**

For distributions $p, q \in \Delta(\mathcal{O})$:

1. **Range**: $0 \leq \text{BC}(p, q) \leq 1$
2. **Perfect agreement**: $\text{BC}(p, q) = 1 \Leftrightarrow p = q$
3. **Joint concavity**: Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$. Therefore $(x,y)\mapsto\sqrt{xy}$ is jointly concave on $\mathbb{R}_{\geq 0}^2$ (extend by continuity on the boundary). Summing over coordinates preserves concavity, so $\text{BC}$ is jointly concave on $\Delta(\mathcal{O})\times\Delta(\mathcal{O})$.
4. **Product structure**: $\text{BC}(p \otimes r, q \otimes s) = \text{BC}(p, q) \cdot \text{BC}(r, s)$

**Proof.**

1. **Range.**
Nonnegativity is obvious. For the upper bound, by Cauchy-Schwarz:

    $$
    \text{BC}(p, q) = \sum_o \sqrt{p(o) q(o)} \leq \sqrt{\sum_o p(o)} \sqrt{\sum_o q(o)} = 1
    $$

2. **Perfect agreement.**
The Cauchy-Schwarz equality condition gives $\text{BC}(p, q) = 1$ iff $\sqrt{p(o)}$ and $\sqrt{q(o)}$ are proportional, i.e., $\frac{\sqrt{p(o)}}{\sqrt{q(o)}}$ is constant over $\{o : p(o) q(o) > 0\}$. Since both are probability distributions, this constant must be 1, giving $p = q$.
3. **Joint concavity.**
Since $g(t)=\sqrt{t}$ is concave on $\mathbb{R}_{>0}$, its perspective $\phi(x,y)=y\,g(x/y)=\sqrt{xy}$ is concave on $\mathbb{R}_{>0}^2$; extend to $\mathbb{R}_{\geq 0}^2$ by continuity. Summing over coordinates preserves concavity.
4. **Product structure.**
Expand the tensor product and factor the sum. □