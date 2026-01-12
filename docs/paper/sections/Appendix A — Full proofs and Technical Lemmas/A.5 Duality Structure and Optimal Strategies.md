## A.5 Duality Structure and Optimal Strategies

### **Theorem A.5.1 (Minimax Duality).**

Let $(\lambda^\star, Q^\star)$ be optimal strategies for the minimax problem in Theorem [A.3.2](#theorem-a32-minimax-equality). Then:

1. $f(\lambda^\star, Q^\star) = \alpha^\star(P)$
2. $\text{supp}(\lambda^\star) \subseteq \{c \in \mathcal{C} : \text{BC}(p_c, q_c^\star) = \alpha^\star(P)\}$
3. If $\lambda^\star_c > 0$, then $\text{BC}(p_c, q_c^\star) = \alpha^\star(P)$

**Proof.** Existence of optimal strategies follows from compactness and continuity.

1. This is immediate from the minimax equality.
2. & 3. For fixed $Q^\star$, the inner optimization $\min_{\lambda \in \Delta(\mathcal{C})} \sum_c \lambda_c a_c$ with $a_c := \text{BC}(p_c, q_c^\star)$ has value $\min_c a_c$ and optimal solutions supported on $\arg\min_c a_c$. Since $(\lambda^\star, Q^\star)$ is optimal for the full problem, $\lambda^\star$ must be optimal for this inner problem, giving the result. □