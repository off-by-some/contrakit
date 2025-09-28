**Proposition 7.1** *(Testing Real vs Frame-Independent)* (App. A.3.2, A.9)

For testing $\mathcal{H}_0: Q \in \mathrm{FI}$ vs $\mathcal{H}_1: P$ with contexts drawn i.i.d. from $\lambda \in \Delta(\mathcal{C})$:

Define $\alpha_\lambda(P) := \max_{Q \in \mathrm{FI}} \sum_c \lambda_c \mathrm{BC}(p(\cdot|c), q(\cdot|c))$ and $E_{\mathrm{BH}}(\lambda) := -\log_2 \alpha_\lambda(P)$.

Then $E_{\text{opt}}(\lambda) \,\ge\, E_{\mathrm{BH}}(\lambda)$, and in the least-favorable mixture:

$$
\inf_\lambda E_{\text{opt}}(\lambda) \,\ge\, \min_\lambda E_{\mathrm{BH}}(\lambda) = K(P)
$$

If the Chernoff optimizer is balanced ($s = 1/2$) under $\lambda^\star$, then $E_{\text{opt}}(\lambda^\star) = K(P)$.

**Proof Strategy:** 

Chernoff bound for composite $\mathcal{H}_0$ yields $E_{\mathrm{BH}}(\lambda)$; minimizing over $\lambda$ gives $K(P)$. Equality at $s=1/2$ is the standard balanced case.

---

### 