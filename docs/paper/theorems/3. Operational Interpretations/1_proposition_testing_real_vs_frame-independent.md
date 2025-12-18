**Proposition 7.1** Testing Real vs Frame-Independent

It's fair to ask whether we can distinguish a real behavior from its frame-independent surrogates. We establish this result formally.

We test $\mathcal{H}_0$ that $Q \in \mathrm{FI}$ versus $\mathcal{H}_1$ that $P$ holds true; contexts are drawn i.i.d. from a distribution $\lambda \in \Delta(\mathcal{C})$. The procedure follows.

We define an agreement coefficient $\alpha_\lambda(P) := \max_{Q \in \mathrm{FI}} \sum_c \lambda_c \mathrm{BC}(p(\cdot|c), q(\cdot|c))$ and an exponent $E_{\mathrm{BH}}(\lambda) := -\log_2 \alpha_\lambda(P)$.

Then $E_{\text{opt}}(\lambda) \,\ge\, E_{\mathrm{BH}}(\lambda)$.

The least-favorable mixture gives us a bound:

$$
\inf_\lambda E_{\text{opt}}(\lambda) \,\ge\, \min_\lambda E_{\mathrm{BH}}(\lambda) = K(P)
$$

Equality holds when the Chernoff optimizer is balanced—at $s = 1/2$ under $\lambda^\star$.

We finally arrive at $E_{\text{opt}}(\lambda^\star) = K(P)$.

We only need to recognize that the contradiction measure emerges naturally from the testing framework.

**Proof Strategy**

The Chernoff bound for composite $\mathcal{H}_0$ yields $E_{\mathrm{BH}}(\lambda)$.

Minimizing over $\lambda$ gives $K(P)$.

Equality at $s=1/2$ is the standard balanced case.

---

### 