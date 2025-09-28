# Simulation Variance Cost

**Proposition 7.2** *(Importance Sampling Penalty)* (App. A.2.2)

To simulate $P$ using a single $Q \in \mathrm{FI}$ with importance weights $w_c = p_c/q_c$:

$$
\inf_{Q \in \mathrm{FI}} \max_{c \in \mathcal{C}} \mathrm{Var}_{Q_c}[w_c] \,\ge\, 2^{2K(P)} - 1
$$

**Proof Strategy:** 

For fixed $c$, $\mathbb{E}_{Q_c}[w_c]=1$ and

$$
\mathbb{E}_{Q_c}[w_c^2] = e^{D_2(p_c \,\|\, q_c)} \,\ge\, e^{D_{1/2}(p_c \,\|\, q_c)} = \mathrm{BC}(p_c,q_c)^{-2}
$$

Thus $\mathrm{Var} \,\ge\, \mathrm{BC}^{-2} - 1$. Taking $\max_c$ and then $\inf_Q$ gives $\alpha^\star(P)^{-2} - 1$ (use $\alpha^\star = \max_Q \min_c \mathrm{BC}(p_c, q_c)$ from App. A.3.2).

---

