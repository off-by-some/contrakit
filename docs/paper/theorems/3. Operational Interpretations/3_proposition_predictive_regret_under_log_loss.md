### Predictive Regret Under Log-Loss

**Proposition 7.3** *(Single-Predictor Penalty)* (App. A.2.2)

Using one predictor $Q \in \mathrm{FI}$ across all contexts under log-loss. Put differently: the single-model prediction penalty.

$$
\inf_{Q \in \mathrm{FI}} \max_{c \in \mathcal{C}} \mathbb{E}_{p_c}\left[\log_2 \frac{p_c(X)}{q_c(X)}\right] \,\ge\, 2K(P) \text{ bits/round}
$$

---