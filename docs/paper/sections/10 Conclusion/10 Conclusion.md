# 10. Conclusion

The fundamental theorem of information theory establishes that entropy $H$ measures the irreducible cost of encoding uncertainty within a coherent probabilistic framework. We have shown that when multiple legitimate frameworks refuse to agree on the interpretation of the same observations, there exists a complementary quantity that measures the irreducible cost of enforcing artificial consensus.

This quantity, which we call the **contradiction** of a behavior $P$, is uniquely determined by six natural axioms to be
$$
K(P) = -\log_2 \alpha^\star(P), \quad \text{where} \quad \alpha^\star(P) = \max_{Q \in \mathrm{FI}} \min_c \mathrm{BC}(p_c, q_c).
$$
Here $\mathrm{FI}$ represents the convex set of frame-independent behaviors, those admitting a unified description, and $\mathrm{BC}$ is the Bhattacharyya affinity between probability distributions.

We show the mathematical structure mirrors that of entropy theory in several essential respects: entropy by additivity, continuity, and the grouping property; contradiction by our axioms A0–A5. Just as entropy has operational meaning through coding theorems, contradiction manifests operationally in three fundamental ways: it governs the error exponents for distinguishing frame-dependent from frame-independent behaviors, it determines the witness overhead required to simulate multi-context data with a single model, and it bounds the irreducible regret when prediction is restricted to unified models.

This is the core insight.

Most significantly, we establish that contradiction obeys a law of additivity—for independent behaviors $P$ and $R$, we have $K(P \otimes R) = K(P) + K(R)$. And this additivity, combined with the natural units of bits per observation, establishes contradiction as a legitimate information measure complementary to entropy.

And the operational interpretation is immediate. In the fundamental tasks analyzed here—asymptotic equipartition, lossless compression, communication with common decoding, and rate-distortion under separated encoding—any attempt to impose a single coherent story across incompatible contexts incurs an exact surcharge of $K(P)$ bits per symbol. When $K(P) = 0$, no surcharge applies and classical Shannon limits are achievable. When $K(P) > 0$, the cost is unavoidable.

The theory thus extends Shannon's framework by providing a second fundamental measure of information complexity, where entropy $H$ prices the cost of uncertainty within a probabilistic model, contradiction $K$ prices the cost of reconciling incompatible models. But together, they provide a complete two-dimensional characterization of informational resources within the scope of this framework: randomness and irreconcilability.

The mathematical development reveals an elegant geometric structure. We show that $\alpha^\star(P) = 1 - \min_{Q \in \mathrm{FI}} \max_c H^2(p_c, q_c)$, establishing that contradiction $K = -\log_2 \alpha^\star$ measures proximity to the frame-independent set in Hellinger geometry, with level sets forming spheres in the space of square-root probability vectors. And this geometry ensures that products of behaviors correspond to addition of contradictions—the essential property that makes bits the natural unit.

We emphasize that these results are not approximations or bounds, but exact equalities, holding under the stated assumptions. The minimax program defining $\alpha^\star(P)$ attains an optimum; the value is unique and computable by standard convex optimization methods.

This precision matters. It guides our work. We build upon it. We extend it further. We refine it continuously. We improve it steadily. For the canonical example of the lenticular coin—a minimal device exhibiting contextual behavior—we obtain the precise value $K = \frac{1}{2}\log_2(3/2) \approx 0.2925$ bits per observation.

Yet the scope of the theory extends naturally beyond its quantum origins. While contradiction recovers contextuality as a special case when $\mathrm{FI}$ is taken to be the set of non-contextual behaviors, the framework applies unchanged to any domain that can specify legitimate contexts and a baseline of unified behaviors, including continuous probability distributions as demonstrated by the exact quantification of contradiction in Gaussian measurement precision conflicts.

This generality holds. This universality suggests that contradiction may prove as fundamental to information theory as entropy itself.

**The central message is simple**: when one story cannot fit the data without distortion, the excess cost is precisely quantifiable. Contradiction $K(P)$ measures this cost in the same units and with the same mathematical precision that entropy measures uncertainty. This completes our understanding of information by accounting not only for what we don't know, but for what cannot be consistently known across incompatible but equally valid perspectives.

In information theory, as in thermodynamics, conservation laws govern the possible. Just as energy cannot be created or destroyed but only transformed, information cannot be reconciled without cost when genuine incompatibilities exist. Contradiction $K(P)$ quantifies this cost exactly, providing the missing piece in our accounting of informational resources.