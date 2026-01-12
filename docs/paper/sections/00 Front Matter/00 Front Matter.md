# A Mathematical Theory of Contradiction



**Author:**
Cassidy Bridges, Independent Researcher.

**Contact:**
cassidybridges@gmail.com,
https://off-by-some.github.io/web/

## Abstract

We introduce an information-theoretic framework for quantifying *perspectival contradiction*, situations where multiple legitimate observational contexts yield data that no single, frame-independent account can reconcile. Starting from six elementary axioms, we prove that any admissible contradiction measure must take the form $K(P) = -\log_2 \alpha^*(P)$, where $\alpha^*(P) = \max_{Q \in \mathrm{FI}} \min_{c} \mathrm{BC}(p_c,q_c)$.

The first set of theorems establishes the axiomatic core. The second set derives operational laws.

We demonstrate computational tractability through a minimal three-view odd-cycle device yielding $K = 0.5 \cdot \log_2(3/2)$ bits per observation, alongside a convex minimax program and plug-in estimator with bootstrap confidence intervals implemented in our reference software. The framework naturally recovers quantum contextuality as a special case while generalizing to arbitrary domains with context-indexed observations, including continuous probability distributions as demonstrated by the exact computation of contradiction for Gaussian measurement precision conflicts.

In essence, while entropy quantifies the cost of randomness within the single frame, $K$ measures the fundamental price of incompatibility across multiple frames of reference.



### Author's Note

This is an early preprint from an independent project. Some results are complete, others are sketched and will develop as the work progresses. I don't present myself as an expert in mathematics or physics; my goal is to share ideas that are both natural and useful, and to invite critique from people with more formal background.

A reference implementation and reproducibility scripts are included. The library will be made public shortly after this preprint is available. Feedback on assumptions, counterexamples, and connections to prior work is warmly welcome. Please cite as a work in progress. Any errors are my own.