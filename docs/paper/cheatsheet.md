# A Mathematical Theory of Contradiction

Consider measurement scenarios where different observers see fundamentally incompatible outcomes. Imagine a classroom with three students—Alice, Bob, and Carol—and three seats arranged in a straight line. The teacher gives each student a pairwise constraint:

* Alice must sit next to Bob
* Bob must sit next to Carol
* Carol must sit next to Alice

![Three students, three seats, three constraints - no arrangement satisfies all rules simultaneously](/figures/three-seats.png)

Each **pairwise context is internally consistent**: if you consider only a given pair (Alice next to Bob, Bob next to Carol, or Carol next to Alice), that local rule can be satisfied. **However, no global seating assignment satisfies all three constraints at once.** At most two constraints can hold; the third is inevitably violated.

This illustrates **irreducible contradiction**: every local context is valid on its own, but no global arrangement reconciles all contexts. Each student can see their pairwise rule satisfied in some assignment, but any global assignment will necessarily leave one constraint broken. The contradiction measure formalizes exactly this classical incompatibility:

$$K(P) = -\log_2 \alpha^\star(P), \quad \text{where} \quad \alpha^\star(P) = \max_{Q \in \mathrm{FI}} \min_c \mathrm{BC}(p_c, q_c)$$

Here, $\alpha^\star$ finds the best possible unified explanation and measures its worst-case agreement with your observations. The worst-matching context determines the overall agreement level, and $K(P)$ converts this into bits—the irreducible "penalty" you pay when trying to reconcile all local contexts into a single global assignment.

For this simple classroom paradox, the best possible global assignment still violates one constraint, which gives a minimal contradiction of

$$K = \frac{1}{2}\log_2\frac{3}{2} \approx 0.29 \text{ bits per observation}$$

This shows that even in the simplest odd-cycle scenario, some irreducible "penalty" must be paid.

## Quick Start

### Essential Properties

$$
K(P) = 0 \quad \text{precisely when} \quad P \in \mathrm{FI}
$$

No contradiction means all perspectives unify into one underlying reality.

$$
K(P \otimes R) = K(P) + K(R)
$$

Contradiction adds up for independent systems. Every operational task costs $K(P)$ extra bits per observation when data contradicts itself.

### Theorems Worth Mentioning

Consider the theory's core results. Six axioms uniquely determine contradiction as the logarithmic transform of minimax Bhattacharyya agreement. That measure captures irreducible perspectival incompatibility.

**Foundational theorems** establish the mathematical core:
- **Theorem 1**: Weakest link aggregators equal the minimum (see Representation Theory)
- **Theorem 2**: Contradiction admits adversarial minimax form (see Representation Theory)
- **Theorem 3**: Bhattacharyya coefficient uniquely satisfies natural properties (see Representation Theory)
- **Theorem 4**: $K(P) = -\log_2 \alpha^\star(P)$ (see Representation Theory)
- **Theorem 5**: Independent contradictions add: $K(P \otimes R) = K(P) + K(R)$ (see Representation Theory)

**Operational theorems** quantify practical costs:
- **Theorem 6**: Typical sets grow by $K(P)$ bits per observation (see Operational Consequences)
- **Theorem 7-8**: Compression rates increase by $K(P)$ (see Operational Consequences)
- **Theorem 9**: Testing frame-independence requires $K(P)$ evidence (see Operational Consequences)
- **Theorem 10**: Witness rate $K(P)$ enables simulation (see Operational Consequences)
- **Theorem 11-12**: Common communication costs $K(P)$ extra bits (see Operational Consequences)
- **Theorem 13**: Channel capacity drops by $K(P)$ (see Operational Consequences)
- **Theorem 14**: Rate-distortion increases by $K(P)$ (see Operational Consequences)
- **Theorem 15**: Geometry explains additive structure (see Geometric Structure)

**Interpretive results** connect to concrete penalties:
- **Theorem 7.4**: Witness-error tradeoff: $E + r \geq K(P)$ (see Testing, Prediction)
- **Theorem 7.5**: Universal adversarial structure (see Testing, Prediction)
- **Theorem 7.9**: Equalizer principle with sparse optimizers (see Geometric Properties)

**Key propositions** bound operational quantities:
- **Proposition 7.1**: Testing bounds: $\inf_\lambda E_{\text{opt}}(\lambda) \geq K(P)$ (see Testing, Prediction)
- **Proposition 7.2**: Simulation variance $\geq 2^{2K(P)} - 1$ (see Testing, Prediction)
- **Proposition 7.3**: Predictive regret $\geq 2K(P)$ bits/round (see Testing, Prediction)
- **Proposition 7.6**: Hellinger sphere structure (see Geometric Properties)
- **Proposition 7.7**: Smoothing bounds on mixing (see Geometric Properties)
- **Proposition 7.8**: Convex program for computation (see Computational Methods)

## Core Measurement Framework

Just like the three-student paradox, each context sees a local rule that makes sense, but we now formalize all such scenarios mathematically.

We model systems where multiple observers examine the same underlying reality through different lenses. Each observer sees a context: a specific subset of measurements. A behavior $P$ records what every possible context would observe. This models situations where the same reality appears differently depending on your observational setup.

The frame-independent baseline represents the easy case. $\mathrm{FI}$ consists of behaviors admitting unified explanations:

$$
\mathrm{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\}
$$

These $q_s$ are deterministic behaviors in the discrete case; they extend to joint density representations for continuous variables. $\mathrm{FI}$ behaviors have no fundamental incompatibility—different viewpoints can be reconciled through one underlying reality.

We quantify agreement between distributions using the Bhattacharyya coefficient. That coefficient measures distributional overlap (The Bhattacharyya coefficient $\mathrm{BC}(p,q)$ measures how similar two distributions are, with 1 meaning identical):

$$
\mathrm{BC}(p,q) = \sum_o \sqrt{p(o) q(o)} \quad (\text{discrete}) \quad \text{or} \quad \int \sqrt{p(x) q(x)} \, dx \quad (\text{continuous})
$$

$\mathrm{BC}$ ranges from 0 (no overlap) to 1 (identical distributions). Perfect agreement occurs when $\mathrm{BC}$ equals 1; that happens precisely when distributions are identical. The coefficient is jointly concave and multiplicative:

$$
\mathrm{BC}(p \otimes r, q \otimes s) = \mathrm{BC}(p,q) \cdot \mathrm{BC}(r,s)
$$

We use this to assess how well different contexts align with each other.

## Contradiction as Adversarial Agreement

Contradiction emerges when no unified explanation adequately fits all observational contexts. The core definition captures this adversarial relationship:

$$
\alpha^\star(P) = \max_{Q \in \mathrm{FI}} \min_{c \in \mathcal{C}} \mathrm{BC}(p_c, q_c), \quad K(P) = -\log_2 \alpha^\star(P)
$$

$\alpha^\star$ finds the best unified explanation and measures its worst-case agreement with your actual data. $K(P)$ transforms this into bits—the logarithmic cost of incompatibility.

This calculation admits an equivalent minimax form:

$$
\alpha^\star(P) = \min_{\lambda \in \Delta(\mathcal{C})} \max_{Q \in \mathrm{FI}} \sum_c \lambda_c \cdot \mathrm{BC}(p_c, q_c)
$$

That formulation reveals the adversarial structure. $\lambda$ represents context weights; nature chooses weights to maximize the apparent contradiction. The weak link—contexts causing the most disagreement—determines the overall measure.

Fundamental bounds constrain the possible values:

$$
K(P) = 0 \iff P \in \mathrm{FI} \iff \alpha^\star(P) = 1, \quad 0 \leq K(P) \leq \frac{1}{2} \log_2 (\max_c |\mathcal{O}_c|)
$$

Zero contradiction occurs only when behaviors lie in $\mathrm{FI}$. The upper bound depends on outcome space sizes—more possible observations allow greater incompatibility.

## Axiomatic Foundations

Six fundamental properties uniquely determine the contradiction measure:

**A0: Label Invariance** - Contradiction measures structural incompatibility, not notational artifacts. Relabeling outcomes or contexts preserves the contradiction level—the pattern of disagreement matters, not what you call the labels.

**A1: Reduction** - Zero contradiction precisely when behaviors are frame-independent. No contradiction means all perspectives unify into one underlying reality—this gives the natural zero point.

**A2: Continuity** - Small probability changes yield small contradiction changes. Tiny tweaks to distributions shouldn't cause huge jumps in measured incompatibility.

**A3: Free Operations** - Monotone under legitimate transformations. Adding noise, averaging perspectives, or combining systems cannot create contradiction where none existed.

**A4: Grouping** - Depends only on refined statistics. How you group observations doesn't change fundamental incompatibility levels—contradiction sees through statistical aggregations.

**A5: Independent Composition** - Additive for disjoint systems:

$$
K(P \otimes R) = K(P) + K(R)
$$

Free operations include:
- Stochastic post-processing within contexts
- Convex mixtures: $K((1-t)P + tQ) \leq \max(K(P), K(Q))$
- Public lotteries over contexts
- Tensoring with FI ancillas: $K(P \otimes R) \leq K(P)$ for $R \in \mathrm{FI}$

## Representation Theory and Uniqueness

The weakest link principle governs reasonable aggregators. Any unanimity-respecting, monotone aggregator with weakest-link properties equals the minimum:

$$
\mathrm{A}(x) = \min_i x_i \quad \text{for all} \quad x \in [0,1]^{\mathcal{C}}
$$

Among fair combination methods, taking the worst-case opinion is uniquely justified. That principle explains why contradiction focuses on the least agreeable context.

Contradiction manifests as a minimax game. Any contradiction measure satisfying the core axioms admits this representation:

$$
K(P) = h\left(\max_Q \min_c F(p_c, q_c)\right) = h\left(\min_\lambda \max_Q \sum_c \lambda_c F(p_c, q_c)\right)
$$

for some strictly decreasing continuous $h$ and agreement kernel $F$. Contradiction is fundamentally adversarial: one player seeks the best unified explanation, the other finds the worst disagreement.

Under refinement separability, product multiplicativity, DPI, joint concavity, and regularity, the agreement kernel is unique:

$$
F(p,q) = \sum_o \sqrt{p(o) q(o)} = \mathrm{BC}(p,q)
$$

The Bhattacharyya coefficient is uniquely determined by natural mathematical properties. No other agreement measure satisfies these fundamental requirements.

The fundamental formula emerges from the complete axiom set:

$$
K(P) = -\log_2 \alpha^\star(P)
$$

Contradiction must be logarithmic in agreement levels. That logarithmic form makes contradiction additive—like information—and gives it the correct units.

For independent systems on disjoint observables with $\mathrm{FI}$ product-closed:

$$
\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R), \quad K(P \otimes R) = K(P) + K(R)
$$

Contradictions from separate systems multiply for agreement but add for bits. That additivity explains why contradiction behaves like information across independent components.

## Operational Consequences in Information Theory

Contradiction imposes fundamental limits on information processing. The asymptotic equipartition property includes a contradiction tax:

$$
\lim_{n\to\infty} \frac{1}{n} \log_2 |\mathcal{S}_n| \geq H(X|C) + K(P)
$$

With many contradictory observations, typical pattern counts grow exponentially at rate $H(X|C) + K(P)$. The $K(P)$ term represents the extra complexity you pay because contexts disagree.

Compression rates increase accordingly. With known contexts, you achieve:

$$
\lim_{n\to\infty} \frac{1}{n} \mathbb{E}[\ell_n^\star] = H(X|C) + K(P)
$$

You need $K(P)$ extra bits per observation to handle context disagreements.

When contexts are latent, compression costs more:

$$
\lim_{n\to\infty} \frac{1}{n} \mathbb{E}[\ell_n^\star] = H(X) + K(P)
$$

Unknown contexts require the full entropy $H(X)$ plus the contradiction penalty.

Witnesses enable different compression regimes. Including witness information at rate $K(P)$, meta-typical sets satisfy:

$$
\frac{1}{n}\log_2|\mathcal T_\varepsilon^n| = \begin{cases}
H(X)+K(P) & \text{(latent contexts)} \\
H(X|C)+K(P) & \text{(known contexts)} \\
H(C)+H(X|C)+K(P) & \text{(contexts in header)}
\end{cases}
$$

Witnesses essentially explain the contradiction, allowing compression as if contexts were unified.

Hypothesis testing reveals another limitation. When testing frame-independence versus contradiction, the optimal type-II error exponent satisfies $E \geq K(P)$. Contradiction fundamentally bounds how sharply you can distinguish unified from contradictory explanations.

Witnessing enables approximation. Rate $K(P)+o(1)$ achieves vanishing total variation: $\mathrm{TV}((X^n, W_n), \tilde{Q}_n) \to 0$. You can simulate contradictory data using a unified model plus $K(P)$ bits of witness information per observation.

Communication faces similar constraints:

- Common message problem: rate $\geq H(X|C) + K(P)$
- Common representation: $\geq H(X|C) + K(P)$ (known contexts) or $H(X) + K(P)$ (latent contexts)

Channel capacity drops when all receivers must decode identically:

$$
C_{\text{common}} = C_{\text{Shannon}} - K(P)
$$

Different interpretations of received signals reduce communication efficiency by $K(P)$ bits.

Rate-distortion with common reconstruction costs extra:

$$
R(D) = R_{\text{Shannon}}(D) + K(P)
$$

You cannot losslessly compress contradictory data to a single representation without paying the contradiction tax.

## Geometric Structure

Contradiction induces a specific geometric structure. The Hellinger metric measures distances:

$$
J(A,B) = \max_c \arccos(\mathrm{BC}(p_c^A, p_c^B))
$$

That metric is subadditive: $J(P \otimes R) \leq J(P) + J(R)$. Yet contradiction itself is log-additive: $K(P \otimes R) = K(P) + K(R)$.

## Testing, Prediction, and Universal Structure

Testing real versus frame-independent data has fundamental limits:

$$
\inf_\lambda E_{\text{opt}}(\lambda) \geq \min_\lambda E_{\mathrm{BH}}(\lambda) = K(P)
$$

The best discrimination performance is bounded by $K(P)$. Contradictory data always introduces at least that much uncertainty.

The witness-error tradeoff quantifies the relationship. For witness rate $r$, type-II exponent $E$ satisfies:

$$
E + r \geq K(P), \quad E^*(r) = K(P) - r \quad \text{for} \quad r \in [0, K(P)]
$$

Importance sampling reveals prediction penalties:

$$
\inf_{Q \in \mathrm{FI}} \max_c \mathrm{Var}_{Q_c}[w_c] \geq 2^{2K(P)} - 1
$$

Single-predictor regret bounds show:

$$
\inf_{Q \in \mathrm{FI}} \max_c \mathbb{E}_{p_c}[\log_2(p_c(X)/q_c(X))] \geq 2K(P) \text{ bits/round}
$$

The universal adversarial structure unifies these results. Optimal $\lambda^\star$ works simultaneously for testing, simulation, and coding.

## Geometric Properties and Smoothing

The Hellinger sphere structure connects contradiction to geometry:

$$
\alpha^\star(P) = 1 - D_H^2(P, \mathrm{FI}) \quad \text{where} \quad D_H^2(P, \mathrm{FI}) = \min_Q \max_c H^2(p_c, q_c)
$$

Level sets $\{P: K(P) = \kappa\}$ form outer Hellinger Chebyshev spheres of radius $\sqrt{1 - 2^{-\kappa}}$ around $\mathrm{FI}$.

The total variation gap shows separation:

$$
d_{\mathrm{TV}}(P, \mathrm{FI}) \geq 1 - 2^{-K(P)}
$$

Smoothing bounds enable interpolation:

$$
K((1-t)P + tR) \leq -\log_2((1-t)2^{-K(P)} + t) \leq (1-t)K(P)
$$

To reduce contradiction to $\leq \kappa$, you need:

$$
t \geq \frac{1 - 2^{-\kappa}}{1 - 2^{-K(P)}}
$$

High stability means reducing significant contradiction requires nearly complete FI mixture.

## Computational Methods

Convex programs enable computation:

$$
\min_\mu \max_c H^2(p_c, q_c(\mu)) \quad \text{where} \quad q_c(\mu) = \sum_s \mu(s) \cdot \delta_{s|_c}
$$

That optimization finds the best unified explanation minimizing worst-case disagreement. Modern solvers handle this efficiently.

The minimax formulation is equivalent:

$$
\max_Q \min_c \mathrm{BC}(p_c, q_c) \quad \text{solved via Sion's theorem duality}
$$

Sion's theorem guarantees this equals the original max-min.

Statistical estimation uses plug-in methods with bootstrap confidence intervals. Regularized estimation adds pseudocounts for small datasets:

$$
\tilde{p}_o = \frac{n_o + \epsilon}{n + k\epsilon} \quad \text{where } k \text{ is number of outcomes}
$$

Small pseudocounts like 0.01 prevent zero probabilities and ensure statistical consistency.

## Worked Examples and Technical Bounds

The uniform law provides a lower bound:

$$
\alpha^\star(P) \geq \min_c \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

Minimax duality reveals the tension structure: $\lambda^\star$ locates the pressure points, $Q^\star$ maximizes agreement, active contexts saturate the bound. The equalizer principle ensures sparse optimizers:

$$
\mathrm{BC}(p_c, q_c^\star) = \alpha^\star(P) \quad \text{for active contexts}
$$

with optimal $Q^\star$ supported on at most $1 + \sum_c (|\mathcal{O}_c| - 1)$ deterministic assignments.

Odd cycles create minimal contradiction:

$$
\alpha^\star(P) \leq \sqrt{2/3}, \quad \mu^\star = \text{Unif}(\{100,010,001,011,101,110\})
$$

Pairwise anti-correlations imply $K(P) > 0$.

The classroom seating paradox demonstrates minimal contradiction. Three students with mutually incompatible seating constraints yield:

$$
\alpha^\star = \sqrt{2/3}, \quad K = \frac{1}{2} \log_2(3/2) \approx 0.29 \text{ bits per observation}
$$

That system represents the simplest contradictory device—no matter how you assign seats globally, you pay 0.29 bits per observation overhead.

Gaussian measurements extend this to continuous variables. For $\mathcal{N}(0,\sigma_1^2)$ versus $\mathcal{N}(0,\sigma_2^2)$ conflicts:

$$
\alpha^\star(P) = \sqrt{\frac{2 \sqrt{\sigma_1 \sigma_2}}{\sigma_1 + \sigma_2}}, \quad K(P) = -\log_2 \sqrt{\frac{2 \sqrt{\sigma_1 \sigma_2}}{\sigma_1 + \sigma_2}}
$$

With $\sigma_1=1, \sigma_2=4$, you get $\alpha^\star \approx 0.894, K \approx 0.161$ bits. Contradiction depends on precision differences—more similar measurements mean less incompatibility.

## Key Identities and Properties

The product law governs composition:

$$
\alpha^\star(P \otimes R) = \alpha^\star(P) \cdot \alpha^\star(R), \quad K(P \otimes R) = K(P) + K(R)
$$

Hellinger geometry connects everything:

$$
H(p,q) = \sqrt{1 - \mathrm{BC}(p,q)}, \quad D_H^2(P, \mathrm{FI}) = \min_Q \max_c H^2(p_c, q_c), \quad \alpha^\star(P) = 1 - D_H^2(P, \mathrm{FI})
$$

Information bounds quantify the gaps:

$$
d_{\mathrm{TV}}(P, \mathrm{FI}) \geq 1 - 2^{-K(P)}, \quad E + r \geq K(P)
$$

Smoothing properties show the interpolation behavior:

$$
K((1-t)P + tR) \leq -\log_2((1-t)2^{-K(P)} + t), \quad t \leq \frac{2^{-\kappa} - 2^{-K(P)}}{1 - 2^{-K(P)}}
$$

Stability is high—reducing significant contradiction to near-zero requires nearly complete $\mathrm{FI}$ mixture.