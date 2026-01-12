## B.4 Consensus as Decoding: $K(P)$ Tax

This section shows that when distributed systems must reach consensus despite frame-dependent disagreements, any protocol forcing a single committed decision pays an unavoidable overhead of exactly $K(P)$ bits per decision beyond the Shannon baseline. This is a direct consequence of the operational theorems in §6.

These are information-rate lower bounds, not round-complexity bounds. They coexist with FLP/Byzantine results: even with perfect channels and replicas, if local perspectives are structurally incompatible**,** a $K(P)$-bit tax must be paid to force consensus.

---

### B.4.1 Setup

Consider a distributed consensus protocol where replicas must agree on proposals. Each replica evaluates proposals using its own local validity predicate—rules derived from the replica's particular message history and context. The question is: what's the fundamental cost of forcing agreement when these local contexts lead to systematically different judgments?

This isn't about noisy communication or Byzantine faults. Even with perfect replicas and reliable channels, structural disagreements emerge when local contexts are legitimately incompatible.

#### Assumptions

Standing assumptions from §2 apply (finite alphabets; $\mathrm{FI}$ convex/compact; product-closure when used).

- **Finite outcomes:** All observables have finite alphabets
- **Frame-independent baseline:** The set FI of frame-independent behaviors is nonempty, convex, and compact
- **Asymptotic regime:** Either i.i.d. decisions or stationary/ergodic sequences where asymptotic equipartition applies
- **Local context availability:** Decoders may use their own local context $C$ during decoding
- **FI product-closure:** $\mathrm{FI}$ is closed under products on disjoint observables.

---

### B.4.2 Mathematical Model

#### Contexts and Behaviors

- **Local validity:** For proposal π, replica i evaluates predicate $V_i(\pi)$ based on its message history, yielding $X_i^{\pi} \in \{YES, NO\}$
- **Contexts:** A context c represents a subset of replicas observed under particular scheduling conditions (e.g., pair {i,j} with their joint reports)
- **System behavior:** The system induces $P = \{p_c\}_{c\in\mathcal{C}}$, a family of outcome distributions, one per context
- **Contradiction measure:** With Bhattacharyya overlap $\mathrm{BC}(p,q)=\sum_o \sqrt{p(o)q(o)}$, we define:

$$
\alpha^\star(P)=\max_{Q\in\mathrm{FI}}\min_{c\in\mathcal{C}}\mathrm{BC}(p_c,q_c),\qquad
K(P)=-\log_2 \alpha^\star(P)
$$

By our axioms, $K(P)$ uniquely measures contradiction: $K(P) = 0$ if and only if all contexts can be reconciled by a single frame-independent account.

---

### B.4.3 Why Odd Cycles Create Contradiction

Suppose an adversary schedules messages so three correct replicas A, B, C produce pairwise disagreements in pairwise contexts:

- Context {A,B}: $p_{AB}(YN)=p_{AB}(NY)=\tfrac12$, others 0
- Context {B,C}: $p_{BC}(YN)=p_{BC}(NY)=\tfrac12$, others 0
- Context {C,A}: $p_{CA}(YN)=p_{CA}(NY)=\tfrac12$, others 0

This creates an "odd cycle" of constraints: $X_A \neq X_B$, $X_B \neq X_C$, $X_C \neq X_A$. Precisely what we saw within our Lenticular Coin. No global assignment can satisfy all three simultaneously.

**Lemma B.4.1:**

If each pairwise context (A,B), (B,C), (C,A) assigns **zero** probability to equal outcomes (i.e., supports only $X_A \neq X_B$, $X_B \neq X_C$, $X_C \neq X_A$), then $P\notin \mathrm{FI}$ and $K(P)>0$.

For this symmetric anti-correlation pattern, $K(P) = \tfrac12\log_2(3/2) \approx 0.29$ bits per decision.

**Proof sketch:**

The optimal frame-independent approximation assigns outcomes uniformly over the six "not-all-equal" patterns of $(X_A, X_B, X_C)$. Each pairwise marginal then has Bhattacharyya overlap $\sqrt{2/3}$ with the observed anti-correlation, giving $\alpha^\star(P) = \sqrt{2/3}$ and $K(P) = \tfrac12\log_2(3/2)$.

---

### B.4.4 Consensus as Common Decoding

A consensus decision is fundamentally a **common representation problem**: we need a finite string Z that every correct replica, using its own local context, can decode to recover the same decision sequence $X^n$.

This is precisely the setup analyzed in our operational theorems. Different replicas effectively use different "codebooks" based on their local contexts, yet they must all decode to the same outcome.

---

### B.4.5 The Fundamental Lower Bound

**Theorem B.4.2 (The K(P) Tax):**

Any consensus scheme outputting a single representation $Z$ decodable by every context requires communication rate: (Theorems 11–12)

$$
\tfrac1n\mathbb{E}[\ell(Z)] \ge
\begin{cases}
H(X\mid C)+K(P)-o(1), & \text{decoders know } C^n\\
H(X)+K(P)-o(1), & \text{decoders don't}
\end{cases}
$$

where $H(X|C)$ is the standard entropy given context information, and $o(1) \to 0$ as block length $n \to \infty$.

**What this means:**

Beyond the usual Shannon entropy cost $H(X|C)$, there's an additional $K(P)$ bits per decision needed to force consensus when contexts naturally disagree.

This overhead is:

- **Unavoidable:** No protocol can do better
- **Tight:** The bound is achievable to within $o(1)$
- **Mechanism-independent:** Whether you use extra metadata, additional rounds, committee proofs, or side channels

**Corollary B.4.3 (Channel Capacity):**

Under source–channel separation, a channel with Shannon capacity $C_{\mathrm{Shannon}}$ can only carry usable consensus payload at rate:

$$
C_{\mathrm{common}} = C_{\mathrm{Shannon}} - K(P)
$$

**Corollary B.4.4 (Witness-Error Tradeoff):**

If r is the witness rate (extra bits beyond Shannon baseline) and E is the level-$\eta$ type-II exponent for detecting frame contradictions, then: (§7.2)

$$
E + r \geq K(P)
$$

You can trade witness bits for statistical detectability, but their sum is bounded by $K(P)$.

### B.4.6 Constructive Witnesses

The bound is achievable: there exist witness strings $W_n$ of rate $K(P)+o(1)$ such that $\mathrm{TV}((X^n,W_n),\tilde{Q}_n)\to 0$. (§6.4)

**Implementation flexibility:**

The same information budget can be realized through:

- Extra metadata fields
- Additional communication rounds
- Cryptographic proofs
- Side channel coordination
- Schema negotiation

The lower bound constrains the total information cost, not the specific mechanism.