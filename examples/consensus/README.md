# Byzantine Consensus: Adaptive K(P)-Based Protocol

## Overview

Consider a leaky pipe network. A plumber doesn't demolish every wall to find the problem. Instead, they trace pressure drops across the system—a subtle change here, an unusual flow there—until the pattern reveals exactly where the leak originates.

![Pipe Network Pressure Analysis](../../figures/pipes.png)
*Visualizing pressure drops in a pipe network reveals exactly where leaks occur.*

Byzantine consensus works identically. Traditional approaches assume every node could be compromised and verify everything uniformly. That guarantees safety but wastes resources checking honest nodes with the same intensity as suspicious ones.

We wondered: what if the system could measure actual disagreement patterns and focus verification where problems appear? This implementation introduces that capability through $K(P)$, a contradiction measure quantifying observable disagreement in bits.

Rather than assuming worst-case faults everywhere, the protocol measures where faults actually manifest and allocates verification resources accordingly. Result: Byzantine consensus that adapts to real fault patterns instead of theoretical maximums.

## How It Works

The system builds on an insight from information theory: when nodes disagree, their messages carry an encoding cost. Perfect agreement between nodes requires minimal communication—just enough to convey the actual information.

But when Byzantine faults create contradictions, additional bits become necessary to resolve those inconsistencies. $K(P)$ measures how many extra bits those contradictions force into the system.

We formalize that through the global agreement coefficient $\alpha^*(P)$, which captures the best achievable consistency across all node pairs:

$$K(P) = -\log_2(\alpha^*(P))$$

where

$$\alpha^*(P) = \max_{Q \in \mathrm{FI}} \min_c \mathrm{BC}(p_c, q_c^*)$$

The framework finds the "unified explanation" Q that best accounts for observed behaviors, measuring agreement through the Bhattacharyya coefficient $BC$. That minimum agreement across all contexts determines the system's contradiction level.

This global measure decomposes into per-context contributions $K_c(P)$, revealing which specific node pairs drive disagreement. High $K_c$ values indicate contexts requiring extra verification, while near-zero values suggest honest interaction.

The protocol uses these measurements to allocate witness weights $\lambda^*$, sending additional verification messages proportional to observed contradiction. That adaptive mechanism integrates directly into consensus execution.

When $K_c$ exceeds thresholds, affected nodes receive extra prepare messages, enhanced auditing, and adjusted commit requirements. Nodes with consistently high contributions face mandatory message signing and reduced trust scores in voting.

This targeted approach maintains Byzantine fault tolerance guarantees while eliminating wasted verification of honest nodes. Less and less do we check nodes that already agree.

## Empirical Results

We tested the protocol across five distinct fault scenarios, each revealing different contradiction patterns.

**Perfect Agreement** establishes the baseline. With $K(P)$ measuring 0.0000 bits, the system confirms no Byzantine faults exist. All three nodes achieve consensus with minimal overhead—just the base entropy cost of 1.0000 bits per message.

That represents the optimum.

**Single Traitor** introduces one corrupted node sending conflicting messages to different peers. $K(P)$ rises to 0.0273 bits, a small but measurable increase. The witness allocation $\lambda^*$ concentrates 0.5000 weight on the N1-N2 context while distributing 0.2500 to each remaining pair.

That targets where the traitor's lies create observable inconsistency. The adaptive protocol eliminates fixed overhead, achieving savings compared to uniform verification.

**Triangle Paradox** creates maximum contradiction for three nodes: each pair observes perfect disagreement, yielding $K(P) = 0.5000$ bits. The Bhattacharyya coefficient drops to its minimum possible value of 0.7071 across all contexts, creating symmetric contradiction.

Despite this pattern, adaptive allocation still improves efficiency—the protocol knows where verification matters rather than guessing.

**Complex Network** extends to four nodes with heterogeneous faults. $K(P)$ measures 0.2934 bits overall, but per-context analysis reveals N2-N4 and N1-N2 pairs contributing 0.2934 bits each while N1-N3 drops to 0.1973 bits.

Node attribution identifies N2 (0.2932 bits) and N4 (0.2931 bits) as likely threats. The protocol applies enhanced verification selectively, reducing overhead versus uniform treatment.

**Asymmetric Byzantine** demonstrates subtle corruption: one node lies differently to each peer, creating 0.1609 bits total contradiction distributed unevenly. Witness allocation reflects this pattern with $\lambda = 0.3700$ for N2-N3, $\lambda = 0.3167$ for N1-N2, and $\lambda = 0.3133$ for N1-N3.

The system adapts its verification strategy to match the specific fault signature.

![Byzantine Adaptive Analysis Dashboard](byzantine_adaptive_analysis.png)
*Visualization showing $K(P)$ values, agreement coefficients, message overhead, per-context contributions, cumulative costs, witness allocation, protocol savings, and node fault attribution across all tested scenarios.*

## What This Means

The 100% savings figures appear in four of five scenarios. These numbers reflect theoretical communication analysis—specifically, the difference between paying $K(P)$ bits uniformly every round versus allocating them dynamically based on witness weights.

In scenarios where contradiction concentrates in specific contexts, the adaptive protocol avoids sending extra bits to honest node pairs that already agree. That represents eliminated waste, not compromised safety.

The actual Byzantine consensus layer maintains full safety guarantees regardless of $K(P)$. The PBFT-style protocol still requires $2f+1$ prepare and commit messages, still tolerates up to $f = \lfloor(n-1)/3\rfloor$ faults, and still ensures agreement among correct nodes.

What $K(P)$ provides is optimization opportunity: knowing which contexts need verification lets the system add redundancy where it helps while skipping redundancy where it doesn't.

In perfect agreement scenarios, $K(P) = 0$ confirms no Byzantine faults exist, so no extra verification becomes necessary. In single traitor scenarios, $K(P)$ localizes contradiction to specific contexts, enabling targeted rather than uniform verification.

The savings represent eliminated waste, not compromised safety. That is the point.

## Implementation Characteristics

The current implementation handles networks up to 10 nodes with $O(n^2)$ context complexity. Beyond that scale, the full optimization becomes computationally expensive, requiring approximations through context sampling and iterative refinement.

The mathematical framework extends to arbitrary network sizes, but practical deployment at scale needs these computational shortcuts.

Fault localization behaves differently at different scales. In small networks ($n \leq 4$), contradiction often spreads evenly across nodes even when faults concentrate in specific locations. This reflects information-theoretic reality: with few observations, distinguishing correlation from causation becomes impossible.

The system reports conservative attribution, preventing false accusations when evidence remains ambiguous. As networks grow, localization precision improves. More contexts provide more measurements, enabling statistical confidence about which nodes actually cause observed disagreement.

The implementation uses multi-tier detection: Byzantine bounds $f = \lfloor(n-1)/3\rfloor$ for theoretical guarantees, plus $3\sigma$ statistical outlier detection for practical fault identification.

Node classification prioritizes conservative thresholds. When $K(P) > 0$ but attribution remains ambiguous, the system reports zero faulty nodes rather than guessing. This design choice prevents false positives—incorrectly flagging honest nodes causes more harm than temporarily missing actual faults that reveal themselves over time.

The protocol currently analyzes static fault patterns injected for testing. Real Byzantine adversaries adapt strategically, potentially hiding contradiction by coordinating lies. Our experiments suggest coordinated attacks can reduce detectable $K(P)$ by 91-95% through carefully distributed, mild inconsistencies that avoid triggering statistical thresholds.

Defending against such adaptive adversaries requires temporal analysis—tracking behavior across consensus rounds to detect patterns invisible in single-round snapshots.

## Running the Analysis

Execute the complete suite:

```bash
cd examples/consensus
python run.py
```

The system generates detailed output showing per-scenario $K(P)$ measurements, node contribution analysis, witness allocation, encoding overhead, fault attribution, mitigation recommendations, consensus execution, and adaptive protocol performance. All visualizations save to `byzantine_adaptive_analysis.png` with 300 DPI resolution.

## Limitations and Future Directions

Several constraints shape this work. $K(P)$ calculation requires solving convex optimization problems that become expensive at scale. The current implementation computes exact solutions for moderate networks, but large deployments need approximation techniques trading precision for speed.

The reactive nature of the approach presents both strength and vulnerability. Measuring observed contradiction enables precise resource allocation but requires faults to manifest before detection. Sophisticated adversaries might exploit this by minimizing observable $K(P)$ through coordination.

Defending robustly requires extending the framework with temporal pattern recognition and historical behavior analysis.

Fault localization thresholds impact performance. Default settings flag nodes contributing $K_c(P) > 0.1$ bits as suspicious, but appropriate values vary by application. Too-sensitive thresholds create false positives; too-lenient ones miss real threats.

Proper deployment requires empirical tuning across diverse fault scenarios specific to each application domain.

Context sampling offers a natural path to scalability. Rather than computing all $O(n^2)$ pair-wise contexts, randomly sampling a fixed subset provides approximate $K(P)$ estimates. Theoretical analysis can bound approximation error, enabling principled trade-offs between accuracy and computational cost.

Temporal analysis addresses adaptive adversaries. By tracking $K_c(P)$ evolution across consensus rounds, the system can detect nodes whose contribution patterns change strategically. Sudden drops in measured contradiction after consistently high levels suggest coordination or adaptation worth investigating.

Hierarchical network structure improves localization precision. Rather than treating all nodes symmetrically, leveraging known topology—data center racks, geographic regions, organizational boundaries—provides additional signal for attributing observed contradiction to likely sources.

## Theoretical Foundation

This work builds on operational information theory, specifically Theorems 11-13 concerning communication costs under behavioral contradiction. Theorem 11 establishes that common messages require $H(X|C) + K(P)$ bits, combining conditional entropy with contradiction overhead.

Our experiments validate this relationship empirically across diverse fault scenarios.

Frame independence characterizes behaviors admitting unified explanations—joint distributions Q that account for all observed contexts. The optimization finding $\alpha^*(P)$ searches over all frame-independent behaviors, identifying the one achieving maximum agreement with observations.

This provides the mathematical basis for $K(P)$ as a measure of irreducible contradiction.

The Bhattacharyya coefficient measures distributional similarity, providing the agreement metric. For two distributions p and q, $BC(p,q) = \sum\sqrt{p(x)q(x)}$ ranges from 0 (complete disagreement) to 1 (perfect match).

This choice over alternatives like KL divergence provides desirable mathematical properties: symmetry, boundedness, and multiplicative decomposition across independent contexts.

Witness allocation $\lambda^*$ emerges from the minimax optimization defining $\alpha^*(P)$. The weights satisfy $\sum_c \lambda_c = 1$ and represent how much each context contributes to the worst-case agreement bound. High $\lambda_c$ values indicate contexts where disagreement forces extra verification bits, providing the natural allocation strategy for adaptive protocols.

Byzantine fault tolerance literature provides context. Fischer, Lynch, and Paterson proved that deterministic asynchronous consensus becomes impossible with even one fault, establishing limits. Lamport's Byzantine Generals Problem formalized the challenge of agreement under arbitrary failures.

Castro and Liskov's PBFT demonstrated practical solutions achieving consensus in polynomial time despite theoretical impossibility results.

Our contribution sits at the intersection: using information-theoretic contradiction measures to optimize practical Byzantine consensus protocols while maintaining safety guarantees. $K(P)$ doesn't solve the impossibility result—it measures the cost of working around it and guides resource allocation for doing so efficiently.

## Summary

Byzantine consensus has traditionally treated all nodes as equally suspicious, verifying everything uniformly to guarantee safety. That works but wastes resources checking honest nodes with the same intensity as potentially malicious ones.

$K(P)$ provides a measurement: an information-theoretic quantification of actual disagreement observable in the network.

By decomposing $K(P)$ into per-context and per-node contributions, the system identifies where faults manifest. Adaptive allocation follows naturally—send extra verification to contexts with high $K_c$, enhance auditing for nodes with high contribution scores, skip redundancy where $K_c$ approaches zero.

The result maintains Byzantine safety guarantees while eliminating systematic waste.

The approach scales theoretically to arbitrary network sizes, though practical implementation requires approximations beyond $n \approx 10$ nodes. Fault localization improves with scale as more contexts provide more statistical confidence.

Conservative thresholds prevent false positives, preferring temporary missed detections over incorrect accusations.

Real deployment must address adaptive adversaries through temporal analysis and historical pattern recognition. Static fault detection catches naive misbehavior but sophisticated attacks require tracking how $K(P)$ patterns evolve.

Future work will extend the framework to handle these dynamic threat models.

This represents progress toward effective Byzantine consensus: protocols that achieve safety guarantees while paying only the communication cost imposed by actual faults rather than theoretical worst cases. The mathematics shows $K(P)$ measures that cost. The implementation demonstrates that measurement enables optimization.