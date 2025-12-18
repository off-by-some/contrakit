# Quantum Contradiction Examples

*We quantify "how quantum" an action is by the minimal info (in bits) needed to reconcile it with a single classical frame. Below classic bounds this cost is zero; above them it becomes positive and grows smoothly. Across CHSH, KCBS and Magic Square we measure ~0.012–0.013 bits for probabilistic violations and ~0.132 bits for algebraic ones, revealing a taxonomy of quantum conflict with a universal currency: bits.*

Specifically, K(P) measures distance from the classical (local/noncontextual) FI set we specify. It measures the classical side-info needed to mimic these behaviors

## The Big Picture

Formally, these examples explore one of the deepest puzzles in physics: why quantum mechanics defies our classical intuitions. Having established the basic setup, we now turn to the contradiction measure $K(P)$.

These examples explore one of the deepest puzzles in physics: why quantum mechanics defies our classical intuitions. We use the contradiction measure $K(P)$ to quantify how much info it costs to reconcile quantum action with classical expectations.

Let us summarize the key insights; each reveals key aspects of quantum-classical boundaries:
- **Classical region:** $K(P) = 0$;
- **Quantum region:** $K(P) > 0$ and grows smoothly with violation strength;
- **Two flavors:** probabilistic violations $\sim 0.012$ bits; algebraic impossibility $\sim 0.132$ bits;


We examine three landmark quantum scenarios that each represent a different facet of quantum nonclassic: Bell inequalities (CHSH) that reveal links exceeding classic limits, contextuality cycles (KCBS) that show measure incompatibilities in single systems, and the Magic Square that shows logical impossibilities resolved through quantum mechanics.

Across all examples, we verify the universal invariant that establishes an info-based boundary between classical and quantum physics: classical behavior costs nothing to explain classically ($K(P) = 0$), while genuinely quantum behavior always incurs a positive info cost ($K(P) > 0$) that grows smoothly as quantum effects strengthen, providing a continuous measure of quantum nonclassic that enables precise quantification of how quantum a given phenomenon truly is.


### At a Glance

$K(P)$ = bits per joint context sample; for $n$ i.i.d. runs, cost $\approx n \cdot K(P)$ bits.

| Scenario | Violation Parameter | Classical Bound | Quantum Point | $`\alpha^*`$ | $K(P)$ (bits) | Character |
|----------|-------------------:|----------------:|--------------:|----------:|-------------:|-----------|
| CHSH | $S$ | 2 | $2\sqrt{2}$ | 0.991445 | 0.012396 | Probabilistic, state-dependent |
| KCBS | $\Sigma\langle E_i \rangle$ | 2 | $\sqrt{5}$ | 0.991206 | 0.012744 | Probabilistic, state-dependent |
| Magic Square | $W$ | ≤ 4 | 6 | 0.912871 | 0.131517 | Algebraic, state-independent |



## Under the Hood: Computing $K(P)$

The library, among other things, converts quantum experiments into precise info costs. Here's Bell's theorem quantified:

```python
import numpy as np
from contrakit import Space, Behavior

# Define Alice and Bob's measure choices
space = Space.create(A0=[-1,+1], A1=[-1,+1], B0=[-1,+1], B1=[-1,+1])

# Quantum correlations at maximum violation (Tsirelson bound)
r = 1/np.sqrt(2)
contexts = {
    ("A0", "B0"): {(+1,+1): (1-r)/4, (+1,-1): (1+r)/4, (-1,+1): (1+r)/4, (-1,-1): (1-r)/4},
    ("A0", "B1"): {(+1,+1): (1-r)/4, (+1,-1): (1+r)/4, (-1,+1): (1+r)/4, (-1,-1): (1-r)/4},
    ("A1", "B0"): {(+1,+1): (1-r)/4, (+1,-1): (1+r)/4, (-1,+1): (1+r)/4, (-1,-1): (1-r)/4},
    ("A1", "B1"): {(+1,+1): (1+r)/4, (+1,-1): (1-r)/4, (-1,+1): (1-r)/4, (-1,-1): (1+r)/4},
}

behavior = Behavior.from_contexts(space, contexts)
print(f"Information cost: {behavior.contradiction_bits:.4f} bits")  # 0.0124
print(f"Classical overlap: {behavior.alpha_star:.4f}")              # 0.9914 (.agreement is another alias available too)
```

* **Result:** Quantum correlations that violate Bell inequalities cost exactly **0.0124 bits per joint context sample** to reconcile with classical physics. For $n$ i.i.d. runs, the info cost scales $\approx n \cdot K(P)$ bits.

* **What $`\alpha^*`$ means:** $`\alpha^*`$ is the best achievable average overlap with any frame-independent (classical) model; $K = -\log_2 \alpha^*$ is the per-run information you must 'carry' to fake frame-independence.


## From "Is It Quantum?" to "How Quantum Is It?"

Consider the following.

Traditional approaches to quantum foundations ask a yes-or-no question: *Can this behavior be explained classically?* This binary perspective, known as **contextuality**, has been enormously successful in identifying quantum phenomena. But it leaves an important question unanswered: when the answer is "no," just how non-classical is the behavior?

Our contradiction measure $K(P)$ transforms this binary question into a quantitative one. Instead of asking whether quantum behavior can be explained classically, we ask: *What is the minimum information cost of forcing a classical explanation?*

### The Power of Quantification

To continue our analysis, we examine the advantages of this quantitative approach.

This shift from binary to quantitative brings several advantages; each addressing limitations of traditional quantum tests:

1. **Precise comparison across phenomena.** Different quantum experiments violate different inequalities with different units and scales; but contradiction is always measured in the same universal currency: bits of information. This lets us directly compare the "quantumness" of Bell violations (~0.012 bits) with Magic Square puzzles (~0.132 bits); and this reveals that algebraic quantum constraints are roughly ten times more costly to explain classically than probabilistic ones.

2. **Localization of quantum effects.** While traditional tests tell you *that* an experiment violates classical bounds; our dual witness $\lambda^*$ tells you *where* the violation is concentrated. Which measure contexts are the bottlenecks? Which experimental configurations contribute most to the quantum advantage? This localization guides both theoretical understanding and experimental design; but it also opens new questions about quantum measure structure.

3. **Engineering applications.** Abstract concepts like "nonclassical correlations" become concrete engineering parameters; the contradiction measure translates directly into resource budgets: How much side-information per experimental run? What sampling strategies minimize detection time? How much noise can the quantum advantage tolerate? These questions have precise, computable answers.

4. **Smooth scaling laws.** In repeated experiments, contradiction costs add linearly: $n$ independent runs require approximately $n \cdot K(P)$ bits of side-information to maintain classical explanations. This clean scaling law enables precise resource planning and statistical analysis. So binary violation tests cannot provide this level of detail.

5. **Boundary precision.** The measure exhibits mathematical elegance at the classical-quantum boundary: exactly zero cost for any classically explainable behavior, positive cost that grows continuously as quantum effects strengthen. This precision enables detection of quantum effects arbitrarily close to classical bounds and solid analysis of near-threshold experiments. Yet the mathematical elegance here is striking. Or consider the implications for quantum computing.


## Key Discoveries

To continue our exploration, we now examine the fundamental discoveries enabled by this quantitative approach.

### A Universal Quantum Signature

Our analysis reveals that the contradiction measure $K(P)$ Behaves like a resource monotone in our setting with consistent mathematical properties:

**Sharp classical-quantum boundary.** The measure exhibits perfect discrimination: exactly zero for any behavior explainable by classical physics, and strictly positive for genuinely quantum phenomena. In our runs, boundary detection achieves $K(P) \leq 10^{-9}$ below classical bounds and $K(P) \geq 10^{-5}$ above them, providing an unambiguous test for quantum effects.

**Continuous quantum scaling.** Beyond the classical boundary, contradiction grows smoothly rather than jumping discontinuously. This continuity enables precise measure of "how quantum" a phenomenon is, and solid analysis of experiments operating near the classical-quantum threshold.

### A taxonomy of Quantum Contradiction Costs

Our three case studies reveal that different types of quantum behavior incur dramatically different information costs:

| Phenomenon | Information Cost | Quantum Parameter | Character |
|------------|------------------|-------------------|-----------|
| **Bell Correlations (CHSH)** | 0.012 bits | $`\alpha^* = 0.991`$ | Probabilistic, state-dependent |
| **Measurement Contextuality (KCBS)** | 0.013 bits | $`\alpha^* = 0.991`$ | Probabilistic, state-dependent |
| **Logical Impossibility (Magic Square)** | 0.132 bits | $`\alpha^* = 0.913`$ | Algebraic, state-independent |

**The fundamental distinction.** Quantum phenomena fall into two categories with vastly different contradiction costs:

- **Probabilistic violations** (Bell inequalities, contextuality cycles) arise from quantum correlations that exceed classical bounds but remain logically consistent. These incur modest information costs (~0.012 bits).

- **Algebraic impossibilities** (Magic Square) arise from measure constraints that cannot be satisfied by any classical assignment, regardless of probabilities. These incur an order of magnitude higher costs (~0.132 bits).

This taxonomy shows that logical contradictions represent a fundamentally more expensive form of quantum behavior than statistical violations. And this distinction remains invisible to traditional binary tests but clearly revealed by information-theoretic analysis.

## Reading the Quantum Landscape: A Guide to Our Visualizations

The three figures below tell a unified story of how quantum behavior emerges from classical foundations. Each visualization maps the journey from classical physics (where contradiction costs nothing) to quantum regimes (where information costs grow continuously).

**Visual language.** Every figure uses the same interpretive framework: horizontal axes show the strength of quantum violations (different units for different experiments), while vertical axes show the information cost in bits. Red vertical lines mark the classical boundaries—the precise points where quantum effects begin. Colored data points trace parameter sweeps, with key configurations labeled for reference.

**Precision at the boundary.** The logarithmic panels in each figure reveal a key detail: our measure achieves machine-level precision at classical boundaries, with contradiction costs dropping to zero within computational limits (better than 10⁻⁹). This precision validates the mathematical sharpness of the classical-quantum transition.

**Universal patterns.** Despite examining very different physical scenarios, all three figures exhibit the same fundamental behavior: zero cost in classical regions, continuous growth in quantum regions, and smooth scaling that reflects the underlying physics. This universality indicates that information cost is a fundamental feature of quantum mechanics itself.

## The Magic Square: When Logic Itself Becomes Quantum

The Magic Square represents the clearest example in our collection—a scenario where quantum mechanics doesn't just violate statistical bounds, but achieves something that classical logic declares impossible.

**The puzzle.** Imagine a 3×3 grid where each cell contains either +1 or -1. The constraint is simple: each row must multiply to +1, and each column must multiply to +1, except the last column which must multiply to -1. Try to fill in such a grid with classical logic—you'll find it's impossible. The mathematical constraint is self-contradictory.

**The quantum solution.** Quantum mechanics solves this "impossible" puzzle effortlessly. Using quantum measures, we can construct a scenario that satisfies all constraints simultaneously, achieving a witness value of W = 6 compared to the classical maximum of W = 4.

**What the figure reveals.** The eight-panel analysis shows this logical impossibility translating into information cost. The contradiction measure reaches $K = 0.132$ bits—roughly ten times higher than the probabilistic violations in Bell tests. **Why Magic Square costs ~10× more:** Because the contradiction is **algebraic** (parity constraints) rather than merely **statistical**; no probability rescaling can satisfy all contexts simultaneously.

**Robustness and universality.** Unlike other quantum effects that depend on carefully prepared states, the Magic Square contradiction is universal—it emerges from the measure structure itself, independent of the quantum state used. The analysis shows this quantum advantage survives substantial noise (up to 30% error rates), making it accessible to real experiments.

**The deeper meaning.** This scenario shows that quantum mechanics doesn't just allow stronger correlations than classical physics—it enables forms of logical consistency that classical reasoning cannot achieve. The information cost quantifies exactly how much extra complexity is needed to force classical logic to account for quantum logical structures.

![Magic Square contradiction analysis](../../figures/magic_square_analysis.png)
*Figure 1: Magic Square analysis showing state-independent algebraic contradiction with $W=6$, $`\alpha^* = \sqrt{5/6}`$, and $`K(P) = \frac{1}{2}\log_2(6/5) \approx 0.1315`$ bits (exact analytic result). The multipanel includes parity constraints, quantum-classical comparison, perturbation robustness, contradiction landscape, and quantum advantage analysis.*

**Reproduce:**
```shell
$ poetry run python -m examples.quantum.magic_squares
```
→ saves `figures/magic_square_analysis.png`

**Alt text:** Multi-panel visualization comparing classical ($W=4$) and quantum ($W=6$) Magic Square solutions, with contradiction measure $K(P)$ showing order-of-magnitude higher cost for algebraic constraints versus probabilistic violations.

### Practical Interpretations

**Information budgets.** The contradiction measure $K(P)$ represents the minimal extra information (in bits) needed per experimental run to maintain a classical explanation of quantum behavior. This transforms abstract quantum effects into concrete resource requirements for classical simulation.

**Optimal experimental strategies.** The dual witness $\lambda^*$ provides mathematically optimal guidance for experimental design. It tells researchers which measure contexts deserve the most attention and how to distribute limited experimental resources to maximize evidence for quantum effects.

**Simulation complexity.** For importance sampling, variance lower bounds scale like $2^{2K}-1$.

**Noise tolerance specifications.** By tracking how contradiction costs degrade under realistic experimental conditions, the framework provides precise specifications for the noise tolerance required to maintain quantum advantages in practical applications.

### Benchmark Results

Our analysis establishes precise contradiction costs for three landmark quantum phenomena:

- **Bell correlations (CHSH)**: At the maximum quantum violation $S = 2\sqrt{2}$, maintaining classical explanations costs $K = 0.012396$ bits per measure
- **Measurement contextuality (KCBS)**: At the quantum optimum $\Sigma\langle E_i \rangle = \sqrt{5}$, the classical explanation cost is $K = 0.012744$ bits per measure
- **Logical impossibility (Magic Square)**: The quantum solution $W = 6$ incurs a cost of $K = 0.131517$ bits—exactly $\frac{1}{2} \cdot \log_2(6/5)$ as predicted by theory

These benchmarks provide reference points for comparing the "quantumness" of other phenomena and guide the development of new quantum technologies.

## Bell's Theorem Made Quantitative: The CHSH Analysis

The CHSH inequality, derived from John Bell's work and showing that quantum particles can exhibit correlations stronger than any classical theory allows, is transformed by our analysis from Bell's binary insight—"quantum correlations violate classical bounds"—into a precise quantitative measure of quantum nonclassic that reveals the fundamental information cost of maintaining classical explanations in the face of quantum correlations.

**The experimental setup.** Two separated particles are measured along different angles. Classical physics predicts that the correlation strength $S$ cannot exceed 2, regardless of any hidden variables or local mechanisms. Quantum mechanics, however, can achieve $S = 2\sqrt{2} \approx 2.828$—a violation that Einstein famously called "spooky action at a distance."

**From threshold to quantity.** The figure traces how information cost grows as we move from classical ($S \leq 2$) to quantum ($S > 2$) regimes. The red line marks the classical boundary—a sharp threshold where contradiction cost jumps from exactly zero to positive values. This transition occurs with mathematical precision: no classical behavior incurs any information cost, while even the smallest quantum violation requires positive information to explain classically.

**The Tsirelson point.** At quantum mechanics' maximum violation ($S = 2\sqrt{2}$), the contradiction cost reaches $K = 0.012$ bits per measure. This small number represents the fundamental information gap between classical and quantum descriptions of nature. To maintain a classical worldview in the face of maximal Bell violations, you must invoke approximately $0.012$ bits of additional information per measure.

**Precision and continuity.** The logarithmic panel reveals the smooth mathematical transition: contradiction costs drop to $K(P) \leq 10^{-9}$ at the classical boundary and grow continuously beyond it. This continuity enables precise detection of quantum effects arbitrarily close to classical limits—a key capability for experiments operating near the quantum threshold.

![CHSH inequality landscape and threshold analysis](../../figures/bell_chsh_analysis.png)
*Figure 2: CHSH inequality analysis showing the full parameter landscape, threshold detection, and boundary zoom revealing the continuous onset of contextuality at $`S = 2`$. The figure traces how $`K(P)`$ grows smoothly from 0 below the classical bound to $`\approx 0.0124`$ bits at the Tsirelson bound ($`S = 2\sqrt{2}`$).*

**Reproduce:**
```shell
$ poetry run python -m examples.quantum.CHSH
```
→ saves `figures/bell_chsh_analysis.png`

**Alt text:** Scatter plot showing CHSH inequality violation ($S$) on x-axis versus contradiction measure $K(P)$ on y-axis, with red boundary line at $S=2$ and logarithmic inset revealing machine precision (zero to within numerical tolerance (≤1e-9) at the classical-quantum transition.

**Note:** Any tiny offset of the last sweep point from exactly $S = 2$ is due to grid resolution in the parameter sweep, not physics—the invariant holds exactly.

## Mathematical Validation

Our theoretical predictions achieve high precision when compared to computational results:

**Exact theoretical agreement.** The fundamental relationship $`K = -\log_2(\alpha^*)`$ holds to better than $10^{-12}$ relative error across all scenarios in our analyses, confirming the mathematical consistency of the framework.

**Physical constraint satisfaction.** All quantum scenarios satisfy fundamental physical principles, including no-signalling constraints, to better than $10^{-12}$ precision in our implementations. This validates that our quantum models represent physically realizable systems.

**Dependencies:** Requires matplotlib and qutip. Install with `poetry install --with examples`

**Numerics:** All runs use SCS via CVXPY; invariants verified to $\leq 10^{-9}$ where expected. Tiny offsets at sweep endpoints reflect grid resolution and solver tolerance.

**Sharp boundary detection.** The classical-quantum boundaries exhibit mathematical sharpness: contradiction costs are effectively zero under this FI baseline in classical regions and strictly positive immediately beyond classical bounds, with transitions detected to better than $10^{-9}$ precision in our runs.

### Universal Scaling Laws

The contradiction measure exhibits consistent mathematical properties that hold across all quantum scenarios:

**Boundary continuity.** At classical-quantum boundaries, contradiction costs transition from exactly zero to positive values with continuous (though often kinked) growth. This continuity enables precise detection of quantum effects arbitrarily close to classical limits.

**Noise degradation.** In our depolarizing-noise sweeps, contradiction remains positive up to ~30% noise; this is model-dependent. All scenarios show predictable degradation patterns. Quantum advantages persist until scenario-specific noise thresholds, beyond which systems cross back into classical regimes. This predictability enables reliable experimental design. 

**Compositional scaling.** For repeated independent experiments, contradiction costs add linearly: $n$ runs require approximately $n \cdot K(P)$ bits of side-information. This clean scaling law enables precise resource planning for large-scale quantum experiments and provides a foundation for statistical analysis of quantum advantages.

## The Measurement Dilemma: KCBS Contextuality

The KCBS scenario reveals a subtle aspect of quantum mechanics: the impossibility of assigning definite values to all quantum observables simultaneously. Unlike Bell inequalities, which focus on correlations between distant particles, KCBS contextuality emerges from the fundamental incompatibility of quantum measures on a single system.

**The pentagonal puzzle.** Imagine five quantum observables arranged in a pentagon, where adjacent observables can be measured together, but non-adjacent ones cannot. Classical physics assumes each observable has a definite value (+1 or -1) independent of which measures we choose to perform. The KCBS inequality tests this assumption by examining the sum of all five expectation values.

**Classical limits and quantum transcendence.** Classical physics constrains this sum to be at most 2. Quantum mechanics, however, can achieve $\Sigma\langle E_i \rangle = \sqrt{5} \approx 2.236$ by exploiting the measure incompatibilities. This violation shows that quantum observables cannot possess definite values independent of measure context—a phenomenon known as contextuality.

**Information cost of contextuality.** At the quantum optimum, maintaining a classical (context-independent) explanation requires $K = 0.013$ bits per measure—nearly identical to the Bell violation cost. This near-equality reveals a connection between different forms of quantum nonclassic, despite their distinct physical origins.

**The measure context dependence.** Unlike the Magic Square, KCBS contextuality requires carefully prepared quantum states. The contradiction emerges only when the system achieves specific quantum correlations, highlighting the delicate balance between state preparation and measure strategy in quantum experiments.

![KCBS contextuality and noise robustness analysis](../../figures/kcbs_contextuality_analysis.png)
*Figure 3: KCBS analysis demonstrating boundary behavior (continuous onset at $`\Sigma\langle E_i \rangle = 2`$), noise robustness under depolarizing channels, and the relationship between violation magnitude and contradiction measure. Shows classical bound at 2, quantum maximum at $`\sqrt{5} \approx 2.236`$ with $`K \approx 0.0127`$ bits, and super-quantum regime at $`\Sigma = 2.5`$ with $`K \approx 0.161`$ bits.*

**Reproduce:**
```shell
$ poetry run python -m examples.quantum.KCBS
```
→ saves `figures/kcbs_contextuality_analysis.png`

**Alt text:** Multi-panel analysis of KCBS contextuality showing pentagonal compatibility graph, parameter sweep from classical bound ($\Sigma = 2$) to quantum optimum ($\Sigma = \sqrt{5}$) to super-quantum regime ($\Sigma = 2.5$), with contradiction measure scaling accordingly.

## Universal Patterns Across Quantum Phenomena

Despite examining three very different physical scenarios—particle correlations, measure contextuality, and logical constraints—we show this analysis reveals consistent universal patterns. Consider the implications: these patterns suggest deep principles underlying quantum mechanics.

**The classical-quantum boundary is mathematically sharp.** Across all scenarios, the transition from classical to quantum behavior occurs at defined thresholds with machine-level accuracy. Below these boundaries, contradiction costs are exactly zero; above them, costs become strictly positive. This sharpness validates the fundamental distinction between classical and quantum physics.

**Information costs provide universal comparison.** While traditional quantum tests use different units and scales (correlation strengths, expectation values, witness functions), contradiction costs are always measured in bits. This common currency reveals that Bell violations and KCBS contextuality incur nearly identical information costs ($\approx 0.012-0.013$ bits), while algebraic constraints like the Magic Square cost an order of magnitude more ($\approx 0.132$ bits).

**Quantum effects scale continuously.** Beyond classical boundaries, contradiction costs grow smoothly rather than jumping discontinuously. This continuity enables precise measure of "quantumness" and solid analysis of near-threshold experiments—capabilities impossible with binary violation tests.

**Symmetry reflects underlying structure.** The uniform distribution of dual witness weights ($\lambda^*$) across measure contexts reflects the inherent symmetries in each scenario: the four-fold symmetry of Bell tests, the five-fold symmetry of KCBS pentagons, and the six-fold symmetry of Magic Square constraints. This mathematical consistency suggests that contradiction measures capture fundamental geometric properties of quantum theory.

## What the Data Says

- **Contradiction is a continuous resource.** It scales smoothly with violation strength and adds linearly across i.i.d. runs.
- **Structure matters.** Algebraic contextuality (Magic Square) is fundamentally more "expensive" than probabilistic violations (CHSH/KCBS) by an order of magnitude.
- **Symmetry shows up in the dual.** Uniform $\lambda^*$ across contexts mirrors each scenario's symmetry—witnesses aren't just certificates; they're lenses on structure.
- **State-independence is loud.** Magic Square's $K(P)$ doesn't rely on a special state: the measure algebra itself carries contradiction.

## From Theory to Practice: Applications of Quantified Quantum Weirdness

The contradiction measure transforms abstract quantum foundations into practical tools with applications spanning fundamental research, experimental design, and technological development.

### Advancing Quantum Foundations

**Graded quantum effects.** Traditional tests answer "Is this quantum?" with yes or no. Our measure answers "How quantum is this?" with precise numerical values, enabling fine-grained analysis of quantum phenomena. Researchers can now quantify the relative "quantumness" of different experiments, states, and protocols using a common information-theoretic scale.

**Universal comparison framework.** The bit-based measure enables direct comparison across disparate quantum phenomena. A Bell violation can be directly compared to a contextuality demonstration or a quantum computational advantage, revealing deep connections between seemingly unrelated quantum effects.

**Geometric insights.** The measure provides information-geometric distances between quantum behaviors and their closest classical approximations. This geometric perspective reveals the structure of quantum theory's departure from classical physics, potentially guiding the development of new quantum theories or modifications.

### Optimizing Quantum Experiments

**Resource allocation.** The dual witness $\lambda^*$ provides optimal strategies for experimental design. Which measure contexts should receive the most attention? How should limited experimental resources be distributed to maximize the evidence for quantum effects? The witness provides mathematically optimal answers.

**Precision requirements.** The measure quantifies exactly how precisely experiments must be performed to detect quantum effects. For near-threshold phenomena, this precision guidance is key for distinguishing genuine quantum behavior from experimental noise.

**Noise tolerance analysis.** By tracking how contradiction costs degrade under noise, researchers can determine the maximum error rates compatible with quantum advantages. This analysis is essential for assessing the feasibility of quantum technologies in realistic, noisy environments.

### Enabling Quantum Technologies

**Classical simulation costs.** The contradiction measure provides lower bounds on the computational resources required to simulate quantum systems classically. These bounds inform the development of quantum algorithms and help identify scenarios where quantum computers offer genuine advantages.

**Communication complexity.** In distributed quantum systems, the measure quantifies the minimum communication required to reconcile different measure contexts. This analysis guides the design of quantum networks and distributed quantum protocols.

**Quantum advantage certification.** The measure provides rigorous methods for certifying when quantum systems outperform classical alternatives, with applications in quantum computing, quantum sensing, and quantum communication. **Certification rule of thumb:** With $m$ samples, a behavior with $K(P) = k$ yields a log-likelihood ratio $\approx m \cdot k$ bits against any frame-independent model.

## The Road Ahead: Expanding the Quantum Information Landscape

The contradiction measure opens new avenues for understanding quantum mechanics that extend beyond the scenarios examined here. Several directions emerge naturally from our framework.

### Scaling to Complex Quantum Systems

**Higher-dimensional quantum systems.** While our examples focus on two-level quantum systems (qubits), the framework readily extends to higher-dimensional systems (qudits). How do contradiction costs scale with system dimension? Do higher-dimensional systems exhibit fundamentally different types of quantum behavior?

**Many-particle quantum systems.** Multipartite quantum systems—involving three, four, or more particles—exhibit rich contextual structures beyond pairwise correlations. The contradiction measure could reveal how quantum effects compound in complex many-body systems, with implications for quantum computing and quantum simulation.

**Continuous variable systems.** Quantum systems with continuous degrees of freedom (like position and momentum) present infinite-dimensional contextuality scenarios. Extending our discrete framework to continuous variables clarifies the quantum-classical boundary in systems like quantum optics and gravitational wave detectors.

### Bridging Quantum and Classical Worlds

**Hybrid quantum-classical systems.** Real quantum technologies operate in environments that are neither purely classical nor purely quantum. How do contradiction costs behave in hybrid systems where quantum and classical components interact? This analysis could guide the design of practical quantum devices.

**Decoherence and environmental effects.** Our noise analysis provides a foundation for understanding how environmental decoherence affects quantum advantages. Future work could develop comprehensive models of how different types of noise—not just depolarizing channels—impact contradiction costs.

### Computational and Algorithmic Advances

**Efficient computation for large systems.** As quantum systems grow in complexity, computing contradiction measures becomes computationally challenging. Advanced optimization techniques, such as column generation for large constraint polytopes, could make the framework scalable to realistic quantum systems with hundreds or thousands of components.

**Machine learning integration.** Could machine learning techniques identify optimal quantum states or measure strategies for maximizing contradiction? The intersection of quantum foundations and artificial intelligence presents unexplored opportunities for discovery.

### Fundamental Questions

**The information-theoretic structure of quantum mechanics.** Our work indicates that information cost is a fundamental feature of quantum theory. Contradiction measures provide new axiomatizations of quantum mechanics based on information-theoretic principles.

**Quantum gravity and spacetime.** If spacetime itself is quantum mechanical, how do contradiction measures behave in quantum gravitational systems? This question connects quantum foundations to fundamental questions in theoretical physics.

## Example Structure

```
examples/quantum/
├── CHSH.py                   # Bell inequality violation analysis
├── KCBS.py                   # Kochen-Specker contextuality cycles
├── magic_squares.py          # Algebraic contextuality example
├── run.py                    # Main execution script for all examples
├── utils.py                  # Visualization and helper functions
└── __init__.py               # Package initialization
```

## How the Examples Use the Library

Each quantum example demonstrates the library's core workflow:

### 1. Define Measurement Space
```python
# CHSH example creates a 4-context measure space
CHSH_SPACE = Space.create(
    A0=[-1, +1], A1=[-1, +1],  # Alice's measures
    B0=[-1, +1], B1=[-1, +1]   # Bob's measures
)
```

### 2. Create Experimental Behavior
```python
# Convert quantum correlations into behavior object
behavior = Behavior.from_contexts(space, contexts)
```

### 3. Compute Contradiction Measures
```python
# Key quantities
K = behavior.contradiction_bits      # Information cost in bits
alpha = behavior.alpha_star          # Best classical overlap
lambda_opt = behavior.least_favorable_lambda()  # Optimal witness
```

### 4. Analyze Results
```python
# Verify the invariant
if S <= 2:
    assert K < 1e-9  # Classical region: zero cost
else:
    assert K > 0     # Quantum region: positive cost
```

Each example includes:
- Parameter sweeps with boundary verification
- Noise robustness analysis with classical crossing detection
- Operational interpretations of contradiction measures
- Comprehensive visualization output (saved to figures/ directory)

**Run all examples:**
```shell
$ poetry run python -m examples.quantum.run
```
→ generates all three analysis figures

### Example Visualizations

The analysis generates detailed visualizations for each scenario:

- **bell_chsh_analysis.png**: CHSH inequality landscape with threshold detection and boundary zoom
- **kcbs_contextuality_analysis.png**: KCBS violation patterns and noise robustness curves
- **magic_square_analysis.png**: Magic Square algebraic constraints and contradiction measures

*All figures include both classical and quantum regimes with precise boundary detection and scaling analysis.*

## References

### Core References

* **Bell nonlocality foundation:**
  J. S. Bell, *On the Einstein Podolsky Rosen paradox*, **Physics Physique Fizika** 1, 195–200 (1964).
* **CHSH inequality:**
  J. F. Clauser, M. A. Horne, A. Shimony, R. A. Holt, *Proposed Experiment to Test Local Hidden-Variable Theories*, **Phys. Rev. Lett.** 23, 880–884 (1969).
* **Tsirelson bound (a.k.a. Cirel'son):**
  B. S. Cirel'son (Tsirelson), *Quantum generalizations of Bell's inequality*, **Lett. Math. Phys.** 4, 93–100 (1980).
* **Kochen–Specker theorem (contextuality foundation):**
  S. Kochen, E. P. Specker, *The Problem of Hidden Variables in Quantum Mechanics*, **J. Math. Mech.** 17, 59–87 (1967).
* **KCBS inequality:**
  A. A. Klyachko, M. A. Can, S. Binicioğlu, A. S. Shumovsky, *Simple Test for Hidden Variables in Spin-1 Systems*, **Phys. Rev. Lett.** 101, 020403 (2008).
* **Mermin–Peres magic square (state-independent contextuality):**
  N. D. Mermin, *Simple Unified Form for the Major No-Hidden-Variables Theorems*, **Phys. Rev. Lett.** 65, 3373–3376 (1990).
  A. Peres, *Two Simple Proofs of the Kochen–Specker Theorem*, **J. Phys. A: Math. Gen.** 24, L175–L178 (1991).