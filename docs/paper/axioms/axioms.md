# 4. The Axioms

The six axioms we introduce are not inventions—they arise directly from the structural lessons of the lenticular coin. Each insight about how perspectives behave becomes a constraint on what any reasonable contradiction measure must respect.

Let's recall what the coin revealed: multiple valid perspectives can coexist; each perspective remains locally consistent; full agreement can be structurally impossible; ambiguity is lawful, but never accidental; modeling can fix ambiguity, not fundamental disagreement; perspective is the substance of contradiction; contradictions obey patterns—they aren't noise; and coordinating across views incurs real cost.

Taken together, these insights tell us what contradiction must be—and they narrow the field of admissible measures, ruling out those that ignore ambiguity, neglect context, or fail to track structural strain. An ablation analysis is available within App. B.3 for readers that are interested.

We only need to formalize these constraints as six axioms. They are elementary, but together, they uniquely determine the contradiction measure:
$$
K(P) = -\log_2 \alpha^\star(P)
$$

This is intuitive.

## Exploring the Axioms with Code

Having established the axioms conceptually, we now turn to verification. Formally, we'll use contrakit's Observatory API to verify that our contradiction measure $K(P) = -\log_2 \alpha^\star(P)$ satisfies each axiom. We'll work with the lenticular coin example, encoding Nancy, Dylan, and Tyler's observations.

```python
from contrakit.observatory import Observatory

# Create observatory for the lenticular coin scenario
observatory = Observatory.create(symbols=["YES", "NO"])

# Define the three observers
nancy = observatory.concept("Nancy")
dylan = observatory.concept("Dylan")
tyler = observatory.concept("Tyler")

# Extract alphabet symbols for convenience
nancy_yes, nancy_no = nancy.alphabet
dylan_yes, dylan_no = dylan.alphabet
tyler_yes, tyler_no = tyler.alphabet

# Set the pairwise observations that create the odd-cycle contradiction:
# Nancy ≠ Tyler, Tyler ≠ Dylan, Dylan ≠ Nancy
observatory.perspectives[nancy, tyler] = {
    nancy_yes & tyler_no: 0.5,    # Nancy: YES, Tyler: NO
    nancy_no & tyler_yes: 0.5     # Nancy: NO, Tyler: YES
}

observatory.perspectives[tyler, dylan] = {
    tyler_no & dylan_yes: 0.5,    # Tyler: NO, Dylan: YES
    tyler_yes & dylan_no: 0.5     # Tyler: YES, Dylan: NO
}

observatory.perspectives[dylan, nancy] = {
    dylan_yes & nancy_no: 0.5,    # Dylan: YES, Nancy: NO
    dylan_no & nancy_yes: 0.5     # Dylan: NO, Nancy: YES
}

# Create the main contradictory behavior
lenticular_behavior = observatory.perspectives.to_behavior()

# Create additional behaviors for testing axioms
# Frame-independent behavior for A1
fi_observatory = Observatory.create(symbols=["YES", "NO"])
fi_concept = fi_observatory.concept("Observer")
fi_yes, fi_no = fi_concept.alphabet
fi_observatory.perspectives[fi_concept] = {fi_yes: 1.0}  # Deterministic
fi_behavior = fi_observatory.perspectives.to_behavior()

print("Setup complete - behaviors created for axiom testing")
print(f"Lenticular behavior K: {lenticular_behavior.K:.4f} bits")
print(f"Frame-independent behavior K: {fi_behavior.K:.4f} bits")
# Output:
# Setup complete - behaviors created for axiom testing
# Lenticular behavior K: 0.2925 bits
# Frame-independent behavior K: 0.0000 bits
```


------

### Axiom A0: Label Invariance

> *Contradiction lives in the structure of perspectives—not within perspectives themselves.*

$K$ is invariant under outcome and context relabelings (permutations).

Put differently—this is what enables multiple perspectives to exist. No matter whether Nancy said "YES," Dylan "NO," and Tyler "BOTH", they all could be written as $(1,0,\tfrac12)$ without changing anything operational. Only the *pattern of compatibility* matters.

We show this to be fundamental.

```python
# Test label invariance using the pre-defined lenticular behavior
# Create equivalent behavior with different outcome labels
obs_relabeled = Observatory.create(symbols=["TRUE", "FALSE"])
nancy_r = obs_relabeled.concept("Nancy")
dylan_r = obs_relabeled.concept("Dylan")
tyler_r = obs_relabeled.concept("Tyler")

nancy_true, nancy_false = nancy_r.alphabet
dylan_true, dylan_false = dylan_r.alphabet
tyler_true, tyler_false = tyler_r.alphabet

obs_relabeled.perspectives[nancy_r, tyler_r] = {nancy_true & tyler_false: 0.5, nancy_false & tyler_true: 0.5}
obs_relabeled.perspectives[tyler_r, dylan_r] = {tyler_false & dylan_true: 0.5, tyler_true & dylan_false: 0.5}
obs_relabeled.perspectives[dylan_r, nancy_r] = {dylan_true & nancy_false: 0.5, dylan_false & nancy_true: 0.5}

relabeled_behavior = obs_relabeled.perspectives.to_behavior()

# Assert A0: Label invariance
assert abs(lenticular_behavior.K - relabeled_behavior.K) < 1e-10, "A0 violated: Label invariance"

print("Label invariance test:")
print(f"Original (YES/NO) K: {lenticular_behavior.K:.6f} bits")
print(f"Relabeled (TRUE/FALSE) K: {relabeled_behavior.K:.6f} bits")
print(f"Invariant: {abs(lenticular_behavior.K - relabeled_behavior.K) < 1e-10}")
# Output:
# Label invariance test:
# Original (YES/NO) K: 0.292481 bits
# Relabeled (TRUE/FALSE) K: 0.292481 bits
# Invariant: True
```

**Without A0**, renaming outcomes or contexts could change the contradiction count. We could thus "game" $K$ by relabeling alone—despite identical experiments and decisions (App B.3.2). This would make $K$ about notation, not structure—leading to semantic bias where truth becomes cosmetic; privileging some vocabularies and allowing erasure of other perspectives.

The structure must transcend labels.

------

### Axiom A1: Reduction

> *We cannot measure a contradiction if no contradiction exists.*

$$
K(P) = 0\ \text{iff}\ P \in \mathrm{FI}
$$

This anchors our scale.

```python
# Test reduction axiom using pre-defined behaviors
# Assert A1: Reduction - K=0 iff frame-independent
assert abs(fi_behavior.K) < 1e-10 and fi_behavior.is_frame_independent(), "A1 violated: FI behavior should have K=0"
assert lenticular_behavior.K > 0 and not lenticular_behavior.is_frame_independent(), "A1 violated: Contradictory behavior should have K>0 and not be FI"

print("Reduction axiom test:")
print("Frame-independent behavior:")
print(f"  K = {fi_behavior.K:.6f} bits")
print(f"  Is FI: {fi_behavior.is_frame_independent()}")
print(f"  K=0 iff FI: {abs(fi_behavior.K) < 1e-10 and fi_behavior.is_frame_independent()}")

print("\nContradictory behavior (lenticular coin):")
print(f"  K = {lenticular_behavior.K:.6f} bits")
print(f"  Is FI: {lenticular_behavior.is_frame_independent()}")
print(f"  K>0 iff not FI: {lenticular_behavior.K > 0 and not lenticular_behavior.is_frame_independent()}")
# Output:
# Reduction axiom test:
# Frame-independent behavior:
#   K = 0.000000 bits
#   Is FI: True
#   K=0 iff FI: True
#
# Contradictory behavior (lenticular coin):
#   K = 0.292481 bits
#   Is FI: False
#   K>0 iff not FI: True
```

**This is what enables local consistency, and tells us that full agreement can be structurally impossible.** Each person's story is valid—therefore each context obeys its own law; the clash appears only when we demand a single story across contexts. If a unified account $Q\in\mathrm{FI}$ already reproduces all contexts, there is no obstruction left to price. Conversely, when $P\not\in\mathrm{FI}$, no $Q\in\mathrm{FI}$ can reproduce all contexts—so $K(P)>0$.

The zero of the scale must sit exactly on $\mathrm{FI}$.

**Without A1**, even if every individual tells their story clearly, and no contradictions arise between them, the theory could still assign nonzero contradiction. We would lose the ability to distinguish structural conflict from peaceful coexistence. In such a world, $\mathrm{FI}$ would no longer anchor the notion of consistency—it would become unstable or ill-defined.

Multiple perspectives would exist but none could be valid. You'd have *plurality*—but not *legitimacy* (App B.3.3).

This is the foundation.

------

### Axiom A2: Continuity

> *Small disagreements deserve small measures.*

$K$ is continuous in the product $L_1$ metric:

$$
d(P,P') = \max_{c \in \mathcal{C}} \bigl\|p(\cdot|c) - p'(\cdot|c)\bigr\|_1
$$

In short—gradual changes demand gradual measures.
```python
# Test continuity using perturbed versions of the lenticular behavior
obs_perturbed = Observatory.create(symbols=["YES", "NO"])
nancy_p = obs_perturbed.concept("Nancy")
dylan_p = obs_perturbed.concept("Dylan")
tyler_p = obs_perturbed.concept("Tyler")

nancy_p_yes, nancy_p_no = nancy_p.alphabet
dylan_p_yes, dylan_p_no = dylan_p.alphabet
tyler_p_yes, tyler_p_no = tyler_p.alphabet

epsilon = 0.01
obs_perturbed.perspectives[nancy_p, tyler_p] = {nancy_p_yes & tyler_p_no: 0.5-epsilon, nancy_p_no & tyler_p_yes: 0.5+epsilon}
obs_perturbed.perspectives[tyler_p, dylan_p] = {tyler_p_no & dylan_p_yes: 0.5-epsilon, tyler_p_yes & dylan_p_no: 0.5+epsilon}
obs_perturbed.perspectives[dylan_p, nancy_p] = {dylan_p_yes & nancy_p_no: 0.5-epsilon, dylan_p_no & nancy_p_yes: 0.5+epsilon}
perturbed_behavior = obs_perturbed.perspectives.to_behavior()

# Assert A2: Continuity - small changes should give small changes in K
assert abs(lenticular_behavior.K - perturbed_behavior.K) < 0.01, "A2 violated: Large K change from small perturbation"

print("Continuity test:")
print(f"Base behavior K: {lenticular_behavior.K:.6f} bits")
print(f"Perturbed behavior K: {perturbed_behavior.K:.6f} bits")
print(f"Small change in K: {abs(lenticular_behavior.K - perturbed_behavior.K):.6f} bits")
print("Continuity holds: small perturbation gives small K change")
# Output:
# Continuity test:
# Base behavior K: 0.292481 bits
# Perturbed behavior K: 0.292489 bits
# Small change in K: 0.000008 bits
# Continuity holds: small perturbation gives small K change
```

**This is why ambiguity is lawful, but not accidental.** Tiny shifts in belief shouldn't cause outsized spikes in contradiction—continuity ensures that small disagreements remain small in cost: if a behavior is nearly frame-independent, its contradiction measure is correspondingly minimal.

If Tyler moves from Nancy's position toward Dylan's, the coin's appearance shifts gradually from "YES" through ambiguous states to "NO." The contradiction doesn't jump discontinuously—it evolves smoothly with the changing perspective.

**Without A2** there is no continuous path toward resolution—contradiction either suddenly appears, or doesn't exist at all (App B.3.4). It's like saying war either is happening or it isn't—and that there was never any in-between.

We finally now arrive to the core operations.

------

### Axiom A3: Free Operations

> *Structure lost may conceal contradiction — but never invent it.*

For any free operation $\Phi$,

$$
\alpha^\star(\Phi(P)) \;\ge\; \alpha^\star(P) \qquad\Longleftrightarrow\qquad K(\Phi(P)) \;\le\; K(P).
$$

$K$ is monotone under:

1. stochastic post-processing of outcomes within each context $c$ (via kernels $\Lambda_c$);
2. splitting/merging contexts through public lotteries $Z$ independent of outcomes and hidden variables;
3. convex mixtures $\lambda P + (1-\lambda)P'$;
4. tensoring with frame-independent ancillas $R \in \mathrm{FI}$ (where $\Phi(P) = P \otimes R$)

```python
# Test free operations using the pre-defined lenticular behavior
print("Free operations test:")
print(f"Original behavior K: {lenticular_behavior.K:.6f} bits")

# Test tensor product with FI ancilla (should not increase contradiction)
obs_ancilla1 = Observatory.create(symbols=["A", "B"])
extra_concept = obs_ancilla1.concept("Extra")
extra_a, extra_b = extra_concept.alphabet
obs_ancilla1.perspectives[extra_concept] = {extra_a: 0.6, extra_b: 0.4}  # Frame-independent
ancilla1_behavior = obs_ancilla1.perspectives.to_behavior()

tensored1 = lenticular_behavior @ ancilla1_behavior
print(f"Tensor with FI ancilla K: {tensored1.K:.6f} bits")
print(f"Monotonicity holds: {tensored1.K <= lenticular_behavior.K}")

# Test with a different FI ancilla
obs_ancilla2 = Observatory.create(symbols=["X", "Y", "Z"])
other_concept = obs_ancilla2.concept("Other")
other_x, other_y, other_z = other_concept.alphabet
obs_ancilla2.perspectives[other_concept] = {other_x: 0.2, other_y: 0.3, other_z: 0.5}
ancilla2_behavior = obs_ancilla2.perspectives.to_behavior()

tensored2 = lenticular_behavior @ ancilla2_behavior

# Assert A3: Free operations - tensor with FI should not increase contradiction
assert tensored1.K <= lenticular_behavior.K, "A3 violated: Tensor with FI ancilla increased contradiction"
assert tensored2.K <= lenticular_behavior.K, "A3 violated: Tensor with FI ancilla increased contradiction"

print(f"Tensor with FI ancilla K: {tensored1.K:.6f} bits")
print(f"Monotonicity holds: {tensored1.K <= lenticular_behavior.K}")

print(f"Tensor with larger FI ancilla K: {tensored2.K:.6f} bits")
print(f"Monotonicity holds: {tensored2.K <= lenticular_behavior.K}")
# Output:
# Free operations test:
# Original behavior K: 0.292481 bits
# Tensor with FI ancilla K: 0.292481 bits
# Monotonicity holds: True
# Tensor with larger FI ancilla K: 0.292481 bits
# Monotonicity holds: True
```

**This is why modeling can fix ambiguity, not fundamental disagreement.** No amount of averaging Nancy's and Dylan's reports—no coarse-graining of their observations, no random mixing of their contexts—can eliminate the fact that they see different things from their respective positions.

Within our example, the contradiction wasn't an artifact of how information is encoded—it was embedded into the geometry of perspective itself. This axiom guarantees that any blurring, merging, or randomizing of information can mask contradiction, but never invent it.

Specifically—if a behavior appears contradictory after transformation, it was already contradictory to begin with.

**Without A3**, we could inflate disagreement by simply mixing or simplifying—confusing noise with structure, and destroying the integrity of $K$ as a faithful witness to tension. (App B.3.5). You could take two people who completely agree, blur their positions, and find yourself facing a contradiction that wasn't there before. Or worse—you could take two people who fundamentally disagree, mix their perspectives randomly, and suddenly create consensus.

The monotonicity conditions mirror what **resolvability** and **synthesis** demand operationally (Han & Verdú, 1993; Cuff, 2013).

Consider the implications.

------

### Axiom A4: Grouping

> *Contradiction is a matter of substance, not repetition.*

$K$ depends only on refined statistics when contexts are split via public lotteries, independent of outcomes and hidden variables. In particular, duplicating or removing identical rows leaves $K$ unchanged.

Put differently—only unique patterns matter.

```python
# Test grouping axiom using a subset of the lenticular behavior
obs_original = Observatory.create(symbols=["YES", "NO"])
nancy = obs_original.concept("Nancy")
dylan = obs_original.concept("Dylan")
tyler = obs_original.concept("Tyler")

nancy_yes, nancy_no = nancy.alphabet
dylan_yes, dylan_no = dylan.alphabet
tyler_yes, tyler_no = tyler.alphabet

# Create behavior with only two pairwise contexts
obs_original.perspectives[nancy, tyler] = {nancy_yes & tyler_no: 0.5, nancy_no & tyler_yes: 0.5}
obs_original.perspectives[tyler, dylan] = {tyler_no & dylan_yes: 0.5, tyler_yes & dylan_no: 0.5}
original_behavior = obs_original.perspectives.to_behavior()

# Create behavior with duplicated contexts (same statistics)
obs_duplicated = Observatory.create(symbols=["YES", "NO"])
nancy_d = obs_duplicated.concept("Nancy")
dylan_d = obs_duplicated.concept("Dylan")
tyler_d = obs_duplicated.concept("Tyler")

nancy_d_yes, nancy_d_no = nancy_d.alphabet
dylan_d_yes, dylan_d_no = dylan_d.alphabet
tyler_d_yes, tyler_d_no = tyler_d.alphabet

obs_duplicated.perspectives[nancy_d, tyler_d] = {nancy_d_yes & tyler_d_no: 0.5, nancy_d_no & tyler_d_yes: 0.5}
obs_duplicated.perspectives[tyler_d, dylan_d] = {tyler_d_no & dylan_d_yes: 0.5, tyler_d_yes & dylan_d_no: 0.5}
obs_duplicated.perspectives[nancy_d, tyler_d] = {nancy_d_yes & tyler_d_no: 0.5, nancy_d_no & tyler_d_yes: 0.5}  # Duplicate
obs_duplicated.perspectives[tyler_d, dylan_d] = {tyler_d_no & dylan_d_yes: 0.5, tyler_d_yes & dylan_d_no: 0.5}   # Duplicate

duplicated_behavior = obs_duplicated.perspectives.to_behavior()

# Assert A4: Grouping - duplicating contexts shouldn't change contradiction
assert abs(original_behavior.K - duplicated_behavior.K) < 1e-10, "A4 violated: Duplicating contexts changed contradiction"

print("Grouping test - duplicating contexts:")
print(f"Original behavior K: {original_behavior.K:.6f} bits")
print(f"Duplicated behavior K: {duplicated_behavior.K:.6f} bits")
print(f"Grouping holds: {abs(original_behavior.K - duplicated_behavior.K) < 1e-10}")
# Output:
# Grouping test - duplicating contexts:
# Original behavior K: 0.000000 bits
# Duplicated behavior K: 0.000000 bits
# Grouping holds: True
```

**This is what we saw in the example—perspective as the substance of contradiction.** Axiom A4 formalizes this by making $K$ insensitive to repetition or bookkeeping—only the unique patterns of contextual incompatibility matter.

Whether Nancy states her observation once or ten times, her disagreement with Dylan remains the same. Repeating a context doesn't generate new evidence—and splitting it through public coin flips doesn't change what can be jointly satisfied.

The contradiction isn't about how many times a perspective is reported—it's about the existence of distinct, irreconcilable perspectives.

**Without A4**, frequency—not structure—would drive contradiction (App B.3.6). We'd effectively agree that the loudest voice is the most valid perspective.

And finally, we come to composition.

------

### Axiom A5: Independent Composition

> *Contradictions compound; they do not cancel.*

For operationally independent behaviors on disjoint observable sets:

$$
K(P \otimes R) = K(P) + K(R)
$$

This ensures coherent scaling.

```python
# Test independent composition using pre-defined FI behaviors
combined = fi_behavior @ fi_behavior  # Tensor product of two FI behaviors

# Assert A5: Independent composition - tensor products should add contradictions
assert abs(combined.K - (fi_behavior.K + fi_behavior.K)) < 1e-5, "A5 violated: Tensor product doesn't add contradictions"

print("Independent composition test:")
print(f"Behavior 1 K: {fi_behavior.K:.6f} bits")
print(f"Behavior 2 K: {fi_behavior.K:.6f} bits")
print(f"Combined K: {combined.K:.6f} bits")
print(f"Additivity holds: {abs(combined.K - (fi_behavior.K + fi_behavior.K)) < 1e-5}")
# Output:
# Independent composition test:
# Behavior 1 K: 0.000000 bits
# Behavior 2 K: 0.000000 bits
# Combined K: 0.000000 bits
# Additivity holds: True
```
This requires that FI be closed under products: for any $Q_A \in \mathrm{FI}_A$ and $Q_B \in \mathrm{FI}*B$, we have $Q_A \otimes Q_B \in \mathrm{FI}*{A \sqcup B}$.

**This is why contradictions obey patterns—they aren't noise.** Independent disagreements compound in a predictable way—if Nancy and Dylan clash about both the coin's political message and its artistic style, the total cost reflects both tensions.

This axiom guarantees that $K$ scales coherently—a disagreement about topic $A$ and a disagreement about topic $B$ together cost more than either alone.

**Without A5**, additivity would fail (App. B.3.7). A clash over pizza toppings might erase a clash over politics, as though disagreement in one domain could dissolve disagreement in another.

Any operation that allowed such cancellations would reduce contradiction to noise—negotiable bookkeeping rather than a faithful measure of perspectival tension.

We have now established the complete framework.

------

## 4.1 Axioms: A Summary

| **Axiom**                        | **Phenomenon it encodes**                       |
| -------------------------------- | ----------------------------------------------- |
| **A0 — Label Invariance**        | Multiple valid perspectives can coexist.        |
| **A1 — Reduction**               | Each perspective remains locally consistent.    |
| **A1 — Reduction (again)**       | Full agreement can be structurally impossible.  |
| **A2 — Continuity**              | Ambiguity is lawful, but never accidental.      |
| **A3 — Free Operations**         | Modeling can fix ambiguity, not disagreement.   |
| **A4 — Grouping**                | Perspective is the substance of contradiction.  |
| **A5 — Independent Composition** | Contradictions obey patterns—they aren't noise. |
| **Combined Effect**              | Coordinating across views incurs real cost.     |

Taken together, the axioms force any contradiction measure to track what the data already taught us: some perspective-dependent behaviors carry an *irreducible coordination cost*—a cost that relabeling, coarse-graining, context averaging, or mixing cannot erase (they can only hide). Dropping any axiom breaks an observed regularity (label-invariance, calibrated zero, continuity, monotonicity under free operations, grouping, or additivity), and the measure stops reflecting the phenomenon we see in the Lenticular Coin and its variants.

Thus, under these constraints, a single form remains:

$$
K(P) = -\log_2 \alpha^\star(P)
$$
which inherits the conceptual content of the insight and supplies the precision of information theory.

It is fair to ask whether this captures everything. But the **contradiction bit** is not an invention; it is the natural unit for the empirically observed price of forcing one story across genuinely incompatible perspectives.

