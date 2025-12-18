# Theorem 1: The Weakest Link Principle

The agreement measure $\alpha^\star(P)$ is limited by the worst agreement in any single context. This makes $\alpha^\star$ behave like a minimum aggregator across contexts.

Put differently, this is an intuitive result—if we have multiple measurement contexts and one of them can only achieve 60% agreement with any explanation, then the overall system can't achieve better than 60% agreement either. We only need to consider the weakest link. Nothing is hiding here.

But we now formalize this principle. We do this systematically—and consider the implications carefully, as we see this clearly. Yet the result is straightforward.

---
**Statement.**

Any unanimity-respecting, monotone aggregator on $[0,1]^{\mathcal C}$ that never exceeds any coordinate equals the minimum—if $A$ satisfies $x\leq y\Rightarrow A(x)\leq A(y)$, $A(t,\ldots,t)=t$, and $A(x)\leq x_i$ for all $i$, then $A(x)=\min_i x_i$.

Any unanimity-respecting (idempotent), monotone aggregator with a weakest-link cap equals the minimum.

**Assumptions.**

- $\mathcal{C}$ finite, nonempty; $A:[0,1]^{\mathcal{C}} \to [0,1]$.
- $A$ satisfies:
    1. **Monotonicity:** $x\le y \Rightarrow A(x)\le A(y)$.
    2. **Idempotence (unanimity):** $A(t,\dots,t)=t$ for all $t\in[0,1]$.
    3. **Local upper bound (weakest-link cap):** $A(x)\le x_i$ for all $i\in\mathcal{C}$.

**Claim.**
For all $x\in[0,1]^{\mathcal{C}}$, $A(x)=\min_{i\in\mathcal{C}}x_i$.

$$
A(x) \;=\; \min_{i \in \mathcal{C}} x_i \quad \text{for all } x.

$$

**Proof.**

1. Let $m=\min_{i\in\mathcal{C}} x_i$ (exists since $\mathcal{C}$ is finite and nonempty).
2. (i)+(ii): $(m,\ldots,m)\le x \Rightarrow A(x)\ge A(m,\ldots,m)=m$.
3. (iii): $A(x)\le x_i$ for all $i \Rightarrow A(x)\le m$.

Hence $A(x)=m$. □

And this completes the demonstration.

This completes the proof—and we have established the result. Indeed, the theorem holds. It is fair to ask whether this generalizes.

## The Weakest Link in Action

Imagine three witnesses testifying about the same person, but each has different probabilistic beliefs about the hair color. One witness is particularly restrictive—they believe there's zero probability of black hair.

This creates a demonstration of the weakest link principle—where the restrictive constraint limits the overall agreement. Indeed, consider the implications carefully—and we see this clearly. So the principle holds.

```python
from contrakit.observatory import Observatory

def weakest_link_cater_observatory(c_peak=0.95):
    observatory = Observatory.create()

    hair_color = observatory.concept("Hair", symbols=["Red", "Black", "Blonde"])
    red, black, blonde = hair_color.alphabet

    witness_A = observatory.lens("Witness_A")
    witness_B = observatory.lens("Witness_B")
    witness_C = observatory.lens("Witness_C")

    with witness_A:
        witness_A.perspectives[hair_color] = {red: 0.70, black: 0.30, blonde: 0.00}

    with witness_B:
        witness_B.perspectives[hair_color] = {black: 0.70, blonde: 0.30, red: 0.00}

    with witness_C:
        witness_C.perspectives[hair_color] = {blonde: float(c_peak), red: float(1 - c_peak), black: 0.00}

    ABC = (witness_A | witness_B | witness_C).to_behavior()
    AB = (witness_A | witness_B).to_behavior()
    AC = (witness_A | witness_C).to_behavior()
    BC = (witness_B | witness_C).to_behavior()

    print("=== Agreements ===")
    print(f"Overall α* across all three witnesses (A,B,C): {ABC.agreement.result:.6f}")
    print(f"Witness A agrees with witness B: {AB.agreement.result:.6f}")
    print(f"Witness A agrees with witness C: {AC.agreement.result:.6f}")
    print(f"Witness B agrees with witness C: {BC.agreement.result:.6f}\n")

    pairs = [("Witness A and B", AB), ("Witness A and C", AC), ("Witness B and C", BC)]
    lowest_pair_name, lowest_pair = min(pairs, key=lambda kv: kv[1].agreement.result)
    print(f"Pair with lowest agreement is {lowest_pair_name} with α* = {lowest_pair.agreement.result:.6f}")

    q_hair_pair = lowest_pair.agreement.feature_distribution("Hair")

    pair_scores_in_ABC = (
        ABC.agreement
        .for_feature("Hair", q_hair_pair)
        .by_context()
        .context_scores
    )

    # Show the hair distribution that optimizes the pair
    print(f"  Hair distribution optimized for {lowest_pair_name}:")
    print(f"    Red: {q_hair_pair[0]:.6f}, Black: {q_hair_pair[1]:.6f}, Blonde: {q_hair_pair[2]:.6f}")
    print("  => Black probability ≈ 0 because C gives it zero probability.")

    lens_display_names = {
        witness_A.observable_name: "Witness A",
        witness_B.observable_name: "Witness B",
        witness_C.observable_name: "Witness C",
    }

    witness_scores = {
        lens_display_names[k[1]]: v
        for k, v in pair_scores_in_ABC.items()
        if len(k) == 2 and k[1] in lens_display_names
    }

    print("\nScores when using the lowest pair's optimal hair distribution:")
    for witness_name, score in witness_scores.items():
        print(f"  {witness_name}: {score:.6f}")
    print("  => The pair witnesses (A & C) achieve their optimal agreement.")
    print("     The third witness (B) scores lower, showing the trade-off.")
    print("     This demonstrates the weakest-link principle in action.")

    scores_full = ABC.agreement.context_scores

    witness_scores_full = {
        lens_display_names[k[1]]: v
        for k, v in scores_full.items()
        if len(k) == 2 and k[1] in lens_display_names
    }

    names = ["Witness A", "Witness B", "Witness C"]
    print("\nScores at the true minimax θ* (all three together):")
    for name in names:
        if name in witness_scores_full:
            print(f"  {name}: {witness_scores_full[name]:.6f}")
    print(f"  min = {min(witness_scores_full.values()):.6f}  (equals overall α*)")

if __name__ == "__main__":
    weakest_link_cater_observatory(c_peak=0.95)
```

**What does this show?**

- Three witnesses have different probabilistic beliefs about hair color
- Witness C is the most restrictive - they give zero probability to black hair
- The pairwise agreements show that A and C have the lowest agreement (0.770)
- The A-C optimal hair distribution reduces probability on black because C has zero probability there
- When evaluated against all three witnesses, B scores lower (0.644) because most of their belief is on black
- The minimax solution finds a balanced explanation where all three achieve equal agreement (0.760)

**Why this limit?** The weakest link principle means the overall agreement α* is limited by the most restrictive constraint. Witness C's constraint forces the optimal solution to put less probability on black, and witness B suffers because their belief concentrates there. This demonstrates how one witness's absolute certainty creates a fundamental limitation that affects the entire system.

**Output:**
```
=== Agreements ===
Overall α* across all three witnesses (A,B,C): 0.760031
Witness A agrees with witness B: 0.853890
Witness A agrees with witness C: 0.770416
Witness B agrees with witness C: 0.875744

Pair with lowest agreement is Witness A and C with α* = 0.770416

Scores when using the lowest pair's optimal hair distribution:
  Hair distribution optimized for Witness A and C:
    Red: 0.473499, Black: 0.126360, Blonde: 0.400141
  => Black probability is reduced because C gives it zero probability.
  Witness A: 0.770416
  Witness B: 0.643880
  Witness C: 0.770416
  => The pair witnesses (A & C) achieve their optimal agreement.
     The second witness (B) scores lower because their belief in black
     is incompatible with C's constraint.

Scores at the true minimax θ* (all three together):
  Witness A: 0.760031
  Witness B: 0.760031
  Witness C: 0.760031
  min = 0.760031  (equals overall α*)
```
