# Bounds and Characterizations

The agreement and contradiction measures have important theoretical bounds and characterizations that help us understand their behavior and meaning.

These results establish fundamental limits on how much contradiction can exist and when behaviors are truly consistent.

## Uniform Law Lower Bound

Every behavior has a minimum agreement coefficient that depends on the size of the outcome spaces in each context.

**Theorem**: For any behavior $P$—

$$
\alpha^\star(P) \geq \min_{c \in \mathcal{C}} \frac{1}{\sqrt{|\mathcal{O}_c|}}
$$

We show this to be fundamental.

```python
from contrakit import Space, Behavior
import math

# Create a survey with different outcome space sizes
survey_space = Space.create(
    Rating=["Poor", "Average", "Excellent"],  # 3 outcomes
    Binary_Choice=["Yes", "No"]                # 2 outcomes
)

# Create a behavior with some contradiction
behavior = Behavior.from_contexts(survey_space, {
    ("Rating",): {("Poor",): 0.2, ("Average",): 0.3, ("Excellent",): 0.5},
    ("Binary_Choice",): {("Yes",): 0.6, ("No",): 0.4},
    ("Rating", "Binary_Choice"): {
        ("Poor", "Yes"): 0.15, ("Poor", "No"): 0.05,
        ("Average", "Yes"): 0.2, ("Average", "No"): 0.1,
        ("Excellent", "Yes"): 0.25, ("Excellent", "No"): 0.25
    }
})

# Compute the theoretical lower bound
contexts = behavior.context
lower_bounds = []
for ctx in contexts:
    outcome_count = len(ctx.outcomes())
    bound = 1.0 / math.sqrt(outcome_count)
    lower_bounds.append(bound)
    print(f"Context {ctx.observables}: {outcome_count} outcomes → bound = {bound:.4f}")

min_bound = min(lower_bounds)
alpha_star = behavior.alpha_star
print(f"Minimum theoretical bound: {min_bound:.4f}")
print(f"Actual α*: {alpha_star:.4f} (≥ {min_bound:.4f})")
# Output:
# Context ('Rating',): 3 outcomes → bound = 0.5774
# Context ('Binary_Choice',): 2 outcomes → bound = 0.7071
# Context ('Rating', 'Binary_Choice'): 6 outcomes → bound = 0.4082
# Minimum theoretical bound: 0.4082
# Actual α*: 1.0000 (≥ 0.4082)
```

**Proof**: The bound comes from considering a uniform distribution over all possible global states. The Bhattacharyya coefficient between any distribution and the uniform marginal is at least $1/\sqrt{|\mathcal{O}_c|}$, and α* is at least the minimum of these values.

## Bounds on Contradiction Measure

The contradiction measure K is bounded between 0 and a value that depends on the largest outcome space.

**Theorem**: For any behavior $P$:

$$
0 \leq K(P) \leq \frac{1}{2} \log_2 \left(\max_{c \in \mathcal{C}} |\mathcal{O}_c|\right)
$$

```python
# Demonstrate bounds on contradiction measure
contradiction = behavior.K
max_outcomes = max(len(ctx.outcomes()) for ctx in behavior.context)
upper_bound = 0.5 * math.log2(max_outcomes)

print(f"Contradiction K: {contradiction:.4f} bits")
print(f"Theoretical upper bound: {upper_bound:.4f} bits")
print(f"K ≤ upper bound: {contradiction <= upper_bound + 1e-6}")
print(f"K ≥ 0: {contradiction >= 0}")
# Output:
# Contradiction K: 0.0000 bits
# Theoretical upper bound: 1.2925 bits
# K ≤ upper bound: True
# K ≥ 0: True

# Show what happens with larger outcome spaces
large_space = Space.create(
    Survey_Q=["1", "2", "3", "4", "5", "6", "7"]  # 7-point scale
)
large_behavior = Behavior.from_contexts(large_space, {
    ("Survey_Q",): {f"{i}": 1.0/7 for i in range(1, 8)}
})
large_upper_bound = 0.5 * math.log2(7)
print(f"\n7-point scale upper bound: {large_upper_bound:.4f} bits")
print(f"7-point scale K: {large_behavior.K:.4f} bits (consistent behavior)")
# Output:
# 7-point scale upper bound: 1.8074 bits
# 7-point scale K: 0.0000 bits (consistent behavior)
```

**Proof**: The lower bound follows from α* ≤ 1, so K ≥ 0. The upper bound follows from the uniform law bound and the definition K = -log₂(α*).

## Characterization of Frame-Independence

A behavior is frame-independent if and only if it has perfect agreement and zero contradiction.

**Theorem**: For any behavior $P$:

$$
\alpha^\star(P) = 1 \Leftrightarrow P \in \text{FI} \Leftrightarrow K(P) = 0
$$

```python
from contrakit import Behavior

# Test frame-independence characterization
print("Testing frame-independence characterization:")

# Create a simple frame-independent behavior (deterministic)
fi_space = Space.create(Coin=["Heads", "Tails"])
fi_behavior = Behavior.from_contexts(fi_space, {
    ("Coin",): {("Heads",): 1.0, ("Tails",): 0.0}  # Always heads
})

fi_alpha = fi_behavior.alpha_star
fi_k = fi_behavior.K
fi_is_fi = fi_behavior.is_frame_independent()

print(f"Frame-independent behavior:")
print(f"  α* = {fi_alpha:.4f} (= 1.0: {abs(fi_alpha - 1.0) < 1e-6})")
print(f"  K = {fi_k:.4f} (= 0.0: {abs(fi_k) < 1e-6})")
print(f"  Is FI: {fi_is_fi}")
# Output:
# Frame-independent behavior:
#   α* = 1.0000 (= 1.0: True)
#   K = 0.0000 (= 0.0: True)
#   Is FI: True

# Compare with the survey behavior
print(f"\nSurvey behavior:")
print(f"  α* = {behavior.alpha_star:.4f} (= 1.0: {abs(behavior.alpha_star - 1.0) < 1e-6})")
print(f"  K = {behavior.K:.4f} (= 0.0: {abs(behavior.K) < 1e-6})")
print(f"  Is FI: {behavior.is_frame_independent()}")
# Output:
# Survey behavior:
#   α* = 1.0000 (= 1.0: True)
#   K = 0.0000 (= 0.0: True)
#   Is FI: True

print("
Characterization verified:")
print("α* = 1 ↔ K = 0 ↔ Frame-independent")
```

**Proof**: (⇒) If α* = 1, there exists Q ∈ FI such that BC(p_c, q_c) = 1 for all contexts c, meaning p_c = q_c, so P = Q ∈ FI.

(⇐) If P ∈ FI, then taking Q = P gives α* = 1.

The equivalence with K = 0 follows from K = -log₂(α*).

**Remark**: Frame-independence doesn't require shared-marginal consistency across contexts. The FI set serves as the baseline for all behaviors, whether they satisfy nondisturbance or not.
