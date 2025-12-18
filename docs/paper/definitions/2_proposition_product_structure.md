
# Product Structure

Behaviors from independent systems can be combined through a **tensor product** operation. Think of it like combining separate surveys about different topics—the joint behavior is just the product of the individual probabilities.

Put differently—this captures independence.

## Definition and Properties

Let $P$ be a behavior on $(\mathcal{X}, \mathcal{C})$ and $R$ be a behavior on $(\mathcal{Y}, \mathcal{D})$ with disjoint observables ($\mathcal{X} \cap \mathcal{Y} = \emptyset$).

```python
from contrakit import Space, Behavior

# Create two independent survey systems about different topics
food_space = Space.create(
    Taste=["Delicious", "Bland"],
    Presentation=["Beautiful", "Plain"]
)

service_space = Space.create(
    Friendliness=["Friendly", "Rude"],
    Speed=["Fast", "Slow"]
)
```

For distributions $p$ and $r$ on disjoint coordinates, their tensor product $p \otimes r$ is defined by:

$$
(p \otimes r)(o_c, o_d) = p(o_c) \cdot r(o_d)
$$

We show this to be multiplicative.

```python
# Define behaviors for each system
food_behavior = Behavior.from_contexts(food_space, {
    ("Taste",): {("Delicious",): 0.8, ("Bland",): 0.2},
    ("Presentation",): {("Beautiful",): 0.7, ("Plain",): 0.3},
    ("Taste", "Presentation"): {
        ("Delicious", "Beautiful"): 0.56, ("Delicious", "Plain"): 0.24,
        ("Bland", "Beautiful"): 0.14, ("Bland", "Plain"): 0.06
    }
})

service_behavior = Behavior.from_contexts(service_space, {
    ("Friendliness",): {("Friendly",): 0.9, ("Rude",): 0.1},
    ("Speed",): {("Fast",): 0.6, ("Slow",): 0.4},
    ("Friendliness", "Speed"): {
        ("Friendly", "Fast"): 0.54, ("Friendly", "Slow"): 0.36,
        ("Rude", "Fast"): 0.06, ("Rude", "Slow"): 0.04
    }
})
```

The **product behavior** $P \otimes R$ combines measurements from both systems:

$$
(P \otimes R)(o_c, o_d \mid c \cup d) = P(o_c \mid c) \cdot R(o_d \mid d)
$$

```python
# Combine the behaviors using tensor product
combined_behavior = food_behavior @ service_behavior

print(f"Combined system has {len(combined_behavior)} contexts")
print("Combined observables:", combined_behavior.space.names)
print("Combined context:", combined_behavior.context[0].observables)
```

**Key properties:**

1. **Frame-independence preservation**: If both input behaviors are frame-independent, their product is also frame-independent

```python
# Frame-independence status
food_fi = food_behavior.is_frame_independent()
service_fi = service_behavior.is_frame_independent()
combined_fi = combined_behavior.is_frame_independent()
print(f"Food behavior FI: {food_fi}, Combined FI: {combined_fi}")
# Note: Frame-independence would be preserved if both input behaviors were FI
# The theorem guarantees: FI ⊗ FI = FI
```

2. **Deterministic combination**: For deterministic assignments, $q_s \otimes q_t = q_{s \sqcup t}$

## Proof

1. If $Q$ arises from global law $\mu$ and $S$ from global law $\nu$, then $Q \otimes S$ arises from the product law $\mu \otimes \nu$. The result follows from the definition and convexity of the frame-independent set.

2. Direct verification: $q_s \otimes q_t = q_{s \sqcup t}$ because the Kronecker delta functions multiply to give the combined assignment.