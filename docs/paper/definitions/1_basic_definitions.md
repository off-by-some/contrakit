# Basic Definitions

In this work, we work with finite sets of possible outcomes for all observables. The frame-independent set (FI) that we'll define later has some nice mathematical properties—it's nonempty, convex, compact, and closed under certain operations.

These properties hold naturally in our finite setting.

## Observable System

An **observable system** consists of a collection of things we can measure or observe. Think of it like a set of questions we can ask about a system.

Formally, we define this as follows.

Formally—let $\mathcal{X} = \{X_1, \ldots, X_n\}$ be a finite set of **observables**. Each observable $x \in \mathcal{X}$ has a finite set of possible **outcomes** $\mathcal{O}_x$ (never empty).

A **context** $c$ is simply a subset of observables that we measure together—like asking multiple questions at once. The possible outcomes for a context $c$ are all combinations of outcomes from its observables: $\mathcal{O}_c := \prod_{x \in c} \mathcal{O}_x$.

**Python example:**
```python
from contrakit import Space

# Create measurement system with varied survey questions
survey_space = Space.create(
    Satisfaction=["Satisfied", "Dissatisfied"],
    Recommendation=["Recommend", "Not_Recommend"]
)

# Examine the system structure
print("Observables:", survey_space.names)
print("Satisfaction outcomes:", survey_space.alphabets['Satisfaction'])
print("Recommendation outcomes:", survey_space.alphabets['Recommendation'])
print("Total combinations:", survey_space.assignment_count())
```

## Behavior

A **behavior** describes how a system responds when we measure different contexts. It's the collection of probability distributions that tell us what outcomes we get when measuring each possible context.

Formally—given a collection $\mathcal{C}$ of contexts (subsets of $\mathcal{X}$), a **behavior** $P$ assigns to each context $c \in \mathcal{C}$ a probability distribution $p_c$ over the possible outcomes $\mathcal{O}_c$:

$$
P = \{p_c \in \Delta(\mathcal{O}_c) : c \in \mathcal{C}\}
$$

Here $\Delta(\mathcal{O}_c)$ means the set of all probability distributions over $\mathcal{O}_c$.

Put differently—this captures the essence.

**Important note**: We don't require that measurements in overlapping contexts are consistent with each other. This means the same observable might give different results when measured in different combinations with others.

And the frame-independent set (defined below) serves as our baseline regardless of this consistency.

**Python example:**
```python
from contrakit import Space, Behavior

# Define the observable system
survey_space = Space.create(
    Satisfaction=["Satisfied", "Dissatisfied"],
    Recommendation=["Recommend", "Not_Recommend"]
)

# Create behavior with probability distributions for each context
behavior = Behavior.from_contexts(survey_space, {
    ("Satisfaction",): {("Satisfied",): 0.8, ("Dissatisfied",): 0.2},
    ("Recommendation",): {("Recommend",): 0.7, ("Not_Recommend",): 0.3},
    ("Satisfaction", "Recommendation"): {
        ("Satisfied", "Recommend"): 0.56, ("Satisfied", "Not_Recommend"): 0.24,
        ("Dissatisfied", "Recommend"): 0.14, ("Dissatisfied", "Not_Recommend"): 0.06
    }
})

print(f"Number of contexts: {len(behavior)}")
print("Satisfaction distribution:", dict(behavior[behavior.context[0]].to_dict()))
print("Joint distribution:", dict(behavior[behavior.context[2]].to_dict()))
```

## Deterministic Global Assignment

Imagine there's a "true" underlying state of the system that determines all measurement outcomes. A **deterministic global assignment** captures this idea—it's like assuming there's a hidden reality that gives definite answers to all possible measurements.

Formally—let $\mathcal{O}_{\mathcal{X}} := \prod_{x \in \mathcal{X}} \mathcal{O}_x$ be the set of all possible complete assignments of outcomes to observables. A **deterministic global assignment** $s$ is an element of this set.

Having established the assignments, we now turn to the induced behaviors. Consider this carefully.

Such an assignment $s$ naturally defines a behavior $q_s$ where measurements are perfectly predictable:

$$
q_s(o \mid c) = \begin{cases} 1 & \text{if } o \text{ matches } s \text{ restricted to context } c \\ 0 & \text{otherwise} \end{cases}
$$


**Python example:**
```python
from contrakit import Space, Behavior
import numpy as np

# Define the observable system
survey_space = Space.create(
    Satisfaction=["Satisfied", "Dissatisfied"],
    Recommendation=["Recommend", "Not_Recommend"]
)

# Show all possible deterministic assignments
print("Possible global assignments:")
for i, assignment in enumerate(survey_space.assignments()):
    print(f"  {i}: {assignment}")

# Create behavior from assignment 3: (Dissatisfied, Not_Recommend)
assignment_probs = np.zeros(survey_space.assignment_count())
assignment_probs[3] = 1.0

contexts = [["Satisfaction"], ["Recommendation"], ["Satisfaction", "Recommendation"]]
deterministic_behavior = Behavior.from_mu(survey_space, contexts, assignment_probs)

print("\nBehavior from (Dissatisfied, Not_Recommend):")
for ctx in deterministic_behavior.context:
    print(f"  {tuple(ctx.observables)}: {dict(deterministic_behavior[ctx].to_dict())}")
```

## Frame-Independent Set

The **frame-independent set** (FI) represents all behaviors that could arise from some deterministic global assignment, possibly mixed together probabilistically. Think of it as the set of all behaviors that could be explained by "hidden variables"—underlying states that determine everything, even if we don't know exactly which state we're dealing with.

Formally—

$$
\text{FI} := \text{conv}\{q_s : s \in \mathcal{O}_{\mathcal{X}}\} \subseteq \prod_{c \in \mathcal{C}} \Delta(\mathcal{O}_c)
$$

This is the convex hull (all probabilistic mixtures) of the deterministic behaviors $q_s$.

Put differently, these are the behaviors without contradiction.

**Python example:**
```python
from contrakit import Space, Behavior
import numpy as np

# Define the observable system
survey_space = Space.create(
    Satisfaction=["Satisfied", "Dissatisfied"],
    Recommendation=["Recommend", "Not_Recommend"]
)

contexts = [["Satisfaction"], ["Recommendation"], ["Satisfaction", "Recommendation"]]

# Create frame-independent behavior (60% positive, 40% negative responses)
assignment_probs = np.array([0.6, 0.0, 0.0, 0.4])
fi_behavior = Behavior.from_mu(survey_space, contexts, assignment_probs)

print("Frame-independent behavior (mixture of customer types):")
for ctx in fi_behavior.context:
    print(f"  {tuple(ctx.observables)}: {dict(fi_behavior[ctx].to_dict())}")
print(f"Is frame-independent: {fi_behavior.is_frame_independent()}")

# Compare with contradictory behavior
contradictory_behavior = Behavior.from_contexts(survey_space, {
    ("Satisfaction",): {("Satisfied",): 0.7, ("Dissatisfied",): 0.3},
    ("Recommendation",): {("Recommend",): 0.7, ("Not_Recommend",): 0.3},
    ("Satisfaction", "Recommendation"): {
        ("Satisfied", "Recommend"): 0.25, ("Satisfied", "Not_Recommend"): 0.45,
        ("Dissatisfied", "Recommend"): 0.45, ("Dissatisfied", "Not_Recommend"): 0.25
    }
})
print(f"\nContradictory behavior is frame-independent: {contradictory_behavior.is_frame_independent()}")
```

### Topological Properties

The frame-independent set has some important mathematical properties:

**Proposition.** The frame-independent set FI is nonempty, convex, and compact.

**Proof.**

- **Nonempty**: FI contains all the deterministic behaviors $q_s$.
- **Convex**: It's defined as a convex hull, so it's convex by construction.
- **Compact**: Since we're working in finite dimensions with finite sets, FI is a polytope (a bounded polyhedron), hence compact.

### Context Simplex

Sometimes we need to consider probability distributions over contexts themselves - like randomly choosing which set of measurements to perform.

The **context simplex** $\Delta(\mathcal{C})$ is the set of all probability distributions over contexts:

$$
\Delta(\mathcal{C}) := \{\lambda \in \mathbb{R}^{\mathcal{C}} : \lambda_c \geq 0 \text{ for each } c \in \mathcal{C}, \sum_{c \in \mathcal{C}} \lambda_c = 1\}
$$

**Python example:**
```python
from contrakit import Space, Behavior
import numpy as np

# Define system with varied survey questions
survey_space = Space.create(
    Food_Quality=["Excellent", "Poor"],
    Service_Speed=["Fast", "Slow"],
    Value_Rating=["Good_Value", "Poor_Value"]
)

contexts = [
    ["Food_Quality"], ["Service_Speed"], ["Value_Rating"],
    ["Food_Quality", "Service_Speed"]
]

behavior = Behavior.from_contexts(survey_space, {
    ("Food_Quality",): {("Excellent",): 0.8, ("Poor",): 0.2},
    ("Service_Speed",): {("Fast",): 0.7, ("Slow",): 0.3},
    ("Value_Rating",): {("Good_Value",): 0.6, ("Poor_Value",): 0.4},
    ("Food_Quality", "Service_Speed"): {
        ("Excellent", "Fast"): 0.56, ("Excellent", "Slow"): 0.24,
        ("Poor", "Fast"): 0.14, ("Poor", "Slow"): 0.06
    }
})

# Context simplex: weight different survey question combinations
context_weights = {
    ("Food_Quality",): 0.3, ("Service_Speed",): 0.3, ("Value_Rating",): 0.2,
    ("Food_Quality", "Service_Speed"): 0.2
}

print(f"Weights sum: {sum(context_weights.values())}")
overall_agreement = behavior.agreement.for_weights(context_weights).result
print(f"Overall survey agreement: {overall_agreement:.6f}")

# Focus on individual questions only
individual_weights = {
    ("Food_Quality",): 0.4, ("Service_Speed",): 0.4, ("Value_Rating",): 0.2,
    ("Food_Quality", "Service_Speed"): 0.0
}
individual_agreement = behavior.agreement.for_weights(individual_weights).result
print(f"Individual question agreement: {individual_agreement:.6f}")
```

## Observatory API: High-Level Interface

The concepts above form the foundation of contrakit's mathematical framework. For practical applications, contrakit provides the **Observatory API** - a high-level interface that makes it easier to work with these concepts without directly managing spaces, contexts, and behaviors.

**Python example:**
```python
from contrakit import Observatory

# Create observatory (equivalent to defining a Space)
observatory = Observatory.create(symbols=["Satisfied", "Dissatisfied", "Recommend", "Not_Recommend"])

# Define concepts (equivalent to observables in a Space)
satisfaction = observatory.concept("Satisfaction", symbols=["Satisfied", "Dissatisfied"])
recommendation = observatory.concept("Recommendation", symbols=["Recommend", "Not_Recommend"])

print("Defined concepts:")
print(f"  {satisfaction.name}: {satisfaction.symbols}")
print(f"  {recommendation.name}: {recommendation.symbols}")

# Set marginal distributions (equivalent to single contexts in a Behavior)
observatory.perspectives[satisfaction] = {"Satisfied": 0.8, "Dissatisfied": 0.2}
observatory.perspectives[recommendation] = {"Recommend": 0.7, "Not_Recommend": 0.3}

print("\nMarginal distributions:")
print(f"  Satisfaction: {dict(observatory.perspectives[satisfaction].distribution.to_dict())}")
print(f"  Recommendation: {dict(observatory.perspectives[recommendation].distribution.to_dict())}")

# Set joint distribution using & syntax (equivalent to joint contexts)
satisfied, dissatisfied = satisfaction.alphabet
recommend, not_recommend = recommendation.alphabet

observatory.perspectives[satisfaction, recommendation] = {
    satisfied & recommend: 0.56,
    satisfied & not_recommend: 0.24,
    dissatisfied & recommend: 0.14,
    dissatisfied & not_recommend: 0.06
}

print("\nJoint distribution:")
joint_dist = observatory.perspectives[satisfaction, recommendation].distribution
print(f"  Joint: {dict(joint_dist.to_dict())}")

# Convert to underlying Behavior (what we've been working with directly)
underlying_behavior = observatory.perspectives.to_behavior()

print(f"\nConverted to behavior with {len(underlying_behavior)} contexts")
print(f"Frame-independent: {underlying_behavior.is_frame_independent()}")

# The Observatory API provides the same mathematical power
# but with more intuitive syntax and higher-level abstractions
```