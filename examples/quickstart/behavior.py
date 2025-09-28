#!/usr/bin/env python3
"""
Golden Example: Multi-Perspective Behaviors in Contrakit

This example demonstrates how to work with multi-perspective behaviors,
the core concept in contrakit. Behaviors represent probability distributions
across different observational contexts, allowing us to detect and quantify
contradictions between incompatible perspectives.

Key Concepts Demonstrated:
- Creating behaviors from context-specific probability distributions
- Computing agreement coefficients and contradiction measures
- Testing frame independence (consistency across all contexts)
- Using different behavior constructors (from_mu, from_counts, random, frame_independent)
- Understanding how behaviors represent multi-perspective observations
- Comparing behaviors across different observational perspectives (union, | operator)
- Analyzing contradiction costs in bits

Behaviors are the fundamental objects for studying contradictions in
multi-perspective data, enabling quantitative analysis of consistency
across different measurement contexts.
"""

from contrakit.space import Space
from contrakit.behavior.behavior import Behavior
import numpy as np
from examples.utils import print_header

# Define our experimental constants
MORNING = "Morning"
EVENING = "Evening"
SUNNY = "Sunny"
RAINY = "Rainy"
H = "Hire"
N = "No_Hire"
REVIEWER_A = "Reviewer_A"
REVIEWER_B = "Reviewer_B"
REVIEWER_C = "Reviewer_C"

print_header("Golden Example: Multi-Perspective Behaviors")

# 1. Creating Behaviors from Contexts
print("\n1. Creating Behaviors from Contexts")
print("-" * 36)

# Create a simple weather observation space
weather_space = Space.create(**{MORNING: [SUNNY, RAINY], EVENING: [SUNNY, RAINY]})

# Define a consistent behavior (no contradiction)
print("Creating a consistent weather behavior...")
consistent_weather = Behavior.from_contexts(weather_space, {
    (MORNING,): {(SUNNY,): 0.7, (RAINY,): 0.3},  # Morning tends to be sunny
    (EVENING,): {(SUNNY,): 0.7, (RAINY,): 0.3},  # Evening also tends to be sunny
    (MORNING, EVENING): {
        (SUNNY, SUNNY): 0.49, (SUNNY, RAINY): 0.21,  # Consistent sunny patterns
        (RAINY, SUNNY): 0.21, (RAINY, RAINY): 0.09   # Consistent rainy patterns
    }
})

print(f"Consistent behavior has {len(consistent_weather)} contexts")
print(f"Agreement coefficient: {consistent_weather.alpha_star:.6f} (perfect agreement)")
print(f"Contradiction cost: {consistent_weather.K:.6f} bits (no contradiction)")


# Create a behavior with moderate tension (different perspectives but still consistent)
moderate_tension_weather = Behavior.from_contexts(weather_space, {
    (MORNING,): {(SUNNY,): 0.6, (RAINY,): 0.4},
    (EVENING,): {(SUNNY,): 0.5, (RAINY,): 0.5},
    (MORNING, EVENING): {
        (SUNNY, SUNNY): 0.3, (SUNNY, RAINY): 0.3,
        (RAINY, SUNNY): 0.2, (RAINY, RAINY): 0.2
    }
})

print(f"Moderate tension behavior has {len(moderate_tension_weather)} contexts")
print(f"Agreement coefficient: {moderate_tension_weather.alpha_star:.6f} (good but imperfect agreement)")
print(f"Contradiction cost: {moderate_tension_weather.K:.6f} bits (some tension)")

# 3. Frame Independence Testing
print("\n3. Frame Independence Testing")
print("-" * 31)

print("Testing frame independence:")
print(f"  Consistent behavior: {consistent_weather.is_frame_independent()}")
print(f"  Moderate tension behavior: {moderate_tension_weather.is_frame_independent()}")

print("\nInterpretation:")
print("  Frame-independent behaviors can be explained by a single underlying reality.")
print("  Frame-dependent behaviors contain irreducible contradictions.")

# 4. Understanding Behavior Contexts
print("\n4. Understanding Behavior Contexts")
print("-" * 35)

print(f"Consistent behavior contexts:")
for i, ctx in enumerate(consistent_weather.distributions.keys(), 1):
    print(f"  {i}. {ctx.observables}: {len(ctx.outcomes())} possible outcomes")

# 5. Creating Behaviors from Global Distributions
print("\n5. Creating Behaviors from Global Distributions")
print("-" * 47)

# Create behavior from global assignment distribution (mu)
print("Creating behavior from global distribution...")
assignment_probs = np.array([
    0.1,  # Morning=Sunny, Evening=Sunny
    0.3,  # Morning=Sunny, Evening=Rainy
    0.2,  # Morning=Rainy, Evening=Sunny
    0.4   # Morning=Rainy, Evening=Rainy
])

# Contexts as list of observable name sequences
contexts = [
    [MORNING],        # Morning observations only
    [EVENING],        # Evening observations only
    [MORNING, EVENING]  # Joint morning-evening observations
]

# Note: from_mu creates marginal distributions from global assignment probabilities
mu_behavior = Behavior.from_mu(weather_space, contexts, assignment_probs)

print("Behavior created from global distribution:")
for ctx in mu_behavior.context:
    ctx_key = tuple(ctx.observables)
    print(f"  {ctx_key}: {dict(mu_behavior[ctx].to_dict())}")

# 6. Creating Behaviors from Count Data
print("\n6. Creating Behaviors from Count Data")
print("-" * 37)

# Simulate count data (like from a survey)
print("Creating behavior from count data...")
count_data = {
    (MORNING,): {(SUNNY,): 70, (RAINY,): 30},
    (EVENING,): {(SUNNY,): 65, (RAINY,): 35},
    (MORNING, EVENING): {
        (SUNNY, SUNNY): 45, (SUNNY, RAINY): 25,
        (RAINY, SUNNY): 20, (RAINY, RAINY): 10
    }
}

count_behavior = Behavior.from_counts(weather_space, count_data, normalize="per_context")

print("Behavior created from counts:")
for ctx in count_behavior.context:
    ctx_key = tuple(ctx.observables)
    print(f"  {ctx_key}: {dict(count_behavior[ctx].to_dict())}")

# 7. Advanced: Hiring Example with Consistent Perspectives
print("\n7. Advanced: Hiring Example with Consistent Perspectives")
print("-" * 52)

# Create a hiring scenario with consistent reviewer behavior
hiring_space = Space.create(**{
    REVIEWER_A: [H, N],
    REVIEWER_B: [H, N],
    REVIEWER_C: [H, N]
})

print("Creating hiring behavior with consistent reviewer perspectives...")
hiring_behavior = Behavior.from_contexts(hiring_space, {
    # Individual reviewer tendencies (consistent with joints)
    (REVIEWER_A,): {(H,): 0.7, (N,): 0.3},  # A hires 70% of candidates
    (REVIEWER_B,): {(H,): 0.4, (N,): 0.6},  # B is more skeptical
    (REVIEWER_C,): {(H,): 0.6, (N,): 0.4},  # C consistent with joint marginals

    # Joint evaluations (consistent with marginals)
    (REVIEWER_A, REVIEWER_B): {
        (H, H): 0.3, (H, N): 0.4,  # A says hire, B says no: 40%
        (N, H): 0.1, (N, N): 0.2   # A says no, B says hire: 10%
    },
    (REVIEWER_B, REVIEWER_C): {
        (H, H): 0.32, (H, N): 0.08,  # B hire, C no: 8%
        (N, H): 0.28, (N, N): 0.32   # B no, C hire: 32%
    },
    (REVIEWER_A, REVIEWER_C): {
        (H, H): 0.6, (H, N): 0.1,    # A hire, C no: 10%
        (N, H): 0.0, (N, N): 0.3     # A no, C hire: 30%
    }
})

print(f"Hiring behavior has {len(hiring_behavior)} contexts")
print(f"Agreement coefficient: {hiring_behavior.alpha_star:.6f}")
print(f"Contradiction cost: {hiring_behavior.contradiction_bits:.6f} bits")

print("\nThis example shows how different reviewer perspectives can be")
print("consistently modeled, even with varying individual tendencies.")

# 8. Computing Agreement with Custom Weights
print("\n8. Computing Agreement with Custom Weights")
print("-" * 42)

# Different trust weights for reviewers
trust_weights = {
    (REVIEWER_A,): 0.6,  # Trust reviewer A most
    (REVIEWER_B,): 0.3,  # Trust B moderately
    (REVIEWER_C,): 0.1   # Trust C least
}

# Use the new fluent agreement API
agreement_score = hiring_behavior.agreement.for_weights(trust_weights).result
print(f"Agreement with custom weights: {agreement_score:.6f}")

# Get the explanation (scenario distribution)
explanation = hiring_behavior.agreement.for_weights(trust_weights).explanation

print("Top 3 scenarios from optimal explanation:")
scenarios = hiring_behavior.agreement.for_weights(trust_weights).scenarios()
for i, (scenario, prob) in enumerate(scenarios[:3]):
    # Map indices to reviewer names and decisions
    names = [REVIEWER_A, REVIEWER_B, REVIEWER_C]
    decisions = [H if s == 0 else N for s in scenario]  # Assuming H=0, N=1 in space order
    scenario_str = ", ".join(f"{name}: {dec}" for name, dec in zip(names, decisions))
    print(f"  {i+1}. {scenario_str} ({prob:.1%})")

print("\nFeature distributions from optimal explanation:")
reviewer_a_dist = hiring_behavior.agreement.for_weights(trust_weights).feature_distribution(REVIEWER_A)
print(f"  {REVIEWER_A} distribution: {reviewer_a_dist}")

# 9. Worst-Case Weights Analysis
print("\n9. Worst-Case Weights Analysis")
print("-" * 32)

worst_weights = hiring_behavior.worst_case_weights
print("Weights that maximize contradiction:")
for ctx_key, weight in worst_weights.items():
    ctx_str = ", ".join(ctx_key)
    print(f"  {ctx_str}: {weight:.3f}")

print("\nInterpretation:")
print("  These weights show which contexts drive the contradiction most strongly.")
print("  Higher weights indicate contexts that are hardest to reconcile.")

# 10. Additional Behavior Operations
print("\n10. Additional Behavior Operations")
print("-" * 35)

# Create a simple behavior for demonstration
simple_space = Space.create(Coin=["Heads", "Tails"])
simple_behavior = Behavior.from_contexts(simple_space, {
    ("Coin",): {("Heads",): 0.6, ("Tails",): 0.4}
})

print("Original simple behavior:")
print(f"  Coin distribution: {dict(simple_behavior[simple_behavior.context[0]].to_dict())}")

# Mix behaviors
fair_coin = Behavior.from_contexts(simple_space, {
    ("Coin",): {("Heads",): 0.5, ("Tails",): 0.5}
})

mixed_behavior = simple_behavior.mix(fair_coin, 0.3)  # 70% biased, 30% fair
print(f"Mixed with fair coin (30%): {dict(mixed_behavior[simple_behavior.context[0]].to_dict())}")

# Rename observables
renamed_behavior = simple_behavior.rename_observables({"Coin": "Flip"})
print(f"Renamed observable: {list(renamed_behavior.space.names)}")

# Compare behaviors
distance = simple_behavior.product_l1_distance(fair_coin)
print(f"L1 distance between biased and fair coin behaviors: {distance:.3f}")

# Random behavior generation
random_behavior = Behavior.random(weather_space, [[MORNING], [EVENING], [MORNING, EVENING]])
print(f"Random behavior has {len(random_behavior)} contexts")
print(f"Random morning: {dict(random_behavior[random_behavior.context[0]].to_dict())}")

# Frame-independent behavior
fi_behavior = Behavior.frame_independent(weather_space, [[MORNING], [EVENING], [MORNING, EVENING]])
print(f"Frame-independent behavior agreement: {fi_behavior.agreement.result:.6f} (should be 1.0)")

# Per-context scores
scores = hiring_behavior.per_context_scores()
print("Per-context agreement scores:")
for ctx, score in zip(hiring_behavior.context, scores):
    ctx_str = ", ".join(ctx.observables)
    print(f"  {ctx_str}: {score:.6f}")

# Aggregate scores
min_score = hiring_behavior.aggregate(lambda x: min(x))
print(f"Worst context agreement: {min_score:.6f}")

# Compare behaviors with different but compatible observational contexts
print("\nBehavior Comparison Across Different Observational Contexts")
print("-" * 60)

# Create an extended weather space with additional observables
extended_space = Space.create(Morning=["Sunny", "Rainy"], Evening=["Sunny", "Rainy"],
                             Humidity=["Dry", "Humid"], Cloud_Cover=["Clear", "Cloudy"])

# Behavior 1: Sky observer (weather and cloud conditions)
sky_observer = Behavior.from_contexts(extended_space, {
    ("Morning",): {("Sunny",): 0.8, ("Rainy",): 0.2},
    ("Cloud_Cover",): {("Clear",): 0.6, ("Cloudy",): 0.4}
})

# Behavior 2: Ground observer (different contexts, no overlap with sky observer)
ground_observer = Behavior.from_contexts(extended_space, {
    ("Evening",): {("Sunny",): 0.7, ("Rainy",): 0.3},
    ("Humidity",): {("Dry",): 0.5, ("Humid",): 0.5}
})

comparison_score = (sky_observer | ground_observer).agreement.result
print(f"Agreement between sky and ground weather observers: {comparison_score:.6f}")
print("  (High agreement means their observations are reconcilable)")
print("  (Low agreement would indicate fundamental conflicts between observers)")

# Demonstrate comparison with custom trust weights
combined_observers = sky_observer | ground_observer
weighted_comparison = combined_observers.agreement.for_weights({
    ("Morning",): 0.3, ("Evening",): 0.3, ("Cloud_Cover",): 0.2, ("Humidity",): 0.2
}).result
print(f"Agreement with balanced trust across all observations: {weighted_comparison:.6f}")

print("The | operator combines behaviors from the same space but different contexts,")
print("creating a merged behavior. This enables cross-validation between different")
print("measurement approaches: (sky_observer | ground_observer).agreement.result")


print("\n")
print_header("Behavior demonstration complete")
