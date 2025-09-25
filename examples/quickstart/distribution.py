#!/usr/bin/env python3
"""
Golden Example: Probability Distributions in Contrakit

This example demonstrates how to work with probability distributions
over observational outcomes. Distributions represent the likelihood of
different measurement results within a specific observational context,
forming the basic building blocks for behavioral patterns and
contradiction analysis.

Key Concepts Demonstrated:
- Creating distributions from outcomes and probabilities
- Using convenience methods (uniform, random, from_dict)
- Accessing and manipulating probability values
- Convex combinations and mixing distributions
- Converting between different representation formats
- Computing distances between distributions

Distributions are fundamental to defining what we observe in different
measurement contexts and enable quantitative analysis of observational patterns.
"""

from contrakit.distribution import Distribution
import numpy as np
from examples.utils import print_header

# Define our experimental constants
HEADS = "Heads"
TAILS = "Tails"
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6

# Weather outcomes
SUNNY = "Sunny"
CLOUDY = "Cloudy"
RAINY = "Rainy"
HOT = "Hot"
WARM = "Warm"
COLD = "Cold"

print_header("Golden Example: Probability Distributions")

# 1. Creating Distributions Directly
print("\n1. Creating Distributions Directly")
print("-" * 35)

# Basic coin flip distribution
coin_fair = Distribution(
    outcomes=((HEADS,), (TAILS,)),
    probs=(0.5, 0.5)
)
print(f"Fair coin: P({HEADS}) = {coin_fair[(HEADS,)]:.1f}, P({TAILS}) = {coin_fair[(TAILS,)]:.1f}")

# Biased coin distribution
coin_biased = Distribution(
    outcomes=((HEADS,), (TAILS,)),
    probs=(0.7, 0.3)
)
print(f"Biased coin: P({HEADS}) = {coin_biased[(HEADS,)]:.1f}, P({TAILS}) = {coin_biased[(TAILS,)]:.1f}")

# Die roll distribution (fair six-sided die)
die_outcomes = ((ONE,), (TWO,), (THREE,), (FOUR,), (FIVE,), (SIX,))
die_fair = Distribution(
    outcomes=die_outcomes,
    probs=(1/6, 1/6, 1/6, 1/6, 1/6, 1/6)
)
print(f"Fair die: P({SIX}) = {die_fair[(SIX,)]:.3f} (1/6 = {1/6:.3f})")

# 2. Creating Distributions with Convenience Methods
print("\n2. Creating Distributions with Convenience Methods")
print("-" * 52)

# Uniform distribution over outcomes
weather_outcomes = ((SUNNY,), (CLOUDY,), (RAINY,))
weather_uniform = Distribution.uniform(weather_outcomes)
print(f"Uniform weather: {dict(weather_uniform.to_dict())}")

# Random distribution (Dirichlet sampling)
weather_random = Distribution.random(weather_outcomes, alpha=1.0)
print(f"Random weather: {dict(weather_random.to_dict())}")

# From dictionary
coin_dict = {(HEADS,): 0.6, (TAILS,): 0.4}
coin_from_dict = Distribution.from_dict(coin_dict)
print(f"From dict: {dict(coin_from_dict.to_dict())}")

# 3. Accessing Probabilities
print("\n3. Accessing Probabilities")
print("-" * 26)

print("Coin flip probabilities:")
print(f"  P({HEADS}) = {coin_fair[(HEADS,)]}")
print(f"  P({TAILS}) = {coin_fair[(TAILS,)]}")
print(f"  P({SIX}) = {coin_fair[(SIX,)]} (zero for impossible outcomes)")

# 4. Joint Distributions (Multiple Observables)
print("\n4. Joint Distributions (Multiple Observables)")
print("-" * 45)

# Weather joint distribution: temperature and conditions
weather_joint_outcomes = (
    (HOT, SUNNY), (HOT, CLOUDY), (HOT, RAINY),
    (WARM, SUNNY), (WARM, CLOUDY), (WARM, RAINY),
    (COLD, SUNNY), (COLD, CLOUDY), (COLD, RAINY)
)

# Realistic weather correlations
weather_joint_probs = (
    0.25, 0.15, 0.05,  # Hot weather
    0.20, 0.15, 0.10,  # Warm weather
    0.05, 0.04, 0.01   # Cold weather
)

weather_joint = Distribution(
    outcomes=weather_joint_outcomes,
    probs=weather_joint_probs
)

print("Joint weather distribution:")
print("  P(Hot, Sunny) =", weather_joint[(HOT, SUNNY)])
print("  P(Warm, Cloudy) =", weather_joint[(WARM, CLOUDY)])
print("  P(Cold, Rainy) =", weather_joint[(COLD, RAINY)])

# 5. Convex Combinations and Mixing
print("\n5. Convex Combinations and Mixing")
print("-" * 36)

# Mix fair and biased coins
coin_mixed = coin_fair.mix(coin_biased, 0.3)  # 70% fair, 30% biased
print(f"Mixed coin (70% fair + 30% biased):")
print(f"  P({HEADS}) = {coin_mixed[(HEADS,)]:.3f}")

# Equal mixture using + operator
coin_average = coin_fair + coin_biased  # Same as mix with weight 0.5
print(f"Average of fair + biased:")
print(f"  P({HEADS}) = {coin_average[(HEADS,)]:.3f}")

# 6. Converting Between Formats
print("\n6. Converting Between Formats")
print("-" * 30)

# To dictionary
weather_dict = weather_joint.to_dict()
print(f"As dictionary: {len(weather_dict)} entries")

# To numpy array
weather_array = weather_joint.to_array()
print(f"As array: shape {weather_array.shape}, sum = {weather_array.sum():.1f}")

# 7. Computing Distances Between Distributions
print("\n7. Computing Distances Between Distributions")
print("-" * 46)

# Compare different weather models
weather_sunny_bias = Distribution(
    outcomes=weather_joint_outcomes,
    probs=(0.40, 0.30, 0.10, 0.10, 0.05, 0.03, 0.01, 0.005, 0.005)  # More sunny
)

weather_rainy_bias = Distribution(
    outcomes=weather_joint_outcomes,
    probs=(0.10, 0.10, 0.20, 0.10, 0.15, 0.10, 0.04, 0.10, 0.11)  # More rainy
)

distance_sunny = weather_joint.l1_distance(weather_sunny_bias)
distance_rainy = weather_joint.l1_distance(weather_rainy_bias)

print(f"L1 distance to sunny-biased model: {distance_sunny:.3f}")
print(f"L1 distance to rainy-biased model: {distance_rainy:.3f}")

# 8. Working with Random Distributions
print("\n8. Working with Random Distributions")
print("-" * 36)

# Set seed for reproducibility
rng = np.random.default_rng(42)

# Generate several random distributions
random_distributions = []
for i in range(3):
    rand_dist = Distribution.random(weather_outcomes, alpha=2.0, rng=rng)
    random_distributions.append(rand_dist)
    print(f"Random dist {i+1}: {dict(rand_dist.to_dict())}")

# Compare the random distributions
dist_01 = random_distributions[0].l1_distance(random_distributions[1])
dist_02 = random_distributions[0].l1_distance(random_distributions[2])
dist_12 = random_distributions[1].l1_distance(random_distributions[2])

print(f"L1 distances between random distributions:")
print(f"  Dist 0-1: {dist_01:.3f}")
print(f"  Dist 0-2: {dist_02:.3f}")
print(f"  Dist 1-2: {dist_12:.3f}")
