#!/usr/bin/env python3
"""
Golden Example: Observable Spaces in Contrakit

This example demonstrates the fundamental concept of observable spaces,
which define what can be measured and what values each measurement can take.
Spaces form the foundation for defining measurement systems and analyzing
consistency across different observables.

Key Concepts Demonstrated:
- Creating observable spaces with different value alphabets
- Using convenience methods (create, binary)
- Space operations (restriction, tensor product)
- Generating assignments and outcomes
- Comparing and transforming spaces

Spaces are essential because they define the structure of what can be observed
in a measurement system, forming the foundation for contexts and behaviors.
"""

from contrakit.space import Space
from examples.utils import print_header

# Define our experiment constants
COIN_FLIP = "Coin"
DIE_ROLL = "Die"
COLOR_PICK = "Color"

# Coin values
HEADS = "Heads"
TAILS = "Tails"

# Die values
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6

# Color values
RED = "Red"
GREEN = "Green"
BLUE = "Blue"

print_header("Golden Example: Observable Spaces")

# 1. Creating Spaces with Different Alphabets
print("\n1. Creating Spaces with Different Value Alphabets")
print("-" * 50)

# Different observables can have different types and numbers of values
coin_space = Space.create(**{COIN_FLIP: [HEADS, TAILS]})
die_space = Space.create(**{DIE_ROLL: [ONE, TWO, THREE, FOUR, FIVE, SIX]})
color_space = Space.create(**{COLOR_PICK: [RED, GREEN, BLUE]})

print("Created individual spaces:")
print(f"  coin_space: {len(coin_space)} observables → {dict(coin_space.alphabets)}")
print(f"  die_space: {len(die_space)} observables → {dict(die_space.alphabets)}")
print(f"  color_space: {len(color_space)} observables → {dict(color_space.alphabets)}")

# 2. Combining Spaces (Tensor Product)
print("\n2. Combining Spaces with Tensor Product")
print("-" * 40)

# Independent experiments can be combined
coin_and_die = coin_space @ die_space
coin_die_color = coin_space @ die_space @ color_space

print("Tensor product combinations:")
print(f"  coin ⊗ die: {len(coin_and_die)} observables")
print(f"    Names: {coin_and_die.names}")
print(f"    Total assignments: {coin_and_die.assignment_count()}")

print(f"  coin ⊗ die ⊗ color: {len(coin_die_color)} observables")
print(f"    Total assignments: {coin_die_color.assignment_count()}")

# 3. Restricting to Subsets (Projection)
print("\n3. Restricting Spaces to Observable Subsets")
print("-" * 44)

# We can focus on just some observables from a larger space
just_coin = coin_die_color | COIN_FLIP
just_die = coin_die_color | DIE_ROLL
coin_and_color = coin_die_color | [COIN_FLIP, COLOR_PICK]

print("Space restrictions:")
print(f"  Full space observables: {coin_die_color.names}")
print(f"  Just coin: {just_coin.names} → {just_coin.assignment_count()} assignments")
print(f"  Just die: {just_die.names} → {just_die.assignment_count()} assignments")
print(f"  Coin + color: {coin_and_color.names} → {coin_and_color.assignment_count()} assignments")

# 4. Convenience Methods
print("\n4. Using Convenience Methods")
print("-" * 28)

# Binary spaces are common in many applications
binary_space = Space.binary("A", "B", "C", "D")
print(f"Binary space (4 observables): {binary_space.names}")
print(f"  Each has alphabet: {binary_space.alphabets['A']}")

# Create a mixed space using create method
mixed_space = Space.create(
    Temperature=["Hot", "Warm", "Cold"],
    Pressure=["High", "Low"],
    Binary_Sensor=[0, 1]
)
print(f"Mixed space: {mixed_space.names}")
print(f"  Alphabets: {dict(mixed_space.alphabets)}")

# 5. Generating All Possible Assignments
print("\n5. Generating All Possible Assignments")
print("-" * 38)

# Small spaces can enumerate all possible outcomes
print(f"All possible coin flips: {list(coin_space.assignments())}")
print(f"All possible die rolls: {list(die_space.assignments())}")

# Combined spaces get large quickly
print(f"Coin+die combinations: {coin_and_die.assignment_count()}")
print("  First few assignments:", list(coin_and_die.assignments())[:3])

# 6. Outcomes for Specific Observable Subsets
print("\n6. Outcomes for Specific Observable Subsets")
print("-" * 44)

# Get all possible outcomes for a subset of observables
coin_outcomes = coin_die_color.outcomes_for([COIN_FLIP])
die_outcomes = coin_die_color.outcomes_for([DIE_ROLL])
coin_die_outcomes = coin_die_color.outcomes_for([COIN_FLIP, DIE_ROLL])

print("Outcomes for observable subsets:")
print(f"  {COIN_FLIP} alone: {coin_outcomes}")
print(f"  {DIE_ROLL} alone: {die_outcomes}")
print(f"  {COIN_FLIP}+{DIE_ROLL}: {len(coin_die_outcomes)} combinations")

# 7. Comparing Spaces
print("\n7. Comparing and Analyzing Space Differences")
print("-" * 46)

# Create two similar spaces
space1 = Space.create(X=[1, 2], Y=["A", "B"])
space2 = Space.create(X=[1, 2], Z=["A", "B"])

differences = space1.difference(space2)
print("Comparing space1 vs space2:")
print(f"  Only in space1: {differences['only_self']}")
print(f"  Only in space2: {differences['only_other']}")
print(f"  Alphabet differences: {differences['alphabet_diffs']}")

# 8. Renaming Observables
print("\n8. Renaming Observables in a Space")
print("-" * 35)

# Rename observables while preserving structure
renamed_space = coin_die_color.rename({
    COIN_FLIP: "Flip_Result",
    DIE_ROLL: "Die_Result",
    COLOR_PICK: "Color_Choice"
})

print("Original space names:", coin_die_color.names)
print("Renamed space names:", renamed_space.names)
print("Alphabets preserved:", dict(renamed_space.alphabets))

