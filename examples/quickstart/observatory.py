#!/usr/bin/env python3
"""
Golden Example: Observatory API in Contrakit

This example demonstrates the Observatory API, a high-level, user-friendly
interface for creating and managing observational behaviors. The Observatory
wraps the core classes (Space, Context, Distribution, Behavior) with a more
intuitive interface for common use cases.

Key Concepts Demonstrated:
- Creating observatories with global alphabets
- Defining concepts with custom or inherited alphabets
- Using ValueHandle objects and syntactic sugar (& operator)
- Setting marginal and joint probability distributions
- Distribution validation and error handling
- Creating Behavior objects from perspective maps
- Using LensScope for different observational perspectives
- Advanced lens composition and meta-analysis
- Joint observations with & syntax across lenses
- Contradictory distributions and agreement analysis

The Observatory API makes it easy to model complex observational systems
with multiple perspectives and measurement contexts.
"""

from contrakit.observatory import Observatory
from examples.utils import print_header

print_header("Golden Example: Observatory API")

# 1. Creating Observatory with Global Alphabet
print("\n1. Creating Observatory with Global Alphabet")
print("-" * 46)

# Create observatory with global alphabet
observatory = Observatory.create(symbols=["Yes", "No"])
print("Created observatory with global alphabet:", [str(v) for v in observatory.alphabet])

# 2. Defining Concepts
print("\n2. Defining Concepts")
print("-" * 18)

# Define concepts - uses global alphabet by default
voter = observatory.concept("Voter")  # Uses ["Yes", "No"]
print(f"Voter concept: {voter.name} with symbols {voter.symbols}")

candidate = observatory.concept("Candidate", symbols=["Qualified", "Unqualified"])
print(f"Candidate concept: {candidate.name} with symbols {candidate.symbols}")

reviewer = observatory.concept("Reviewer", symbols=["Maybe", *observatory.alphabet])  # ["Maybe", "Yes", "No"]
print(f"Reviewer concept: {reviewer.name} with symbols {reviewer.symbols}")

# 3. Setting Marginal Distributions
print("\n3. Setting Marginal Distributions")
print("-" * 32)

# Set distributions using ValueHandle objects
yes, no = voter.alphabet
observatory.perspectives[voter] = {yes: 0.6, no: 0.4}
print(f"Voter distribution: {dict(observatory.perspectives[voter].distribution.to_dict())}")

qualified, unqualified = candidate.alphabet
observatory.perspectives[candidate] = {qualified: 0.7, unqualified: 0.3}
print(f"Candidate distribution: {dict(observatory.perspectives[candidate].distribution.to_dict())}")

# 4. Setting Joint Distributions with & Syntax
print("\n4. Setting Joint Distributions with & Syntax")
print("-" * 43)

# Joint distributions using & operator syntactic sugar
maybe_vote, hire, no_hire = reviewer.alphabet  # ["Maybe", "Yes", "No"]
observatory.perspectives[voter, candidate] = {
    yes & qualified: 0.3,        # Voter: Yes, Candidate: Qualified
    yes & unqualified: 0.2,      # Voter: Yes, Candidate: Unqualified
    no & qualified: 0.1,         # Voter: No, Candidate: Qualified
    no & unqualified: 0.4        # Voter: No, Candidate: Unqualified
}
print("Joint distribution set using & syntax")

# 5. Distribution Validation and Error Handling
print("\n5. Distribution Validation and Error Handling")
print("-" * 46)

# Validate distributions
observatory.perspectives.validate()
print("Distributions validated successfully")

# Show error handling for invalid distributions
try:
    # This would fail - probabilities don't sum to 1
    observatory.perspectives[voter] = {yes: 0.7, no: 0.5}  # 1.2 > 1.0
except ValueError as e:
    print(f"Caught expected error: {e}")

# Reset to valid distribution
observatory.perspectives[voter] = {yes: 0.6, no: 0.4}

# 6. Creating Behavior Objects and Analyzing Contradiction
print("\n6. Creating Behavior Objects and Analyzing Contradiction")
print("-" * 55)

# Create behavior object
behavior = observatory.perspectives.to_behavior()
print(f"Created behavior with {len(behavior)} contexts")
print(f"Overall agreement: {behavior.agreement.result:.6f}")
print(f"Contradiction bits: {behavior.contradiction_bits:.3f}")

# 7. Setting Up Contradictory Distributions
print("\n7. Setting Up Contradictory Distributions")
print("-" * 41)

# Create contradictory scenario: marginals vs joint
observatory.perspectives[voter] = {yes: 0.6}  # Marginal: Voter says Yes 60%
observatory.perspectives[candidate] = {qualified: 0.7}  # Marginal: Candidate Qualified 70%

# But joint contradicts: Voter never says Yes when Candidate is Qualified
observatory.perspectives[voter, candidate] = {
    yes & qualified: 0.0,        # Impossible combination
    yes & unqualified: 0.42,     # 0.6 * 0.3 = 0.18, but adjusted
    no & qualified: 0.49,        # 0.4 * 0.7 = 0.28, but adjusted
    no & unqualified: 0.09       # 0.4 * 0.3 = 0.12, but adjusted
}

behavior_contradictory = observatory.perspectives.to_behavior()
print(f"Contradictory behavior agreement: {behavior_contradictory.agreement.result:.6f}")
print(f"Contradictory behavior contradiction: {behavior_contradictory.contradiction_bits:.3f} bits")

# 8. Using Lenses for Different Perspectives
print("\n8. Using Lenses for Different Perspectives")
print("-" * 45)

# Use lenses for different perspectives
with observatory.lens(reviewer) as lens_r:
    local_concept = lens_r.define("LocalConcept", symbols=["Value1", "Value2"])
    val1, val2 = local_concept.alphabet
    lens_r.perspectives[local_concept] = {val1: 0.7, val2: 0.3}
    behavior_r = lens_r.to_behavior()
    print(f"Lens behavior created with {len(behavior_r)} contexts")
    print(f"Lens agreement: {behavior_r.agreement.result:.6f}")

# 9. Advanced Lens Features: Meta-Lenses
print("\n9. Advanced Lens Features: Meta-Lenses")
print("-" * 37)

# Create observable lens for meta-analysis
with observatory.lens("Quality_Assessment", symbols=["High", "Medium", "Low"]) as quality_lens:
    high, medium, low = quality_lens.alphabet
    print(f"Observable lens created: {quality_lens.name} with symbols {quality_lens.symbols}")

    # Create meta-lens that observes the quality assessment
    with observatory.lens("Meta_Reviewer") as meta_lens:
        # Meta-lens can observe other lenses and concepts
        meta_lens.perspectives[quality_lens] = {high: 0.5, medium: 0.3, low: 0.2}
        meta_behavior = meta_lens.to_behavior()
        print(f"Meta-lens behavior created with {len(meta_behavior)} contexts")

# 10. Lens Composition and Joint Observations
print("\n10. Lens Composition and Joint Observations")
print("-" * 46)

# Create multiple lenses for composition with joint observations
with observatory.lens("Perspective_A", symbols=["Pro_Hire", "Anti_Hire"]) as lens_a:
    pro_hire, anti_hire = lens_a.alphabet
    lens_a.perspectives[voter] = {yes: 0.8, no: 0.2}

with observatory.lens("Perspective_B", symbols=["Good_Fit", "Poor_Fit"]) as lens_b:
    good_fit, poor_fit = lens_b.alphabet
    lens_b.perspectives[candidate] = {qualified: 0.9, unqualified: 0.1}

# Compose lenses
composition = lens_a | lens_b  # Union composition
print(f"Lens composition created with {len(composition.lenses)} lenses")

# Advanced: Joint observations across lenses using & syntax
with observatory.lens("Joint_Observer", symbols=["Agree", "Disagree"]) as joint_lens:
    agree, disagree = joint_lens.alphabet
    # Observe both lenses jointly using their local alphabets
    joint_lens.perspectives[lens_a & lens_b] = {
        (pro_hire.value, good_fit.value): 0.6,     # Agree case
        (pro_hire.value, poor_fit.value): 0.1,     # Disagree case
        (anti_hire.value, good_fit.value): 0.1,    # Disagree case
        (anti_hire.value, poor_fit.value): 0.2     # Agree case
    }

composed_behavior = composition.to_behavior()
joint_behavior = joint_lens.to_behavior()
print(f"Composed behavior agreement: {composed_behavior.agreement.result:.6f}")
print(f"Joint observer agreement: {joint_behavior.agreement.result:.6f}")
print(f"Joint observer contradiction: {joint_behavior.contradiction_bits:.3f} bits")

# 11. ValueHandle Advanced Operations
print("\n11. ValueHandle Advanced Operations")
print("-" * 36)

# Demonstrate ValueHandle properties
print(f"ValueHandle string repr: {str(yes)}")
print(f"ValueHandle repr: {repr(yes)}")
print(f"ValueHandle equality: {yes == yes}, different instances: {yes is not voter.alphabet[0]}")

# & operator for joint outcomes
joint_outcome = yes & qualified
print(f"Joint outcome with &: {joint_outcome}")

# 12. Cross-Observatory ValueHandle Usage
print("\n12. Cross-Observatory ValueHandle Usage")
print("-" * 41)

# Create second observatory and demonstrate ValueHandle sharing
observatory2 = Observatory.create(symbols=["Shared_Yes", "Shared_No"])
shared_yes, shared_no = observatory2.alphabet

# Use ValueHandles from first observatory in second
observatory3 = Observatory.create(symbols=["Mixed", shared_yes, shared_no, "Extra"])
mixed, yes_shared, no_shared, extra = observatory3.alphabet
print(f"Cross-observatory alphabet: {[str(vh) for vh in observatory3.alphabet]}")

# Create concept using mixed ValueHandle types
concept_mixed = observatory3.concept("MixedConcept", symbols=[mixed, yes, extra])
print(f"Mixed concept symbols: {[str(vh) for vh in concept_mixed.alphabet]}")
