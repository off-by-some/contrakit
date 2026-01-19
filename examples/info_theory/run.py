"""
Testing K(P) against Shannon information theory quantities.

Computes K(P) and relevant Shannon measures for different scenarios.
Minimal interpretation - let the numbers speak.
"""

import math
from contrakit import Space, Behavior, Observatory
import numpy as np

def shannon_entropy(probs):
    """H(X) = -Σ p(x) log₂ p(x)"""
    return sum(-p * math.log2(p) for p in probs if p > 0)

def conditional_entropy(joint_dist, marginal_y):
    """H(X|Y) = Σ p(y) H(X|Y=y)"""
    h_cond = 0
    for y, p_y in marginal_y.items():
        if p_y > 0:
            conditional_probs = [joint_dist.get((x, y), 0) / p_y 
                               for x in set(x for (x, _) in joint_dist.keys())]
            h_cond += p_y * shannon_entropy([p for p in conditional_probs if p > 0])
    return h_cond

def mutual_information(joint_dist, marginal_x, marginal_y):
    """I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    h_x = shannon_entropy(list(marginal_x.values()))
    h_y = shannon_entropy(list(marginal_y.values()))
    h_xy = shannon_entropy(list(joint_dist.values()))
    return h_x + h_y - h_xy

def check_joint_exists(constraints):
    """Try to find joint distribution satisfying all pairwise constraints.
    Returns (exists: bool, max_violation: float)"""
    # Simple satisfiability check for small discrete cases
    # Returns max constraint violation if no solution exists
    return None, None  # Placeholder - would need SAT solver for general case

print("=" * 80)
print("SCENARIO 1: Two contradictory rules for same input")
print("=" * 80)

# Create two-context scenario: parity vs roundness for digit "7"
obs1 = Observatory.create(symbols=['0', '1'])
label = obs1.concept('Label')

parity = obs1.lens('Parity')
with parity:
    parity.perspectives[label] = {'1': 1.0, '0': 0.0}

roundness = obs1.lens('Roundness')
with roundness:
    roundness.perspectives[label] = {'0': 1.0, '1': 0.0}

behavior1 = (parity | roundness).to_behavior()

# Shannon analysis assuming contexts equally likely
p_context = [0.5, 0.5]
p_label_0 = 0.5 * 0.0 + 0.5 * 1.0  
p_label_1 = 0.5 * 1.0 + 0.5 * 0.0
p_labels = [p_label_0, p_label_1]

H_X = shannon_entropy(p_labels)
H_X_given_C = 0.0  # Deterministic given context
H_C = shannon_entropy(p_context)
I_X_C = H_X - H_X_given_C

print(f"\nContrakit measures:")
print(f"  K(P) = {behavior1.K:.4f} bits")
print(f"  α*   = {behavior1.alpha_star:.4f}")

print(f"\nShannon measures:")
print(f"  H(Label)         = {H_X:.4f} bits")
print(f"  H(Label|Context) = {H_X_given_C:.4f} bits")
print(f"  H(Context)       = {H_C:.4f} bits")
print(f"  I(Label;Context) = {I_X_C:.4f} bits")

print(f"\nNumerical relationships:")
print(f"  |K(P) - H(C)|     = {abs(behavior1.K - H_C):.6f}")
print(f"  |K(P) - I(X;C)|   = {abs(behavior1.K - I_X_C):.6f}")
print(f"  K(P) / I(X;C)     = {behavior1.K / I_X_C:.4f}")

print("\n" + "=" * 80)
print("SCENARIO 2: Triangle disagreement (A≠B, B≠C, A=C)")
print("=" * 80)

space2 = Space.create(A=['0','1'], B=['0','1'], C=['0','1'])
behavior2 = Behavior.from_contexts(space2, {
    ('A','B'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('B','C'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('A','C'): {('0','0'): 1.0, ('1','1'): 0.0}
})

print(f"\nContrakit measures:")
print(f"  K(P) = {behavior2.K:.4f} bits")
print(f"  α*   = {behavior2.alpha_star:.4f}")

# Try to construct consistent joint distribution
# Check if any P(A,B,C) satisfies all three pairwise constraints
print(f"\nPairwise constraints:")
print(f"  P(A≠B) = 1.0")
print(f"  P(B≠C) = 1.0")
print(f"  P(A=C) = 1.0")

# Enumerate all 8 possible joint states
states = [(a,b,c) for a in [0,1] for b in [0,1] for c in [0,1]]
print(f"\nJoint state validity:")
for state in states:
    a, b, c = state
    satisfies_AB = (a != b)
    satisfies_BC = (b != c)
    satisfies_AC = (a == c)
    all_satisfied = satisfies_AB and satisfies_BC and satisfies_AC
    print(f"  P(A={a},B={b},C={c}): AB={'✓' if satisfies_AB else '✗'}, "
          f"BC={'✓' if satisfies_BC else '✗'}, AC={'✓' if satisfies_AC else '✗'} "
          f"→ {'VALID' if all_satisfied else 'invalid'}")

valid_count = sum(1 for state in states 
                  if (state[0] != state[1]) and (state[1] != state[2]) and (state[0] == state[2]))
print(f"\nNumber of valid joint states: {valid_count}/8")

witnesses = behavior2.worst_case_weights
print(f"\nWitness weights (which contexts constrain agreement):")
for ctx, weight in sorted(witnesses.items(), key=lambda x: x[1], reverse=True):
    if weight > 1e-6:
        print(f"  {ctx}: λ = {weight:.3f}")

print("\n" + "=" * 80)
print("SCENARIO 3: Varying context probabilities")
print("=" * 80)
print("Test: Does K(P) change with context probabilities like I(X;C)?")

# Same contradictory rules, different context weights
context_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []

for w in context_weights:
    # Context probabilities
    p_parity_w = w
    p_roundness_w = 1 - w
    
    # Marginal label distribution
    p_label_0_w = p_parity_w * 0.0 + p_roundness_w * 1.0
    p_label_1_w = p_parity_w * 1.0 + p_roundness_w * 0.0
    
    # Shannon quantities
    H_X_w = shannon_entropy([p_label_0_w, p_label_1_w])
    H_C_w = shannon_entropy([p_parity_w, p_roundness_w])
    I_X_C_w = H_X_w  # Since H(X|C) = 0
    
    results.append({
        'weight': w,
        'H_X': H_X_w,
        'H_C': H_C_w,
        'I_X_C': I_X_C_w
    })
    
print(f"\nContext weight | H(X)   | H(C)   | I(X;C) | K(P)")
print("-" * 60)
for r in results:
    # K(P) doesn't change - it's structural
    print(f"  {r['weight']:.1f}         | {r['H_X']:.4f} | {r['H_C']:.4f} | {r['I_X_C']:.4f} | {behavior1.K:.4f}")

print(f"\nObservations:")
print(f"  K(P) = {behavior1.K:.4f} (constant)")
print(f"  I(X;C) varies: {results[0]['I_X_C']:.4f} to {results[-1]['I_X_C']:.4f}")
print(f"  K(P) matches I(X;C) only at: {[r['weight'] for r in results if abs(r['I_X_C'] - behavior1.K) < 0.01]}")

print("\n" + "=" * 80)
print("SCENARIO 4: Probabilistic disagreement (not deterministic)")
print("=" * 80)

obs4 = Observatory.create(symbols=['0', '1'])
label4 = obs4.concept('Label')

# Weak disagreement: 60% vs 40%
weak_A = obs4.lens('Weak_A')
with weak_A:
    weak_A.perspectives[label4] = {'1': 0.6, '0': 0.4}

weak_B = obs4.lens('Weak_B')
with weak_B:
    weak_B.perspectives[label4] = {'0': 0.6, '1': 0.4}

behavior4_weak = (weak_A | weak_B).to_behavior()

# Strong disagreement: 90% vs 10%
strong_A = obs4.lens('Strong_A')
with strong_A:
    strong_A.perspectives[label4] = {'1': 0.9, '0': 0.1}

strong_B = obs4.lens('Strong_B')
with strong_B:
    strong_B.perspectives[label4] = {'0': 0.9, '1': 0.1}

behavior4_strong = (strong_A | strong_B).to_behavior()

print(f"\nWeak disagreement (60/40 vs 40/60):")
print(f"  K(P) = {behavior4_weak.K:.4f} bits")
print(f"  α*   = {behavior4_weak.alpha_star:.4f}")

print(f"\nStrong disagreement (90/10 vs 10/90):")
print(f"  K(P) = {behavior4_strong.K:.4f} bits")
print(f"  α*   = {behavior4_strong.alpha_star:.4f}")

print(f"\nDeterministic disagreement (100/0 vs 0/100):")
print(f"  K(P) = {behavior1.K:.4f} bits")
print(f"  α*   = {behavior1.alpha_star:.4f}")

print(f"\nK(P) progression: {behavior4_weak.K:.4f} → {behavior4_strong.K:.4f} → {behavior1.K:.4f}")

print("\n" + "=" * 80)
print("SCENARIO 5: Three-way vs two-way contradictions")
print("=" * 80)

# Two contradictory contexts
obs5a = Observatory.create(symbols=['0', '1'])
label5a = obs5a.concept('Label')
ctx1 = obs5a.lens('Ctx1')
with ctx1:
    ctx1.perspectives[label5a] = {'1': 1.0, '0': 0.0}
ctx2 = obs5a.lens('Ctx2')
with ctx2:
    ctx2.perspectives[label5a] = {'0': 1.0, '1': 0.0}

behavior5_two = (ctx1 | ctx2).to_behavior()

# Three contradictory contexts
obs5b = Observatory.create(symbols=['0', '1'])
label5b = obs5b.concept('Label')
ctx1b = obs5b.lens('Ctx1')
with ctx1b:
    ctx1b.perspectives[label5b] = {'1': 1.0, '0': 0.0}
ctx2b = obs5b.lens('Ctx2')
with ctx2b:
    ctx2b.perspectives[label5b] = {'0': 1.0, '1': 0.0}
ctx3b = obs5b.lens('Ctx3')
with ctx3b:
    ctx3b.perspectives[label5b] = {'1': 0.5, '0': 0.5}

behavior5_three = (ctx1b | ctx2b | ctx3b).to_behavior()

print(f"\nTwo contradictory contexts:")
print(f"  K(P) = {behavior5_two.K:.4f} bits")
print(f"  α*   = {behavior5_two.alpha_star:.4f}")

print(f"\nThree contexts (third is 50/50):")
print(f"  K(P) = {behavior5_three.K:.4f} bits")
print(f"  α*   = {behavior5_three.alpha_star:.4f}")

print(f"\nDifference: ΔK = {abs(behavior5_three.K - behavior5_two.K):.4f} bits")

print("\n" + "=" * 80)
print("SCENARIO 6: Product of independent systems")
print("=" * 80)

# System 1: Simple binary contradiction
obs6a = Observatory.create(symbols=['0', '1'])
pred6a = obs6a.concept('Pred1')
lens6a_A = obs6a.lens('Sys1_A')
with lens6a_A:
    lens6a_A.perspectives[pred6a] = {'1': 1.0}
lens6a_B = obs6a.lens('Sys1_B')
with lens6a_B:
    lens6a_B.perspectives[pred6a] = {'0': 1.0}
beh6_1 = (lens6a_A | lens6a_B).to_behavior()

# System 2: Different binary contradiction
obs6b = Observatory.create(symbols=['0', '1'])
pred6b = obs6b.concept('Pred2')
lens6b_X = obs6b.lens('Sys2_X')
with lens6b_X:
    lens6b_X.perspectives[pred6b] = {'1': 1.0}
lens6b_Y = obs6b.lens('Sys2_Y')
with lens6b_Y:
    lens6b_Y.perspectives[pred6b] = {'0': 1.0}
beh6_2 = (lens6b_X | lens6b_Y).to_behavior()

# Product
product_beh = beh6_1 @ beh6_2

print(f"\nSystem 1: K₁ = {beh6_1.K:.4f} bits")
print(f"System 2: K₂ = {beh6_2.K:.4f} bits")
print(f"Product:  K  = {product_beh.K:.4f} bits")
print(f"\nK₁ + K₂ = {beh6_1.K + beh6_2.K:.4f}")
print(f"Difference from additivity: {abs(product_beh.K - (beh6_1.K + beh6_2.K)):.6f}")

print("\n" + "=" * 80)
print("DATA SUMMARY TABLE")
print("=" * 80)
print(f"\n{'Scenario':<40} | {'K(P)':<8} | {'α*':<8} | {'Notes'}")
print("-" * 80)
print(f"{'Binary contradiction (50/50)':<40} | {behavior1.K:<8.4f} | {behavior1.alpha_star:<8.4f} | I(X;C)={I_X_C:.4f}")
print(f"{'Triangle impossibility':<40} | {behavior2.K:<8.4f} | {behavior2.alpha_star:<8.4f} | {valid_count}/8 valid states")
print(f"{'Weak disagreement (60/40)':<40} | {behavior4_weak.K:<8.4f} | {behavior4_weak.alpha_star:<8.4f} |")
print(f"{'Strong disagreement (90/10)':<40} | {behavior4_strong.K:<8.4f} | {behavior4_strong.alpha_star:<8.4f} |")
print(f"{'Two contexts':<40} | {behavior5_two.K:<8.4f} | {behavior5_two.alpha_star:<8.4f} |")
print(f"{'Three contexts':<40} | {behavior5_three.K:<8.4f} | {behavior5_three.alpha_star:<8.4f} |")
print(f"{'Product of two systems':<40} | {product_beh.K:<8.4f} | {product_beh.alpha_star:<8.4f} | K₁+K₂={beh6_1.K + beh6_2.K:.4f}")