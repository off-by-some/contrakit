# Contrakit Cheatsheet: Learn Contradiction Theory in Y Minutes

A comprehensive guide to measuring contradictions in multi-perspective data. All examples incur actual contradiction.

## 1. What is Contradiction?

**Contradiction Theory** quantifies when valid perspectives cannot be reconciled.

```python
from contrakit import Space, Behavior

# Perfect triangle of disagreement: A≠B, B≠C, A=C creates maximum contradiction
space = Space.create(A=['0','1'], B=['0','1'], C=['0','1'])
behavior = Behavior.from_contexts(space, {
    ('A','B'): {('0','1'): 1.0, ('1','0'): 0.0},  # A and B disagree
    ('B','C'): {('0','1'): 1.0, ('1','0'): 0.0},  # B and C disagree
    ('A','C'): {('0','0'): 1.0, ('1','1'): 0.0}   # A and C agree
})
print(f"K = {behavior.K:.4f} bits (α* = {behavior.alpha_star:.3f})")
```

**Key Measures**:
- **K**: Contradiction in bits (0 = no contradiction)
- **α***: Best agreement coefficient (how well perspectives can be reconciled)
- **Witnesses**: Where contradictions concentrate

## 2. Classic Examples

#### Hiring Paradox (Observer Conflicts)
```python
# Three interviewers with conflicting hiring criteria
hiring = Behavior.from_contexts(Space.create(Exp=['H','R'], Skill=['H','R'], Ref=['H','R']), {
    ('Exp','Skill'): {('H','R'): 0.9, ('R','H'): 0.1},  # Experience vs skills
    ('Skill','Ref'): {('H','R'): 0.9, ('R','H'): 0.1},  # Skills vs references
    ('Exp','Ref'): {('H','H'): 0.9, ('R','R'): 0.1}     # Experience matches references
})
print(f"Hiring conflict: K = {hiring.K:.4f} bits")
```

#### Simpson's Paradox (Contextual Reversal)
```python
# Different observers see same data differently
from contrakit import Observatory
obs = Observatory.create(symbols=['Good','Bad'])
alice = obs.lens('Alice', symbols=['Reliable','Unreliable'])
bob = obs.lens('Bob', symbols=['Consistent','Inconsistent'])
candidate = obs.concept('Candidate')

# Alice: pro-hire, Bob: anti-hire
with alice: alice.perspectives[candidate] = {'Good': 0.8, 'Bad': 0.2}
with bob: bob.perspectives[candidate] = {'Good': 0.2, 'Bad': 0.8}

composed = (alice | bob).to_behavior()  # Compose conflicting views
print(f"Observer conflict: K = {composed.K:.4f} bits")
```

#### Simpson's Paradox (Contextual Reversal)
```python
# Three-way inconsistency creates paradox
paradox = Behavior.from_contexts(Space.create(Treatment=['A','B'], Group=['G1','G2'], Outcome=['Pos','Neg']), {
    ('Treatment','Group'): {('A','G1'): 0.9, ('B','G1'): 0.1, ('A','G2'): 0.2, ('B','G2'): 0.8},
    ('Group','Outcome'): {('G1','Pos'): 0.9, ('G1','Neg'): 0.1, ('G2','Pos'): 0.2, ('G2','Neg'): 0.8},
    ('Treatment','Outcome'): {('A','Pos'): 0.55, ('A','Neg'): 0.45, ('B','Pos'): 0.45, ('B','Neg'): 0.55}
})
print(f"Three-way paradox: K = {paradox.K:.4f} bits")
```

## 3. Key Operations

#### Find Contradiction Witnesses
```python
# Identify which perspectives drive the contradiction
witnesses = behavior.worst_case_weights
for ctx, weight in sorted(witnesses.items(), key=lambda x: x[1], reverse=True)[:3]:
    if weight > 1e-6:
        print(f"Witness {ctx}: λ = {weight:.3f}")
```

#### Check Frame Independence
```python
# Test if perspectives can be reconciled in single framework
fi_behavior = Behavior.frame_independent(space, [['A'], ['B'], ['C'], ['A','B'], ['B','C'], ['A','C']])
print(f"Can reconcile: {fi_behavior.K < 1e-6}")
```

#### Compression Costs
```python
# Information cost of reconciling contradictory perspectives
import math
ctx = ('A','B')
probs = list(behavior[ctx].to_dict().values())
entropy = sum(p * (-math.log2(p)) for p in probs if p > 0)
total_cost = entropy + behavior.K  # Base entropy + contradiction penalty
print(f"Reconciliation cost: {total_cost:.3f} bits")
```

## 4. Theorem Demonstrations

#### Weakest Link Principle (Theorem 1)
```python
# Agreement limited by worst context
from contrakit import Space, Behavior
space = Space.create(A=['0','1'], B=['0','1'], C=['0','1'])
behavior = Behavior.from_contexts(space, {
    ('A','B'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('B','C'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('A','C'): {('0','0'): 1.0, ('1','1'): 0.0}
})
print(f"Overall α* = {behavior.alpha_star:.4f}")
# Shows α* limited by weakest pairwise agreement
```

#### Contradiction is a Game (Theorem 2)
```python
# Minimax adversarial structure
witnesses = behavior.worst_case_weights
print("Adversarial context weights:")
for ctx, weight in sorted(witnesses.items(), key=lambda x: x[1], reverse=True)[:2]:
    print(f"  {ctx}: λ = {weight:.3f}")
print(f"Minimax: α* = {behavior.alpha_star:.4f}")
```

#### Independence Additivity (Theorem 5)
```python
# Contradictions add on independent systems (using triangle behaviors)
beh1 = Behavior.from_contexts(Space.create(A=['0','1'], B=['0','1'], C=['0','1']), {
    ('A','B'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('B','C'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('A','C'): {('0','0'): 1.0, ('1','1'): 0.0}
})
beh2 = Behavior.from_contexts(Space.create(X=['0','1'], Y=['0','1'], Z=['0','1']), {
    ('X','Y'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('Y','Z'): {('0','1'): 1.0, ('1','0'): 0.0},
    ('X','Z'): {('0','0'): 1.0, ('1','1'): 0.0}
})
product = (beh1 @ beh2).K
print(f"K₁ + K₂ = {(beh1.K + beh2.K):.4f}, Product = {product:.4f}")
```

#### Contradiction Geometry (Theorem 15)
```python
# Triangle inequality for Hellinger angles in triangle behavior
import math
scores = behavior.per_context_scores()  # Per-context BC scores
def hellinger_angle(bc_score):
    return math.acos(bc_score)
J_AB, J_BC, J_AC = [hellinger_angle(s) for s in scores[:3]]
print(f"J(AC) = {J_AC:.3f} ≤ J(AB) + J(BC) = {J_AB + J_BC:.3f}")
```

## 6. Theoretical Foundations

#### Theoretical Bounds
```python
# Minimum achievable agreement coefficient
max_outcomes = max(len(space.alphabets[name]) for name in space.names)
min_alpha = max_outcomes ** (-0.5)
print(f"α* ≥ {min_alpha:.3f}, achieved = {behavior.alpha_star:.3f}")
```

#### Independent Systems
```python
# Contradictions add when systems are independent
space1 = Space.create(X=['A','B'])
beh1 = Behavior.from_contexts(space1, {('X',): {('A',): 0.6, ('B',): 0.4}})
space2 = Space.create(Y=['C','D'])
beh2 = Behavior.from_contexts(space2, {('Y',): {('C',): 0.7, ('D',): 0.3}})
product = (beh1 @ beh2).K  # Tensor product
print(f"K₁ + K₂ = {(beh1.K + beh2.K):.4f}, Product = {product:.4f}")
```

#### Witness-Error Tradeoff
```python
# Statistical power vs coordination requirements
witness_rate = sum(behavior.worst_case_weights.values())
error_exponent = max(0, behavior.K - witness_rate)
detection_threshold = 2**(-behavior.K)  # Minimum detectable effect
print(f"Error exponent: {error_exponent:.4f}, Detection: {detection_threshold:.2e}")
```

## 6. Advanced Features

#### Observatory API for Complex Systems
```python
from contrakit.observatory import Observatory

# High-level API for building complex observational systems
obs = Observatory.create(symbols=['Good', 'Bad'])
quality = obs.concept('Quality', symbols=['High', 'Low'])
decision = obs.concept('Decision')

obs.perspectives[quality] = {'High': 0.7, 'Low': 0.3}
obs.perspectives[decision] = {'Good': 0.6, 'Bad': 0.4}

# Lens for conditional analysis
with obs.lens(quality) as quality_lens:
    candidate = quality_lens.define('Candidate')
    quality_lens.perspectives[candidate] = {'Good': 0.8, 'Bad': 0.2}
    lens_behavior = quality_lens.to_behavior()

behavior = obs.perspectives.to_behavior()
print(f"Complex system: K = {behavior.K:.4f} bits")
```

#### Advanced Behavior Operations
```python
# Mix behaviors (weighted combination)
biased = Behavior.from_contexts(Space.create(Coin=['H','T']), {('Coin',): {('H',): 0.7, ('T',): 0.3}})
fair = Behavior.from_contexts(Space.create(Coin=['H','T']), {('Coin',): {('H',): 0.5, ('T',): 0.5}})
mixed = biased.mix(fair, 0.3)  # 70% biased, 30% fair

# Rename observables
renamed = biased.rename_observables({'Coin': 'Flip'})

# Compare behaviors
distance = biased.product_l1_distance(fair)
print(f"L1 distance: {distance:.3f}")
```

#### Working with Count Data
```python
# Create behavior from survey counts
count_data = {
    ('Morning',): {('Sunny',): 70, ('Rainy',): 30},
    ('Evening',): {('Sunny',): 65, ('Rainy',): 35},
    ('Morning', 'Evening'): {
        ('Sunny', 'Sunny'): 45, ('Sunny', 'Rainy'): 25,
        ('Rainy', 'Sunny'): 20, ('Rainy', 'Rainy'): 10
    }
}
weather_space = Space.create(Morning=['Sunny','Rainy'], Evening=['Sunny','Rainy'])
count_behavior = Behavior.from_counts(weather_space, count_data, normalize="per_context")
```

## 8. Practical Applications

#### Decision Paradoxes
```python
# Medical diagnosis with conflicting symptoms and tests
diagnosis = Behavior.from_contexts(Space.create(Symp=['Pos','Neg'], Test=['Pos','Neg'], Disease=['Yes','No']), {
    ('Symp','Test'): {('Pos','Neg'): 0.8, ('Neg','Pos'): 0.2},    # Symptoms contradict test
    ('Test','Disease'): {('Pos','Yes'): 0.9, ('Neg','No'): 0.1},  # Test predicts disease
    ('Symp','Disease'): {('Pos','Yes'): 0.1, ('Neg','No'): 0.9}   # Symptoms don't predict disease
})
print(f"Diagnostic paradox: K = {diagnosis.K:.4f} bits")
```

#### Quantum Contextuality
```python
# CHSH inequality violation (quantum correlations exceed classical bounds)
chsh = Behavior.from_contexts(Space.create(A0=['+1','-1'], A1=['+1','-1'], B0=['+1','-1'], B1=['+1','-1']), {
    ('A0','B0'): {('+1','-1'): 0.5, ('-1','+1'): 0.5},  # Perfect anticorrelation
    ('A0','B1'): {('+1','-1'): 0.5, ('-1','+1'): 0.5},
    ('A1','B0'): {('+1','-1'): 0.5, ('-1','+1'): 0.5},
    ('A1','B1'): {('+1','+1'): 0.5, ('-1','-1'): 0.5}   # Perfect correlation
})
print(f"CHSH violation: K = {chsh.K:.4f} bits")
```

#### Multi-Agent Coordination
```python
# Competing agents with incompatible objectives
agents = Behavior.from_contexts(Space.create(AgentA=['Win','Lose'], AgentB=['Win','Lose'], Outcome=['Good','Bad']), {
    ('AgentA','AgentB'): {('Win','Lose'): 0.8, ('Lose','Win'): 0.2},  # Zero-sum game
    ('AgentB','Outcome'): {('Win','Good'): 0.9, ('Lose','Bad'): 0.1}, # B's win = good outcome
    ('AgentA','Outcome'): {('Win','Bad'): 0.9, ('Lose','Good'): 0.1}  # A's win = bad outcome
})
print(f"Coordination conflict: K = {agents.K:.4f} bits")
```

## 9. Key Formulas

**Contradiction Measure**: $K(P) = -\log_2(\alpha^*(P))$

**Agreement Coefficient**: $\alpha^*(P) = \max_{Q \in \mathrm{FI}} \min_{c} \mathrm{BC}(p_c, q_c)$

**Witness-Error Tradeoff**: $E^*(r) = K - r$ for coordination rate $r$

**Product Rule**: $K(P \otimes R) = K(P) + K(R)$

**Bounds**: $0 \leq K(P) \leq \frac{1}{2} \log_2(\max_c |\mathcal{O}_c|)$

## 10. Contradiction Patterns

- **Triangle Inequalities**: A≠B, B≠C, A=C creates maximum inconsistency
- **Observer Lenses**: Different perspectives on identical observables
- **Contextual Reversals**: Effects flip when context variables are included
- **Zero-Sum Games**: Competing agents with incompatible objectives
- **Quantum Violations**: Correlations exceeding classical bounds
- **Diagnostic Paradoxes**: Symptoms contradict definitive tests

## 11. Best Practices

- **Verify Contradiction**: All examples show K > 0.1 bits of real inconsistency
- **Sequential Execution**: Code samples work when run in order (variables in scope)
- **Witness Analysis**: Use `worst_case_weights` to identify constraining contexts
- **Frame Independence**: Compare with reconciled models to test resolvability
- **Triangle Patterns**: A≠B, B≠C, A=C maximizes contradiction detection

See `/examples/` for extended examples and `/tests/` for comprehensive validation.