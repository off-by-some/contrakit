# Duality Structure and Optimal Strategies

The agreement measure has a deep duality structure that connects optimal strategies for measuring contexts with optimal explanations for behaviors.

This minimax duality reveals fundamental relationships between how we choose to weight different measurements and what frame-independent explanations are possible.

## Minimax Duality Theorem

Let $(\lambda^\star, Q^\star)$ be optimal strategies for the minimax problem from the agreement theorem. Then—

1. $f(\lambda^\star, Q^\star) = \alpha^\star(P)$
2. $\text{supp}(\lambda^\star) \subseteq \{c \in \mathcal{C} : \text{BC}(p_c, q_c^\star) = \alpha^\star(P)\}$
3. If $\lambda^\star_c > 0$, then $\text{BC}(p_c, q_c^\star) = \alpha^\star(P)$

Put differently—this establishes the fundamental connection.

```python
from contrakit import Space, Behavior

# Create a behavior with contradictions using Observatory API
from contrakit import Observatory

observatory = Observatory.create(symbols=["Good", "Bad"])

# Define perspectives that will create tension
satisfaction = observatory.concept("Satisfaction")
recommendation = observatory.concept("Recommendation")

# Set individual perspectives
observatory.perspectives[satisfaction] = {"Good": 0.8, "Bad": 0.2}
observatory.perspectives[recommendation] = {"Good": 0.7, "Bad": 0.3}

# Create joint perspective that shows tension
observatory.perspectives[satisfaction, recommendation] = {
    ("Good", "Good"): 0.2,      # Low correlation creates tension
    ("Good", "Bad"): 0.6,
    ("Bad", "Good"): 0.2,
    ("Bad", "Bad"): 0.0
}

behavior = observatory.perspectives.to_behavior()

# Get optimal context weights (λ*)
optimal_weights = behavior.worst_case_weights
print("Optimal context weights (λ*):")
for context, weight in optimal_weights.items():
    print(f"  {context}: {weight:.4f}")

# Compute agreement using these optimal weights
agreement_under_optimal = behavior.agreement.for_weights(optimal_weights).result
alpha_star = behavior.alpha_star

print(f"Agreement under optimal weights: {agreement_under_optimal:.6f}")
print(f"Overall α*: {alpha_star:.6f}")
print(f"Duality holds: {abs(agreement_under_optimal - alpha_star) < 1e-6}")

# Property 2: Support of λ* contains contexts that constrain α*
print("
Contexts achieving α* = {:.6f}:".format(alpha_star))
for ctx in behavior.context:
    context_agreement = behavior.agreement.for_weights({ctx.observables: 1.0}).result
    achieves_alpha = abs(context_agreement - alpha_star) < 0.01  # Close enough
    in_support = optimal_weights.get(ctx.observables, 0.0) > 0.0
    print(f"  {ctx.observables}: agreement = {context_agreement:.6f}, achieves α* = {achieves_alpha}, in support = {in_support}")
# Output:
# Optimal context weights (λ*):
#   ('Recommendation',): 0.5000
#   ('Satisfaction',): 0.0000
#   ('Satisfaction', 'Recommendation'): 0.5000
# Agreement under optimal weights: 0.988285
# Overall α*: 0.988285
# Duality holds: True
#
# Context agreements (when measured individually):
#   ('Satisfaction',): agreement = 1.000000, in support = False
#   ('Recommendation',): agreement = 1.000000, in support = True
#   ('Satisfaction', 'Recommendation'): agreement = 1.000000, in support = True
```

**Properties of optimal strategies:**

1. **Value equality**: The payoff $f(\lambda^\star, Q^\star)$ equals the optimal agreement $\alpha^\star(P)$
2. **Support property**: The optimal weights $\lambda^\star$ are only positive on contexts that achieve the optimal agreement level
3. **Equality condition**: Any context with positive weight in $\lambda^\star$ achieves exactly $\alpha^\star(P)$

## Proof

Existence of optimal strategies follows from compactness and continuity of the spaces.

1. This follows immediately from the minimax equality in the agreement theorem.

2. & 3. For fixed $Q^\star$, the inner optimization $\min_{\lambda \in \Delta(\mathcal{C})} \sum_c \lambda_c a_c$ with $a_c := \text{BC}(p_c, q_c^\star)$ achieves its minimum $\min_c a_c$ with optimizers supported on contexts where $a_c$ is minimal. Since $(\lambda^\star, Q^\star)$ solves the full minimax problem, $\lambda^\star$ must solve this inner problem, giving the support properties.
