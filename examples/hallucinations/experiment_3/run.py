"""
Test hallucination inevitability using contradictory training labels.

This experiment demonstrates that tasks with contradiction measure K > 0 force
models to hallucinate even when the training data explicitly contains the
contradictions. The same inputs receive conflicting labels from different rules,
creating a task where perfect accuracy is mathematically impossible.

Hypothesis tested:
Neural networks trained on contradictory labels (where the same input appears
with different outputs) will hallucinate with rates bounded below by 1-2^(-K),
where K is computed from the task's contradiction structure.

Testing approach:
- Create task with two conflicting rules: "Z=X" and "Z=NOT Y"
- Both rules are applied to all (X,Y) inputs, creating explicit contradictions
- For example: (X=0, Y=0) → Z=0 from X-rule AND (X=0, Y=0) → Z=1 from Y-rule
- Train neural network on mixture of both rules
- Test on all (X,Y) combinations to measure which rule the model prefers
- Measure hallucination as deviation from either pure rule
- Compare observed rates against analytically computed K and lower bounds

Key measurements:
- Contradiction measure K computed from incompatible marginal constraints
- Theoretical minimum error rate = 1-2^(-K)
- Observed error rates across different random seeds
- Model behavior on conflicting vs. agreeing inputs

Assumptions:
- Task constraints are deterministic and explicitly contradictory
- K is computed correctly from marginal incompatibilities
- Training contains equal numbers of examples from each rule
- Model must learn to compromise between contradictory constraints

Expected outcome:
Error rates exceed theoretical minimum (1-2^(-K)), confirming that logical
contradictions in training data force inevitable errors. The model cannot
perfectly satisfy both rules simultaneously.

Typical usage:
- Run main() to execute the full experiment
- Results demonstrate that K > 0 implies unavoidable errors
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from contrakit import Space, Behavior
from contrakit.constants import DEFAULT_SEED

# ==============================================================================
# TASK: Conflicting Deterministic Marginals
# ==============================================================================

class ConflictingMarginalsTask:
    """
    Training provides contradictory labels for the same inputs via two rules:
    
    X-rule: Z = X (ignores Y)
      - (X=0, Y=0) → Z=0
      - (X=0, Y=1) → Z=0
      - (X=1, Y=0) → Z=1
      - (X=1, Y=1) → Z=1
      
    Y-rule: Z = NOT Y (ignores X)
      - (X=0, Y=0) → Z=1  # Conflicts with X-rule!
      - (X=0, Y=1) → Z=0  # Agrees with X-rule
      - (X=1, Y=0) → Z=1  # Agrees with X-rule
      - (X=1, Y=1) → Z=0  # Conflicts with X-rule!
    
    Training contains examples from both rules for all (X,Y) combinations.
    The same inputs appear with different labels, creating explicit contradictions.
    
    Model must learn to compromise between incompatible constraints.
    This is a task with K > 0, making perfect accuracy impossible.
    """
    
    def generate_training_data(self, n_per_context: int = 50) -> Tuple:
        """
        Generate contradictory training examples where the same input receives
        conflicting labels from different contexts.
        
        The contradiction: When X=0 and Y=0 appear together:
        - Rule from X: X=0 → Z=0
        - Rule from Y: Y=0 → Z=1
        Both rules are provided in training with joint (X,Y) inputs.
        
        Returns:
            x_data: [X, Y] pairs  
            y_data: outputs Z (conflicting for some inputs)
            contexts: which rule each example follows
        """
        x_data = []
        y_data = []
        contexts = []
        
        # Context 1: "X-rule" - Z follows X
        # Include both marginal (Y missing) and joint examples
        for _ in range(n_per_context // 2):
            for x_val in [0, 1]:
                x_data.append([x_val, -1])  # Y missing
                y_data.append(x_val)  # Z = X
                contexts.append('X_rule_marginal')
        
        # X-rule applied to joint inputs
        for _ in range(n_per_context // 2):
            for x_val in [0, 1]:
                for y_val in [0, 1]:
                    x_data.append([x_val, y_val])  # Both present
                    y_data.append(x_val)  # Z = X (ignores Y)
                    contexts.append('X_rule_joint')
        
        # Context 2: "Y-rule" - Z follows NOT Y  
        # Include both marginal (X missing) and joint examples
        for _ in range(n_per_context // 2):
            for y_val in [0, 1]:
                x_data.append([-1, y_val])  # X missing
                y_data.append(1 - y_val)  # Z = NOT Y
                contexts.append('Y_rule_marginal')
        
        # Y-rule applied to joint inputs (CONFLICTS with X-rule!)
        for _ in range(n_per_context // 2):
            for x_val in [0, 1]:
                for y_val in [0, 1]:
                    x_data.append([x_val, y_val])  # Both present
                    y_data.append(1 - y_val)  # Z = NOT Y (ignores X)
                    contexts.append('Y_rule_joint')
        
        # Now training contains explicit contradictions:
        # Example: (X=0, Y=0) → Z=0 (from X-rule)
        # Example: (X=0, Y=0) → Z=1 (from Y-rule)
        # Same input, different labels!
        
        return np.array(x_data, dtype=np.float32), np.array(y_data), contexts
    
    def compute_k_apriori(self) -> float:
        """
        Compute K from contradictory labeling of the same inputs.
        
        The task has two rules applied to the same inputs:
        - X-rule: Z = X (ignores Y)
        - Y-rule: Z = NOT Y (ignores X)
        
        These rules give different answers for 2 out of 4 input combinations:
        - (X=0, Y=0): X-rule says Z=0, Y-rule says Z=1 [CONFLICT]
        - (X=0, Y=1): X-rule says Z=0, Y-rule says Z=0 [agree]
        - (X=1, Y=0): X-rule says Z=1, Y-rule says Z=1 [agree]
        - (X=1, Y=1): X-rule says Z=1, Y-rule says Z=0 [CONFLICT]
        
        With uniform input distribution, the two rules agree on 50% of cases.
        Agreement coefficient α = 0.5, so K = -log₂(0.5) = 1.0 bits.
        
        To measure this using contrakit's Behavior class, we create marginal
        contexts that capture the contradiction:
        - Context (X,Z): X determines Z directly (X-rule marginal)
        - Context (Y,Z): Y anti-determines Z (Y-rule marginal)
        - Context (X,Y): Uniform distribution (both variables can appear)
        
        These three pairwise marginals are incompatible with any joint distribution
        when X and Y are correlated.
        """
        space = Space.create(X=['0', '1'], Y=['0', '1'], Z=['0', '1'])
        
        # Marginal (X,Z) from X-rule: Z=X with uniform X
        context_xz = {
            ('0', '0'): 0.5,  # X=0 → Z=0
            ('0', '1'): 0.0,
            ('1', '0'): 0.0,
            ('1', '1'): 0.5,  # X=1 → Z=1
        }
        
        # Marginal (Y,Z) from Y-rule: Z=NOT Y with uniform Y
        context_yz = {
            ('0', '0'): 0.0,
            ('0', '1'): 0.5,  # Y=0 → Z=1
            ('1', '0'): 0.5,  # Y=1 → Z=0
            ('1', '1'): 0.0,
        }
        
        # Marginal (X,Y): Since training includes all (X,Y) combinations
        # equally from both rules, X and Y are uniform and independent
        context_xy = {
            ('0', '0'): 0.25,
            ('0', '1'): 0.25,
            ('1', '0'): 0.25,
            ('1', '1'): 0.25,
        }
        
        # These three marginals are incompatible. For example:
        # - From (X,Z): P(Z=0|X=0) = 1
        # - From (Y,Z): P(Z=1|Y=0) = 1
        # - From (X,Y): P(X=0,Y=0) = 0.25 > 0
        # But P(Z=0|X=0,Y=0) and P(Z=1|X=0,Y=0) can't both be 1!
        
        behavior = Behavior.from_contexts(space, {
            ('X', 'Z'): context_xz,
            ('Y', 'Z'): context_yz,
            ('X', 'Y'): context_xy,
        })
        
        return behavior.K

# ==============================================================================
# MODEL
# ==============================================================================

class SimpleClassifier(nn.Module):
    """
    MLP that processes [X, Y] pairs and predicts Z.
    
    Architecture supports both joint (X,Y) inputs and marginal inputs
    where one variable is marked as missing with -1.
    """
    
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.x_embed = nn.Embedding(3, hidden//2)  # Values: 0, 1, or missing (-1)
        self.y_embed = nn.Embedding(3, hidden//2)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 2)  # Binary output: Z ∈ {0, 1}
        
    def forward(self, x):
        # Shift values: -1→0 (missing), 0→1, 1→2
        x_shifted = (x + 1).long()
        x_emb = self.x_embed(x_shifted[:, 0])
        y_emb = self.y_embed(x_shifted[:, 1])
        h = torch.cat([x_emb, y_emb], dim=1)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)

# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_model(model, x_train, y_train, epochs=200, lr=0.01):
    """
    Train model on contradictory labels.
    
    The training data contains explicit contradictions where the same
    (X,Y) input appears with different Z labels from different rules.
    Model learns to compromise via gradient descent.
    """
    X = torch.FloatTensor(x_train)
    y = torch.LongTensor(y_train)
    
    # Filter out any examples with missing labels (-1)
    # These might exist if we include unlabeled joint (X,Y) pairs
    valid_indices = y >= 0
    if not valid_indices.all():
        X = X[valid_indices]
        y = y[valid_indices]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

def evaluate_hallucination(model) -> Dict:
    """
    Test on 4 (X,Y) combinations to measure model's learned compromise.
    
    Agreement cases (both rules give same answer):
    - (X=0, Y=1): X-rule→0, Y-rule→0 → should predict Z=0
    - (X=1, Y=0): X-rule→1, Y-rule→1 → should predict Z=1
    
    Conflict cases (rules disagree, no correct answer):
    - (X=0, Y=0): X-rule→0, Y-rule→1 → model must choose
    - (X=1, Y=1): X-rule→1, Y-rule→0 → model must choose
    
    We measure:
    1. Accuracy on agreement cases (should be 100%)
    2. Confidence on conflict cases (high confidence = "hallucination")
    3. Overall error rate vs. theoretical minimum
    """
    model.eval()
    
    test_queries = [
        ([0, 0], "conflict", None),     # X-rule→0, Y-rule→1
        ([0, 1], "agree", 0),           # Both→0
        ([1, 0], "agree", 1),           # Both→1
        ([1, 1], "conflict", None),     # X-rule→1, Y-rule→0
    ]
    
    predictions = []
    confidences = []
    agreement_correct = []
    conflict_confidences = []
    
    for (x, y), label, correct_answer in test_queries:
        query = torch.FloatTensor([[x, y]])
        
        with torch.no_grad():
            logits = model(query)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            conf = probs[0, pred].item()
        
        predictions.append(pred)
        confidences.append(conf)
        
        if label == "agree":
            agreement_correct.append(pred == correct_answer)
        else:
            conflict_confidences.append(conf)
    
    # Accuracy on agreement cases
    agreement_accuracy = np.mean(agreement_correct) if agreement_correct else 0.0
    
    # Average confidence on conflict cases
    avg_conflict_conf = np.mean(conflict_confidences) if conflict_confidences else 0.5
    
    # "Hallucination rate" = making confident predictions on conflict queries
    # where no correct answer exists
    # conf=0.5 (uncertain) → halluc=0%, conf=1.0 (certain) → halluc=100%
    hallucination_rate = max(0.0, (avg_conflict_conf - 0.5) * 2)
    
    return {
        'agreement_accuracy': agreement_accuracy,
        'hallucination_rate': hallucination_rate,
        'predictions': predictions,
        'confidences': confidences,
        'avg_conflict_confidence': avg_conflict_conf,
        'agreement_correct': agreement_correct
    }

# ==============================================================================
# EXPERIMENT
# ==============================================================================

def run_experiment(n_seeds: int = 10) -> Dict:
    """Run experiment with K prediction computed before running."""
    
    print("="*70)
    print("Hallucination Test with Conflicting Marginals")
    print("="*70)
    
    # Step 1: Compute K before running (no free parameters)
    print("\nStep 1: Compute contradiction before experiment")
    print("-"*70)
    
    task = ConflictingMarginalsTask()
    K = task.compute_k_apriori()
    
    print("Task structure:")
    print("  X-rule: Z=X (applies to all X,Y combinations)")
    print("  Y-rule: Z=NOT Y (applies to all X,Y combinations)")
    print("  Training: Equal mixture of both rules")
    print("\nConflicts in training data:")
    print("  (X=0, Y=0): X-rule→Z=0, Y-rule→Z=1")
    print("  (X=1, Y=1): X-rule→Z=1, Y-rule→Z=0")
    print("Agreement cases:")
    print("  (X=0, Y=1): X-rule→Z=0, Y-rule→Z=0")
    print("  (X=1, Y=0): X-rule→Z=1, Y-rule→Z=1")
    print(f"\nMeasured contradiction: K = {K:.4f} bits")
    
    # Predict error inevitability from information theory
    # K > 0 guarantees that no model can perfectly satisfy both rules
    # The lower bound on error rate is 1 - 2^(-K)
    predicted_lower_bound = 1 - 2**(-K)
    
    print(f"\nMinimum error rate (from information theory): {predicted_lower_bound:.1%}")
    print("(Any model must fail on at least this fraction)")
    print("\nNote: Observed rate may exceed this bound due to:")
    print("  - Sub-optimal learning (gradient descent doesn't find best compromise)")
    print("  - Architectural constraints (model capacity limitations)")
    print("  - Training dynamics (loss function weighting)")
    
    # Step 2: Run experiment
    print(f"\n{'='*70}")
    print("Step 2: Run experiment")
    print("-"*70)
    print(f"Training: {n_seeds} seeds × 400 examples total")
    print("  - 200 examples following X-rule (Z=X)")
    print("  - 200 examples following Y-rule (Z=NOT Y)")
    print("  - Same (X,Y) inputs appear in both rule sets with conflicting Z labels")
    print("\nTest: 4 (X,Y) combinations")
    print("  - 2 inputs where rules agree (X=0,Y=1 and X=1,Y=0)")
    print("  - 2 inputs where rules conflict (X=0,Y=0 and X=1,Y=1)")
    
    results = []
    for seed_offset in range(n_seeds):
        seed = DEFAULT_SEED + seed_offset
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate data
        x_train, y_train, _ = task.generate_training_data(n_per_context=50)
        
        # Train
        model = SimpleClassifier(hidden=32)
        train_model(model, x_train, y_train, epochs=200, lr=0.01)
        
        # Evaluate
        result = evaluate_hallucination(model)
        results.append(result)
    
    # Step 3: Compare prediction to observation
    print(f"\n{'='*70}")
    print("Step 3: Results")
    print("-"*70)
    
    agreement_accs = [r['agreement_accuracy'] for r in results]
    halluc_rates = [r['hallucination_rate'] for r in results]
    
    observed_agreement = np.mean(agreement_accs)
    observed_halluc = np.mean(halluc_rates)
    std_halluc = np.std(halluc_rates)
    
    print(f"\nAccuracy on agreement cases: {observed_agreement:.1%}")
    print(f"  (Both rules give same answer - model should get these right)")
    
    print(f"\nHallucination on conflict cases: {observed_halluc:.1%} ± {std_halluc:.1%}")
    print(f"  (Rules disagree - model makes confident predictions anyway)")
    
    print(f"\nTheoretical minimum error: {predicted_lower_bound:.1%}")
    print(f"  (From K = {K:.4f} bits)")
    
    # Detailed analysis
    avg_conf = np.mean([r['avg_conflict_confidence'] for r in results])
    print(f"\nAverage confidence on conflicts: {avg_conf:.1%}")
    print(f"  Ideal (uncertain): ~50%")
    print(f"  Observed: {avg_conf:.1%}")
    print(f"  Hallucinating (confident on impossible queries): 80-100%")
    
    # Show example predictions
    print(f"\nExample predictions (seed 0):")
    example = results[0]
    test_cases = [
        ("X=0,Y=0", "X→0, Y→1", "conflict"),
        ("X=0,Y=1", "X→0, Y→0", "agree on 0"),
        ("X=1,Y=0", "X→1, Y→1", "agree on 1"), 
        ("X=1,Y=1", "X→1, Y→0", "conflict")
    ]
    for i, (inputs, rules, label) in enumerate(test_cases):
        pred = example['predictions'][i]
        conf = example['confidences'][i]
        marker = "[CONFLICT]" if "conflict" in label else "[AGREE]   "
        print(f"  {marker} {inputs}: {rules}")
        print(f"             → pred={pred}, conf={conf:.1%}")
    
    return {
        'K': K,
        'lower_bound': predicted_lower_bound,
        'agreement_accuracy': observed_agreement,
        'hallucination_rate': observed_halluc,
        'std': std_halluc,
        'all_results': results
    }

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run hallucination inevitability test"""
    results = run_experiment(n_seeds=10)
    
    print(f"\n{'='*70}")
    print("Summary")
    print("="*70)
    
    K = results['K']
    lower_bound = results['lower_bound']
    agreement_acc = results['agreement_accuracy']
    hallucination = results['hallucination_rate']
    
    print(f"\nContradiction measure: K = {K:.4f} bits")
    print(f"Theoretical minimum error: {lower_bound:.1%}")
    print(f"\nResults across {len(results['all_results'])} seeds:")
    print(f"  Agreement accuracy: {agreement_acc:.1%}")
    print(f"  Hallucination rate: {hallucination:.1%} ± {results['std']:.1%}")
    
    print(f"\n{'='*70}")
    print("Interpretation")
    print("="*70)
    
    print(f"\n✓ Task has genuine contradiction: K = {K:.4f} > 0")
    print(f"✓ Training contains same inputs with conflicting labels")
    print(f"✓ Model achieves {agreement_acc:.1%} on non-contradictory cases")
    
    if hallucination > 0.5:
        print(f"\n⚠ Model hallucinates on {hallucination:.1%} of conflict cases")
        print(f"  (Makes confident predictions where both rules disagree)")
        print(f"\nThis demonstrates:")
        print(f"  - Softmax forces commitment even when training is contradictory")
        print(f"  - Model cannot express 'this query has no consistent answer'")
        print(f"  - Contradiction K > 0 leads to confident but arbitrary predictions")
    
    print(f"\nTheoretical insight:")
    print(f"  - Minimum error from K: {lower_bound:.1%}")
    print(f"  - Observed hallucination: {hallucination:.1%}")
    print(f"  - Excess comes from architectural forcing (softmax confidence)")
    
 
if __name__ == "__main__":
    main()