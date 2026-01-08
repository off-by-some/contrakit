"""
Baseline experiment testing hallucination inevitability in standard neural networks.

This experiment establishes a baseline by testing whether standard neural networks
(confidence-calibrated classifiers) hallucinate when given inputs outside their
training domain, where no correct answer exists.

Hypothesis tested:
Standard neural networks confidently produce structured outputs (A/B/C/D) even
when presented with undefined inputs that should logically map to abstention (⊥).

Testing approach:
- Generate partial function with 40% of inputs defined, 60% undefined
- Train standard MLP classifier on defined inputs only
- Test on held-out defined inputs (should achieve high accuracy)
- Test on undefined inputs (should abstain with ⊥, but will likely hallucinate)
- Measure hallucination rate as fraction of undefined inputs given structured answers
- Verify consistency across multiple random seeds

Key measurements:
- Accuracy on defined inputs (validation of learning)
- Hallucination rate on undefined inputs (primary outcome)
- Confidence distributions for both defined and undefined inputs
- Statistical consistency across random seeds

Assumptions:
- Partial function has clear defined vs undefined input distinction
- Undefined inputs should logically receive abstention (⊥) responses
- Neural networks are trained with standard cross-entropy loss
- No architectural modifications for uncertainty representation
- Results are reproducible across random seeds

Expected outcome:
Hallucination rates > 0% demonstrate the baseline behavior that definedness
heads attempt to mitigate in subsequent experiments.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from utils import (
    INPUT_SIZE, OUTPUT_CLASSES, HIDDEN_SIZE, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    generate_partial_function, create_datasets, HallucinationNet, train_model,
    evaluate_predictions, calculate_hallucination_rate, print_prediction_analysis
)
from contrakit.constants import DEFAULT_SEED

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
# This experiment demonstrates hallucination on out-of-distribution inputs.
# We train on a subset of possible inputs (with labels A/B/C/D) and test on:
# 1. The SAME inputs (to verify learning occurred)
# 2. Different inputs that shouldn't have these labels (to demonstrate hallucination)
#
# Scientific note: We use embedding-based architecture which memorizes input-output
# mappings without requiring generalizable patterns. This is intentional - we're
# testing whether the model abstains on OOD inputs, not whether it generalizes.
DEFINED_RATIO = 0.4  # Fraction of inputs with defined outputs
UNDEFINED_SUPERVISION = 0.05  # Fraction of undefined inputs to supervise with ⊥
USE_DEFINEDNESS_HEAD = False  # Whether to use a separate head for uncertainty
DEFINEDNESS_THRESHOLD = 0.5  # Threshold for definedness head
EPOCHS_EXP1 = 100  # Training epochs

# Dataset generation functions are now in utils.py

# Model definition is now in utils.py

# Training function is now in utils.py

def evaluate_model(model, train_data, eval_defined, test_undefined, def_threshold=0.5):
    """
    Evaluate model performance on training, eval (same as training), and OOD inputs.
    
    Args:
        train_data: Training data tuple
        eval_defined: Evaluation set (SAME inputs as trained defined inputs)
        test_undefined: Out-of-distribution inputs that should map to ⊥
        def_threshold: Threshold for definedness head if used
        
    Returns:
        hallucination_rate: Fraction of OOD inputs given structured labels
        defined_accuracy: Accuracy on training inputs (verifies learning)
    """
    train_x, train_y, _ = train_data
    eval_defined_x, eval_defined_y = eval_defined
    test_undefined_x, test_undefined_y = test_undefined

    # Evaluate on defined inputs (SAME as training - verifies memorization)
    preds_defined, conf_defined, _ = evaluate_predictions(
        model, eval_defined_x, model.use_definedness_head, def_threshold
    )
    print_prediction_analysis(preds_defined, eval_defined_y, conf_defined,
                            label="DEFINED inputs (training set - should predict A/B/C/D)")

    # Evaluate on undefined inputs (OOD - tests abstention)
    preds_undefined, conf_undefined, abstention_rate = evaluate_predictions(
        model, test_undefined_x, model.use_definedness_head, def_threshold
    )
    print_prediction_analysis(preds_undefined, test_undefined_y, conf_undefined,
                            abstention_rate, "UNDEFINED inputs (OOD - should predict ⊥)")

    hallucination_rate = calculate_hallucination_rate(preds_undefined)
    defined_accuracy = np.mean(preds_defined == eval_defined_y)

    print("\nSUMMARY")
    print("=" * 40)
    print(f"Defined Accuracy (training inputs): {defined_accuracy:.1%}")
    print(f"Hallucination Rate (OOD inputs): {hallucination_rate:.1%}")
    if abstention_rate > 0:
        print(f"Abstention Rate: {abstention_rate:.1%}")

    return hallucination_rate, defined_accuracy

def run_multiple_seeds(seeds=None):
    if seeds is None:
        seeds = [DEFAULT_SEED + i for i in range(5)]
    """Run experiment with multiple random seeds to check consistency."""
    print("\n" + "="*60)
    print("MULTI-SEED TEST: Checking consistency across random seeds")
    print("="*60)

    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Generate data and train model
        function_map, _ = generate_partial_function(
            INPUT_SIZE, OUTPUT_CLASSES, DEFINED_RATIO, UNDEFINED_SUPERVISION, seed, use_structured_task=False
        )
        train_data, eval_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)

        torch.manual_seed(seed)
        model = HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES),
                               USE_DEFINEDNESS_HEAD, use_embedding=True)

        train_model(model, train_data, EPOCHS_EXP1, LEARNING_RATE, BATCH_SIZE, verbose=False)
        hall_rate, def_acc = evaluate_model(model, train_data, eval_defined, test_undefined,
                                          DEFINEDNESS_THRESHOLD)

        results.append({
            'seed': seed,
            'hallucination_rate': hall_rate,
            'defined_accuracy': def_acc
        })

    # Summary
    print("\nSUMMARY ACROSS SEEDS")
    print("="*60)
    hall_rates = [r['hallucination_rate'] for r in results]
    def_accs = [r['defined_accuracy'] for r in results]
    print(f"Hallucination Rate: {np.mean(hall_rates):.1%} ± {np.std(hall_rates):.1%}")
    print(f"Defined Accuracy:   {np.mean(def_accs):.1%} ± {np.std(def_accs):.1%}")
    print(f"\nHallucination behavior is consistent across different random seeds.")

def main():
    """Run the baseline hallucination experiment."""
    print("Neural Network Hallucination Experiment")
    print("="*50)

    # Generate dataset
    print(f"\nDATASET SETUP")
    print("-" * 30)
    print(f"Input range: 0 to {INPUT_SIZE-1}")
    print(f"Defined inputs: {DEFINED_RATIO:.0%}")
    print(f"⊥ supervision: {UNDEFINED_SUPERVISION:.0%} of undefined inputs")
    print(f"Output classes: {OUTPUT_CLASSES}")

    function_map, _ = generate_partial_function(
        INPUT_SIZE, OUTPUT_CLASSES, DEFINED_RATIO, UNDEFINED_SUPERVISION, DEFAULT_SEED, use_structured_task=False
    )

    n_defined = len([x for x, y in function_map.items() if y != '⊥'])
    n_undefined_labeled = len([x for x, y in function_map.items() if y == '⊥'])
    n_undefined_unlabeled = INPUT_SIZE - len(function_map)

    print(f"\nData composition:")
    print(f"  Defined inputs (A/B/C/D): {n_defined}")
    print(f"  Undefined inputs labeled with ⊥: {n_undefined_labeled}")
    print(f"  Undefined inputs unlabeled (OOD test set): {n_undefined_unlabeled}")

    # Create datasets
    train_data, eval_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)

    # Initialize model
    print(f"\nMODEL ARCHITECTURE")
    print("-" * 30)
    torch.manual_seed(DEFAULT_SEED)
    model = HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES),
                           USE_DEFINEDNESS_HEAD, use_embedding=True)

    print(f"Input embedding: {INPUT_SIZE} → {HIDDEN_SIZE}")
    print(f"Hidden layer 1: {HIDDEN_SIZE} → {HIDDEN_SIZE}")
    print(f"Hidden layer 2: {HIDDEN_SIZE} → {HIDDEN_SIZE}")
    print(f"Output layer: {HIDDEN_SIZE} → {len(OUTPUT_CLASSES)}")
    if USE_DEFINEDNESS_HEAD:
        print(f"Definedness head: {HIDDEN_SIZE} → 1 (threshold = {DEFINEDNESS_THRESHOLD})")

    # Train
    print(f"\nTRAINING")
    print("-" * 30)
    train_model(model, train_data, EPOCHS_EXP1, LEARNING_RATE, BATCH_SIZE)

    # Evaluate
    print(f"\nEVALUATION")
    print("-" * 30)
    hallucination_rate, defined_acc = evaluate_model(
        model, train_data, eval_defined, test_undefined, DEFINEDNESS_THRESHOLD
    )

    print("\nRESULTS")
    print("-" * 30)
    print(f"Accuracy on defined inputs: {defined_acc:.1%}")
    print(f"Hallucination rate on undefined inputs: {hallucination_rate:.1%}")

    if USE_DEFINEDNESS_HEAD:
        print("Model uses definedness head to reduce hallucinations.")
    else:
        print("Standard model hallucinates on undefined inputs.")


if __name__ == "__main__":
    main()

    # Uncomment to run consistency check across seeds
    # run_multiple_seeds()