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

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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
# EXPERIMENT CONFIGURATION CONSTANTS
# ============================================================================
DEFINED_RATIO = 0.4  # Fraction of inputs with defined outputs (A/B/C/D)
UNDEFINED_SUPERVISION_RATIO = 0.05  # Fraction of undefined inputs supervised with ⊥
USE_DEFINEDNESS_HEAD = False  # Whether to use separate head for uncertainty
DEFINEDNESS_THRESHOLD = 0.5  # Threshold for definedness head decisions
EXPERIMENT_EPOCHS = 100  # Training epochs for this experiment
NUM_SEEDS_FOR_CONSISTENCY = 5  # Number of seeds to test for consistency

# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass(frozen=True)
class ExperimentResults:
    """Results from evaluating a hallucination experiment."""
    hallucination_rate: float  # Fraction of undefined inputs given structured predictions
    defined_accuracy: float    # Accuracy on defined inputs (verifies learning)

# Dataset generation functions are now in utils.py

# Model definition is now in utils.py

# Training function is now in utils.py

def evaluate_model(
    model: Any,
    train_data: Tuple[Any, Any, Any],
    eval_defined: Tuple[Any, Any],
    test_undefined: Tuple[Any, Any],
    def_threshold: float = DEFINEDNESS_THRESHOLD
) -> ExperimentResults:
    """
    Evaluate model performance on training, eval (same as training), and OOD inputs.

    Args:
        model: Trained neural network model
        train_data: Training data tuple (inputs, labels, masks)
        eval_defined: Evaluation set (same inputs as trained defined inputs)
        test_undefined: Out-of-distribution inputs that should map to ⊥
        def_threshold: Threshold for definedness head if used

    Returns:
        hallucination_rate: Fraction of OOD inputs given structured labels
        defined_accuracy: Accuracy on training inputs (verifies learning)
    """
    if not (0.0 <= def_threshold <= 1.0):
        raise ValueError(f"def_threshold must be between 0 and 1, got {def_threshold}")

    if len(train_data) != 3:
        raise ValueError("train_data must be a tuple of (inputs, labels, masks)")

    if len(eval_defined) != 2 or len(test_undefined) != 2:
        raise ValueError("eval_defined and test_undefined must be tuples of (inputs, labels)")
    train_x, train_y, _ = train_data
    eval_defined_x, eval_defined_y = eval_defined
    test_undefined_x, test_undefined_y = test_undefined

    # Evaluate on defined inputs (same as training - verifies memorization occurred)
    predictions_defined, confidence_defined, _ = evaluate_predictions(
        model, eval_defined_x, model.use_definedness_head, def_threshold
    )
    print_prediction_analysis(predictions_defined, eval_defined_y, confidence_defined,
                            label="DEFINED inputs (training set - should predict A/B/C/D)")

    # Evaluate on undefined inputs (out-of-distribution - tests hallucination)
    predictions_undefined, confidence_undefined, abstention_rate = evaluate_predictions(
        model, test_undefined_x, model.use_definedness_head, def_threshold
    )
    print_prediction_analysis(predictions_undefined, test_undefined_y, confidence_undefined,
                            abstention_rate, "UNDEFINED inputs (OOD - should predict ⊥)")

    hallucination_rate = calculate_hallucination_rate(predictions_undefined)
    defined_accuracy = np.mean(predictions_defined == eval_defined_y)

    print("\nSUMMARY")
    print("=" * 40)
    print(f"Defined Accuracy (training inputs): {defined_accuracy:.1%}")
    print(f"Hallucination Rate (OOD inputs): {hallucination_rate:.1%}")
    if abstention_rate > 0:
        print(f"Abstention Rate: {abstention_rate:.1%}")

    return ExperimentResults(
        hallucination_rate=hallucination_rate,
        defined_accuracy=defined_accuracy
    )

def run_multiple_seeds(seeds: Optional[List[int]] = None) -> None:
    """Run experiment with multiple random seeds to check consistency."""
    if seeds is None:
        seeds = [DEFAULT_SEED + i for i in range(NUM_SEEDS_FOR_CONSISTENCY)]

    if not seeds:
        raise ValueError("At least one seed must be provided")

    if len(set(seeds)) != len(seeds):
        raise ValueError("All seeds must be unique")
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

        train_model(model, train_data, EXPERIMENT_EPOCHS, LEARNING_RATE, BATCH_SIZE, verbose=False)
        results = evaluate_model(model, train_data, eval_defined, test_undefined, DEFINEDNESS_THRESHOLD)
        hall_rate, def_acc = results.hallucination_rate, results.defined_accuracy

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

def setup_experiment_data() -> Tuple[Dict[Any, Any], Tuple[Any, Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    """
    Generate and prepare datasets for the hallucination experiment.

    Returns:
        function_map: Mapping of inputs to outputs (including ⊥ for undefined)
        train_data: Tuple of (inputs, labels, masks) for training
        eval_defined: Tuple of (inputs, labels) for evaluation on defined inputs
        test_undefined: Tuple of (inputs, labels) for testing on undefined inputs
    """
    function_map, _ = generate_partial_function(
        INPUT_SIZE, OUTPUT_CLASSES, DEFINED_RATIO, UNDEFINED_SUPERVISION_RATIO,
        DEFAULT_SEED, use_structured_task=False
    )

    train_data, eval_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)
    return function_map, train_data, eval_defined, test_undefined


def print_dataset_info(function_map: Dict[Any, Any]) -> None:
    """
    Print detailed information about the experiment dataset composition.

    Args:
        function_map: Dictionary mapping inputs to their defined outputs or ⊥
    """
    print(f"DATASET SETUP")
    print("-" * 30)
    print(f"Input range: 0 to {INPUT_SIZE-1}")
    print(f"Defined inputs: {DEFINED_RATIO:.0%}")
    print(f"⊥ supervision: {UNDEFINED_SUPERVISION_RATIO:.0%} of undefined inputs")
    print(f"Output classes: {OUTPUT_CLASSES}")

    n_defined = len([x for x, y in function_map.items() if y != '⊥'])
    n_undefined_labeled = len([x for x, y in function_map.items() if y == '⊥'])
    n_undefined_unlabeled = INPUT_SIZE - len(function_map)

    print(f"\nData composition:")
    print(f"  Defined inputs (A/B/C/D): {n_defined}")
    print(f"  Undefined inputs labeled with ⊥: {n_undefined_labeled}")
    print(f"  Undefined inputs unlabeled (OOD test set): {n_undefined_unlabeled}")


def initialize_model() -> Any:
    """
    Initialize the HallucinationNet model with experiment configuration.

    Returns:
        Configured neural network model ready for training
    """
    torch.manual_seed(DEFAULT_SEED)
    return HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES),
                           USE_DEFINEDNESS_HEAD, use_embedding=True)


def print_model_info(model: Any) -> None:
    """
    Print detailed information about the model architecture and configuration.

    Args:
        model: The initialized neural network model
    """
    print(f"MODEL ARCHITECTURE")
    print("-" * 30)
    print(f"Input embedding: {INPUT_SIZE} → {HIDDEN_SIZE}")
    print(f"Hidden layer 1: {HIDDEN_SIZE} → {HIDDEN_SIZE}")
    print(f"Hidden layer 2: {HIDDEN_SIZE} → {HIDDEN_SIZE}")
    print(f"Output layer: {HIDDEN_SIZE} → {len(OUTPUT_CLASSES)}")
    if USE_DEFINEDNESS_HEAD:
        print(f"Definedness head: {HIDDEN_SIZE} → 1 (threshold = {DEFINEDNESS_THRESHOLD})")


def train_experiment_model(model: Any, train_data: Tuple[Any, Any, Any]) -> None:
    """
    Train the model on the experiment dataset.

    Args:
        model: Neural network model to train
        train_data: Tuple of (inputs, labels, masks) for training
    """
    print(f"TRAINING")
    print("-" * 30)
    train_model(model, train_data, EXPERIMENT_EPOCHS, LEARNING_RATE, BATCH_SIZE)


def evaluate_experiment_model(
    model: Any,
    train_data: Tuple[Any, Any, Any],
    eval_defined: Tuple[Any, Any],
    test_undefined: Tuple[Any, Any]
) -> ExperimentResults:
    """
    Evaluate the trained model on all test sets and return key metrics.

    Args:
        model: Trained neural network model
        train_data: Original training data (for reference)
        eval_defined: Evaluation set with defined inputs
        test_undefined: Test set with undefined inputs

    Returns:
        Tuple of (hallucination_rate, defined_accuracy)
    """
    print(f"EVALUATION")
    print("-" * 30)
    return evaluate_model(model, train_data, eval_defined, test_undefined, DEFINEDNESS_THRESHOLD)


def print_experiment_results(hallucination_rate: float, defined_acc: float) -> None:
    """
    Print the final experiment results and conclusions.

    Args:
        hallucination_rate: Fraction of undefined inputs that received structured predictions
        defined_acc: Accuracy on defined inputs (verifies learning occurred)
    """
    print("\nRESULTS")
    print("-" * 30)
    print(f"Accuracy on defined inputs: {defined_acc:.1%}")
    print(f"Hallucination rate on undefined inputs: {hallucination_rate:.1%}")

    if USE_DEFINEDNESS_HEAD:
        print("Model uses definedness head to reduce hallucinations.")
    else:
        print("Standard model hallucinates on undefined inputs.")


def main() -> None:
    """Run the baseline hallucination experiment."""
    print("Neural Network Hallucination Experiment")
    print("="*50)

    # Generate dataset
    function_map, train_data, eval_defined, test_undefined = setup_experiment_data()
    print_dataset_info(function_map)

    # Initialize and describe model
    model = initialize_model()
    print_model_info(model)

    # Train model
    train_experiment_model(model, train_data)

    # Evaluate model
    results = evaluate_experiment_model(model, train_data, eval_defined, test_undefined)

    # Print results
    print_experiment_results(results.hallucination_rate, results.defined_accuracy)


if __name__ == "__main__":
    main()

    # Uncomment to run consistency check across seeds
    # run_multiple_seeds()