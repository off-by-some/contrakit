"""
Architectural comparison: Standard vs definedness-head models for hallucination mitigation.

This experiment compares standard neural network classifiers against models with
dedicated definedness heads to test whether architectural modifications can
mitigate hallucination on undefined inputs.

Hypothesis tested:
Definedness heads can learn to abstain on undefined inputs during training,
but show limited generalization to novel undefined inputs, resulting in
hallucination rates that decrease but remain substantial.

Testing approach:
- Train both standard and definedness-head models on identical partial functions
- Standard model: Single output head producing confident classifications
- Definedness-head model: Dual heads for classification + definedness probability
- Compare hallucination rates across different training data compositions
- Analyze why definedness heads fail to generalize to test-time undefined inputs
- Examine training vs test performance discrepancies

Key measurements:
- Hallucination rates for both architectures across defined ratios
- Definedness head accuracy on training data vs generalization to test data
- Confidence distributions and abstention behavior
- Diagnostic analysis of definedness head failure modes

Assumptions:
- Definedness heads use sigmoid outputs with configurable thresholds
- Training includes supervision on some undefined inputs with ⊥ labels
- Test undefined inputs are truly out-of-distribution (never seen during training)
- Standard cross-entropy loss plus optional BCE loss for definedness

Expected outcome:
Definedness heads reduce but do not eliminate hallucination, due to limited
generalization from sparse supervision on undefined inputs. This motivates
the need for mathematically grounded approaches in subsequent experiments.

Typical usage:
- Run model_comparison() to compare architectures across defined ratios
- Use analyze_definedness_head_detailed() for diagnostic analysis
- Results saved as model_comparison.png in figures directory
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import (
    run_experiment, OUTPUT_CLASSES, generate_partial_function, create_datasets,
    HallucinationNet, train_model, INPUT_SIZE, HIDDEN_SIZE, EPOCHS, LEARNING_RATE, BATCH_SIZE
)
from contrakit.constants import FIGURES_DIR, DEFAULT_SEED

def analyze_definedness_head_detailed(defined_ratio=0.4, seed=DEFAULT_SEED):
    """Analyze why definedness head doesn't generalize well to test data."""

    # Generate data
    function_map, _ = generate_partial_function(
        INPUT_SIZE, OUTPUT_CLASSES, defined_ratio, 0.05, seed
    )
    train_data, test_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)
    train_x, train_y, train_defined = train_data
    test_defined_x, _ = test_defined
    test_undefined_x, _ = test_undefined

    # Train model
    torch.manual_seed(seed)
    model = HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES),
                           use_definedness_head=True)

    train_model(model, train_data, EPOCHS, LEARNING_RATE, BATCH_SIZE, verbose=False)

    # Analyze training performance
    model.eval()
    with torch.no_grad():
        _, train_definedness = model(torch.LongTensor(train_x))
        train_definedness = train_definedness.squeeze().numpy()

    train_defined_array = train_defined.numpy() if hasattr(train_defined, 'numpy') else train_defined
    defined_mask = train_defined_array == 1.0
    undefined_mask = train_defined_array == 0.0

    defined_scores = train_definedness[defined_mask]
    undefined_scores = train_definedness[undefined_mask]

    train_undefined_acc = (undefined_scores < 0.5).mean()

    # Analyze test performance
    with torch.no_grad():
        _, test_undef_definedness = model(torch.LongTensor(test_undefined_x))
        test_undef_definedness = test_undef_definedness.squeeze().numpy()

    test_undefined_acc = (test_undef_definedness < 0.5).mean()

    # Generalization analysis
    gap = train_undefined_acc - test_undefined_acc
    coverage_ratio = len(undefined_scores) / len(test_undef_definedness)

    return {
        'train_accuracy': train_undefined_acc,
        'test_accuracy': test_undefined_acc,
        'generalization_gap': gap,
        'coverage_ratio': coverage_ratio,
        'n_undefined_train': len(undefined_scores),
        'n_undefined_test': len(test_undef_definedness)
    }

def main():
    """Compare standard vs definedness-head models across training ratios."""
    print("Comparing Standard vs Definedness-Head Models")
    print("=" * 55)

    # Test different proportions of defined inputs
    defined_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\nTesting Standard Model (no definedness head)")
    print("-" * 45)

    standard_results = []
    for ratio in defined_ratios:
        print(f"Defined ratio: {ratio:.0%}")
        result = run_experiment(defined_ratio=ratio, use_definedness_head=False, seed=DEFAULT_SEED)
        standard_results.append(result)
        print(f"  Hallucination rate: {result['hallucination_rate']:.1%}")

    print("\nTesting Definedness-Head Model")
    print("-" * 35)

    definedness_results = []
    for ratio in defined_ratios:
        print(f"Defined ratio: {ratio:.0%}")
        result = run_experiment(defined_ratio=ratio, use_definedness_head=True, seed=DEFAULT_SEED)
        definedness_results.append(result)
        print(f"  Hallucination rate: {result['hallucination_rate']:.1%}")
        print(f"  Abstention rate: {result['abstention_rate']:.1%}")

    # Compare results
    print("\nRESULTS COMPARISON")
    print("-" * 25)

    standard_rates = [r['hallucination_rate'] for r in standard_results]
    definedness_rates = [r['hallucination_rate'] for r in definedness_results]

    print("Standard Model:")
    print(f"  Mean hallucination: {np.mean(standard_rates):.1%}")
    print(f"  Range: {min(standard_rates):.1%} to {max(standard_rates):.1%}")

    print("Definedness-Head Model:")
    print(f"  Mean hallucination: {np.mean(definedness_rates):.1%}")
    print(f"  Range: {min(definedness_rates):.1%} to {max(definedness_rates):.1%}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot hallucination rates
    ax1.plot(defined_ratios, standard_rates, 'o-', label='Standard Model',
             linewidth=2, markersize=8, color='red')
    ax1.plot(defined_ratios, definedness_rates, 's-', label='Definedness Head',
             linewidth=2, markersize=8, color='blue')

    ax1.set_xlabel('Fraction of Defined Training Inputs')
    ax1.set_ylabel('Hallucination Rate')
    ax1.set_title('Hallucination Rate vs Training Data Composition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot variances
    models = ['Standard\nModel', 'Definedness\nHead']
    variances = [np.var(standard_rates), np.var(definedness_rates)]
    colors = ['red', 'blue']

    bars = ax2.bar(models, variances, color=colors, alpha=0.7, width=0.6)
    ax2.set_ylabel('Variance in Hallucination Rate')
    ax2.set_title('Consistency Across Training Conditions')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    output_path = FIGURES_DIR / 'model_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {output_path}")

    # Diagnostic analysis of definedness head performance
    print("\nDIAGNOSTIC ANALYSIS")
    print("-" * 20)
    print("Why does the definedness head underperform?")

    diag_result = analyze_definedness_head_detailed(defined_ratio=0.4, seed=DEFAULT_SEED)

    print(f"\nTraining performance on undefined inputs: {diag_result['train_accuracy']:.1%}")
    print(f"Test performance on undefined inputs: {diag_result['test_accuracy']:.1%}")
    print(f"Generalization gap: {diag_result['generalization_gap']:+.1%}")
    print(f"Training coverage: {diag_result['coverage_ratio']:.1%}")
    print(f"  ({diag_result['n_undefined_train']} labeled undefined examples in training)")
    print(f"  ({diag_result['n_undefined_test']} undefined examples in test)")

    if diag_result['generalization_gap'] > 0.1:
        print("\nThe definedness head shows poor generalization.")
        print("It performs well on training data but poorly on unseen test data.")
        print("This suggests memorization rather than learning general patterns.")

    print("\nSUMMARY")
    print("-" * 15)
    variance_ratio = np.var(definedness_rates) / np.var(standard_rates)
    print(f"Variance ratio (definedness/standard): {variance_ratio:.2f}")

    if np.mean(definedness_rates) < np.mean(standard_rates):
        print("Definedness head reduces hallucination rates modestly.")
        print("However, limited training supervision and poor generalization")
        print("prevent more significant improvements.")
    else:
        print("Definedness head does not significantly reduce hallucination rates.")
        print("The small amount of supervision (5% of undefined inputs) is insufficient.")

    return standard_results, definedness_results

if __name__ == "__main__":
    main()

