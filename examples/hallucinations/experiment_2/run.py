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
    seeds = [DEFAULT_SEED, DEFAULT_SEED + 1, DEFAULT_SEED + 2]

    print("\nTesting Standard Model (no definedness head)")
    print("-" * 45)

    standard_results = []
    for ratio in defined_ratios:
        print(f"Defined ratio: {ratio:.0%}")
        ratio_results = []
        for seed in seeds:
            result = run_experiment(defined_ratio=ratio, use_definedness_head=False, seed=seed)
            ratio_results.append(result)
        
        mean_hallucination = np.mean([r['hallucination_rate'] for r in ratio_results])
        std_hallucination = np.std([r['hallucination_rate'] for r in ratio_results])
        standard_results.append({
            'mean': mean_hallucination,
            'std': std_hallucination,
            'all_results': ratio_results
        })
        print(f"  Hallucination rate: {mean_hallucination:.1%} ± {std_hallucination:.1%}")

    print("\nTesting Definedness-Head Model")
    print("-" * 35)

    definedness_results = []
    for ratio in defined_ratios:
        print(f"Defined ratio: {ratio:.0%}")
        ratio_results = []
        for seed in seeds:
            result = run_experiment(defined_ratio=ratio, use_definedness_head=True, seed=seed)
            ratio_results.append(result)
        
        mean_hallucination = np.mean([r['hallucination_rate'] for r in ratio_results])
        std_hallucination = np.std([r['hallucination_rate'] for r in ratio_results])
        mean_abstention = np.mean([r['abstention_rate'] for r in ratio_results])
        definedness_results.append({
            'mean': mean_hallucination,
            'std': std_hallucination,
            'mean_abstention': mean_abstention,
            'all_results': ratio_results
        })
        print(f"  Hallucination rate: {mean_hallucination:.1%} ± {std_hallucination:.1%}")
        print(f"  Abstention rate: {mean_abstention:.1%}")

    # Compare results
    print("\nRESULTS COMPARISON")
    print("-" * 25)

    standard_rates = [r['mean'] for r in standard_results]
    definedness_rates = [r['mean'] for r in definedness_results]

    print("Standard Model:")
    print(f"  Mean hallucination: {np.mean(standard_rates):.1%}")
    print(f"  Range: {min(standard_rates):.1%} to {max(standard_rates):.1%}")

    print("Definedness-Head Model:")
    print(f"  Mean hallucination: {np.mean(definedness_rates):.1%}")
    print(f"  Range: {min(definedness_rates):.1%} to {max(definedness_rates):.1%}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot hallucination rates with error bars
    standard_stds = [r['std'] for r in standard_results]
    definedness_stds = [r['std'] for r in definedness_results]
    
    ax1.errorbar(defined_ratios, standard_rates, yerr=standard_stds, 
                 fmt='o-', label='Standard Model',
                 linewidth=2, markersize=8, color='red', capsize=5)
    ax1.errorbar(defined_ratios, definedness_rates, yerr=definedness_stds,
                 fmt='s-', label='Definedness Head',
                 linewidth=2, markersize=8, color='blue', capsize=5)

    ax1.set_xlabel('Fraction of Defined Training Inputs')
    ax1.set_ylabel('Hallucination Rate')
    ax1.set_title('Hallucination Rate vs Training Data Composition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot mean standard deviation across seeds (model stability)
    models = ['Standard\nModel', 'Definedness\nHead']
    mean_stds = [np.mean(standard_stds), np.mean(definedness_stds)]
    colors = ['red', 'blue']

    bars = ax2.bar(models, mean_stds, color=colors, alpha=0.7, width=0.6)
    ax2.set_ylabel('Mean Standard Deviation Across Seeds')
    ax2.set_title('Model Stability (Within Condition Variability)')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, std in zip(bars, mean_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    output_path = FIGURES_DIR / 'model_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {output_path}")

    # Diagnostic analysis of definedness head performance
    print("\nDIAGNOSTIC ANALYSIS")
    print("-" * 20)
    print("Why does the definedness head underperform?")
    print("Running diagnostic at 40% defined ratio across multiple seeds...")

    diag_results = [analyze_definedness_head_detailed(defined_ratio=0.4, seed=seed) for seed in seeds]
    
    mean_train_acc = np.mean([d['train_accuracy'] for d in diag_results])
    std_train_acc = np.std([d['train_accuracy'] for d in diag_results])
    mean_test_acc = np.mean([d['test_accuracy'] for d in diag_results])
    std_test_acc = np.std([d['test_accuracy'] for d in diag_results])
    mean_gap = np.mean([d['generalization_gap'] for d in diag_results])
    std_gap = np.std([d['generalization_gap'] for d in diag_results])

    print(f"\nTraining performance on undefined inputs: {mean_train_acc:.1%} ± {std_train_acc:.1%}")
    print(f"Test performance on undefined inputs: {mean_test_acc:.1%} ± {std_test_acc:.1%}")
    print(f"Generalization gap: {mean_gap:+.1%} ± {std_gap:.1%}")
    print(f"Training coverage: {diag_results[0]['coverage_ratio']:.1%}")
    print(f"  ({diag_results[0]['n_undefined_train']} labeled undefined examples in training)")
    print(f"  ({diag_results[0]['n_undefined_test']} undefined examples in test)")

    if mean_gap > 0.1:
        print("\nThe definedness head shows poor generalization.")
        print("It performs well on training data but poorly on unseen test data.")
        print("This suggests memorization rather than learning general patterns.")

    print("\nSUMMARY")
    print("-" * 15)
    mean_std_ratio = np.mean(definedness_stds) / np.mean(standard_stds)
    print(f"Mean std ratio (definedness/standard): {mean_std_ratio:.2f}")
    print(f"This measures within-condition variability (lower = more stable).")

    if np.mean(definedness_rates) < np.mean(standard_rates):
        improvement = (np.mean(standard_rates) - np.mean(definedness_rates)) * 100
        print(f"\nDefinedness head reduces hallucination by {improvement:.1f} percentage points.")
        print("However, limited training supervision and poor generalization")
        print("prevent more significant improvements.")
    else:
        print("\nDefinedness head does not significantly reduce hallucination rates.")
        print("The small amount of supervision (5% of undefined inputs) is insufficient.")

    return standard_results, definedness_results

if __name__ == "__main__":
    main()

