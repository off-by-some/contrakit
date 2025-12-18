"""
Test relationship between mathematical task structure and hallucination rates.

This experiment uses contradiction theory (contrakit) to analyze how the
mathematical structure of partial function learning tasks predicts hallucination
behavior in neural networks.

Hypothesis tested:
Hallucination rates in neural networks are predictable from the contradiction
measure K of the learning task, where K quantifies the incompatibility between
different "views" or contexts of the same underlying function.

Testing approach:
- Create partial functions with varying defined/undefined input ratios
- Model task as having two behavioral views: defined vs undefined input contexts
- Compute contradiction measure K between these behavioral contexts
- Train neural networks on the partial function and measure hallucination rates
- Compare observed hallucination rates against theoretical predictions from K
- Test statistical significance of relationship between K and hallucination

Key measurements:
- Contradiction measure K between defined and undefined input behaviors
- Theoretical hallucination bounds derived from K
- Observed hallucination rates across different defined ratios
- Agreement coefficients and frame-independence measures
- Statistical correlation between task structure and network behavior

Assumptions:
- Task can be modeled as having distinct behavioral contexts (defined/undefined)
- Contrakit correctly computes contradiction measures between contexts
- Neural networks learn frame-independent representations
- Hallucination rates are measurable and statistically stable

Expected outcome:
Tasks with higher contradiction (K > 0) show predictable hallucination rates,
demonstrating that mathematical task structure determines neural network
behavior rather than just training data statistics.

Typical usage:
- Run experiment_across_defined_ratios() to test relationship systematically
- Use create_task_behavior() to construct contrakit behavior representations
- Results show how mathematical structure predicts empirical outcomes
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from utils import (
    HallucinationNet, generate_partial_function, create_datasets,
    train_model, INPUT_SIZE, OUTPUT_CLASSES, HIDDEN_SIZE,
    LEARNING_RATE, EPOCHS, BATCH_SIZE, calculate_hallucination_rate
)
from contrakit import Observatory
from contrakit.constants import DEFAULT_SEED

def create_task_behavior(function_map, test_defined_x, test_undefined_x):
    """
    Create a behavior representing the task structure.
    Models the task as having two different "views" that may conflict.
    """
    undefined_idx = OUTPUT_CLASSES.index('⊥')

    # Distribution for defined inputs
    defined_dist = np.zeros(len(OUTPUT_CLASSES))
    for x in test_defined_x:
        if x in function_map and function_map[x] != '⊥':
            label_idx = OUTPUT_CLASSES.index(function_map[x])
            defined_dist[label_idx] += 1
    if len(test_defined_x) > 0:
        defined_dist /= len(test_defined_x)

    # Distribution for undefined inputs (always ⊥)
    undefined_dist = np.zeros(len(OUTPUT_CLASSES))
    undefined_dist[undefined_idx] = 1.0

    # Create observatory and define the task structure
    obs = Observatory.create(symbols=OUTPUT_CLASSES)
    output = obs.concept("Output")

    defined_lens = obs.lens("DefinedRegion")
    undefined_lens = obs.lens("UndefinedRegion")

    with defined_lens:
        defined_lens.perspectives[output] = {
            val: float(prob)
            for val, prob in zip(output.alphabet, defined_dist)
        }

    with undefined_lens:
        undefined_lens.perspectives[output] = {
            val: float(prob)
            for val, prob in zip(output.alphabet, undefined_dist)
        }

    behavior = (defined_lens | undefined_lens).to_behavior()
    return behavior

# measure_hallucination_rate is now in utils.py

def run_experiment(defined_ratio, undefined_supervision=0.05, seed=DEFAULT_SEED):
    """Run experiment and measure task properties and hallucination rate."""
    # Generate data
    function_map, _ = generate_partial_function(
        INPUT_SIZE, OUTPUT_CLASSES, defined_ratio, undefined_supervision, seed
    )
    train_data, test_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)
    test_defined_x, _ = test_defined
    test_undefined_x, _ = test_undefined

    # Analyze task structure
    task_behavior = create_task_behavior(function_map, test_defined_x, test_undefined_x)

    # Train model
    torch.manual_seed(seed)
    model = HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES),
                           use_definedness_head=False)
    train_model(model, train_data, EPOCHS, LEARNING_RATE, BATCH_SIZE, verbose=False)

    # Evaluate
    with torch.no_grad():
        output = model(torch.LongTensor(test_undefined_x))
        preds_undefined = torch.argmax(output, dim=1).numpy()
        hallucination_rate = calculate_hallucination_rate(preds_undefined)

    return {
        'task_complexity': task_behavior.K,
        'task_agreement': task_behavior.alpha_star,
        'task_frame_independent': task_behavior.is_frame_independent(),
        'hallucination_rate': hallucination_rate,
        'defined_ratio': defined_ratio,
        'n_defined': len(test_defined_x),
        'n_undefined': len(test_undefined_x),
    }

def main():
    print("="*70)
    print("EXPERIMENT: Task Contradiction and Hallucination")
    print("="*70)
    print("\nThis test examines how hallucination relates to task properties.")
    print("We vary the proportion of 'defined' vs 'undefined' inputs in training.")
    print("\nDefined inputs: Model learns to predict A/B/C/D labels")
    print("Undefined inputs: Model should abstain (⊥) but often hallucinates")

    # Sweep defined ratios
    results = []
    defined_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n{'='*70}")
    print("RESULTS BY DATA COMPOSITION")
    print('='*70)

    for ratio in defined_ratios:
        print(f"\nData composition: {ratio:.0%} defined, {(1-ratio):.0%} undefined")
        print("-"*50)

        result = run_experiment(defined_ratio=ratio, seed=DEFAULT_SEED)
        results.append(result)

        print(f"Task complexity: {result['task_complexity']:.4f}")
        print(f"Task agreement: {result['task_agreement']:.4f}")
        print(f"Frame independent: {'Yes' if result['task_frame_independent'] else 'No'}")
        print(f"Hallucination rate:      {result['hallucination_rate']:.1%}")
        print(f"Training set size:       {INPUT_SIZE} examples")
        print(f"Defined examples:        {result['n_defined']}")
        print(f"Undefined examples:      {result['n_undefined']}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print('='*70)
    print(f"{'Defined %':<12} {'Complexity':<12} {'Agreement':<12} {'Frame Indep':<12} {'Hallucination':<15}")
    print("-" * 75)
    for r in results:
        fi_str = "Yes" if r['task_frame_independent'] else "No"
        print(f"{r['defined_ratio']:>8.0%}    {r['task_complexity']:>10.4f}  {r['task_agreement']:>10.4f}  {fi_str:>10s}  {r['hallucination_rate']:>12.1%}")

    print(f"\n{'='*70}")
    print("OBSERVATIONS")
    print('='*70)

    complexities = [r['task_complexity'] for r in results]
    hall_rates = [r['hallucination_rate'] for r in results]

    # Key observations
    complexity_range = max(complexities) - min(complexities)
    complexity_constant = complexity_range < 1e-6

    print("\n1. Task complexity:")
    if complexity_constant:
        print(f"  - Remains constant at {complexities[0]:.4f} across all data compositions")
    else:
        print(f"  - Varies with data composition")


    print("\n2. Hallucination rate:")
    print(f"   - Ranges from {min(hall_rates):.1%} to {max(hall_rates):.1%}")




    print('\n' + '='*70)

if __name__ == "__main__":
    main()
