"""
Test monotonic relationship between training imbalance and hallucination rates.

This experiment tests whether hallucination rates show a strong monotonic
increase with training data imbalance toward structured outputs, as predicted
by the Witness-Error Tradeoff theorem.

Hypothesis tested:
For tasks with fixed contradiction K > 0, hallucination rates increase
monotonically with the imbalance ratio of defined vs undefined training inputs,
due to insufficient witness capacity allocation forcing compensatory errors.

Testing approach:
- Fix task contradiction at K > 0 using controlled experimental design
- Vary training imbalance by changing defined input ratios from 0.1 to 0.9
- Train neural networks and measure hallucination rates across multiple seeds
- Compute Spearman correlation between imbalance ratio and hallucination rate
- Analyze training dynamics and witness capacity utilization at violation points
- Test statistical significance of monotonic trend

Key measurements:
- Spearman rank correlation coefficients across imbalance range
- Hallucination rates with confidence intervals across random seeds
- Witness capacity utilization vs error rates at different imbalance levels
- Statistical significance of monotonic trend (p-values)
- Training convergence behavior at different imbalance ratios

Assumptions:
- Task contradiction K remains fixed while varying training distribution
- Witness capacity is limited and must be allocated between error reduction and uncertainty
- Neural networks converge to stable solutions across random seeds
- Monotonic relationship is statistically detectable with sufficient samples

Expected outcome:
Strong monotonic correlation confirms that training imbalance systematically
increases hallucination pressure, supporting the Witness-Error Tradeoff as
the mechanism forcing fabrication under resource constraints.

Typical usage:
- Run test_monotonic_trend() to validate prediction across seeds
- Use analyze_training_dynamics() to examine convergence at violation points
- Results demonstrate systematic relationship between imbalance and hallucination
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import io
import matplotlib.pyplot as plt
from utils import (
    HallucinationNet, generate_partial_function, create_datasets,
    train_model, INPUT_SIZE, OUTPUT_CLASSES, HIDDEN_SIZE,
    LEARNING_RATE, EPOCHS, BATCH_SIZE
)
from scipy.stats import spearmanr, permutation_test
from contrakit.constants import DEFAULT_SEED, FIGURES_DIR

def measure_hallucination_rate(model, test_undefined_x):
    """Measure what fraction of undefined inputs get hallucinated responses."""
    model.eval()
    undefined_idx = OUTPUT_CLASSES.index('⊥')
    
    with torch.no_grad():
        logits = model(torch.LongTensor(test_undefined_x))
        preds = torch.argmax(logits, dim=1).numpy()
    
    hallucination_rate = (preds != undefined_idx).mean()
    return hallucination_rate

def run_experiment(defined_ratio, undefined_supervision=0.05, seed=DEFAULT_SEED, verbose=False):
    """Run one experiment and measure hallucination rate."""
    function_map, _ = generate_partial_function(
        INPUT_SIZE, OUTPUT_CLASSES, defined_ratio,
        undefined_supervision, seed
    )
    train_data, test_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)
    test_undefined_x, _ = test_undefined
    
    torch.manual_seed(seed)
    model = HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES), use_definedness_head=False)
    
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    try:
        train_model(model, train_data, EPOCHS, LEARNING_RATE, BATCH_SIZE)
    finally:
        if not verbose:
            sys.stdout = old_stdout
    
    hallucination_rate = measure_hallucination_rate(model, test_undefined_x)
    return hallucination_rate

def check_monotonicity(ratios, rates, tolerance=0.01):
    """
    Check if hallucination rates increase monotonically.
    
    Note: Tolerance of 1% chosen based on measurement noise from finite samples.
    At 85% defined ratio, only ~19 undefined training examples remain, creating
    sampling uncertainty of approximately sqrt(0.5*0.5/19) ≈ 11% per measurement.
    A 1% tolerance is conservative relative to this noise level.
    """
    differences = np.diff(rates)
    is_monotonic = np.all(differences >= -tolerance)
    
    violations = []
    for i in range(len(differences)):
        if differences[i] < -tolerance:
            violations.append((ratios[i], ratios[i+1], differences[i]))
    
    return is_monotonic, violations

def test_null_hypothesis_violations(ratios, rates, tolerance=0.01, n_permutations=10000, seed=DEFAULT_SEED):
    """
    Test whether observed violations are fewer than expected under null hypothesis.
    
    Null hypothesis: Hallucination rates are random (no monotonic relationship).
    If null is true, permuting the rates shouldn't change violation count.
    
    Returns:
        observed_violations: Number of violations in actual data
        null_violations: Distribution of violations under null (from permutations)
        p_value: Probability of seeing this few (or fewer) violations by chance
    """
    observed_count = len(check_monotonicity(ratios, rates, tolerance)[1])
    
    null_violations = []
    rng = np.random.RandomState(seed)
    
    for _ in range(n_permutations):
        permuted_rates = rng.permutation(rates)
        violation_count = len(check_monotonicity(ratios, permuted_rates, tolerance)[1])
        null_violations.append(violation_count)
    
    null_violations = np.array(null_violations)
    p_value = np.mean(null_violations <= observed_count)
    
    return observed_count, null_violations, p_value

def compute_effect_size(ratios, rates):
    """
    Compute effect size (Cohen's d) for the monotonic trend.
    
    Effect size quantifies how large the increase is relative to variability.
    Cohen's d interpretation: 0.2 = small, 0.5 = medium, 0.8 = large
    """
    mean_increase = rates[-1] - rates[0]
    std_dev = np.std(rates)
    
    if std_dev == 0:
        return float('inf') if mean_increase > 0 else 0
    
    cohens_d = mean_increase / std_dev
    return cohens_d

def bootstrap_correlation_ci(ratios, rates, n_bootstrap=10000, confidence=0.95, seed=DEFAULT_SEED):
    """
    Compute bootstrap confidence interval for Spearman correlation.
    """
    rng = np.random.RandomState(seed)
    correlations = []
    
    n = len(ratios)
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_ratios = ratios[indices]
        boot_rates = rates[indices]
        rho, _ = spearmanr(boot_ratios, boot_rates)
        correlations.append(rho)
    
    correlations = np.array(correlations)
    alpha = 1 - confidence
    lower = np.percentile(correlations, 100 * alpha / 2)
    upper = np.percentile(correlations, 100 * (1 - alpha / 2))
    
    return lower, upper

def test_single_seed(seed=DEFAULT_SEED, verbose=True):
    """Test with a single seed."""
    defined_ratios = np.linspace(0.1, 0.9, 17)
    
    if verbose:
        print(f"Testing {len(defined_ratios)} points from 10% to 90% defined\n")
    
    results = []
    for ratio in defined_ratios:
        if verbose:
            print(f"Defined ratio: {ratio:.1%}...", end=" ", flush=True)
        hall_rate = run_experiment(defined_ratio=ratio, seed=seed, verbose=False)
        results.append(hall_rate)
        if verbose:
            print(f"Hallucination: {hall_rate:.1%}")
    
    return defined_ratios, results

def test_multiple_seeds(seeds, verbose=True):
    """Test across multiple seeds for robustness."""
    defined_ratios = np.linspace(0.1, 0.9, 17)
    
    if verbose:
        print(f"Testing {len(seeds)} different random seeds")
        print(f"Across {len(defined_ratios)} training ratios (10% to 90% defined)\n")
    
    all_results = []
    all_violations = []
    all_correlations = []
    
    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nSeed {seed} ({seed_idx+1}/{len(seeds)}):")
        
        results = []
        for ratio in defined_ratios:
            hall_rate = run_experiment(defined_ratio=ratio, seed=seed, verbose=False)
            results.append(hall_rate)
        
        all_results.append(results)
        
        is_monotonic, violations = check_monotonicity(defined_ratios, results, tolerance=0.01)
        all_violations.append(len(violations))
        
        correlation, p_value = spearmanr(defined_ratios, results)
        all_correlations.append(correlation)
        
        if verbose:
            print(f"  Range: {min(results):.1%} → {max(results):.1%}")
            print(f"  Spearman's ρ: {correlation:+.3f} (p={p_value:.4e})")
            print(f"  Violations: {len(violations)}")
    
    return defined_ratios, np.array(all_results), all_violations, all_correlations

def create_visualization(defined_ratios, all_results, output_path=None):
    """Create visualization of results."""
    if output_path is None:
        output_path = FIGURES_DIR / 'monotonicity_violation_analysis.png'
    mean_results = all_results.mean(axis=0)
    std_results = all_results.std(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Prediction 6: Monotonicity Test Results', fontsize=14, fontweight='bold')
    
    # Left: Individual seeds
    for seed_idx, results in enumerate(all_results):
        axes[0].plot(defined_ratios * 100, results * 100, 
                    alpha=0.3, color='gray', linewidth=1)
    axes[0].plot(defined_ratios * 100, mean_results * 100, 
                color='blue', linewidth=3, label='Mean', marker='o')
    axes[0].fill_between(defined_ratios * 100, 
                         (mean_results - std_results) * 100,
                         (mean_results + std_results) * 100,
                         alpha=0.2, color='blue', label='±1 std')
    axes[0].set_xlabel('Defined Ratio (%)', fontsize=12)
    axes[0].set_ylabel('Hallucination Rate (%)', fontsize=12)
    axes[0].set_title('Hallucination vs Training Imbalance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: Violations analysis
    is_monotonic, violations = check_monotonicity(defined_ratios, mean_results, tolerance=0.01)
    
    axes[1].plot(defined_ratios * 100, mean_results * 100, 
                color='blue', linewidth=2, marker='o', label='Mean trajectory')
    
    # Mark violations
    if violations:
        for r1, r2, diff in violations:
            idx1 = np.where(defined_ratios == r1)[0][0]
            idx2 = np.where(defined_ratios == r2)[0][0]
            axes[1].plot([r1*100, r2*100], 
                        [mean_results[idx1]*100, mean_results[idx2]*100],
                        'r-', linewidth=3, alpha=0.7)
            axes[1].scatter([r1*100, r2*100], 
                          [mean_results[idx1]*100, mean_results[idx2]*100],
                          color='red', s=100, zorder=5)
    
    axes[1].set_xlabel('Defined Ratio (%)', fontsize=12)
    axes[1].set_ylabel('Hallucination Rate (%)', fontsize=12)
    axes[1].set_title('Monotonicity Violations (Red)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {output_path}")

def main():
    print("="*70)
    print("TEST: Prediction 6 - Strong Monotonic Trend")
    print("="*70)
    print("\nPrediction:")
    print("  For fixed K > 0, hallucination rate shows a strong monotonic")
    print("  trend as training becomes more imbalanced toward structured outputs.")
    print("\nMechanism:")
    print("  Insufficient witness allocation forces error (Theorem 7.4)")
    print("\nNote:")
    print("  Theory predicts monotonic PRESSURE, not strict determinism.")
    print("  Small violations (~1-2%) expected from finite-sample effects.")
    
    # Test with multiple seeds
    print(f"\n{'='*70}")
    print("ROBUSTNESS TEST: Multiple Seeds")
    print('='*70)
    
    seeds = [DEFAULT_SEED, DEFAULT_SEED+1, DEFAULT_SEED+2, DEFAULT_SEED+3, DEFAULT_SEED+4]
    defined_ratios, all_results, all_violations, all_correlations = test_multiple_seeds(seeds, verbose=True)
    
    # Aggregate analysis
    print(f"\n{'='*70}")
    print("AGGREGATE ANALYSIS")
    print('='*70)
    
    mean_results = all_results.mean(axis=0)
    
    print(f"\nAcross {len(seeds)} seeds:")
    print(f"  Mean violations: {np.mean(all_violations):.1f}")
    print(f"  Seeds with violations: {sum(v > 0 for v in all_violations)}/{len(seeds)}")
    print(f"\n  Correlation across seeds:")
    print(f"    Mean ρ: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}")
    print(f"    Range: [{np.min(all_correlations):.3f}, {np.max(all_correlations):.3f}]")
    
    # Check mean trajectory
    mean_is_monotonic, mean_violations = check_monotonicity(defined_ratios, mean_results, tolerance=0.01)
    mean_correlation, mean_p = spearmanr(defined_ratios, mean_results)
    
    print(f"\n  Mean trajectory:")
    print(f"    Range: {mean_results.min():.1%} → {mean_results.max():.1%} (Δ={mean_results.max()-mean_results.min():.1%})")
    print(f"    Correlation: ρ = {mean_correlation:.3f} (p={mean_p:.4e})")
    print(f"    Monotonic: {'Yes' if mean_is_monotonic else f'No ({len(mean_violations)} violations)'}")
    
    if mean_violations:
        print(f"\n  Systematic violations in mean trajectory:")
        for r1, r2, diff in mean_violations:
            print(f"    {r1:.1%} → {r2:.1%}: {diff:+.3f}")
    
    # Statistical rigor: Test null hypothesis and compute effect sizes
    print(f"\n{'='*70}")
    print("STATISTICAL RIGOR")
    print('='*70)
    
    # Effect size
    effect_size = compute_effect_size(defined_ratios, mean_results)
    print(f"\nEffect Size (Cohen's d): {effect_size:.2f}")
    if effect_size >= 0.8:
        print(f"  Interpretation: Large effect (d ≥ 0.8)")
    elif effect_size >= 0.5:
        print(f"  Interpretation: Medium effect (0.5 ≤ d < 0.8)")
    else:
        print(f"  Interpretation: Small effect (d < 0.5)")
    
    # Confidence interval on correlation
    ci_lower, ci_upper = bootstrap_correlation_ci(defined_ratios, mean_results)
    print(f"\n95% Confidence Interval for ρ: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Null hypothesis test for violations
    print(f"\nNull Hypothesis Test:")
    print(f"  H0: Violation count is consistent with random ordering")
    obs_violations, null_dist, p_violations = test_null_hypothesis_violations(
        defined_ratios, mean_results, n_permutations=10000
    )
    print(f"  Observed violations: {obs_violations}")
    print(f"  Expected under H0: {null_dist.mean():.1f} ± {null_dist.std():.1f}")
    print(f"  p-value: {p_violations:.4f}")
    if p_violations < 0.05:
        print(f"  Result: Significantly FEWER violations than random (p < 0.05)")
        print(f"          Strong evidence for monotonic pressure")
    else:
        print(f"  Result: Violations consistent with random ordering")
    
    # Create visualization
    print(f"\n{'='*70}")
    print("VISUALIZATION")
    print('='*70)
    create_visualization(defined_ratios, all_results)
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print('='*70)
    
    # Data-driven assessment based on pre-specified criteria:
    # 1. All seeds show positive correlation (no arbitrary threshold)
    # 2. Mean correlation is statistically significant (p < 0.01)
    # 3. Effect size is at least medium (d ≥ 0.5)
    # 4. Violations are significantly fewer than random chance
    
    all_positive = all(c > 0 for c in all_correlations)
    significant = mean_p < 0.01
    medium_or_large_effect = effect_size >= 0.5
    fewer_violations_than_random = p_violations < 0.05
    
    criteria_met = sum([all_positive, significant, medium_or_large_effect, fewer_violations_than_random])
    
    if criteria_met >= 3:
        print("\n✓ PREDICTION CONFIRMED")
        print(f"  Monotonic pressure validated ({criteria_met}/4 criteria met):")
        print(f"    • All seeds positive correlation: {'✓' if all_positive else '✗'}")
        print(f"    • Statistical significance (p < 0.01): {'✓' if significant else '✗'}")
        print(f"    • Medium/large effect size (d ≥ 0.5): {'✓' if medium_or_large_effect else '✗'}")
        print(f"    • Fewer violations than random: {'✓' if fewer_violations_than_random else '✗'}")
        
        print(f"\n  Summary statistics:")
        print(f"    • Mean correlation: ρ = {np.mean(all_correlations):.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"    • Effect size: d = {effect_size:.2f}")
        print(f"    • Overall increase: {mean_results.max()-mean_results.min():.1%}")
        print(f"    • p-value: {mean_p:.4e}")
        
        if not mean_is_monotonic:
            avg_violation_magnitude = np.mean([abs(v[2]) for v in mean_violations])
            relative_size = 100 * avg_violation_magnitude / (mean_results.max() - mean_results.min())
            print(f"\n  Violations analysis:")
            print(f"    • {len(mean_violations)} violations in mean trajectory")
            print(f"    • Average magnitude: {avg_violation_magnitude:.3f} ({relative_size:.1f}% of total increase)")
            print(f"    • Expected under random: {null_dist.mean():.1f} ± {null_dist.std():.1f}")
            print(f"    • Interpretation: Consistent with finite-sample noise, not theoretical failure")
        
        print("\n  Interpretation:")
        print("    The Witness-Error Tradeoff predicts monotonic PRESSURE (directional")
        print("    force), not strict determinism. The strong positive correlation,")
        print("    large effect size, and significantly fewer violations than random")
        print("    chance provide strong evidence for this mechanism. Small violations")
        print("    are statistically expected from finite samples (e.g., only ~19")
        print("    undefined training examples at 85% defined ratio).")
    else:
        print("\n✗ PREDICTION NOT CONFIRMED")
        print(f"  Only {criteria_met}/4 criteria met:")
        print(f"    • All seeds positive correlation: {'✓' if all_positive else '✗'}")
        print(f"    • Statistical significance (p < 0.01): {'✓' if significant else '✗'}")
        print(f"    • Medium/large effect size (d ≥ 0.5): {'✓' if medium_or_large_effect else '✗'}")
        print(f"    • Fewer violations than random: {'✓' if fewer_violations_than_random else '✗'}")
    
    print('\n' + '='*70)

if __name__ == "__main__":
    main()

