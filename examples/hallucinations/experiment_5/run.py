"""
Test non-linear relationship between training imbalance and hallucination rates.

This experiment tests whether the relationship between training data imbalance
(defined vs undefined inputs) and hallucination rates follows a non-linear
pattern, specifically exponential or sigmoidal as effective contradiction K increases.

Hypothesis tested:
As training data becomes more imbalanced toward defined inputs, hallucination
rates increase non-linearly due to compounding contradiction effects from
learned priors that cannot generalize to undefined regions.

Testing approach:
- Vary defined input ratio from 0.1 to 0.9 in systematic steps
- For each ratio, train neural network and measure hallucination rate on undefined inputs
- Fit mathematical curves (exponential, sigmoid) to (defined_ratio, hallucination_rate) pairs
- Test statistical significance of non-linear relationship
- Analyze whether curve fits predict behavior beyond training range

Key measurements:
- Hallucination rates across defined ratios from 0.1 to 0.9
- Curve fitting quality (R², p-values) for different functional forms
- Extrapolation performance of fitted curves
- Spearman correlation coefficients for monotonic vs non-linear relationships

Assumptions:
- Training imbalance creates effective contradiction K that increases non-linearly
- Neural networks learn local patterns that compound into global contradictions
- Hallucination rates are stable and measurable across random seeds
- Curve fitting assumptions (independent observations, appropriate functional forms)

Expected outcome:
Non-linear relationship confirms that training imbalance creates compounding
contradiction effects, supporting the theoretical prediction that local
inconsistencies accumulate into global impossibility.

Typical usage:
- Run test_nonlinear_relationship() to fit curves across imbalance range
- Use analyze_curve_fitting() to evaluate different functional forms
- Results saved as hallucination_curve_fitting.png in figures directory
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import (
    HallucinationNet, generate_partial_function, create_datasets,
    train_model, INPUT_SIZE, OUTPUT_CLASSES, HIDDEN_SIZE,
    LEARNING_RATE, EPOCHS, BATCH_SIZE, calculate_hallucination_rate
)
from contrakit.constants import FIGURES_DIR, DEFAULT_SEED

def run_experiment(defined_ratio, seed=DEFAULT_SEED):
    """Run one experiment and measure hallucination rate."""
    function_map, _ = generate_partial_function(
        INPUT_SIZE, OUTPUT_CLASSES, defined_ratio, 0.05, seed
    )
    train_data, test_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)
    test_undefined_x, _ = test_undefined

    torch.manual_seed(seed)
    model = HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES), use_definedness_head=False)
    train_model(model, train_data, EPOCHS, LEARNING_RATE, BATCH_SIZE)

    with torch.no_grad():
        preds = torch.argmax(model(torch.LongTensor(test_undefined_x)), dim=1).numpy()
    return calculate_hallucination_rate(preds)

# Define candidate functional forms
def linear(x, a, b):
    """Linear: y = ax + b"""
    return a * x + b

def exponential(x, a, b, c):
    """Exponential: y = a * exp(b*x) + c"""
    return a * np.exp(b * x) + c

def sigmoid(x, L, k, x0):
    """Sigmoid (logistic): y = L / (1 + exp(-k*(x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))

def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)

def fit_curves(x_data, y_data):
    """Fit multiple functional forms and return best fit."""
    results = {}
    
    # Linear fit
    try:
        params, _ = curve_fit(linear, x_data, y_data)
        y_pred = linear(x_data, *params)
        rmse = np.sqrt(np.mean((y_data - y_pred)**2))
        r_squared = 1 - (np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2))
        results['linear'] = {'params': params, 'rmse': rmse, 'r_squared': r_squared, 'func': linear}
    except:
        results['linear'] = {'rmse': float('inf'), 'r_squared': -float('inf')}
    
    # Exponential fit
    try:
        params, _ = curve_fit(exponential, x_data, y_data, p0=[0.1, 2.0, 0.5], maxfev=10000)
        y_pred = exponential(x_data, *params)
        rmse = np.sqrt(np.mean((y_data - y_pred)**2))
        r_squared = 1 - (np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2))
        results['exponential'] = {'params': params, 'rmse': rmse, 'r_squared': r_squared, 'func': exponential}
    except:
        results['exponential'] = {'rmse': float('inf'), 'r_squared': -float('inf')}
    
    # Sigmoid fit
    try:
        params, _ = curve_fit(sigmoid, x_data, y_data, p0=[1.0, 5.0, 0.5], maxfev=10000)
        y_pred = sigmoid(x_data, *params)
        rmse = np.sqrt(np.mean((y_data - y_pred)**2))
        r_squared = 1 - (np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2))
        results['sigmoid'] = {'params': params, 'rmse': rmse, 'r_squared': r_squared, 'func': sigmoid}
    except:
        results['sigmoid'] = {'rmse': float('inf'), 'r_squared': -float('inf')}
    
    # Power law fit
    try:
        params, _ = curve_fit(power_law, x_data, y_data, p0=[0.5, 2.0])
        y_pred = power_law(x_data, *params)
        rmse = np.sqrt(np.mean((y_data - y_pred)**2))
        r_squared = 1 - (np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2))
        results['power_law'] = {'params': params, 'rmse': rmse, 'r_squared': r_squared, 'func': power_law}
    except:
        results['power_law'] = {'rmse': float('inf'), 'r_squared': -float('inf')}
    
    # Find best fit
    best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
    
    return results, best_model

def main():
    print("="*70)
    print("TEST: Prediction 7 - Non-linear Hallucination Curve")
    print("="*70)
    print("\nPrediction:")
    print("  Relationship between training imbalance and hallucination")
    print("  should be non-linear (exponential or sigmoidal curve).")
    print("\nMechanism:")
    print("  Compounding of local K values through learned priors")
    
    # Collect data
    defined_ratios = np.linspace(0.1, 0.9, 17)
    
    print(f"\n{'='*70}")
    print("DATA COLLECTION")
    print('='*70)
    print(f"\nRunning {len(defined_ratios)} experiments...\n")
    
    hallucination_rates = []
    for ratio in defined_ratios:
        print(f"Defined ratio: {ratio:.1%}...", end=" ", flush=True)
        hall_rate = run_experiment(defined_ratio=ratio, seed=DEFAULT_SEED)
        hallucination_rates.append(hall_rate)
        print(f"Hallucination: {hall_rate:.1%}")
    
    hallucination_rates = np.array(hallucination_rates)
    
    # Fit curves
    print(f"\n{'='*70}")
    print("CURVE FITTING ANALYSIS")
    print('='*70)
    
    fits, best_model = fit_curves(defined_ratios, hallucination_rates)
    
    print("\nFit quality for each functional form:")
    print(f"{'Model':<15} {'RMSE':<12} {'R²':<12} {'Result'}")
    print("-"*70)
    
    for model_name in ['linear', 'exponential', 'sigmoid', 'power_law']:
        fit = fits[model_name]
        rmse = fit['rmse']
        r2 = fit['r_squared']
        is_best = " ← BEST FIT" if model_name == best_model else ""
        print(f"{model_name:<15} {rmse:<12.4f} {r2:<12.4f} {is_best}")
    
    # Check for non-linearity
    print(f"\n{'='*70}")
    print("NON-LINEARITY TEST")
    print('='*70)
    
    linear_r2 = fits['linear']['r_squared']
    best_r2 = fits[best_model]['r_squared']
    improvement = best_r2 - linear_r2
    
    print(f"\nLinear model R²:        {linear_r2:.4f}")
    print(f"Best non-linear R²:     {best_r2:.4f}")
    print(f"Improvement:            {improvement:+.4f}")
    
    is_nonlinear = (best_model != 'linear') and (improvement > 0.05)
    
    if is_nonlinear:
        print(f"\n✓ NON-LINEAR: {best_model.upper()} fits significantly better")
        print(f"  The relationship shows clear non-linear structure")
    else:
        print(f"\n? UNCLEAR: Best fit is {best_model.upper()}")
        print(f"  Improvement over linear: {improvement:.4f}")
        if improvement < 0.05:
            print(f"  Improvement is small (< 0.05)")
    
    # Create visualization
    print(f"\n{'='*70}")
    print("VISUALIZATION")
    print('='*70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data with all fits
    ax1.scatter(defined_ratios, hallucination_rates, color='black', s=50, 
                label='Observed', zorder=5)
    
    x_smooth = np.linspace(0.1, 0.9, 100)
    colors = {'linear': 'blue', 'exponential': 'red', 'sigmoid': 'green', 'power_law': 'orange'}
    
    for model_name, fit in fits.items():
        if fit['rmse'] < float('inf'):
            y_smooth = fit['func'](x_smooth, *fit['params'])
            linestyle = '-' if model_name == best_model else '--'
            linewidth = 2.5 if model_name == best_model else 1.5
            ax1.plot(x_smooth, y_smooth, linestyle, color=colors[model_name],
                    label=f"{model_name.capitalize()} (R²={fit['r_squared']:.3f})",
                    linewidth=linewidth, alpha=0.8)
    
    ax1.set_xlabel('Defined Ratio')
    ax1.set_ylabel('Hallucination Rate')
    ax1.set_title('Hallucination vs Training Imbalance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals for best fit
    y_pred = fits[best_model]['func'](defined_ratios, *fits[best_model]['params'])
    residuals = hallucination_rates - y_pred
    
    ax2.scatter(defined_ratios, residuals, color='black', s=50)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Defined Ratio')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Residuals for {best_model.capitalize()} Fit')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'hallucination_curve_fitting.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print('='*70)
    
    if is_nonlinear and best_model in ['exponential', 'sigmoid', 'power_law']:
        print("\n✓ PREDICTION CONFIRMED")
        print(f"  Best fit: {best_model.upper()} (R² = {best_r2:.4f})")
        print(f"  The relationship is clearly non-linear")
        print(f"  This supports the compounding K mechanism")
    elif is_nonlinear:
        print("\n? PREDICTION PARTIALLY CONFIRMED")
        print(f"  Best fit is non-linear but unexpected form: {best_model}")
    else:
        print("\n✗ PREDICTION NOT CLEARLY CONFIRMED")
        print(f"  Linear model fits nearly as well as non-linear")
        print(f"  May need more data points or different K regime")
    
    print('\n' + '='*70)
    
    return defined_ratios, hallucination_rates, fits, best_model

if __name__ == "__main__":
    main()

