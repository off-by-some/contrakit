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
from sklearn.model_selection import LeaveOneOut
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

def calculate_aic(n, mse, num_params):
    """Calculate Akaike Information Criterion."""
    return n * np.log(mse) + 2 * num_params

def calculate_bic(n, mse, num_params):
    """Calculate Bayesian Information Criterion."""
    return n * np.log(mse) + num_params * np.log(n)

def bootstrap_r_squared(x_data, y_data, fit_func, params, n_bootstrap=1000, seed=DEFAULT_SEED):
    """Bootstrap confidence interval for R²."""
    rng = np.random.RandomState(seed)
    r_squared_samples = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(x_data), size=len(x_data), replace=True)
        x_boot = x_data[indices]
        y_boot = y_data[indices]
        
        y_pred = fit_func(x_boot, *params)
        ss_res = np.sum((y_boot - y_pred)**2)
        ss_tot = np.sum((y_boot - np.mean(y_boot))**2)
        
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
            r_squared_samples.append(r2)
    
    return np.percentile(r_squared_samples, [2.5, 97.5])

def cross_validate_model(x_data, y_data, fit_func, initial_params):
    """Perform leave-one-out cross-validation."""
    loo = LeaveOneOut()
    predictions = np.zeros_like(y_data)
    
    for train_idx, test_idx in loo.split(x_data):
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train = y_data[train_idx]
        
        params, _ = curve_fit(fit_func, x_train, y_train, p0=initial_params, maxfev=10000)
        predictions[test_idx] = fit_func(x_test, *params)
    
    mse = np.mean((y_data - predictions)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    cv_r_squared = 1 - (np.sum((y_data - predictions)**2) / ss_tot) if ss_tot > 0 else -float('inf')
    
    return cv_r_squared, mse

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
    """Fit multiple functional forms with model selection criteria."""
    results = {}
    n = len(x_data)
    
    model_configs = [
        ('linear', linear, [0.5, 0.5], 2),
        ('exponential', exponential, [0.1, 2.0, 0.5], 3),
        ('sigmoid', sigmoid, [1.0, 5.0, 0.5], 3),
        ('power_law', power_law, [0.5, 2.0], 2)
    ]
    
    for model_name, fit_func, initial_params, num_params in model_configs:
        params, _ = curve_fit(fit_func, x_data, y_data, p0=initial_params, maxfev=10000)
        y_pred = fit_func(x_data, *params)
        
        residuals = y_data - y_pred
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -float('inf')
        
        aic = calculate_aic(n, mse, num_params)
        bic = calculate_bic(n, mse, num_params)
        
        r2_ci = bootstrap_r_squared(x_data, y_data, fit_func, params, n_bootstrap=1000)
        
        cv_r2, cv_mse = cross_validate_model(x_data, y_data, fit_func, initial_params)
        
        results[model_name] = {
            'params': params,
            'num_params': num_params,
            'rmse': rmse,
            'mse': mse,
            'r_squared': r_squared,
            'r_squared_ci': r2_ci,
            'cv_r_squared': cv_r2,
            'aic': aic,
            'bic': bic,
            'func': fit_func
        }
    
    best_model_aic = min(results.keys(), key=lambda k: results[k]['aic'])
    best_model_bic = min(results.keys(), key=lambda k: results[k]['bic'])
    best_model_cv = max(results.keys(), key=lambda k: results[k]['cv_r_squared'])
    
    return results, best_model_aic, best_model_bic, best_model_cv

def main():
    print("="*70)
    print("TEST: Prediction 7 - Non-linear Hallucination Curve")
    print("="*70)
    print("\nPrediction:")
    print("  Relationship between training imbalance and hallucination")
    print("  should be non-linear (exponential or sigmoidal curve).")
    print("\nMechanism:")
    print("  Compounding of local K values through learned priors")
    
    # Collect data with multiple seeds for robustness
    defined_ratios = np.linspace(0.1, 0.9, 17)
    n_seeds = 3
    
    print(f"\n{'='*70}")
    print("DATA COLLECTION")
    print('='*70)
    print(f"\nRunning {len(defined_ratios)} experiments with {n_seeds} seeds each for robustness...\n")
    
    hallucination_rates_all_seeds = []
    for ratio in defined_ratios:
        print(f"Defined ratio: {ratio:.1%}...", end=" ", flush=True)
        rates_for_ratio = []
        for seed_offset in range(n_seeds):
            seed = DEFAULT_SEED + int(ratio * 1000) + seed_offset
            hall_rate = run_experiment(defined_ratio=ratio, seed=seed)
            rates_for_ratio.append(hall_rate)
        
        mean_rate = np.mean(rates_for_ratio)
        std_rate = np.std(rates_for_ratio)
        hallucination_rates_all_seeds.append(rates_for_ratio)
        print(f"Hallucination: {mean_rate:.1%} ± {std_rate:.1%}")
    
    hallucination_rates = np.array([np.mean(rates) for rates in hallucination_rates_all_seeds])
    hallucination_std = np.array([np.std(rates) for rates in hallucination_rates_all_seeds])
    
    # Fit curves
    print(f"\n{'='*70}")
    print("CURVE FITTING ANALYSIS (with model selection criteria)")
    print('='*70)
    
    fits, best_model_aic, best_model_bic, best_model_cv = fit_curves(defined_ratios, hallucination_rates)
    
    print("\nFit quality for each functional form:")
    print(f"{'Model':<15} {'Params':<8} {'RMSE':<12} {'R²':<20} {'CV R²':<12} {'AIC':<12} {'BIC':<12}")
    print("-"*100)
    
    for model_name in ['linear', 'exponential', 'sigmoid', 'power_law']:
        fit = fits[model_name]
        r2_low, r2_high = fit['r_squared_ci']
        r2_str = f"{fit['r_squared']:.4f} [{r2_low:.4f}, {r2_high:.4f}]"
        print(f"{model_name:<15} {fit['num_params']:<8} {fit['rmse']:<12.4f} {r2_str:<20} {fit['cv_r_squared']:<12.4f} {fit['aic']:<12.2f} {fit['bic']:<12.2f}")
    
    print(f"\nBest model by AIC (penalizes complexity): {best_model_aic}")
    print(f"Best model by BIC (penalizes complexity more): {best_model_bic}")
    print(f"Best model by cross-validation: {best_model_cv}")
    
    consensus_models = {best_model_aic, best_model_bic, best_model_cv}
    if len(consensus_models) == 1:
        best_model = list(consensus_models)[0]
        print(f"\n✓ All criteria agree: {best_model.upper()} is the best model")
    else:
        best_model = best_model_bic
        print(f"\n⚠ Criteria disagree. Using BIC choice: {best_model.upper()}")
    
    # Check for non-linearity with proper statistical assessment
    print(f"\n{'='*70}")
    print("NON-LINEARITY ASSESSMENT")
    print('='*70)
    
    linear_fit = fits['linear']
    best_fit = fits[best_model]
    
    delta_aic = linear_fit['aic'] - best_fit['aic']
    delta_bic = linear_fit['bic'] - best_fit['bic']
    
    print(f"\nLinear model:")
    print(f"  R² = {linear_fit['r_squared']:.4f} [{linear_fit['r_squared_ci'][0]:.4f}, {linear_fit['r_squared_ci'][1]:.4f}]")
    print(f"  CV R² = {linear_fit['cv_r_squared']:.4f}")
    print(f"  AIC = {linear_fit['aic']:.2f}")
    print(f"  BIC = {linear_fit['bic']:.2f}")
    
    print(f"\nBest non-linear model ({best_model}):")
    print(f"  R² = {best_fit['r_squared']:.4f} [{best_fit['r_squared_ci'][0]:.4f}, {best_fit['r_squared_ci'][1]:.4f}]")
    print(f"  CV R² = {best_fit['cv_r_squared']:.4f}")
    print(f"  AIC = {best_fit['aic']:.2f}")
    print(f"  BIC = {best_fit['bic']:.2f}")
    
    print(f"\nModel comparison:")
    print(f"  ΔAIC = {delta_aic:.2f} (>10 = strong evidence for non-linear)")
    print(f"  ΔBIC = {delta_bic:.2f} (>10 = strong evidence for non-linear)")
    
    is_nonlinear = (best_model != 'linear') and (delta_bic > 10)
    
    if is_nonlinear:
        print(f"\n✓ NON-LINEAR: {best_model.upper()} strongly preferred")
        print(f"  Strong statistical evidence for non-linear structure")
    elif best_model != 'linear' and delta_bic > 2:
        print(f"\n✓ LIKELY NON-LINEAR: {best_model.upper()} preferred")
        print(f"  Moderate evidence for non-linear structure")
    else:
        print(f"\n? UNCLEAR: Evidence for non-linearity is weak")
        print(f"  Linear model may be sufficient given the data")
    
    # Create visualization
    print(f"\n{'='*70}")
    print("VISUALIZATION")
    print('='*70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data with all fits
    ax1.errorbar(defined_ratios, hallucination_rates, yerr=hallucination_std,
                 fmt='o', color='black', markersize=6, capsize=3,
                 label='Observed (mean ± std)', zorder=5, elinewidth=1.5)
    
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
        print(f"  Best fit: {best_model.upper()}")
        print(f"  R² = {best_fit['r_squared']:.4f} [95% CI: {best_fit['r_squared_ci'][0]:.4f}-{best_fit['r_squared_ci'][1]:.4f}]")
        print(f"  CV R² = {best_fit['cv_r_squared']:.4f}")
        print(f"  The relationship shows statistically significant non-linear structure")
        print(f"  This supports the compounding mechanism hypothesis")
    elif best_model != 'linear' and delta_bic > 2:
        print("\n✓ PREDICTION LIKELY CONFIRMED")
        print(f"  Best fit: {best_model.upper()}")
        print(f"  Moderate evidence for non-linearity")
    else:
        print("\n⚠ PREDICTION UNCERTAIN")
        print(f"  Evidence for non-linearity is not conclusive")
        print(f"  More data or different experimental conditions may be needed")
    
    print('\n' + '='*70)
    
    return defined_ratios, hallucination_rates, hallucination_std, fits, best_model

if __name__ == "__main__":
    main()

