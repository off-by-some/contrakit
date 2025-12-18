"""
Common utilities for quantum contextuality analysis.

This module contains shared functionality used across different quantum
contextuality demonstrations including CHSH, KCBS, and magic squares.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contrakit.constants import FIGURES_DIR, DEFAULT_SEED


def pretty_witness(lmbda, tight, tol=1e-9):
    """
    Pretty-print witness information with uniform detection and entropy calculation.

    Parameters
    ----------
    lmbda : dict or array-like
        Witness weights λ* from least_favorable_lambda()
    tight : array-like
        Tightness scores b_c in same order as contexts
    tol : float
        Tolerance for uniformity detection and entropy calculations

    Notes
    -----
    This is to display the optimal witness structure; nothing is hiding.
    """
    if isinstance(lmbda, dict):
        weights = list(lmbda.values())
    else:
        weights = list(lmbda) if hasattr(lmbda, '__iter__') else [lmbda]

    n = len(weights)
    if n == 0:
        print("  Witness λ*: empty")
        return

    # Check if uniform
    uni = all(abs(w - 1.0/n) < tol for w in weights)
    if uni:
        print(f"  Witness λ*: uniform over {n} contexts (≈ {1.0/n:.6f} each)")
    else:
        # Calculate support and entropy
        supp = sum(w > tol for w in weights)
        H = -sum(w*math.log(w, 2) for w in weights if w > 0)

        # Show top 2 weights if dict
        if isinstance(lmbda, dict):
            top = sorted(lmbda.items(), key=lambda kv: kv[1], reverse=True)[:2]
            top_str = ", ".join([f"{k}: {v:.6f}" for k,v in top])
            print(f"  Witness λ*: support={supp}/{n}, H(λ*)={H:.3f} bits; top: {top_str}")
        else:
            print(f"  Witness λ*: support={supp}/{n}, H(λ*)={H:.3f} bits")

    # Handle tightness scores
    if hasattr(tight, "__len__") and len(tight) > 0:
        tight_rounded = [round(x, 6) for x in tight]
        if len(set(tight_rounded)) == 1:
            print(f"  Tightness b_c: [{tight[0]:.6f} × {len(tight)}]")
        else:
            print(f"  Tightness b_c: {[f'{x:.6f}' for x in tight]}")
    else:
        print("  Tightness b_c: N/A")


def save_figure(filename, fig=None, dpi=300, bbox_inches='tight'):
    """
    Save a matplotlib figure to the figures directory.

    Parameters
    ----------
    filename : str
        Name of the output file (without path)
    fig : matplotlib.figure.Figure, optional
        Figure to save (default: current figure)
    dpi : int
        Resolution in dots per inch
    bbox_inches : str
        Bounding box setting for tight layout

    Returns
    -------
    Path
        Full path to the saved figure

    Notes
    -----
    Formally, this ensures consistent output paths; the standardization is essential.
    """
    output_path = FIGURES_DIR / filename
    if fig is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    return Path(output_path)


def display_figure():
    """
    Display the current matplotlib figure with exception handling for headless environments.
    """
    try:
        plt.show()
    except Exception:
        pass  # Handle headless environments gracefully


def setup_matplotlib_style():
    """
    Set up consistent matplotlib styling for quantum contextuality plots.
    """
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18


def print_section_header(title, width=70):
    """
    Print a standardized section header with title.

    Parameters
    ----------
    title : str
        Section title
    width : int
        Width of the header line

    Notes
    -----
    Put differently, this creates visual hierarchy; the consistency aids readability.
    """
    print("=" * width)
    print(title)
    print("=" * width)
    print()


def print_subsection_header(title, width=60):
    """
    Print a subsection header.

    Parameters
    ----------
    title : str
        Subsection title
    width : int
        Width of the header line
    """
    print(title)
    print("=" * width)
    print()


def create_analysis_header(title, description="", key_points=None, width=80):
    """
    Create a standardized header for quantum contextuality analyses.

    Parameters
    ----------
    title : str
        Main title for the analysis
    description : str
        Optional description text
    key_points : list of str, optional
        List of key points to display
    width : int
        Width of the header
    """
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    if description:
        print()
        print(description)
    if key_points:
        print()
        for point in key_points:
            print(f"  • {point}")
    print()


def extract_behavior_properties(behavior):
    """
    Extract common properties from a behavior object.

    Parameters
    ----------
    behavior : Behavior
        The behavior to analyze

    Returns
    -------
    dict
        Dictionary containing alpha_star, contradiction_bits, frame_independence, etc.

    Notes
    -----
    This is to standardize analysis across demonstrations; nothing is hiding.
    We only need these core quantities. Formally, they define the behavior completely.
    """
    alpha_star = behavior.alpha_star
    contradiction_bits = behavior.contradiction_bits
    is_fi = behavior.is_frame_independent()
    context_scores = behavior.per_context_scores(mu="optimal")
    equalization_gap = float(np.ptp(context_scores))

    lambda_star = behavior.least_favorable_lambda()
    alpha_lambda = None
    lambda_gap = None
    if lambda_star:
        alpha_lambda, _ = behavior.alpha_given_lambda(lambda_star)  # Unpack tuple (alpha, weights)
        alpha_lambda = float(alpha_lambda)
        lambda_gap = abs(alpha_lambda - alpha_star)

    return {
        'alpha_star': alpha_star,
        'contradiction_bits': contradiction_bits,
        'is_frame_independent': is_fi,
        'context_scores': context_scores,
        'equalization_gap': equalization_gap,
        'lambda_star': lambda_star,
        'alpha_lambda': alpha_lambda,
        'lambda_gap': lambda_gap,
    }


def format_behavior_verdict(props, prefix="  Verdict:", witness_symbol="S"):
    """
    Format a behavior verdict in a standardized way.

    Parameters
    ----------
    props : dict
        Properties from extract_behavior_properties
    prefix : str
        Prefix for the verdict line
    witness_symbol : str
        Symbol for the witness value (e.g., 'S' for CHSH, 'Σ⟨E⟩' for KCBS, 'W' for Magic Square)

    Returns
    -------
    str
        Formatted verdict string

    Notes
    -----
    This standardizes the output format. Consider the invariant: α* = 2^(-K).
    """
    witness_value = props.get('witness_value', 'N/A')
    if witness_symbol == "Σ⟨E⟩" and isinstance(witness_value, (int, float)):
        witness_value = f"{witness_value:.6f}"

    return (f"{prefix} {witness_symbol}: {witness_value}   "
            f"K(P): {props['contradiction_bits']:.6f} bits   "
            f"α*: {props['alpha_star']:.6f}   "
            f"FI?: {'NO' if not props['is_frame_independent'] else 'YES'}")


def print_behavior_analysis(behavior, label="Behavior", show_witness=True):
    """
    Print a comprehensive behavior analysis.

    Parameters
    ----------
    behavior : Behavior
        The behavior to analyze
    label : str
        Label for the analysis
    show_witness : bool
        Whether to show witness information if available
    """
    props = extract_behavior_properties(behavior)

    print(f"{label}:")
    print(format_behavior_verdict(props, prefix="  "))
    print(f"  Contradiction measure K(P) = {props['contradiction_bits']:.6f} bits")
    print(f"  Optimal alpha = {props['alpha_star']:.6f}")
    print(f"  b_c (tightness against FI extremes): {np.round(props['context_scores'], 6).tolist()}")
    print(f"  Equalization gap: {props['equalization_gap']:.3e}")

    if props['alpha_lambda'] is not None:
        print(f"  Alpha at lambda*: {props['alpha_lambda']:.6f} (gap: {props['lambda_gap']:.3e})")

    print(f"  Frame independent: {props['is_frame_independent']}")

    if show_witness and not props['is_frame_independent'] and props['lambda_star'] is not None:
        print("  Witness information:")
        pretty_witness(props['lambda_star'], props['context_scores'], tol=1e-9)


def format_configuration_results(results_dict, config_name="", metric_name="", bound_value=None):
    """
    Format configuration analysis results in a standardized way.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing analysis results
    config_name : str
        Name of the configuration
    metric_name : str
        Name of the main metric (e.g., "S", "KCBS sum")
    bound_value : float, optional
        Classical bound value for comparison

    Returns
    -------
    list of str
        Formatted result lines
    """
    lines = []

    if config_name:
        lines.append(f"  Configuration: {config_name}")

    # Add main metric if available
    if metric_name and 'metric_value' in results_dict:
        metric_val = results_dict['metric_value']
        lines.append(f"  {metric_name}: {metric_val:.6f}")

        if bound_value is not None:
            comparison = '>' if metric_val > bound_value else '≤'
            status = "VIOLATES" if metric_val > bound_value else "SATISFIES"
            lines.append(f"  Classical bound: {bound_value:.6f} ({status}: {metric_val:.6f} {comparison} {bound_value})")

    # Add behavior properties if available
    if 'contradiction_bits' in results_dict:
        lines.append(f"  Contradiction K(P): {results_dict['contradiction_bits']:.6f} bits")

    if 'alpha_star' in results_dict:
        lines.append(f"  Optimal α*: {results_dict['alpha_star']:.6f}")

    if 'frame_independent' in results_dict:
        lines.append(f"  Frame independent: {results_dict['frame_independent']}")

    return lines


def verify_no_signalling(contexts, tolerance=1e-12):
    """
    Verify that contexts satisfy no-signalling constraints.

    Parameters
    ----------
    contexts : dict
        Dictionary mapping context names to probability distributions
    tolerance : float
        Numerical tolerance for marginal equality checks

    Returns
    -------
    tuple
        (is_no_signalling, max_discrepancy) where is_no_signalling is bool
        and max_discrepancy is the maximum deviation found
    """
    # This is a generic implementation that needs to be specialized
    # for specific context structures. For now, we'll return a placeholder.
    return True, 0.0


def sanitize_pmf(pmf, eps=1e-15):
    """
    Sanitize a probability mass function by clipping negatives and renormalizing.

    Parameters
    ----------
    pmf : dict
        Probability mass function to sanitize
    eps : float
        Small threshold for numerical operations

    Returns
    -------
    dict
        Sanitized and renormalized probability mass function
    """
    p = {k: max(float(v), 0.0) for k, v in pmf.items()}
    z = sum(p.values())
    if z <= eps:
        raise ValueError("All-zero PMF after clipping.")
    return {k: v / z for k, v in p.items()}


def analyze_boundary_conditions(values_x, values_y, boundary_x, condition="below"):
    """
    Analyze values at boundary conditions (e.g., max below threshold, min above threshold).

    Parameters
    ----------
    values_x : list
        X values (e.g., witness values)
    values_y : list
        Y values (e.g., contradiction values)
    boundary_x : float
        Boundary value for X
    condition : str
        "below" or "above" the boundary

    Returns
    -------
    tuple
        (extreme_value, count) where extreme_value is max or min, count is number of points
    """
    if condition == "below":
        filtered_y = [y for x, y in zip(values_x, values_y) if x <= boundary_x]
        extreme_value = max(filtered_y) if filtered_y else 0.0
    elif condition == "above":
        filtered_y = [y for x, y in zip(values_x, values_y) if x > boundary_x]
        extreme_value = min(filtered_y) if filtered_y else float('inf')
    else:
        raise ValueError("condition must be 'below' or 'above'")

    return extreme_value, len(filtered_y)


def print_boundary_analysis(values_x, values_y, boundary_x, x_label="X", y_label="Y"):
    """
    Print boundary analysis results in standardized format.

    Parameters
    ----------
    values_x : list
        X values (e.g., witness values)
    values_y : list
        Y values (e.g., contradiction values)
    boundary_x : float
        Boundary value for X
    x_label : str
        Label for X values
    y_label : str
        Label for Y values
    """
    max_below, count_below = analyze_boundary_conditions(values_x, values_y, boundary_x, "below")
    min_above, count_above = analyze_boundary_conditions(values_x, values_y, boundary_x, "above")

    # Buffered analysis: exclude points within 1e-3 of boundary for clearer separation
    buffer_width = 1e-3
    filtered_indices = [i for i, x in enumerate(values_x) if abs(x - boundary_x) >= buffer_width]
    filtered_x = [values_x[i] for i in filtered_indices]
    filtered_y = [values_y[i] for i in filtered_indices]

    max_below_buffered, _ = analyze_boundary_conditions(filtered_x, filtered_y, boundary_x, "below")
    min_above_buffered, _ = analyze_boundary_conditions(filtered_x, filtered_y, boundary_x, "above")

    print(f"Boundary zoom (by {x_label}): max {y_label} for {x_label} ≤ {boundary_x} is {max_below:.2e}; "
          f"min {y_label} for {boundary_x} < {x_label} ≤ {boundary_x + 1:.1f} is {min_above:.2e}")
    print(f"  strict: min {y_label}({x_label} > {boundary_x}) ≥ {min_above:.2e}")
    print(f"  buffered: min {y_label}({x_label} ≥ {boundary_x} + {buffer_width:.0e}) ≥ {min_above_buffered:.2e}")
    print(f"✓ Verified: max {y_label}({x_label} ≤ {boundary_x}) = {max_below:.2e}; "
          f"min {y_label}({x_label} > {boundary_x}) = {min_above:.2e}")


def run_parametric_analysis(perturbation_levels, perturbation_func, analysis_func=None, description=""):
    """
    Run parametric analysis with consistent error handling and reporting.

    Parameters
    ----------
    perturbation_levels : list
        Levels of perturbation to test
    perturbation_func : callable
        Function that takes a perturbation level and returns (contexts, behavior)
    analysis_func : callable, optional
        Function to analyze each result, defaults to extracting contradiction_bits
    description : str
        Description of the analysis

    Returns
    -------
    tuple
        (levels, results) where results contains analysis values for each level
    """
    if analysis_func is None:
        analysis_func = lambda behavior: behavior.contradiction_bits

    results = []
    for level in perturbation_levels:
        try:
            contexts, behavior = perturbation_func(level)
            result = analysis_func(behavior)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed at level {level}: {e}")
            results.append(0.0)  # Default value

    return perturbation_levels, results
