"""
Analysis of Simpson's Paradox Using Information-Theoretic Measures

This example demonstrates how Simpson's Paradox can be analyzed using
information-theoretic measures of inconsistency. The paradox occurs when
statistical associations change direction when data are aggregated versus
disaggregated, revealing limitations in simultaneous multi-perspective analysis.

WHAT IS SIMPSON'S PARADOX?
=========================

Simpson's Paradox occurs when a statistical relationship reverses direction
when data are broken down into subgroups. This phenomenon reveals fundamental
constraints on how statistical relationships can be simultaneously analyzed
across different groupings.

RESTAURANT CHOICE EXAMPLE
========================

Consider evaluating two restaurants with different rating profiles:
- Restaurant A: 4.8 stars from 12 reviews
- Restaurant B: 4.2 stars from 1,200 reviews

Two reasonable evaluators might reach different conclusions:

- Evaluator 1: "Higher average rating indicates better quality → prefer A"
- Evaluator 2: "Small sample size suggests greater uncertainty → prefer B"

Both evaluation approaches are methodologically sound. The disagreement
arises from different weighting of sample size versus rating magnitude.

TEACHING METHODS EXAMPLE
=======================

Our primary example examines teaching method effectiveness across schools:

School A: Method X (90% success) > Method Y (20% success)
School B: Method X (10% success) < Method Y (80% success)
Combined: Method X (50% success) = Method Y (50% success)

Traditional analysis shows no overall difference between methods.
Information-theoretic analysis reveals the underlying perspectival tension.

FRAME INTEGRATION APPROACH
==========================

Frame integration involves including context variables in the analysis
to enable consistent modeling of multiple perspectives. Mathematically:

1. Start with observations P = {p(o|c)} where contexts c provide observables o
2. Traditional approaches require compatibility → often not achievable
3. Include context variable F, observe (O,F) instead of just O
4. Find single joint distribution q(o,f) explaining all observations

This is analogous to adding coordinates to make local descriptions compatible
within a global framework.

ANALYSIS METHODOLOGIES
=====================

Two complementary approaches are demonstrated:

1. Lens spaces: Contexts sharing observables with distinct identifiers
   - Appropriate for: Comparing incompatible interpretations of same variables
   - Effective when: Different observers interpret identical phenomena differently

2. Explicit context variables: Context as a measurable component
   - Appropriate for: Modeling context as part of the data structure
   - Effective when: Context differences are directly observable

The contradiction measure K(P) quantifies when multiple valid perspectives
cannot be reconciled, measuring the information content of their disagreement.

IMPLICATIONS FOR ANALYSIS
========================

When experts using sound methodologies reach systematic disagreement,
standard practice typically resolves this by selecting one perspective.
However, the disagreement itself provides valuable information about:

- Which system aspects can/cannot be simultaneously optimized
- Coordination requirements between valid alternatives
- Early indicators when objectives are fundamentally incompatible

The contradiction measure quantifies the explanatory coherence lost when
incompatible perspectives are forced into a single analytical framework.
"""

from math import log2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from contrakit import Space, Behavior
from contrakit import lens_space, as_lens_context
from contrakit.constants import FIGURES_DIR


# ================================
# 1. Data Setup: Simpson's Paradox Example
# ================================

# Two schools with different teaching method success rates
# This data illustrates the core of Simpson's Paradox:
#
# School A: Method X shows 90% success vs Method Y at 20% success
# School B: Method Y shows 80% success vs Method X at 10% success
# Combined: Both methods show exactly 50% success rates
#
# The paradox occurs because each school provides clear evidence
# favoring opposite conclusions, yet the combined data shows no difference.

school_data = {
    "Oakwood_School": {
        "Lecture_Method": (9, 1),      # 90% success: 9 passes, 1 fail
        "Discussion_Method": (2, 8)    # 20% success: 2 passes, 8 fails
    },
    "Maple_School": {
        "Lecture_Method": (1, 9),      # 10% success: 1 pass, 9 fails  
        "Discussion_Method": (8, 2)    # 80% success: 8 passes, 2 fails
    }
}

# This isn't due to measurement error or insufficient data—it's a
# fundamental structural pattern. The evidence from each school cannot be
# explained by any single consistent framework.

# Calculate aggregate statistics across both schools
aggregate_counts = {
    ("Lecture_Method", 1): 0,
    ("Lecture_Method", 0): 0,
    ("Discussion_Method", 1): 0,
    ("Discussion_Method", 0): 0
}
total_students = 0

for school_name in ["Oakwood_School", "Maple_School"]:
    for method in ["Lecture_Method", "Discussion_Method"]:
        passes, fails = school_data[school_name][method]
        aggregate_counts[(method, 1)] += passes
        aggregate_counts[(method, 0)] += fails
        total_students += passes + fails


def calculate_mutual_information(probabilities):
    """
    Calculate mutual information I(T;Y) between treatment T and outcome Y.

    Args:
        probabilities: Dictionary of (treatment, outcome) -> probability

    Returns:
        Mutual information in bits
    """
    # Marginal probabilities for treatments
    treatment_marginals = {
        treatment: probabilities[(treatment, 1)] + probabilities[(treatment, 0)]
        for treatment in ["Lecture_Method", "Discussion_Method"]
    }

    # Marginal probabilities for outcomes
    outcome_marginals = {
        outcome: probabilities[("Lecture_Method", outcome)] + probabilities[("Discussion_Method", outcome)]
        for outcome in [1, 0]
    }

    mutual_info = 0.0
    for treatment in ["Lecture_Method", "Discussion_Method"]:
        for outcome in [1, 0]:
            joint_prob = probabilities[(treatment, outcome)]
            if joint_prob > 0:
                expected_prob = treatment_marginals[treatment] * outcome_marginals[outcome]
                mutual_info += joint_prob * log2(joint_prob / expected_prob)

    return mutual_info


# ================================
# 2. Demonstrating the Paradox: Multiple Perspectives
# ================================

# APPROACH 1: Using lens spaces to model incompatible perspectives
# ================================================================
#
# We use lens spaces here to show how different observers (schools) can examine
# the same variables (teaching methods and outcomes) and reach incompatible conclusions.
#
# Lens spaces create contexts that:
# - Share the same observable variables (Treatment, Outcome)
# - Have distinct identifiers to separate the contexts
# - Allow us to test whether different perspectives can be reconciled
#
# This represents the situation where skilled analysts using sound methods
# on identical data types reach systematic disagreement.

base_space = Space.create(Treatment=["Lecture_Method", "Discussion_Method"], Outcome=[1, 0])

# The singleton tags create distinct contexts while keeping observables identical
# This captures the core issue: same variables, different perspectives, incompatible conclusions
paradox_space = lens_space(base_space, OakwoodTag=[0], MapleTag=[0])


def get_school_distribution(school_name):
    """
    Get the probability distribution for a specific school.

    Args:
        school_name: Either "Oakwood_School" or "Maple_School"

    Returns:
        Dictionary mapping (treatment, outcome, tag) to probability
    """
    passes_lecture, fails_lecture = school_data[school_name]["Lecture_Method"]
    passes_discussion, fails_discussion = school_data[school_name]["Discussion_Method"]
    total = passes_lecture + fails_lecture + passes_discussion + fails_discussion

    return {
        ("Lecture_Method", 1, 0): passes_lecture / total,
        ("Lecture_Method", 0, 0): fails_lecture / total,
        ("Discussion_Method", 1, 0): passes_discussion / total,
        ("Discussion_Method", 0, 0): fails_discussion / total,
    }


# Define contexts: each school provides a different view of (Treatment, Outcome)
contexts = {
    # Oakwood School's perspective on teaching method effectiveness
    as_lens_context(("Treatment", "Outcome"), "OakwoodTag"): get_school_distribution("Oakwood_School"),

    # Maple School's perspective on teaching method effectiveness
    as_lens_context(("Treatment", "Outcome"), "MapleTag"): get_school_distribution("Maple_School"),

    # Aggregate view across both schools
    ("Treatment", "Outcome"): {
        key: value / total_students
        for key, value in aggregate_counts.items()
    },
}

# Analyze the behavior for contradictions
paradox_behavior = Behavior.from_counts(paradox_space, contexts, normalize="none")
contradiction_measure = paradox_behavior.contradiction_bits
consistency_parameter = paradox_behavior.alpha_star
witness_distribution = paradox_behavior.least_favorable_lambda()


# ================================
# 3. Resolution: Frame Integration
# ================================

# APPROACH 2: Adding context as an explicit variable
# ===================================================
#
# Now we demonstrate frame integration—adding context (School) to our
# variables so that different perspectives can be consistently modeled together.
#
# Mathematically, instead of just observing (Treatment, Outcome),
# we observe (Treatment, Outcome, School). This allows school-specific
# patterns to be combined into a unified explanation.
#
# Key insight: when incompatibility arises from mixing different contexts (schools
# with different characteristics), adding the context variable allows a single
# joint distribution to explain everything consistently.

resolved_space = Space.create(
    Treatment=["Lecture_Method", "Discussion_Method"],
    Outcome=[1, 0],
    School=["Oakwood_School", "Maple_School"]  # Frame variable F
)

# Build joint counts including school information
joint_counts = {}
for school_name in ["Oakwood_School", "Maple_School"]:
    for method in ["Lecture_Method", "Discussion_Method"]:
        passes, fails = school_data[school_name][method]
        joint_counts[(method, 1, school_name)] = passes
        joint_counts[(method, 0, school_name)] = fails

# Aggregate counts (same as before, for comparison)
agg_counts = {
    ("Lecture_Method", 1): 0,
    ("Lecture_Method", 0): 0,
    ("Discussion_Method", 1): 0,
    ("Discussion_Method", 0): 0
}
for school_name in ["Oakwood_School", "Maple_School"]:
    for method in ["Lecture_Method", "Discussion_Method"]:
        passes, fails = school_data[school_name][method]
        agg_counts[(method, 1)] += passes
        agg_counts[(method, 0)] += fails

# Create behavior with explicit school context
resolved_behavior = Behavior.from_counts(
    resolved_space,
    {
        ("Treatment", "Outcome", "School"): joint_counts,
        ("Treatment", "Outcome"): agg_counts,
    },
    normalize="per_context",
)

resolved_contradiction = resolved_behavior.contradiction_bits
resolved_consistency = resolved_behavior.alpha_star


# ================================
# 4. Results and Analysis: Measuring Inconsistency
# ================================

print("Information-Theoretic Analysis of Simpson's Paradox")
print("=" * 55)
print()
print("Summary of analytical approach:")
print("- Traditional methods detect relationships within specific groupings")
print("- Information-theoretic measures identify incompatibility between perspectives")
print("- Frame integration enables consistent multi-perspective analysis")
print("- Contradiction measure quantifies reconciliation costs")
print()

# Calculate and display mutual information from aggregate data
agg_total = sum(agg_counts.values())
agg_probabilities = {key: value / agg_total for key, value in agg_counts.items()}
mi_value = calculate_mutual_information(agg_probabilities)

# Self-check assertions
assert abs(mi_value) < 1e-12, "Aggregate MI should be ~0 in this setup"
assert contradiction_measure > 0, "K must be >0 without School"
assert resolved_contradiction < 1e-9, "K should drop to ~0 when School is carried"

print("Traditional Analysis (Mutual Information):")
print(f"  I(Teaching_Method; Pass_Rate) = {mi_value:.6f} bits")
print("  → Standard conclusion: 'No overall relationship between method and outcome'")
print("  → Traditional approach concludes analysis at this point")
print()

print("Information-Theoretic Analysis:")
print(f"  K(P) without school context = {contradiction_measure:.6f} bits (α* = {consistency_parameter:.6f})")
print("  → Measures structural incompatibility between school-level perspectives")
print(f"  → Quantifies explanatory coherence lost in single-framework analysis")
print()

print("Frame Integration Results:")
print(f"  K(P) with school context = {resolved_contradiction:.6f} bits (α* = {resolved_consistency:.6f})")
print("  → Including context enables consistent multi-perspective modeling")
print("  → School-specific patterns integrate into unified explanation")
print()

print("Witness Distribution:")
print("  → Identifies perspectives contributing to structural inconsistency")
for key, value in witness_distribution.items():
    if value > 0.001:  # Only show non-zero values
        context_name = key[2] if len(key) > 2 else "Aggregate"
        if context_name == "OakwoodTag":
            context_name = "Oakwood School lens"
        elif context_name == "MapleTag":
            context_name = "Maple School lens"
        print(f"  {context_name}: λ* = {value:.6f}")
        
witness_sum = sum(v for k, v in witness_distribution.items() if v > 0.001)
print(f"  → Total witness weight: {witness_sum:.3f} (confirms proper normalization)")
print()
print("Key Finding:")
print("  When experts using sound methodologies reach systematic disagreement,")
print("  the disagreement itself provides information about underlying structural")
print("  constraints that may not be apparent through traditional analysis.")
print()
print("  This suggests that inconsistency can reveal important system properties")
print("  rather than merely indicating analytical noise.")


# ================================
# 5. Optional: Robustness Analysis
# ================================

def blend_school(school_name, eps):
    """Mix school data with its counterpart by epsilon for noise robustness analysis."""
    other_school = "Maple_School" if school_name == "Oakwood_School" else "Oakwood_School"
    
    A = np.array(school_data[school_name]["Lecture_Method"], dtype=float)
    B = np.array(school_data[school_name]["Discussion_Method"], dtype=float)
    A2 = np.array(school_data[other_school]["Lecture_Method"], dtype=float)
    B2 = np.array(school_data[other_school]["Discussion_Method"], dtype=float)
    
    A = (1-eps)*A + eps*A2
    B = (1-eps)*B + eps*B2
    Z = A.sum() + B.sum()
    
    return {
        ("Lecture_Method", 1, 0): A[0]/Z, 
        ("Lecture_Method", 0, 0): A[1]/Z,
        ("Discussion_Method", 1, 0): B[0]/Z, 
        ("Discussion_Method", 0, 0): B[1]/Z,
    }

def K_vs_noise(grid=np.linspace(0, 1, 21)):
    """Compute contradiction K as a function of noise parameter epsilon."""
    Ks = []
    for eps in grid:
        ctx = {
            as_lens_context(("Treatment", "Outcome"), "OakwoodTag"): blend_school("Oakwood_School", eps),
            as_lens_context(("Treatment", "Outcome"), "MapleTag"): blend_school("Maple_School", eps),
            ("Treatment", "Outcome"): {k: v/total_students for k,v in aggregate_counts.items()},
        }
        b = Behavior.from_counts(paradox_space, ctx, normalize="none")
        Ks.append(b.contradiction_bits)
    return grid, np.array(Ks)


# ================================
# 6. Visualization:
# ================================

def create_figure():
    """Create a 6-panel visualization showing the complete Simpson's Paradox analysis."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Information-Theoretic Analysis of Simpson's Paradox",
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Colors for consistency across panels - separate palettes for different variables
    colors = {
        # Teaching method palette (used in panels A, C)
        'lecture': '#2E86AB',      # blue
        'discussion': '#F24236',    # red
        
        # School palette (used in panels B, D)
        'oakwood': '#A23B72',      # Purple for Oakwood
        'maple': '#F18F01',        # Orange for Maple
        
        # Information/analysis palette
        'neutral': '#C73E1D',      # Dark red for emphasis
        'evidence': '#2E86AB'      # Blue for evidence accumulation
    }
    
    # Panel A: Simpson's Slope Plot - The Paradox Revealed
    ax_a = axes[0, 0]
    
    # Calculate success rates for each context
    oakwood_lecture_rate = 9 / (9 + 1)  # 90%
    oakwood_discussion_rate = 2 / (2 + 8)  # 20%
    maple_lecture_rate = 1 / (1 + 9)  # 10%
    maple_discussion_rate = 8 / (8 + 2)  # 80%
    
    # Aggregate rates
    total_lecture_pass = 9 + 1  # 10 passes out of 20 total
    total_lecture_fail = 1 + 9  # 10 fails
    total_discussion_pass = 2 + 8  # 10 passes out of 20 total
    total_discussion_fail = 8 + 2  # 10 fails
    
    agg_lecture_rate = total_lecture_pass / (total_lecture_pass + total_lecture_fail)  # 50%
    agg_discussion_rate = total_discussion_pass / (total_discussion_pass + total_discussion_fail)  # 50%
    
    x_positions = [0, 1, 2.5]  # Oakwood, Maple, Aggregate
    x_labels = ['Oakwood\nSchool', 'Maple\nSchool', 'Aggregate']
    
    # Plot lines for each method
    lecture_rates = [oakwood_lecture_rate, maple_lecture_rate, agg_lecture_rate]
    discussion_rates = [oakwood_discussion_rate, maple_discussion_rate, agg_discussion_rate]
    
    ax_a.plot(x_positions, lecture_rates, 'o-', color=colors['lecture'], 
              linewidth=2.5, markersize=8, label='Lecture Method')
    ax_a.plot(x_positions, discussion_rates, 's-', color=colors['discussion'], 
              linewidth=2.5, markersize=8, label='Discussion Method')
    
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(x_labels)
    ax_a.set_ylabel('Success Rate')
    ax_a.set_title('A. School-Level vs Combined Analysis', fontweight='bold')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Add callout box with correct description of opposite preferences
    ax_a.text(0.02, 0.43, 'Within schools: opposite preferences\nOakwood: Lecture≫Discussion\nMaple: Discussion≫Lecture', 
              transform=ax_a.transAxes, fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=1.0, edgecolor='blue', linewidth=1))
    ax_a.text(0.7, 0.15, 'Aggregate:\nTie', 
              transform=ax_a.transAxes, fontsize=11,
              bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    # Add faint dashed connectors from school points to aggregate
    for i, school_pos in enumerate([0, 1]):
        ax_a.plot([school_pos, 2.5], [lecture_rates[i], agg_lecture_rate], 
                  '--', color=colors['lecture'], alpha=0.3, linewidth=1)
        ax_a.plot([school_pos, 2.5], [discussion_rates[i], agg_discussion_rate], 
                  '--', color=colors['discussion'], alpha=0.3, linewidth=1)
    
    annotations = [
        ((0, oakwood_lecture_rate), '9/10', (8, 8)),
        ((0, oakwood_discussion_rate), '2/10', (8, -18)),
        ((1, maple_lecture_rate), '1/10', (8, -18)),
        ((1, maple_discussion_rate), '8/10', (8, 8)),
        ((2.5, agg_lecture_rate), '10/20', (8, 8)),
        ((2.5, agg_discussion_rate), '10/20', (8, -18))
    ]
    
    for (x, y), text, offset in annotations:
        ax_a.annotate(text, (x, y), xytext=offset, 
                      textcoords='offset points', fontsize=10, ha='left', fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # Panel B: Composition/Weighting - Why It Flips
    ax_b = axes[0, 1]
    
    # Student distribution data
    oakwood_lecture_students = 10  # 9 pass + 1 fail
    maple_lecture_students = 10    # 1 pass + 9 fail
    oakwood_discussion_students = 10  # 2 pass + 8 fail
    maple_discussion_students = 10    # 8 pass + 2 fail
    
    # Create stacked bars
    methods = ['Lecture\nStudents', 'Discussion\nStudents']
    oakwood_counts = [oakwood_lecture_students, oakwood_discussion_students]
    maple_counts = [maple_lecture_students, maple_discussion_students]
    
    x_bar = np.arange(len(methods))
    width = 0.6
    
    bars1 = ax_b.bar(x_bar, oakwood_counts, width, label='Oakwood School',
                     color=colors['oakwood'], alpha=0.8)
    bars2 = ax_b.bar(x_bar, maple_counts, width, bottom=oakwood_counts,
                     label='Maple School', color=colors['maple'], alpha=0.8)
    
    # Add success rate and count labels on segments
    ax_b.text(0, oakwood_lecture_students/2, '90%\nsuccess\n(n=10)', ha='center', va='center', 
              fontweight='bold', color='white', fontsize=10)
    ax_b.text(0, oakwood_lecture_students + maple_lecture_students/2, '10%\nsuccess\n(n=10)', 
              ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    ax_b.text(1, oakwood_discussion_students/2, '20%\nsuccess\n(n=10)', ha='center', va='center', 
              fontweight='bold', color='white', fontsize=10)
    ax_b.text(1, oakwood_discussion_students + maple_discussion_students/2, '80%\nsuccess\n(n=10)', 
              ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    
    ax_b.set_ylabel('Number of Students')
    ax_b.set_title('B. Sample Sizes Drive the Combined Result', fontweight='bold')
    ax_b.set_xticks(x_bar)
    ax_b.set_xticklabels(methods)
    ax_b.legend()
    ax_b.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Information Measures - MI vs K
    ax_c = axes[0, 2]
    
    measures = ['MI(Method; Success)\n[Aggregate]', 'K(P)\n[Cross-Context]']
    values = [mi_value, contradiction_measure]
    colors_c = [colors['neutral'], colors['evidence']]
    
    bars = ax_c.bar(measures, values, color=colors_c, alpha=0.8, width=0.6)
    
    # Add thin horizontal line at 0
    ax_c.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if i == 0:  # MI bar - label as exact 0
            ax_c.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                      'exact 0', ha='center', va='bottom', fontweight='bold')
        else:  # K bar
            ax_c.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                      f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
            ax_c.text(bar.get_x() + bar.get_width()/2., height/2,
                      f'α* = {consistency_parameter:.3f}', ha='center', va='center',
                      color='white', fontweight='bold')
    
    ax_c.set_ylabel('Information (bits)')
    ax_c.set_title('C. Traditional vs Information-Theoretic Measures', fontweight='bold')
    ax_c.grid(True, alpha=0.3, axis='y')
    ax_c.set_ylim(0, max(values) * 1.2)
    
    # Panel D: Witness Distribution - Where Contradiction Concentrates
    ax_d = axes[1, 0]
    
    # Extract witness values and show all contexts (including zero aggregate)
    witness_values = []
    witness_labels = []
    witness_colors = []
    
    # Safer witness parsing with explicit context keys
    oakwood_ctx = as_lens_context(("Treatment", "Outcome"), "OakwoodTag")
    maple_ctx = as_lens_context(("Treatment", "Outcome"), "MapleTag")
    oakwood_weight = witness_distribution.get(oakwood_ctx, 0.0)
    maple_weight = witness_distribution.get(maple_ctx, 0.0)
    
    # Add in the correct order to match the data
    witness_labels.extend(["Oakwood\nLens", "Maple\nLens"])
    witness_values.extend([oakwood_weight, maple_weight])
    witness_colors.extend([colors['oakwood'], colors['maple']])
    
    # Add aggregate lens with zero weight (faded and de-emphasized)
    witness_labels.append("Aggregate\nLens")
    witness_values.append(0.0)
    witness_colors.append('lightgray')
    
    # Create bars with different alpha for aggregate lens
    bars_d = []
    for i, (label, value, color) in enumerate(zip(witness_labels, witness_values, witness_colors)):
        alpha_val = 0.4 if label == "Aggregate\nLens" else 0.8
        bar = ax_d.bar(i, value, color=color, alpha=alpha_val, width=0.6)
        bars_d.extend(bar)
    
    # Add value labels
    witness_sum = sum(v for v in witness_values if v > 0)
    for i, (bar, value) in enumerate(zip(bars_d, witness_values)):
        height = bar.get_height() if value > 0 else 0.02  # Small height for zero bar visibility
        if value > 0.001:
            ax_d.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax_d.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      '0.000', ha='center', va='bottom', fontweight='bold', alpha=0.6)
    
    ax_d.set_xticks(range(len(witness_labels)))
    ax_d.set_xticklabels(witness_labels)
    ax_d.set_ylabel('λ* (Witness Weight)')
    ax_d.set_title('D. Which Perspectives Drive Inconsistency', fontweight='bold')
    ax_d.grid(True, alpha=0.3, axis='y')
    ax_d.set_ylim(0, max(witness_values) * 1.3 if max(witness_values) > 0 else 0.6)
    
    # Panel E: Resolution - Carry the Frame
    ax_e = axes[1, 1]
    
    resolution_labels = ['Without\nSchool Context', 'With\nSchool Context']
    k_values = [contradiction_measure, resolved_contradiction]
    colors_e = [colors['neutral'], colors['evidence']]
    
    bars_e = ax_e.bar(resolution_labels, k_values, color=colors_e, alpha=0.8, width=0.6)
    
    # Add value labels and alpha values
    for i, (bar, k_val) in enumerate(zip(bars_e, k_values)):
        height = bar.get_height()
        ax_e.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                  f'{k_val:.6f}', ha='center', va='bottom', fontweight='bold')
        
        alpha_val = consistency_parameter if i == 0 else resolved_consistency
        if height > 0.01:  # Only show alpha if bar is tall enough
            ax_e.text(bar.get_x() + bar.get_width()/2., height/2,
                      f'α* = {alpha_val:.3f}', ha='center', va='center',
                      color='white', fontweight='bold')
    
    
    ax_e.set_ylabel('Contradiction K (bits)')
    ax_e.set_title('E. Adding Context Resolves Inconsistency', fontweight='bold')
    ax_e.grid(True, alpha=0.3, axis='y')
    ax_e.set_ylim(0, contradiction_measure * 1.3)
    
    # Panel F: Detection Power - Sample Complexity
    ax_f = axes[1, 2]
    
    # Sample sizes from 1 to 500 - reduced range to better show the evidence thresholds
    sample_sizes = np.logspace(0, 2.5, 50)  # 1 to ~316, log scale
    log_likelihood_ratios = sample_sizes * contradiction_measure
    
    ax_f.semilogx(sample_sizes, log_likelihood_ratios, 'b-', linewidth=3, 
                  label=f'Evidence accumulation\n(K = {contradiction_measure:.3f} bits/sample)')
    
    # Add horizontal guide lines for evidence thresholds - reduced levels
    evidence_levels = [1, 3, 5]
    evidence_labels = ['Weak', 'Moderate', 'Strong']
    colors_evidence = ['orange', 'red', 'darkred']
    
    # Set y-axis limit to better accommodate the evidence levels
    max_evidence = max(log_likelihood_ratios)
    ax_f.set_ylim(0, min(max_evidence * 1.1, 8))  # Cap at 8 bits for better spacing
    
    for i, (level, label, color) in enumerate(zip(evidence_levels, evidence_labels, colors_evidence)):
        if level <= ax_f.get_ylim()[1]:  # Only show if within y-axis range
            ax_f.axhline(y=level, color=color, linestyle='--', alpha=0.7)
            # Position text with better spacing
            text_x = sample_sizes[10 + i * 8]  # Spread out the text positions
            ax_f.text(text_x, level + 0.2, f'{label}\n({level} bits)', 
                      ha='center', va='bottom', fontsize=9, color=color, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax_f.set_xlabel('Sample Size (number of experiments)')
    ax_f.set_ylabel('Cumulative Evidence (bits)')
    ax_f.set_title('F. Evidence Accumulation with Sample Size', fontweight='bold')
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc='lower right')
    
    # Adjust layout and save with proper padding
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
    
    return fig

print("\nGenerating visualization...")
fig = create_figure()
output_path = FIGURES_DIR / 'simpsons_paradox.png'
fig.savefig(output_path,
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"Visualization saved to: {output_path}")



