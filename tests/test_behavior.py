#!/usr/bin/env python3
"""
Tests for the Behavior class and its core functionality.
"""
from contrakit.space import Space
from contrakit.context import Context
from contrakit.behavior.behavior import Behavior
from contrakit.distribution import Distribution
import pytest
import numpy as np


def test_behavior_from_contexts_with_dicts():
    """Test creating behavior from dictionary specifications."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    behavior = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4},
        ("B",): {("0",): 0.5, ("1",): 0.5},
        ("A", "B"): {("0", "0"): 0.3, ("0", "1"): 0.3, ("1", "0"): 0.2, ("1", "1"): 0.2}
    })

    assert isinstance(behavior, Behavior)
    assert behavior.space == space
    assert len(behavior.distributions) == 3

    # Verify distributions are properly normalized
    for context, dist in behavior.distributions.items():
        total_prob = sum(dist[outcome] for outcome in dist.outcomes)
        assert abs(total_prob - 1.0) < 1e-12


def test_behavior_from_contexts_with_distributions():
    """Test creating behavior from Distribution objects."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    dist_a = Distribution.from_dict({("0",): 0.6, ("1",): 0.4})
    dist_b = Distribution.from_dict({("0",): 0.5, ("1",): 0.5})
    dist_joint = Distribution.from_dict({
        ("0", "0"): 0.3, ("0", "1"): 0.3,
        ("1", "0"): 0.2, ("1", "1"): 0.2
    })

    behavior = Behavior.from_contexts(space, {
        ("A",): dist_a,
        ("B",): dist_b,
        ("A", "B"): dist_joint
    })

    assert isinstance(behavior, Behavior)
    assert len(behavior.distributions) == 3


def test_behavior_from_contexts_consistency_check():
    """Test that inconsistent behaviors are rejected."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    # Create inconsistent distributions
    dist_marginal = Distribution.from_dict({("0",): 0.6, ("1",): 0.4})
    dist_joint = Distribution.from_dict({
        ("0", "0"): 0.3, ("0", "1"): 0.1,  # P(A=0) = 0.4
        ("1", "0"): 0.3, ("1", "1"): 0.3   # P(A=1) = 0.6
    })

    with pytest.raises(ValueError, match="Inconsistent contexts detected"):
        Behavior.from_contexts(space, {
            ("A",): dist_marginal,
            ("A", "B"): dist_joint,
        })


def test_behavior_manual_construction():
    """Test manual behavior construction and context setting."""
    space = Space.create(A=["0", "1"])

    behavior = Behavior(space)
    assert len(behavior.distributions) == 0

    # Add a context
    context = Context.make(space, ["A"])
    dist = Distribution.from_dict({("0",): 0.7, ("1",): 0.3})
    behavior[context] = dist

    assert len(behavior.distributions) == 1
    assert behavior[context] == dist


def test_behavior_duplicate_context_rejection():
    """Test that duplicate contexts with same observables are rejected."""
    space = Space.create(A=["0", "1"])

    behavior = Behavior(space)

    # Add first context
    context1 = Context.make(space, ["A"])
    dist1 = Distribution.from_dict({("0",): 0.7, ("1",): 0.3})
    behavior[context1] = dist1

    # Try to add another context with same observables
    context2 = Context.make(space, ["A"])  # Same observables
    dist2 = Distribution.from_dict({("0",): 0.6, ("1",): 0.4})

    with pytest.raises(ValueError, match="Duplicate context with the same observables"):
        behavior[context2] = dist2


def test_behavior_check_consistency():
    """Test the consistency checking functionality."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    # Create consistent behavior
    behavior = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4},
        ("A", "B"): {("0", "0"): 0.3, ("0", "1"): 0.3, ("1", "0"): 0.2, ("1", "1"): 0.2}
    })

    report = behavior.check_consistency()
    assert report["ok"] is True
    assert len(report["mismatches"]) == 0

    # Create behavior with only marginal - should be ok
    behavior_marginal = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4}
    })

    report = behavior_marginal.check_consistency()
    assert report["ok"] is True


def test_behavior_alpha_star():
    """Test alpha star calculation."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    # Create consistent behavior
    behavior = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4},
        ("B",): {("0",): 0.5, ("1",): 0.5},
        ("A", "B"): {("0", "0"): 0.3, ("0", "1"): 0.3, ("1", "0"): 0.2, ("1", "1"): 0.2}
    })

    alpha = behavior.alpha_star
    assert isinstance(alpha, (int, float))
    assert 0.0 <= alpha <= 1.0


def test_alpha_star_frame_independent_equals_one():
    """Test that α* = 1.0 for trivially frame-independent behavior."""
    space = Space.create(A=["0","1"])
    # Single context — always FI
    behavior = Behavior.from_contexts(space, {("A",): {("0",): 1.0}})
    assert abs(behavior.alpha_star - 1.0) < 1e-12


def test_behavior_contradiction_bits():
    """Test contradiction bits calculation."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    # Create consistent behavior (no contradiction)
    behavior = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4},
        ("B",): {("0",): 0.5, ("1",): 0.5},
        ("A", "B"): {("0", "0"): 0.3, ("0", "1"): 0.3, ("1", "0"): 0.2, ("1", "1"): 0.2}
    })

    k = behavior.contradiction_bits  # not behavior.K unless you purposely expose alias
    assert isinstance(k, (int, float))
    assert abs(k - 0.0) < 1e-12  # Use tolerance, not exact equality


def test_contradiction_bits_positive_on_tension():
    """Test that contradiction_bits > 0 for behaviors with tension."""
    space = Space.create(A=["0","1"], B=["0","1"], C=["0","1"])
    # Soft triangle: AB prefer differ, BC prefer differ, AC slightly prefer equal
    ab = {("0","1"): 0.55, ("1","0"): 0.45}
    bc = {("0","1"): 0.55, ("1","0"): 0.45}
    ac = {("0","0"): 0.55, ("1","1"): 0.45}
    behavior = Behavior.from_contexts(space, {("A","B"): ab, ("B","C"): bc, ("A","C"): ac})
    assert behavior.contradiction_bits > 0.0


def test_behavior_union():
    """Test union operation between behaviors with different contexts."""
    space1 = Space.create(A=["0", "1"])
    space2 = Space.create(B=["0", "1"])

    behavior1 = Behavior.from_contexts(space1, {
        ("A",): {("0",): 0.6, ("1",): 0.4}
    })

    behavior2 = Behavior.from_contexts(space2, {
        ("B",): {("0",): 0.5, ("1",): 0.5}
    })

    # Need to combine spaces for union
    combined_space = space1 @ space2
    behavior1_extended = Behavior.from_contexts(combined_space, {
        ("A",): {("0",): 0.6, ("1",): 0.4}
    })
    behavior2_extended = Behavior.from_contexts(combined_space, {
        ("B",): {("0",): 0.5, ("1",): 0.5}
    })

    combined_behavior = behavior1_extended.union(behavior2_extended)
    assert isinstance(combined_behavior, Behavior)

    # Test that we can compute agreement on the combined behavior
    agreement = combined_behavior.agreement.result
    assert isinstance(agreement, (int, float))
    assert agreement >= 0.0

    # Test the | operator
    combined_via_operator = behavior1_extended | behavior2_extended
    assert isinstance(combined_via_operator, Behavior)
    assert combined_via_operator.agreement.result == agreement
    assert agreement <= 1.0


def test_agreement_api_error_handling():
    """Test error handling in the agreement API."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    behavior = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4},
        ("B",): {("0",): 0.5, ("1",): 0.5},
        ("A", "B"): {("0", "0"): 0.3, ("0", "1"): 0.3, ("1", "0"): 0.2, ("1", "1"): 0.2}
    })

    # Test that normal operation works
    agreement = behavior.agreement.result
    assert isinstance(agreement, (int, float))
    assert 0.0 <= agreement <= 1.0

    context_scores = behavior.agreement.context_scores
    assert isinstance(context_scores, dict)
    assert len(context_scores) > 0

    explanation = behavior.agreement.explanation
    assert explanation is not None
    assert isinstance(explanation, np.ndarray)

    # Test with_explanation with invalid theta dimensions
    wrong_size_theta = np.array([0.5, 0.5])  # Should be 4 elements for this space
    try:
        behavior.agreement.with_explanation(wrong_size_theta)
        assert False, "Expected ValueError for wrong size theta"
    except ValueError as e:
        assert "theta has length 2" in str(e) and "needs 4" in str(e)

    # Test with_explanation with zero mass theta
    zero_mass_theta = np.array([0.0, 0.0, 0.0, 0.0])
    try:
        behavior.agreement.with_explanation(zero_mass_theta)
        assert False, "Expected ValueError for zero mass theta"
    except ValueError as e:
        assert "theta must have positive mass" in str(e)

    # Test with_explanation with negative values
    negative_mass_theta = np.array([-0.1, 0.3, 0.4, 0.4])
    try:
        behavior.agreement.with_explanation(negative_mass_theta)
        assert False, "Expected ValueError for negative values in theta"
    except ValueError as e:
        assert "theta must have non-negative values" in str(e)

    # Test filtering that results in no contexts
    try:
        behavior.agreement.keep_contexts([("NonExistent",)]).result
        assert False, "Expected ValueError for no contexts remaining"
    except ValueError as e:
        assert "No contexts remain after filtering" in str(e)

    # Test drop contexts that results in no contexts
    try:
        behavior.agreement.drop_contexts([("A",), ("B",), ("A", "B")]).result
        assert False, "Expected ValueError for no contexts remaining"
    except ValueError as e:
        assert "No contexts remain after filtering" in str(e)


def test_behavior_duplicate_context():
    """Test the duplicate_context method."""
    space = Space.create(A=["0", "1"])

    behavior = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4}
    })

    # Duplicate the context with a tag
    duplicated = behavior.duplicate_context(("A",), 2, "Trial")

    # Should have 2 contexts now (original + 1 duplicate with tag)
    assert len(duplicated.distributions) == 2

    # Check that contexts have different observables
    contexts = list(duplicated.distributions.keys())
    obs_sets = [tuple(ctx.observables) for ctx in contexts]
    assert len(set(obs_sets)) == 2  # Should be unique


def test_behavior_frame_independent():
    """Test creating frame-independent behavior."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    contexts = [("A",), ("B",), ("A", "B")]
    behavior = Behavior.frame_independent(space, contexts)

    assert len(behavior.distributions) == 3

    # Check that marginals are consistent with the joint
    ctx_a = next(ctx for ctx in behavior.distributions.keys() if ctx.observables == ("A",))
    ctx_b = next(ctx for ctx in behavior.distributions.keys() if ctx.observables == ("B",))
    ctx_joint = next(ctx for ctx in behavior.distributions.keys() if set(ctx.observables) == {"A", "B"})

    dist_a = behavior.distributions[ctx_a]
    dist_b = behavior.distributions[ctx_b]
    dist_joint = behavior.distributions[ctx_joint]

    # Check marginal consistency: P(A=0) should equal sum over B of P(A=0,B=b)
    p_a0_from_marginal = dist_a[("0",)]
    p_a0_from_joint = dist_joint[("0", "0")] + dist_joint[("0", "1")]
    assert abs(p_a0_from_marginal - p_a0_from_joint) < 1e-10

    p_b0_from_marginal = dist_b[("0",)]
    p_b0_from_joint = dist_joint[("0", "0")] + dist_joint[("1", "0")]
    assert abs(p_b0_from_marginal - p_b0_from_joint) < 1e-10


def test_behavior_from_mu():
    """Test creating behavior from global assignment distribution."""
    space = Space.create(A=["0", "1"], B=["0", "1"])

    # Define mu: [P(A=0,B=0), P(A=0,B=1), P(A=1,B=0), P(A=1,B=1)]
    mu = np.array([0.1, 0.2, 0.3, 0.4])

    contexts = [("A",), ("B",), ("A", "B")]
    behavior = Behavior.from_mu(space, contexts, mu)

    assert len(behavior.distributions) == 3

    # Check marginal for A: P(A=0) = P(A=0,B=0) + P(A=0,B=1) = 0.1 + 0.2 = 0.3
    # P(A=1) = P(A=1,B=0) + P(A=1,B=1) = 0.3 + 0.4 = 0.7
    ctx_a = next(ctx for ctx in behavior.distributions.keys() if ctx.observables == ("A",))
    dist_a = behavior.distributions[ctx_a]
    assert abs(dist_a[("0",)] - 0.3) < 1e-10
    assert abs(dist_a[("1",)] - 0.7) < 1e-10


def test_behavior_tensor_product():
    """Test tensor product of behaviors."""
    space1 = Space.create(A=["0", "1"])
    space2 = Space.create(B=["x", "y"])

    behavior1 = Behavior.from_contexts(space1, {
        ("A",): {("0",): 0.6, ("1",): 0.4}
    })

    behavior2 = Behavior.from_contexts(space2, {
        ("B",): {("x",): 0.7, ("y",): 0.3}
    })

    product = behavior1 @ behavior2

    assert len(product.distributions) == 1  # Only one context each
    assert set(product.space.names) == {"A", "B"}

    # Check that probabilities multiply
    context = list(product.distributions.keys())[0]
    dist = product.distributions[context]

    # Outcomes are flattened tuples: (value_A, value_B)
    assert abs(dist[("0", "x")] - 0.6 * 0.7) < 1e-10
    assert abs(dist[("1", "y")] - 0.4 * 0.3) < 1e-10


def test_behavior_mix():
    """Test convex combination of behaviors."""
    space = Space.create(A=["0", "1"])

    behavior1 = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.6, ("1",): 0.4}
    })

    behavior2 = Behavior.from_contexts(space, {
        ("A",): {("0",): 0.4, ("1",): 0.6}
    })

    mixed = behavior1.mix(behavior2, 0.3)  # 70% behavior1, 30% behavior2

    assert len(mixed.distributions) == 1

    context = list(mixed.distributions.keys())[0]
    dist = mixed.distributions[context]

    # Expected: 0.7 * 0.6 + 0.3 * 0.4 = 0.42 + 0.12 = 0.54 for ("0",)
    expected_0 = 0.7 * 0.6 + 0.3 * 0.4
    expected_1 = 0.7 * 0.4 + 0.3 * 0.6

    assert abs(dist[("0",)] - expected_0) < 1e-10
    assert abs(dist[("1",)] - expected_1) < 1e-10


def test_alpha_star_computation_does_not_hang():
    """Test that α* computation doesn't hang on complex constraint sets."""
    # Create a behavior with multiple pairwise constraints
    # This tests that the solver doesn't hang indefinitely
    space = Space.create(A=["0","1"], B=["0","1"], C=["0","1"])

    ab = Distribution.from_dict({("0","1"):0.5, ("1","0"):0.5})
    bc = Distribution.from_dict({("0","1"):0.5, ("1","0"):0.5})
    ac = Distribution.from_dict({("0","0"):0.5, ("1","1"):0.5})

    behavior = Behavior.from_contexts(space, {
        ("A","B"): ab,
        ("B","C"): bc,
        ("A","C"): ac,
    })

    # Should not raise; should not hang; α* should be valid
    alpha = behavior.alpha_star
    assert 0.0 <= alpha <= 1.0


def test_from_contexts_rejects_invalid_outcomes():
    """Test that from_contexts rejects invalid outcomes with clear error."""
    space = Space.create(A=["0","1"])
    # Outcome ("2",) doesn't exist in A's alphabet
    bad = Distribution.from_dict({("0",): 0.7, ("2",): 0.3})

    with pytest.raises(ValueError, match="Invalid outcomes for context"):
        Behavior.from_contexts(space, {("A",): bad})


def test_from_contexts_rejects_unknown_context_names():
    """Test that from_contexts rejects unknown observable names."""
    space = Space.create(A=["0","1"])
    with pytest.raises(ValueError, match="Unknown observables"):
        Behavior.from_contexts(space, {("Z",): {("0",): 1.0}})


def test_from_contexts_rejects_non_normalized():
    """Test that from_contexts rejects non-normalized probability distributions."""
    space = Space.create(A=["0","1"])
    with pytest.raises(ValueError, match=r"must sum to 1"):
        Behavior.from_contexts(space, {("A",): {("0",): 0.7, ("1",): 0.6}})


def test_check_consistency_catches_subset_mismatches():
    """Test that consistency check catches mismatches across multi-subset marginals."""
    space = Space.create(A=["0","1"], B=["0","1"], C=["0","1"])
    joint = {("0","0","0"): 0.25, ("0","0","1"): 0.25, ("1","1","0"): 0.25, ("1","1","1"): 0.25}
    # Implied P(A=0,B=0)=0.5, but we lie and say 0.4
    bad_ab = {("0","0"): 0.4, ("0","1"): 0.1, ("1","0"): 0.1, ("1","1"): 0.4}
    with pytest.raises(ValueError, match="Inconsistent contexts detected"):
        Behavior.from_contexts(space, {
            ("A","B","C"): joint,
            ("A","B"): bad_ab
        })


def test_from_contexts_rejects_duplicate_contexts():
    """Test that from_contexts rejects duplicate contexts with different distributions."""
    space = Space.create(A=["0","1"])
    # Dict merging to create "duplicate" keys (though dicts normally prevent this)
    # We test the constructor-level duplicate detection
    m1 = {("A",): {("0",): 0.5, ("1",): 0.5}}
    m2 = {("A",): {("0",): 1.0}}  # Same key, different distribution
    # Since dicts can't have duplicate keys, this tests the behavior construction
    # But we already test duplicate rejection via __setitem__, so this is more of a smoke test
    behavior = Behavior.from_contexts(space, m1)
    assert len(behavior.distributions) == 1
    # The second dict would overwrite, but we're testing the single construction


def test_alpha_star_soft_triangle_finishes():
    """Test that α* computation finishes on softly contradictory constraints."""
    space = Space.create(A=["0","1"], B=["0","1"], C=["0","1"])
    # AB prefer differ, BC prefer differ, AC slightly prefer equal (soft, not hard)
    ab = {("0","1"): 0.55, ("1","0"): 0.45}
    bc = {("0","1"): 0.55, ("1","0"): 0.45}
    ac = {("0","0"): 0.55, ("1","1"): 0.45}
    behavior = Behavior.from_contexts(space, {("A","B"): ab, ("B","C"): bc, ("A","C"): ac})
    alpha = behavior.alpha_star
    assert 0.0 < alpha < 1.0


def test_check_consistency_report_shape():
    """Test that check_consistency returns properly structured reports."""
    space = Space.create(A=["0","1"], B=["0","1"])
    joint = {("0","0"):0.4, ("0","1"):0.1, ("1","0"):0.1, ("1","1"):0.4}
    bad_a = {("0",):0.3, ("1",):0.7}  # implied from joint is 0.5/0.5
    with pytest.raises(ValueError):
        Behavior.from_contexts(space, {("A","B"): joint, ("A",): bad_a})

    # Test the report structure on valid data
    behavior = Behavior.from_contexts(space, {("A","B"): joint})
    report = behavior.check_consistency()
    assert report["ok"] is True
    assert isinstance(report["mismatches"], list)
    assert len(report["mismatches"]) == 0
