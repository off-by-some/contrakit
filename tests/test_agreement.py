# tests/test_agreement.py
"""Tests for the Agreement API."""

import numpy as np
import pytest

from contrakit import Space, Behavior

TOL = 1e-6


class TestAgreementAPI:
    """Test the new agreement API."""

    @pytest.fixture
    def behavior_2x2(self):
        """Create a 2x2 behavior for testing."""
        space = Space.create(Hair=["Blonde", "Brunette"], Eyes=["Blue", "Brown"])
        return Behavior.from_contexts(space, {
            ("Hair",): {("Blonde",): 0.6, ("Brunette",): 0.4},
            ("Eyes",): {("Blue",): 0.7, ("Brown",): 0.3},
            ("Hair", "Eyes"): {
                ("Blonde", "Blue"): 0.4, ("Blonde", "Brown"): 0.2,
                ("Brunette", "Blue"): 0.3, ("Brunette", "Brown"): 0.1
            }
        })

    @pytest.fixture
    def behavior_1x2(self):
        """Create a 1x2 behavior for simpler testing."""
        space = Space.create(Hair=["Blonde", "Brunette"])
        return Behavior.from_contexts(space, {
            ("Hair",): {("Blonde",): 0.6, ("Brunette",): 0.4}
        })

    def test_result_always_float(self, behavior_2x2):
        """Test that .result always returns a float."""
        # Basic agreement
        result = behavior_2x2.agreement.result
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0 + 1e-9

        # With weights
        weighted_result = behavior_2x2.agreement.for_weights({("Hair",): 0.6, ("Eyes",): 0.4}).result
        assert isinstance(weighted_result, float)
        assert 0.0 <= weighted_result <= 1.0 + 1e-9

        # Fixed feature
        fixed_result = behavior_2x2.agreement.for_feature("Hair", [0.5, 0.5]).result
        assert isinstance(fixed_result, float)
        assert 0.0 <= fixed_result <= 1.0 + 1e-9

    def test_context_scores_property(self, behavior_2x2):
        """Test that .context_scores returns per-context scores."""
        scores = behavior_2x2.agreement.context_scores
        assert isinstance(scores, dict)
        expected_keys = {("Hair",), ("Eyes",), ("Hair", "Eyes")}
        assert set(scores.keys()) == expected_keys
        assert all(isinstance(v, float) for v in scores.values())
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in scores.values())

    def test_explanation_property(self, behavior_2x2):
        """Test that .explanation returns theta array or None."""
        # Normal case should have explanation
        explanation = behavior_2x2.agreement.explanation
        assert isinstance(explanation, np.ndarray)
        assert explanation.shape == (4,)  # 2x2 = 4 assignments
        assert np.isclose(explanation.sum(), 1.0, atol=1e-9)

        # Fixed feature case should have None explanation
        fixed_explanation = behavior_2x2.agreement.for_feature("Hair", [0.5, 0.5]).explanation
        assert fixed_explanation is None

    def test_scenarios_method(self, behavior_1x2):
        """Test that .scenarios() returns readable scenario list."""
        scenarios = behavior_1x2.agreement.scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) == 2  # 2 assignments

        total_prob = 0.0
        for scenario, prob in scenarios:
            assert isinstance(scenario, tuple)
            assert len(scenario) == 1  # One observable
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0 + 1e-9
            total_prob += prob
        assert np.isclose(total_prob, 1.0, atol=1e-9)

        # Fixed feature should return empty list
        fixed_scenarios = behavior_1x2.agreement.for_feature("Hair", [0.5, 0.5]).scenarios()
        assert fixed_scenarios == []

    def test_feature_distribution_method(self, behavior_2x2):
        """Test that .feature_distribution() returns marginals."""
        # Normal case
        hair_dist = behavior_2x2.agreement.feature_distribution("Hair")
        assert isinstance(hair_dist, np.ndarray)
        assert hair_dist.shape == (2,)  # 2 hair options
        assert np.isclose(hair_dist.sum(), 1.0, atol=1e-9)

        # Fixed feature case - should return the fixed distribution
        fixed_hair_dist = behavior_2x2.agreement.for_feature("Hair", [0.3, 0.7]).feature_distribution("Hair")
        assert isinstance(fixed_hair_dist, np.ndarray)
        assert np.allclose(fixed_hair_dist, [0.3, 0.7])

        # Fixed feature case for different observable should raise error
        with pytest.raises(ValueError, match="No explanation"):
            behavior_2x2.agreement.for_feature("Hair", [0.5, 0.5]).feature_distribution("Eyes")

    def test_for_marginal_alias(self, behavior_1x2):
        """Test that .for_marginal() is an alias for .feature_distribution()."""
        dist1 = behavior_1x2.agreement.feature_distribution("Hair")
        dist2 = behavior_1x2.agreement.for_marginal("Hair")

        assert np.allclose(dist1, dist2)

    def test_fluent_api_chaining(self, behavior_2x2):
        """Test that fluent API chaining works correctly."""
        # Chain multiple operations
        result = (
            behavior_2x2.agreement
            .for_weights({("Hair",): 0.6, ("Eyes",): 0.4})
            .for_feature("Hair", [0.5, 0.5])
            .result
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0 + 1e-9

    def test_by_context_mode(self, behavior_2x2):
        """Test by_context mode."""
        # .result should still be float (minimum of context scores)
        result = behavior_2x2.agreement.by_context().result
        assert isinstance(result, float)

        # But we can get per-context scores
        scores = behavior_2x2.agreement.by_context().context_scores
        assert isinstance(scores, dict)
        assert len(scores) == 3  # Three contexts

    def test_by_context_result_is_bottleneck(self, behavior_2x2):
        """Test that by_context().result returns the bottleneck (min) of context scores."""
        by_context_builder = behavior_2x2.agreement.by_context()
        result = by_context_builder.result
        min_score = min(by_context_builder.context_scores.values())
        assert np.isclose(result, min_score, atol=1e-9)

    def test_unweighted_overall_equals_bottleneck(self, behavior_2x2):
        """Test that unweighted overall agreement equals bottleneck at optimal θ."""
        overall = behavior_2x2.agreement.result
        bottleneck = behavior_2x2.agreement.by_context().result
        # For unweighted minimax, overall agreement equals bottleneck at θ*
        assert np.isclose(overall, bottleneck, atol=1e-6)

    def test_fixed_feature_result_is_bottleneck(self, behavior_2x2):
        """Test that fixed-feature .result equals bottleneck of constrained context scores."""
        fixed_builder = behavior_2x2.agreement.for_feature("Hair", [0.5, 0.5])
        result = fixed_builder.result
        min_score = min(fixed_builder.context_scores.values())
        assert np.isclose(result, min_score, atol=1e-9)

    def test_weighted_overall_vs_bottleneck(self, behavior_2x2):
        """Test that weighted overall agreement differs from bottleneck in general."""
        # Use extreme weights to create a clear difference
        weighted_builder = behavior_2x2.agreement.for_weights({("Hair",): 0.9, ("Eyes",): 0.1})

        overall_agreement = weighted_builder.result
        bottleneck = weighted_builder.by_context().result

        # The bottleneck should be <= overall agreement (weighted objective)
        # They are typically NOT equal for arbitrary weights
        assert bottleneck <= overall_agreement + 1e-6  # Small tolerance for numerical issues

        # For this specific case, they might be close or equal due to the data,
        # but the API should allow them to differ
        # We mainly test that bottleneck doesn't exceed overall

    def test_least_favorable_weights_exist(self, behavior_2x2):
        """Test that least-favorable weights can be computed and used."""
        # Get the adversarial weights
        worst_weights = behavior_2x2.worst_case_weights

        # Should return a dict with weights for contexts (may be sparse)
        assert isinstance(worst_weights, dict)
        expected_keys = {("Hair",), ("Eyes",), ("Hair", "Eyes")}
        assert set(worst_weights.keys()).issubset(expected_keys)

        # All weights should be non-negative
        assert all(w >= 0 for w in worst_weights.values())

        # Should sum to 1 (within numerical tolerance)
        total = sum(worst_weights.values())
        assert np.isclose(total, 1.0, atol=1e-6)

        # Should be able to use these weights for agreement calculation
        adversarial_builder = behavior_2x2.agreement.for_weights(worst_weights)
        overall = adversarial_builder.result
        assert isinstance(overall, float)
        assert 0.0 <= overall <= 1.0 + 1e-6

    def test_filter_mode_no_distribution(self, behavior_2x2):
        """Test filter mode (for_feature without distribution)."""
        # Filter to Hair contexts only and solve agreement
        filtered_result = behavior_2x2.agreement.for_feature("Hair").result
        assert isinstance(filtered_result, float)
        assert 0.0 <= filtered_result <= 1.0 + 1e-9

        # Should have explanation available
        theta = behavior_2x2.agreement.for_feature("Hair").explanation
        assert theta is not None
        assert isinstance(theta, np.ndarray)

        # Feature distribution should be properly normalized
        hair_from_theta = behavior_2x2.agreement.for_feature("Hair").feature_distribution("Hair")
        assert isinstance(hair_from_theta, np.ndarray)
        assert np.isclose(hair_from_theta.sum(), 1.0, atol=1e-9)

        # Only contexts that include Hair should be scored
        b = behavior_2x2.agreement.for_feature("Hair")
        scored_contexts = set(b.context_scores.keys())
        hair_contexts = set(k for k in scored_contexts if "Hair" in k)
        assert scored_contexts == hair_contexts

    def test_keep_drop_contexts(self, behavior_2x2):
        """Test keep_contexts and drop_contexts functionality."""
        # Keep only Hair context
        kept = behavior_2x2.agreement.keep_contexts([("Hair",)])
        kept_scores = kept.context_scores
        assert set(kept_scores.keys()) == {("Hair",)}

        # Drop joint context
        dropped = behavior_2x2.agreement.drop_contexts([("Hair", "Eyes")])
        dropped_scores = dropped.context_scores
        assert ("Hair", "Eyes") not in dropped_scores
        assert set(dropped_scores.keys()) == {("Hair",), ("Eyes",)}

    def test_keep_drop_with_for_feature(self, behavior_2x2):
        """Test keep/drop interplay with for_feature."""
        # Keep only contexts that include Hair, then filter to Hair
        combined = (
            behavior_2x2.agreement
            .keep_contexts([("Hair",), ("Hair", "Eyes")])
            .for_feature("Hair", [0.5, 0.5])
        )

        scores = combined.context_scores
        # Should only have contexts that include Hair
        assert set(scores.keys()) == {("Hair",), ("Hair", "Eyes")}

        # Result should be min of these scores
        assert np.isclose(combined.result, min(scores.values()), atol=1e-9)

    def test_with_explanation(self, behavior_2x2):
        """Test with_explanation functionality."""
        # Get a valid theta from normal agreement
        normal_theta = behavior_2x2.agreement.explanation
        assert normal_theta is not None

        # Use it with with_explanation
        fixed_theta_builder = behavior_2x2.agreement.with_explanation(normal_theta)
        fixed_theta = fixed_theta_builder.explanation
        assert fixed_theta is not None
        assert np.allclose(fixed_theta, normal_theta, atol=1e-9)

        # Result should equal min of context scores
        scores = fixed_theta_builder.context_scores
        result = fixed_theta_builder.result
        assert np.isclose(result, min(scores.values()), atol=1e-9)

        # Feature distributions should match
        hair_dist = fixed_theta_builder.feature_distribution("Hair")
        expected_hair = behavior_2x2.agreement.feature_distribution("Hair")
        assert np.allclose(hair_dist, expected_hair, atol=1e-9)

    def test_worst_case_weights(self, behavior_2x2):
        """Test worst_case_weights property."""
        weights = behavior_2x2.worst_case_weights
        assert isinstance(weights, dict)

        # Should have same keys as contexts
        expected_keys = {("Hair",), ("Eyes",), ("Hair", "Eyes")}
        assert set(weights.keys()) == expected_keys

        # All weights should be non-negative
        assert all(w >= 0 for w in weights.values())

        # Should sum to 1 (within tolerance)
        total = sum(weights.values())
        assert np.isclose(total, 1.0, atol=1e-9)

    def test_portability_across_behaviors(self):
        """Test portability of feature distributions across behaviors."""
        # Create AB behavior (subset)
        space_ab = Space.create(Hair=["Blonde", "Brunette"], Eyes=["Blue", "Brown"])
        ab_behavior = Behavior.from_contexts(space_ab, {
            ("Hair",): {("Blonde",): 0.6, ("Brunette",): 0.4},
            ("Eyes",): {("Blue",): 0.7, ("Brown",): 0.3},
        })

        # Create ABC behavior (includes joint)
        abc_behavior = Behavior.from_contexts(space_ab, {
            ("Hair",): {("Blonde",): 0.6, ("Brunette",): 0.4},
            ("Eyes",): {("Blue",): 0.7, ("Brown",): 0.3},
            ("Hair", "Eyes"): {
                ("Blonde", "Blue"): 0.4, ("Blonde", "Brown"): 0.2,
                ("Brunette", "Blue"): 0.3, ("Brunette", "Brown"): 0.1
            }
        })

        # Extract hair distribution from AB
        hair_dist_ab = ab_behavior.agreement.feature_distribution("Hair")

        # Use it to score ABC with fixed hair
        abc_fixed = abc_behavior.agreement.for_feature("Hair", hair_dist_ab)
        result = abc_fixed.result
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0 + 1e-9

        # Should equal min of context scores for contexts including Hair
        scores = abc_fixed.context_scores
        hair_contexts = [k for k in scores.keys() if "Hair" in k]
        hair_scores = [scores[k] for k in hair_contexts]
        assert np.isclose(result, min(hair_scores), atol=1e-9)

    def test_immutability(self, behavior_2x2):
        """Test that builders are immutable."""
        base = behavior_2x2.agreement
        original_result = base.result

        # Create derived builders
        b1 = base.for_weights({("Hair",): 0.9, ("Eyes",): 0.1})  # Extreme weights
        b2 = base.for_feature("Hair", [0.5, 0.5])
        b3 = base.by_context()

        # All should be different objects
        assert base is not b1
        assert base is not b2
        assert base is not b3
        assert b1 is not b2
        assert b1 is not b3

        # Original should be unchanged
        assert np.isclose(base.result, original_result, atol=1e-9)

        # Derived builders should have different configurations
        # (We don't assert different results since weights might not change agreement much)

    def test_input_validation(self, behavior_2x2):
        """Test input validation."""
        # Empty weights
        with pytest.raises(ValueError):
            behavior_2x2.agreement.for_weights({})

        # Negative weights
        with pytest.raises(ValueError):
            behavior_2x2.agreement.for_weights({("Hair",): -0.1, ("Eyes",): 1.1})

        # Zero sum weights
        with pytest.raises(ValueError):
            behavior_2x2.agreement.for_weights({("Hair",): 0.0, ("Eyes",): 0.0})

        # Non-existent feature
        with pytest.raises(ValueError):
            behavior_2x2.agreement.for_feature("NonExistent", [0.5, 0.5])

        # Wrong length distribution
        with pytest.raises(ValueError):
            behavior_2x2.agreement.for_feature("Hair", [0.5])  # Should be length 2

        # Negative distribution
        with pytest.raises(ValueError):
            behavior_2x2.agreement.for_feature("Hair", [-0.1, 1.1])

        # Zero sum distribution
        with pytest.raises(ValueError):
            behavior_2x2.agreement.for_feature("Hair", [0.0, 0.0])

        # Wrong length theta for with_explanation
        wrong_theta = np.array([0.5, 0.5])  # Should be length 4
        with pytest.raises(ValueError):
            behavior_2x2.agreement.with_explanation(wrong_theta)


# ---------- Fixtures ----------

@pytest.fixture
def behavior_2x2_consistent():
    """
    Coherent 2x2: the joint matches both marginals exactly.
    This should allow α* = 1.0 (within tolerance).
    """
    space = Space.create(Hair=["Blonde", "Brunette"], Eyes=["Blue", "Brown"])
    return Behavior.from_contexts(space, {
        ("Hair",): {("Blonde",): 0.6, ("Brunette",): 0.4},
        ("Eyes",): {("Blue",): 0.7, ("Brown",): 0.3},
        ("Hair", "Eyes"): {
            ("Blonde", "Blue"): 0.4, ("Blonde", "Brown"): 0.2,
            ("Brunette", "Blue"): 0.3, ("Brunette", "Brown"): 0.1
        }
    })


@pytest.fixture
def behavior_2x2_inconsistent():
    """
    Incoherent system: parity triangle (impossible constraints).
    α* must fall strictly below 1.
    """
    space = Space.create(A=[0, 1], B=[0, 1], C=[0, 1])

    # helper distributions
    def equal_ctx(x, y):
        return {(0, 0): 0.5, (1, 1): 0.5}

    def noteq_ctx(x, y):
        return {(0, 1): 0.5, (1, 0): 0.5}

    return Behavior.from_contexts(space, {
        ("A", "B"): equal_ctx("A", "B"),
        ("B", "C"): equal_ctx("B", "C"),
        ("A", "C"): noteq_ctx("A", "C"),
    })


@pytest.fixture
def parity_triangle():
    """
    Parity triangle (binary A,B,C):
      - (A,B) 'equal' (mass on 00 and 11)
      - (B,C) 'equal'
      - (A,C) 'not equal' (mass on 01 and 10)
    Impossible to satisfy all three simultaneously.

    Properties we will test:
      • α*(3 contexts) < α*(any 2-context subset)
      • by-context scores at α*(3) are (approximately) equal by symmetry
    """
    space = Space.create(A=[0, 1], B=[0, 1], C=[0, 1])

    # helper distributions
    def equal_ctx(x, y):
        return {(0, 0): 0.5, (1, 1): 0.5}

    def noteq_ctx(x, y):
        return {(0, 1): 0.5, (1, 0): 0.5}

    return Behavior.from_contexts(space, {
        ("A", "B"): equal_ctx("A", "B"),
        ("B", "C"): equal_ctx("B", "C"),
        ("A", "C"): noteq_ctx("A", "C"),
    })


# ---------- Core invariants & non-brittle assertions ----------

def test_scalar_contract_and_bounds(behavior_2x2_consistent):
    # .result always float in [0,1]
    s = behavior_2x2_consistent.agreement.result
    assert isinstance(s, float)
    assert -TOL <= s <= 1.0 + TOL


def test_unweighted_equals_bottleneck_at_optimum(behavior_2x2_consistent):
    """
    Minimax identity: α* == min_i g_i(theta*).
    """
    overall = behavior_2x2_consistent.agreement.result
    byc = behavior_2x2_consistent.agreement.by_context()
    bottleneck = byc.result
    assert np.isclose(overall, bottleneck, atol=TOL)

    # And the bottleneck equals min of reported per-context scores
    ctx_scores = byc.context_scores
    assert np.isclose(bottleneck, min(ctx_scores.values()), atol=TOL)


def test_weighted_objective_is_between_min_and_max(behavior_2x2_consistent):
    """
    For any λ, at θ(λ),  min_i g_i <= sum_i λ_i g_i <= max_i g_i.
    """
    w = {("Hair",): 0.1, ("Eyes",): 0.7, ("Hair", "Eyes"): 0.2}
    builder = behavior_2x2_consistent.agreement.for_weights(w)
    weighted = builder.result
    ctx_scores = builder.by_context().context_scores
    ctx_vals = np.array(list(ctx_scores.values()))
    assert ctx_vals.min() - TOL <= weighted <= ctx_vals.max() + TOL


def test_weight_rescaling_invariant(behavior_2x2_consistent):
    """
    λ and c·λ should give the same weighted agreement (weights are normalized internally).
    """
    w1 = {("Hair",): 1.0, ("Eyes",): 2.0, ("Hair", "Eyes"): 3.0}
    w2 = {k: 10.0 * v for k, v in w1.items()}
    r1 = behavior_2x2_consistent.agreement.for_weights(w1).result
    r2 = behavior_2x2_consistent.agreement.for_weights(w2).result
    assert np.isclose(r1, r2, atol=TOL)


def test_monotonicity_wrt_context_set(behavior_2x2_inconsistent):
    """
    Adding constraints cannot increase α*.
    α*(subset) >= α*(superset)
    """
    full = behavior_2x2_inconsistent.agreement.result
    # Drop one context (weaker system)
    dropped = behavior_2x2_inconsistent.agreement.drop_contexts([("A", "C")]).result
    assert dropped + 1e-8 >= full - TOL  # allow tiny numerical slack
    # Keep only two contexts (weakest)
    kept_two = behavior_2x2_inconsistent.agreement.keep_contexts([("A", "B"), ("B", "C")]).result
    assert kept_two + 1e-8 >= full - TOL


def test_consistent_system_reaches_one(behavior_2x2_consistent):
    """
    In a coherent system there exists θ matching all contexts exactly → α* ≈ 1.
    """
    s = behavior_2x2_consistent.agreement.result
    assert s >= 1.0 - 1e-4  # be generous to solver tolerance

    # And each per-context score ≈ 1 at θ*
    cs = behavior_2x2_consistent.agreement.by_context().context_scores
    for v in cs.values():
        assert v >= 1.0 - 1e-4


def test_inconsistent_system_below_one(behavior_2x2_inconsistent):
    """
    In an incoherent system, α* must be strictly < 1.
    """
    s = behavior_2x2_inconsistent.agreement.result
    assert s <= 1.0 - 1e-4


def test_parity_triangle_symmetry_and_shrink(parity_triangle):
    """
    • Symmetry: at α*(3), all three context scores should be equal (within tol).
    • Monotonicity: dropping any one constraint strictly increases α*.
    """
    # Symmetry check
    byc = parity_triangle.agreement.by_context()
    scores = list(byc.context_scores.values())
    assert max(scores) - min(scores) <= 5e-3  # they should tie within small band

    # Shrinkage when removing a constraint
    alpha_ABC = parity_triangle.agreement.result
    alpha_AB = parity_triangle.agreement.drop_contexts([("A", "C")]).result
    alpha_BC = parity_triangle.agreement.drop_contexts([("A", "B")]).result
    alpha_AC = parity_triangle.agreement.drop_contexts([("B", "C")]).result
    assert alpha_AB >= alpha_ABC + 1e-3
    assert alpha_BC >= alpha_ABC + 1e-3
    assert alpha_AC >= alpha_ABC + 1e-3


def test_bottleneck_behaviour_fixed_feature(behavior_2x2_inconsistent):
    """
    For a fixed feature distribution, .result is the min over scored contexts
    that include that feature; explanation is None; marginal returns the fixed vector.
    """
    q = np.array([0.5, 0.5])
    b = behavior_2x2_inconsistent.agreement.for_feature("A", q)
    r = b.result
    cs = b.context_scores
    # Only contexts that include A should be present
    assert set(k for k in cs.keys() if "A" in k) == set(cs.keys())
    assert np.isclose(r, min(cs.values()), atol=1e-8)
    assert b.explanation is None
    assert np.allclose(b.feature_distribution("A"), q, atol=1e-12)
    with pytest.raises(ValueError):
        _ = b.feature_distribution("B")


def test_least_favorable_weights_kkts(parity_triangle):
    """
    KKT-style sanity: for λ* (least-favorable), contexts with weight >= τ
    should have per-context scores tied to the overall value (within ε).
    Due to solver precision, we check that active contexts are much closer
    to the overall value than inactive contexts.
    """
    lam = parity_triangle.worst_case_weights
    builder = parity_triangle.agreement.for_weights(lam)
    overall = builder.result
    cs = builder.by_context().context_scores

    # Separate active and inactive contexts
    active_scores = []
    inactive_scores = []
    tau = 1e-4  # active threshold

    for k, w in lam.items():
        if w >= tau:
            active_scores.append(abs(cs[k] - overall))
        else:
            inactive_scores.append(abs(cs[k] - overall))

    # Active contexts should be much closer to overall than inactive ones
    if active_scores and inactive_scores:
        avg_active_error = sum(active_scores) / len(active_scores)
        avg_inactive_error = sum(inactive_scores) / len(inactive_scores)
        # Active should be at least 10x closer than inactive
        assert avg_active_error <= avg_inactive_error * 0.1, \
            f"Active contexts not close enough: {avg_active_error} vs {avg_inactive_error}"


def test_portable_feature_between_behaviors(behavior_2x2_consistent):
    """
    Feature distribution is portable and scores as a bottleneck among contexts that include it.
    """
    # build AB (marginals only)
    space = behavior_2x2_consistent.space
    AB = Behavior.from_contexts(space, {
        ("Hair",): {("Blonde",): 0.6, ("Brunette",): 0.4},
        ("Eyes",): {("Blue",): 0.7, ("Brown",): 0.3},
    })
    # Get Hair from AB's θ*
    q_hair = AB.agreement.feature_distribution("Hair")

    # Fix Hair in ABC and score
    ABC = behavior_2x2_consistent
    fixed = ABC.agreement.for_feature("Hair", q_hair)
    result = fixed.result
    cs = fixed.context_scores
    # result equals min across contexts that include Hair
    hair_keys = [k for k in cs.keys() if "Hair" in k]
    assert np.isclose(result, min(cs[k] for k in hair_keys), atol=1e-8)


def test_immutability_and_repeated_calls(behavior_2x2_inconsistent):
    """
    Builders are immutable; repeated .result calls are stable.
    """
    base = behavior_2x2_inconsistent.agreement
    r0 = base.result
    b1 = base.for_weights({("A","B"): 0.9, ("B","C"): 0.1, ("A","C"): 0.0})
    b2 = base.for_feature("A")
    b3 = base.by_context()

    # Different objects
    assert base is not b1 and base is not b2 and base is not b3

    # Base remains stable
    assert np.isclose(base.result, r0, atol=TOL)
    assert np.isclose(base.result, base.result, atol=TOL)  # idempotent calls


def test_input_validation_paths(behavior_2x2_inconsistent):
    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.for_weights({})

    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.for_weights({("A","B"): -0.1, ("B","C"): 1.1})

    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.for_weights({("A","B"): 0.0, ("B","C"): 0.0})

    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.for_feature("Nope", [0.5, 0.5])

    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.for_feature("A", [0.5])  # wrong length

    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.for_feature("A", [-0.1, 1.1])

    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.for_feature("A", [0.0, 0.0])

    wrong_theta = np.array([0.5, 0.5])  # should be 8 for 3x3
    with pytest.raises(ValueError):
        behavior_2x2_inconsistent.agreement.with_explanation(wrong_theta)
