#!/usr/bin/env python3
from contrakit.observatory import Observatory, NoConceptsDefinedError, LensBuilder, EmptyBehaviorError, ValueHandle
import pytest

def test_observatory_should_create():
    """Test that observatory creates correctly."""
    obs = Observatory.create()
    assert isinstance(obs, Observatory)


def test_concept_alphabet_should_retain_proper_names():
    """Test that value handles retain their string values correctly."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, no_hire = reviewer_a.alphabet
    assert hire.value == "Hire"
    assert no_hire.value == "No_Hire"


def test_observatory_should_create_concepts_correctly():
    """Test that concepts can be defined and tracked properly."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    candidate = observatory.concept("Candidate")
    reviewer_a = observatory.concept("Reviewer_A")
    assert set(observatory._space.names) == {candidate.name, reviewer_a.name}


def test_value_handle_syntactic_sugar_should_work():
    """Test that ValueHandle & operator creates proper tuples."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, no_hire = reviewer_a.alphabet
    joint_outcome = hire & no_hire
    assert joint_outcome == (hire.value, no_hire.value)


def test_perspectives_should_be_accessible():
    """Test that perspectives can be accessed from observatory after defining concepts."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    concept = observatory.concept("TestConcept")
    perspectives = observatory.perspectives
    assert type(perspectives).__name__ == "PerspectiveMap"


def test_joint_distribution_should_be_settable():
    """Test that joint distributions can be set and retrieved correctly."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    hire, no_hire = reviewer_a.alphabet

    perspectives = observatory.perspectives
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.3,
        hire & no_hire: 0.4,
        no_hire & hire: 0.1,
        no_hire & no_hire: 0.2
    }

    joint_dist = perspectives[reviewer_a, reviewer_b].distribution
    assert joint_dist[(hire.value, hire.value)] == pytest.approx(0.3)


def test_marginal_distribution_should_auto_complete():
    """Test that marginal distributions auto-complete missing outcomes."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer = observatory.concept("Reviewer")
    hire, no_hire = reviewer.alphabet

    perspectives = observatory.perspectives
    perspectives[reviewer] = {hire: 0.7}

    marg_dist = perspectives[reviewer].distribution
    assert marg_dist[(hire.value,)] == pytest.approx(0.7)
    assert marg_dist[(no_hire.value,)] == pytest.approx(0.3)


def test_perspectives_should_validate_distributions():
    """Test that perspectives can validate distribution consistency."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    hire, no_hire = reviewer_a.alphabet

    perspectives = observatory.perspectives
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.3,
        hire & no_hire: 0.4,
        no_hire & hire: 0.1,
        no_hire & no_hire: 0.2
    }

    # Should not raise an exception
    perspectives.validate()


def test_perspectives_should_create_behavior():
    """Test that perspectives can create behavior objects."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    hire, no_hire = reviewer_a.alphabet

    perspectives = observatory.perspectives
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.3,
        hire & no_hire: 0.4,
        no_hire & hire: 0.1,
        no_hire & no_hire: 0.2
    }

    behavior = perspectives.to_behavior()
    assert type(behavior).__name__ == "Behavior"


def test_behavior_should_have_agreement():
    """Test that behavior objects have agreement property."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    hire, no_hire = reviewer_a.alphabet

    perspectives = observatory.perspectives
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.3,
        hire & no_hire: 0.4,
        no_hire & hire: 0.1,
        no_hire & no_hire: 0.2
    }

    behavior = perspectives.to_behavior()
    assert isinstance(behavior.agreement, (int, float))
    assert 0.0 <= behavior.agreement <= 1.0


def test_accessing_perspectives_before_defining_concepts_should_raise_error():
    """Test that accessing perspectives before defining concepts raises ValueError."""
    try:
        _ = Observatory.create().perspectives
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Perspectives not available" in str(e)


def test_accessing_space_before_defining_concepts_should_return_none():
    """Test that accessing space before defining concepts returns None."""
    space = Observatory.create()._space
    assert space is None


def test_non_normalized_distribution_should_raise_value_error():
    """Test that non-normalized distributions raise ValueError."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    concept = observatory.concept("X")
    hire, no_hire = concept.alphabet
    perspectives = observatory.perspectives

    try:
        perspectives[concept] = {hire: 0.5, no_hire: 0.6}  # Sums to 1.1
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must sum to 1" in str(e)


def test_accessing_context_with_undefined_concept_should_raise_key_error():
    """Test that accessing context with concept not in perspectives raises KeyError."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    x_concept = observatory.concept("X")
    y_concept = observatory.concept("Y")
    perspectives = observatory.perspectives

    # Define a concept after perspectives are accessed
    z_concept = observatory.concept("Z")

    # Try to access a context that includes the newly defined concept
    try:
        _ = perspectives[x_concept, y_concept, z_concept]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_distribution_should_provide_to_dict():
    """Test that Distribution objects provide to_dict() method."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    concept = observatory.concept("Test")
    hire, no_hire = concept.alphabet

    perspectives = observatory.perspectives
    perspectives[concept] = {hire: 0.6}

    dist_wrapper = perspectives[concept]
    dist_dict = dist_wrapper.distribution.to_dict()

    expected_keys = {(hire.value,), (no_hire.value,)}
    assert set(dist_dict.keys()) == expected_keys
    assert dist_dict[(hire.value,)] == pytest.approx(0.6)
    assert dist_dict[(no_hire.value,)] == pytest.approx(0.4)


def test_value_handle_should_have_string_representation():
    """Test that ValueHandle has proper string representation."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, no_hire = reviewer_a.alphabet

    assert str(hire) == "Hire"
    assert str(no_hire) == "No_Hire"


def test_value_handle_should_have_repr():
    """Test that ValueHandle has proper repr."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, _ = reviewer_a.alphabet

    assert repr(hire) == "ValueHandle('Hire')"


def test_value_handle_should_support_and_operator():
    """Test that ValueHandle supports & operator for joint outcomes."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, no_hire = reviewer_a.alphabet

    assert (hire & no_hire) == (hire.value, no_hire.value)
    assert (no_hire & hire) == (no_hire.value, hire.value)


def test_space_should_have_correct_properties():
    """Test that Space object has correct properties and representation."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])

    concept_a = observatory.concept("A")
    concept_b = observatory.concept("B")
    hire, no_hire = concept_a.alphabet

    space = observatory._space
    assert len(space) == 2
    assert concept_a in space
    assert concept_b in space
    assert space[concept_a] == (hire.value, no_hire.value)
    assert "Space" in repr(space)


def test_concepts_should_be_properly_stored():
    """Test that concepts are properly stored in observatory."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    concept_a = observatory.concept("A")
    concept_b = observatory.concept("B")

    assert len(observatory._concepts) == 2
    assert concept_a in observatory._concepts.values()
    assert concept_b in observatory._concepts.values()


def test_compositional_behavior():
    """Test how different components interact and compose."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    candidate = observatory.concept("Candidate")
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    reviewer_c = observatory.concept("Reviewer_C")

    # Set up some distributions
    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    perspectives[reviewer_a] = {hire: 0.6}
    perspectives[reviewer_b] = {hire: 0.4}
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.24,      # 0.6 * 0.4
        hire & no_hire: 0.36,   # 0.6 * 0.6
        no_hire & hire: 0.16,   # 0.4 * 0.4
        no_hire & no_hire: 0.24  # 0.4 * 0.6
    }

    behavior = perspectives.to_behavior()

    # Check that behavior has the expected contexts
    assert len(behavior.distributions) == 3  # Reviewer_A, Reviewer_B, Reviewer_A+Reviewer_B

    # Check agreement is reasonable
    agreement = behavior.agreement
    assert 0.0 <= agreement <= 1.0


def test_concept_should_return_concept_handles():
    """Test that concept() returns ConceptHandle objects."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    A = observatory.concept("A", symbols=["Yes", "No"])
    B = observatory.concept("B", symbols=["Yes", "No"])
    C = observatory.concept("C", symbols=["Yes", "No"])
    assert A.name == "A" and B.name == "B" and C.name == "C"


def test_concept_should_create_space():
    """Test that concept() creates proper Space object."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    observatory.concept("A", symbols=["Yes", "No"])
    observatory.concept("B", symbols=["Yes", "No"])
    observatory.concept("C", symbols=["Yes", "No"])

    assert observatory._space is not None
    assert set(observatory._space.names) == {"A", "B", "C"}


def test_concept_should_enable_perspectives():
    """Test that perspectives work after defining concepts."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    observatory.concept("A", symbols=["Yes", "No"])
    observatory.concept("B", symbols=["Yes", "No"])
    observatory.concept("C", symbols=["Yes", "No"])

    perspectives = observatory.perspectives
    assert type(perspectives).__name__ == "PerspectiveMap"


def test_empty_symbols_should_prevent_concept_definition():
    """Test that defining concepts fails with empty symbols."""
    obs = Observatory.create()
    try:
        obs.concept("X", symbols=[])
        assert False, "Should fail with empty symbols"
    except ValueError:
        pass


def test_single_value_concept_should_work():
    """Test that single-value concepts work correctly."""
    obs = Observatory.create()
    X = obs.concept("X", symbols=["Only"])
    perspectives = obs.perspectives

    # Should work with single value
    perspectives[X] = {X.alphabet[0]: 1.0}
    marg = perspectives[X].distribution
    assert marg[(("Only",))] == 1.0


def test_concept_redefinition_should_raise_error():
    """Test that re-defining concepts raises an error."""
    obs = Observatory.create()
    obs.concept("Y", symbols=["Yes", "No"])
    assert "Y" in obs._space.names

    # Should NOT be able to redefine existing concepts
    try:
        obs.concept("Y", symbols=["Yes", "No"])
        assert False, "Should have raised ValueError for duplicate concept"
    except ValueError as e:
        assert "Concept 'Y' already exists" in str(e)


def test_inconsistent_distributions_should_have_valid_agreement():
    """Test that inconsistent distributions produce valid agreement scores."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # Set up inconsistent distributions (similar to fixture)
    perspectives[reviewer_a] = {hire: 0.8}
    perspectives[reviewer_b] = {hire: 0.2}
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.1,
        hire & no_hire: 0.1,
        no_hire & hire: 0.1,
        no_hire & no_hire: 0.7
    }

    behavior = perspectives.to_behavior()
    assert 0.0 <= behavior.agreement <= 1.0


def test_invalid_probabilities_should_raise_value_error():
    """Test that invalid probability distributions raise ValueError."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    try:
        perspectives[reviewer_a, reviewer_b] = {
            hire & hire: 0.5,
            hire & no_hire: 0.6,  # These sum to 1.1 > 1.0
            no_hire & hire: 0.0,
            no_hire & no_hire: 0.0
        }
        assert False, "Should have raised ValueError for invalid probabilities"
    except ValueError as e:
        assert "must sum to 1" in str(e)


def test_behavior_should_have_correct_space():
    """Test that behavior object has correct space reference."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    candidate = observatory.concept("Candidate")
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    perspectives[reviewer_a] = {hire: 0.6}
    perspectives[reviewer_b] = {hire: 0.4}

    behavior = perspectives.to_behavior()
    assert behavior.space == observatory._space
    assert set(behavior.space.names) == {candidate.name, reviewer_a.name, reviewer_b.name}


def test_behavior_distributions_should_have_context_keys():
    """Test that behavior distributions have Context objects as keys."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    perspectives[reviewer_a] = {hire: 0.6}
    perspectives[reviewer_b] = {hire: 0.4}
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.24,
        hire & no_hire: 0.36,
        no_hire & hire: 0.16,
        no_hire & no_hire: 0.24
    }

    behavior = perspectives.to_behavior()
    context_keys = list(behavior.distributions.keys())
    assert len(context_keys) == 3  # Reviewer_A, Reviewer_B, Reviewer_A+Reviewer_B

    from contrakit.context import Context
    for context in context_keys:
        assert isinstance(context, Context)
        assert context.space == behavior.space


def test_behavior_should_have_valid_agreement_range():
    """Test that behavior agreement is within valid range."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    perspectives[reviewer_a] = {hire: 0.6}

    behavior = perspectives.to_behavior()
    assert isinstance(behavior.agreement, (int, float))
    assert 0.0 <= behavior.agreement <= 1.0


def test_marginal_distribution_overwriting_should_work():
    """Test that marginal distributions can be overwritten multiple times."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # First assignment
    perspectives[reviewer_a] = {hire: 0.6}
    dist1 = perspectives[reviewer_a].distribution
    assert dist1[(hire.value,)] == pytest.approx(0.6)
    assert dist1[(no_hire.value,)] == pytest.approx(0.4)

    # Overwrite with different values
    perspectives[reviewer_a] = {hire: 0.8}
    dist2 = perspectives[reviewer_a].distribution
    assert dist2[(hire.value,)] == pytest.approx(0.8)
    assert dist2[(no_hire.value,)] == pytest.approx(0.2)


def test_joint_distribution_overwriting_should_work():
    """Test that joint distributions can be overwritten."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # First joint assignment
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.5,
        hire & no_hire: 0.3,
        no_hire & hire: 0.1,
        no_hire & no_hire: 0.1
    }

    joint1 = perspectives[reviewer_a, reviewer_b].distribution
    assert joint1[(hire.value, hire.value)] == pytest.approx(0.5)

    # Overwrite joint
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.2,
        hire & no_hire: 0.4,
        no_hire & hire: 0.3,
        no_hire & no_hire: 0.1
    }

    joint2 = perspectives[reviewer_a, reviewer_b].distribution
    assert joint2[(hire.value, hire.value)] == pytest.approx(0.2)
    assert joint2[(hire.value, no_hire.value)] == pytest.approx(0.4)


def test_distribution_overwriting_should_not_affect_others():
    """Test that overwriting one distribution doesn't affect others."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # Set up joint distribution
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.2,
        hire & no_hire: 0.4,
        no_hire & hire: 0.3,
        no_hire & no_hire: 0.1
    }

    # Set up marginal for reviewer_b
    perspectives[reviewer_b] = {hire: 0.9}

    # Check that joint distribution is unchanged
    joint = perspectives[reviewer_a, reviewer_b].distribution
    assert joint[(hire.value, hire.value)] == pytest.approx(0.2)

    # Check that marginal for reviewer_b is correct
    marg_b = perspectives[reviewer_b].distribution
    assert marg_b[(hire.value,)] == pytest.approx(0.9)


def test_contradictory_joint_distributions_should_validate():
    """Test that contradictory joint distributions still validate without error."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # Set up marginals
    perspectives[reviewer_a] = {hire: 0.6}
    perspectives[reviewer_b] = {hire: 0.4}

    # Joint distribution that contradicts marginals
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.8,     # Would imply P(A=Hire) ≈ 0.8, P(B=Hire) ≈ 0.8
        hire & no_hire: 0.1,  # Would imply P(A=Hire) ≈ 0.1, P(B=No_Hire) ≈ 0.1
        no_hire & hire: 0.05, # Would imply P(A=No_Hire) ≈ 0.05, P(B=Hire) ≈ 0.05
        no_hire & no_hire: 0.05 # Would imply P(A=No_Hire) ≈ 0.05, P(B=No_Hire) ≈ 0.05
    }

    # This should validate without raising ValueError
    perspectives.validate()


def test_contradictory_distributions_should_produce_valid_agreement():
    """Test that contradictory distributions produce valid agreement scores."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # Set up contradictory distributions
    perspectives[reviewer_a] = {hire: 0.6}
    perspectives[reviewer_b] = {hire: 0.4}
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.8,
        hire & no_hire: 0.1,
        no_hire & hire: 0.05,
        no_hire & no_hire: 0.05
    }

    behavior = perspectives.to_behavior()
    assert 0.0 <= behavior.agreement <= 1.0


def test_probability_tolerance_should_accept_near_one():
    """Test that distributions summing to nearly 1.0 are accepted."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # Test distribution very close to 1.0 (should pass)
    perspectives[reviewer_a] = {hire: 0.999999999, no_hire: 0.000000001}
    perspectives.validate()  # Should not raise


def test_probability_tolerance_should_reject_over_one():
    """Test that distributions summing to over 1.0 are rejected."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")

    hire, no_hire = reviewer_a.alphabet
    perspectives = observatory.perspectives

    # Test distribution significantly over 1.0 (should fail)
    try:
        perspectives[reviewer_a] = {hire: 0.6, no_hire: 0.5}  # Sums to 1.1
        assert False, "Should have raised ValueError for sum over 1.0"
    except ValueError as e:
        assert "must sum to 1" in str(e)

# Global Alphabet Tests

def test_observatory_create_with_global_alphabet():
    """Test that Observatory.create() sets up global alphabet correctly."""
    obs = Observatory.create(symbols=["Yes", "No", "Maybe"])

    assert len(obs.alphabet) == 3
    assert all(isinstance(vh, ValueHandle) for vh in obs.alphabet)
    assert [str(vh) for vh in obs.alphabet] == ["Yes", "No", "Maybe"]
    assert [vh.value for vh in obs.alphabet] == ["Yes", "No", "Maybe"]


def test_observatory_create_without_global_alphabet():
    """Test that Observatory.create() without symbols has empty alphabet."""
    obs = Observatory.create()

    assert obs.alphabet == ()
    assert len(obs.alphabet) == 0


def test_alphabet_property_returns_value_handle_tuple():
    """Test that obs.alphabet returns Tuple[ValueHandle, ...] like concept.alphabet."""
    obs = Observatory.create(symbols=["Yes", "No"])
    concept = obs.concept("Test", symbols=["A", "B"])

    # Both should return tuples of ValueHandle objects
    assert isinstance(obs.alphabet, tuple)
    assert isinstance(concept.alphabet, tuple)
    assert all(isinstance(vh, ValueHandle) for vh in obs.alphabet)
    assert all(isinstance(vh, ValueHandle) for vh in concept.alphabet)


def test_concept_uses_global_alphabet_when_no_symbols_provided():
    """Test that concept() uses global alphabet when symbols=None."""
    obs = Observatory.create(symbols=["Yes", "No"])
    concept = obs.concept("Voter")

    assert [vh.value for vh in concept.alphabet] == ["Yes", "No"]
    assert [str(vh) for vh in concept.alphabet] == ["Yes", "No"]


def test_concept_with_value_handle_objects_in_symbols():
    """Test that concept() accepts ValueHandle objects in symbols parameter."""
    obs1 = Observatory.create(symbols=["Yes", "No"])
    yes, no = obs1.alphabet

    obs2 = Observatory.create()
    candidate = obs2.concept("Candidate", symbols=["Qualified", "Unqualified", yes, no])

    expected_symbols = ["Qualified", "Unqualified", "Yes", "No"]
    assert [vh.value for vh in candidate.alphabet] == expected_symbols
    assert [str(vh) for vh in candidate.alphabet] == expected_symbols


def test_observatory_create_with_value_handle_objects():
    """Test that Observatory.create() accepts ValueHandle objects in symbols parameter."""
    obs1 = Observatory.create(symbols=["Yes", "No"])
    yes, no = obs1.alphabet

    obs2 = Observatory.create(symbols=["Maybe", yes, no])

    expected_symbols = ["Maybe", "Yes", "No"]
    assert [vh.value for vh in obs2.alphabet] == expected_symbols
    assert [str(vh) for vh in obs2.alphabet] == expected_symbols


def test_value_handle_objects_work_across_observatories():
    """Test that ValueHandle objects from one observatory work in another."""
    obs1 = Observatory.create(symbols=["Yes", "No"])
    yes1, no1 = obs1.alphabet

    obs2 = Observatory.create(symbols=["Maybe", yes1, no1])
    maybe, yes2, no2 = obs2.alphabet

    # ValueHandle objects should be different instances but have same values
    assert yes1.value == yes2.value == "Yes"
    assert no1.value == no2.value == "No"
    assert yes1 is not yes2  # Different instances
    assert no1 is not no2    # Different instances

    # Should be able to create concepts using these ValueHandle objects
    concept = obs2.concept("Test", symbols=[maybe, yes2, no2])
    assert [vh.value for vh in concept.alphabet] == ["Maybe", "Yes", "No"]


def test_perspectives_accept_value_handle_objects():
    """Test that perspectives accept ValueHandle objects as keys and values."""
    obs = Observatory.create(symbols=["Yes", "No"])
    yes, no = obs.alphabet

    voter = obs.concept("Voter")
    obs.perspectives[voter] = {yes: 0.6, no: 0.4}

    distribution = obs.perspectives[voter].distribution
    assert distribution[(yes.value,)] == pytest.approx(0.6)
    assert distribution[(no.value,)] == pytest.approx(0.4)


def test_joint_distributions_with_value_handle_objects():
    """Test joint distributions using ValueHandle objects from & operator."""
    obs = Observatory.create(symbols=["Yes", "No"])
    yes, no = obs.alphabet

    voter = obs.concept("Voter")
    candidate = obs.concept("Candidate", symbols=["Qualified", "Unqualified"])

    # Use & operator with ValueHandle objects
    obs.perspectives[voter, candidate] = {
        yes & candidate.alphabet[0]: 0.3,  # yes & qualified
        yes & candidate.alphabet[1]: 0.2,  # yes & unqualified
        no & candidate.alphabet[0]: 0.1,   # no & qualified
        no & candidate.alphabet[1]: 0.4    # no & unqualified
    }

    joint_dist = obs.perspectives[voter, candidate].distribution
    assert joint_dist[(yes.value, "Qualified")] == pytest.approx(0.3)
    assert joint_dist[(yes.value, "Unqualified")] == pytest.approx(0.2)
    assert joint_dist[(no.value, "Qualified")] == pytest.approx(0.1)
    assert joint_dist[(no.value, "Unqualified")] == pytest.approx(0.4)


def test_global_alphabet_value_handle_uniqueness():
    """Test that global alphabet ValueHandle objects are unique per observatory."""
    obs1 = Observatory.create(symbols=["Yes", "No"])
    obs2 = Observatory.create(symbols=["Yes", "No"])

    yes1, no1 = obs1.alphabet
    yes2, no2 = obs2.alphabet

    # Same values but different ValueHandle instances
    assert yes1.value == yes2.value == "Yes"
    assert no1.value == no2.value == "No"
    assert yes1 is not yes2
    assert no1 is not no2

    # Different concepts (different observatories)
    assert yes1.concept is not yes2.concept
    assert yes1.concept.name == "__global__"
    assert yes2.concept.name == "__global__"


def test_concept_with_mixed_value_handle_and_string_values():
    """Test concept creation with mix of ValueHandle objects and strings."""
    obs1 = Observatory.create(symbols=["Yes", "No"])
    yes, no = obs1.alphabet

    obs2 = Observatory.create()
    mixed_concept = obs2.concept("Mixed", symbols=["Start", yes, "Middle", no, "End"])

    expected = ["Start", "Yes", "Middle", "No", "End"]
    assert [vh.value for vh in mixed_concept.alphabet] == expected
    assert [str(vh) for vh in mixed_concept.alphabet] == expected


def test_global_alphabet_preserves_order():
    """Test that global alphabet preserves the order of provided values."""
    values = ["Third", "First", "Second", "Z", "A", "M"]
    obs = Observatory.create(symbols=values)

    # Order should be preserved exactly as provided
    assert [vh.value for vh in obs.alphabet] == values
    assert [str(vh) for vh in obs.alphabet] == values

    # Test that concepts using global alphabet also preserve order
    concept = obs.concept("Test")
    assert [vh.value for vh in concept.alphabet] == values
    assert [str(vh) for vh in concept.alphabet] == values


def test_concept_alphabet_preserves_order():
    """Test that concept alphabet preserves the order of provided values."""
    obs = Observatory.create()
    values = ["Gamma", "Alpha", "Beta", "Omega", "Delta"]
    concept = obs.concept("Ordered", symbols=values)

    # Order should be preserved exactly as provided
    assert [vh.value for vh in concept.alphabet] == values
    assert [str(vh) for vh in concept.alphabet] == values


def test_concept_with_value_handles_preserves_order():
    """Test that concept creation with ValueHandle objects preserves order."""
    obs1 = Observatory.create(symbols=["X", "Y", "Z"])
    x, y, z = obs1.alphabet

    obs2 = Observatory.create()
    # Create concept with ValueHandle objects in specific order
    order = [z, x, y, z, x]  # Note: duplicates should be preserved
    concept = obs2.concept("Handles", symbols=order)

    expected_values = ["Z", "X", "Y", "Z", "X"]
    assert [vh.value for vh in concept.alphabet] == expected_values
    assert [str(vh) for vh in concept.alphabet] == expected_values


def test_observatory_create_with_value_handles_preserves_order():
    """Test that Observatory.create with ValueHandle objects preserves order."""
    obs1 = Observatory.create(symbols=["A", "B", "C"])
    a, b, c = obs1.alphabet

    # Create second observatory with ValueHandle objects in different order
    order = [c, a, b, a, c]
    obs2 = Observatory.create(symbols=order)

    expected_values = ["C", "A", "B", "A", "C"]
    assert [vh.value for vh in obs2.alphabet] == expected_values
    assert [str(vh) for vh in obs2.alphabet] == expected_values


def test_concept_ordering_with_duplicates():
    """Test that ordering is preserved even with duplicate values."""
    obs = Observatory.create()
    # Create concept with duplicate values in specific order
    values = ["A", "B", "A", "C", "B", "A"]
    concept = obs.concept("Duplicates", symbols=values)

    # Duplicates should be preserved in the exact order provided
    assert [vh.value for vh in concept.alphabet] == values
    assert [str(vh) for vh in concept.alphabet] == values
    assert len(concept.alphabet) == len(values)


def test_global_alphabet_ordering_with_duplicates():
    """Test that global alphabet preserves order with duplicate values."""
    # Create observatory with duplicate values in specific order
    values = ["X", "Y", "X", "Z", "Y", "X"]
    obs = Observatory.create(symbols=values)

    # Order should be preserved exactly
    assert [vh.value for vh in obs.alphabet] == values
    assert [str(vh) for vh in obs.alphabet] == values
    assert len(obs.alphabet) == len(values)

    # Concept using global alphabet should also preserve order
    concept = obs.concept("Test")
    assert [vh.value for vh in concept.alphabet] == values
    assert [str(vh) for vh in concept.alphabet] == values


def test_mixed_value_types_preserve_order():
    """Test that mixed string and ValueHandle values preserve order."""
    obs1 = Observatory.create(symbols=["P", "Q"])
    p, q = obs1.alphabet

    obs2 = Observatory.create()
    # Mix strings and ValueHandle objects in specific order
    mixed_order = ["Start", p, "Middle", q, "End", p, q]
    concept = obs2.concept("MixedOrder", symbols=mixed_order)

    expected = ["Start", "P", "Middle", "Q", "End", "P", "Q"]
    assert [vh.value for vh in concept.alphabet] == expected
    assert [str(vh) for vh in concept.alphabet] == expected


def test_meta_lens_joint_and_composition():
    """Test meta-lenses with joint observations using & syntax and composition."""
    obs = Observatory.create(symbols=["Hire", "No_Hire"])
    hire, no_hire = obs.alphabet
    
    # Level 1: two reviewers
    alice = obs.lens("Alice", symbols=["Reliable", "Unreliable"])
    bob = obs.lens("Bob", symbols=["Consistent", "Inconsistent"])
    
    candidate = obs.concept("Candidate")
    
    # Alice: slightly pro-hire
    with alice:
        alice.perspectives[candidate] = {hire: 0.7, no_hire: 0.3}
    
    # Bob: slightly anti-hire
    with bob:
        bob.perspectives[candidate] = {hire: 0.3, no_hire: 0.7}
    
    # Level 2: Auditor observing Alice & Bob jointly
    auditor = obs.lens("Auditor", symbols=[
        "BothGood", "BothBad", "Disagree"
    ])
    
    reliable, unreliable = alice.alphabet
    consistent, inconsistent = bob.alphabet
    both_good, both_bad, disagree = auditor.alphabet
    
    with auditor:
        # Test joint observation using & syntax
        auditor.perspectives[alice & bob] = {
            (reliable, consistent): 0.4,      # Both good
            (unreliable, inconsistent): 0.3,  # Both bad
            (reliable, inconsistent): 0.2,    # Disagree
            (unreliable, consistent): 0.1     # Disagree
        }
    
    # Behaviors
    alice_behavior = alice.to_behavior()
    bob_behavior = bob.to_behavior()
    auditor_behavior = auditor.to_behavior()
    
    # --- Numerical Checks ---
    
    # 1. Marginals should normalize
    alice_contexts = list(alice_behavior.distributions.keys())
    alice_dist = alice_behavior.distributions[alice_contexts[0]]
    assert abs(sum(alice_dist.probs) - 1.0) < 1e-9
    
    bob_contexts = list(bob_behavior.distributions.keys())
    bob_dist = bob_behavior.distributions[bob_contexts[0]]
    assert abs(sum(bob_dist.probs) - 1.0) < 1e-9
    
    # 2. Auditor joint should normalize
    auditor_contexts = list(auditor_behavior.distributions.keys())
    joint_dist = auditor_behavior.distributions[auditor_contexts[0]]
    assert abs(sum(joint_dist.probs) - 1.0) < 1e-9
    
    # 3. Contradiction bits check
    composed = alice | bob
    composed_behavior = composed.to_behavior()
    # Should be > 0 because Alice and Bob disagree
    assert composed_behavior.contradiction_bits > 0.0
    
    # 4. Agreement is bounded
    assert 0.0 <= composed_behavior.agreement <= 1.0
    assert 0.0 <= auditor_behavior.agreement <= 1.0
    
    # 5. Test that & syntax works for creating joint contexts
    joint_context = alice & bob
    assert isinstance(joint_context, tuple)
    assert len(joint_context) == 2
    assert joint_context[0] is alice
    assert joint_context[1] is bob
    
    # 6. Test mixed concept-lens joint observations
    auditor2 = obs.lens("Auditor2", symbols=["Good", "Bad"])
    good, bad = auditor2.alphabet
    
    with auditor2:
        # Joint observation of concept and lens
        auditor2.perspectives[candidate & alice] = {
            (hire, reliable): 0.5,
            (hire, unreliable): 0.2,
            (no_hire, reliable): 0.2,
            (no_hire, unreliable): 0.1
        }
    
    auditor2_behavior = auditor2.to_behavior()
    assert 0.0 <= auditor2_behavior.agreement <= 1.0

