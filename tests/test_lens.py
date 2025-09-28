#!/usr/bin/env python3
from contrakit.observatory import Observatory, NoConceptsDefinedError, LensBuilder, EmptyBehaviorError
import pytest


# LensBuilder Tests

def test_lens_should_create_with_concept():
    """Test that lens creates correctly with a concept."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    lens = observatory.lens(reviewer_a)
    assert isinstance(lens, LensBuilder)
    assert lens.name == reviewer_a.name


def test_lens_should_work_as_context_manager():
    """Test that lens works as a context manager."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    with observatory.lens(reviewer_a) as lens:
        assert isinstance(lens, LensBuilder)


def test_lens_with_statement_should_define_concepts():
    """Test that with statement allows defining concepts in lens."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")

    # Use with statement to define concepts within lens
    with observatory.lens(reviewer_a) as lens_a:
        candidate = lens_a.define("Candidate")
        assert candidate.name == "Candidate"
        assert "Candidate" in observatory._space.names


def test_lens_with_statement_should_setup_perspectives():
    """Test that with statement allows setting up perspectives in lens."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, no_hire = reviewer_a.alphabet

    with observatory.lens(reviewer_a) as lens_a:
        candidate = lens_a.define("Candidate")
        lens_a.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
        behavior = lens_a.to_behavior()
        # Verify context is in base space and includes candidate
        ctxs = [tuple(ctx.observables) for ctx in behavior.distributions.keys()]
        assert (candidate.name,) in ctxs


def test_lens_with_statement_should_generate_behavior():
    """Test that with statement allows generating behavior from lens."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, no_hire = reviewer_a.alphabet

    with observatory.lens(reviewer_a) as lens_a:
        candidate = lens_a.define("Candidate")
        lens_a.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
        behavior = lens_a.to_behavior()

        assert type(behavior).__name__ == "Behavior"
        agreement = behavior.agreement.result
        assert isinstance(agreement, (int, float))
        assert agreement >= 0.0 - 1e-10
        assert agreement <= 1.0 + 1e-10


def test_lens_behavior_should_contain_expected_distributions():
    """Test that lens-generated behavior contains expected distributions."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    hire, no_hire = reviewer_a.alphabet

    with observatory.lens(reviewer_a) as lens_a:
        candidate = lens_a.define("Candidate")
        lens_a.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
        behavior = lens_a.to_behavior()

        # Verify the behavior contains the expected distributions
        assert len(behavior.distributions) == 1
        context_key = list(behavior.distributions.keys())[0]
        assert candidate.name in context_key.observables


def test_multiple_lenses_should_create_independent_behaviors():
    """Test that multiple lenses create independent behaviors."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])

    # Define two lens concepts
    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    # Define shared concept once globally so lenses don't duplicate it
    candidate = observatory.concept("Candidate")
    hire, no_hire = reviewer_a.alphabet

    with observatory.lens(reviewer_a) as lens_a:
        lens_a.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
        behavior_a = lens_a.to_behavior()

    with observatory.lens(reviewer_b) as lens_b:
        lens_b.perspectives[candidate] = {hire: 0.3, no_hire: 0.7}
        behavior_b = lens_b.to_behavior()

    # Verify both behaviors exist
    assert behavior_a is not None
    assert behavior_b is not None


def test_multiple_lenses_should_have_valid_agreements():
    """Test that multiple lenses produce behaviors with valid agreements."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])

    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    candidate = observatory.concept("Candidate")
    hire, no_hire = reviewer_a.alphabet

    with observatory.lens(reviewer_a) as lens_a:
        lens_a.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
        behavior_a = lens_a.to_behavior()

    with observatory.lens(reviewer_b) as lens_b:
        lens_b.perspectives[candidate] = {hire: 0.3, no_hire: 0.7}
        behavior_b = lens_b.to_behavior()

    # Verify agreements are valid numbers in range [0, 1]
    agreement_a = behavior_a.agreement.result
    agreement_b = behavior_b.agreement.result
    assert isinstance(agreement_a, (int, float))
    assert isinstance(agreement_b, (int, float))
    assert agreement_a >= 0.0 - 1e-10
    assert agreement_a <= 1.0 + 1e-10
    assert agreement_b >= 0.0 - 1e-10
    assert agreement_b <= 1.0 + 1e-10


def test_multiple_lenses_should_have_same_distribution_count():
    """Test that multiple lenses have the same number of distributions."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])

    reviewer_a = observatory.concept("Reviewer_A")
    reviewer_b = observatory.concept("Reviewer_B")
    candidate = observatory.concept("Candidate")
    hire, no_hire = reviewer_a.alphabet

    with observatory.lens(reviewer_a) as lens_a:
        lens_a.perspectives[candidate] = {hire: 0.8, no_hire: 0.2}
        behavior_a = lens_a.to_behavior()

    with observatory.lens(reviewer_b) as lens_b:
        lens_b.perspectives[candidate] = {hire: 0.3, no_hire: 0.7}
        behavior_b = lens_b.to_behavior()

    # Both should have 1 distribution (single context each)
    assert len(behavior_a.distributions) == 1
    assert len(behavior_b.distributions) == 1


def test_lens_should_allow_concept_definition():
    """Test that lens allows defining concepts locally."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    lens = observatory.lens(reviewer_a)

    candidate = lens.define("Candidate")
    assert candidate.name == "Candidate"
    assert "Candidate" in observatory._space.names


def test_lens_should_prevent_duplicate_concept_definition():
    """Test that lens prevents defining the same concept twice."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    lens = observatory.lens(reviewer_a)

    lens.define("Candidate")
    try:
        lens.define("Candidate")
        assert False, "Should have raised ValueError for duplicate concept"
    except ValueError as e:
        # Now enforced by global concept uniqueness
        assert "Concept 'Candidate' already exists" in str(e)


def test_lens_should_expose_raw_behavior_with_lens_axis():
    """Raw behavior should include a hidden lens axis when requested."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    lens = observatory.lens(reviewer_a)
    candidate = lens.define("Candidate")
    hire, no_hire = reviewer_a.alphabet
    lens.perspectives[candidate] = {hire: 1.0}

    # Base behavior has no lens axis
    b_base = lens.to_behavior()
    assert set(b_base.space.names) == set(observatory._space.names)

    # Raw behavior includes an extra singleton axis (the lens tag)
    b_raw = lens.to_behavior_raw()
    assert len(b_raw.space.names) == len(observatory._space.names) + 1
    # The extra axis should be a singleton alphabet
    extra_names = set(b_raw.space.names) - set(observatory._space.names)
    assert len(extra_names) == 1
    extra = extra_names.pop()
    assert len(b_raw.space.alphabets[extra]) == 1


def test_lens_should_provide_perspectives_after_concept_definition():
    """Test that lens provides perspectives after defining concepts."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    lens = observatory.lens(reviewer_a)

    # Define a concept to enable perspectives
    lens.define("Candidate")

    perspectives = lens.perspectives
    assert type(perspectives).__name__ == "LensPerspectiveProxy"


def test_lens_should_generate_behavior():
    """Test that lens can generate behavior from perspectives."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    lens = observatory.lens(reviewer_a)

    candidate = lens.define("Candidate")
    hire, no_hire = reviewer_a.alphabet

    # Set up a simple distribution
    lens.perspectives[candidate] = {hire: 0.6}

    behavior = lens.to_behavior()
    assert type(behavior).__name__ == "Behavior"


def test_lens_behavior_should_have_valid_agreement():
    """Test that lens-generated behavior has valid agreement."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")
    lens = observatory.lens(reviewer_a)

    candidate = lens.define("Candidate")
    hire, no_hire = reviewer_a.alphabet

    lens.perspectives[candidate] = {hire: 0.6}

    behavior = lens.to_behavior()
    agreement = behavior.agreement.result
    assert isinstance(agreement, (int, float))
    assert 0.0 <= agreement <= 1.0


def test_empty_lens_should_raise_no_concepts_defined_error():
    """Test that lens with no concepts available raises NoConceptsDefinedError."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    # Create a fresh observatory with no concepts defined
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])

    with observatory.lens("Alice") as lens:
        # Don't define any concepts
        with pytest.raises(NoConceptsDefinedError) as exc_info:
            lens.to_behavior()

        assert "no concepts defined" in str(exc_info.value).lower()
        assert "Call define() first" in str(exc_info.value)


def test_lens_with_concepts_but_no_distributions_should_raise_empty_behavior_error():
    """Test that lens with concepts but no distributions raises EmptyBehaviorError."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])

    reviewer_a = observatory.concept("Reviewer_A")

    with observatory.lens(reviewer_a) as lens:
        # Define concept but don't set any distributions
        candidate = lens.define("Candidate")

        # Access perspectives to initialize them (but don't set distributions)
        _ = lens.perspectives

        with pytest.raises(EmptyBehaviorError) as exc_info:
            lens.to_behavior()

        assert "no distributions defined" in str(exc_info.value).lower()
        assert "allow_empty=True" in str(exc_info.value)


def test_lens_allow_empty_behavior_should_create_empty_behavior():
    """Test that allow_empty=True creates empty behavior successfully."""
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])
    reviewer_a = observatory.concept("Reviewer_A")

    with observatory.lens(reviewer_a) as lens:
        # Define concept but don't set any distributions
        candidate = lens.define("Candidate")

        # Access perspectives to initialize them (but don't set distributions)
        _ = lens.perspectives

        # Should work with allow_empty=True
        behavior = lens.to_behavior(allow_empty=True)

        assert type(behavior).__name__ == "Behavior"
        assert len(behavior.distributions) == 0
        # Note: Empty behaviors don't have meaningful agreement values
        # so we don't test behavior.agreement here


def test_empty_lens_with_allow_empty_should_still_raise_no_concepts_error():
    """Test that lens with no concepts still raises error even with allow_empty=True."""
    # Create a fresh observatory with no concepts defined
    observatory = Observatory.create(symbols=["Hire", "No_Hire"])

    with observatory.lens("Alice") as lens:
        # Don't define any concepts, but try allow_empty=True
        with pytest.raises(NoConceptsDefinedError):
            lens.to_behavior(allow_empty=True)


def test_lens_compose_method_should_work():
    """Test that lens compose method creates LensComposition."""
    observatory = Observatory.create(symbols=["Yes", "No"])
    lens_a = observatory.lens("A")
    lens_b = observatory.lens("B")

    composition = lens_a.compose(lens_b)
    assert composition.__class__.__name__ == "LensComposition"
    assert isinstance(composition.lenses, tuple)
    assert len(composition.lenses) == 2
    assert lens_a in composition.lenses
    assert lens_b in composition.lenses


def test_lens_or_operator_should_work():
    """Test that | operator creates LensComposition."""
    observatory = Observatory.create(symbols=["Yes", "No"])
    lens_a = observatory.lens("A")
    lens_b = observatory.lens("B")

    composition = lens_a | lens_b
    assert composition.__class__.__name__ == "LensComposition"
    assert isinstance(composition.lenses, tuple)
    assert len(composition.lenses) == 2


def test_lens_composition_to_behavior_should_work():
    """Test that LensComposition can create behavior."""
    observatory = Observatory.create(symbols=["Yes", "No"])
    concept = observatory.concept("Test")
    yes, no = concept.alphabet

    lens_a = observatory.lens("A")
    lens_b = observatory.lens("B")

    # Set up simple distributions
    with lens_a:
        lens_a.perspectives[concept] = {yes: 0.6, no: 0.4}

    with lens_b:
        lens_b.perspectives[concept] = {yes: 0.4, no: 0.6}

    # Create composition and behavior
    composition = lens_a | lens_b
    behavior = composition.to_behavior()

    assert type(behavior).__name__ == "Behavior"
    assert len(behavior.distributions) == 2  # One for each lens


def test_lens_composition_properties_should_work():
    """Test that LensComposition properties work correctly."""
    observatory = Observatory.create(symbols=["Yes", "No"])
    concept = observatory.concept("Test")
    yes, no = concept.alphabet

    lens_a = observatory.lens("A")
    lens_b = observatory.lens("B")

    # Set up simple distributions that create contradiction
    with lens_a:
        lens_a.perspectives[concept] = {yes: 0.8, no: 0.2}

    with lens_b:
        lens_b.perspectives[concept] = {yes: 0.2, no: 0.8}

    # Create composition
    composition = lens_a | lens_b

    # Test perspective contributions
    contributions = composition.perspective_contributions
    assert isinstance(contributions, dict)
    assert "A" in contributions
    assert "B" in contributions
    assert abs(contributions["A"] - contributions["B"]) < 0.1  # Should be roughly equal

    # Test witness distribution alias
    witness = composition.witness_distribution
    assert witness == contributions

    # Test behavior creation
    behavior = composition.to_behavior()
    assert isinstance(behavior.contradiction_bits, float)
    assert behavior.contradiction_bits > 0  # Should have contradiction


def test_lens_composition_immutable():
    """Test that LensComposition is immutable (frozen dataclass)."""
    observatory = Observatory.create(symbols=["Yes", "No"])

    lens_a = observatory.lens("A")
    lens_b = observatory.lens("B")

    composition = lens_a | lens_b

    # Should not be able to modify the lenses tuple
    with pytest.raises(AttributeError):
        composition.lenses = ()

    # Should not be able to modify the tuple contents (tuples are immutable)
    assert isinstance(composition.lenses, tuple)
    assert len(composition.lenses) == 2


def test_lens_composition_chaining_should_work():
    """Test that LensComposition supports chaining with | operator."""
    observatory = Observatory.create(symbols=["Yes", "No"])
    concept = observatory.concept("Test")
    yes, no = concept.alphabet

    lens_a = observatory.lens("A")
    lens_b = observatory.lens("B")
    lens_c = observatory.lens("C")

    # Set up simple distributions
    with lens_a:
        lens_a.perspectives[concept] = {yes: 0.6, no: 0.4}

    with lens_b:
        lens_b.perspectives[concept] = {yes: 0.4, no: 0.6}

    with lens_c:
        lens_c.perspectives[concept] = {yes: 0.5, no: 0.5}

    # Test chaining: (lens_a | lens_b) | lens_c should work
    chained_composition = (lens_a | lens_b) | lens_c

    assert chained_composition.__class__.__name__ == "LensComposition"
    assert isinstance(chained_composition.lenses, tuple)
    assert len(chained_composition.lenses) == 3

    # Test that equivalent chained composition works
    equivalent_composition = lens_a | lens_b | lens_c
    assert len(equivalent_composition.lenses) == 3

    # Both should produce valid behaviors
    chained_behavior = chained_composition.to_behavior()
    equivalent_behavior = equivalent_composition.to_behavior()

    assert type(chained_behavior).__name__ == "Behavior"
    assert type(equivalent_behavior).__name__ == "Behavior"

    # Both should have the same number of distributions
    assert len(chained_behavior.distributions) == len(equivalent_behavior.distributions) == 3
